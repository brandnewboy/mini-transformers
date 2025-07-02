import argparse
import os
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, TextIO

from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, AutoConfig, LogitsProcessorList, LogitsProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors
import shutil
import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:108"

avai_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
model:
    transformer:
        embedding:
        rotary_pos_emb:
        encoder:
            layers:
                0:
                1:
                ...
                27:
            final_layernorm:
        output_layer:
'''


class ChatGLM2FirstHalf(PreTrainedModel):
    def __init__(self, original_model, split_layer):
        super().__init__(original_model.config)
        self.embedding = original_model.transformer.embedding
        self.rotary_pos_emb = original_model.transformer.rotary_pos_emb
        # TODO # TODO 拷贝encoder 直接copy内存占用大
        self.encoder = original_model.transformer.encoder
        self.encoder.layers = nn.ModuleList(list(original_model.transformer.encoder.layers)[:split_layer])

        self.pre_seq_len = self.config.pre_seq_len
        self.prefix_projection = self.config.prefix_projection
        self.num_layers = self.config.num_layers
        self.multi_query_group_num = self.config.multi_query_group_num
        self.kv_channels = self.config.kv_channels
        self.seq_length = self.config.seq_length

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        prompt = tokenizer.build_prompt(query, history=history)
        inputs = tokenizer([prompt], return_tensors="pt")
        # TODO
        inputs = inputs.to(self.device)
        return inputs

    def forward(
            self,
            input_ids,
            position_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.BoolTensor] = None,
            full_attention_mask: Optional[torch.BoolTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ):
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if self.pre_seq_len is not None:
            if past_key_values is None:
                past_key_values = self.get_prompt(batch_size=batch_size, device=input_ids.device,
                                                  dtype=inputs_embeds.dtype)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask.new_ones((batch_size, self.pre_seq_len)),
                                            attention_mask], dim=-1)

        if full_attention_mask is None:
            if (attention_mask is not None and not attention_mask.all()) or (past_key_values and seq_length != 1):
                full_attention_mask = self.get_masks(input_ids, past_key_values, padding_mask=attention_mask)

        # Rotary positional embeddings
        rotary_pos_emb = self.rotary_pos_emb(self.seq_length)
        if position_ids is not None:
            rotary_pos_emb = rotary_pos_emb[position_ids]
        else:
            rotary_pos_emb = rotary_pos_emb[None, :seq_length]
        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states
        )

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        ), {
            'full_attention_mask': full_attention_mask,
            'rotary_pos_emb': rotary_pos_emb,
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'output_hidden_states': output_hidden_states,
        }


'''
AutoTokenizer.from_pretrained(
            self.tokenizer_dir,
            trust_remote_code=self.trust_remote_code
        )
'''


class ChatGLM2SecondHalf(PreTrainedModel):
    def __init__(self, original_model, split_layer):
        super().__init__(original_model.config)
        self.max_sequence_length = original_model.config.max_length
        self.rotary_pos_emb = original_model.transformer.rotary_pos_emb
        # TODO 拷贝encoder
        self.encoder = original_model.transformer.encoder
        self.encoder.layers = nn.ModuleList(list(original_model.transformer.encoder.layers)[split_layer:])
        self.output_layer = original_model.transformer.output_layer

    def forward(
            self,
            inputs_embeds,
            intermediate_data,
            return_dict: Optional[bool] = None,

            labels: Optional[torch.Tensor] = None,
            return_last_logit: Optional[bool] = False,
    ):
        full_attention_mask = intermediate_data['full_attention_mask']
        rotary_pos_emb = intermediate_data['rotary_pos_emb']
        past_key_values = intermediate_data['past_key_values']
        use_cache = intermediate_data['use_cache']
        output_hidden_states = intermediate_data['output_hidden_states']

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Run encoder.
        hidden_states, presents, all_hidden_states, all_self_attentions = self.encoder(
            inputs_embeds,
            full_attention_mask,
            rotary_pos_emb=rotary_pos_emb,
            kv_caches=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
        )
        # if not return_dict:
        #     return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        transformer_outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        # transformer_outputs = self.encoder(
        #     inputs_embeds,
        #     full_attention_mask,
        #     rotary_pos_emb=rotary_pos_emb,
        #     kv_caches=past_key_values,
        #     use_cache=use_cache,
        #     output_hidden_states=output_hidden_states,
        # )

        hidden_states = transformer_outputs[0]
        if return_last_logit:
            hidden_states = hidden_states[-1:]
        lm_logits = self.output_layer(hidden_states)
        lm_logits = lm_logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:
            lm_logits = lm_logits.to(torch.float32)

            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            lm_logits = lm_logits.to(hidden_states.dtype)
            loss = loss.to(hidden_states.dtype)

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def build_inputs(self, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        prompt = tokenizer.build_prompt(query, history=history)
        inputs = tokenizer([prompt], return_tensors="pt")
        # TODO
        inputs = inputs.to(self.device)
        return inputs

    @torch.inference_mode()
    def chat(
            self,
            tokenizer,
            query: str,
            history: List[Tuple[str, str]] = None,
            max_length: int = 8192,
            num_beams=1,
            do_sample=True,
            top_p=0.8,
            temperature=0.8,
            logits_processor=None,
            **kwargs
    ):
        if history is None:
            history = []
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "do_sample": do_sample, "top_p": top_p,
                      "temperature": temperature, "logits_processor": logits_processor, **kwargs}
        inputs = self.build_inputs(tokenizer, query, history=history)
        outputs = self.generate(**inputs, **gen_kwargs)
        outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
        response = tokenizer.decode(outputs)
        # response = self.process_response(response)
        history = history + [(query, response)]
        return response, history


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class ChatGLM2Splitter:
    def __init__(
            self,
            model_name_or_path: str = "THUDM/chatglm2-6b",
            trust_remote_code: bool = True
    ):
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.tokenizer = None
        self.model = None
        self.config = None

    def load_model(self, device='cuda'):
        """加载完整的ChatGLM2-6B模型"""
        print(f"正在加载模型: {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.float32,
            device=device
        )
        self.config = self.model.config
        print(f"模型加载完成：总层数: {self.config.num_layers}")
        print(f"模型加载完成：参数量: {sum(p.numel() for p in self.model.parameters()):,}")
        return self.model, self.tokenizer

    def split_model(self, split_layer: int) -> Tuple[nn.Module, nn.Module]:
        if self.model is None:
            self.load_model()

        if split_layer < 0 or split_layer >= self.config.num_layers:
            raise ValueError(f"分割层必须在 [0, {self.config.num_layers - 1}] 范围内")

        print(f"正在将模型分割为两部分，分割点: 第 {split_layer} 层")

        '''
            处理层参数，分开存储到硬盘，再加载到各个部分的模型
        '''
        # original_encoder = self.model.transformer.encoder
        # first_encoder_layers = nn.ModuleList(list(self.model.transformer.encoder.layers)[:split_layer])
        # second_encoder_layers = nn.ModuleList(list(self.model.transformer.encoder.layers)[split_layer:])
        # first_encoder_file_path = os.path.join(self.dir, "encoders/first_encoder.safetensors")
        # second_encoder_file_path = os.path.join(self.dir, "encoders/second_encoder.safetensors")
        # original_encoder_file_path = os.path.join(self.dir, "encoders/original_encoder.safetensors")
        #
        # save_safetensors(first_encoder_layers.state_dict(), first_encoder_file_path)
        # save_safetensors(second_encoder_layers.state_dict(), second_encoder_file_path)
        # save_safetensors(original_encoder.state_dict(), original_encoder_file_path)

        first_half = ChatGLM2FirstHalf(self.model, split_layer)
        second_half = ChatGLM2SecondHalf(self.model, split_layer)

        print(f"模型分割完成:")
        print(f"  - 前半部分: {sum(p.numel() for p in first_half.parameters()):,} 参数")
        print(f"  - 后半部分: {sum(p.numel() for p in second_half.parameters()):,} 参数")

        return first_half, second_half

    def save_split_models(self, first_half: nn.Module, second_half: nn.Module,
                          output_dir: str, split_layer: int, save_tokenizer: bool = True):
        first_half_dir = os.path.join(output_dir, "first_half")
        second_half_dir = os.path.join(output_dir, "second_half")

        os.makedirs(first_half_dir, exist_ok=True)
        os.makedirs(second_half_dir, exist_ok=True)

        # 保存模型权重
        print(f"保存前半部分模型到: {first_half_dir}")
        save_safetensors(first_half.state_dict(), os.path.join(first_half_dir, "split_model.safetensors"))

        print(f"保存后半部分模型到: {second_half_dir}")
        save_safetensors(second_half.state_dict(), os.path.join(second_half_dir, "split_model.safetensors"))

        # 保存配置文件
        config_path = os.path.join(output_dir, "config.json")
        first_config_path = os.path.join(output_dir, "first_half/config.json")
        second_config_path = os.path.join(output_dir, "second_half/config.json")
        first_config = second_config = self.config
        first_config.num_layers = split_layer
        second_config.num_layers -= split_layer
        self.config.to_json_file(config_path)
        first_config.to_json_file(first_config_path)
        second_config.to_json_file(second_config_path)

        # 复制配置文件到各部分目录
        shutil.copy(config_path, os.path.join(first_half_dir, "config.json"))
        shutil.copy(config_path, os.path.join(second_half_dir, "config.json"))

        # 复制代码文件
        code_files = ["configuration_chatglm.py", "modeling_chatglm.py", "tokenization_chatglm.py", "quantization.py"]
        for file in code_files:
            src_file = os.path.join(self.model_name_or_path, file)
            if os.path.exists(src_file):
                shutil.copy(src_file, first_half_dir)
                shutil.copy(src_file, second_half_dir)
                shutil.copy(src_file, output_dir)

        # 保存分割信息
        split_info = {
            "split_layer": split_layer,
            "total_layers": self.config.num_layers,
            "model_name": self.model_name_or_path,
            "first_half_params": sum(p.numel() for p in first_half.parameters()),
            "second_half_params": sum(p.numel() for p in second_half.parameters())
        }

        with open(os.path.join(first_half_dir, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)
        with open(os.path.join(second_half_dir, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)
        with open(os.path.join(output_dir, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)

        # 保存tokenizer
        if save_tokenizer:
            print(f"保存tokenizer到: {output_dir}")
            # 删除self.tokenizer上的某些特定数据
            # 定义需要删除的属性列表，将这里替换为你实际要删除的属性名
            # attributes_to_remove = ["eos_token", "pad_token", "unk_token"]
            # for attr in attributes_to_remove:
            #     if hasattr(self.tokenizer, attr):
            #         delattr(self.tokenizer, attr)
            self.tokenizer.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(first_half_dir)
            self.tokenizer.save_pretrained(second_half_dir)

        print(f"模型分割并保存完成，分割点: 第 {split_layer} 层")
        return first_half_dir, second_half_dir


class ChatGLM2DistributedInference:
    """ChatGLM2-6B 分布式推理工具，使用分割后的模型进行推理"""

    def __init__(self):
        pass

    def generate(
            self,
            prompt: str,
            max_length: int = 2048,
            temperature: float = 0.8,
            top_p: float = 0.9,
            first_device: str = "cuda:0",
            second_device: str = "cuda:1",
            use_cache: bool = True
    ) -> str:
        pass


def split_model(
        model_name_or_path: str,
        output_dir: str,
        split_layer: int,
        device='cuda',
):
    splitter = ChatGLM2Splitter(model_name_or_path)
    splitter.load_model(device)
    first_half, second_half = splitter.split_model(split_layer)
    splitter.save_split_models(first_half, second_half, output_dir, split_layer)


SPLIT_MODEL_PARTS = {
    "first": "first_half",
    "second": "second_half"
}


def load_model(
        dir: str,
        model_clazz: ChatGLM2FirstHalf or ChatGLM2SecondHalf,
        tokenizer_dir: Optional[str] = None,
        device='cuda',
):
    if tokenizer_dir is None:
        tokenizer_dir = dir
    print(f"加载tokenizer from: {tokenizer_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=True
    )

    print(f"加载配置 from: {dir}")
    config = AutoConfig.from_pretrained(dir, trust_remote_code=True)

    print(f"加载分割信息 from: {os.path.join(dir, 'split_info.json')}")
    with open(os.path.join(dir, 'split_info.json'), "r") as f:
        split_info = json.load(f)

    torch.cuda.empty_cache()

    print(f"加载分割模型 from: {dir} 到 {device}")
    model = model_clazz(
        AutoModel.from_config(
            config, trust_remote_code=True,
            device=device
        ),
        # original_tokenizer=tokenizer,
        split_layer=split_info['split_layer']
    )
    state_dict = load_safetensors(os.path.join(dir, "split_model.safetensors"))
    model.load_state_dict(state_dict, strict=False)  # 设置 strict 为 False

    print(f"模型加载完成, 模型配置:")
    print(f"  - 总层数: {config.num_layers}")
    print(f"  - 参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  - 分割点: 第 {split_info['split_layer']} 层")
    return model, tokenizer


def logits_to_text(logits, tokenizer):
    # 从 logits 中获取每个位置概率最大的词元 ID
    predicted_ids = logits.argmax(dim=-1)
    # 使用分词器将词元 ID 解码成文本
    text = tokenizer.decode(predicted_ids[0], skip_special_tokens=True)
    return text


def first_inference(prompt_str, model_path, intermediate_data_file_path, hidden_states_file_path, device='cpu'):
    model_first, tokenizer = load_model(dir=model_path, model_clazz=ChatGLM2FirstHalf, device=device)
    model_first.eval()
    print(model_first)
    inputs = model_first.build_inputs(tokenizer, prompt_str)
    first_output, intermediate_data = model_first(**inputs)
    intermediate_data = {
        'full_attention_mask': intermediate_data['full_attention_mask'],
        'rotary_pos_emb': intermediate_data['rotary_pos_emb'],
        'past_key_values': intermediate_data['past_key_values'],
        'use_cache': intermediate_data['use_cache'],
        'output_hidden_states': intermediate_data['output_hidden_states'],
        'inputs': inputs
    }
    print(f'中间层输出: {first_output.last_hidden_state}')
    data = {"last_hidden_state": first_output.last_hidden_state}
    save_safetensors(data, hidden_states_file_path)
    print(f'保存中间数据: {intermediate_data}')
    torch.save(intermediate_data, intermediate_data_file_path)


def second_inference(
        model_path,
        intermediate_data_file_path,
        hidden_states_file_path,
        device='cpu'
):
    model_second, tokenizer = load_model(dir=model_path, model_clazz=ChatGLM2SecondHalf, device=device)
    model_second.eval()
    print(model_second)
    data = load_safetensors(hidden_states_file_path)
    intermediate_data = torch.load(intermediate_data_file_path)
    final_output = model_second(data['last_hidden_state'], intermediate_data, return_dict=True)
    text = logits_to_text(final_output.logits, tokenizer)
    print(f'文本: {text}')


# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=False, type=str, default="E:\models\chatglm", help="模型名称或路径")
    parser.add_argument('-o', '--output_path', required=False, type=str, default="E:\models\chatglm_split",
                        help="分割模型保存路径")
    args = parser.parse_args()

    split_model(args.model, args.output_path, 14, 'cpu')

    prompt_str = '你好'
    hidden_states_file_path = 'E:\models\chatglm_split\hidden_states\/first_hidden_states.safetensors'
    intermediate_data_file_path = 'E:\models\chatglm_split\hidden_states\intermediate_data.pt'

    first_inference(
        prompt_str,
        'E:\models\chatglm_split\/first_half',
        intermediate_data_file_path,
        hidden_states_file_path,
        'cpu'
    )

    second_inference(
        'E:\models\chatglm_split\/second_half',
        intermediate_data_file_path,
        hidden_states_file_path,
        'cpu'
    )
