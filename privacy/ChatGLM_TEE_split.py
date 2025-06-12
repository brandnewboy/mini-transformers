import os
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Any, Tuple
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import shutil
import json

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

class ChatGLM2FirstHalf(nn.Module):
    def __init__(self, original_model, split_layer):
        super().__init__()
        self.config = original_model.config
        # 通过 transformer 访问 embedding
        self.embedding = original_model.transformer.embedding
        # 通过 transformer 访问 rotary_pos_emb
        self.rotary_pos_emb = original_model.transformer.rotary_pos_emb
        self.encoder = nn.ModuleList(list(original_model.transformer.encoder.layers)[:split_layer])
        # 通过 transformer 访问 final_layernorm
        self.final_layernorm = original_model.transformer.encoder.final_layernorm

    def forward(
            self,
            input_ids: torch.LongTensor,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        batch_size, seq_length = input_ids.shape

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        # 处理旋转位置编码
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 计算KV缓存的长度
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]
            position_ids = position_ids[:, past_length:]

        # 前向传播前半部分
        hidden_states = inputs_embeds
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.encoder):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_past = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                presents = presents + (layer_outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2 if use_cache else 1],)

        # 应用最后的层归一化
        hidden_states = self.final_layernorm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ChatGLM2SecondHalf(nn.Module):
    def __init__(self, original_model, split_layer):
        super().__init__()
        self.config = original_model.config
        self.encoder = nn.ModuleList(list(original_model.transformer.encoder.layers)[split_layer:])
        self.output_layer = original_model.transformer.output_layer

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            position_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: bool = False,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ):
        batch_size, seq_length = hidden_states.shape[:2]

        # 处理旋转位置编码
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

        # 前向传播后半部分
        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        for i, layer in enumerate(self.encoder):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_past = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_value=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                presents = presents + (layer_outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2 if use_cache else 1],)

        # 应用输出层（生成logits）
        logits = self.output_layer(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [logits, hidden_states, presents, all_hidden_states, all_self_attentions] if
                         v is not None)

        return {
            'logits': logits,
            'hidden_states': hidden_states,
            'past_key_values': presents,
            'attentions': all_self_attentions,
        }


class ChatGLM2Splitter:
    """ChatGLM2-6B 模型分割工具，支持按层分割模型并保存为独立部分"""

    def __init__(self, model_name_or_path: str = "THUDM/chatglm2-6b",
                 trust_remote_code: bool = True):
        """
        初始化模型分割器

        Args:
            model_name_or_path: 模型名称或路径
            trust_remote_code: 是否信任远程代码
        """
        self.model_name_or_path = model_name_or_path
        self.trust_remote_code = trust_remote_code
        self.tokenizer = None
        self.model = None
        self.config = None

    def load_model(self):
        """加载完整的ChatGLM2-6B模型"""
        print(f"正在加载模型: {self.model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, trust_remote_code=self.trust_remote_code
        )
        self.model = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=self.trust_remote_code,
            torch_dtype=torch.float16
        )
        self.config = self.model.config
        print(f"模型加载完成，总层数: {self.config.num_layers}")
        return self.model, self.tokenizer

    def split_model(self, split_layer: int) -> Tuple[nn.Module, nn.Module]:
        """
        按指定层分割模型为两部分

        Args:
            split_layer: 分割点所在层（0-based索引）

        Returns:
            Tuple[前半部分模型, 后半部分模型]
        """
        if self.model is None:
            self.load_model()

        if split_layer < 0 or split_layer >= self.config.num_layers:
            raise ValueError(f"分割层必须在 [0, {self.config.num_layers - 1}] 范围内")

        print(f"正在将模型分割为两部分，分割点: 第 {split_layer} 层")

        print(self.model)
        # 创建分割后的模型
        first_half = ChatGLM2FirstHalf(self.model, split_layer)
        second_half = ChatGLM2SecondHalf(self.model, split_layer)

        print(f"模型分割完成:")
        print(f"  - 前半部分: {sum(p.numel() for p in first_half.parameters()):,} 参数")
        print(f"  - 后半部分: {sum(p.numel() for p in second_half.parameters()):,} 参数")

        return first_half, second_half

    def save_split_models(self, first_half: nn.Module, second_half: nn.Module,
                          output_dir: str, split_layer: int, save_tokenizer: bool = True):
        """
        保存分割后的模型到指定目录

        Args:
            first_half: 模型前半部分
            second_half: 模型后半部分
            output_dir: 输出目录
            split_layer: 分割层
            save_tokenizer: 是否保存tokenizer
        """
        # 创建输出目录
        first_half_dir = os.path.join(output_dir, "first_half")
        second_half_dir = os.path.join(output_dir, "second_half")

        os.makedirs(first_half_dir, exist_ok=True)
        os.makedirs(second_half_dir, exist_ok=True)

        # 保存模型权重
        print(f"保存前半部分模型到: {first_half_dir}")
        torch.save(first_half.state_dict(), os.path.join(first_half_dir, "pytorch_model.bin"))
        # 保存tokenizer
        if save_tokenizer:
            print(f"前半部分tokenizer: {first_half_dir}")
            self.tokenizer.save_pretrained(first_half_dir)

        print(f"保存后半部分模型到: {second_half_dir}")
        torch.save(second_half.state_dict(), os.path.join(second_half_dir, "pytorch_model.bin"))
        # 保存tokenizer
        if save_tokenizer:
            print(f"前半部分tokenizer: {second_half_dir}")
            self.tokenizer.save_pretrained(second_half_dir)

        # 保存配置文件
        config_path = os.path.join(output_dir, "config.json")
        self.config.to_json_file(config_path)

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

        # 保存分割信息
        split_info = {
            "split_layer": split_layer,
            "total_layers": self.config.num_layers,
            "model_name": self.model_name_or_path,
            "first_half_params": sum(p.numel() for p in first_half.parameters()),
            "second_half_params": sum(p.numel() for p in second_half.parameters())
        }

        with open(os.path.join(output_dir, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)

        # # 保存tokenizer
        # if save_tokenizer:
        #     print(f"保存tokenizer到: {output_dir}")
        #     self.tokenizer.save_pretrained(output_dir)

        print(f"模型分割并保存完成，分割点: 第 {split_layer} 层")
        return first_half_dir, second_half_dir


class ChatGLM2DistributedInference:
    """ChatGLM2-6B 分布式推理工具，使用分割后的模型进行推理"""

    def __init__(self, first_half_dir: str, second_half_dir: str,
                 tokenizer_dir: Optional[str] = None,
                 trust_remote_code: bool = True):
        """
        初始化分布式推理工具

        Args:
            first_half_dir: 模型前半部分目录
            second_half_dir: 模型后半部分目录
            tokenizer_dir: tokenizer目录，若为None则使用first_half_dir
            trust_remote_code: 是否信任远程代码
        """
        self.first_half_dir = first_half_dir
        self.second_half_dir = second_half_dir
        self.tokenizer_dir = tokenizer_dir or first_half_dir
        self.trust_remote_code = trust_remote_code

        # 加载分割信息
        split_info_path = os.path.dirname(first_half_dir) + "/split_info.json"
        with open(split_info_path, "r") as f:
            self.split_info = json.load(f)

        print(f"加载分割信息: 分割点在第 {self.split_info['split_layer']} 层")

        # 初始化模型和tokenizer
        self.tokenizer = None
        self.first_half = None
        self.second_half = None
        self.config = None

    def load_models(self, first_device: str = "cuda:0", second_device: str = "cuda:1"):
        """
        加载分割后的模型到指定设备

        Args:
            first_device: 前半部分模型设备
            second_device: 后半部分模型设备
        """
        # 加载tokenizer
        print(f"加载tokenizer from: {self.tokenizer_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_dir, trust_remote_code=self.trust_remote_code
        )

        # 加载配置
        config_path = os.path.join(self.first_half_dir, "config.json")
        self.config = AutoConfig.from_pretrained(config_path,
                                                 trust_remote_code=self.trust_remote_code,
                                                 torch_dtype = torch.float16
                                                 )

        # 创建并加载前半部分模型
        print(f"加载前半部分模型 from: {self.first_half_dir} 到 {first_device}")
        self.first_half = ChatGLM2FirstHalf(AutoModel.from_config(self.config), self.split_info['split_layer'])
        self.first_half.load_state_dict(torch.load(
            os.path.join(self.first_half_dir, "pytorch_model.bin"),
            map_location=first_device
        ), strict=False)  # 设置 strict 为 False
        self.first_half.to(first_device)
        self.first_half.eval()

        # 创建并加载后半部分模型
        print(f"加载后半部分模型 from: {self.second_half_dir} 到 {second_device}")
        self.second_half = ChatGLM2SecondHalf(AutoModel.from_config(self.config), self.split_info['split_layer'])
        self.second_half.load_state_dict(torch.load(
            os.path.join(self.second_half_dir, "pytorch_model.bin"),
            map_location=second_device
        ), strict=False)  # 设置 strict 为 False
        self.second_half.to(second_device)
        self.second_half.eval()

        print("分布式模型加载完成")
        # 打印模型相关信息
        print(f"模型配置:")
        print(f"  - 总层数: {self.config.num_layers}")
        print(f"  - 前半部分参数数量: {sum(p.numel() for p in self.first_half.parameters()):,}")
        print(f"  - 后半部分参数数量: {sum(p.numel() for p in self.second_half.parameters()):,}")
        print(f"  - 分割点: 第 {self.split_info['split_layer']} 层")
        return self.tokenizer, self.first_half, self.second_half

    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 2048, temperature: float = 0.8,
                 top_p: float = 0.9, first_device: str = "cuda:0", second_device: str = "cuda:1",
                 use_cache: bool = True) -> str:
        """
        使用分布式模型生成回答

        Args:
            prompt: 输入提示
            max_length: 最大生成长度
            temperature: 温度参数
            top_p: 核采样参数
            first_device: 前半部分模型设备
            second_device: 后半部分模型设备
            use_cache: 是否使用KV缓存

        Returns:
            生成的回答文本
        """
        if self.tokenizer is None or self.first_half is None or self.second_half is None:
            self.load_models(first_device, second_device)

        # 编码输入
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(first_device)
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(first_device)

        # 初始化生成长度和KV缓存
        generated_ids = [input_ids[0].tolist()]
        past_key_values = None

        print(f"开始生成回答，输入长度: {input_ids.shape[1]}")

        for i in range(max_length):
            # 前半部分模型推理
            first_outputs = self.first_half(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache
            )

            # 将中间结果移至后半部分模型设备
            hidden_states = first_outputs.last_hidden_state.to(second_device)
            # TODO 存入文件，两部分之间通过该文件进行数据传输
            if use_cache:
                past_key_values = [
                    (k.to(second_device), v.to(second_device))
                    for k, v in first_outputs.past_key_values
                ]

            # 后半部分模型推理
            second_outputs = self.second_half(
                hidden_states=hidden_states,
                attention_mask=attention_mask.to(second_device) if attention_mask is not None else None,
                past_key_values=past_key_values,
                use_cache=use_cache
            )

            # 获取logits并计算下一个token
            logits = second_outputs['logits'][:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            probs, indices = torch.topk(probs, k=min(50, probs.shape[-1]))
            probs = probs / probs.sum()

            # 核采样
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # 应用采样
            sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum()
            next_token_idx = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices[0, next_token_idx].unsqueeze(0)

            # 更新生成的ids和输入
            generated_ids[0].append(next_token.item())
            input_ids = next_token.to(first_device)

            # 更新注意力掩码
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=first_device)], dim=1)

            # 检查是否生成结束标记
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            # 定期打印进度
            if (i + 1) % 50 == 0:
                print(f"已生成 {i + 1} 个token")

        # 解码生成的ids
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # 提取回答部分（移除输入提示）
        response = generated_text[len(prompt):].strip()

        print(f"回答生成完成，总长度: {len(generated_text)}")
        return response


# 使用示例
if __name__ == "__main__":
    # 1. 分割模型
    splitter = ChatGLM2Splitter(model_name_or_path="F:\models")
    model, tokenizer = splitter.load_model()

    # 选择分割层（例如将28层模型分为前14层和后14层）
    split_layer = 14
    first_half, second_half = splitter.split_model(split_layer)

    # 保存分割后的模型
    output_dir = "D:\Coding_Personal\py\LLM_Learn_Transformers\data\chatglm2_6b_split"
    first_half_dir, second_half_dir = splitter.save_split_models(
        first_half, second_half, output_dir, split_layer
    )

    # 2. 使用分割后的模型进行推理
    inference = ChatGLM2DistributedInference(first_half_dir, second_half_dir)

    # 加载模型到指定设备
    tokenizer, first_half_model, second_half_model = inference.load_models(
        first_device="cuda", second_device="cuda"
    )

    # 进行推理
    prompt = "你好，请问你能做什么？"
    response = inference.generate(prompt, max_length=512)

    print(f"输入: {prompt}")
    print(f"输出: {response}")