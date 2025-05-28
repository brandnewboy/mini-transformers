import torch
import secretflow as sf
from secretflow.device import PYU, TEEU
from transformers import AutoModelForCausalLM, AutoTokenizer
from secretflow.security.aggregation import SecureAggregator


def split_and_freeze_model(model):
    """冻结模型前半部分参数"""
    num_layers = len(model.transformer.layers)
    split_index = num_layers // 2

    # 冻结前半部分模型参数
    for layer in model.transformer.layers[:split_index]:
        for param in layer.parameters():
            param.requires_grad = False

    return model, model.transformer.layers[split_index:]


def prepare_data(tokenizer):
    """准备训练数据"""
    texts = ["你好，世界", "今天天气如何"]
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    inputs['labels'] = inputs.input_ids.clone()  # 添加labels字段用于训练
    return inputs


def apply_spf_strategy(model, selection_ratio=0.5):
    """
    应用SPF (Sparsification Parameter Fine-tuning) 策略选择需要训练的参数
    """
    trainable_params = []
    frozen_params = []
    trainable_indices = {}

    # 遍历模型的所有模块
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            if 'query_key_value' in name or 'dense_h_to_4h' in name:
                weight = module.weight

                if 'query_key_value' in name:
                    hidden_size = weight.shape[1]
                    q_weight, k_weight, v_weight = weight.chunk(3, dim=0)

                    num_heads = model.config.num_attention_heads
                    head_dim = hidden_size // num_heads

                    for mat, mat_name in [(q_weight, 'q'), (k_weight, 'k'), (v_weight, 'v')]:
                        head_l1_norms = []
                        for i in range(num_heads):
                            start_idx = i * head_dim
                            end_idx = (i + 1) * head_dim
                            head = mat[start_idx:end_idx]
                            l1_norm = torch.norm(head, p=1)
                            head_l1_norms.append((i, l1_norm))

                        head_l1_norms.sort(key=lambda x: x[1], reverse=True)
                        k = int(len(head_l1_norms) * selection_ratio)
                        selected_heads = [idx for idx, _ in head_l1_norms[:k]]

                        trainable_indices[f"{name}_{mat_name}"] = selected_heads

                        for i in range(num_heads):
                            start_idx = i * head_dim
                            end_idx = (i + 1) * head_dim

                            if i in selected_heads:
                                trainable_params.append(mat[start_idx:end_idx])
                            else:
                                frozen_params.append(mat[start_idx:end_idx])

                else:
                    l1_norms = torch.norm(weight, p=1, dim=1)
                    sorted_indices = torch.argsort(l1_norms, descending=True)
                    k = int(len(sorted_indices) * selection_ratio)
                    selected_indices = sorted_indices[:k]

                    trainable_indices[name] = selected_indices.tolist()

                    for i in range(len(weight)):
                        if i in selected_indices:
                            trainable_params.append(weight[i])
                        else:
                            frozen_params.append(weight[i])

    return trainable_params, frozen_params, trainable_indices


def apply_lora_to_attention_output(model, r=8, lora_alpha=32, lora_dropout=0.1):
    """对多头注意力机制结束后的线性层应用LoRA"""
    from peft import LoraConfig, get_peft_model, TaskType

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=["dense"]
    )

    lora_model = get_peft_model(model, config)
    return lora_model


@TEEU.device
def train_model_in_teeu(model, data, num_epochs, learning_rate, selection_ratio, use_lora):
    """在TEEU内部执行完整的训练流程"""

    class PartialModel(torch.nn.Module):
        def __init__(self, layers):
            super().__init__()
            self.layers = torch.nn.ModuleList(layers)
            self.config = model[0].config

        def forward(self, input_ids, labels=None):
            hidden_states = input_ids
            for layer in self.layers:
                hidden_states = layer(hidden_states)[0]

            logits = hidden_states

            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    labels.view(-1)
                )
                return {'loss': loss, 'logits': logits}

            return {'logits': logits}

    partial_model = PartialModel(model)

    if use_lora:
        partial_model = apply_lora_to_attention_output(partial_model)

    trainable_params, _, _ = apply_spf_strategy(partial_model, selection_ratio=selection_ratio)

    optimizer_params = []
    if use_lora:
        lora_params = [p for n, p in partial_model.named_parameters() if "lora" in n.lower() and p.requires_grad]
        optimizer_params.extend(lora_params)

    optimizer_params.extend(trainable_params)
    optimizer = torch.optim.AdamW(optimizer_params, lr=learning_rate)

    for epoch in range(num_epochs):
        total_loss = 0
        partial_model.train()

        for i in range(len(data['input_ids'])):
            input_ids = data['input_ids'][i].unsqueeze(0)
            labels = data['labels'][i].unsqueeze(0)

            outputs = partial_model(input_ids, labels=labels)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch + 1}, Loss: {total_loss / len(data["input_ids"])}')

    return partial_model


@PYU.device
def process_input_in_client(tokenizer, input_text):
    """在客户端处理输入文本"""
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    return inputs


@PYU.device
def run_front_model(model_front, input_ids):
    """在客户端运行模型前半部分"""
    model_front.eval()
    with torch.no_grad():
        hidden_states = input_ids
        for layer in model_front.transformer.layers:
            hidden_states = layer(hidden_states)[0]
    return hidden_states


@TEEU.device
def run_latter_model(model_latter, hidden_states):
    """在TEEU中运行模型后半部分"""
    model_latter.eval()
    with torch.no_grad():
        for layer in model_latter.layers:
            hidden_states = layer(hidden_states)[0]
        logits = hidden_states
    return logits


def secure_inference(model_front, model_latter, tokenizer, pyu_device, teeu_device, input_text):
    """安全推理流程：结合前半部分和后半部分模型"""
    # 在客户端处理输入文本
    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs.input_ids

    # 在客户端运行前半部分模型
    front_output = run_front_model(model_front, input_ids)

    # 将前半部分输出传输到TEEU
    front_output_in_teeu = front_output.to(teeu_device)

    # 在TEEU中运行后半部分模型
    logits = run_latter_model(model_latter, front_output_in_teeu)

    # 将结果返回客户端并解码
    logits_numpy = logits.to(pyu_device).reveal()
    output_text = tokenizer.decode(logits_numpy.argmax(dim=-1)[0], skip_special_tokens=True)

    return output_text


def main():
    """主函数"""
    # 初始化SecretFlow
    sf.init(
        address='127.0.0.1:10000',
        cluster_config={
            'parties': {
                'alice': {'address': '127.0.0.1:20001', 'listen_address': '0.0.0.0:20001'},
                'carol': {'address': '127.0.0.1:20002', 'listen_address': '0.0.0.0:20002'}
            },
            'self_party': 'alice'
        },
        tee_simulation=True
    )

    # 定义设备
    alice = PYU('alice')
    carol = TEEU('carol')

    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b")
    model = AutoModelForCausalLM.from_pretrained("THUDM/chatglm-6b")

    # 对模型进行分层和参数冻结
    model, sliced_model = split_and_freeze_model(model)

    # 准备数据并发送到TEEU
    data = prepare_data(tokenizer)
    data_in_teeu = {k: carol(lambda x: x)(v) for k, v in data.items()}

    # 将模型后半部分发送到TEEU
    sliced_model_in_teeu = carol(lambda x: x)(sliced_model)

    # 在TEEU中训练模型
    print("开始在TEEU中训练...")
    trained_model = train_model_in_teeu(
        sliced_model_in_teeu,
        data_in_teeu,
        num_epochs=3,
        learning_rate=1e-4,
        selection_ratio=0.5,
        use_lora=True
    )


    # 进行安全推理
    input_text = "请介绍一下Python"
    print(f"\n输入: {input_text}")
    result = secure_inference(model, trained_model, tokenizer, alice, carol, input_text)
    print(f"输出: {result}")


if __name__ == "__main__":
    main()