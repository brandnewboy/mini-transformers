import torch
from model import Model, enc  # 从model.py导入Model类和token编码器

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_full_model(model_path):
    """加载完整模型权重"""
    model = Model().to(device)
    model.load_state_dict(torch.load(model_path))
    return model


def load_split_model():
    """加载分块层权重"""
    model = Model().to(device)
    # 加载词嵌入层
    model.token_embedding_table.load_state_dict(torch.load("D:/Coding_Personal/py/LLM_Learn_Transformers/data/token_embedding.pth"))
    # 加载Transformer块
    for i, block in enumerate(model.transformer_blocks):
        block.load_state_dict(torch.load(f"D:/Coding_Personal/py/LLM_Learn_Transformers/data/transformer_block_{i}.pth"))
    # 加载最终线性层
    model.final_linear_layer.load_state_dict(torch.load("D:/Coding_Personal/py/LLM_Learn_Transformers/data/final_linear.pth"))
    return model


def evaluate_model(model):
    """执行推理生成文本"""
    model.eval()
    start = 'The'
    start_ids = enc.encode(start)
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]  # 扩展batch维度
    y = model.generate(x, max_new_tokens=100)
    print('---------------')
    print(enc.decode(y[0].tolist()))
    print('---------------')


if __name__ == "__main__":
    # 选择加载模式：0为完整权重，1为分块权重
    load_mode = 0
    if load_mode == 0:
        model = load_full_model("D:/Coding_Personal/py/LLM_Learn_Transformers/data/model.pth")
    else:
        model = load_split_model()

    evaluate_model(model)