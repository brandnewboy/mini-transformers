import torch
from model import Model, enc, trainModelForExport
from model_utils import generate_splitter

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Model()
class TestModelUtils:
    def __init__(self, _model = model):
        self.model = _model


    def test_model_splitting(self, prompt = 'The salesperson'):
        start = 'The salesperson'
        start_ids = enc.encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        # 完整模型推理
        logits_full = self.model.generate(x, max_new_tokens=100)

        # 拆分模型推理
        logits_split = generate_splitter(self.model, x)

        # 对比结果
        # self.assertTrue(torch.allclose(logits_full, logits_split), "拆分前后模型生成结果不一致")

        print("完整模型生成结果：", enc.decode(logits_full[0].tolist()))
        print("========================================")
        print("拆分模型生成结果：", enc.decode(logits_split[0].tolist()))

if __name__ == '__main__':
    # 先进行训练
    trainModelForExport(model)
    # 然后进行测试
    model.eval()
    test_model_utils = TestModelUtils(model)
    test_model_utils.test_model_splitting()