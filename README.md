# 🌟 mini-transformer 简易 Transformer 复现项目 🚀

![Python](https://img.shields.io/badge/Python-3.11.0%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1%2B-orange?logo=pytorch)

---

## 项目简介 🌟
本项目基于经典论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 复现了一个简易的 Transformer 模型。Transformer 模型以其强大的并行计算能力和卓越的序列处理能力，在自然语言处理、计算机视觉等多个领域取得了巨大成功。同时，项目还包含一个出于隐私保护目的的模型切片案例，探索模型隐私保护的新方法。

---

## 核心代码 📜
核心代码位于 `model.py` 文件中，该文件实现了 Transformer 模型的主要组件，包括多头注意力机制、前馈神经网络、编码器和解码器等。

---

## 项目结构 📁
```plaintext
mini-transformer/
├── .ipynb_checkpoints/    # 📋 Jupyter Notebook 检查点目录
├── data/                  # 📊 数据存放目录
├── privacy/               # 🔒 隐私保护相关代码目录
│   ├── ChatGLM_TEE_split.py
│   └── ChatGLM_privacy.py
├── tests/                 # 🧪 测试代码目录
├── model.py               # 🧠 核心 Transformer 模型代码
├── model_utils.py         # 🛠️ 模型工具函数
├── lora.py                # 📈 LoRA 相关代码
├── main.py                # 🚀 主程序入口
├── _transformer.py        # 🤖 Transformer 相关代码
├── demo.ipynb             # 📓 演示 Jupyter Notebook
├── test.ipynb             # 🧐 测试 Jupyter Notebook
├── README.md              # 📖 项目说明文档
└── sales_textbook.txt     # 📖 销售教材文本数据
```

## 模型切片隐私保护 🛡️
`privacy_slice.py` 中包含了一个模型切片的案例，通过将模型划分为多个部分，实现对模型隐私的保护。这一方法可以有效防止模型参数泄露，增强模型的安全性。

## 未来计划
目前很多切片与隐私部分的代码存在很大问题，后续会进行修改......

