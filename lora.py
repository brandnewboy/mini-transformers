from functools import partial

from model import Model
import torch
import torch.nn as nn


my_model = Model()
print('\n')
print('===================================================================================')
print('===================== your old model not with LoRA layers: ========================')
print(my_model)

def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False


# create LoRA layer
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        std_dev = 1 / torch.sqrt(torch.tensor(rank).float())
        self.W_a = nn.Parameter(torch.randn(in_dim, rank) * std_dev)
        self.W_b = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x):
        x = self.alpha * x @ self.W_a @ self.W_b
        return x

class LinearWithLoRA(nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        x = self.linear(x)
        x = x + self.lora(x)
        return x

# Add LoRA layers to the model
lora_rank = 8
lora_alpha = 16
lora_query = True
lora_key = False
lora_value = True
lora_projection = False
lora_mlp = False
lora_head = False

layers = []
assign_lora = partial(LinearWithLoRA, rank=lora_rank, alpha=lora_alpha)

if __name__ == "__main__":
    freeze_layers(model=my_model)
    for layer in my_model.transformer_blocks:
        if lora_query:
            for i, linear in enumerate(layer.attention.attention_heads):
                lin = layer.attention.attention_heads[i].q_lin
                layer.attention.attention_heads[i].q_lin = assign_lora(lin)

    print('\n')
    print('===================================================================================')
    print('======================= your new model with LoRA layers: ==========================')
    print(my_model)