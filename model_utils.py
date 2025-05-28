import torch
import math
from model import context_length
device = "cuda" if torch.cuda.is_available() else "cpu"
def first_half(model, idx):
    """执行模型推理的前半部分操作。"""
    B, T = idx.shape
    token_embeddings = model.token_embedding_table(idx)
    context_length = model.transformer_blocks[0].attention.attention_heads[0].mask.size(0)
    d_model = model.token_embedding_table.embedding_dim
    
    position_encoding_lookup_table = torch.zeros(context_length, d_model, device=device)
    position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    position_encoding_lookup_table[:, 0::2] = torch.sin(position * div_term)
    position_encoding_lookup_table[:, 1::2] = torch.cos(position * div_term)
    position_embeddings = position_encoding_lookup_table[:T, :].to(device)
    
    x = token_embeddings + position_embeddings
    num_blocks = len(model.transformer_blocks)
    half_blocks = num_blocks // 2
    
    for block in model.transformer_blocks[:half_blocks]:
        x = block(x)
    
    return x


import torch.nn.functional as F


def second_half(model, x):
    """执行模型推理的后半部分操作。"""
    num_blocks = len(model.transformer_blocks)
    half_blocks = num_blocks // 2
    
    for block in model.transformer_blocks[half_blocks:]:
        x = block(x)

    logits = model.final_linear_layer(x)
    
    return logits

def generate_splitter(model ,idx, max_new_tokens=100):
    # idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
        # Crop idx to the max size of our positional embeddings table
        idx_crop = idx[:, -context_length:]
        # Get predictions
        x = first_half(model, idx_crop)
        logits = second_half(model, x)
        # Get the last time step from logits where the dimensions of the logits are (B,T,C)
        logits_last_timestep = logits[:, -1, :]
        # Apply softmax to get probabilities
        probs = F.softmax(input=logits_last_timestep, dim=-1)
        # Sample from the probabilities' distribution.
        idx_next = torch.multinomial(input=probs, num_samples=1)
        # Append the sampled indexes idx_next to idx
        idx = torch.cat((idx, idx_next), dim=1)
    return idx