#! /usr/bin/env python3

import torch
from torch import nn

# 初始化模型参数
d_model = 512
nhead = 8
num_layers = 6

# 初始化一个Transformer模型
model = nn.Transformer(d_model, nhead, num_layers)

# 生成输入数据
src = torch.rand((10, 32, d_model))  # (seq_length, batch_size, d_model)
tgt = torch.rand((20, 32, d_model))  # (seq_length, batch_size, d_model)

# 前向传播
out = model(src, tgt)
print(out.shape)  # 输出: torch.Size([20, 32, 512])
