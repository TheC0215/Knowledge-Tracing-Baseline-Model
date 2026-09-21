"""pyKT 各模型共享的小工具（对齐 pykt/models/utils.py，仅做自包含与设备处理）：
- transformer_FFN / ut_mask / lt_mask / pos_encode / get_clones：与 pyKT 相同；
- CosinePositionalEmbedding：simpleKT/UKT 共用（pyKT 中各自文件里各有一份，这里收敛成一份）；
- 与 pyKT 的唯一差异：掩码/位置张量允许显式指定 device（默认从调用处传入），
  避免 pyKT 用全局 device 变量导致"机器有 CUDA 时 CPU 测试报错"的问题。
"""
import copy
import math

import torch
from torch import nn
from torch.nn import Sequential, Linear, ReLU, Dropout


class transformer_FFN(nn.Module):
    """两层的 Position-wise FFN（pyKT 原样）。"""

    def __init__(self, emb_size, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
            Linear(self.emb_size, self.emb_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.emb_size, self.emb_size),
            # Dropout(self.dropout),
        )

    def forward(self, in_fea):
        return self.FFN(in_fea)


def ut_mask(seq_len, device=None):
    """Upper Triangular Mask：对角线(含)以上为 True，表示"不允许 attend"（严格因果）。
    pyKT 原样，仅加 device 参数。"""
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool).to(device)


def lt_mask(seq_len, device=None):
    """Lower Triangular Mask（pyKT 原样，本项目未使用，仅为对齐保留）。"""
    return torch.tril(torch.ones(seq_len, seq_len), diagonal=-1).to(dtype=torch.bool).to(device)


def pos_encode(seq_len, device=None):
    """位置编码索引 [0, 1, ..., seq_len-1]（pyKT 原样，仅加 device）。"""
    return torch.arange(seq_len).unsqueeze(0).to(device)


def get_clones(module, N):
    """克隆 N 个 nn.Module（deepcopy，pyKT 原样）。"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class CosinePositionalEmbedding(nn.Module):
    """余弦位置编码（simpleKT/UKT 用，pyKT 原样：sin/cos 覆盖偶/奇维，前 0.1*randn 噪声被覆盖）。"""

    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        # 返回 (1, seq_len, d_model)，按输入序列长度截断
        return self.weight[:, :x.size(1), :]
