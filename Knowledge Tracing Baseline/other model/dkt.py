import torch
import torch.nn as nn


class DKT(nn.Module):
    """经典 DKT（对齐 pykt/models/dkt.py 的 qid 版）：
    交互嵌入 x = q + num_c * r → LSTM → 全连接(输出 num_c 维) → sigmoid，
    预测时按当前题目 gather 取对应列。
    接口适配本项目框架：返回 (P, None)。"""

    def __init__(self, pro_max, emb_size=200, dropout=0.2):
        super().__init__()
        self.pro_max = pro_max
        self.interaction_emb = nn.Embedding(pro_max * 2, emb_size)
        self.lstm = nn.LSTM(emb_size, emb_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(emb_size, pro_max)

    def forward(self, last_problem, last_ans, next_problem, next_ans):
        # x_t = (题目, 作答) 组合索引；h_t 编码第 t 步之前的历史（先预测后含当前作答）
        x = last_problem + self.pro_max * last_ans.long()
        xemb = self.interaction_emb(x)
        h, _ = self.lstm(xemb)
        h = self.dropout(h)
        y = torch.sigmoid(self.out(h))                                 # (batch, seq, pro_max)
        P = y.gather(-1, next_problem.unsqueeze(-1)).squeeze(-1)       # (batch, seq)
        return P, None
