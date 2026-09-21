"""DKVMN（动态键值记忆网络）——对齐 pykt/models/dkvmn.py 的 qid 版，只改数据映射。

pyKT 侧（train_model.py / evaluate_model.py）的调用约定：
    y = model(cc, cr)，其中 cc = cat(c[:,0:1], cshft)、cr = cat(r[:,0:1], rshft)，
    是把 199 长的错位窗口"前插一列"还原成 200 长的原窗口（行 j ↔ 原窗口第 j 步交互）。
    y 行 j = 对题目 cc[j] 的预测，且 read 用 Mv[:, :-1]，即"先读后写"：
    行 j 只看到写入了 0..j-1 步作答的记忆（Mv[j]），因此 y 行 j 没有用到 cc[j] 自己的作答。
    训练/评测取 y[:,1:] 与错位标签 rshft 对齐。

本项目 loader 给出的正是同一个错位窗口：last_problem[t]=窗口 t、next_problem[t]=窗口 t+1
（L = 窗口长-1）。因此端口 forward 内部同样前插一列构造 200 行序列：
    q_ = cat([last_problem[:, :1], next_problem], dim=1)      # 每行是"该步的题目"
    r_ = cat([last_ans[:, :1], next_ans], dim=1)              # 每行是"该步的作答"
之后照抄 pyKT 的 w/写/读逻辑，返回 p[:, 1:]（丢掉首列，与框架其他模型一致：
位置 t 的预测 next_problem[t] 用到窗口第 0..t 步的作答，不包含 next_ans[t]，无泄漏）。
"""
import torch
from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_


class DKVMN(Module):
    """动态键值记忆网络。
    dim_s: 记忆槽向量维度（pyKT 默认 200）；size_m: 记忆槽数量（pyKT 默认 50）。
    """

    def __init__(self, pro_max, dim_s=200, size_m=50, dropout=0.2):
        super().__init__()
        self.pro_max = pro_max
        self.dim_s = dim_s
        self.size_m = size_m

        # 键嵌入（按题目）；Mk: 记忆键矩阵; Mv0: 初始记忆值矩阵（可学习）
        self.k_emb_layer = Embedding(self.pro_max, self.dim_s)
        self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
        self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        # 值嵌入：x = 题目 + pro_max * 作答（与框架 DKT 的 interaction 索引一致）
        self.v_emb_layer = Embedding(self.pro_max * 2, self.dim_s)

        # 读：拼接(权重和×记忆值, 当前键) → f 层 → p 层
        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)

        # 写：擦除/加和门
        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, last_problem, last_ans, next_problem, next_ans):
        # 前插一列还原 200 长原窗口（行为窗口第 0..199 步；行 199 的作答不会被任何预测读取）
        q = torch.cat([last_problem[:, :1], next_problem], dim=1)          # (B, T) 题目
        r = torch.cat([last_ans[:, :1].long(), next_ans.long()], dim=1)    # (B, T) 作答 0/1
        batch_size, T = q.shape
        x = q + self.pro_max * r                    # (题目, 作答) 组合索引
        k = self.k_emb_layer(q)                     # 键
        v = self.v_emb_layer(x)                     # 值（当前步交互的向量）

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)   # 初始记忆
        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.t()), dim=-1)  # 相关权重 (B,T,size_m)

        # 写门（pyKT 原样）：e 为擦除门、a 为加和门
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        # 逐步写入：Mv[j] = 写入第 0..j-1 步之后的记忆（Mv 共 T+1 个）
        for et, at, wt in zip(e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)   # (B, T+1, size_m, dim_s)

        # 读：Mv[:, :-1] → 行 j 读到的是"写入 0..j-1 步之后"的记忆（先读后写，无当前步作答泄漏）
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        p = self.p_layer(self.dropout_layer(f))
        p = torch.sigmoid(p).squeeze(-1)
        # 丢首列（行 0 无任何历史），与框架其余模型对齐：P[t] = 对 next_problem[t] 的预测
        return p[:, 1:], None
