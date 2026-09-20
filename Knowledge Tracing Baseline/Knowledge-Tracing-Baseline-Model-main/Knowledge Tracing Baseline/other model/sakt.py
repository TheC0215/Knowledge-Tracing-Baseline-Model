"""SAKT（自注意力知识追踪）——对齐 pykt/models/sakt.py 的 qid 版，只改接口/设备处理。

pyKT 侧调用约定（train/evaluate 相同）：y = model(c, r, cshft)，c/r/cshft 都是
199 长的错位窗口数组（c[t]=窗口第 t 步、cshft[t]=窗口第 t+1 步），预测 y 直接与
错位标签 rshft 对齐，无需切片。
模型语义：行 t 用 query=exercise_emb(下一题 cshft[t]) 对键值=带位置的交互序列
x[t']=(q[t'], r[t'])（t' <= t，严格因果：ut_mask 挡住行 t 对角线上方）做注意力，
输出 sigmoid 后即 P(答对 cshft[t]) —— 即对"窗口第 t+1 题"的预测只使用窗口第
0..t 步的作答（r[t] 是最新的历史作答，r[t+1] 不可见），无泄漏。

本端口映射：q=last_problem、r=last_ans、qry=next_problem，三者均为 199 长 → 直接
forward，P (B, 199) 与框架 next_ans 对齐。与 pyKT 的唯一差异：把全局 device 改为
按输入张量取设备（ut_mask 显式放 q 所在设备），避免 GPU 上报错。
"""
import torch
from torch.nn import Module, Embedding, Linear, MultiheadAttention, LayerNorm, Dropout

# 共享小工具（transformer_FFN / ut_mask / pos_encode / get_clones 已收敛到 utils.py）
from utils import transformer_FFN, pos_encode, ut_mask, get_clones


class SAKT(Module):
    """num_c: 本题设定下的题目总数 pro_max（问题即"概念"）；
    seq_len: 位置编码最大长度（pyKT 传 maxlen=200，实际输入恒为 199，索引不越界）。
    """

    def __init__(self, num_c, seq_len=200, emb_size=256, num_attn_heads=8,
                 dropout=0.2, num_en=1, emb_type="qid"):
        super().__init__()
        self.model_name = "sakt"
        self.emb_type = emb_type

        self.num_c = num_c          # 题目数（与 pyKT 的 num_c 概念数对齐）
        self.seq_len = seq_len
        self.emb_size = emb_size
        self.num_attn_heads = num_attn_heads
        self.dropout = dropout
        self.num_en = num_en        # 脚本配置为 1（pyKT 类默认 2，per-script 覆盖为 1）

        # qid 分支：num_c*2 行 = (题目, 作答 0/1) 组合交互嵌入；题目嵌入用于"下一题"查询
        self.interaction_emb = Embedding(num_c * 2, emb_size)
        self.exercise_emb = Embedding(num_c, emb_size)
        self.position_emb = Embedding(seq_len, emb_size)

        self.blocks = get_clones(Blocks(emb_size, num_attn_heads, dropout), self.num_en)

        self.dropout_layer = Dropout(dropout)
        self.pred = Linear(self.emb_size, 1)

    def base_emb(self, q, r, qry):
        # 交互索引 x = 题目 + num_c * 作答（与框架 DKT 的 interaction 索引一致）
        x = q + self.num_c * r
        qshftemb, xemb = self.exercise_emb(qry), self.interaction_emb(x)

        # 只有交互流加位置编码，查询流（下一题）不加（pyKT 原样）
        posemb = self.position_emb(pos_encode(xemb.shape[1], device=q.device))
        xemb = xemb + posemb
        return qshftemb, xemb

    def forward(self, last_problem, last_ans, next_problem, next_ans):
        q = last_problem           # (B, 199) 当前窗口题目
        r = last_ans               # (B, 199) 当前窗口作答 0/1（float/long 均可，内部当 long 用）
        qry = next_problem         # (B, 199) 下一题（预测目标）

        emb_type = self.emb_type
        qshftemb, xemb = None, None
        if emb_type.startswith("qid"):
            qshftemb, xemb = self.base_emb(q, r.long(), qry)

        for i in range(self.num_en):
            xemb = self.blocks[i](qshftemb, xemb, xemb)

        p = torch.sigmoid(self.pred(self.dropout_layer(xemb))).squeeze(-1)
        return p, None


class Blocks(Module):
    """单层自注意力块：因果自注意力 → LN → FFN → LN（pyKT 原样）。"""

    def __init__(self, emb_size, num_attn_heads, dropout) -> None:
        super().__init__()

        self.attn = MultiheadAttention(emb_size, num_attn_heads, dropout=dropout)
        self.attn_dropout = Dropout(dropout)
        self.attn_layer_norm = LayerNorm(emb_size)

        self.FFN = transformer_FFN(emb_size, dropout)
        self.FFN_dropout = Dropout(dropout)
        self.FFN_layer_norm = LayerNorm(emb_size)

    def forward(self, q=None, k=None, v=None):
        # nn.MultiheadAttention 用 (L, B, E) 布局
        q, k, v = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
        # 严格因果：行 t 只能 attend 键 0..t（含自己，即"当前交互已发生"）；
        # 预测的是下一题，所以行 t 用交互 <= t 预测 q[t+1] 答对概率，无泄漏
        causal_mask = ut_mask(seq_len=k.shape[0], device=k.device)
        attn_emb, _ = self.attn(q, k, v, attn_mask=causal_mask)

        attn_emb = self.attn_dropout(attn_emb)
        attn_emb, q = attn_emb.permute(1, 0, 2), q.permute(1, 0, 2)

        attn_emb = self.attn_layer_norm(q + attn_emb)

        emb = self.FFN(attn_emb)
        emb = self.FFN_dropout(emb)
        emb = self.FFN_layer_norm(attn_emb + emb)
        return emb
