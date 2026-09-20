"""AKT（Context-Aware Knowledge Tracing，带单调注意力/Rasch 题目难度）——
对齐 pykt/models/akt.py，只改接口与设备处理。

pyKT 侧约定：y, reg_loss = model(cc, cr, cq)，其中 cc/cr/cq 是"前插一列"还原出的
200 长原窗口（行 j ↔ 原窗口第 j 步交互），训练/评测取 y[:,1:]。
模型内部 mask=0 的层对 value 流做 zero_pad（行 j 的注意力分布用行 j-1 的分布、且只能看
<=j-1 的键）+ 严格因果注意力 → 行 j 的预测是"作答第 j 步之前"对题目 cc[j] 的预测，
行 j 自己（及以后）的作答不会被看到 → 无泄漏。

本端口 forward 里同样前插一列：
    q_data = pid_data = cat([last_problem[:, :1], next_problem], dim=1)  （q 与 pid 同一空间）
    target = cat([last_ans[:, :1], next_ans], dim=1)
返回 preds[:, 1:]（丢弃首列），位置 t 的预测对应当前步下一题 next_problem[t]，
只使用窗口 0..t 步作答 —— 与框架 DKT 等模型语义一致。

与 pyKT 的两处有意差异（详见类内注释/报告）：
1. reset() 只把难度参数 difficult_param 清零。pyKT 用"参数行数 == n_pid+1"来挑参数，
   在概念级数据上 num_c(概念数) != num_q(题数) 时只会命中 difficult_param；本题设定下
   q 空间 == pid 空间 == 题目空间（n_question+1 == n_pid+1），照抄会把 q_embed_diff
   也清零，且 u_q ≡ 0、d_ct ≡ 0 时 Rasch 支路梯度恒为 0（等于被永久关掉）。因此按
   pyKT 在 assist2009 上的数值初始化意图：仅难度 u_q 置 0，d_ct 保持随机初始化。
2. 原代码返回 (preds, c_reg_loss)（Rasch L2 正则）。本框架 run.py 只用第一个返回值
   算 BCE 损失，第二项无处安放 → 略去 c_reg_loss（l2=1e-5 量级，影响可忽略）。
"""
import math
from enum import IntEnum

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class AKT(nn.Module):
    """AKT：n_question/n_pid 在本框架都等于题目数 pro_max（q_data 与 pid_data 同一空间）。
    d_model: 注意力维度；n_blocks: 堆叠块数（blocks_1 n_blocks 层编码作答、blocks_2 2*n_blocks 层）；
    kq_same=1 表示 query 与 key 共用投影；final_fc_dim: 预测头第一层宽度。
    """

    def __init__(self, pro_max, d_model=256, n_blocks=4, dropout=0.2, d_ff=512,
                 kq_same=1, final_fc_dim=512, num_attn_heads=8, separate_qa=False,
                 emb_type="qid"):
        super().__init__()
        """
        Input:
            pro_max: 题目总数（本框架 == 每步"概念"空间 == pid 空间）
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully connected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "akt"
        self.n_question = pro_max
        self.n_pid = pro_max          # 本框架题目即 pid（Rasch 难度也按题目）
        self.dropout = dropout
        self.kq_same = kq_same
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)  # 题目难度 u_q（标量）
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)  # question emb 的难度调制向量 d_ct
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)  # interaction 难度调制向量

        if emb_type.startswith("qid"):
            # n_question+1, d_model
            self.q_embed = nn.Embedding(self.n_question, embed_l)   # c_ct
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)  # interaction emb
            else:  # false default
                self.qa_embed = nn.Embedding(2, embed_l)            # e_(ct,rt) 中只按作答 0/1 取值

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                  d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,
                                  kq_same=self.kq_same, emb_type=emb_type)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(256, 1)
        )
        self.reset()

    def reset(self):
        """难度参数清零初始化（AKT 论文：难度 u_q 从 0 起步，d_ct 保持随机）。
        注意：pyKT 用 p.size(0) == n_pid+1 挑选参数，那在概念级数据(num_c != num_q)下
        只会命中 difficult_param；本框架 n_question+1 == n_pid+1，若照抄会把 q_embed_diff
        也清零（u_q≡0 ∧ d_ct≡0 → 梯度恒 0，Rasch 支路被永久关闭），故改为只清零难度。
        """
        torch.nn.init.constant_(self.difficult_param.weight, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen, d_model # c_ct
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct + g_rt = e_(ct,rt)
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, last_problem, last_ans, next_problem, next_ans):
        # 前插一列还原原窗口（行为窗口第 0..199 步；预测时行 j 看不到行 >= j 的作答）
        q_data = torch.cat([last_problem[:, :1], next_problem], dim=1)       # (B, T)
        target = torch.cat([last_ans[:, :1].long(), next_ans.long()], dim=1)
        pid_data = q_data    # 本框架 pid == 题目本身

        emb_type = self.emb_type
        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        pid_embed_data = None
        if self.n_pid > 0:  # have problem id
            q_embed_diff_data = self.q_embed_diff(q_data)  # d_ct
            pid_embed_data = self.difficult_param(pid_data)  # u_q（标量，广播到最后一维）
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # u_q*d_ct + c_ct # question encoder

            qa_embed_diff_data = self.qa_embed_diff(target)  # f_(ct,rt)（按作答 0/1 取值）
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    qa_embed_diff_data  # u_q*f_(ct,rt) + e_(ct,rt)
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data + q_embed_diff_data)  # + u_q*(h_rt+d_ct)
        # BS, seqlen, d_model
        # Pass to the decoder
        # output shape BS, seqlen, d_model
        d_output = self.model(q_embed_data, qa_embed_data, pid_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        preds = torch.sigmoid(output)
        # 丢弃首列（行 0 无历史作答），对齐框架：P[t] = 对 next_problem[t] 答对的预测
        return preds[:, 1:], None


class Architecture(nn.Module):
    """两层式 Transformer：blocks_1 编码 (question, answer) 交互流 y；
    blocks_2 中第一层做 question 自身 self-attention（apply_pos=False），
    之后交替以 y 为 value（zero_pad 保证看不到当前步作答）。
    """

    def __init__(self, n_blocks, d_model, d_feature, d_ff, n_heads, dropout, kq_same, emb_type):
        super().__init__()
        self.d_model = d_model

        self.blocks_1 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same,
                             emb_type=emb_type)
            for _ in range(n_blocks)
        ])
        self.blocks_2 = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                             d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same,
                             emb_type=emb_type)
            for _ in range(n_blocks * 2)
        ])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data):
        y = qa_embed_data        # 作答交互流
        x = q_embed_data         # 题目流

        # encoder：编码 qa，对 0~t-1 时刻前的 qa 信息进行编码（yt^）
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, pdiff=pid_embed_data)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:  # peek current question（只看题目，不掺杂作答）
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False, pdiff=pid_embed_data)
                flag_first = False
            else:  # don't peek current response
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True, pdiff=pid_embed_data)
                # mask=0：不能看到当前的 response；value 流 zero_pad 后第一行全 0，
                # 实现"第一题只有 question 信息，无 qa 信息"
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    """单层 Transformer 块：多头注意力 + LayerNorm + FFN（含残差与 dropout，pyKT 原样）。"""

    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same, emb_type):
        super().__init__()
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None):
        """
        mask=0：只能看过去；mask=1：可看当前与过去。
        返回经注意力与（可选）FFN 后的 query。
        """
        seqlen = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)
        if mask == 0:  # 需要 zero-padding：当前步作答不可见
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, pdiff=pdiff)
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, pdiff=pdiff)

        query = query + self.dropout1(query2)   # 残差
        query = self.layer_norm1(query)         # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout(  # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)  # 残差
            query = self.layer_norm2(query)        # lay norm
        return query


class MultiHeadAttention(nn.Module):
    """多头注意力（单调注意力用 gammas 加权距离衰减，pyKT 原样）。"""

    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, emb_type="qid"):
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))   # 每个头的单调衰减速率
        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad, pdiff=None):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        gammas = self.gammas
        if self.emb_type.find("pdiff") == -1:
            # 单调衰减只由 gamma 控制（题目难度不参与距离调制；'qid' 恒走此分支，与 pyKT 一致）
            pdiff = None
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad, gammas, pdiff)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, gamma=None, pdiff=None):
    """带单调距离衰减的注意力（AKT 论文公式 1；pyKT 原样，仅把全局 device 改为按输入张量）。
    衰减系数由每个头可学习的 gamma（Softplus 后取负）与"未来距离"共同决定：
    先对 mask 之后的分布做累计求和，dist_scores 惩罚 attend 到较远未来位置。
    """
    dev = q.device
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # BS, heads, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(dev)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)          # BS, heads, seqlen, seqlen
        scores_ = scores_ * mask.float().to(dev)
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, heads, sl, sl
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(dev)
        # bs, heads, sl, sl：已 attend 的总概率 × 位置距离（惩罚"跨过"远期位置）
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()

    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1, heads, 1, 1：一个头一个 gamma（论文里的 theta）
    # 先 exp(gamma*distance) 再 clamp 到 [1e-5, 1e5]
    if pdiff is None:
        total_effect = torch.clamp(torch.clamp(
            (dist_scores * gamma).exp(), min=1e-5), max=1e5)
    else:
        diff = pdiff.unsqueeze(1).expand(pdiff.shape[0], dist_scores.shape[1],
                                         pdiff.shape[1], pdiff.shape[2])
        diff = diff.sigmoid().exp()
        total_effect = torch.clamp(torch.clamp(
            (dist_scores * gamma * diff).exp(), min=1e-5), max=1e5)

    scores = scores * total_effect
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    if zero_pad:
        # 第一行 score 置 0（丢弃行 0 的原始分布；行 0 在严格因果下无任何可 attend 的键，
        # 不加处理会退化成均匀分布泄漏未来作答）。行 r>=1 保持原位：行 r 只能看键行 < r，
        # 即 value 行 <= r-1 —— 看不到当前步自己的作答 r[r]（与 pyKT 的 "第一行score置0" 一致）
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(dev)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output
