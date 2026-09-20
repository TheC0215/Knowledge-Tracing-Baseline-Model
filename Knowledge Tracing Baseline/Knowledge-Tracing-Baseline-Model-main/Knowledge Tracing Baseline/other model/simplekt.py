"""simpleKT（简化的 AKT，无单调注意力/无难度加权的 Rasch 全维向量版）——
对齐 pykt/models/simplekt.py 的 qid 版，只改接口/设备处理。

pyKT 侧约定（train/evaluate）：y = model(dcur)（dcur 内含 qseqs/cseqs/rseqs 与
错位数组），模型 forward 内部把 q/c/r 各"前插一列"还原 200 长原窗口，外部再取
y[:,1:] 与错位标签 rshft 对齐。本端口把这些都收进 forward：q_data(题目)=pid_data
同空间，内部前插一列后跑 pyKT 原版流程，返回 preds[:, 1:]（P[t]=对 next_problem[t]
的预测，只使用窗口 0..t 步作答）。

与 pyKT 的两处有意差异：
1. reset() 只把难度参数 difficult_param 清零（pyKT 按"参数行数==n_pid+1"挑选，
   在 n_question+1 == n_pid+1 的问题级设定下会把 q_embed_diff 一并清零 → u_q≡0 ∧
   d_ct≡0 梯度恒 0、Rasch 支路永久失效；按 pyKT assist2009 数值初始化意图只清难度）。
2. Rasch L2 正则(c_reg_loss) 无处安放 → 略去（l2=1e-5 量级，影响可忽略）。
另外 pyKT 构造器里的 num_layers/nheads/loss1/2/3/start 等参数在本 qid 路径完全未用，
故不保留（见移植报告）。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from utils import CosinePositionalEmbedding


class simpleKT(nn.Module):
    """n_question/n_pid 都等于题目数 pro_max（q 空间 == pid 空间）。
    d_model: 注意力维度；n_blocks: 堆叠块数；final_fc_dim(2): 预测头宽度；
    seq_len: 余弦位置编码长度上限（pyKT 传 200）。
    """

    def __init__(self, pro_max, d_model=256, d_ff=256, n_blocks=2, num_attn_heads=4,
                 dropout=0.1, final_fc_dim=256, final_fc_dim2=256, kq_same=1,
                 separate_qa=False, seq_len=200, l2=1e-5, emb_type="qid"):
        super().__init__()
        self.model_name = "simplekt"
        self.n_question = pro_max
        self.n_pid = pro_max          # 本框架题目即 pid
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model
        if self.n_pid > 0:
            if emb_type.find("scalar") != -1:
                self.difficult_param = nn.Embedding(self.n_pid + 1, 1)   # 标量难度 u_q
            else:
                self.difficult_param = nn.Embedding(self.n_pid + 1, embed_l)  # 全维难度向量 u_q
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)  # d_ct
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

        if emb_type.startswith("qid"):
            # n_question+1, d_model（embedding 行数与 pyKT 一致：行 0 是 pad 占位行）
            self.q_embed = nn.Embedding(self.n_question, embed_l)   # c_ct
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
            else:  # false default
                self.qa_embed = nn.Embedding(2, embed_l)            # g_rt（按作答 0/1 取值）

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=pro_max, n_blocks=n_blocks, n_heads=num_attn_heads,
                                  dropout=dropout, d_model=d_model,
                                  d_feature=d_model / num_attn_heads, d_ff=d_ff,
                                  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )

        self.reset()

    def reset(self):
        """仅清零难度参数 u_q（原因见文件头注释；pyKT 原 reset 按行数匹配，会连
        q_embed_diff 一起清零导致 Rasch 支路零梯度死亡）。"""
        for p in self.parameters():
            if p is self.difficult_param.weight:
                torch.nn.init.constant_(p, 0.)

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
        # 前插一列还原 200 长原窗口（q 与 c 同空间 → pid_data = q_data；行 j ↔ 窗口第 j 步）
        pid_data = torch.cat([last_problem[:, :1], next_problem], dim=1)
        q_data = pid_data
        target = torch.cat([last_ans[:, :1].long(), next_ans.long()], dim=1)

        emb_type = self.emb_type

        # Batch First
        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        if self.n_pid > 0 and emb_type.find("norasch") == -1:  # have problem id
            if emb_type.find("aktrasch") == -1:
                # 只有 question 流做 Rasch 调制（'qid' 不含 'aktrasch' → qa 流不加，与 pyKT 一致）
                q_embed_diff_data = self.q_embed_diff(q_data)    # d_ct
                pid_embed_data = self.difficult_param(pid_data)  # u_q
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # u_q * d_ct + c_ct # question encoder
            else:
                # 'aktrasch' 分支（本项目 emb='qid' 用不到，保留结构以对齐 pyKT）
                q_embed_diff_data = self.q_embed_diff(q_data)
                pid_embed_data = self.difficult_param(pid_data)
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  # u_q * d_ct + c_ct

                qa_embed_diff_data = self.qa_embed_diff(target)
                qa_embed_data = qa_embed_data + pid_embed_data * \
                    (qa_embed_diff_data + q_embed_diff_data)  # + u_q * (h_rt + d_ct)

        # BS, seqlen, d_model → decoder
        d_output = self.model(q_embed_data, qa_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)
        # 丢弃首列：P[t] = 对 next_problem[t] 的预测，只用到窗口 0..t 步的作答
        return preds[:, 1:], None


class Architecture(nn.Module):
    """简单两层（n_blocks 个）mask=0 的 TransformerLayer：query=题目流 x、value=作答流 y；
    y 未经独立编码，直接由原始交互嵌入 + 余弦位置编码构成（simpleKT 与 AKT 的差异）。
    """

    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'simplekt'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed   # 作答交互流（value）
        x = q_pos_embed    # 题目流（query）

        # encoder：mask=0 → 行 r 看不到当前步自己的作答（value 行 <= r-1），
        # 首行分布被置 0（"第一题只有 question 信息、无 qa 信息"）
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        return x


class TransformerLayer(nn.Module):
    """多头注意力 + 两层 LayerNorm + 两层 FFN（带残差与 dropout，pyKT 原样）。"""

    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        """
        mask=0：只能看过去；mask=1：可看当前与过去（本文件只用 0）。
        返回经注意力与（可选）FFN 后的 query。
        """
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)
        if mask == 0:  # 需要 zero-padding：当前步作答不可见
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1(query2)   # 残差 1
        query = self.layer_norm1(query)         # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout(  # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2(query2)  # 残差
            query = self.layer_norm2(query)        # layer norm
        return query


class MultiHeadAttention(nn.Module):
    """多头注意力（无单调 gammas，simpleKT 不含量化/距离衰减机制）。"""

    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
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

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)

        # perform linear operation and split into h heads
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

        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output = self.out_proj(concat)
        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    """缩放点积因果注意力（pyKT 原样，仅按输入张量取设备）。"""
    dev = q.device
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # BS, heads, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    # 行 r 只能 attend 键行 < r（mask=0 时 nopeek 为 k=0 的上三角）
    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        # 首行 score 置 0（严格因果下行 0 无可 attend 键，不处理会成均匀分布泄漏未来）
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(dev)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output
