"""UKT（不确定性感知知识追踪，随机嵌入 mean/cov + Wasserstein 注意力）——
对齐 pykt/models/ukt.py（emb_type='stoc_qid'、atten_type='w2'、use_CL=False），
只改接口/设备处理，并按脚本配置删除对比学习(CL)全部相关代码。

pyKT 侧约定（train/evaluate）：y = model(dcur)（forward 内部把 q/c/r 各"前插一列"
还原 200 长原窗口），外部取 y[:,1:] 与错位标签对齐。本端口把这些收进 forward：
q 流（pid）与 c 流（题目/概念）在本框架同属题目空间 → pid_data = q_data，
前插一列后跑 pyKT 原版流程（含 TransformerLayer 第 430-433 行的 LN 施加在
query2 上、残差加和结果被覆盖的原版 bug —— 为保证与原实现行为逐位一致，
该 quirk 原样保留），返回 preds[:, 1:]。

删除清单（use_CL=False 下 pyKT 本身不走的路径）：WassersteinNCELoss、
d2s_1overx、wasserstein_distance、r_aug/shft_r_aug 增广数据、masks 池化、
temp 不确定性度量（仅用于 CL 汇报）、y2/y3（恒 0 占位）、use_uncertainty_aug 等。
保留：uattention（w2 默认）与带 gammas 的 attention（dp 备选）两个实现、
Architecture 的 ELU+1 处理、stoc_qid 下 qa 流不加 Rasch 的结构、reset 偏差
（只清难度参数，原因同 simplekt.py 文件头注释）。
"""
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import xavier_uniform_, constant_

from utils import CosinePositionalEmbedding


class UKT(nn.Module):
    """n_question/n_pid 都等于题目数 pro_max（q 流 == pid 流）。
    mean/cov 两组嵌入表达"随机知识状态"，注意力用 Wasserstein 距离（'w2'，
    即 uattention）；cov 流经 ELU+1 保证正定。
    """

    def __init__(self, pro_max, d_model=256, d_ff=512, n_blocks=4, num_attn_heads=8,
                 dropout=0.2, final_fc_dim=256, final_fc_dim2=256, kq_same=1,
                 separate_qa=False, seq_len=200, use_CL=False, atten_type='w2',
                 l2=1e-5, emb_type="stoc_qid"):
        super().__init__()
        self.model_name = "ukt"
        self.n_question = pro_max
        self.dropout = dropout
        self.kq_same = kq_same
        self.n_pid = pro_max          # 本框架题目即 pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.use_CL = use_CL          # 恒 False（脚本默认 use_CL=0；不再支持 CL 路径）
        self.atten_type = atten_type  # 'w2'（默认，Wasserstein 注意力）或 'dp'（点积+单调衰减）

        embed_l = d_model

        self.embed_l = d_model
        if self.n_pid > 0:
            if emb_type.find("scalar") != -1:
                self.difficult_param = nn.Embedding(self.n_pid + 1, 1)   # 标量难度
            else:
                self.difficult_param = nn.Embedding(self.n_pid + 1, embed_l)  # 全维难度
            self.q_embed_diff = nn.Embedding(self.n_question + 1, embed_l)  # d_ct
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l)

        if emb_type.startswith("qid") or emb_type.startswith("stoc"):
            # mean embedding / covariance embedding
            self.mean_q_embed = nn.Embedding(self.n_question, embed_l)
            self.cov_q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa:
                self.mean_qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
                self.cov_qa_embed = nn.Embedding(2 * self.n_question + 1, embed_l)
            else:  # false default
                self.mean_qa_embed = nn.Embedding(2, embed_l)
                self.cov_qa_embed = nn.Embedding(2, embed_l)

        # Architecture Object. It contains stack of attention block
        self.model = Architecture(n_question=pro_max, n_blocks=n_blocks, n_heads=num_attn_heads,
                                  dropout=dropout, d_model=d_model,
                                  d_feature=d_model / num_attn_heads, d_ff=d_ff,
                                  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)

        # 预测头输入 = mean/cov 输出 + mean/cov question 嵌入（共 4*d_model）
        self.out = nn.Sequential(
            nn.Linear(embed_l + embed_l + embed_l + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )
        self.reset()

    def reset(self):
        """仅清零难度参数（原因同 simplekt：按 pyKT 行数匹配的 reset 会把
        q_embed_diff 一起清零 → Rasch 支路零梯度死亡）。"""
        for p in self.parameters():
            if p is self.difficult_param.weight:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        # 式(1)：随机嵌入 = mean + cov；qa 嵌入为 g_rt + question 嵌入（不作答维）
        q_mean_embed_data = self.mean_q_embed(q_data)
        q_cov_embed_data = self.cov_q_embed(q_data)

        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_mean_embed_data = self.mean_qa_embed(qa_data)
            qa_cov_embed_data = self.cov_qa_embed(qa_data)
        else:
            qa_mean_embed_data = self.mean_qa_embed(target) + q_mean_embed_data
            qa_cov_embed_data = self.cov_qa_embed(target) + q_cov_embed_data

        return q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data

    def forward(self, last_problem, last_ans, next_problem, next_ans):
        # 前插一列还原 200 长原窗口（q 与 c 同空间 → pid_data = q_data）
        pid_data = torch.cat([last_problem[:, :1], next_problem], dim=1)
        q_data = pid_data
        target = torch.cat([last_ans[:, :1].long(), next_ans.long()], dim=1)

        emb_type = self.emb_type
        # 随机嵌入（use_CL=False → 无增广支路）
        q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data = \
            self.base_emb(q_data, target)

        if self.n_pid > 0 and emb_type.find("norasch") == -1:  # have problem id
            if emb_type.find("aktrasch") == -1:
                # 'stoc_qid'：难度只调制 question 流的 mean 与 cov（qa 流不加，与 pyKT 一致）
                q_embed_diff_data = self.q_embed_diff(q_data)      # d_ct
                pid_embed_data = self.difficult_param(pid_data)    # u_q
                q_mean_embed_data = q_mean_embed_data + pid_embed_data * q_embed_diff_data
                q_cov_embed_data = q_cov_embed_data + pid_embed_data * q_embed_diff_data
            else:
                # 'aktrasch' 分支（本项目用不到，保留结构以对齐 pyKT）
                q_embed_diff_data = self.q_embed_diff(q_data)
                pid_embed_data = self.difficult_param(pid_data)
                q_mean_embed_data = q_mean_embed_data + pid_embed_data * q_embed_diff_data
                q_cov_embed_data = q_cov_embed_data + pid_embed_data * q_embed_diff_data

                qa_embed_diff_data = self.qa_embed_diff(target)
                qa_mean_embed_data = qa_mean_embed_data + pid_embed_data * \
                    (qa_embed_diff_data + q_embed_diff_data)
                qa_cov_embed_data = qa_cov_embed_data + pid_embed_data * \
                    (qa_embed_diff_data + q_embed_diff_data)

        # BS, seqlen, d_model → decoder（mean/cov 双流）
        mean_d_output, cov_d_output = self.model(
            q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data,
            self.atten_type)

        if emb_type == "stoc_qid":
            concat_q = torch.cat([mean_d_output, cov_d_output,
                                  q_mean_embed_data, q_cov_embed_data], dim=-1)
        else:
            concat_q = torch.cat([mean_d_output, mean_d_output,
                                  q_cov_embed_data, q_cov_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)
        m = nn.Sigmoid()
        preds = m(output)
        # 丢弃首列：P[t] = 对 next_problem[t] 的预测，只用到窗口 0..t 步的作答
        return preds[:, 1:], None


class Architecture(nn.Module):
    """mean/cov 双流位置编码（余弦，各自一份对象——pe 被 sin/cos 完全覆盖，两者数值相同）
    + ELU+1 正定化（式(3)）+ n_blocks 层 mask=0 的 TransformerLayer。
    """

    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        self.position_mean_embeddings = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)
        self.position_cov_embeddings = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

        if model_type in {'ukt'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])

    def forward(self, q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data,
                qa_cov_embed_data, atten_type='w2'):
        # 式(2)：位置编码
        mean_q_posemb = self.position_mean_embeddings(q_mean_embed_data)
        cov_q_posemb = self.position_cov_embeddings(q_cov_embed_data)

        q_mean_embed_data = q_mean_embed_data + mean_q_posemb
        q_cov_embed_data = q_cov_embed_data + cov_q_posemb

        qa_mean_posemb = self.position_mean_embeddings(qa_mean_embed_data)
        qa_cov_posemb = self.position_cov_embeddings(qa_cov_embed_data)

        qa_mean_embed_data = qa_mean_embed_data + qa_mean_posemb
        qa_cov_embed_data = qa_cov_embed_data + qa_cov_posemb

        # 式(3)：cov 用 ELU+1 保持正定
        elu_act = torch.nn.ELU()
        q_cov_embed_data = elu_act(q_cov_embed_data) + 1
        qa_cov_embed_data = elu_act(qa_cov_embed_data) + 1

        y_mean = qa_mean_embed_data   # 作答交互流 mean（value）
        y_cov = qa_cov_embed_data     # 作答交互流 cov（value）
        x_mean = q_mean_embed_data    # 题目流 mean（query）
        x_cov = q_cov_embed_data      # 题目流 cov（query）

        # encoder：mask=0 → 行 r 看不到当前步自己的作答（首行分布置 0）
        for block in self.blocks_2:
            x_mean, x_cov = block(mask=0, query_mean=x_mean, query_cov=x_cov,
                                  key_mean=x_mean, key_cov=x_cov,
                                  values_mean=y_mean, values_cov=y_cov,
                                  atten_type=atten_type, apply_pos=True)
        return x_mean, x_cov


class TransformerLayer(nn.Module):
    """双流（mean/cov）Transformer 层。注意：cov 流在每个 LN 前都经过 ELU+1；
    末段两个 LN 施加在 query2（FFN 输出）而不是残差加和结果上 —— 这是 pyKT
    ukt.py 第 430-433 行的原版写法（残差加和实际被覆盖丢弃），为保证与 pyKT
    逐位一致，此处原样保留该行为。
    """

    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two dropout layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.mean_linear1 = nn.Linear(d_model, d_ff)
        self.cov_linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.mean_linear2 = nn.Linear(d_ff, d_model)
        self.cov_linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation2 = nn.ELU()

    def forward(self, mask, query_mean, query_cov, key_mean, key_cov,
                values_mean, values_cov, atten_type='w2', apply_pos=True):
        seqlen, batch_size = query_mean.size(1), query_mean.size(0)

        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query_mean.device)

        if mask == 0:  # 需要 zero-padding：当前步作答不可见
            query2_mean, query2_cov = self.masked_attn_head(
                query_mean, query_cov, key_mean, key_cov, values_mean, values_cov,
                mask=src_mask, atten_type=atten_type, zero_pad=True)
        else:
            query2_mean, query2_cov = self.masked_attn_head(
                query_mean, query_cov, key_mean, key_cov, values_mean, values_cov,
                mask=src_mask, atten_type=atten_type, zero_pad=False)

        # 残差 + LN1（cov 流 LN1 前 ELU+1）
        query_mean = query_mean + self.dropout1(query2_mean)
        query_cov = query_cov + self.dropout1(query2_cov)

        query_mean = self.layer_norm1(query_mean)
        query_cov = self.layer_norm1(self.activation2(query_cov) + 1)
        # 式(6)：FFN（mean/cov 各自独立的两层 MLP）
        if apply_pos:
            query2_mean = self.mean_linear2(self.dropout(
                self.activation(self.mean_linear1(query_mean))))
            query2_cov = self.cov_linear2(self.dropout(
                self.activation(self.cov_linear1(query_cov))))

            query_mean = query_mean + self.dropout2(query2_mean)
            query_cov = query_cov + self.dropout2(query2_cov)
            # 原版 quirk：LN2 作用在 query2 上（残差加和结果未参与输出），原样保留
            query_mean = self.layer_norm2(query2_mean)
            query_cov = self.layer_norm2(self.activation2(query2_cov) + 1)

        return query_mean, query_cov


class MultiHeadAttention(nn.Module):
    """双流多头注意力：'w2' → uattention（Wasserstein 距离打分 + scores**2 聚合 cov）；
    'dp' → attention（mean/cov 各自点积 + 单调距离衰减 gammas）。"""

    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.activation = nn.ELU()
        self.v_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_cov_linear = nn.Linear(d_model, d_model, bias=bias)

        self.k_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_cov_linear = nn.Linear(d_model, d_model, bias=bias)

        if kq_same is False:
            self.q_mean_linear = nn.Linear(d_model, d_model, bias=bias)
            self.q_cov_linear = nn.Linear(d_model, d_model, bias=bias)

        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_mean_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_cov_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))   # 'dp' 用的单调衰减参数
        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_mean_linear.weight)
        xavier_uniform_(self.k_cov_linear.weight)

        xavier_uniform_(self.v_mean_linear.weight)
        xavier_uniform_(self.v_cov_linear.weight)

        if self.kq_same is False:
            xavier_uniform_(self.q_mean_linear.weight)
            xavier_uniform_(self.q_cov_linear.weight)

        if self.proj_bias:
            constant_(self.k_mean_linear.bias, 0.)
            constant_(self.k_cov_linear.bias, 0.)

            constant_(self.v_mean_linear.bias, 0.)
            constant_(self.v_cov_linear.bias, 0.)

            if self.kq_same is False:
                constant_(self.q_mean_linear.bias, 0.)
                constant_(self.q_cov_linear.bias, 0.)

            constant_(self.out_mean_proj.bias, 0.)
            constant_(self.out_cov_proj.bias, 0.)

    def forward(self, q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, mask, atten_type, zero_pad):
        bs = q_mean.size(0)

        # 线性投影并拆多头
        k_mean = self.k_mean_linear(k_mean).view(bs, -1, self.h, self.d_k)
        k_cov = self.k_cov_linear(k_cov).view(bs, -1, self.h, self.d_k)

        if self.kq_same is False:
            q_mean = self.q_mean_linear(q_mean).view(bs, -1, self.h, self.d_k)
            q_cov = self.q_cov_linear(q_cov).view(bs, -1, self.h, self.d_k)
        else:
            q_mean = self.k_mean_linear(q_mean).view(bs, -1, self.h, self.d_k)
            q_cov = self.k_cov_linear(q_cov).view(bs, -1, self.h, self.d_k)

        v_mean = self.v_mean_linear(v_mean).view(bs, -1, self.h, self.d_k)
        v_cov = self.v_cov_linear(v_cov).view(bs, -1, self.h, self.d_k)

        k_mean = k_mean.transpose(1, 2)
        q_mean = q_mean.transpose(1, 2)
        v_mean = v_mean.transpose(1, 2)
        k_cov = k_cov.transpose(1, 2)
        q_cov = q_cov.transpose(1, 2)
        v_cov = v_cov.transpose(1, 2)

        # calculate attention using function we will define next
        gammas = self.gammas
        if atten_type == 'w2':
            scores_mean, scores_cov = uattention(
                q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, self.d_k,
                mask, self.dropout, zero_pad, gammas)
        elif atten_type == 'dp':
            scores_mean, scores_cov = attention(
                q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, self.d_k,
                mask, self.dropout, zero_pad, gammas)

        # concatenate heads and put through final linear layer
        concat_mean = scores_mean.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        concat_cov = scores_cov.transpose(1, 2).contiguous().view(bs, -1, self.d_model)

        output_mean = self.out_mean_proj(concat_mean)
        output_cov = self.out_cov_proj(concat_cov)

        return output_mean, output_cov


def attention(q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, d_k, mask, dropout, zero_pad, gamma):
    """'dp'：mean/cov 各自缩放点积 + 每头 gamma 的单调距离衰减（pyKT 原样）。"""
    dev = q_mean.device
    scores_mean = torch.matmul(q_mean, k_mean.transpose(-2, -1)) / math.sqrt(d_k)
    scores_cov = torch.matmul(q_cov, k_cov.transpose(-2, -1)) / math.sqrt(d_k)

    bs, head, seqlen = scores_mean.size(0), scores_mean.size(1), scores_mean.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(dev)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_mean_ = scores_mean.masked_fill(mask == 0, -1e32)
        scores_cov_ = scores_cov.masked_fill(mask == 0, -1e32)

        scores_mean_ = F.softmax(scores_mean_, dim=-1)
        scores_cov_ = F.softmax(scores_cov_, dim=-1)

        scores_mean_ = scores_mean_ * mask.float().to(dev)
        scores_cov_ = scores_cov_ * mask.float().to(dev)

        distcum_scores_mean = torch.cumsum(scores_mean_, dim=-1)
        distcum_scores_cov = torch.cumsum(scores_cov_, dim=-1)

        disttotal_scores_mean = torch.sum(scores_mean_, dim=-1, keepdim=True)
        disttotal_scores_cov = torch.sum(scores_cov_, dim=-1, keepdim=True)

        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(dev)

        dist_scores_mean = torch.clamp(
            (disttotal_scores_mean - distcum_scores_mean) * position_effect, min=0.)
        dist_scores_cov = torch.clamp(
            (disttotal_scores_cov - distcum_scores_cov) * position_effect, min=0.)

        dist_scores_mean = dist_scores_mean.sqrt().detach()
        dist_scores_cov = dist_scores_cov.sqrt().detach()

    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1, heads, 1, 1：每头一个衰减参数

    total_effect_mean = torch.clamp(torch.clamp(
        (dist_scores_mean * gamma).exp(), min=1e-5), max=1e5)
    total_effect_cov = torch.clamp(torch.clamp(
        (dist_scores_cov * gamma).exp(), min=1e-5), max=1e5)

    scores_mean = scores_mean * total_effect_mean
    scores_cov = scores_cov * total_effect_cov

    scores_mean.masked_fill_(mask == 0, -1e32)
    scores_cov.masked_fill_(mask == 0, -1e32)

    scores_mean = F.softmax(scores_mean, dim=-1)
    scores_cov = F.softmax(scores_cov, dim=-1)

    if zero_pad:
        # 首行 score 置 0（严格因果下行 0 无可 attend 键）
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(dev)
        scores_mean = torch.cat([pad_zero, scores_mean[:, :, 1:, :]], dim=2)
        scores_cov = torch.cat([pad_zero, scores_cov[:, :, 1:, :]], dim=2)

    scores_mean = dropout(scores_mean)
    scores_cov = dropout(scores_cov)

    output_mean = torch.matmul(scores_mean, v_mean)
    output_cov = torch.matmul(scores_cov, v_cov)
    return output_mean, output_cov


def uattention(q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, d_k, mask, dropout, zero_pad, gamma):
    """'w2'（默认）：score = -Wasserstein 距离/sqrt(d_k)（式(4) 的矩阵化版本），
    再加每头 gamma 的单调距离衰减；cov 流用 score**2 聚合（二阶矩），pyKT 原样。"""
    dev = q_mean.device
    scores = (-wasserstein_distance_matmul(q_mean, q_cov, k_mean, k_cov)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    x1 = torch.arange(seqlen).expand(seqlen, -1).to(dev)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(dev)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2)[None, None, :, :].type(torch.FloatTensor).to(dev)
        dist_scores = torch.clamp(
            (disttotal_scores - distcum_scores) * position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach()

    m = nn.Softplus()
    gamma = -1. * m(gamma).unsqueeze(0)  # 1, heads, 1, 1
    total_effect = torch.clamp(torch.clamp(
        (dist_scores * gamma).exp(), min=1e-5), max=1e5)

    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)

    if zero_pad:
        # 首行 score 置 0
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(dev)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)

    output_mean = torch.matmul(scores, v_mean)
    output_cov = torch.matmul(scores ** 2, v_cov)

    return output_mean, output_cov


def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    """式(4) 的矩阵化 Wasserstein 距离：mean 项 (m1-m2)^2 展开成 matmul 形式，
    cov 项用 sqrt 后二阶展开（cov 已由 ELU+1 保证为正）。"""
    mean1_2 = torch.sum(mean1 ** 2, -1, keepdim=True)  # BS, heads, seqlen, 1
    mean2_2 = torch.sum(mean2 ** 2, -1, keepdim=True)
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2)

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)),
                                torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) \
        + cov1_2 + cov2_2.transpose(-1, -2)

    return ret + cov_ret  # BS, heads, seqlen, seqlen
