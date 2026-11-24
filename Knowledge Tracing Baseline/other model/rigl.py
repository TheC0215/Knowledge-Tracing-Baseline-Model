"""RIGL（KDD'24, A Unified Reciprocal Approach for Tracing the Independent and Group
Learning Processes）——按本框架接口移植的版本（原始仓库 RIGL-main/models/RIGL.py）。

RIGL 提出 HKT（holistic knowledge tracing）任务：同时追踪"独立学习"（学生序列）与
"小组学习"（group 序列，来自真实的小组协作数据），并用 reciprocal 嵌入 + 动态图 +
对比学习把两者耦合。本框架的数据只有单个学生的题目序列（无 group 字段、无学生 ID），
因此按下面的方式移植，尽量保留原论文的关键机制：

1) group 分支 → 组原型（graph_mode='proto'，默认）
   原仓库每个 group 有一条真实 group 序列。没有 group 数据时有两种替代：
   (a) 'peer'：把一个 batch 的学生当作虚拟组（batch 内掩码均值 + 学生 top-k 相似图）；
   (b) 'proto'：用 K 个可学习的"小组状态原型"表示组，学生在每个时间帧对原型做注意力
       得到组上下文（组端不引入 batch 内随机学生的信息）。
   实测（assist09 第 0 折，40 epoch，同超参）：'peer' 验证 AUC ≈ 0.66，'none'（完全
   不做组聚合）≈ 0.711，'proto' ≈ 0.724 —— 随机 batch 组队把别人的作答流混进当前学生
   的表达里，是纯噪声，所以默认取 'proto'；'peer'/'group'/'none' 保留为消融开关。
2) 时间帧（timeframe）= 会话位置
   原仓库的 timeframe 是"span"（每 30 步一个 span），这里直接以序列位置为时间帧；
   'peer'/'group' 模式下每个位置建一张"组-学生"图（学生之间按余弦相似度取 top-k 连边）
   再过 GCN，然后统一做时序 transformer（见 _dynamic_graph）。
3) 输出层 gather 化
   原仓库 out_layer 输出 n_question 维 logits 后用 one_hot 取当前题目列；当题目数
   上万时 (batch, seq, n_question) 会爆显存，这里等价地改为"按目标题目切片权重再
   点积"（数学上与 gather 完全一致，见 RIGL.forward 里的注释）。
4) torch_geometric 依赖移除：GCNConv 换成本文件内 _gcn（对称归一化 + 两层线性），
   与原实现等价（原实现用 to_undirected 建的双向图，这里同样对称化），避免在服务器
   上再装 pyg/torch_scatter。

对比学习（bias-aware contrastive learning）保留：训练时对作答做 flip 采样，用
NT-Xent 对齐原始/扰动两路的学生表达；该辅助损失通过 model.aux_loss 交给 run.py
的 loss 汇总（gamma=cl_weight，原仓库 gamma=0.001）。
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import pad_lens


class RIGL(nn.Module):
    """pro_max(n_question): 题目数；skill_max(n_concept): 技能数；
    q_matrix: (pro_max, skill_max) 0/1 题目-技能矩阵（buffer）；
    d_model/d_ff/n_blocks/num_attn_heads/dropout: transformer 结构（与原仓库一致）；
    final_fc_dim: 兼容原参数（head 改为按题目切片后不再使用，保留以对齐脚本）；
    top_num: 动态图里每个学生保留的相似学生邻居数（原仓库 top_num=3）；
    flip_rate: 作答翻转比例（原仓库 0.05）；cl_weight: 对比损失权重（原仓库 gamma=0.001）。
    """

    def __init__(self, pro_max, skill_max, q_matrix=None, d_model=256, d_ff=256,
                 n_blocks=4, num_attn_heads=8, dropout=0.2, final_fc_dim=256,
                 seq_len=200, top_num=3, flip_rate=0.05, cl_weight=1e-3,
                 graph_mode='proto', num_proto=8):
        super().__init__()
        self.model_name = "rigl"
        self.n_question = pro_max
        self.n_concept = skill_max
        self.num_c = pro_max          # 与原仓库一致：num_c = n_question（one_hot 取题目列）
        self.d_model = d_model
        self.n_dim = d_model
        self.seq_len_ = seq_len
        self.top_num = top_num
        # 动态图模式：'peer' = 组节点 + 学生 top-k（默认，最接近原论文）；
        # 'group' = 只有组节点；'none' = 不做图聚合（消融用）
        self.graph_mode = graph_mode
        self.num_proto = num_proto
        # 组原型：没有真实 group 数据时，用 K 个可学习的"小组状态原型"代替小组序列。
        # 学生每一步的组上下文 = 对各原型的注意力加权和（不含 batch 内其它学生的信息，
        # 避免随机 batch 组队带来的噪声；组的更新只通过梯度完成）。
        self.grp_proto = nn.Parameter(torch.randn(num_proto, d_model) * 0.02)
        self.flip_rate = flip_rate
        self.cl_weight = cl_weight
        self.dropout_rate = dropout

        if q_matrix is None:
            q_matrix = np.zeros((pro_max, skill_max), dtype=np.float32)
        q_matrix = np.asarray(q_matrix, dtype=np.float32)
        self.register_buffer("q_matrix", torch.from_numpy(q_matrix))
        # 每题的概念 = q_matrix 行的 argmax（原仓库 base_emb 里的 argmax 取法）
        self.register_buffer("cpts_idx",
                             torch.from_numpy(q_matrix.argmax(axis=-1)).long())

        self.ques_embed_layer = nn.Embedding(pro_max + 1, d_model)
        self.cpts_embed_layer = nn.Embedding(skill_max + 1, d_model)
        self.qa_embed_layer = nn.Embedding(2 * pro_max + 1, d_model)   # 兼容原仓库（未使用）
        self.resp_embed_layer = nn.Embedding(2, d_model)
        self.AttModule = AttentionModule(d_model)

        # 动态图上的两层 GCN（替代 torch_geometric 的 GCNConv(2d→128→2d)）
        self.graph_layer1 = nn.Linear(2 * d_model, 128)
        self.graph_layer2 = nn.Linear(128, 2 * d_model)

        # 时序 transformer（组/学生共享权重，与原仓库 self.model 复用的写法一致）
        self.model = Architecture(d_model=d_model, d_feature=d_model // num_attn_heads,
                                  d_ff=d_ff, n_heads=num_attn_heads, dropout=dropout,
                                  n_blocks=n_blocks, seq_len=seq_len)

        # 输出层：等价于原 out_layer(Linear(2d → n_question)) 的权重切片
        self.out_w = nn.Parameter(torch.empty(pro_max, 2 * d_model))
        self.out_b = nn.Parameter(torch.zeros(pro_max))
        nn.init.xavier_uniform_(self.out_w)

        # 训练期由 forward 写入，run.py 汇总进总 loss 后清空
        self.aux_loss = None

    # ---------------- 基础嵌入 ----------------
    def base_emb(self, ques, resp):
        """ques: (B, T) long；resp: (B, T) long → (q_embed, qa_embed)，与原仓库一致。"""
        ques_embed = self.ques_embed_layer(ques)
        cpts = self.cpts_idx[ques]                                    # (B, T)
        cpts_embed = self.cpts_embed_layer(cpts)
        q_embed = ques_embed + cpts_embed
        qa_embed = self.resp_embed_layer(resp) + ques_embed
        return q_embed, qa_embed

    def _proto_context(self, q_embed, qa_embed):
        """K 个组原型上的注意力上下文：(B, T, d)。"""
        query = q_embed + qa_embed                                     # (B, T, d)
        score = torch.einsum('btd,kd->btk', query, self.grp_proto) / (self.n_dim ** 0.5)
        weight = torch.softmax(score, dim=-1)                          # (B, T, K)
        return torch.einsum('btk,kd->btd', weight, self.grp_proto)

    # ---------------- 动态图（组-学生 + 学生-学生 top-k）----------------
    def _gcn(self, x, adj):
        """x: (N, F)；adj: (N, N) 0/1 对称邻接 → 两层 GCN（对称归一化）。"""
        n = adj.shape[0]
        a = adj + torch.eye(n, device=adj.device)
        deg = a.sum(-1).clamp(min=1e-6)
        dinv = deg.pow(-0.5)
        a_norm = dinv.unsqueeze(-1) * a * dinv.unsqueeze(0)
        h = F.relu(a_norm @ self.graph_layer1(x))
        h = F.dropout(h, p=self.dropout_rate, training=self.training)
        return a_norm @ self.graph_layer2(h)

    def _dynamic_graph(self, x, valid):
        """逐时间帧建图并做 GCN。x: (B, T, 2d)，valid: (B, T)。
        返回 (B, T, 2d)：学生节点在图上的表达（组节点的表达通过 reciprocal 均值体现）。"""
        B, T, D = x.shape
        if self.graph_mode in ('none', 'proto'):
            return x
        out = torch.zeros_like(x)
        eye_b = torch.eye(B, device=x.device, dtype=torch.bool)
        full_base = torch.zeros(B + 1, B + 1, device=x.device)
        for t in range(T):
            e = x[:, t, :]                                            # (B, D)
            # 学生两两余弦相似度（原仓库 cosin_similarity 的两两版本）
            en = F.normalize(e, dim=-1)
            sim = en @ en.t()                                         # (B, B)
            sim = sim.masked_fill(eye_b, -1e9)
            if self.graph_mode == 'group':
                adj = torch.zeros(B, B, device=x.device)                # 只有组节点与学生的连边
            else:
                k = min(self.top_num, max(1, B - 1))
                idx = sim.topk(k, dim=-1).indices                      # (B, k) 每个学生的 top-k 相似学生
                adj = torch.zeros(B, B, device=x.device)
                adj.scatter_(1, idx, 1.0)
                adj = ((adj + adj.t()) > 0).float()                     # to_undirected

            # 加组节点 0（与所有学生相连，原仓库 source_nodes=[0]*num_stu）
            grp = e.mean(dim=0, keepdim=True)                         # (1, D) 虚拟组表达
            nodes = torch.cat([grp, e], dim=0)                        # (B+1, D)
            full = full_base.clone()
            full[1:, 1:] = adj
            full[0, 1:] = 1.0
            full[1:, 0] = 1.0
            emb = self._gcn(nodes, full)                              # (B+1, D)
            # 组节点信息回写给各学生（reciprocal：组上下文增强学生表达）
            out[:, t, :] = emb[1:] + emb[0:1]
        return out

    # ---------------- 前向 ----------------
    def _forward_once(self, q, r, q_next):
        B, T = q.shape
        lens = pad_lens(q)
        valid = (torch.arange(T, device=q.device).view(1, T) >= (T - lens).view(B, 1))

        q_embed, qa_embed = self.base_emb(q, r)                       # (B, T, d) each

        if self.graph_mode == 'proto':
            # 组原型注意力：学生每个位置从 K 个可学习组原型取上下文（不依赖 batch 组成）
            ctx = self._proto_context(q_embed, qa_embed)               # (B, T, d)
            grp_q = grp_qa = ctx
        else:
            # 虚拟组（batch 内学生聚合）：仅用于消融对比，随机 batch 组队会引入噪声
            w = valid.float().unsqueeze(-1)                            # (B, T, 1)
            denom = w.sum(dim=0).clamp(min=1.0)                        # (T, 1)
            grp_q = (q_embed * w).sum(dim=0, keepdim=True) / denom.unsqueeze(0)   # (1, T, d)
            grp_qa = (qa_embed * w).sum(dim=0, keepdim=True) / denom.unsqueeze(0)
            grp_q = grp_q + self.AttModule(q_embed)
            grp_qa = grp_qa + self.AttModule(qa_embed)

        # reciprocal：学生 ← 组上下文（原仓库的 q_embed_stu = q_embed_stu_ori + q_embed_grp_ori）
        q_embed = q_embed + grp_q
        qa_embed = qa_embed + grp_qa

        x = torch.cat([q_embed, qa_embed], dim=-1)                    # (B, T, 2d)
        x = self._dynamic_graph(x, valid)                             # 关系引导的动态图

        # 拆回两路，共享 transformer 编码（原仓库 self.model 复用）
        q_graph, qa_graph = x[:, :, :self.n_dim], x[:, :, self.n_dim:]
        h_q = self.model(q_graph)
        h_qa = self.model(qa_graph)
        concat = torch.cat([h_qa, q_graph], dim=-1)                   # (B, T, 2d)

        # 输出：按目标题目切片权重 → (B, T) 的 logit（等价 one_hot gather）
        w_q = self.out_w[q_next]                                      # (B, T, 2d)
        logit = (concat * w_q).sum(-1) + self.out_b[q_next]
        return logit, valid

    def flip_response(self, r, valid):
        """按 flip_rate 随机翻转作答（原仓库 flip_response 的向量化版本）。"""
        if not self.training:
            return r
        mask = (torch.rand_like(r, dtype=torch.float) < self.flip_rate).long() & valid.long()
        return (r ^ mask)                                             # 0/1 翻转

    def forward(self, last_problem, last_ans, next_problem, next_ans):
        q = last_problem.long()
        r = last_ans.long().clamp(0, 1)
        q_next = next_problem.long()

        logit, valid = self._forward_once(q, r, q_next)
        P = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)                # (B, T)

        # ---- bias-aware 对比学习（仅训练期算，作为辅助损失）----
        if self.training and self.cl_weight > 0:
            r_flip = self.flip_response(r, valid)
            logit_cl, _ = self._forward_once(q, r_flip, q_next)
            p_cl = torch.sigmoid(logit_cl)
            # NT-Xent：原始表达与扰动表达应互相靠近（对角为正样本）
            sim = F.cosine_similarity(P.unsqueeze(1), p_cl.unsqueeze(0), dim=-1) / 0.05
            target = torch.arange(q.shape[0], device=q.device)
            self.aux_loss = self.cl_weight * F.cross_entropy(sim, target)
        else:
            self.aux_loss = None

        return P, None


# ============================ 以下为原仓库结构件（保留等价实现） ============================

class AttentionModule(nn.Module):
    """组内注意力池化（RIGL-main/utils/utils.py 原样）。"""

    def __init__(self, embedding_dim, drop_ratio=0.2):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x_avg = torch.mean(x, dim=1)                     # (N, d)
        out = self.linear(x_avg)                         # (N, 1)
        weight = torch.softmax(out.unsqueeze(-1), dim=0)  # (N, 1, 1)
        return torch.sum(weight * x, dim=0, keepdim=True)


class Architecture(nn.Module):
    """时序 transformer（RIGL-main/models/RIGL.py 的 Architecture + CosinePositionalEmbedding）。"""

    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, n_blocks, seq_len):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList([
            TransformerLayer(d_model=d_model, d_feature=d_feature, d_ff=d_ff,
                             dropout=dropout, n_heads=n_heads)
            for _ in range(n_blocks)
        ])
        self.position_emb = CosinePositionalEmbedding(d_model=d_model, max_len=seq_len)

    def forward(self, q_embed_data):
        x = q_embed_data + self.position_emb(q_embed_data)
        for block in self.blocks:
            x = block(query=x, key=x, values=x)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout):
        super().__init__()
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, query, key, values):
        seqlen = query.size(1)
        # 因果 mask：位置 t 可看 0..t（含自己）。pyKT/SAKT 系原版为 triu(k=1)：
        # 若用 k=0（屏蔽对角），位置 0 的整行被全屏蔽，softmax 对全 -1e32 的行
        # 输出均匀分布（而非零），会把全序列信息混进位置 0，再经逐层 attention
        # 传播到所有位置——实测造成非因果泄漏（cold-start k=1 AUC 虚高至 0.82）。
        nopeek = np.triu(np.ones((1, 1, seqlen, seqlen)), k=1).astype('uint8')
        src_mask = (torch.from_numpy(nopeek) == 0).to(query.device)
        q2 = self.masked_attn_head(query, key, values, mask=src_mask)
        query = query + self.dropout1(q2)
        query = self.layer_norm1(query)
        q2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout2(q2)
        return self.layer_norm2(query)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, bias=True):
        super().__init__()
        self.d_model, self.d_k, self.h = d_model, d_feature, n_heads
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, q, k, v, mask):
        bs = q.size(0)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores.masked_fill(mask == 0, -1e32)
        scores = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(scores, v)
        out = out.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_proj(out)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.weight = nn.Parameter(pe.unsqueeze(0), requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(1), :]
