"""GKT（基于图的深度知识追踪）——对齐 pykt/models/gkt.py（+gkt_utils.py 建图），
只改接口/设备处理/技能映射，并保留 MLP、EraseAddGate、GRUCell 等全部模块结构。

pyKT 在"每步单概念"的数据上训练：forward(q, r) 逐位置把 (概念, 作答) 当作一次
交互：aggregate → 图邻居聚合(erase-add gate) → GRU 更新全部概念状态 → 对下一位置
的概念 gather 出预测概率。本题设定是问题级数据（pro_max 个问题、每个问题映射到
1..max_concepts 个技能），因此本端口把 GKT 的"概念空间"改成"技能空间"
num_c = 技能数（assist09 为 123），并做两处等价/必要的适配：

1. 每步多技能子步更新：位置 t 的问题若含多个技能，则按技能升序拆成多个"子步"，
   每个子步 = pyKT 原版的一步更新（该子步的技能作为 qt，只对本批中确实含有该
   技能的学生的状态做更新，其余学生 qt=-1 跳过 —— 即 pyKT 原版的 mask 语义）；
   同一位置内的子步顺序复合（上一步子步的结果作为下一步子步的 ht 输入），
   与 pyKT 逐位置单步更新完全等价（单技能问题时子步数=1，逐位一致）。
2. 预测 gather 换成 pro2skill 均值：位置 t 处理完后，对下一题 next_problem[t] 的
   全部技能列取 sigmoid 概率的均值作为 P[t]（单技能问题 == pyKT 的 one_hot gather；
   技能数 0 的防御性回退 0.5，数据中不会出现）。

其余差异（数值上等价，见各注释）：
- one_hot_feat.mm(interaction 全表) 的直接嵌入取行（数学恒等，省掉 2*num_c 的 one-hot 表）；
- graph / pro2skill 注册为 buffer（requires_grad=False 的 nn.Parameter 等价，
  但不计入 state_dict 的可训练参数统计）；
- 建图只用训练数据（pyKT 用 train+test 拼接建图），且按本框架原始会话文件
  train_question.txt 的"3 行块"格式解析（跨 200 窗口切分点不额外建边）。
"""
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class GKT(nn.Module):
    """num_c: 技能空间大小（== graph/pro2skill 的列数，assist09 为 123）；
    pro2skill: (pro_max, num_c) 的 0/1 题目-技能隶属矩阵（buffer，不参与梯度）；
    graph: (num_c, num_c) 行归一化技能转移矩阵（buffer，不参与梯度）；
    hidden_dim: 各 MLP/GRU 隐层宽度；emb_size: 交互/技能嵌入宽度（脚本里 = hidden_dim=64）；
    graph_type: 仅作记录（建图发生在 main_baseline，'transition' 或 'dense'）。
    """

    def __init__(self, num_c, pro2skill, graph, hidden_dim=64, emb_size=None,
                 dropout=0.5, bias=True, graph_type="transition"):
        super(GKT, self).__init__()
        self.model_name = "gkt"
        self.num_c = num_c            # 技能空间大小（对齐 pyKT 的 num_c 概念空间）
        self.hidden_dim = hidden_dim
        if emb_size is None:
            emb_size = hidden_dim     # pyKT 脚本 args.emb_size = args.hidden_dim
        self.emb_size = emb_size
        self.res_len = 2
        self.graph_type = graph_type
        self.emb_type = "qid"

        # 图与题目-技能隶属都是统计量：注册为 buffer（pyKT 用 requires_grad=False 的
        # nn.Parameter，行为等价；用 buffer 避免它们混入可训练参数统计）
        self.register_buffer("graph", torch.as_tensor(graph, dtype=torch.float32))
        self.register_buffer("pro2skill", torch.as_tensor(pro2skill, dtype=torch.float32))

        if pro2skill.shape[0] != 0:  # 非空即检查形状，便于尽早发现调用错误
            assert self.pro2skill.size(1) == self.num_c

        # 交互嵌入：行 = 技能*2 + 作答（2*num_c 行）；技能嵌入最后一行是 pad 行
        # （padding_idx=-1 → 由 torch 换算成 num_c 行，即最后一行不产生梯度，与 pyKT 相同）
        self.interaction_emb = nn.Embedding(self.res_len * num_c, emb_size)
        self.emb_c = nn.Embedding(num_c + 1, emb_size, padding_idx=-1)

        # f_self 函数
        mlp_input_dim = hidden_dim + emb_size
        self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)

        # f_neighbor 函数（f_in 与 f_out）
        self.f_neighbor_list = nn.ModuleList()
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))

        # Erase & Add Gate
        self.erase_add_gate = EraseAddGate(hidden_dim, num_c)
        # Gate Recurrent Unit
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        # prediction layer
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)

    # 一个"子步"的聚合，见论文 Section 3.2.1（与 pyKT 相同；xt/qt 为该子步内的交互）
    def _aggregate(self, xt, qt, ht, batch_size):
        r"""qt=-1 的学生不参与（对应 pyKT 的 pad 语义）。
        xt: [batch_size]（= qt*2 + 作答，仅 qt!=-1 的行被查表）
        qt: [batch_size]（该子步的技能 id，-1 表示跳过）
        ht: [batch_size, num_c, hidden_dim]
        tmp_ht: [batch_size, num_c, hidden_dim + emb_size]
        """
        device = qt.device
        qt_mask = torch.ne(qt, -1)  # [batch_size]
        # pyKT: one_hot(xt[qt_mask]).mm(交互嵌入全表) —— 与直接按行取交互嵌入恒等
        res_embedding = self.interaction_emb(xt[qt_mask])  # [mask_num, emb_size]
        mask_num = res_embedding.shape[0]

        # 技能列特征：默认全部取 num_c(pad 行)，qt!=-1 的学生逐列取技能嵌入 0..num_c-1
        concept_idx_mat = self.num_c * torch.ones((batch_size, self.num_c), device=device).long()
        concept_idx_mat[qt_mask, :] = torch.arange(self.num_c, device=device)
        concept_embedding = self.emb_c(concept_idx_mat)  # [batch_size, num_c, emb_size]

        # 把当前交互嵌入写进该学生锚点技能所在的列
        index_tuple = (torch.arange(mask_num, device=device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(index_tuple, res_embedding)
        tmp_ht = torch.cat((ht, concept_embedding), dim=-1)  # [batch_size, num_c, hidden+emb]
        return tmp_ht

    # GNN 邻居聚合，见论文 3.3.2 式(1)（与 pyKT 相同）
    def _agg_neighbors(self, tmp_ht, qt):
        r"""tmp_ht: [batch_size, num_c, hidden_dim + emb_size]
        qt: [batch_size]（该子步各生的锚点技能）
        m_next: [batch_size, num_c, hidden_dim]
        """
        device = qt.device
        qt_mask = torch.ne(qt, -1)
        masked_qt = qt[qt_mask]  # [mask_num]
        masked_tmp_ht = tmp_ht[qt_mask]  # [mask_num, num_c, hidden+emb]
        mask_num = masked_tmp_ht.shape[0]
        self_index_tuple = (torch.arange(mask_num, device=device), masked_qt.long())
        self_ht = masked_tmp_ht[self_index_tuple]  # [mask_num, hidden+emb]：锚点技能处自身特征
        self_features = self.f_self(self_ht)  # [mask_num, hidden_dim]

        expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(1, self.num_c, 1)
        neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht), dim=-1)  # [mask_num, num_c, 2*(hidden+emb)]

        # 图权重：出边（锚点→其它技能）与入边（其它技能→锚点）
        adj = self.graph[masked_qt.long(), :].unsqueeze(dim=-1)          # [mask_num, num_c, 1]
        reverse_adj = self.graph[:, masked_qt.long()].transpose(0, 1).unsqueeze(dim=-1)
        neigh_features = adj * self.f_neighbor_list[0](neigh_ht) + \
            reverse_adj * self.f_neighbor_list[1](neigh_ht)  # [mask_num, num_c, hidden]

        m_next = tmp_ht[:, :, :self.hidden_dim]           # 初始 = 旧 ht
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        return m_next

    # 更新子步：GNN 聚合 → Erase & Add Gate → GRU（与 pyKT 相同）
    def _update(self, tmp_ht, ht, qt):
        r"""h_next: [batch_size, num_c, hidden_dim]（qt=-1 的行保持旧 ht 不变）"""
        device = qt.device
        qt_mask = torch.ne(qt, -1)
        mask_num = qt_mask.nonzero().shape[0]
        m_next = self._agg_neighbors(tmp_ht, qt)  # [batch_size, num_c, hidden_dim]
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])  # [mask_num, num_c, hidden]
        # GRU 把每个技能的旧状态推进到新状态
        h_next = m_next
        res = self.gru(m_next[qt_mask].reshape(-1, self.hidden_dim),
                       ht[qt_mask].reshape(-1, self.hidden_dim))  # [mask_num*num_c, hidden]
        index_tuple = (torch.arange(mask_num, device=device),)
        h_next[qt_mask] = h_next[qt_mask].index_put(index_tuple,
                                                    res.reshape(-1, self.num_c, self.hidden_dim))
        return h_next

    def forward(self, last_problem, last_ans, next_problem, next_ans):
        """P[t] = 对 next_problem[t]（=窗口第 t+1 步题目）答对的预测，
        只使用窗口第 0..t 步的作答 —— 与框架其它模型语义一致。
        返回 (P (B, 199), None)。
        """
        # 前插一列还原原窗口（行 j ↔ 窗口第 j 步的题目/作答）
        q_data = torch.cat([last_problem[:, :1], next_problem], dim=1)   # (B, T)
        r_data = torch.cat([last_ans[:, :1].long(), next_ans.long()], dim=1)
        device = q_data.device

        batch_size, seq_len = q_data.shape
        ht = torch.zeros((batch_size, self.num_c, self.hidden_dim), device=device)

        pred_list = []
        for i in range(seq_len):
            qi = q_data[:, i]                     # 当前步题目 (B,)
            ri = r_data[:, i]                     # 当前步作答 (B,)
            p2s = self.pro2skill[qi]              # (B, num_c) 0/1 题目-技能隶属
            num_sk = p2s.sum(dim=1)               # 每题技能数（单技能问题 = 1）
            max_sk = int(num_sk.max().item()) if batch_size > 0 else 0

            # 每步按技能升序拆子步：第 k 个子步的锚点 = 各生第 k+1 个技能（不足者 -1 跳过）
            for k in range(max_sk):
                cum = p2s.cumsum(dim=1)                       # 列 j = 技能 <= j 的个数
                is_kth = (p2s == 1) & (cum == k + 1)          # 第 k+1 个技能所在的列
                has_k = is_kth.any(dim=1)
                col_k = is_kth.to(torch.long).argmax(dim=1)   # 无第 k+1 个技能时该值被覆盖
                qt = torch.where(has_k, col_k,
                                 torch.full_like(col_k, -1)).long()
                xt = qt * 2 + ri                                # 交互特征 = 技能*2 + 作答
                tmp_ht = self._aggregate(xt, qt, ht, batch_size)
                h_next = self._update(tmp_ht, ht, qt)
                ht = h_next                                   # 未更新的行 = 旧 ht（等价 pyKT 的 mask 赋值）

            # 整步处理完后的全技能预测概率（每步每个学生都至少更新过 1 个子步）
            yt = torch.sigmoid(self.predict(ht).squeeze(-1))    # (B, num_c)
            if i < seq_len - 1:
                # 对下一题的全部技能列取均值（单技能 == pyKT 的 one_hot gather）
                p2s_next = self.pro2skill[q_data[:, i + 1]]     # (B, num_c)
                cnt = p2s_next.sum(dim=1)
                pred = (p2s_next * yt).sum(dim=1) / cnt.clamp(min=1)
                pred = torch.where(cnt > 0, pred, torch.full_like(pred, 0.5))
                pred_list.append(pred)
        pred_res = torch.stack(pred_list, dim=1)  # (B, seq_len - 1) = (B, 199)
        return pred_res, None


# Multi-Layer Perceptron(MLP) layer（pyKT 原样）
class MLP(nn.Module):
    """Two-layer fully-connected ReLU net with batch norm."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            # batch_size == 1 or 0 will cause BatchNorm error, so return the input directly
            return inputs
        if len(inputs.size()) == 3:
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.norm(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        else:  # len(input_size()) == 2
            return self.norm(inputs)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)


class EraseAddGate(nn.Module):
    """Erase & Add Gate（DKVMN 风格的擦除加和门；pyKT 原样：只用一个输入矩阵自建两个门）"""

    def __init__(self, feature_dim, num_c, bias=True):
        super(EraseAddGate, self).__init__()
        self.weight = nn.Parameter(torch.rand(num_c))   # 每个技能一个可学习的更新强度
        self.reset_parameters()
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        erase_gate = torch.sigmoid(self.erase(x))
        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x
        add_feat = torch.tanh(self.add(x))
        res = tmp_x + self.weight.unsqueeze(dim=1) * add_feat
        return res


# ------------------------------ 数据端建图工具 ------------------------------

def build_pro2skill(ques_skill_path):
    """读 ques_skill.csv（列 0 题目 id、列 1 技能 id）→ (pro_max, skill_max) 0/1 矩阵。
    注意：技能编号以该 csv 为准（train_skill.txt 的编号范围与其不一致，不能混用）。
    返回 (pro2skill torch.FloatTensor, pro_max, skill_max)。
    """
    df = pd.read_csv(ques_skill_path)
    pro = df.values[:, 0].astype(np.int64)
    sk = df.values[:, 1].astype(np.int64)
    pro_max = int(pro.max()) + 1
    skill_max = int(sk.max()) + 1
    pro2skill = torch.zeros((pro_max, skill_max))
    pro2skill[torch.from_numpy(pro).long(), torch.from_numpy(sk).long()] = 1.
    return pro2skill, pro_max, skill_max


def _read_question_txt(path):
    """按框架 loader 的 3 行块格式解析问题序列文件（每 3 行一个用户：
    uid / 题目列表 / 作答列表；题目行与作答行按逗号分隔），返回 problem_list。
    （与 load_data.py getReader 同构；末尾不完整的块自动忽略。）"""
    problem_list = []
    with open(path, 'r') as f:
        lines = [ln.strip() for ln in f]
    for idx in range(0, len(lines) - 2, 3):   # 块首行 = 用户 id 行，块内第 2/3 行 = 题目/作答
        try:
            probs = list(map(int, lines[idx + 1].split(','))) if lines[idx + 1] else []
        except Exception:
            continue
        problem_list.append(probs)
    return problem_list


def build_transition_graph(train_txt_path, pro2skill):
    """从原始会话序列（train_question.txt，每行一个完整用户会话）统计技能转移并
    行归一化，得到 (skill_max, skill_max) 图。连续题目对 (p_i, p_{i+1}) 之间统计
    全部 (s ∈ S(p_i), s' ∈ S(p_{i+1})) 对；对角清零后行归一化（行和为 0 的孤立行
    保持 0 —— pyKT 的 inverse-vectorize 写法）。会话切分成 200 窗口的位置不额外建边。
    """
    skill_max = pro2skill.shape[1]
    p2s_np = pro2skill.numpy()
    graph = np.zeros((skill_max, skill_max))
    problem_list = _read_question_txt(train_txt_path)
    for probs in problem_list:
        for a, b in zip(probs[:-1], probs[1:]):
            if a < 0 or b < 0 or a >= p2s_np.shape[0] or b >= p2s_np.shape[0]:
                continue
            sa = np.nonzero(p2s_np[a])[0]     # 题目 a 的技能
            sb = np.nonzero(p2s_np[b])[0]     # 题目 b 的技能
            for s in sa:
                for s2 in sb:
                    graph[s, s2] += 1
    np.fill_diagonal(graph, 0)
    # 行归一化（pyKT 同款：行和为 0 时保持 0，避免除零）
    rowsum = graph.sum(1)
    def inv(x):
        return 0. if x == 0 else 1. / x
    r_inv = np.vectorize(inv)(rowsum)
    graph = np.diag(r_inv).dot(graph)
    return torch.from_numpy(graph).float()


def build_dense_graph(skill_max):
    """稠密均匀图：非对角 1/(num_c-1)（pyKT 原样，graph_type='dense' 时用）。"""
    graph = 1. / (skill_max - 1) * np.ones((skill_max, skill_max))
    np.fill_diagonal(graph, 0)
    return torch.from_numpy(graph).float()
