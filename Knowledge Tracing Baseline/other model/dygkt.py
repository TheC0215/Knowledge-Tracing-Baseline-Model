"""DyGKT（Dynamic Graph based Knowledge Tracing）——按本框架接口移植的版本
（原始仓库 DyGKT-main/models/DyGKT.py + models/modules.py）。

原始仓库把 KT 当"时序图上的链路预测"任务：节点 = 学生 + 题目，边 = 一次作答，
用 NeighborSampler 在全局时序图上按时间采样邻居，src/dst 两个 GRU 更新器分别编码
学生侧与题目侧邻居序列，最后 MergeLayer 输出"这条边成立（答对）"的概率。

本框架的输入是 (last_problem, last_ans, next_problem, next_ans)，没有真实时间戳、
没有学生 ID、也不是边批（edge batch）的形式，因此下面三处必须做等价化改造，其余
模块（节点/边/时间投影、双衰减时间编码、GRU 更新器、Merge 预测头）与论文一致：

1) 时序图的采样范围 → "会话内动态图"
   把每个学生的会话看作一张即时图：位置 t 预测 next_problem[t] 时，候选邻居是其
   历史交互 (≤ t) 中与该目标题目"同技能"的位置（+ 未来位置一律屏蔽）。这与
   DyGKT 的邻居采样语义一致（只用过去、按技能/共现相关性挑邻居），只是邻居集合
   被限定在会话内，避免跨用户全局建图时的时间不可比问题（本框架数据无 ts）。
2) 真实时间戳 → 位置间隔（伪时间）
   DyGKT 的 TimeDualDecayEncoder 吃"相邻时间戳差"，这里吃"距上次同技能交互的位置
   间隔"，阈值 3600*24 秒换成 time_threshold 个位置（默认 20），保留短期/长期双
   通道结构。
3) link predictor → 逐位置二分类
   原仓库输出 (src_node_emb, dst_node_emb) 再经 MergeLayer 得 1 个 logit；这里对
   每个位置 t 输出 P(答对 next_problem[t])，与框架其它模型的 (batch, seq) 对齐。

结论：dense 训练协议、早停、五折划分与既有基线完全一致，可直接横向比较。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import pad_lens


class DyGKT(nn.Module):
    """pro_max: 题目数（= 图节点数）；skill_max: 技能数；
    pro2skill: (pro_max, skill_max) 0/1 隶属矩阵（buffer，用于取题目所属技能）；
    d_node: 节点/边嵌入维度（原仓库 node_dim=edge_dim=64）；
    time_dim: 时间编码维度（原仓库 time_dim=16）；
    num_neighbors: 会话内邻居上限（原仓库 num_neighbors=50，这里受限于最长的历史，
                   实际由 mask 决定；该参数保留用于对齐原配置）；
    ablation: 与论文一致的消融开关（'-1' 默认；'time' 屏蔽时间特征；'skill' 屏蔽
              节点特征；'dual' 改用单通道 TimeEncoder）。
    """

    def __init__(self, pro_max, skill_max, pro2skill=None, d_node=64, time_dim=16,
                 num_neighbors=50, dropout=0.5, time_threshold=20, ablation='-1'):
        super(DyGKT, self).__init__()
        self.pro_max = pro_max
        self.skill_max = skill_max
        self.node_dim = d_node
        self.edge_dim = d_node
        self.time_dim = time_dim
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.time_threshold = time_threshold
        self.ablation = ablation

        if pro2skill is None:
            pro2skill = torch.zeros(pro_max, skill_max)
        # 题目 → 技能（q_matrix 的 argmax，与 DyGKT 用 node_raw_features[:,0] 取技能一致）
        q2skill = torch.as_tensor(np.asarray(pro2skill)).argmax(dim=-1).long()
        self.register_buffer("q2skill", q2skill)

        # ---- 节点/边/时间/结构 投影（对应原仓库 projection_layer）----
        # 原仓库：Linear(node_raw_feat_dim → node_dim)；这里节点原始特征是"题目 + 技能"
        self.node_proj = nn.Linear(d_node, d_node, bias=True)
        self.node_emb = nn.Embedding(pro_max, d_node)
        self.skill_emb = nn.Embedding(skill_max, d_node)
        self.edge_proj = nn.Linear(1, d_node, bias=True)          # edge_raw_features 维度 1
        self.time_proj = nn.Linear(time_dim, d_node, bias=True)
        self.struct_proj = nn.Linear(1, 1, bias=True)             # 共现/同技能结构特征 → 标量先验

        self.output_layer = nn.Linear(d_node, d_node, bias=True)

        # 双衰减时间编码（原仓库 TimeDualDecayEncoder；dual 消融用单通道 TimeEncoder）
        if ablation == 'dual':
            self.time_encoder = TimeEncoder(time_dim)
        else:
            self.time_encoder = TimeDualDecayEncoder(time_dim, threshold=time_threshold)

        # src = 学生侧邻居序列更新器；dst = 题目侧更新器（原仓库两个 DyKT_Seq）
        self.src_node_updater = DyKT_Seq(edge_dim=self.edge_dim, node_dim=self.node_dim)
        self.dst_node_updater = DyKT_Seq(edge_dim=self.edge_dim, node_dim=self.node_dim)

        # link predictor（原仓库 MergeLayer(node_dim*2 → node_dim → 1)）
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * d_node, d_node),
            nn.ReLU(),
            nn.Linear(d_node, 1),
        )
        self.drop = nn.Dropout(dropout)

    # ---------- 内部工具 ----------
    def _gap_features(self, skill, valid):
        """返回 (距上次同技能交互的位置间隔, 目标题目距其上次出现的位置间隔)。
        skill: (B, T) long；valid: (B, T) bool。无历史时用 T（最大间隔）。
        """
        B, T = skill.shape
        S = self.skill_max
        device = skill.device
        pos = torch.arange(T, device=device, dtype=torch.float32).view(1, T, 1)

        onehot = F.one_hot(skill.clamp(min=0), num_classes=S).float()      # (B, T, S)
        val = onehot * (pos + 1.0)                                        # 位置写成 1-based
        cm = torch.cummax(val, dim=1).values                              # 含自身的累积最大
        prev = torch.cat([torch.zeros_like(cm[:, :1]), cm[:, :-1]], dim=1)  # 严格早于 t
        prev_idx = (prev - 1.0).clamp(min=0.0)                            # (B, T, S) 0-based 位置

        gap_now = pos - torch.gather(prev_idx, 2, skill.unsqueeze(-1))     # (B, T, 1)
        gap_now = torch.where(prev.gather(2, skill.unsqueeze(-1)) <= 0,
                              torch.full_like(gap_now, float(T)), gap_now)

        return gap_now.squeeze(-1), prev_idx

    def _encode_time(self, gap, device):
        """gap: (B, T) float → (B, T, node_dim)"""
        return self.time_proj(self.time_encoder(gap))

    # ---------- 前向 ----------
    def forward(self, last_problem, last_ans, next_problem, next_ans):
        device = last_problem.device
        B, T = last_problem.shape
        lens = pad_lens(last_problem)                                     # (B,) 左侧 padding 推出真实长度
        valid = (torch.arange(T, device=device).view(1, T)
                 >= (T - lens).view(B, 1))                                # (B, T) bool

        skill = self.q2skill[last_problem]                                # (B, T)
        skill_next = self.q2skill[next_problem]                           # (B, T)

        # ---- 节点特征：题目嵌入 + 技能嵌入（原仓库 node_raw_features 的 feature_Linear 分支）----
        node_feat = self.node_emb(last_problem) + self.skill_emb(skill)
        node_feat = self.node_proj(self.drop(node_feat))                  # (B, T, d)

        # ---- 边特征：本次作答（0/1）→ Linear(1, d) ----
        edge_feat = self.edge_proj(last_ans.float().unsqueeze(-1))        # (B, T, d)

        # ---- 时间特征：距上次同技能交互的位置间隔 + 双衰减编码 ----
        gap_now, prev_idx = self._gap_features(skill, valid)              # (B, T)
        time_feat = self._encode_time(gap_now, device)                    # (B, T, d)

        # 目标题目距上次出现的位置间隔（第 t 步用 t 时刻的历史视角）
        pos = torch.arange(T, device=device, dtype=torch.float32).view(1, T)
        prev_for_next = torch.gather(prev_idx, 2, skill_next.unsqueeze(-1)).squeeze(-1)
        gap_next = (pos + 1.0) - prev_for_next
        gap_next = torch.where(prev_for_next <= 0, torch.full_like(gap_next, float(T)), gap_next)
        time_feat_next = self._encode_time(gap_next.clamp(min=0), device)  # (B, T, d)

        if self.ablation == 'time':
            time_feat = time_feat * 0
            time_feat_next = time_feat_next * 0
        if self.ablation == 'skill':
            node_feat = node_feat * 0

        # ---- 会话内动态图的邻居聚合（TGAT 式的邻居注意力，等价于按相关性采样邻居）----
        # 邻居 = 历史中与目标题目同技能的位置；严格因果（只看 t 及以前）
        neigh = (skill.unsqueeze(2) == skill_next.unsqueeze(1))            # (B, T, T) [t, s]
        causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
        neigh = neigh & causal.unsqueeze(0) & valid.unsqueeze(1)
        # 至少保留自身位置，避免整行被 mask 后 softmax 退化为均匀分布
        neigh = neigh | torch.eye(T, device=device, dtype=torch.bool).unsqueeze(0)
        if self.ablation == 'counter':                                    # 消融：去掉结构特征
            neigh = torch.zeros_like(neigh)

        key = node_feat + edge_feat + time_feat                            # (B, T, d)
        query = node_feat + self.skill_emb(skill_next)                     # 以"下一题"为查询
        score = torch.einsum('btd,bsd->bts', query, key) / (self.node_dim ** 0.5)
        # 结构特征：同技能指示（1.0）经 struct_proj 投成标量先验，加到邻居得分上
        struct_bias = self.struct_proj(torch.ones(1, 1, device=device))    # (1, 1)
        score = score + struct_bias * neigh.float()
        score = score.masked_fill(~neigh, -1e9)
        attn = torch.softmax(score, dim=-1)
        agg = torch.einsum('bts,bsd->btd', self.drop(attn), key)           # (B, T, d) 邻居聚合

        # ---- src（学生侧）更新器：把邻居聚合结果当作邻居序列喂 GRU ----
        msg = agg + edge_feat + time_feat
        src_emb = self.src_node_updater.update(msg) + msg                  # (B, T, d)
        src_emb = self.output_layer(src_emb)                              # (B, T, d)

        # ---- dst（题目侧）更新器：目标题目节点的历史边序列（这里用其时间/结构特征）----
        dst_msg = edge_feat + time_feat_next
        dst_emb = self.dst_node_updater.update(dst_msg) + dst_msg + node_feat
        dst_emb = self.output_layer(dst_emb)                              # (B, T, d)

        # ---- link predictor：Merge(src, dst) → 1 logit → sigmoid ----
        logit = self.link_predictor(torch.cat([src_emb, dst_emb], dim=-1)).squeeze(-1)
        P = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)                     # (B, T)
        return P, None


class DyKT_Seq(nn.Module):
    """GRU 序列更新器（原仓库 DyKT_Seq：GRU(input=edge_dim, hidden=node_dim)）。

    唯一改动：原实现在"某个节点的邻居序列"上跑 GRU 并只取最后隐状态，输出
    (batch, node_dim) 给当前时刻用；本框架要把整个会话逐位置预测，所以保留 GRU 的
    全部隐状态 (batch, seq, node_dim)，语义上等价于"对每个位置都取一次最后隐状态"。
    """

    def __init__(self, edge_dim: int, node_dim: int):
        super(DyKT_Seq, self).__init__()
        self.patch_enc_layer = nn.Linear(edge_dim, node_dim)
        self.hid_node_updater = nn.GRU(input_size=edge_dim, hidden_size=node_dim, batch_first=True)

    def update(self, x):
        outputs, hidden = self.hid_node_updater(x)
        return outputs                                                 # (batch, seq, node_dim)


class TimeEncoder(nn.Module):
    """单通道余弦时间编码（原仓库 modules.TimeEncoder，消融 ablation='dual' 时用）。"""

    def __init__(self, time_dim: int):
        super(TimeEncoder, self).__init__()
        self.time_dim = time_dim
        self.w = nn.Linear(1, time_dim)
        self.w.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)).reshape(time_dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

    def forward(self, gaps):
        return torch.cos(self.w(gaps.unsqueeze(-1)))


class TimeDualDecayEncoder(nn.Module):
    """短期/长期双通道时间编码（原仓库 modules.TimeDualDecayEncoder）。
    唯一改动：输入从"相邻时间戳差"改为"位置间隔"（本框架无真实时间戳），
    阈值从 3600*24 秒改为 time_threshold 个位置。"""

    def __init__(self, time_dim: int, threshold: float = 20.0):
        super(TimeDualDecayEncoder, self).__init__()
        self.time_dim = time_dim
        self.threshold = threshold

        self.w_short = nn.Linear(1, time_dim)
        self.w_short.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)).reshape(time_dim, -1))
        self.w_short.bias = nn.Parameter(torch.zeros(time_dim))

        self.w_long = nn.Linear(1, time_dim)
        self.w_long.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32)).reshape(time_dim, -1))
        self.w_long.bias = nn.Parameter(torch.zeros(time_dim))

        self.f = nn.ReLU()
        self.w_o = nn.Linear(time_dim, time_dim)
        self.w_o.weight = nn.Parameter(
            torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim * time_dim, dtype=np.float32))
            .reshape(time_dim, -1))
        self.w_o.bias = nn.Parameter(torch.zeros(time_dim))

    def forward(self, gaps):
        g = gaps.unsqueeze(-1)                                            # (B, T, 1)
        short_mask = (g <= self.threshold).float()
        short = self.f(self.w_short(g * short_mask))
        long = self.f(self.w_long(g * (1 - short_mask)))
        return self.w_o(short + long)                                     # (B, T, time_dim)
