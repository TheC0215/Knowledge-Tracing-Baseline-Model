"""MIKT（WWW'24, Interpretable Knowledge Tracing with Multiscale State Representation）
——按本框架接口移植的版本（原始仓库 MIKT-master/model.py）。

接口约定与框架内其它模型一致：
    P, _ = model(last_problem, last_ans, next_problem, next_ans)
    P: (batch, seq_len) 每个位置 t 预测"下一题 next_problem[t] 答对"的概率
    last_problem/last_ans 为 (batch, seq_len)，next_problem/next_ans 为其向右错一位

与原始仓库的差异（只有 3 处，均为"去掉全局变量 + 贴合框架张量类型"）：
1. 原实现用 glo.py 的全局字典传 state_d / max_seq / pro2skill，这里改成构造参数 +
   注册 buffer（pro2skill 随 .to(device) 自动搬设备，不再有"忘记 .to(device)"的坑）；
2. 框架的 next_ans 是 float 张量，而 ans_embed 需要 long 索引 → 统一 .long()；
3. 原仓库返回 (P, contrast_loss=0)，这里返回 (P, None)（框架忽略第二项；CL 项在原
   实现里恒为 0，未参与优化，故此处不引入）。

模型要点（与论文一致）：
- 多尺度状态：技能级状态 skill_state(技能数 × state_d) 与 全局状态 all_state(1 × state_d)，
  技能状态按"技能遗忘门 + 时间间隔编码(last_skill_time)"衰减，全局状态经 LSTM 遗忘；
- 题目表示：题目-技能隶属注意力(pro2skill 掩码) + 难度偏移(pro_diff) 得到题目嵌入；
- 预测：能力-难度(Rasch)形式 P = sigmoid(5 * (ability - difficulty))；
- 逐位置循环更新状态（seq 长 199，与 DKT 的逐位预测语义一致，无信息泄漏）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MIKT(nn.Module):
    """pro_max: 题目数；skill_max: 技能数（pro2skill 的列数）；
    pro2skill: (pro_max, skill_max) 0/1 隶属矩阵（buffer）；
    embed: 题目/技能嵌入维度；state_d: 状态维度；max_seq: 时间间隔编码的最大长度
    （=框架的 max_problem_num，位置间隔不会超过它）；p: dropout。
    """

    def __init__(self, pro_max, skill_max, pro2skill, embed=64, state_d=64,
                 max_seq=200, p=0.4):
        super(MIKT, self).__init__()
        self.pro_max = pro_max
        self.skill_max = skill_max
        # 隶属矩阵是数据统计量：注册为 buffer（不参与梯度、随模型搬设备）
        self.register_buffer("pro2skill", torch.as_tensor(pro2skill, dtype=torch.float32))
        assert self.pro2skill.shape == (pro_max, skill_max), \
            f'pro2skill 形状应为 ({pro_max}, {skill_max})，实际 {tuple(self.pro2skill.shape)}'

        self.pro_embed = nn.Parameter(torch.rand(pro_max, embed))
        nn.init.xavier_uniform_(self.pro_embed)

        self.skill_embed = nn.Parameter(torch.rand(skill_max, embed))
        nn.init.xavier_uniform_(self.skill_embed)

        self.var = nn.Parameter(torch.rand(pro_max, embed))
        self.change = nn.Parameter(torch.rand(pro_max, 1))

        d = embed
        self.pos_embed = nn.Parameter(torch.rand(max_seq, embed))
        nn.init.xavier_uniform_(self.pos_embed)

        # ---- 多尺度状态（技能级 + 全局）----
        self.skill_state = nn.Parameter(torch.rand(skill_max, state_d))
        self.time_state = nn.Parameter(torch.rand(max_seq, state_d))
        self.all_state = nn.Parameter(torch.rand(1, state_d))

        self.all_forget = nn.Sequential(
            nn.Linear(2 * state_d, state_d),
            nn.ReLU(),
            nn.Linear(state_d, state_d),
            nn.Sigmoid()
        )

        self.ans_embed = nn.Embedding(2, d)
        self.lstm = nn.LSTM(d, d, batch_first=True)

        self.now_obtain = nn.Sequential(
            nn.Linear(d, state_d),
            nn.Tanh(),
            nn.Linear(state_d, state_d),
            nn.Tanh()
        )

        self.pro_diff_embed = nn.Parameter(torch.rand(pro_max, d))
        self.pro_diff = nn.Embedding(pro_max, 1)

        self.pro_linear = nn.Linear(d, d)
        self.skill_linear = nn.Linear(d, d)
        self.pro_change = nn.Linear(d, d)

        self.pro_guess = nn.Embedding(pro_max, 1)
        self.pro_divide = nn.Embedding(pro_max, 1)

        self.pro_ability = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Linear(d, 1)
        )

        self.obtain1_linear = nn.Linear(d, d)
        self.obtain2_linear = nn.Linear(d, d)

        self.pro_diff_judge = nn.Linear(d, 1)

        self.all_obtain = nn.Linear(d, d)

        self.skill_forget = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d, d)
        )

        self.do_attn = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d, 1)
        )

        self.predict_attn = nn.Linear(3 * d, d)

        self.dropout = nn.Dropout(p=p)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, last_problem, last_ans, next_problem, next_ans):
        device = last_problem.device

        seq = last_problem.shape[1]
        batch = last_problem.shape[0]

        pro2skill = self.pro2skill                      # (pro_max, skill_max)

        pro_embed = self.pro_embed
        skill_embed = self.skill_embed

        # 题目-技能软表示：题目隶属技能的嵌入均值
        skill_mean = torch.matmul(pro2skill, skill_embed) / (
                torch.sum(pro2skill, dim=-1, keepdims=True) + 1e-8)      # (pro_max, d)

        pro_idx = torch.arange(self.pro_max, device=device)
        pro_diff = torch.sigmoid(self.pro_diff(pro_idx))                 # (pro_max, 1)

        # 题目嵌入 = 隶属技能注意力(掩码为 pro2skill) + 难度偏移 * 技能均值
        q_pro = self.pro_linear(pro_embed)
        q_skill = self.skill_linear(self.skill_embed)
        attn = torch.matmul(q_pro, q_skill.transpose(-1, -2)) / math.sqrt(q_pro.shape[-1])
        attn = torch.masked_fill(attn, pro2skill == 0, -1e9)
        attn = torch.softmax(attn, dim=-1)
        skill_attn = torch.matmul(attn, skill_embed)

        now_embed = skill_attn + pro_diff * self.pro_change(skill_mean)
        pro_embed = self.dropout(now_embed)                              # (pro_max, d)

        # 交互表示：题目嵌入 + 作答嵌入（当前位置与下一位置各取一次）
        last_pro_rasch = F.embedding(last_problem, pro_embed)             # (batch, seq, d)
        next_pro_rasch = F.embedding(next_problem, pro_embed)
        next_pro_diff = F.embedding(next_problem, pro_diff)               # (batch, seq, 1)
        next_pro_guess = torch.sigmoid(self.pro_guess(next_problem))      # (batch, seq, 1)
        next_pro_divide = self.pro_divide(next_problem)

        next_X = next_pro_rasch + self.ans_embed(next_ans.long())
        X = last_pro_rasch + self.ans_embed(last_ans.long())

        time_embed = self.time_state                                     # (max_seq, state_d)

        res_p = []
        # 未出现过的技能：上次交互时间记 0（原实现语义）
        last_skill_time = torch.zeros((batch, self.skill_max), device=device).long()

        skill_state = self.skill_state.unsqueeze(0).repeat(batch, 1, 1)   # (batch, skill_max, state_d)
        all_state = self.all_state.repeat(batch, 1)                      # (batch, state_d)

        for now_step in range(seq):
            now_pro = next_problem[:, now_step]                           # (batch,)
            now_pro2skill = F.embedding(now_pro, pro2skill).unsqueeze(1)   # (batch, 1, skill_max)

            now_pro_embed = next_pro_rasch[:, now_step]                   # (batch, d)

            f1 = now_pro_embed.unsqueeze(1)                               # (batch, 1, d)
            f2 = skill_state                                             # (batch, skill_max, state_d)

            # 距该技能上次交互的位置间隔（多尺度时间信号）
            skill_time_gap = now_step - now_pro2skill.squeeze(1) * last_skill_time   # (batch, skill_max)
            skill_time_gap_embed = F.embedding(skill_time_gap.clamp_(0, time_embed.shape[0] - 1).long(),
                                              time_embed)                 # (batch, skill_max, state_d)

            # 全局状态：LSTM 做一次遗忘（原实现对单步序列做 LSTM，等价于一层门控）
            state_sequence = all_state.unsqueeze(1)                        # (batch, 1, d)
            lstm_output, _ = self.lstm(state_sequence)
            forget_now_all_state = lstm_output[:, -1, :]                   # (batch, d)

            effect_all_state = forget_now_all_state.unsqueeze(1).repeat(1, f2.shape[1], 1)

            # 技能状态遗忘门（时间间隔 + 全局状态作为条件），非本技能列不衰减
            skill_forget = torch.sigmoid(self.skill_forget(
                self.dropout(torch.cat([skill_state, skill_time_gap_embed, effect_all_state], dim=-1))))
            skill_forget = torch.masked_fill(skill_forget, now_pro2skill.transpose(-1, -2) == 0, 1)
            skill_state = skill_state * skill_forget

            # 题目对技能状态的注意力（掩码为本题目所属技能）
            now_pro_skill_attn = torch.matmul(f1, skill_state.transpose(-1, -2)) / f1.shape[-1]
            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, now_pro2skill == 0, -1e9)
            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)

            now_need_state = torch.matmul(now_pro_skill_attn, skill_state).squeeze(1)   # (batch, state_d)

            # 技能状态与全局状态的自适应融合
            all_attn = torch.sigmoid(self.predict_attn(
                self.dropout(torch.cat([now_need_state, forget_now_all_state, now_pro_embed], dim=-1))))

            now_need_state = torch.cat([(1 - all_attn) * now_need_state,
                                        all_attn * forget_now_all_state], dim=-1)

            last_skill_time = torch.masked_fill(last_skill_time,
                                                now_pro2skill.squeeze(1) == 1, now_step)

            # 能力-难度(Rasch)：sigmoid(5 * (ability - difficulty))
            now_ability = torch.sigmoid(self.pro_ability(torch.cat([now_need_state, now_pro_embed], dim=-1)))
            now_diff = F.embedding(now_pro, pro_diff)                      # (batch, 1)

            # clamp 到开区间，避免 BCELoss 在概率精确等于 0/1 时 log(0)=inf
            now_output = torch.sigmoid(5 * (now_ability - now_diff)).squeeze(-1).clamp(1e-6, 1 - 1e-6)
            res_p.append(now_output)

            # 用当前交互更新全局状态与技能状态（含当前作答，供下一步使用）
            now_X = next_X[:, now_step]                                    # (batch, d)
            all_state = forget_now_all_state + torch.tanh(self.all_obtain(self.dropout(now_X)))

            to_get = torch.tanh(self.now_obtain(self.dropout(now_X))).unsqueeze(1)   # (batch, 1, d)
            now_pro_skill_attn = torch.matmul(to_get, skill_state.transpose(-1, -2)) / to_get.shape[-1]
            now_pro_skill_attn = torch.masked_fill(now_pro_skill_attn, now_pro2skill == 0, -1e9)
            now_pro_skill_attn = torch.softmax(now_pro_skill_attn, dim=-1)
            now_get = torch.matmul(now_pro_skill_attn.transpose(-1, -2), to_get)     # (batch, skill_max, state_d)

            skill_state = skill_state + now_get

        P = torch.vstack(res_p).T                                          # (batch, seq)
        return P, None
