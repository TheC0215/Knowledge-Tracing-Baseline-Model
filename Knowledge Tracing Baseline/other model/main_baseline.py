"""六个 pyKT 基准模型（DKVMN/AKT/SAKT/simpleKT/UKT/GKT）的统一五折交叉验证主脚本。

用法示例：
    python main_baseline.py --model_name gkt
    python main_baseline.py --model_name akt --dataset assist09 --n_folds 5

结构与 main_dkt.py 完全一致（同协议、同 run_epoch、同早停规则），只是把模型换成
注册表里的六个基线，且 batch_size / 学习率 / 结构超参按 pyKT 对应 wandb 脚本对齐
（见下方 MODELS 注册表；其中 batch_size 按本框架约定：akt=64、gkt=16、其余=256，
注意 pyKT wandb 脚本把 dkvmn/sakt/simplekt 也设成 64，此处以本框架统一 256 为准）。

GKT 特殊：数据端需要 (pro_max x num_c) 题目-技能矩阵与 num_c x num_c 技能转移图，
两者都在脚本启动时从数据一次性构建（与 pyKT 的 get_gkt_graph 相同思想，但 pyKT
用 train+test 拼接建图；这里只用训练用户会话建图，测试折不参与）。
"""
import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn

from akt import AKT
from dkt import DKT
from dkvmn import DKVMN
from gkt import GKT, build_dense_graph, build_pro2skill, build_transition_graph
from run import run_epoch
from sakt import SAKT
from simplekt import simpleKT
from ukt import UKT

# 数据在上级目录 模型/data 下（以本脚本位置为锚点，与运行目录无关）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, '..', 'data')

mp2path = {
    'assist09': {
        'ques_skill_path': os.path.join(DATA_ROOT, 'assist09', 'ques_skill.csv'),
        'train_path': os.path.join(DATA_ROOT, 'assist09', 'train_question.txt'),
        'test_path': os.path.join(DATA_ROOT, 'assist09', 'test_question.txt'),
        'train_skill_path': os.path.join(DATA_ROOT, 'assist09', 'train_skill.txt'),
        'test_skill_path': os.path.join(DATA_ROOT, 'assist09', 'test_skill.txt'),
        'fold_path': os.path.join(DATA_ROOT, 'assist09', 'train_fold.txt'),
        'skill_max': 999},
    'assist12': {
        'ques_skill_path': os.path.join(DATA_ROOT, 'assist12', 'ques_skill.csv'),
        'train_path': os.path.join(DATA_ROOT, 'assist12', 'train_question.txt'),
        'test_path': os.path.join(DATA_ROOT, 'assist12', 'test_question.txt'),
        'train_skill_path': os.path.join(DATA_ROOT, 'assist12', 'train_skill.txt'),
        'test_skill_path': os.path.join(DATA_ROOT, 'assist12', 'test_skill.txt'),
        'fold_path': os.path.join(DATA_ROOT, 'assist12', 'train_fold.txt'),
        'skill_max': 999},
    'NeurIPS 2020': {
        'ques_skill_path': os.path.join(DATA_ROOT, 'NeurIPS 2020', 'ques_skill.csv'),
        'train_path': os.path.join(DATA_ROOT, 'NeurIPS 2020', 'train_question.txt'),
        'test_path': os.path.join(DATA_ROOT, 'NeurIPS 2020', 'test_question.txt'),
        'train_skill_path': os.path.join(DATA_ROOT, 'NeurIPS 2020', 'train_skill.txt'),
        'test_skill_path': os.path.join(DATA_ROOT, 'NeurIPS 2020', 'test_skill.txt'),
        'fold_path': os.path.join(DATA_ROOT, 'NeurIPS 2020', 'train_fold.txt'),
        'skill_max': 999},
}

# 模型注册表：结构超参与 lr 取自 pyKT examples/wandb_*_train.py（用户核对版），
# batch_size 按用户映射（akt=64、gkt=16、其余 256）
MODELS = {
    'dkt': dict(cls=DKT, params=dict(emb_size=200, dropout=0.2),
                lr=1e-3, batch_size=256),
    'dkvmn': dict(cls=DKVMN, params=dict(dim_s=200, size_m=50, dropout=0.2),
                  lr=1e-3, batch_size=256),
    'akt': dict(cls=AKT, params=dict(d_model=256, d_ff=512, n_blocks=4,
                                     num_attn_heads=8, dropout=0.2),
                lr=1e-4, batch_size=64),
    'sakt': dict(cls=SAKT, params=dict(emb_size=256, num_attn_heads=8, num_en=1,
                                       dropout=0.2),
                 lr=1e-3, batch_size=256),
    'simplekt': dict(cls=simpleKT, params=dict(d_model=256, d_ff=256, n_blocks=2,
                                               num_attn_heads=4, dropout=0.1,
                                               final_fc_dim=256, final_fc_dim2=256),
                     lr=1e-4, batch_size=256),
    'ukt': dict(cls=UKT, params=dict(d_model=256, d_ff=512, n_blocks=4,
                                     num_attn_heads=8, dropout=0.2, use_CL=False,
                                     emb_type='stoc_qid'),
                lr=1e-4, batch_size=256),
    'gkt': dict(cls=GKT, params=dict(hidden_dim=64, emb_size=64, dropout=0.5,
                                     graph_type='transition'),
                lr=1e-2, batch_size=16),
}

def build_model(model_name, pro_max, ques_skill_path, train_path, device):
    """按注册表构建模型；GKT 额外从数据构建题目-技能矩阵与转移图。"""
    cfg = MODELS[model_name]
    if model_name == 'gkt':
        pro2skill, _, skill_max = build_pro2skill(ques_skill_path)
        if cfg['params']['graph_type'] == 'transition':
            graph = build_transition_graph(train_path, pro2skill)
        else:  # 'dense'：均匀稠密图
            graph = build_dense_graph(skill_max)
        model = cfg['cls'](num_c=skill_max, pro2skill=pro2skill, graph=graph,
                           **cfg['params'])
    else:
        kwargs = dict(pro_max=pro_max) if model_name != 'sakt' else dict(num_c=pro_max)
        model = cfg['cls'](**kwargs, **cfg['params'])
    return model.to(device)


def main():
    parser = argparse.ArgumentParser(description='pyKT 六个基线模型的五折交叉验证')
    parser.add_argument('--model_name', type=str, required=True,
                        choices=sorted(MODELS.keys()),
                        help='模型名：dkt/dkvmn/akt/sakt/simplekt/ukt/gkt')
    parser.add_argument('--dataset', type=str, default='assist09', choices=list(mp2path.keys()))
    parser.add_argument('--n_folds', type=int, default=5)
    args = parser.parse_args()

    model_name = args.model_name
    dataset = args.dataset
    n_folds = args.n_folds
    cfg = MODELS[model_name]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 对齐 main_dkt.py 的训练协议
    epochs = 200
    min_seq = 3
    max_seq = 200
    grad_clip = 15.0
    patience = 10

    p = mp2path[dataset]
    train_skill_path, test_skill_path = p['train_skill_path'], p['test_skill_path']
    train_path, test_path = p['train_path'], p['test_path']
    fold_path = p['fold_path']
    pro_max = 1 + int(max(pd.read_csv(p['ques_skill_path']).values[:, 0]))

    criterion = nn.BCELoss()
    classify = nn.CrossEntropyLoss()

    avg_auc, avg_acc = 0.0, 0.0
    fold_aucs = []
    all_folds = set(range(n_folds))

    for now_step in range(n_folds):
        train_folds = all_folds - {now_step}

        model = build_model(model_name, pro_max, p['ques_skill_path'], train_path, device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        n_params = sum(p_.numel() for p_ in model.parameters())
        print(f'model: {model_name} ({model.__class__.__name__}), trainable params: {n_params}, '
              f'batch_size: {cfg["batch_size"]}, lr: {cfg["lr"]}, dataset: {dataset}')

        best_valid_auc, best_valid_acc, bad_cnt = 0.0, 0.0, 0
        ckpt_name = f"./{model_name.upper()}_{dataset}_{now_step}_model.pkl"

        t0 = time.time()
        for epoch in range(epochs):
            train_loss, train_acc, train_auc = run_epoch(
                classify, train_skill_path, model, optimizer, pro_max, train_path,
                cfg['batch_size'], True, min_seq, max_seq, criterion, device,
                grad_clip, folds=train_folds, fold_path=fold_path)
            print(f'epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, '
                  f'train_auc: {train_auc:.4f}')

            valid_loss, valid_acc, valid_auc = run_epoch(
                classify, train_skill_path, model, optimizer, pro_max, train_path,
                cfg['batch_size'], False, min_seq, max_seq, criterion, device,
                grad_clip, folds={now_step}, fold_path=fold_path)
            print(f'epoch: {epoch}, valid_loss: {valid_loss:.4f}, valid_acc: {valid_acc:.4f}, '
                  f'valid_auc: {valid_auc:.4f}')

            # 早停 + 模型选择：只看验证集 AUC
            if valid_auc >= best_valid_auc:
                best_valid_auc = valid_auc
                best_valid_acc = valid_acc
                bad_cnt = 0
                torch.save(model.state_dict(), ckpt_name)
            else:
                bad_cnt += 1
                if bad_cnt >= patience:
                    print(f'fold {now_step}: 验证 AUC 连续 {patience} 轮不涨，早停于 epoch {epoch}')
                    break

        fold_time = time.time() - t0
        print(f'fold {now_step}: 训练耗时 {fold_time:.1f} 秒')

        print('*******************************************************************************')
        print(f'fold {now_step}: best_valid_auc: {best_valid_auc:.4f}, best_valid_acc: {best_valid_acc:.4f}')

        # 载入 best checkpoint，在测试集上评一次（测试集只用于最终报告）
        model.load_state_dict(torch.load(ckpt_name))
        test_loss, test_acc, test_auc = run_epoch(
            classify, test_skill_path, model, optimizer, pro_max, test_path,
            cfg['batch_size'], False, min_seq, max_seq, criterion, device, grad_clip)
        print(f'fold {now_step}: test_loss: {test_loss:.4f}, test_acc: {test_acc:.4f}, '
              f'test_auc: {test_auc:.4f}')
        print('*******************************************************************************')

        avg_auc += test_auc
        avg_acc += test_acc
        fold_aucs.append(test_auc)

    avg_auc /= n_folds
    avg_acc /= n_folds

    # ============ 保存结果：总体 AUC/ACC ============
    result_file = f'./{model_name.upper()}_{dataset}_results.txt'
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write(f'模型: {model_name.upper()} | 数据集: {dataset}\n')
        f.write(f'总体({n_folds}折平均): AUC={avg_auc:.4f} ACC={avg_acc:.4f}\n')
        f.write('逐折总体 AUC: ' + ', '.join(
            f'fold{i}={a:.4f}' for i, a in enumerate(fold_aucs)) + '\n')
    print(f'结果已保存 -> {result_file}')

    print('*******************************************************************************')
    print('*******************************************************************************')
    print(f'final_avg_acc: {avg_acc:.4f}, final_avg_auc: {avg_auc:.4f}')
    print('*******************************************************************************')
    print('*******************************************************************************')


if __name__ == '__main__':
    main()
