import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from sklearn import metrics

from tqdm import tqdm
from load_data import getLoader

def run_epoch(classify, skill_path, model, optimizer, max_problem, path, batch_size, is_train, min_problem_num,
              max_problem_num, criterion, device, grad_clip, folds=None, fold_path=None):
    total_correct = 0
    total_num = 0
    total_loss = []

    labels = []
    outputs = []

    if is_train:
        model.train()
    else:
        model.eval()

    data_loader = getLoader(max_problem, skill_path, path, batch_size, is_train, min_problem_num, max_problem_num,
                            folds=folds, fold_path=fold_path)

    for i, data in tqdm(enumerate(data_loader), desc='加载中...'):

        last_problem, last_ans, next_problem, next_ans, mask = data

        if is_train:
            predict, attn = model(last_problem, last_ans, next_problem, next_ans.long())

            next_predict = torch.masked_select(predict, mask)
            next_true = torch.masked_select(next_ans, mask)

            kt_loss = criterion(next_predict, next_true)

            # 可选辅助损失钩子：模型 forward 里写入 model.aux_loss（如 RIGL 的对比损失），
            # 这里汇总后清空。对没有该属性的模型（其余基线）完全无影响。
            aux = getattr(model, 'aux_loss', None)
            if aux is not None:
                kt_loss = kt_loss + aux
                model.aux_loss = None

            optimizer.zero_grad()
            kt_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            labels.extend(next_true.view(-1).data.cpu().numpy())
            outputs.extend(next_predict.view(-1).data.cpu().numpy())
            total_loss.append(kt_loss.item())
            total_num += len(next_true)
            to_pred = (next_predict >= 0.5).long()
            total_correct += (next_true == to_pred).sum()
        else:
            with torch.no_grad():
                predict, attn = model(last_problem, last_ans, next_problem, next_ans.long())
                next_predict = torch.masked_select(predict, mask)
                next_true = torch.masked_select(next_ans, mask)
                kt_loss = criterion(next_predict, next_true)
                labels.extend(next_true.view(-1).data.cpu().numpy())
                outputs.extend(next_predict.view(-1).data.cpu().numpy())
                total_loss.append(kt_loss.item())
                total_num += len(next_true)
                to_pred = (next_predict >= 0.5).long()
                total_correct += (next_true == to_pred).sum()

    avg_loss = np.average(total_loss)
    acc = total_correct * 1.0 / total_num
    auc = metrics.roc_auc_score(labels, outputs)

    return avg_loss, acc, auc
