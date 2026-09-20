# Knowledge Tracing Baseline

知识追踪（Knowledge Tracing）基线仓库：统一的数据预处理流程 + pyKT 风格的六种经典深度知识追踪基线模型的五折交叉验证。

- `preprocessdata.py`：数据集清洗、划分、编码，输出标准格式的序列文件
- `other model/main_baseline.py`：统一入口，运行对应基线模型
- `data/`：存放原始数据集与预处理产物

## 支持的模型

模型实现对齐 [pyKT](https://pykt.org/)（`pykt/models/` 的 qid 版），接口适配本仓库的数据格式。

| 模型 | 论文 | 会议/年份 |
| ---- | ---- | --------- |
| DKT | Deep Knowledge Tracing | NIPS 2015 |
| DKVMN | Dynamic Key-Value Memory Networks for Knowledge Tracing | WWW 2017 |
| SAKT | A Self-Attentive Model for Knowledge Tracing | EDM 2019 |
| GKT | Graph-based Knowledge Tracing: Modeling Student Proficiency Using Graph Neural Network | WWW 2019 |
| AKT | Context-Aware Attentive Knowledge Tracing | KDD 2020 |
| simpleKT | simpleKT: A Simple But Tough-to-Beat Baseline for Knowledge Tracing | ICLR 2023 |
| UKT | Uncertainty-aware Knowledge Tracing | AAAI 2025 |

## 目录结构

```
Knowledge Tracing Baseline/
├── preprocessdata.py        # 数据预处理主脚本
├── data/                    # 数据集目录
│   ├── assist09/            # ASSISTments 2009-2010（原始 CSV 放这里）
│   ├── assist12/            # ASSISTments 2012-2013
│   └── NeurIPS 2020/        # NeurIPS 2020 Education Challenge（Task 3&4）
└── other model/             # 模型实现与训练入口
    ├── main_baseline.py     # 训练主脚本（五折交叉验证）
    ├── run.py               # 单 epoch 训练/评估
    ├── load_data.py         # 数据加载器
    ├── utils.py             # 公共工具函数
    ├── dkt.py               # 各模型实现
    ├── dkvmn.py
    ├── akt.py
    ├── sakt.py
    ├── gkt.py
    ├── simplekt.py
    └── ukt.py
```

## 环境依赖

- Python 3.8+
- numpy、pandas、scikit-learn、tqdm
- PyTorch（支持 CPU / CUDA）

```bash
pip install numpy pandas scikit-learn tqdm torch
```

## 快速开始

### 1. 准备数据集

把原始 CSV 放入对应的 `data/<数据集>/` 目录：

| 目录 | 需要的文件 | 数据来源 |
| ---- | ---------- | -------- |
| `data/assist09/` | `skill_builder_data_corrected_collapsed.csv` | [ASSISTments 2009-2010 skill builder 数据](https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010) |
| `data/assist12/` | `2012-2013-data-with-predictions-4-final.csv` | [ASSISTments 2012-2013 school data with affect](https://sites.google.com/site/assistmentsdata/datasets/2012-13-school-data-with-affect) |
| `data/NeurIPS 2020/` | `train_task_3_4.csv`、`question_metadata_task_3_4.csv` | [NeurIPS 2020 Education Challenge](https://competitions.codalab.org/competitions/25449) 官方下载：[data.zip](https://dqanonymousdata.blob.core.windows.net/neurips-public/data.zip) |

### 2. 数据预处理

在 `preprocessdata.py` 中设置 `dataname`（`"assist09"` / `"assist12"` / `"NeurIPS 2020"`），然后运行：

```bash
python preprocessdata.py
```

脚本依次完成：清洗（去重、排序、移除空技能/脚手架题目）→ 按学生 8:2 划分训练/测试集 → 编码题目/技能 → 训练学生随机分 5 折 → 构建题目-技能映射 → 生成序列文件（超长会话按 `MAX_SEQ=200` 切段）。

在 `data/<数据集>/` 下生成：

| 文件 | 说明 |
| ---- | ---- |
| `ques_skill.csv` | 题目→技能映射（两列：`ques,skill`） |
| `train_question.txt` / `test_question.txt` | 题目序列（3 行块：长度 / 题目 / 作答） |
| `train_skill.txt` / `test_skill.txt` | 技能序列（3 行块：长度 / 技能 / 作答） |
| `train_fold.txt` | 每个训练块对应的折号（与训练文件逐块对齐） |

### 3. 训练与评估

```bash
cd "other model"
python main_baseline.py --model_name gkt --dataset assist09 --n_folds 5
```

参数：

| 参数 | 说明 |
| ---- | ---- |
| `--model_name` | 模型名，必填：`dkt` / `dkvmn` / `akt` / `sakt` / `simplekt` / `ukt` / `gkt` |
| `--dataset` | 数据集：`assist09`（默认）/ `assist12` / `NeurIPS 2020` |
| `--n_folds` | 交叉验证折数，默认 5 |

各模型的超参与学习率（与 pyKT 对应脚本对齐）：

| 模型 | 学习率 | batch_size |
| ---- | ------ | ---------- |
| DKT | 1e-3 | 256 |
| DKVMN | 1e-3 | 256 |
| AKT | 1e-4 | 64 |
| SAKT | 1e-3 | 256 |
| simpleKT | 1e-4 | 256 |
| UKT | 1e-4 | 256 |
| GKT | 1e-2 | 16 |

训练结束后，在 `other model/` 下输出：

- `{模型}_{数据集}_results.txt`：总体 AUC/ACC（各折平均）与逐折 AUC
- `{模型}_{数据集}_{折号}_model.pkl`：每折最优 checkpoint（按验证 AUC 选择）

## 训练协议

- 训练/测试按**学生** 8:2 划分，测试学生完全不参与训练
- 训练学生随机分 5 折（pyKT 风格），轮流用 4 折训练、1 折验证
- 每折最多 200 epoch，验证 AUC 连续 10 轮不涨则早停
- 序列窗口：最小 3 步、最大 200 步；梯度裁剪 15.0
- 评价指标：AUC 与 ACC（预测概率 ≥ 0.5 判为答对）
- 测试集仅在每折选出最优 checkpoint 后用于最终评测

## 自定义数据集

在 `preprocessdata.py` 的 `DATASET_CONFIGS` 中新增一条配置（参考注释掉的 `mydata` 模板），并把原始 CSV 放到 `data/<数据集名>/`：

- 交互表有知识点列：填写 `col_skill`
- 交互表没有知识点列：像 `NeurIPS 2020` 一样提供 `meta_file` / `meta_problem` / `meta_skill` 指向题目元数据表

同时在 `main_baseline.py` 的 `mp2path` 中注册该数据集的输出路径。

## 致谢

模型实现基于 [pyKT](https://github.com/pykt-team/pykt-toolkit)（MIT License）移植，数据预处理与训练框架参考 pyKT 的公共脚本。

## License

本项目采用 [MIT License](LICENSE.md)。
