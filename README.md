# Model Merging Project

机器学习导论课程项目：基于 MergeKit 的模型合并实验，并复现了 EMR-Merging (NeurIPS 2024) 官方代码。


## Level 1: 模型合并

将数学专家和代码专家进行合并，验证融合后是否同时保留两种能力。

- 基座模型：mistralai/Mistral-7B-v0.1
- 数学专家：meta-math/MetaMath-Mistral-7B
- 代码专家：beowolx/CodeNinja-1.0-OpenChat-7B
- 合并算法：SLERP (t=0.25/0.5/0.75)、TIES-Merging、Task Arithmetic
- 评估基准：GSM8K（数学）、SAT Math（数学）、HumanEval（代码）、MBPP（代码）

### 核心思路

数学专家和代码专家均基于同一个基座模型（Mistral-7B-v0.1）微调而来，具备相同的网络架构和权重空间，因此可以直接合并。

- **SLERP**：直接在数学专家和代码专家的权重之间进行球面线性插值，生成最终合并模型。基座模型本身不参与合并计算。本实验测试了 t=0.25、0.5、0.75 三种插值比例。

- **TIES-Merging**：本实验中，由于数学专家（32001）和代码专家（32002）词表大小不一致，无法直接计算任务向量。因此先用 `mergekit-tokensurgeon` 将两者的词表统一为 32002，再以对齐后的模型计算任务向量进行合并。本实验取density为 0.3 以剔除微小噪声。

- **Task Arithmetic**：与 TIES 类似，合并前需先通过 `mergekit-tokensurgeon` 统一词表。同样基于任务向量，将各专家的任务向量加权求和后加回基座模型。。

### 文件说明

- `configs/` ： 合并配置文件
- `results/` ： 完整评估结果
- `requirements.txt` ： Python 依赖列表

### 合并示例

```bash
mergekit-yaml configs/slerp-experts.yaml ./merged_experts --cuda
```

## Level 2: EMR-Merging 复现

跑通EMR-Merging (NeurIPS 2024)，在视觉（ViT）和自然语言处理（RoBERTa、GPT-2）两个领域多个任务上验证了论文报告的性能。

### 核心修改说明

- **动态模型参数**：`src/main_emr_merging.py` 中移除对 `ViT-B-32` 的硬编码，通过 `argparse` 引入 `--model` 参数，支持 `ViT-B-32`、`ViT-B-16`、`ViT-L-14` 等架构灵活切换。
- **环境兼容性修复**：修改 `task_vectors.py` 中的 `torch.load` 加载方式，增加 `map_location='cpu'` 及 `state_dict` 适配，解决跨环境加载权重时的 `ModuleNotFoundError: No module named 'src'` 问题。
- **评估列表命令行覆盖**：支持通过 `--train-dataset` 参数动态覆盖 `eval_datasets` 列表，便于快速验证单个或部分数据集。

### 文件结构

```bash
EMR_Merging/
├── merge_vit/                              # ViT实验
│   ├── src/
│   │   ├── home/
│   │   │   └── emr-merging/         
│   │   │       ├── data/                   # 数据集
│   │   │       ├── checkpoints/            # 各架构模型权重 (ViT-B-32、B-16、L-14)
│   │   │       └── logs/                   # 测试结果日志
│   │   ├── main_emr_merging.py             # 核心入口
│   │   ├── task_vectors.py                 # 任务向量计算与合并算子
│   │   └── ...
│   └── ...
│
└── merge_lm/                               # NLP实验
    ├── merge_roberta_glue.py               # RoBERTa GLUE 实验入口
    ├── merge_gpt_glue.py                   # GPT-2 GLUE 实验入口
    ├── ckpts/                              # 模型权重目录 (RoBERTa和GPT-2)
    ├── save_merge_logs/                    # 实验结果日志
    │   └── emr-merging/
    │       ├── gpt2/                       # GPT-2实验日志
    │       └── roberta-base/               # RoBERTa实验日志
    └── ...
```

### 运行示例
ViT实验：在merge_vit/目录下

```bash
# 示例：使用 ViT-B-16 在 SVHN 数据集上验证
python -m src.main_emr_merging --model ViT-B-16 --train-dataset SVHN
```
NLP实验：在merge_lm/目录下

```bash
# RoBERTa 实验（8 个 GLUE 任务）
python merge_roberta_glue.py --gpu 0

# GPT-2 实验（7 个 GLUE 任务）
python merge_gpt_glue.py --gpu 0
```