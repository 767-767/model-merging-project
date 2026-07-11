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

跑通EMR-Merging (NeurIPS 2024)，在MNIST和SVHN上分别验证了ViT-B-32, ViT-B-16和ViT-L-14在论文报告中的性能。

### 核心修改说明

- `src/main_emr_merging.py`：移除了原代码中对 ViT-B-32 的硬编码，通过 argparse 引入 --model 参数，支持通过命令行参数灵活切换 ViT-B-32, ViT-B-16, ViT-L-14 等不同架构的模型。

### 文件结构

```bash
EMR_Merging/merge_vit/src/
├── home/                    # 运行时资源目录
│   └── emr-merging/         
│        ├── data/           # 数据集文件
│        ├── checkpoints/    # 各架构模型权重
│        └── logs            # 测试结果日志    
├── main_emr_merging.py      # 核心入口
├── task_vectors.py          # 任务向量计算与合并算子
├── utils.py                 # 数据与权重路径管理          
└── ...
```

### 运行示例
在 src 目录下，使用以下命令对指定模型及数据集进行验证

```bash
# 示例：使用 ViT-B-16 在 SVHN 数据集上验证
python main_emr_merging.py --model ViT-B-16 --train-dataset SVHN
```