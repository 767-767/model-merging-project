# Model Merging Project

基于 MergeKit 的模型合并实验，将数学专家和代码专家两个模型进行合并，验证融合后是否同时保留两种能力。

## 实验内容

- 基座模型：mistralai/Mistral-7B-v0.1
- 数学专家：meta-math/MetaMath-Mistral-7B
- 代码专家：beowolx/CodeNinja-1.0-OpenChat-7B
- 合并算法：SLERP (Spherical Linear Interpolation)

## 核心思路

数学专家（MetaMath-Mistral-7B）和代码专家（CodeNinja-1.0-OpenChat-7B）均基于同一个基座模型（Mistral-7B-v0.1）微调而来，具备相同的网络架构和权重空间，因此可以直接合并。

在本实验中，SLERP 算法直接在数学专家和代码专家的权重之间进行球面线性插值，生成最终合并模型。基座模型本身不参与合并计算。

## 文件说明

- `configs/slerp-experts.yaml` ： SLERP 合并配置文件
- `EMR_Merging/` ： Level 2 论文复现代码
- `scripts/` ： 评估脚本
- `results/` ： 完整评估结果（JSON 格式）
- `requirements.txt` ： Python 依赖列表

## 如何运行合并

```bash
mergekit-yaml configs/slerp-experts.yaml ./merged_experts --cuda