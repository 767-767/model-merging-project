# Model Merging Project

基于 MergeKit 的模型合并实验，将数学专家和代码专家合并到 Mistral-7B 基座模型上。

## 实验内容
- 基座模型：mistralai/Mistral-7B-v0.1
- 数学专家：meta-math/MetaMath-Mistral-7B
- 代码专家：beowolx/CodeNinja-1.0-OpenChat-7B
- 合并算法：SLERP

## 文件说明
- `configs/slerp-merge.yaml` ： SLERP 合并配置文件
- `EMR_Merging/` ： Level 2 论文复现代码
- `scripts/` ： 评估脚本（待添加）
- `results/` ： 评估结果（待添加）
- `requirements.txt` ： Python 依赖列表

## 如何运行合并
```bash
mergekit-yaml configs/slerp-merge.yaml ./merged_math --cuda