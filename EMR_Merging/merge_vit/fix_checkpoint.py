import torch
import pickle
import sys

# 修复的路径
file_path = "/root/autodl-tmp/EMR_Merging/merge_vit/checkpoints/ViT-B-16/Cars/finetuned.pt"

print(f"正在读取并修复: {file_path}")

try:
    # 尝试用 pickle 直接读取（这会绕过 torch.load 的严格校验）
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # 用当前的 torch 环境重新保存
    torch.save(data, file_path)
    print("成功！文件已使用当前环境格式重新保存。")
except Exception as e:
    print(f"修复失败: {e}")
