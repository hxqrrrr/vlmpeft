# VLMPEFT: 

## 项目概述

vlmpeft 是一个基于 **PyTorch** 和 **Hugging Face** 的视觉-语言模型（Vision-Language Model, VLM）微调项目，利用 **LoRA（Low-Rank Adaptation）** 实现参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）。项目支持多种模型（如 CLIP 和 LLaVA）和任务（VQA、图像-文本匹配、图像描述生成），目前适配 Flickr8k 数据集，并具备扩展性。

### 功能亮点

- **多模型支持**：包括 CLIP（如 openai/clip-vit-base-patch16）和 LLaVA（如 liuhaotian/llava-v1.5-7b）。
- **任务切换**：通过配置文件动态切换 VQA、contrastive 和 captioning 任务。
- **高效微调**：LoRA 减少训练参数，提升计算效率。
- **模块化设计**：便于扩展新模型和任务。

------

## 安装指南

在项目根目录运行以下命令安装依赖：

```
pip install -r requirements.txt
```

### 数据准备

1. 下载 Flickr8k
   - 从 [Kaggle](https://www.kaggle.com/adityajn105/flickr8k) 下载数据集。

------

## 使用步骤

```
python main.py --config configs/clip_contrastive.yaml
```

------

## 任务支持（正在实现）

| 任务            | 描述             | 支持模型    | 输入        | 输出       | 注意事项                           |
| --------------- | ---------------- | ----------- | ----------- | ---------- | ---------------------------------- |
| **VQA**         | 回答图像相关问题 | CLIP, LLaVA | 图像 + 问题 | 分类答案   | Flickr8k 需模拟标签，推荐 VQA v2.0 |
| **Contrastive** | 图像-文本匹配    | CLIP        | 图像 + 文本 | 相似度矩阵 | 适合 Flickr8k                      |
| **Captioning**  | 生成图像描述     | LLaVA       | 图像 + 提示 | 文本序列   | 当前需改进为自回归生成             |

## 
