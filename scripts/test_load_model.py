import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vlm_with_lora import VLMWithLoRA
import torch
import torch.distributed as dist
import logging
import torch.nn as nn
def load_base_model(model_name, model_type, lora_rank, task_type):
    model = VLMWithLoRA(
        model_name=model_name,
        model_type=model_type,
        lora_rank=lora_rank,
        num_answers=1000,  # 假设 VQA 任务有 1000 个可能的答案
        task_type=task_type
    )
    return model
# 创建 VLMWithLoRA 实例
model_name = "openai/clip-vit-base-patch16"  # CLIP 模型名称
model_type = "CLIP"  # 模型类型
lora_rank = 16  # LoRA 秩
task_type = "contrastive"  # 任务类型
model = load_base_model(model_name, model_type, lora_rank, task_type)
logging.info(f"Rank : 加载 {model_type} 模型 ({model_name}) 与 LoRA (rank={lora_rank})")



# 打印可训练参数信息
model.print_trainable_parameters()

    
