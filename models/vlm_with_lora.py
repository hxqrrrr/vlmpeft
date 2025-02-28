import torch
import torch.nn as nn
from transformers import CLIPModel, ViTModel  # 示例扩展模型
from peft import LoraConfig, get_peft_model
from typing import Dict, Callable

class VLMWithLoRA(nn.Module):
    # 模型加载器注册表
    _model_loaders: Dict[str, Callable] = {}

    def __init__(self, model_name: str, model_type: str, lora_rank: int, num_answers: int):
        """
        初始化 VLMWithLoRA 类，支持多种模型类型并应用LoRA。

        Args:
            model_name (str): Hugging Face 模型名称，例如 "openai/clip-vit-base-patch32"。
            model_type (str): 模型类型，例如 "CLIP" 或 "ViT"。
            lora_rank (int): LoRA的秩，控制可训练参数量。
            num_answers (int): VQA任务的答案类别数。
        """
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type
        self.lora_rank = lora_rank
        self.num_answers = num_answers
        self.model = self._load_model()
        self._configure_task_head()

    def _load_model(self):
        """
        加载指定类型的模型并应用LoRA。

        Returns:
            model: 配置了LoRA的模型实例。

        Raises:
            ValueError: 如果模型类型未注册。
        """
        if self.model_type not in self._model_loaders:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types: {list(self._model_loaders.keys())}")
        
        # 调用注册的加载器
        base_model = self._model_loaders[self.model_type](self.model_name)
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],  # 默认针对注意力层，可根据模型调整
            lora_dropout=0.1
        )
        return get_peft_model(base_model, lora_config)

    def _configure_task_head(self):
        """
        根据模型类型配置任务头部（VQA为例）。
        """
        if self.model_type == "CLIP":
            hidden_size = self.model.config.projection_dim
            self.vqa_head = nn.Linear(hidden_size * 2, self.num_answers)  # 图像+文本嵌入拼接
        elif self.model_type == "ViT":
            hidden_size = self.model.config.hidden_size
            self.vqa_head = nn.Linear(hidden_size, self.num_answers)  # ViT仅图像输入
        else:
            raise NotImplementedError(f"Task head not implemented for {self.model_type}")

    def forward(self, **inputs):
        """
        前向传播，生成VQA预测。

        Args:
            **inputs: 模型输入，根据模型类型不同（如 pixel_values, input_ids 等）。

        Returns:
            logits (torch.Tensor): VQA预测，形状 [batch_size, num_answers]。
        """
        if self.model_type == "CLIP":
            pixel_values = inputs["pixel_values"]
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            outputs = self.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            image_embeds = outputs.image_embeds  # [batch_size, hidden_size]
            text_embeds = outputs.text_embeds    # [batch_size, hidden_size]
            # 计算相似度矩阵
            logits_per_image = torch.matmul(image_embeds, text_embeds.t()) / self.model.logit_scale.exp()
            return logits_per_image
        
        elif self.model_type == "ViT":
            pixel_values = inputs["pixel_values"]
            outputs = self.model(pixel_values=pixel_values)
            image_embeds = outputs.pooler_output  # ViT的CLS token输出
            return self.vqa_head(image_embeds)
        else:
            raise NotImplementedError(f"Forward not implemented for {self.model_type}")

    def print_trainable_parameters(self):
        """
        打印模型的总参数量和可训练参数量。
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {trainable_params / total_params * 100:.2f}%")

    @classmethod
    def register_model_loader(cls, model_type: str, loader: Callable):
        """
        注册新的模型加载器。

        Args:
            model_type (str): 模型类型名称，例如 "CLIP" 或 "ViT"。
            loader (Callable): 加载模型的函数，接受 model_name 并返回模型实例。
        """
        cls._model_loaders[model_type] = loader

# 注册默认模型加载器
def load_clip_model(model_name: str):
    return CLIPModel.from_pretrained(model_name)

def load_vit_model(model_name: str):
    return ViTModel.from_pretrained(model_name)

VLMWithLoRA.register_model_loader("CLIP", load_clip_model)
VLMWithLoRA.register_model_loader("ViT", load_vit_model)