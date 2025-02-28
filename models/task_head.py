import torch
import torch.nn as nn

class TaskHead(nn.Module):
    def __init__(self, task_type: str, input_dim: int, output_dim: int = None):
        """
        初始化任务头部。

        Args:
            task_type (str): 任务类型，例如 "vqa", "contrastive", "captioning"。
            input_dim (int): 输入维度（模型输出的特征维度）。
            output_dim (int, optional): 输出维度（VQA 的答案数或词汇表大小）。
        """
        super().__init__()
        self.task_type = task_type
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 配置任务头部
        if task_type == "vqa":
            if output_dim is None:
                raise ValueError("VQA task requires output_dim (number of answers)")
            self.head = nn.Linear(input_dim, output_dim)
        elif task_type == "contrastive":
            self.head = None  # 无需头部，直接返回相似度
        elif task_type == "captioning":
            if output_dim is None:
                raise ValueError("Captioning task requires output_dim (vocab size)")
            self.head = nn.Linear(input_dim, output_dim)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def forward(self, image_embeds=None, text_embeds=None, last_hidden_state=None):
        """
        根据任务类型处理模型输出。

        Args:
            image_embeds (torch.Tensor): 图像嵌入（CLIP）。
            text_embeds (torch.Tensor): 文本嵌入（CLIP）。
            last_hidden_state (torch.Tensor): LLaVA 的最后一层隐藏状态。

        Returns:
            torch.Tensor: 任务输出。
        """
        if self.task_type == "vqa":
            if image_embeds is not None and text_embeds is not None:  # CLIP
                combined = torch.cat([image_embeds, text_embeds], dim=-1)
                return self.head(combined)
            elif last_hidden_state is not None:  # LLaVA
                return self.head(last_hidden_state[:, 0, :])  # CLS token
        elif self.task_type == "contrastive":
            if image_embeds is None or text_embeds is None:
                raise ValueError("Contrastive task requires both image and text embeds")
            return torch.matmul(image_embeds, text_embeds.t())
        elif self.task_type == "captioning":
            if last_hidden_state is None:
                raise ValueError("Captioning task requires last_hidden_state")
            return self.head(last_hidden_state)
        else:
            raise NotImplementedError(f"Task '{self.task_type}' not implemented")