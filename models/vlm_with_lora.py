import torch
import torch.nn as nn
from transformers import CLIPModel, LlavaForConditionalGeneration, AutoTokenizer, CLIPProcessor
from peft import LoraConfig, get_peft_model
from typing import Dict, Callable
from models.task_head import TaskHead
class VLMWithLoRA(nn.Module):
    _model_loaders: Dict[str, Callable] = {}
    _supported_tasks: Dict[str, list] = {
        "CLIP": ["vqa", "contrastive"],
        "LLaVA": ["vqa", "captioning"]
    }

    def __init__(self, model_name: str, model_type: str, lora_rank: int, num_answers: int = None, task_type: str = "vqa"):
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type
        self.lora_rank = lora_rank
        self.task_type = task_type

        if task_type not in self._supported_tasks.get(model_type, []):
            raise ValueError(f"Task '{task_type}' is not supported by model type '{model_type}'. Supported tasks: {self._supported_tasks[model_type]}")

        self.model = self._load_model()
        
        # 配置任务头部
        if self.model_type == "CLIP":
            input_dim = self.model.config.projection_dim * 2 if task_type == "vqa" else self.model.config.projection_dim
            self.processor = CLIPProcessor.from_pretrained(model_name)
        elif self.model_type == "LLaVA":
            input_dim = self.model.config.hidden_size
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.image_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        
        output_dim = num_answers if task_type == "vqa" else (self.model.config.vocab_size if task_type == "captioning" else None)
        self.task_head = TaskHead(task_type, input_dim, output_dim)

    def _load_model(self):
        if self.model_type not in self._model_loaders:
            raise ValueError(f"Unsupported model type: {self.model_type}. Supported types: {list(self._model_loaders.keys())}")
        
        base_model = self._model_loaders[self.model_type](self.model_name)
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=32,
            target_modules=self._get_lora_target_modules(),
            lora_dropout=0.1
        )
        return get_peft_model(base_model, lora_config)

    def _get_lora_target_modules(self):
        if self.model_type == "CLIP":
            return ["q_proj", "v_proj"]
        elif self.model_type == "LLaVA":
            return ["q_proj", "v_proj", "query", "value"]
        else:
            raise NotImplementedError(f"LoRA target modules not defined for {self.model_type}")

    def _process_inputs(self, images=None, text=None, **inputs):
        if self.model_type == "CLIP":
            if images is not None and text is not None:
                processed = self.processor(images=images, text=text, return_tensors="pt", padding=True)
                return {
                    "pixel_values": processed["pixel_values"],
                    "input_ids": processed["input_ids"],
                    "attention_mask": processed["attention_mask"]
                }
            return inputs
        elif self.model_type == "LLaVA":
            processed = {}
            if images is not None:
                processed["pixel_values"] = self.image_processor(images=images, return_tensors="pt")["pixel_values"]
            if text is not None:
                tokenized = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
                processed["input_ids"] = tokenized["input_ids"]
                processed["attention_mask"] = tokenized["attention_mask"]
            return {**processed, **inputs}

    def forward(self, images=None, text=None, **inputs):
        inputs = self._process_inputs(images, text, **inputs)
        
        # 确保所有张量移动到模型设备
        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        if self.model_type == "CLIP":
            pixel_values = inputs["pixel_values"]
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            outputs = self.model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            return self.task_head(image_embeds=outputs.image_embeds, text_embeds=outputs.text_embeds)
        
        elif self.model_type == "LLaVA":
            pixel_values = inputs["pixel_values"]
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            outputs = self.model(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask)
            return self.task_head(last_hidden_state=outputs.last_hidden_state)

    def print_trainable_parameters(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable percentage: {trainable_params / total_params * 100:.2f}%")

    @classmethod
    def register_model_loader(cls, model_type: str, loader: Callable):
        cls._model_loaders[model_type] = loader

# 模型加载器
def load_clip_model(model_name: str):
    return CLIPModel.from_pretrained(model_name)

def load_llava_model(model_name: str):
    return LlavaForConditionalGeneration.from_pretrained(model_name)

VLMWithLoRA.register_model_loader("CLIP", load_clip_model)
VLMWithLoRA.register_model_loader("LLaVA", load_llava_model)