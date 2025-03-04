import torch
import torch.nn as nn
from transformers import CLIPModel, LlavaForConditionalGeneration, AutoTokenizer, CLIPProcessor
from peft import LoraConfig, get_peft_model
from typing import Dict, Callable, List
from models.task_head import TaskHead
import logging

class VLMWithLoRA(nn.Module):
    _model_loaders: Dict[str, Callable] = {}
    _supported_tasks: Dict[str, list] = {
        "CLIP": ["vqa", "contrastive"],
        "LLaVA": ["vqa", "captioning"]
    }

    def __init__(self, model_name: str, model_type: str, lora_rank: int, 
                 num_answers: int = None, task_type: str = "vqa", 
                 encoder_part: str = None):  # 新增参数
        super().__init__()
        self.model_name = model_name
        self.model_type = model_type
        self.lora_rank = lora_rank
        self.task_type = task_type
        self.encoder_part = encoder_part  # 保存编码器部分信息

        if task_type not in self._supported_tasks.get(model_type, []):
            raise ValueError(f"Task '{task_type}' is not supported by model type '{model_type}'. "
                            f"Supported tasks: {self._supported_tasks[model_type]}")

        # 根据编码器部分加载相应的模型
        self.model = self._load_model()
        
        # 配置任务头部
        if self.model_type == "CLIP":
            if encoder_part == "vision":
                input_dim = self.model.config.vision_config.hidden_size
            elif encoder_part == "text":
                input_dim = self.model.config.text_config.hidden_size
            else:
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

    def split_model(self, num_gpus):
        """
        将模型分割为多个部分，以便在多个 GPU 上运行
        """
        logging.info(f"将 {self.model_type} 模型分割为 {num_gpus} 部分")
        
        # 对于 CLIP 模型，我们可以将其分为视觉编码器和文本编码器
        if self.model_type == "CLIP":
            # 如果 GPU 数量超过 2，我们只使用前 2 个 GPU
            if num_gpus > 2:
                logging.warning(f"GPU 数量 ({num_gpus}) 超过 2，将只使用 2 个 GPU 进行模型分割")
                num_gpus = 2
            
            # 从 PEFT 模型中提取基础模型
            logging.info("从 PEFT 模型中提取基础模型")
            base_model = self.model.base_model
            
            # 分割模型
            if num_gpus == 1:
                # 如果只有 1 个 GPU，返回整个模型
                return [self]
            elif num_gpus >= 2:
                # 如果有 2 个或更多 GPU，分割为视觉编码器和文本编码器
                logging.info("CLIP 模型分割为视觉编码器和文本编码器")
                
                # 创建视觉编码器部分
                vision_encoder = VLMWithLoRA(
                    model_name=self.model_name,
                    model_type="CLIP",  # 使用相同的模型类型
                    lora_rank=self.lora_rank,
                    task_type=self.task_type,
                    encoder_part="vision"  # 新增参数指定编码器部分
                )
                
                # 创建文本编码器部分
                text_encoder = VLMWithLoRA(
                    model_name=self.model_name,
                    model_type="CLIP",  # 使用相同的模型类型
                    lora_rank=self.lora_rank,
                    task_type=self.task_type,
                    encoder_part="text"  # 新增参数指定编码器部分
                )
                
                return [vision_encoder, text_encoder]
        elif self.model_type == "LLaVA":
            logging.warning(f"LLaVA 模型分割尚未实现，将返回完整模型的 {num_gpus} 个副本")
            return [self.model for _ in range(num_gpus)]
        else:
            logging.warning(f"未知模型类型 {self.model_type}，将返回完整模型的 {num_gpus} 个副本")
            return [self.model for _ in range(num_gpus)]
    
    def _split_clip_model(self, clip_model, num_gpus):
        """
        专门为 CLIP 模型设计的分割函数
        将视觉编码器和文本编码器分配到不同的 GPU
        """
        # 如果只有一个 GPU，返回完整模型
        if num_gpus == 1:
            return [clip_model]
        
        # 如果有两个 GPU，将视觉编码器和文本编码器分开
        if num_gpus == 2:
            class VisionEncoderPart(nn.Module):
                def __init__(self, clip_model):
                    super().__init__()
                    self.vision_model = clip_model.vision_model
                    self.visual_projection = clip_model.visual_projection
                    
                def forward(self, pixel_values=None, **kwargs):
                    vision_outputs = self.vision_model(pixel_values=pixel_values)
                    image_embeds = vision_outputs[1]
                    image_embeds = self.visual_projection(image_embeds)
                    return image_embeds
            
            class TextEncoderPart(nn.Module):
                def __init__(self, clip_model):
                    super().__init__()
                    self.text_model = clip_model.text_model
                    self.text_projection = clip_model.text_projection
                    
                def forward(self, input_ids=None, attention_mask=None, **kwargs):
                    text_outputs = self.text_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                    text_embeds = text_outputs[1]
                    text_embeds = self.text_projection(text_embeds)
                    return text_embeds
            
            vision_part = VisionEncoderPart(clip_model)
            text_part = TextEncoderPart(clip_model)
            
            logging.info("CLIP 模型分割为视觉编码器和文本编码器")
            return [vision_part, text_part]
        
        # 如果有更多 GPU，进一步分割编码器内部
        parts = []
        
        # 分割视觉编码器
        vision_layers = list(clip_model.vision_model.encoder.layers)
        vision_layers_per_gpu = max(1, len(vision_layers) // (num_gpus // 2))
        
        for i in range(0, len(vision_layers), vision_layers_per_gpu):
            end_idx = min(i + vision_layers_per_gpu, len(vision_layers))
            
            class VisionEncoderPartial(nn.Module):
                def __init__(self, clip_model, start_idx, end_idx, is_first=False, is_last=False):
                    super().__init__()
                    self.is_first = is_first
                    self.is_last = is_last
                    
                    if is_first:
                        self.embeddings = clip_model.vision_model.embeddings
                    
                    self.layers = nn.ModuleList(vision_layers[start_idx:end_idx])
                    
                    if is_last:
                        self.post_layernorm = clip_model.vision_model.post_layernorm
                        self.visual_projection = clip_model.visual_projection
                
                def forward(self, hidden_states=None, pixel_values=None, **kwargs):
                    if self.is_first and pixel_values is not None:
                        hidden_states = self.embeddings(pixel_values)
                    
                    for layer in self.layers:
                        hidden_states = layer(hidden_states)[0]
                    
                    if self.is_last:
                        pooled_output = self.post_layernorm(hidden_states[:, 0, :])
                        image_embeds = self.visual_projection(pooled_output)
                        return image_embeds
                    
                    return hidden_states
            
            is_first = (i == 0)
            is_last = (end_idx == len(vision_layers))
            
            part = VisionEncoderPartial(clip_model, i, end_idx, is_first, is_last)
            parts.append(part)
        
        # 分割文本编码器
        text_layers = list(clip_model.text_model.encoder.layers)
        text_layers_per_gpu = max(1, len(text_layers) // (num_gpus - len(parts)))
        
        for i in range(0, len(text_layers), text_layers_per_gpu):
            end_idx = min(i + text_layers_per_gpu, len(text_layers))
            
            class TextEncoderPartial(nn.Module):
                def __init__(self, clip_model, start_idx, end_idx, is_first=False, is_last=False):
                    super().__init__()
                    self.is_first = is_first
                    self.is_last = is_last
                    
                    if is_first:
                        self.embeddings = clip_model.text_model.embeddings
                    
                    self.layers = nn.ModuleList(text_layers[start_idx:end_idx])
                    
                    if is_last:
                        self.final_layer_norm = clip_model.text_model.final_layer_norm
                        self.text_projection = clip_model.text_projection
                
                def forward(self, hidden_states=None, input_ids=None, attention_mask=None, **kwargs):
                    if self.is_first and input_ids is not None:
                        hidden_states = self.embeddings(input_ids)
                    
                    for layer in self.layers:
                        hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
                    
                    if self.is_last:
                        pooled_output = self.final_layer_norm(hidden_states[:, 0, :])
                        text_embeds = self.text_projection(pooled_output)
                        return text_embeds
                    
                    return hidden_states
            
            is_first = (i == 0)
            is_last = (end_idx == len(text_layers))
            
            part = TextEncoderPartial(clip_model, i, end_idx, is_first, is_last)
            parts.append(part)
        
        # 如果部分数量不等于 GPU 数量，调整部分数量
        while len(parts) < num_gpus:
            # 复制最后一个部分
            parts.append(parts[-1])
        
        if len(parts) > num_gpus:
            parts = parts[:num_gpus]
        
        logging.info(f"CLIP 模型分割为 {len(parts)} 部分")
        for i, part in enumerate(parts):
            logging.info(f"部分 {i}: {type(part).__name__}")
        
        return parts

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