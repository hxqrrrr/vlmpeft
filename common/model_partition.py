# import torch
# import torch.nn as nn
# import logging
# from transformers import CLIPModel, CLIPProcessor

# def split_model(model, num_gpus):
#     """
#     专门为 CLIP 模型设计的分割函数
#     将视觉编码器和文本编码器分配到不同的 GPU
#     """
#     # 检查是否是 VLMWithLoRA 实例
#     if model.__class__.__name__ == 'VLMWithLoRA':
#         logging.info(f"检测到 VLMWithLoRA 实例，尝试提取内部模型")
#         if hasattr(model, 'model') and hasattr(model.model, 'base_model'):
#             model = model.model.base_model
#             logging.info(f"成功提取 CLIP 基础模型")
#         else:
#             logging.warning(f"无法提取内部模型，将尝试直接分割 VLMWithLoRA")
    
#     if not isinstance(model, CLIPModel):
#         logging.warning(f"输入模型不是 CLIPModel，而是 {type(model)}，尝试通用分割")
#         # 使用通用分割方法，而不是递归调用自己
#         return split_generic_model(model, num_gpus)
    
#     logging.info("分割 CLIP 模型")
    
#     # 如果只有一个 GPU，返回完整模型
#     if num_gpus == 1:
#         return [model]
    
#     # 如果有两个 GPU，将视觉编码器和文本编码器分开
#     if num_gpus == 2:
#         class VisionEncoderPart(nn.Module):
#             def __init__(self, clip_model):
#                 super().__init__()
#                 self.vision_model = clip_model.vision_model
#                 self.visual_projection = clip_model.visual_projection
                
#             def forward(self, pixel_values=None, **kwargs):
#                 vision_outputs = self.vision_model(pixel_values=pixel_values)
#                 image_embeds = vision_outputs[1]
#                 image_embeds = self.visual_projection(image_embeds)
#                 return image_embeds
        
#         class TextEncoderPart(nn.Module):
#             def __init__(self, clip_model):
#                 super().__init__()
#                 self.text_model = clip_model.text_model
#                 self.text_projection = clip_model.text_projection
                
#             def forward(self, input_ids=None, attention_mask=None, **kwargs):
#                 text_outputs = self.text_model(
#                     input_ids=input_ids,
#                     attention_mask=attention_mask
#                 )
#                 text_embeds = text_outputs[1]
#                 text_embeds = self.text_projection(text_embeds)
#                 return text_embeds
        
#         vision_part = VisionEncoderPart(model)
#         text_part = TextEncoderPart(model)
        
#         logging.info("CLIP 模型分割为视觉编码器和文本编码器")
#         return [vision_part, text_part]
    
#     # 如果有更多 GPU，进一步分割编码器内部
#     parts = []
    
#     # 分割视觉编码器
#     vision_layers = list(model.vision_model.encoder.layers)
#     vision_layers_per_gpu = max(1, len(vision_layers) // (num_gpus // 2))
    
#     for i in range(0, len(vision_layers), vision_layers_per_gpu):
#         end_idx = min(i + vision_layers_per_gpu, len(vision_layers))
        
#         class VisionEncoderPartial(nn.Module):
#             def __init__(self, clip_model, start_idx, end_idx, is_first=False, is_last=False):
#                 super().__init__()
#                 self.is_first = is_first
#                 self.is_last = is_last
                
#                 if is_first:
#                     self.embeddings = clip_model.vision_model.embeddings
                
#                 self.layers = nn.ModuleList(vision_layers[start_idx:end_idx])
                
#                 if is_last:
#                     self.post_layernorm = clip_model.vision_model.post_layernorm
#                     self.visual_projection = clip_model.visual_projection
            
#             def forward(self, hidden_states=None, pixel_values=None, **kwargs):
#                 if self.is_first and pixel_values is not None:
#                     hidden_states = self.embeddings(pixel_values)
                
#                 for layer in self.layers:
#                     hidden_states = layer(hidden_states)[0]
                
#                 if self.is_last:
#                     pooled_output = self.post_layernorm(hidden_states[:, 0, :])
#                     image_embeds = self.visual_projection(pooled_output)
#                     return image_embeds
                
#                 return hidden_states
        
#         is_first = (i == 0)
#         is_last = (end_idx == len(vision_layers))
        
#         part = VisionEncoderPartial(model, i, end_idx, is_first, is_last)
#         parts.append(part)
    
#     # 分割文本编码器
#     text_layers = list(model.text_model.encoder.layers)
#     text_layers_per_gpu = max(1, len(text_layers) // (num_gpus - len(parts)))
    
#     for i in range(0, len(text_layers), text_layers_per_gpu):
#         end_idx = min(i + text_layers_per_gpu, len(text_layers))
        
#         class TextEncoderPartial(nn.Module):
#             def __init__(self, clip_model, start_idx, end_idx, is_first=False, is_last=False):
#                 super().__init__()
#                 self.is_first = is_first
#                 self.is_last = is_last
                
#                 if is_first:
#                     self.embeddings = clip_model.text_model.embeddings
                
#                 self.layers = nn.ModuleList(text_layers[start_idx:end_idx])
                
#                 if is_last:
#                     self.final_layer_norm = clip_model.text_model.final_layer_norm
#                     self.text_projection = clip_model.text_projection
            
#             def forward(self, hidden_states=None, input_ids=None, attention_mask=None, **kwargs):
#                 if self.is_first and input_ids is not None:
#                     hidden_states = self.embeddings(input_ids)
                
#                 for layer in self.layers:
#                     hidden_states = layer(hidden_states, attention_mask=attention_mask)[0]
                
#                 if self.is_last:
#                     pooled_output = self.final_layer_norm(hidden_states[:, 0, :])
#                     text_embeds = self.text_projection(pooled_output)
#                     return text_embeds
                
#                 return hidden_states
        
#         is_first = (i == 0)
#         is_last = (end_idx == len(text_layers))
        
#         part = TextEncoderPartial(model, i, end_idx, is_first, is_last)
#         parts.append(part)
    
#     # 如果部分数量不等于 GPU 数量，调整部分数量
#     while len(parts) < num_gpus:
#         # 复制最后一个部分
#         parts.append(parts[-1])
    
#     if len(parts) > num_gpus:
#         parts = parts[:num_gpus]
    
#     logging.info(f"CLIP 模型分割为 {len(parts)} 部分")
#     for i, part in enumerate(parts):
#         logging.info(f"部分 {i}: {type(part).__name__}")
    
#     return parts

# def split_generic_model(model, num_parts):
#     """将通用模型分割成多个部分"""
#     if isinstance(model, nn.Sequential):
#         # 如果模型是顺序模型，直接按层分割
#         layers_per_part = max(1, len(model) // num_parts)
#         model_parts = []
        
#         for i in range(num_parts):
#             start_idx = i * layers_per_part
#             end_idx = min((i + 1) * layers_per_part, len(model))
#             if start_idx >= len(model):
#                 # 创建一个空的模型部分
#                 part = nn.Sequential()
#             else:
#                 part = nn.Sequential(*list(model.children())[start_idx:end_idx])
#             model_parts.append(part)
        
#         logging.info(f"顺序模型被分割为 {num_parts} 部分")
#         for i, part in enumerate(model_parts):
#             logging.info(f"部分 {i}: {len(part)} 层")
        
#         return model_parts
    
#     elif hasattr(model, 'encoder') and hasattr(model, 'decoder'):
#         # 对于编码器-解码器模型
#         encoder_layers = list(model.encoder.children())
#         decoder_layers = list(model.decoder.children())
        
#         total_layers = len(encoder_layers) + len(decoder_layers)
#         layers_per_part = max(1, total_layers // num_parts)
        
#         model_parts = []
#         for i in range(num_parts):
#             # 创建自定义模型部分
#             part = create_model_part(model, encoder_layers, decoder_layers, 
#                                     i, layers_per_part, total_layers)
#             model_parts.append(part)
        
#         return model_parts
    
#     else:
#         # 对于其他类型的模型，尝试按层分割
#         try:
#             all_modules = list(model.modules())
#             # 过滤掉容器模块
#             leaf_modules = [m for m in all_modules if len(list(m.children())) == 0]
            
#             modules_per_part = max(1, len(leaf_modules) // num_parts)
#             model_parts = []
            
#             for i in range(num_parts):
#                 start_idx = i * modules_per_part
#                 end_idx = min((i + 1) * modules_per_part, len(leaf_modules))
                
#                 # 创建一个包含特定模块的模型部分
#                 part = create_custom_model_part(leaf_modules[start_idx:end_idx])
#                 model_parts.append(part)
            
#             return model_parts
#         except Exception as e:
#             logging.error(f"无法分割模型: {str(e)}")
#             # 回退到复制整个模型
#             return [model for _ in range(num_parts)]

# def create_custom_model_part(modules):
#     """创建包含指定模块的自定义模型部分"""
#     class ModelPart(nn.Module):
#         def __init__(self, modules_list):
#             super().__init__()
#             self.modules_list = nn.ModuleList(modules_list)
        
#         def forward(self, x):
#             for module in self.modules_list:
#                 x = module(x)
#             return x
    
#     return ModelPart(modules)

# def create_model_part(model, encoder_layers, decoder_layers, part_idx, layers_per_part, total_layers):
#     """创建编码器-解码器模型的一部分"""
#     class EncoderDecoderPart(nn.Module):
#         def __init__(self, original_model, start_idx, end_idx, encoder_len):
#             super().__init__()
#             self.original_model = original_model
#             self.start_idx = start_idx
#             self.end_idx = end_idx
#             self.encoder_len = encoder_len
            
#             # 确定这部分是否包含编码器和/或解码器层
#             self.has_encoder = start_idx < encoder_len
#             self.has_decoder = end_idx > encoder_len
            
#             # 创建层列表
#             self.layers = nn.ModuleList()
#             for i in range(start_idx, end_idx):
#                 if i < encoder_len:
#                     self.layers.append(encoder_layers[i])
#                 else:
#                     self.layers.append(decoder_layers[i - encoder_len])
        
#         def forward(self, x):
#             for layer in self.layers:
#                 x = layer(x)
#             return x
    
#     start_idx = part_idx * layers_per_part
#     end_idx = min((part_idx + 1) * layers_per_part, total_layers)
    
#     return EncoderDecoderPart(model, start_idx, end_idx, len(encoder_layers))

# def load_clip_model():
#     """
#     加载 CLIP 模型
#     """
#     try:
#         from transformers import CLIPModel, CLIPProcessor
        
#         model_name = "openai/clip-vit-base-patch32"
#         model = CLIPModel.from_pretrained(model_name)
#         processor = CLIPProcessor.from_pretrained(model_name)
        
#         logging.info(f"成功加载 CLIP 模型: {model_name}")
#         return model, processor
#     except Exception as e:
#         logging.error(f"加载 CLIP 模型失败: {str(e)}")
        
#         # 创建一个简单的替代模型
#         class SimpleCLIPModel(nn.Module):
#             def __init__(self):
#                 super().__init__()
#                 self.vision_model = nn.Sequential(
#                     nn.Conv2d(3, 64, kernel_size=3, padding=1),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2),
#                     nn.Conv2d(64, 128, kernel_size=3, padding=1),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2),
#                     nn.Flatten(),
#                     nn.Linear(128 * 56 * 56, 512),
#                     nn.ReLU(),
#                     nn.Linear(512, 512)
#                 )
                
#                 self.text_model = nn.Sequential(
#                     nn.Embedding(10000, 128),
#                     nn.LSTM(128, 256, batch_first=True),
#                     nn.Linear(256, 512)
#                 )
                
#                 self.visual_projection = nn.Linear(512, 512)
#                 self.text_projection = nn.Linear(512, 512)
            
#             def forward(self, pixel_values=None, input_ids=None, **kwargs):
#                 if pixel_values is not None:
#                     vision_outputs = self.vision_model(pixel_values)
#                     image_embeds = self.visual_projection(vision_outputs)
#                 else:
#                     image_embeds = None
                
#                 if input_ids is not None:
#                     text_outputs = self.text_model(input_ids)
#                     text_embeds = self.text_projection(text_outputs)
#                 else:
#                     text_embeds = None
                
#                 return {"image_embeds": image_embeds, "text_embeds": text_embeds}
        
#         model = SimpleCLIPModel()
#         processor = None
        
#         logging.warning("使用简单替代模型代替 CLIP")
#         return model, processor