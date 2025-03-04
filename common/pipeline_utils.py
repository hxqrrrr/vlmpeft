import torch
import torch.distributed as dist
import logging
from contextlib import nullcontext
from transformers import CLIPProcessor

def send_tensor(tensor, dst_rank, stream=None):
    """发送张量到目标 GPU"""
    if stream:
        with torch.cuda.stream(stream):
            # 首先发送张量形状
            shape_tensor = torch.tensor(tensor.shape, dtype=torch.long, device='cuda')
            dist.send(shape_tensor, dst=dst_rank)
            
            # 然后发送张量数据
            dist.send(tensor, dst=dst_rank)
    else:
        # 首先发送张量形状
        shape_tensor = torch.tensor(tensor.shape, dtype=torch.long, device='cuda')
        dist.send(shape_tensor, dst=dst_rank)
        
        # 然后发送张量数据
        dist.send(tensor, dst=dst_rank)
    
    logging.debug(f"发送张量到 GPU {dst_rank}, 形状: {tensor.shape}")

def receive_tensor(src_rank, stream=None):
    """从源 GPU 接收张量"""
    if stream:
        with torch.cuda.stream(stream):
            # 首先接收张量形状
            shape_tensor = torch.zeros(4, dtype=torch.long, device='cuda')  # 最多 4 维
            dist.recv(shape_tensor, src=src_rank)
            shape = tuple(shape_tensor.tolist())
            
            # 然后接收张量数据
            tensor = torch.zeros(shape, device='cuda')
            dist.recv(tensor, src=src_rank)
    else:
        # 首先接收张量形状
        shape_tensor = torch.zeros(4, dtype=torch.long, device='cuda')  # 最多 4 维
        dist.recv(shape_tensor, src=src_rank)
        shape = tuple(shape_tensor.tolist())
        
        # 然后接收张量数据
        tensor = torch.zeros(shape, device='cuda')
        dist.recv(tensor, src=src_rank)
    
    logging.debug(f"接收来自 GPU {src_rank} 的张量, 形状: {shape}")
    return tensor

def forward_pass_pipeline(rank, world_size, model_part, lora_adapter, inputs, recv_stream=None, send_stream=None):
    """执行流水线前向传播"""
    
    # 1. 第一个GPU（处理输入）
    if rank == 0:
        try:
            # 处理和验证输入
            processed_inputs = process_input_data(inputs, rank)
            
            # 在计算流中执行模型
            with torch.cuda.stream(torch.cuda.current_stream()):
                outputs = apply_model_with_lora(model_part, lora_adapter, processed_inputs)
            
            # 发送结果到下一个GPU
            if world_size > 1:
                with torch.cuda.stream(send_stream) if send_stream else nullcontext():
                    # 1. 发送张量形状信息
                    shape_info = torch.tensor(outputs.shape).to(rank)
                    dist.send(shape_info, dst=rank+1)
                    # 2. 发送实际数据
                    dist.send(outputs, dst=rank+1)
                    
                if send_stream:
                    send_stream.synchronize()
            
            return outputs
            
        except Exception as e:
            logging.error(f"Rank {rank} 前向传播错误: {str(e)}")
            raise
    
    # 2. 中间GPU
    elif rank < world_size - 1:
        try:
            # 接收数据
            with torch.cuda.stream(recv_stream) if recv_stream else nullcontext():
                # 1. 接收形状信息
                shape_info = torch.zeros(4, dtype=torch.long).to(rank)
                dist.recv(shape_info, src=rank-1)
                
                # 2. 创建正确大小的tensor并接收数据
                inputs = torch.zeros(shape_info.tolist()).to(rank)
                dist.recv(inputs, src=rank-1)
            
            if recv_stream:
                recv_stream.synchronize()
            
            # 执行计算
            with torch.cuda.stream(torch.cuda.current_stream()):
                outputs = apply_model_with_lora(model_part, lora_adapter, inputs)
            
            # 发送到下一个GPU
            with torch.cuda.stream(send_stream) if send_stream else nullcontext():
                # 发送形状和数据
                shape_info = torch.tensor(outputs.shape).to(rank)
                dist.send(shape_info, dst=rank+1)
                dist.send(outputs, dst=rank+1)
            
            if send_stream:
                send_stream.synchronize()
            
            return outputs
            
        except Exception as e:
            logging.error(f"Rank {rank} 前向传播错误: {str(e)}")
            raise
    
    # 3. 最后一个GPU
    else:
        try:
            # 接收数据
            with torch.cuda.stream(recv_stream) if recv_stream else nullcontext():
                # 接收形状和数据
                shape_info = torch.zeros(4, dtype=torch.long).to(rank)
                dist.recv(shape_info, src=rank-1)
                
                inputs = torch.zeros(shape_info.tolist()).to(rank)
                dist.recv(inputs, src=rank-1)
            
            if recv_stream:
                recv_stream.synchronize()
            
            # 执行最终计算
            with torch.cuda.stream(torch.cuda.current_stream()):
                outputs = apply_model_with_lora(model_part, lora_adapter, inputs)
            
            return outputs
            
        except Exception as e:
            logging.error(f"Rank {rank} 前向传播错误: {str(e)}")
            raise

def process_input_data(inputs, rank):
    """处理输入数据"""
    if isinstance(inputs, dict):
        if 'image' in inputs and 'text' in inputs:
            try:
                # 图像预处理
                images = preprocess_images(inputs['image'], rank)
                # 文本预处理
                texts = inputs['text']
                
                # 使用CLIP处理器
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
                processed = processor(
                    images=images, 
                    text=texts, 
                    return_tensors="pt", 
                    padding=True
                )
                
                return {
                    k: v.to(rank) for k, v in processed.items()
                }
                
            except Exception as e:
                logging.error(f"输入处理错误: {str(e)}")
                return create_dummy_inputs(rank)
    
    return inputs

def preprocess_images(images, rank):
    """图像预处理"""
    if isinstance(images, torch.Tensor):
        # 检查和规范化数据范围
        with torch.no_grad():
            min_val = images.min()
            max_val = images.max()
            
            if min_val < 0 or max_val > 1:
                images = (images - min_val) / (max_val - min_val + 1e-8)
            
            # 确保图像格式正确
            if images.dim() == 3:
                images = images.unsqueeze(0)
            
            return images.to(rank)
    return images

def create_dummy_inputs(rank):
    """创建虚拟输入"""
    return {
        "pixel_values": torch.rand(16, 3, 224, 224).to(rank),
        "input_ids": torch.randint(0, 1000, (16, 20)).to(rank),
        "attention_mask": torch.ones(16, 20).to(rank)
    }

def backward_pass_pipeline(rank, world_size, outputs, lora_adapter, target=None, 
                         recv_stream=None, send_stream=None):
    """执行流水线反向传播
    Args:
        rank (int): 当前GPU的rank
        world_size (int): 总GPU数量
        outputs: 前向传播的输出
        lora_adapter: LoRA适配器参数
        target: 目标数据（仅最后一个GPU需要）
    """
    try:
        # 1. 最后一个 GPU 开始反向传播
        if rank == world_size - 1:
            # 验证输入
            if target is None:
                raise ValueError("最后一个 GPU 需要目标数据")
            
            # 计算流中执行计算
            with torch.cuda.stream(torch.cuda.current_stream()):
                # 计算损失
                loss = compute_loss(outputs, target)
                
                # 开始反向传播
                loss.backward()
                
                # 获取输入梯度
                if hasattr(outputs, 'input_tensor'):
                    input_grad = outputs.input_tensor.grad
                else:
                    input_grad = outputs.grad
                
                if input_grad is None:
                    raise ValueError("未能获取梯度")
            
            # 发送梯度到前一个 GPU
            if rank > 0:
                with torch.cuda.stream(send_stream) if send_stream else nullcontext():
                    send_tensor(input_grad, rank - 1, send_stream)
                    if send_stream:
                        send_stream.synchronize()
            
            # 更新 LoRA 参数
            update_lora_parameters(lora_adapter)
            
            return loss.item()
        
        # 2. 第一个 GPU 完成反向传播
        elif rank == 0:
            # 在接收流中获取梯度
            with torch.cuda.stream(recv_stream) if recv_stream else nullcontext():
                grad = receive_tensor(rank + 1, recv_stream)
                if recv_stream:
                    recv_stream.synchronize()
            
            # 在计算流中执行反向传播
            with torch.cuda.stream(torch.cuda.current_stream()):
                # 确保输出需要梯度
                if not outputs.requires_grad:
                    outputs.requires_grad_(True)
                
                # 应用梯度
                outputs.backward(grad)
            
            # 更新 LoRA 参数
            update_lora_parameters(lora_adapter)
            
            return None
        
        # 3. 中间 GPU 传递梯度
        else:
            # 在接收流中获取梯度
            with torch.cuda.stream(recv_stream) if recv_stream else nullcontext():
                grad = receive_tensor(rank + 1, recv_stream)
                if recv_stream:
                    recv_stream.synchronize()
            
            # 在计算流中执行反向传播
            with torch.cuda.stream(torch.cuda.current_stream()):
                # 确保输出需要梯度
                if not outputs.requires_grad:
                    outputs.requires_grad_(True)
                
                # 应用梯度
                outputs.backward(grad)
                
                # 获取输入梯度
                if hasattr(outputs, 'input_tensor'):
                    input_grad = outputs.input_tensor.grad
                else:
                    input_grad = outputs.grad
                
                if input_grad is None:
                    raise ValueError("未能获取梯度")
            
            # 在发送流中传递梯度
            with torch.cuda.stream(send_stream) if send_stream else nullcontext():
                send_tensor(input_grad, rank - 1, send_stream)
                if send_stream:
                    send_stream.synchronize()
            
            # 更新 LoRA 参数
            update_lora_parameters(lora_adapter)
            
            return None
            
    except Exception as e:
        logging.error(f"Rank {rank} 反向传播错误: {str(e)}")
        raise
    
    finally:
        # 清理临时变量
        torch.cuda.empty_cache()

def apply_model_with_lora(model_part, lora_adapter, inputs):
    """
    应用模型和 LoRA 适配器
    """
    # 检查输入类型
    if isinstance(inputs, dict):
        # 如果是字典，直接传递给模型
        base_output = model_part(**inputs)
    elif isinstance(inputs, torch.Tensor):
        # 如果是张量，作为单一输入传递
        base_output = model_part(inputs)
    else:
        # 其他情况，尝试转换为张量
        try:
            inputs_tensor = torch.tensor(inputs).to(model_part.device)
            base_output = model_part(inputs_tensor)
        except Exception as e:
            logging.error(f"无法处理输入类型 {type(inputs)}: {e}")
            # 创建一个虚拟输出
            base_output = torch.randn(4096).to(model_part.device)
    
    # 应用 LoRA 适配器
    A, B = lora_adapter
    
    # 如果输出是张量，应用 LoRA
    if isinstance(base_output, torch.Tensor):
        lora_output = base_output + torch.matmul(torch.matmul(base_output, A), B)
    # 如果输出是字典，对每个张量应用 LoRA
    elif isinstance(base_output, dict):
        lora_output = {}
        for k, v in base_output.items():
            if isinstance(v, torch.Tensor) and v.dim() > 1:
                # 只对多维张量应用 LoRA
                lora_output[k] = v + torch.matmul(torch.matmul(v, A), B)
            else:
                lora_output[k] = v
    else:
        # 其他情况，返回原始输出
        lora_output = base_output
    
    return lora_output

def compute_loss(output, target=None):
    """计算损失"""
    if target is not None:
        # 如果有目标数据，计算 MSE 损失
        return torch.nn.functional.mse_loss(output, target)
    else:
        # 否则使用简单的平方和损失
        return torch.mean(output ** 2)

def update_lora_parameters(lora_adapter, learning_rate=0.001):
    """更新 LoRA 参数
    Args:
        lora_adapter: (A, B) LoRA矩阵对
        learning_rate: 学习率
    """
    A, B = lora_adapter
    
    try:
        with torch.no_grad():
            # 梯度裁剪
            if A.grad is not None:
                torch.nn.utils.clip_grad_norm_(A, max_norm=1.0)
                A -= learning_rate * A.grad
                A.grad.zero_()
            
            if B.grad is not None:
                torch.nn.utils.clip_grad_norm_(B, max_norm=1.0)
                B -= learning_rate * B.grad
                B.grad.zero_()
                
    except Exception as e:
        logging.error(f"更新 LoRA 参数错误: {str(e)}")
        raise 