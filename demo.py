import torch
import torch.distributed as dist
from torch.cuda import Stream
from collections import deque
from torch.utils.data import DataLoader
import os
import argparse
import logging
import torch.nn as nn
import time
import yaml  # 添加 yaml 导入
import glob

# 导入我们的自定义模块

from common.pipeline_utils import forward_pass_pipeline, backward_pass_pipeline
from common.task_scheduler import AdvancedTaskScheduler
from common.memory_manager import MemoryManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 初始化分布式环境
def init_distributed():
    # 获取当前进程的 rank（从环境变量或命令行参数）
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args, _ = parser.parse_known_args()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    
    # 检测可用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    logging.info(f"检测到 {num_gpus} 个可用 GPU")
    
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    
    # 设置当前进程使用的 GPU
    torch.cuda.set_device(local_rank)
    
    # 获取分布式信息
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    logging.info(f"进程 {rank}/{world_size} 使用 GPU {local_rank}")
    
    return rank, world_size

# 加载 YAML 配置文件
def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 修改 generate_tasks_from_config 函数，使其处理单个任务配置
def generate_tasks_from_config(config_path):
    """从配置文件生成单个任务"""
    config = load_config(config_path)
    
    # 获取全局配置
    model_config = config.get('model', {})
    lora_config = config.get('lora', {})
    training_config = config.get('training', {})
    task_config = config.get('task', {})
    
    # 创建 LoRA 参数
    lora_params = initialize_lora({"rank": lora_config.get('rank', 16)})
    
    # 处理数据集
    dataset_configs = task_config.get('dataset', [])
    if not dataset_configs:
        logging.warning(f"配置文件 {config_path} 中没有找到数据集配置，将使用默认随机数据集")
        dataset = create_dummy_dataset("default")
    else:
        # 使用第一个数据集配置
        dataset_config = dataset_configs[0]
        dataset_name = dataset_config.get('name', '')
        dataset_path = dataset_config.get('path', '')
        
        if dataset_path and os.path.exists(dataset_path):
            # 加载实际数据集
            dataset = load_dataset(dataset_path, dataset_config.get('type', ''))
        else:
            # 创建随机数据集
            dataset = create_dummy_dataset(dataset_name)
    
    # 创建数据加载器
    batch_size = training_config.get('batch_size', 16)
    dataset_loader = DataLoader(dataset, batch_size=batch_size)
    
    # 创建任务
    task = LoRATask(lora_params, dataset_loader)
    
    # 添加任务元数据
    task.name = os.path.basename(config_path).split('.')[0]
    task.model_config = model_config
    task.lora_config = lora_config
    task.training_config = training_config
    task.task_config = task_config
    
    return task

# 添加一个新函数，用于从多个配置文件生成任务队列
def generate_tasks_from_configs(config_paths):
    """从多个配置文件生成任务队列"""
    task_queue = deque()
    
    for config_path in config_paths:
        if os.path.exists(config_path):
            task = generate_tasks_from_config(config_path)
            task_queue.append(task)
            logging.info(f"从配置文件 {config_path} 生成任务: {task.name}")
        else:
            logging.error(f"配置文件 {config_path} 不存在")
    
    return task_queue

def create_dummy_dataset(name, size=10):
    """创建随机数据集"""
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, name, size=10):
            self.name = name
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            # 创建符合 CLIP 模型输入要求的数据
            image = torch.rand(3, 224, 224)  # [0, 1] 范围内的随机图像
            text = f"这是数据集 {self.name} 中的示例文本 {idx}"
            return {"image": image, "text": text}
    
    return DummyDataset(name, size)

def load_dataset(path, dataset_type):
    """加载实际数据集"""
    if dataset_type == "image_text_pair":
        # 实现加载图像-文本对数据集的逻辑
        # 这里可以使用 torchvision.datasets 或自定义加载逻辑
        pass
    else:
        # 其他类型的数据集
        pass
    
    # 临时返回随机数据集
    return create_dummy_dataset(path.split('/')[-1])

# 主程序
def mLoRA_framework(config_paths=None, user_requests=None):
    """实现 mLoRA 流水线并行训练"""
    try:
        # 初始化分布式环境
        rank, num_gpus = init_distributed()
        logging.info(f"Rank {rank}: 初始化完成，GPU 数量: {num_gpus}")
        
        # 创建内存管理器
        memory_manager = MemoryManager()
        memory_manager.start_monitoring()
        
        # 检查模型一致性
        if config_paths and len(config_paths) > 1:
            is_consistent, conflict_info = check_model_consistency(config_paths)
            if not is_consistent:
                if rank == 0:  # 只在主进程打印错误
                    base = conflict_info["base"]
                    conflict = conflict_info["conflict"]
                    logging.error(f"检测到模型配置不一致:")
                    logging.error(f"  基准配置 ({base['path']}): {base['name']} ({base['type']})")
                    logging.error(f"  冲突配置 ({conflict['path']}): {conflict['name']} ({conflict['type']})")
                    logging.error(f"不允许使用不同的模型配置。请确保所有配置文件使用相同的模型。")
                
                # 所有进程都应该退出
                if dist.is_initialized():
                    dist.barrier()  # 确保所有进程同步
                    dist.destroy_process_group()
                return
        
        # 加载基础模型
        if config_paths and len(config_paths) > 0:
            # 从第一个配置文件加载模型配置
            config = load_config(config_paths[0])
            model_config = config.get('model', {})
            lora_config = config.get('lora', {})
            task_config = config.get('task', {})
            
            vlm_model = load_base_model(
                model_name=model_config.get('name', "openai/clip-vit-base-patch16"),
                model_type=model_config.get('type', "CLIP"),
                lora_rank=lora_config.get('rank', 16),
                task_type=task_config.get('type', "contrastive")
            )
        else:
            # 使用默认配置
            vlm_model = load_base_model()
        
        logging.info(f"Rank {rank}: 加载基础模型完成")
        
        # 使用 VLMWithLoRA 的 split_model 方法分割模型
        model_parts = vlm_model.split_model(num_gpus)
        my_model_part = model_parts[rank].cuda() if rank < len(model_parts) else None
        
        if my_model_part is not None:
            my_model_part.eval()  # 冻结基础模型
            logging.info(f"Rank {rank}: 获得模型部分 {rank}/{len(model_parts)}")
        else:
            logging.warning(f"Rank {rank}: 没有分配到模型部分，将保持空闲")
        
        # 生成任务队列
        if config_paths:
            # 从多个配置文件生成任务
            task_queue = generate_tasks_from_configs(config_paths)
            logging.info(f"Rank {rank}: 从 {len(config_paths)} 个配置文件生成了 {len(task_queue)} 个任务")
        else:
            logging.error(f"Rank {rank}: 没有提供配置文件或用户请求")
            return
        
        # 初始化调度器和分析器
        scheduler = AdvancedTaskScheduler(num_gpus)
        analyzer = MemoryAnalyzer()
        
        try:
            # 执行流水线训练
            pipeline_train(rank, num_gpus, my_model_part, task_queue, scheduler, analyzer, memory_manager)
        except TypeError as e:
            if "'bool' object is not callable" in str(e):
                logging.error(f"Rank {rank}: 调度器返回了错误的类型。请检查 AdvancedTaskScheduler.schedule 方法的返回值。")
            raise
        
        # 停止内存监控
        memory_manager.stop_monitoring()
        
        logging.info(f"Rank {rank}: 训练完成，清理资源")
        
    except Exception as e:
        logging.error(f"Rank {rank if 'rank' in locals() else '?'}: 发生错误: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
    
    finally:
        # 确保总是清理分布式环境
        if dist.is_initialized():
            dist.destroy_process_group()
            logging.info(f"Rank {rank if 'rank' in locals() else '?'}: 已销毁进程组")

def pipeline_train(rank, world_size, model_part, task_queue, scheduler, analyzer, memory_manager):
    """执行流水线训练"""
    # 创建一个模型部分列表，其中只有当前进程的模型部分是有效的
    model_parts = [None] * world_size
    model_parts[rank] = model_part
    
    # 获取调度计划
    pipeline_plan = scheduler.schedule(task_queue, model_parts, analyzer)
    logging.info(f"Rank {rank}: 调度计划: {pipeline_plan}")
    # 确保所有进程都有任务执行
    has_tasks = len(pipeline_plan[rank]) > 0 if rank < len(pipeline_plan) else False
    
    # 同步所有进程，确保它们同时开始
    logging.info(f"Rank {rank}: 等待所有进程准备就绪")
    dist.barrier()
    
    if not has_tasks:
        logging.warning(f"Rank {rank}: 没有分配任务，将退出")
        return
    
    logging.info(f"Rank {rank}: 所有进程已准备就绪，开始训练")
    
    # 创建 CUDA 流用于异步操作
    compute_stream = torch.cuda.Stream()
    recv_stream = torch.cuda.Stream()
    send_stream = torch.cuda.Stream()
    
    # 为每个任务创建 LoRA 适配器
    lora_adapters = {}
    
    # 执行流水线
    for epoch in range(3):  # 训练 3 个 epoch
        logging.info(f"Rank {rank}: 开始 Epoch {epoch+1}/3")
        
        for task_idx, (task, stage) in enumerate(pipeline_plan[rank]):
            task_id = id(task)
            
            # 如果是第一次处理这个任务，初始化 LoRA 适配器
            if task_id not in lora_adapters:
                lora_adapters[task_id] = task.lora_params
                logging.info(f"Rank {rank}: 初始化任务 {task_id} 的 LoRA 适配器")
            
            # 获取数据批次
            try:
                # 尝试获取数据批次
                batch_iter = iter(task.dataset)
                try:
                    batch = next(batch_iter)
                    logging.info(f"Rank {rank}: 获取批次数据，类型: {type(batch)}")
                    
                    # 处理批次数据
                    if isinstance(batch, dict):
                        inputs = batch
                    elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                        inputs = {"image": batch[0], "text": batch[1]}
                    else:
                        logging.warning(f"Rank {rank}: 未知批次格式: {type(batch)}")
                        inputs = {"dummy": torch.randn(16, 3, 224, 224).to(rank)}
                
                    logging.info(f"Rank {rank}: 处理批次数据，格式: {type(batch)}")
                    
                except StopIteration:
                    logging.warning(f"Rank {rank}: 数据集已耗尽，使用虚拟数据")
                    inputs = {"pixel_values": torch.randn(16, 3, 224, 224).to(rank)}
            
            except Exception as e:
                logging.error(f"Rank {rank}: 获取数据批次失败: {str(e)}")
                inputs = {"pixel_values": torch.randn(16, 3, 224, 224).to(rank)}
            
            # 将输入数据移动到当前设备
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(rank)
                    logging.info(f"Rank {rank}: 输入 '{k}' 形状: {v.shape}")
            
            # 根据阶段执行不同的操作
            with torch.cuda.stream(compute_stream):
                # 处理 CLIP 模型分割情况
                if stage in ["F_vision", "F_text", "B_vision", "B_text"]:
                    # 解析阶段
                    phase, part = stage.split('_')
                    is_forward = (phase == "F")
                    is_vision = (part == "vision")
                    
                    logging.info(f"Rank {rank}: 执行任务 {task_id} 的 {stage} 阶段")
                    
                    if is_forward:
                        # 前向传播
                        if is_vision and rank == 0:  # GPU 0 处理视觉部分
                            # 处理视觉输入
                            vision_inputs = {k: v for k, v in inputs.items() if k in ["image", "pixel_values"]}
                            outputs = forward_pass_pipeline(
                                rank, world_size, model_part, lora_adapters[task_id],
                                vision_inputs, recv_stream, send_stream
                            )
                            # 存储输出用于反向传播
                            task.vision_outputs = outputs
                            
                        elif not is_vision and rank == 1:  # GPU 1 处理文本部分
                            # 处理文本输入
                            text_inputs = {k: v for k, v in inputs.items() if k in ["text", "input_ids", "attention_mask"]}
                            outputs = forward_pass_pipeline(
                                rank, world_size, model_part, lora_adapters[task_id],
                                text_inputs, recv_stream, send_stream
                            )
                            # 存储输出用于反向传播
                            task.text_outputs = outputs
                    else:
                        # 反向传播
                        if is_vision and rank == 0:  # GPU 0 处理视觉部分
                            # 获取前向传播的输出
                            outputs = task.vision_outputs
                            backward_pass_pipeline(
                                rank, world_size, model_part, lora_adapters[task_id],
                                outputs, recv_stream, send_stream
                            )
                            
                        elif not is_vision and rank == 1:  # GPU 1 处理文本部分
                            # 获取前向传播的输出
                            outputs = task.text_outputs
                            backward_pass_pipeline(
                                rank, world_size, model_part, lora_adapters[task_id],
                                outputs, recv_stream, send_stream
                            )
                            
                            # 更新 LoRA 参数（只在 GPU 1 上更新）
                            from common.pipeline_utils import update_lora_parameters
                            update_lora_parameters(lora_adapters[task_id])
                            logging.info(f"Rank {rank}: 更新任务 {task_id} 的 LoRA 参数")
                            
                            # 保存 LoRA 参数
                            save_lora_parameters(lora_adapters[task_id], task_id)
                else:
                    # 处理常规情况
                    if stage == "F":  # 前向传播
                        logging.info(f"Rank {rank}: 执行任务 {task_id} 的前向传播")
                        
                        # 执行前向传播
                        outputs = forward_pass_pipeline(
                            rank, world_size, model_part, lora_adapters[task_id],
                            inputs, recv_stream, send_stream
                        )
                        
                        # 存储输出用于反向传播
                        task.outputs = outputs
                        
                    elif stage == "B":  # 反向传播
                        logging.info(f"Rank {rank}: 执行任务 {task_id} 的反向传播")
                        
                        # 获取前向传播的输出
                        outputs = task.outputs
                        
                        # 执行反向传播
                        backward_pass_pipeline(
                            rank, world_size, model_part, lora_adapters[task_id],
                            outputs, recv_stream, send_stream
                        )
                        
                        # 更新 LoRA 参数
                        if rank == 0:  # 只在第一个 GPU 上更新参数
                            from common.pipeline_utils import update_lora_parameters
                            update_lora_parameters(lora_adapters[task_id])
                            logging.info(f"Rank {rank}: 更新任务 {task_id} 的 LoRA 参数")
                            
                            # 保存 LoRA 参数
                            save_lora_parameters(lora_adapters[task_id], task_id)
            
            # 同步流
            compute_stream.synchronize()
            
            # 记录任务完成
            logging.info(f"Rank {rank}: 完成任务 {task_id} 的 {stage} 阶段")
            
            # 更新内存使用情况
            current_memory = torch.cuda.memory_allocated()
            memory_manager.update_task_memory(task_id, current_memory)
            
            # 模拟一些延迟，以便观察流水线效果
            time.sleep(0.1)
    
    # 同步所有进程，确保它们同时结束
    dist.barrier()
    logging.info(f"Rank {rank}: 流水线训练完成")

class LoRATask:
    def __init__(self, lora_params, dataset):
        self.lora_params = lora_params
        self.dataset = dataset
        self.outputs = None  # 存储前向传播的输出

class MemoryAnalyzer:
    def __init__(self):
        self.memory_model = {}  # 任务 -> 内存估计
    
    def estimate_memory(self, task):
        # 初始估计：基础模型 + LoRA + 激活值
        if task not in self.memory_model:
            base_size = 14e9 / 4  # 假设 3.5GB (模型被分成 4 部分)
            lora_size = 256e3  # 256KB
            act_size = 1e9  # 1GB 激活值
            return base_size + lora_size + act_size
        return self.memory_model[task]
    
    def update(self, task, metrics):
        # 更新内存估计
        actual_memory = metrics["memory_used"]
        self.memory_model[task] = actual_memory
        logging.info(f"任务 {id(task)}: 更新内存估计为 {actual_memory / 1e9:.2f} GB")

def initialize_lora(config):
    """初始化 LoRA 参数，确保正确设置梯度"""
    rank = config["rank"]
    # 创建参数并明确设置 requires_grad=True
    A = torch.nn.Parameter(torch.randn(4096, rank), requires_grad=True)
    B = torch.nn.Parameter(torch.randn(rank, 4096), requires_grad=True)
    
    # 移动到 GPU
    A = A.cuda()
    B = B.cuda()
    
    logging.info(f"初始化 LoRA 参数: A={A.shape}, B={B.shape}, requires_grad={A.requires_grad}")
    
    return (A, B)

def save_lora_parameters(lora_adapter, task_id):
    """保存 LoRA 参数"""
    A, B = lora_adapter
    
    # 创建保存目录
    os.makedirs("saved_loras", exist_ok=True)
    
    # 保存参数
    torch.save({
        "A": A.data,
        "B": B.data
    }, f"saved_loras/lora_{task_id}.pt")
    
    logging.info(f"保存任务 {task_id} 的 LoRA 参数")

def load_base_model(model_name=None, model_type=None, lora_rank=None, task_type=None):
    """
    加载基础模型，使用 VLMWithLoRA 类
    """
    from models.vlm_with_lora import VLMWithLoRA
    
    # 创建 VLMWithLoRA 实例
    model_name = model_name or "openai/clip-vit-base-patch16"
    model_type = model_type or "CLIP"  # 确保使用 "CLIP" 而不是 "CLIP_VISION"
    lora_rank = lora_rank or 16
    task_type = task_type or "contrastive"
    
    # 检查分布式环境状态
    if torch.distributed.is_initialized():
        rank_info = f"Rank {dist.get_rank()}: "
    else:
        rank_info = ""
    
    logging.info(f"{rank_info}加载 {model_type} 模型 ({model_name}) 与 LoRA (rank={lora_rank})")
    
    try:
        model = VLMWithLoRA(
            model_name=model_name,
            model_type=model_type,  # 使用原始的 "CLIP"
            lora_rank=lora_rank,
            num_answers=1000,
            task_type=task_type
        )
        
        # 打印可训练参数信息
        model.print_trainable_parameters()
        
        return model
        
    except Exception as e:
        logging.error(f"{rank_info}加载 VLMWithLoRA 失败: {str(e)}")
        raise  # 在开发阶段，让错误传播以便调试

def check_model_consistency(config_paths):
    """检查多个配置文件中的模型配置是否一致"""
    if not config_paths or len(config_paths) <= 1:
        return True, None
    
    # 加载第一个配置作为基准
    base_config = load_config(config_paths[0])
    base_model_config = base_config.get('model', {})
    base_model_name = base_model_config.get('name')
    base_model_type = base_model_config.get('type')
    
    # 检查其他配置
    for path in config_paths[1:]:
        config = load_config(path)
        model_config = config.get('model', {})
        model_name = model_config.get('name')
        model_type = model_config.get('type')
        
        # 检查模型名称和类型是否一致
        if model_name != base_model_name or model_type != base_model_type:
            return False, {
                "base": {"name": base_model_name, "type": base_model_type, "path": config_paths[0]},
                "conflict": {"name": model_name, "type": model_type, "path": path}
            }
    
    return True, None

# 用户请求示例
if __name__ == "__main__":
    import sys
    import glob
    
    config_queue = ["hxq/vlmpeft/configs/clip_contrastive.yaml",
                    "hxq/vlmpeft/configs/clip_contrastive.yaml"]
    
    # 使用硬编码的配置队列
    if all(os.path.exists(path) for path in config_queue):
        mLoRA_framework(config_paths=config_queue)
    else:
        # 使用默认配置
        default_config = "hxq/vlmpeft/configs/clip_contrastive.yaml"
        if os.path.exists(default_config):
            mLoRA_framework(config_paths=[default_config])
        else:
            logging.error("找不到有效的配置文件")