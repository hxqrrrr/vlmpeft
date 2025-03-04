import torch
import logging
import numpy as np
from collections import defaultdict

class AdvancedTaskScheduler:
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        
        # 获取每个 GPU 的信息
        self.gpu_info = []
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            free_memory = props.total_memory - torch.cuda.memory_allocated(i)
            
            self.gpu_info.append({
                "index": i,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_memory": props.total_memory,
                "free_memory": free_memory,
                "multi_processor_count": props.multi_processor_count,
                "tasks": []
            })
            
            logging.info(f"GPU {i} ({props.name}): "
                        f"计算能力 {props.major}.{props.minor}, "
                        f"内存 {free_memory/1e9:.2f}GB/{props.total_memory/1e9:.2f}GB, "
                        f"多处理器数量 {props.multi_processor_count}")
    
    def schedule(self, task_queue, model_parts, analyzer):
        """
        为每个 GPU 分配任务和执行阶段
        
        Args:
            task_queue: 任务队列
            model_parts: 每个 GPU 上的模型部分
            analyzer: 内存分析器
            
        Returns:
            每个 GPU 的任务和阶段列表
        """
        # 获取有效 GPU 的索引列表（只包含有模型部分的 GPU）
        valid_gpus = [i for i, part in enumerate(model_parts) if part is not None]
        num_valid_gpus = len(valid_gpus)
        
        logging.info(f"有效 GPU 数量: {num_valid_gpus}, 索引: {valid_gpus}")
        
        # 如果没有有效的 GPU，返回空计划
        if num_valid_gpus == 0:
            logging.warning("没有有效的 GPU，返回空计划")
            return [[] for _ in range(self.num_gpus)]
        
        # 分析 GPU 计算能力和负载
        gpu_scores = self._analyze_gpus()
        gpu_loads = self._analyze_loads(task_queue, analyzer)
        
        # 打印 GPU 计算能力和负载
        logging.info(f"GPU 计算能力得分: {gpu_scores}")
        logging.info(f"GPU 负载分布: {gpu_loads}")
        
        # 初始化每个 GPU 的任务列表
        gpu_tasks = [[] for _ in range(self.num_gpus)]
        
        # 检查是否是 CLIP 模型分割情况（2 个有效 GPU）
        if num_valid_gpus == 2 and valid_gpus == [0, 1]:
            logging.info("检测到 CLIP 模型分割情况，使用协同调度策略")
            
            # 对于 CLIP 模型，视觉编码器在 GPU 0，文本编码器在 GPU 1
            # 每个任务需要两个 GPU 协同工作
            for task in task_queue:
                # GPU 0 处理视觉部分的前向和反向传播
                gpu_tasks[0].append((task, "F_vision"))
                gpu_tasks[0].append((task, "B_vision"))
                
                # GPU 1 处理文本部分的前向和反向传播
                gpu_tasks[1].append((task, "F_text"))
                gpu_tasks[1].append((task, "B_text"))
                
                # 更新 GPU 负载
                memory_estimate = analyzer.estimate_memory(task) / 1e9  # 转换为 GB
                gpu_loads[0] += memory_estimate / 2  # 假设负载平均分配
                gpu_loads[1] += memory_estimate / 2
        else:
            # 使用轮询方式分配任务
            for i, task in enumerate(task_queue):
                # 选择下一个有效的 GPU（轮询）
                gpu_idx = valid_gpus[i % num_valid_gpus]
                
                # 分配前向传播和反向传播
                gpu_tasks[gpu_idx].append((task, "F"))  # 前向传播
                gpu_tasks[gpu_idx].append((task, "B"))  # 反向传播
                
                # 更新 GPU 负载
                gpu_loads[gpu_idx] += analyzer.estimate_memory(task) / 1e9  # 转换为 GB
        
        # 打印调度计划
        for gpu_id, tasks in enumerate(gpu_tasks):
            logging.info(f"GPU {gpu_id} 调度计划: {len(tasks)} 个阶段")
            for i, (task, stage) in enumerate(tasks):
                logging.info(f"  阶段 {i}: {stage} 阶段, 任务 ID {id(task)}")
        
        return gpu_tasks
    
    def _analyze_gpus(self):
        """分析 GPU 计算能力，返回每个 GPU 的得分"""
        gpu_scores = []
        total_score = 0
        
        for info in self.gpu_info:
            # 计算得分 = 计算能力 * 多处理器数量
            compute_capability = float(info["compute_capability"])
            score = compute_capability * info["multi_processor_count"]
            gpu_scores.append(score)
            total_score += score
        
        # 归一化得分
        if total_score > 0:
            gpu_scores = [score / total_score for score in gpu_scores]
        else:
            gpu_scores = [1.0 / len(self.gpu_info)] * len(self.gpu_info)
        
        return gpu_scores
    
    def _analyze_loads(self, task_queue, analyzer):
        """分析 GPU 当前负载，返回每个 GPU 的负载得分"""
        gpu_loads = [0.0] * self.num_gpus
        
        # 计算每个 GPU 上已有任务的负载
        for gpu_idx, info in enumerate(self.gpu_info):
            for task in info["tasks"]:
                memory_used = analyzer.estimate_memory(task) / 1e9  # 转换为 GB
                gpu_loads[gpu_idx] += memory_used
        
        return gpu_loads
    
    def _assign_tasks_to_gpus(self, task_queue, gpu_scores, model_parts, analyzer):
        """将任务分配给 GPU，考虑任务复杂度和 GPU 能力"""
        assignments = {}
        gpu_loads = [0.0] * self.num_gpus
        
        # 估计每个任务的复杂度
        task_complexities = {}
        for task in task_queue:
            # 复杂度 = 内存需求 * 数据集大小
            memory_needed = analyzer.estimate_memory(task)
            dataset_size = len(task.dataset.dataset) if hasattr(task.dataset, 'dataset') else 10
            complexity = memory_needed * dataset_size
            task_complexities[id(task)] = complexity
        
        # 归一化任务复杂度
        total_complexity = sum(task_complexities.values())
        for task_id in task_complexities:
            task_complexities[task_id] /= total_complexity
        
        # 按复杂度排序任务（从高到低）
        sorted_tasks = sorted(task_queue, key=lambda t: task_complexities[id(t)], reverse=True)
        
        # 分配任务
        for task in sorted_tasks:
            task_id = id(task)
            task_complexity = task_complexities[task_id]
            
            # 找到负载最小的 GPU
            min_load_idx = np.argmin(gpu_loads)
            
            # 检查该 GPU 是否有足够内存
            memory_needed = analyzer.estimate_memory(task)
            if self.gpu_info[min_load_idx]["free_memory"] >= memory_needed:
                # 分配任务
                assignments[task_id] = min_load_idx
                gpu_loads[min_load_idx] += task_complexity / gpu_scores[min_load_idx]
                self.gpu_info[min_load_idx]["free_memory"] -= memory_needed
            else:
                # 找到有足够内存的 GPU
                for gpu_idx in range(self.num_gpus):
                    if self.gpu_info[gpu_idx]["free_memory"] >= memory_needed:
                        assignments[task_id] = gpu_idx
                        gpu_loads[gpu_idx] += task_complexity / gpu_scores[gpu_idx]
                        self.gpu_info[gpu_idx]["free_memory"] -= memory_needed
                        break
                else:
                    # 如果没有 GPU 有足够内存，分配给负载最小的 GPU
                    logging.warning(f"没有 GPU 有足够内存运行任务 {task_id}，分配给负载最小的 GPU {min_load_idx}")
                    assignments[task_id] = min_load_idx
                    gpu_loads[min_load_idx] += task_complexity / gpu_scores[min_load_idx]
        
        logging.info(f"GPU 负载分布: {gpu_loads}")
        return assignments 