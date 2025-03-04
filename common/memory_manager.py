import torch
import gc
import logging
import psutil
import time
from threading import Thread

class MemoryManager:
    def __init__(self, monitoring_interval=5):
        self.monitoring_interval = monitoring_interval
        self.active_tasks = {}  # task_id -> 内存使用情况
        self.monitoring_thread = None
        self.stop_monitoring = False
    
    def register_task(self, task_id, initial_memory=0):
        """注册一个新任务"""
        self.active_tasks[task_id] = {
            "initial_memory": initial_memory,
            "peak_memory": initial_memory,
            "start_time": time.time()
        }
    
    def unregister_task(self, task_id):
        """注销一个任务并释放其内存"""
        if task_id in self.active_tasks:
            task_info = self.active_tasks.pop(task_id)
            duration = time.time() - task_info["start_time"]
            peak_memory = task_info["peak_memory"]
            
            logging.info(f"任务 {task_id} 完成: 持续时间 {duration:.2f}s, 峰值内存 {peak_memory/1e9:.2f}GB")
            
            # 强制垃圾回收
            gc.collect()
            torch.cuda.empty_cache()
            
            # 返回释放的内存估计
            return peak_memory
        return 0
    
    def update_task_memory(self, task_id, current_memory):
        """更新任务的内存使用情况"""
        if task_id in self.active_tasks:
            self.active_tasks[task_id]["peak_memory"] = max(
                self.active_tasks[task_id]["peak_memory"],
                current_memory
            )
    
    def start_monitoring(self):
        """开始内存监控线程"""
        if self.monitoring_thread is None:
            self.stop_monitoring = False
            self.monitoring_thread = Thread(target=self._monitor_memory)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """停止内存监控线程"""
        if self.monitoring_thread is not None:
            self.stop_monitoring = True
            self.monitoring_thread.join()
            self.monitoring_thread = None
    
    def _monitor_memory(self):
        """内存监控线程函数"""
        while not self.stop_monitoring:
            try:
                # 获取系统内存信息
                system_memory = psutil.virtual_memory()
                
                # 获取 GPU 内存信息
                gpu_memory = []
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    total = torch.cuda.get_device_properties(i).total_memory
                    gpu_memory.append({
                        "device": i,
                        "allocated": allocated,
                        "reserved": reserved,
                        "total": total,
                        "free": total - allocated
                    })
                
                # 记录内存使用情况
                logging.debug(f"系统内存: {system_memory.percent}% 已用")
                for gpu in gpu_memory:
                    logging.debug(f"GPU {gpu['device']}: "
                                f"{gpu['allocated']/1e9:.2f}GB/{gpu['total']/1e9:.2f}GB 已分配, "
                                f"{gpu['free']/1e9:.2f}GB 可用")
                
                # 检查是否需要释放内存
                for gpu in gpu_memory:
                    if gpu['free'] < 1e9:  # 如果可用内存小于 1GB
                        logging.warning(f"GPU {gpu['device']} 内存不足，强制垃圾回收")
                        gc.collect()
                        torch.cuda.empty_cache()
            
            except Exception as e:
                logging.error(f"内存监控错误: {str(e)}")
            
            # 等待下一次监控
            time.sleep(self.monitoring_interval)
    
    def get_memory_stats(self):
        """获取当前内存统计信息"""
        stats = {
            "system": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent
            },
            "gpu": []
        }
        
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i)
            reserved = torch.cuda.memory_reserved(i)
            total = torch.cuda.get_device_properties(i).total_memory
            stats["gpu"].append({
                "device": i,
                "allocated": allocated,
                "reserved": reserved,
                "total": total,
                "free": total - allocated,
                "percent": allocated / total * 100
            })
        
        return stats 