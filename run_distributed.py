import os
import sys
import time
import torch
import subprocess
import psutil
import json
import signal
from datetime import datetime

# 获取当前脚本的目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_gpu_info():
    """获取 GPU 信息"""
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_info = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            gpu_info.append({
                "index": i,
                "name": gpu_name.decode('utf-8') if isinstance(gpu_name, bytes) else gpu_name,
                "total_memory": info.total / (1024**3),  # GB
                "used_memory": info.used / (1024**3),    # GB
                "free_memory": info.free / (1024**3)     # GB
            })
        return gpu_info
    except (ImportError, Exception) as e:
        print(f"无法获取 GPU 信息: {e}")
        print("请安装 pynvml: pip install pynvml")
        return []

def get_process_info(process):
    """获取进程信息"""
    try:
        p = psutil.Process(process.pid)
        return {
            "pid": process.pid,
            "status": p.status(),
            "cpu_percent": p.cpu_percent(),
            "memory_percent": p.memory_percent(),
            "memory_info": {
                "rss": p.memory_info().rss / (1024**3),  # GB
                "vms": p.memory_info().vms / (1024**3)   # GB
            },
            "create_time": datetime.fromtimestamp(p.create_time()).strftime('%Y-%m-%d %H:%M:%S')
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return {"pid": process.pid, "status": "无法获取详细信息"}

def get_available_gpu_indices():
    """获取可用的 GPU 索引列表"""
    try:
        import pynvml
        pynvml.nvmlInit()
        available_gpus = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                # 尝试获取 GPU 信息，如果成功则认为 GPU 可用
                pynvml.nvmlDeviceGetMemoryInfo(handle)
                available_gpus.append(i)
            except pynvml.NVMLError:
                print(f"警告: GPU {i} 不可用或出现错误")
                continue
        return available_gpus
    except (ImportError, Exception) as e:
        print(f"无法获取 GPU 信息: {e}")
        return []

def main():
    # 获取可用的 GPU 索引列表
    available_gpus = get_available_gpu_indices()
    num_gpus = len(available_gpus)
    print(f"检测到 {num_gpus} 个可用 GPU，将启动 {num_gpus} 个进程")
    
    # 获取启动前的 GPU 信息
    print("启动前 GPU 状态:")
    gpu_info_before = get_gpu_info()
    for gpu in gpu_info_before:
        print(f"GPU {gpu['index']} ({gpu['name']}): 已用 {gpu['used_memory']:.2f}GB / 总计 {gpu['total_memory']:.2f}GB")
    
    if num_gpus <= 0:
        print("没有检测到可用的 GPU，将使用 CPU 运行")
        demo_path = os.path.join(SCRIPT_DIR, "demo.py")
        subprocess.run([sys.executable, demo_path])
        return {}
    
    # 设置环境变量
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["WORLD_SIZE"] = str(num_gpus)
    
    # 启动多个进程
    processes = []
    process_info = {}
    
    def signal_handler(sig, frame):
        print("接收到中断信号，正在终止所有进程...")
        for p in processes:
            if p.poll() is None:  # 如果进程还在运行
                p.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # 构建 demo.py 的完整路径
    demo_path = os.path.join(SCRIPT_DIR, "demo.py")
    print(f"将运行脚本: {demo_path}")
    
    if not os.path.exists(demo_path):
        print(f"错误: 找不到脚本 {demo_path}")
        return {}
    
    for i in range(num_gpus):
        env = os.environ.copy()
        env["RANK"] = str(i)
        env["LOCAL_RANK"] = str(available_gpus[i])
        
        cmd = [sys.executable, demo_path]
        process = subprocess.Popen(cmd, env=env)
        processes.append(process)
        
        # 等待进程启动
        time.sleep(1)
        
        # 收集进程信息
        process_info[str(i)] = {
            "rank": i,
            "pid": process.pid,
            "gpu_id": available_gpus[i],
            "cmd": " ".join(cmd),
            "details": get_process_info(process)
        }
        
        print(f"启动进程 {i}: PID={process.pid}, GPU={available_gpus[i]}")
    
    # 保存进程信息到文件
    info_path = os.path.join(SCRIPT_DIR, "process_info.json")
    with open(info_path, "w") as f:
        json.dump(process_info, f, indent=4)
    
    print(f"进程信息已保存到 {info_path}")
    
    try:
        # 监控进程状态
        while any(p.poll() is None for p in processes):
            time.sleep(5)
            
            # 更新进程信息
            for i, p in enumerate(processes):
                if p.poll() is None:  # 如果进程还在运行
                    process_info[str(i)]["details"] = get_process_info(p)
            
            # 更新 GPU 信息
            gpu_info_current = get_gpu_info()
            
            # 打印状态更新
            print("\n" + "="*50)
            print(f"状态更新 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
            for i, p in enumerate(processes):
                status = "运行中" if p.poll() is None else f"已结束 (返回码: {p.poll()})"
                print(f"进程 {i} (PID={p.pid}): {status}")
                
                if p.poll() is None and i < len(gpu_info_current):
                    gpu = gpu_info_current[i]
                    print(f"  GPU {gpu['index']} 使用: {gpu['used_memory']:.2f}GB / {gpu['total_memory']:.2f}GB")
            print("="*50 + "\n")
            
            # 更新进程信息文件
            with open(info_path, "w") as f:
                json.dump(process_info, f, indent=4)
    
    except KeyboardInterrupt:
        print("接收到中断信号，正在终止所有进程...")
        for p in processes:
            if p.poll() is None:
                p.terminate()
    
    # 等待所有进程完成
    for i, process in enumerate(processes):
        return_code = process.wait()
        print(f"进程 {i} (PID={process.pid}) 已结束，返回码: {return_code}")
    
    # 获取结束后的 GPU 信息
    print("\n启动后 GPU 状态:")
    gpu_info_after = get_gpu_info()
    for gpu in gpu_info_after:
        print(f"GPU {gpu['index']} ({gpu['name']}): 已用 {gpu['used_memory']:.2f}GB / 总计 {gpu['total_memory']:.2f}GB")
    
    # 返回最终的进程信息
    return process_info

if __name__ == "__main__":
    
    process_info = main()
    if process_info:
        print("\n进程执行摘要:")
        for rank, info in process_info.items():
            print(f"Rank {rank} (PID {info['pid']}): GPU {info['gpu_id']}") 