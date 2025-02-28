import argparse
import torch
import os
import sys
from PIL import Image

# 添加项目根目录和 datasets 到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../datasets'))

from datasets.flickr8k_dataset import Flickr8kDataset
from common.utils import load_config
from models.vlm_with_lora import VLMWithLoRA

def infer_single(model, image_path, text, device):
    """
    对单张图像和文本进行推理。

    Args:
        model: 加载的 VLMWithLoRA 模型。
        image_path (str): 图像文件路径。
        text (str): 输入文本（问题或描述）。
        device: 计算设备（CPU 或 GPU）。

    Returns:
        输出结果（根据任务类型变化）。
    """
    model.eval()
    with torch.no_grad():
        image = Image.open(image_path).convert("RGB")
        outputs = model(images=image, text=text)
        
        if model.task_type == "vqa":
            pred_idx = torch.argmax(outputs, dim=-1).item()
            return f"Predicted answer index: {pred_idx}"
        elif model.task_type == "contrastive":
            similarity = outputs.item()
            return f"Similarity score: {similarity:.4f}"
        elif model.task_type == "captioning":
            caption_ids = torch.argmax(outputs, dim=-1)
            caption = model.tokenizer.decode(caption_ids[0], skip_special_tokens=True)
            return f"Generated caption: {caption}"

def infer_dataset(model, dataset, device, max_samples=5):
    """
    对数据集进行批量推理。

    Args:
        model: 加载的 VLMWithLoRA 模型。
        dataset: Flickr8kDataset 实例。
        device: 计算设备。
        max_samples (int): 推理的最大样本数。

    Returns:
        None（打印结果）。
    """
    model.eval()
    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            item = dataset[i]
            images = item["images"]  # 单张图像
            text = item["text"]
            outputs = model(images=images, text=text)
            
            if model.task_type == "vqa":
                pred_idx = torch.argmax(outputs, dim=-1).item()
                print(f"Sample {i+1}: Predicted answer index: {pred_idx}")
            elif model.task_type == "contrastive":
                similarity = outputs.item()
                print(f"Sample {i+1}: Similarity score: {similarity:.4f}")
            elif model.task_type == "captioning":
                caption_ids = torch.argmax(outputs, dim=-1)
                caption = model.tokenizer.decode(caption_ids[0], skip_special_tokens=True)
                print(f"Sample {i+1}: Generated caption: {caption}")

if __name__ == "__main__":
    # 设置代理（可选）
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Inference with VLMWithLoRA")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--image", type=str, help="Path to single image for inference")
    parser.add_argument("--text", type=str, help="Input text for single inference")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型并加载检查点
    model = VLMWithLoRA(
        model_name=config["model_name"],
        model_type=config["model_type"],
        lora_rank=config["lora_rank"],
        num_answers=config.get("num_answers"),
        task_type=config["task_type"]
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {args.checkpoint}")

    # 推理模式
    if args.image and args.text:
        # 单张图像推理
        result = infer_single(model, args.image, args.text, device)
        print(result)
    else:
        # 数据集批量推理
        dataset = Flickr8kDataset(config["data_path"], task_type=config["task_type"])
        infer_dataset(model, dataset, device, max_samples=5)