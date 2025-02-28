import argparse
import torch
import os
import sys

# 添加 datasets 文件夹到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../datasets'))

from torch.utils.data import DataLoader
from datasets.flickr8k_dataset import Flickr8kDataset

from common.utils import load_config, save_checkpoint
from torch.nn.utils.rnn import pad_sequence
from common.processor import VLMProcessor
from models.vlm_with_lora import VLMWithLoRA

# 自定义collate_fn处理变长序列
def collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=0)
    attention_mask = pad_sequence([item["attention_mask"] for item in batch], batch_first=True, padding_value=0)
    return pixel_values, input_ids, attention_mask

if __name__ == '__main__':
    print(sys.path)
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Fine-tune VLM with LoRA on Flickr8k")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    config["learning_rate"] = float(config["learning_rate"])  # 确保学习率是浮点数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化处理器和数据集
    processor = VLMProcessor(config["model_name"])
    train_dataset = Flickr8kDataset(config["data_path"], processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )

    # 初始化模型
    model = VLMWithLoRA(
        model_name=config["model_name"],
        model_type=config["model_type"],
        lora_rank=config["lora_rank"],
        num_answers=config["num_answers"]
    ).to(device)
    model.print_trainable_parameters()

    # 设置优化器和损失函数（对比损失）
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    # 训练循环
    model.train()
    for epoch in range(config["num_epochs"]):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            pixel_values, input_ids, attention_mask = [x.to(device) for x in batch]
            
            # 前向传播
            logits_per_image = model(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
            labels = torch.arange(len(pixel_values)).to(device)  # 对角线为正样本
            loss = (torch.nn.CrossEntropyLoss()(logits_per_image, labels) + 
            torch.nn.CrossEntropyLoss()(logits_per_image.t(), labels)) / 2
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if (step + 1) % config["log_interval"] == 0:
                print(f"Epoch [{epoch+1}/{config['num_epochs']}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] completed, Average Loss: {avg_loss:.4f}")
        
        # 保存检查点
        checkpoint_path = os.path.join(config["checkpoint_dir"], f"checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    print("Training completed!")