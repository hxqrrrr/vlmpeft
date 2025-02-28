import argparse
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../datasets'))
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets.flickr8k_dataset import Flickr8kDataset
from common.utils import load_config, save_checkpoint,collate_fn
from models.vlm_with_lora import VLMWithLoRA

if __name__ == '__main__':
    print(sys.path)
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
    
    parser = argparse.ArgumentParser(description="Fine-tune VLM with LoRA on Flickr8k")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    config["learning_rate"] = float(config["learning_rate"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Flickr8kDataset(config["data_path"], task_type=config["task_type"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, config["task_type"]),
        num_workers=0
    )

    model = VLMWithLoRA(
        model_name=config["model_name"],
        model_type=config["model_type"],
        lora_rank=config["lora_rank"],
        num_answers=config.get("num_answers"),
        task_type=config["task_type"]
    ).to(device)
    model.print_trainable_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    model.train()
    for epoch in range(config["num_epochs"]):
        total_loss = 0
        # 使用 tqdm 添加进度条
        with tqdm(train_loader, desc=f"Epoch [{epoch+1}/{config['num_epochs']}]", unit="batch") as pbar:
            for step, batch in enumerate(pbar):
                outputs = model(images=batch["images"], text=batch["text"])

                if config["task_type"] == "vqa":
                    labels = batch["labels"].to(device)
                    loss = torch.nn.CrossEntropyLoss()(outputs, labels)
                elif config["task_type"] == "contrastive":
                    labels = torch.arange(len(batch["images"])).to(device)
                    loss = (torch.nn.CrossEntropyLoss()(outputs, labels) + 
                            torch.nn.CrossEntropyLoss()(outputs.t(), labels)) / 2
                elif config["task_type"] == "captioning":
                    labels = model.tokenizer(batch["text"], return_tensors="pt", padding=True, truncation=True)["input_ids"].to(device)
                    loss = torch.nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)), labels.view(-1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                # 更新进度条显示当前损失
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                if (step + 1) % config["log_interval"] == 0:
                    print(f"Epoch [{epoch+1}/{config['num_epochs']}], Step [{step+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config['num_epochs']}] completed, Average Loss: {avg_loss:.4f}")
        
        checkpoint_path = os.path.join(config["checkpoint_dir"], f"checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch, checkpoint_path)

    print("Training completed!")