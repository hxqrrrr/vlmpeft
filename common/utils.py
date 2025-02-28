import yaml
import torch
import os

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def save_checkpoint(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }, path)

def collate_fn(batch, task_type):
    images = [item["images"] for item in batch]
    texts = [item["text"] for item in batch]
    
    if task_type == "vqa":
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return {"images": images, "text": texts, "labels": labels}
    elif task_type in ["contrastive", "captioning"]:
        return {"images": images, "text": texts}
    else:
        raise ValueError(f"Unsupported task type: {task_type}")