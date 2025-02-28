from torch.utils.data import Dataset
from PIL import Image
import json
import os
# vaq尚未实现


class VQADataset(Dataset):
    def __init__(self, data_path, processor, split="train"):
        # 加载VQA数据（假设JSON格式）
        json_path = os.path.join(data_path, f"vqa_v2_{split}.json")
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.processor = processor
        self.image_dir = os.path.join(data_path, "images")
        self.answer_map = self._build_answer_map()

    def _build_answer_map(self):
        # 构建答案到索引的映射
        answers = set(entry["answer"] for entry in self.data)
        return {ans: idx for idx, ans in enumerate(answers)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_path = os.path.join(self.image_dir, entry["image"])
        question = entry["question"]
        answer = self.answer_map[entry["answer"]]

        # 预处理图像和文本
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, text=question, return_tensors="pt", padding=True)
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),  # [C, H, W]
            "input_ids": inputs["input_ids"].squeeze(0),        # [seq_len]
            "attention_mask": inputs["attention_mask"].squeeze(0),  # [seq_len]
            "label": answer
        }