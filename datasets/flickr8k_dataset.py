from torch.utils.data import Dataset
from PIL import Image
import os

class Flickr8kDataset(Dataset):
    def __init__(self, data_path, task_type="contrastive"):
        self.data_path = data_path
        self.task_type = task_type
        self.image_dir = os.path.join(data_path, "Images")
        self.captions = self._load_captions(os.path.join(data_path, "captions.txt"))

    def _load_captions(self, captions_file):
        captions = {}
        try:
            with open(captions_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[1:]:  # 跳过第一行标题
                    if "," in line:
                        image_name, caption = line.strip().split(",", 1)
                        captions[image_name] = caption.strip()
                        # print(f"Loaded: {image_name}, {caption[:30]}...")  # 可选调试
                    else:
                        print(f"Skipping invalid line: {line.strip()}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Captions file not found at: {captions_file}")
        print(f"Loaded {len(captions)} captions.")
        return captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        image_name = list(self.captions.keys())[idx]
        caption = self.captions[image_name]
        image_path = os.path.join(self.image_dir, image_name)
        
        
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found at: {image_path}")
        
        if self.task_type == "vqa":
            question = f"Is there a {caption.split()[1]} in the image?"
            label = 1
            return {"images": image, "text": question, "label": label}
        elif self.task_type == "contrastive":
            return {"images": image, "text": caption}
        elif self.task_type == "captioning":
            return {"images": image, "text": caption}

    def __repr__(self):
        return f"Flickr8kDataset(data_path={self.data_path}, task_type={self.task_type})"