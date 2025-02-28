from torch.utils.data import Dataset
from PIL import Image
import os

class Flickr8kDataset(Dataset):
    def __init__(self, data_path, processor):
        self.data_path = data_path
        self.processor = processor
        self.image_dir = os.path.join(data_path, "images")
        self.captions = self._load_captions(os.path.join(data_path, "captions.txt"))

    def _load_captions(self, captions_file):
        captions = {}
        with open(captions_file, 'r') as f:
            for line in f:
                if "," in line:
                    image_name, caption = line.strip().split(",", 1)
                    captions[image_name] = caption.strip()
                else:
                    print(f"Skipping line due to incorrect format: {line.strip()}")
        print(f"Loaded {len(captions)} captions.")
        return captions

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.captions):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.captions)} items")
        
        image_name = list(self.captions.keys())[idx]
        caption = self.captions[image_name]
        image_path = os.path.join(self.image_dir, image_name)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        inputs = self.processor(images=image, text=caption, return_tensors="pt", padding=True)
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0)
        }

    def __repr__(self):
        return f"Flickr8kDataset(data_path={self.data_path}, processor={self.processor})"