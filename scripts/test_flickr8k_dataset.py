import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 添加 vlmpeft/ 到路径
from datasets.flickr8k_dataset import Flickr8kDataset

import os

class TestFlickr8kDataset(unittest.TestCase):
    def setUp(self):
        # 测试用数据集路径
        self.data_path = "C:/Users/hxq11/Desktop/vlmpeft/data/raw/"
        self.task_types = ["vqa", "contrastive", "captioning"]

    def test_dataset_loading(self):
        # 测试数据集加载
        for task_type in self.task_types:
            dataset = Flickr8kDataset(self.data_path, task_type=task_type)
            self.assertGreater(len(dataset), 0, f"Dataset should load captions for {task_type}")
            print(f"Loaded {len(dataset)} captions for {task_type}")

    def test_getitem(self):
        # 测试 __getitem__ 返回格式
        for task_type in self.task_types:
            dataset = Flickr8kDataset(self.data_path, task_type=task_type)
            item = dataset[0]
            self.assertIn("images", item, f"{task_type} should return 'images'")
            self.assertIn("text", item, f"{task_type} should return 'text'")
            self.assertIsInstance(item["images"], Image.Image, f"Images should be PIL Image for {task_type}")
            self.assertIsInstance(item["text"], str, f"Text should be string for {task_type}")
            if task_type == "vqa":
                self.assertIn("label", item, "VQA should return 'label'")
                self.assertIsInstance(item["label"], int, "Label should be integer")

    def test_image_exists(self):
        # 测试图像文件是否可访问
        dataset = Flickr8kDataset(self.data_path, task_type="contrastive")
        item = dataset[0]
        image_path = os.path.join(dataset.image_dir, list(dataset.captions.keys())[0])
        self.assertTrue(os.path.exists(image_path), f"Image file should exist at: {image_path}")

if __name__ == '__main__':
    unittest.main()