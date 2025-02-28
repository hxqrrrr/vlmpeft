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

# 移除 VLMProcessor 的导入
# from common.processor import VLMProcessor

from models.vlm_with_lora import VLMWithLoRA
from PIL import Image
from transformers import LlavaForConditionalGeneration
dataset = Flickr8kDataset("C:/Users/hxq11/Desktop/vlmpeft/data/raw/", task_type="contrastive")
print(dataset[0])  # 应返回 {"images": <PIL.Image>, "text": "caption"}