import unittest
import torch
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # 添加 vlmpeft/ 到路径
from models.vlm_with_lora import VLMWithLoRA
from PIL import Image

class TestVLMWithLoRA(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_configs = [
            {"model_name": "openai/clip-vit-base-patch16", "model_type": "CLIP", "lora_rank": 16, "task_type": "contrastive"},
            {"model_name": "liuhaotian/llava-v1.5-7b", "model_type": "LLaVA", "lora_rank": 16, "task_type": "vqa", "num_answers": 3129},
            {"model_name": "liuhaotian/llava-v1.5-7b", "model_type": "LLaVA", "lora_rank": 16, "task_type": "captioning"}
        ]
        self.image = Image.new("RGB", (224, 224), color="red")
        self.text = "Test input"

    def test_model_loading(self):
        for config in self.model_configs:
            model = VLMWithLoRA(**config).to(self.device)
            self.assertTrue(hasattr(model, "model"), f"Model should be loaded for {config['task_type']}")
            model.print_trainable_parameters()

    def test_forward(self):
        for config in self.model_configs:
            model = VLMWithLoRA(**config).to(self.device)
            outputs = model(images=self.image, text=self.text)
            if config["task_type"] == "vqa":
                self.assertEqual(outputs.shape, (1, config["num_answers"]), f"VQA output shape mismatch")
            elif config["task_type"] == "contrastive":
                self.assertEqual(outputs.shape, (1, 1), f"Contrastive output shape mismatch")
            elif config["task_type"] == "captioning":
                self.assertEqual(outputs.dim(), 3, f"Captioning output should be 3D (batch, seq, vocab)")

    def test_input_processing(self):
        model = VLMWithLoRA("openai/clip-vit-base-patch16", "CLIP", 16, task_type="contrastive").to(self.device)
        inputs = model._process_inputs(images=self.image, text=self.text)
        self.assertIn("pixel_values", inputs)
        self.assertIn("input_ids", inputs)
        self.assertIn("attention_mask", inputs)
        self.assertEqual(inputs["pixel_values"].device, self.device)

if __name__ == '__main__':
    unittest.main()