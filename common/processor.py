from transformers import CLIPProcessor


class VLMProcessor:
    def __init__(self, model_name, processor_type='CLIP'):
        if processor_type == 'CLIP':
            self.processor = CLIPProcessor.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported processor type: {processor_type}")

    def __call__(self, images, text, **kwargs):
        """
        处理图像和文本输入。

        Args:
            images: PIL图像或图像列表。
            text: 字符串或字符串列表。
            **kwargs: 其他参数，如 return_tensors, padding 等。

        Returns:
            dict: 处理后的输入张量。
        """
        return self.processor(images=images, text=text, **kwargs)

