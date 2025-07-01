import torch
import numpy as np
from PIL import Image

class RemoveAlphaChannel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),  # 输入图像，支持单张或批量
                "背景颜色": (["黑", "白", "灰", "红", "绿", "蓝"], {
                    "default": "黑"
                }),  # 背景颜色选择，使用简洁中文选项
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("图像",)
    FUNCTION = "remove_alpha"
    CATEGORY = "Muye/图像"

    def remove_alpha(self, 图像: torch.Tensor, 背景颜色: str) -> tuple:
        # 英文到中文的颜色映射
        color_name_map = {
            "black": "黑",
            "white": "白",
            "gray": "灰",
            "red": "红",
            "green": "绿",
            "blue": "蓝"
        }
        
        # 如果传入的是英文值，转换为中文值
        背景颜色 = color_name_map.get(背景颜色, 背景颜色)

        # 定义背景颜色（RGB）
        color_map = {
            "黑": (0, 0, 0),
            "白": (255, 255, 255),
            "灰": (128, 128, 128),
            "红": (255, 0, 0),
            "绿": (0, 255, 0),
            "蓝": (0, 0, 255)
        }
        bg_color = color_map[背景颜色]

        # 确保输入是 batch 格式 (B, H, W, C)
        if 图像.dim() == 3:  # 单张图像 (H, W, C)
            图像 = 图像.unsqueeze(0)  # 转换为 (1, H, W, C)

        batch_size, height, width, channels = 图像.shape

        # 检查输入通道数
        if channels not in [3, 4]:
            raise ValueError(f"输入图像必须有 3 或 4 个通道，当前为 {channels} 个通道")

        # 创建输出张量列表
        output_images = []

        for i in range(batch_size):
            img = 图像[i]  # 提取单张图像 (H, W, C)
            img_np = img.cpu().numpy()  # 转换为 numpy 数组，值在 [0, 1]

            # 处理 Alpha 通道
            if channels == 4:
                # 分离 RGB 和 Alpha 通道
                rgb = img_np[:, :, :3]  # (H, W, 3)
                alpha = img_np[:, :, 3:4]  # (H, W, 1)

                # 创建背景图像（归一化到 [0, 1]）
                bg = np.ones_like(rgb) * np.array(bg_color) / 255.0

                # 根据 Alpha 通道混合 RGB 和背景
                alpha = alpha.repeat(3, axis=2)  # 扩展 Alpha 到 3 通道
                result = rgb * alpha + bg * (1 - alpha)
            else:
                # 如果没有 Alpha 通道，直接使用 RGB
                result = img_np[:, :, :3]

            # 转换为张量
            result_tensor = torch.from_numpy(result).float().to(图像.device)
            output_images.append(result_tensor)

        # 堆叠输出图像
        output = torch.stack(output_images, dim=0)  # (B, H, W, 3)

        return (output,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "RemoveAlphaChannel": RemoveAlphaChannel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemoveAlphaChannel": "移除透明通道"
}