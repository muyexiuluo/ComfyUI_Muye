import os
import torch
import numpy as np
from PIL import Image
import folder_paths


class MuyeLoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {
                    "image_upload": True,  # 支持图片拖拽
                    "tooltip": "选择或拖拽图片文件"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("图像", "遮罩", "文件名称")
    FUNCTION = "load_image"
    CATEGORY = "Muye/图像"

    def load_image(self, image):
        # 获取图片路径
        image_path = folder_paths.get_annotated_filepath(image)
        
        # 加载图片
        i = Image.open(image_path)
        if i is None:
            raise ValueError(f"无法加载图片: {image_path}")

        # 处理图片格式（参考官方 LoadImage）
        image = i.convert("RGB")  # 强制转换为 RGB，确保无 alpha 影响
        image = np.array(image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image)[None,]  # 形状 (1, H, W, 3)

        # 处理遮罩
        mask = None
        if "A" in i.getbands():
            # 提取 alpha 通道
            mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
            mask = 1. - mask  # 反转遮罩
            mask = torch.from_numpy(mask)
        else:
            # 无 alpha 通道，生成全零遮罩
            mask = torch.zeros((i.size[1], i.size[0]), dtype=torch.float32, device="cpu")

        # 确保遮罩格式
        mask = mask.to("cpu")
        mask = torch.clamp(mask, 0, 1)

        # 调试日志：记录输出尺寸
        print(f"图像 - 尺寸: {image_tensor.shape}")
        print(f"遮罩 - 尺寸: {mask.shape}, 设备: {mask.device}, 值范围: {mask.min().item()} - {mask.max().item()}")

        # 提取文件名
        file_name = os.path.basename(image_path)

        # 返回输出
        return (image_tensor,  # 图像（原始 RGB，无遮罩）
                mask,         # 遮罩
                file_name)    # 文件名称

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        mtime = os.path.getmtime(image_path)
        return mtime

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True

# 节点映射
NODE_CLASS_MAPPINGS = {
    "MuyeLoadImage": MuyeLoadImage
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuyeLoadImage": "加载图片（文件名）"
}