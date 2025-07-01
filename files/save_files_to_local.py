import os
import torch
import numpy as np
from PIL import Image
import shutil



class SaveFilesToLocal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图片文件名": ("STRING", {"default": ""}),
                "图片后缀": ("STRING", {"default": ".png"}),
                "图片保存路径": ("STRING", {"default": "请输入"}),
                "文本文件名": ("STRING", {"default": ""}),
                "文本后缀": ("STRING", {"default": ".txt"}),
                "文本保存路径": ("STRING", {"default": "请输入"}),
                "附加": (["追加", "覆盖"], {"default": "追加"}),
            },
            "optional": {
                "图片": ("IMAGE",),
                "文本": ("STRING", {"forceInput": True, "is_list": True}),  
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("图片名称", "文本名称")
    OUTPUT_IS_LIST = (True, True)
    OUTPUT_NODE = True
    FUNCTION = "save_files"
    CATEGORY = "Muye/文件"

    def save_files(self, 图片文件名, 图片后缀, 图片保存路径,
                   文本文件名, 文本后缀, 文本保存路径,
                   附加, 图片=None, 文本=None, 视频=None, 音频=None):
        # 初始化输出名称列表
        image_names = []
        text_names = []

        # 确保保存路径存在
        def ensure_directory(path):
            if path and path != "请输入" and not os.path.exists(path):
                os.makedirs(path)

        # 处理文件名重复（追加模式）
        def get_unique_filename(base_name, extension, directory, mode):
            filename = f"{base_name}{extension}"
            full_path = os.path.join(directory, filename)

            if mode == "覆盖":
                return full_path, filename

            counter = 1
            new_base_name = base_name
            while os.path.exists(full_path):
                new_base_name = f"{base_name}_{counter:02d}"
                filename = f"{new_base_name}{extension}"
                full_path = os.path.join(directory, filename)
                counter += 1

            return full_path, filename

        # 解析非文本输入（文件名等）
        def parse_input(input_data, default_prefix):
            if not input_data:
                return []

            if isinstance(input_data, str):
                items = [item.strip() for item in input_data.split(",") if item.strip()]
                return items if items else [f"{default_prefix}_0"]
            if isinstance(input_data, list):
                items = []
                for item in input_data:
                    if isinstance(item, str):
                        items.extend([sub_item.strip() for sub_item in item.split(",") if sub_item.strip()])
                    elif isinstance(item, list):
                        items.extend(parse_input(item, default_prefix))
                    elif item:
                        items.append(str(item))
                return items if items else [f"{default_prefix}_0"]
            return [str(input_data)] if input_data else [f"{default_prefix}_0"]

        # 循环文件名并添加序号
        def cycle_filenames(filename_list, target_length):
            if not filename_list:
                return [f"file_{i}" for i in range(target_length)]

            result = []
            for i in range(target_length):
                base_name = filename_list[i % len(filename_list)]
                if i >= len(filename_list):
                    base_name = f"{base_name}_{(i // len(filename_list)):02d}"
                result.append(base_name)
            return result

        # 1. 处理图片
        if 图片 is not None and 图片保存路径 != "请输入":
            ensure_directory(图片保存路径)
            if len(图片.shape) == 3:  # 单张图片 (H, W, C)
                images = [图片]
            else:  # 图片列表 (B, H, W, C)
                images = 图片

            image_filename_list = parse_input(图片文件名, "image")
            image_filename_list = cycle_filenames(image_filename_list, len(images))

            print(f"图片数量: {len(images)}, 文件名列表: {image_filename_list}")

            for idx, img in enumerate(images):
                base_name = image_filename_list[idx]
                img = img.cpu().numpy()
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                pil_image = Image.fromarray(img)
                extension = 图片后缀 if 图片后缀 else ".png"
                full_path, filename = get_unique_filename(base_name, extension, 图片保存路径, 附加)
                pil_image.save(full_path)
                image_names.append(filename)
                print(f"保存图片: {filename}")

        # 2. 处理文本（沿用 ComfyUI-Easy-Use 逻辑）
        if 文本 is not None and 文本保存路径 != "请输入":
            ensure_directory(文本保存路径)
            # 直接处理文本输入（列表或单字符串）
            texts = [文本] if isinstance(文本, str) else 文本
            texts = [str(txt).strip() for txt in texts if str(txt).strip()]
            
            if not texts:
                print("警告: 文本输入为空，跳过文本保存")
            else:
                # 解析文件名
                text_filename_list = parse_input(文本文件名, "text")
                text_filename_list = cycle_filenames(text_filename_list, len(texts))

                # 调试日志
                print(f"文本数量: {len(texts)}, 原始文本输入: {texts[:3]}... (总计 {len(texts)} 项)")
                print(f"文件名输入: {文本文件名}, 解析后文件名列表: {text_filename_list[:10]}... (总计 {len(text_filename_list)} 项)")

                for idx, txt in enumerate(texts):
                    base_name = text_filename_list[idx]
                    extension = 文本后缀 if 文本后缀 else ".txt"
                    full_path, filename = get_unique_filename(base_name, extension, 文本保存路径, 附加)
                    with open(full_path, "w", encoding="utf-8") as f:
                        f.write(txt)  # 直接写入完整文本
                    text_names.append(filename)
                    print(f"保存文本: {filename} (内容: {txt[:100]}...)")



        # （已移除视频和音频保存功能）

        return (image_names, text_names)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "SaveFilesToLocal": SaveFilesToLocal
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveFilesToLocal": "保存文件（列表）到本地"
}