import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# 获取系统字体列表
import platform

def get_system_fonts():
    fonts = []
    if platform.system() == "Windows":
        font_dir = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
        for file in os.listdir(font_dir):
            if file.lower().endswith(('.ttf', '.otf')):
                fonts.append(os.path.splitext(file)[0])
    elif platform.system() == "Darwin":
        font_dir = "/System/Library/Fonts"
        for file in os.listdir(font_dir):
            if file.lower().endswith(('.ttf', '.otf')):
                fonts.append(os.path.splitext(file)[0])
    else:
        font_dir = "/usr/share/fonts"
        for root, dirs, files in os.walk(font_dir):
            for file in files:
                if file.lower().endswith(('.ttf', '.otf')):
                    fonts.append(os.path.splitext(file)[0])
    return sorted(list(set(fonts)))

# 颜色映射
COLOR_MAP = {
    "黑": (0, 0, 0),
    "白": (255, 255, 255),
    "灰": (128, 128, 128),
    "红": (255, 0, 0),
    "绿": (0, 255, 0),
    "蓝": (0, 0, 255),
}

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any = AnyType("*")

def normalize_input(input_data, debug_name="输入"):
    import torch
    import numpy as np
    from PIL import Image
    # PIL Image
    if isinstance(input_data, Image.Image):
        array = np.array(input_data).astype(np.float32) / 255.0
        if input_data.mode == "L":
            array = array[..., np.newaxis]
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
        return tensor.permute(0, 2, 3, 1)
    # list of PIL
    if isinstance(input_data, list) and all(isinstance(item, Image.Image) for item in input_data):
        tensors = []
        for img in input_data:
            array = np.array(img).astype(np.float32) / 255.0
            if img.mode == "L":
                array = array[..., np.newaxis]
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append(tensor.permute(1, 2, 0))
        return torch.stack(tensors, dim=0)
    # numpy
    if isinstance(input_data, np.ndarray):
        array = input_data.astype(np.float32) / 255.0 if input_data.max() > 1.0 else input_data.astype(np.float32)
        if array.ndim == 2:
            array = array[..., np.newaxis]
        elif array.ndim == 3:
            if array.shape[-1] in [1, 3, 4]:
                array = array[np.newaxis, ...]
            else:
                array = array[..., np.newaxis]
        elif array.ndim == 4:
            if array.shape[1] in [1, 3, 4]:
                array = array.transpose(0, 2, 3, 1)
        else:
            raise ValueError(f"不支持的 numpy 数组维度：{array.shape}")
        return torch.from_numpy(array)
    # torch.Tensor
    if 'torch' in sys.modules and isinstance(input_data, torch.Tensor):
        tensor = input_data.cpu().detach().float()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1).unsqueeze(0)
        elif tensor.ndim == 3:
            if tensor.shape[0] in [1, 3, 4]:
                tensor = tensor.permute(1, 2, 0).unsqueeze(0)
            else:
                tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 4:
            if tensor.shape[1] in [1, 3, 4]:
                tensor = tensor.permute(0, 2, 3, 1)
        elif tensor.ndim == 2 and tensor.shape[1] < 5:
            tensor = tensor.unsqueeze(0).unsqueeze(2)
        else:
            raise ValueError(f"不支持的张量维度：{tensor.shape}")
        return torch.clamp(tensor, 0, 1)
    # bytes
    if isinstance(input_data, (bytes, bytearray)):
        from PIL import Image
        from io import BytesIO
        img = Image.open(BytesIO(input_data)).convert("RGBA")
        return normalize_input(img, debug_name)
    # to_pil
    if hasattr(input_data, "to_pil") and callable(input_data.to_pil):
        return normalize_input(input_data.to_pil(), debug_name)
    # fallback
    try:
        array = np.array(input_data, dtype=np.float32)
        return normalize_input(array, debug_name)
    except Exception as e:
        raise ValueError(f"无法处理 {debug_name} 的输入类型：{type(input_data)}，错误：{str(e)}")


class TextOverlayImage:
    @classmethod
    def INPUT_TYPES(cls):
        fonts = get_system_fonts()
        font_choices = fonts
        # 默认字体优先 NotoSansSC-VF
        default_font = "NotoSansSC-VF" if "NotoSansSC-VF" in fonts else (font_choices[0][0] if font_choices else "Arial")
        if not font_choices:
            font_choices = ["Arial"]
        return {
            "required": {
                "图像": ("IMAGE",),
                "文本": ("STRING",),
                "字体": (tuple(font_choices), {"default": default_font}),
                "字体大小": ("INT", {"default": 128, "min": 1, "max": 512}),
                "字体颜色": (tuple(COLOR_MAP.keys()), {"default": "黑"}),
                "排版": (("横向", "纵向"), {"default": "横向"}),
                "X位置": ("INT", {"default": 384, "min": 0}),
                "Y位置": ("INT", {"default": 512, "min": 0}),
            },
            "optional": {
                "文字颜色": (any,),
            },
            "hidden": {},
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("图像", "文字遮罩")
    FUNCTION = "overlay"
    CATEGORY = "Muye/文本"

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return True

    # ...已移除 get_font_choices 和 INPUT_WIDGETS...

    def overlay(self, 图像, 文本, 文字颜色=None, 字体="Arial", 字体大小=32, 字体颜色="红", 排版="横向", X位置=384, Y位置=512):
        import torch
        # 1. 图像格式校正，兼容 torch.Tensor/np.ndarray
        img = 图像.clone().detach() if isinstance(图像, torch.Tensor) else torch.from_numpy(np.array(图像)).float() / 255.0
        if img.dim() == 3:
            img = img.unsqueeze(0)  # (1, H, W, C)

        # 2. 生成文字遮罩（单通道，白字黑底）
        from PIL import Image, ImageDraw, ImageFont
        import platform, os
        arr = (img[0].cpu().numpy() * 255).astype(np.uint8)
        h, w = arr.shape[:2]
        mask = Image.new("L", (w, h), 0)
        mask_draw = ImageDraw.Draw(mask)
        # 字体
        font_path = None
        font_dirs = []
        if platform.system() == "Windows":
            font_dirs = [os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')]
        elif platform.system() == "Darwin":
            font_dirs = ["/System/Library/Fonts"]
        else:
            font_dirs = ["/usr/share/fonts"]
        for d in font_dirs:
            for ext in [".ttf", ".otf"]:
                p = os.path.join(d, 字体+ext)
                if os.path.exists(p):
                    font_path = p
                    break
            if font_path:
                break
        if font_path:
            font = ImageFont.truetype(font_path, 字体大小)
        else:
            font = ImageFont.load_default()
        # 排版
        if 排版 == "纵向":
            文本 = "\n".join(list(文本))
        # 计算文字尺寸，实现居中定位
        text_bbox = mask_draw.textbbox((0, 0), 文本, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        draw_x = int(X位置 - text_w / 2)
        draw_y = int(Y位置 - text_h / 2)
        mask_draw.text((draw_x, draw_y), 文本, font=font, fill=255)
        mask_np = np.array(mask)

        # 3. 遮罩上色逻辑（与 mask_colorize.py 完全一致）
        color_map = {
            "黑": (0, 0, 0),
            "白": (255, 255, 255),
            "灰": (128, 128, 128),
            "红": (255, 0, 0),
            "绿": (0, 255, 0),
            "蓝": (0, 0, 255)
        }
        if 文字颜色 is not None:
            if isinstance(文字颜色, (list, tuple, np.ndarray)) and len(文字颜色) == 3:
                if max(文字颜色) <= 1.0:
                    color = tuple(int(c * 255) for c in 文字颜色)
                else:
                    color = tuple(int(c) for c in 文字颜色)
            else:
                color = color_map.get(字体颜色, (255, 0, 0))
        else:
            color = color_map.get(字体颜色, (255, 0, 0))

        mask_tensor = torch.from_numpy(mask_np).float().to(img.device)
        if mask_tensor.dim() == 3:
            mask_tensor = mask_tensor.max(dim=0).values
        mask_tensor = (mask_tensor > 0.5).float()
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

        color_arr = torch.tensor(color, dtype=img.dtype, device=img.device) / 255.0
        color_img = torch.ones_like(img) * color_arr
        out = img * (1 - mask_tensor) + color_img * mask_tensor
        out = out.clamp(0, 1)
        # 遮罩输出 shape: (1, H, W)，float32，0/1
        mask_out = mask_tensor[0, ..., 0].unsqueeze(0)  # (1, H, W)
        return (out, mask_out)

NODE_CLASS_MAPPINGS = {
    "TextOverlayImage": TextOverlayImage
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TextOverlayImage": "文字叠加图像"
}
