import torch
import numpy as np

class MaskColorize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE", {"defaultInput": True, "label": "输入图像"}),
                "遮罩": ("MASK", {"defaultInput": True, "label": "输入遮罩"}),
                "颜色选择": (["黑", "白", "灰", "红", "绿", "蓝"], {"default": "白", "label": "内置颜色"}),
            },
            "optional": {
                "颜色输入": ("COLOR", {"label": "颜色输入（可选）"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("上色图像",)
    FUNCTION = "colorize"
    CATEGORY = "Muye/Mask"

    def colorize(self, 图像: torch.Tensor, 遮罩: torch.Tensor, 颜色选择: str, 颜色输入=None):
        # 颜色映射
        color_map = {
            "黑": (0, 0, 0),
            "白": (255, 255, 255),
            "灰": (128, 128, 128),
            "红": (255, 0, 0),
            "绿": (0, 255, 0),
            "蓝": (0, 0, 255)
        }
        if 颜色输入 is not None:
            # 颜色输入格式为 [R, G, B]，范围 0-255 或 0-1
            if isinstance(颜色输入, (list, tuple, np.ndarray)) and len(颜色输入) == 3:
                if max(颜色输入) <= 1.0:
                    color = tuple(int(c * 255) for c in 颜色输入)
                else:
                    color = tuple(int(c) for c in 颜色输入)
            else:
                color = color_map.get(颜色选择, (255, 0, 0))
        else:
            color = color_map.get(颜色选择, (255, 0, 0))

        # 图像和遮罩格式校正
        img = 图像.clone().detach()
        if img.dim() == 3:
            img = img.unsqueeze(0)  # (1, H, W, C)
        mask = 遮罩
        if mask.dim() == 3:
            mask = mask.max(dim=0).values
        mask = mask.cpu().numpy()
        mask = (mask > 0.5).astype(np.float32)
        mask = torch.from_numpy(mask).to(img.device)
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)

        # 上色
        color_arr = torch.tensor(color, dtype=img.dtype, device=img.device) / 255.0
        color_img = torch.ones_like(img) * color_arr
        out = img * (1 - mask) + color_img * mask
        out = out.clamp(0, 1)
        return (out,)

NODE_CLASS_MAPPINGS = {
    "MaskColorize": MaskColorize
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskColorize": "遮罩区域上色"
}
