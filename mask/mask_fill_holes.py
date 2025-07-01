import torch
import numpy as np
from scipy.ndimage import binary_fill_holes

class MaskFillHoles:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "遮罩": ("MASK", {"defaultInput": True}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("填充遮罩",)
    FUNCTION = "fill_holes"
    CATEGORY = "Muye/Mask"

    def fill_holes(self, 遮罩):
        # 支持 2D/3D/4D，统一转为 2D
        if isinstance(遮罩, torch.Tensor):
            mask = 遮罩
        else:
            mask = torch.tensor(遮罩)
        if mask.dim() == 4:
            mask = mask.flatten(0, 1).max(dim=0).values
        elif mask.dim() == 3:
            if mask.shape[0] == 1:
                mask = mask.squeeze(0)
            else:
                mask = mask.max(dim=0).values
        elif mask.dim() == 2:
            pass
        else:
            raise ValueError(f"不支持的遮罩维度: {mask.shape}")
        # 转为 numpy 二值
        mask_np = mask.detach().cpu().numpy() > 0.5
        # 填充空洞
        filled_np = binary_fill_holes(mask_np)
        # 转回 torch
        filled_mask = torch.from_numpy(filled_np.astype(np.float32)).to(mask.device)
        return (filled_mask,)

NODE_CLASS_MAPPINGS = {
    "MaskFillHoles": MaskFillHoles
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskFillHoles": "遮罩填充漏洞"
}
