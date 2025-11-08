import torch
import numpy as np
from typing import Tuple


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x


class ImageInfoNode:
    @classmethod
    def INPUT_TYPES(s):
        # 将反转作为可点击的布尔参数（默认 False），而不是连线接口
        return {"required": {"image": ("IMAGE",), "反转": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("宽", "高", "比例")
    FUNCTION = "get_info"
    CATEGORY = "Muye/图像"

    def get_info(self, image, 反转: bool = False) -> Tuple[int, int, str]:
        """
        Reads upstream image, returns width(int), height(int), ratio(str like "9:16" or "1.5:1").
        """
        img = to_numpy(image)
        if img is None:
            # no image provided, return zeros/empty
            return (0, 0, "0:0")

        # Handle batch/channel/temp dims: aim to find H and W
        # Common shapes: (H,W,3), (1,H,W,3), (C,H,W) etc.
        if isinstance(img, np.ndarray):
            shape = img.shape
            if len(shape) == 4:
                # (N,H,W,3) or (N,C,H,W) - prefer (N,H,W,3)
                if shape[0] == 1 and shape[-1] == 3:
                    H, W = shape[1], shape[2]
                elif shape[0] == 1 and shape[1] in (1,3):
                    # (N,C,H,W)
                    H, W = shape[2], shape[3]
                else:
                    # fallback
                    H, W = shape[1], shape[2]
            elif len(shape) == 3:
                # (H,W,3) or (C,H,W)
                if shape[2] in (1,3):
                    H, W = shape[0], shape[1]
                else:
                    # assume (C,H,W)
                    H, W = shape[1], shape[2]
            elif len(shape) == 2:
                H, W = shape[0], shape[1]
            else:
                # unknown shape
                return (0, 0, "0:0")
        else:
            return (0, 0, "0:0")

        # ensure ints
        H = int(H)
        W = int(W)

        # 如果要求反转输出，则交换 W 和 H
        if 反转:
            W_out, H_out = H, W
        else:
            W_out, H_out = W, H

        # compute ratio: support common integer ratios AND decimal cinematic ratios
        # We'll output a simplified integer ratio when possible, otherwise format as decimal with 2 dp like "2.40:1".
        def compute_ratio(w, h):
            from math import gcd
            if h == 0 or w == 0:
                return "0:0"
            # try integer reduced ratio first
            g = gcd(w, h)
            a = w // g
            b = h // g
            # if reduced denominator is small (<=100) and the division yields reasonable ints, use a:b
            if b <= 100:
                return f"{a}:{b}"
            # otherwise fall back to floating ratio expressed as X:1 with 2 decimal places
            val = w / h
            return f"{val:.2f}:1"

        ratio = compute_ratio(W_out, H_out)

        return (int(W_out), int(H_out), ratio)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageInfo": ImageInfoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageInfo": "图像比例/分辨率读取",
}
