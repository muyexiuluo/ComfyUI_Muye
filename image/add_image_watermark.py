"""
水印大师 - 添加成品水印图像节点
参考自 Comfyui-ergouzi-DGNJD (EGCPSYTJNode: 2🐕添加成品水印图像)
功能：将已有的水印图片叠加到底图上，支持缩放、定位、旋转、透明度、遮罩等
"""

import os
import sys
import numpy as np
import torch
from PIL import Image, ImageOps


MAX_RESOLUTION = 16384


def tensor2pil(image: torch.Tensor) -> Image.Image:
    """将 ComfyUI IMAGE 张量转为 PIL Image"""
    return Image.fromarray(np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


def pil2tensor(image: Image.Image) -> torch.Tensor:
    """将 PIL Image 转为 ComfyUI IMAGE 张量"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def common_upscale(samples, 缩放宽度, 缩放高度, upscale_method, crop):
    """通用图像缩放函数（参考 ComfyUI 原生实现）"""
    if crop == "center":
        old_w = samples.shape[3]
        old_h = samples.shape[2]
        old_aspect = old_w / old_h
        new_aspect = 缩放宽度 / 缩放高度
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_w - old_w * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_h - old_h * (old_aspect / new_aspect)) / 2)
        s = samples[:, :, y : old_h - y, x : old_w - x]
    else:
        s = samples

    if upscale_method == "bislerp":
        return bislerp(s, 缩放宽度, 缩放高度)
    elif upscale_method == "lanczos":
        return lanczos(s, 缩放宽度, 缩放高度)
    else:
        return torch.nn.functional.interpolate(
            s, size=(缩放高度, 缩放宽度), mode=upscale_method
        )


# ============================================================
# 核心节点：水印大师 - 添加成品水印图像
# ============================================================
class WatermarkMaster:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "原图": ("IMAGE",),
                "水印图片": ("IMAGE",),
                "缩放模式": (
                    ["不缩放", "保持比例铺满", "按照缩放倍数缩放", "按照输入宽高缩放"],
                ),
                "缩放方法": (["nearest-exact", "bilinear", "area"],),
                "缩放倍数": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.1},
                ),
                "缩放宽度": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64},
                ),
                "缩放高度": (
                    "INT",
                    {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 64},
                ),
                "初始位置": (
                    ["居中", "上", "下", "左", "右", "上 左", "上 右", "下 左", "下 右"],
                ),
                "横向位移": (
                    "INT",
                    {"default": 0, "min": -48000, "max": 48000, "step": 10},
                ),
                "竖向位移": (
                    "INT",
                    {"default": 0, "min": -48000, "max": 48000, "step": 10},
                ),
                "旋转角度": (
                    "INT",
                    {"default": 0, "min": -180, "max": 180, "step": 5},
                ),
                "水印透明度": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0,
                        "max": 100,
                        "step": 5,
                        "display": "slider",
                    },
                ),
            },
            "optional": {
                "水印遮罩": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_watermark"
    CATEGORY = "Muye/文本"

    def apply_watermark(
        self,
        原图,
        水印图片,
        缩放模式,
        缩放方法,
        缩放倍数,
        缩放宽度,
        缩放高度,
        横向位移,
        竖向位移,
        旋转角度,
        水印透明度,
        初始位置,
        水印遮罩=None,
    ):
        # ---- 缩放处理 ----
        if 缩放模式 != "不缩放":
            wm_size = 水印图片.size()
            wm_size = (wm_size[2], wm_size[1])  # (W, H)

            if 缩放模式 == "保持比例铺满":
                h_ratio = 原图.size()[1] / wm_size[1]
                w_ratio = 原图.size()[2] / wm_size[0]
                ratio = min(h_ratio, w_ratio)
                wm_size = tuple(round(d * ratio) for d in wm_size)
            elif 缩放模式 == "按照缩放倍数缩放":
                wm_size = tuple(int(d * 缩放倍数) for d in wm_size)
            elif 缩放模式 == "按照输入宽高缩放":
                wm_size = (缩放宽度, 缩放高度)

            samples = 水印图片.movedim(-1, 1)
            水印图片 = common_upscale(samples, wm_size[0], wm_size[1], 缩放方法, False)
            水印图片 = 水印图片.movedim(1, -1)

        # ---- 张量转 PIL ----
        wm_pil = tensor2pil(水印图片)
        wm_pil = wm_pil.convert("RGBA")
        wm_pil.putalpha(Image.new("L", wm_pil.size, 255))

        # ---- 遮罩处理 ----
        if 水印遮罩 is not None:
            mask_pil = tensor2pil(水印遮罩)
            mask_pil = mask_pil.resize(wm_pil.size)
            wm_pil.putalpha(ImageOps.invert(mask_pil))

        # ---- 旋转 ----
        if 旋转角度 != 0:
            wm_pil = wm_pil.rotate(旋转角度, expand=True)

        # ---- 透明度处理（0=完全可见，100=完全透明） ----
        r, g, b, a = wm_pil.split()
        a = a.point(lambda x: max(0, int(x * (1 - 水印透明度 / 100))))
        wm_pil.putalpha(a)

        # ---- 计算位置 ----
        orig_w = 原图.size()[2]
        orig_h = 原图.size()[1]
        wm_w, wm_h = wm_pil.size

        loc_x = None
        loc_y = None

        if 初始位置 == "居中":
            loc_x = int(横向位移 + (orig_w - wm_w) / 2)
            loc_y = int(竖向位移 + (orig_h - wm_h) / 2)
        elif 初始位置 == "上":
            loc_x = int(横向位移 + (orig_w - wm_w) / 2)
            loc_y = int(竖向位移)
        elif 初始位置 == "下":
            loc_x = int(横向位移 + (orig_w - wm_w) / 2)
            loc_y = int(竖向位移 + orig_h - wm_h)
        elif 初始位置 == "左":
            loc_y = int(竖向位移 + (orig_h - wm_h) / 2)
            loc_x = int(横向位移)
        elif 初始位置 == "右":
            loc_x = int(横向位移 + orig_w - wm_w)
            loc_y = int(竖向位移 + (orig_h - wm_h) / 2)
        elif 初始位置 == "上 左":
            loc_x = int(横向位移)
            loc_y = int(竖向位移)
        elif 初始位置 == "上 右":
            loc_x = int(orig_w - wm_w + 横向位移)
            loc_y = int(竖向位移)
        elif 初始位置 == "下 左":
            loc_x = int(横向位移)
            loc_y = int(orig_h - wm_h + 竖向位移)
        elif 初始位置 == "下 右":
            loc_x = int(横向位移 + orig_w - wm_w)
            loc_y = int(竖向位移 + orig_h - wm_h)

        location = (loc_x, loc_y) if (loc_x is not None and loc_y is not None) else (横向位移, 竖向位移)

        # ---- 逐帧合成（支持 batch） ----
        img_list = torch.unbind(原图, dim=0)
        result_list = []

        for tensor in img_list:
            img_pil = tensor2pil(tensor)
            if 水印遮罩 is None:
                img_pil.paste(wm_pil, location)
            else:
                img_pil.paste(wm_pil, location, wm_pil)

            result_list.append(pil2tensor(img_pil))

        # ---- 合并输出 ----
        output = torch.stack([t.squeeze() for t in result_list])

        return (output,)


NODE_CLASS_MAPPINGS = {
    "AddImageWatermark": WatermarkMaster,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AddImageWatermark": "添加图像水印",
}
