import torch
import numpy as np
import cv2

class 图像列表批次统一尺寸:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像列表": ("IMAGE", {"forceInput": True, "tooltip": "支持list/tuple/batch tensor"}),
                "以最大尺寸为准": ("BOOLEAN", {"default": True}),
                "以最小尺寸为准": ("BOOLEAN", {"default": False}),
                "自定义尺寸": ("BOOLEAN", {"default": False}),
                "自定义宽度": ("INT", {"default": 1024, "min": 1, "max": 8192}),
                "自定义高度": ("INT", {"default": 1024, "min": 1, "max": 8192}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("统一尺寸图像",)
    FUNCTION = "resize_batch"
    CATEGORY = "Muye/图像"

    def resize_batch(self, 图像列表, 以最大尺寸为准, 以最小尺寸为准, 自定义尺寸, 自定义宽度, 自定义高度):
        # LayerUtility节点真实逻辑：输入为list/tuple自动stack为(B,H,W,3)，输入为tensor直接返回
        if isinstance(图像列表, torch.Tensor):
            # 输入为tensor，直接处理每张，返回同shape tensor
            if 图像列表.ndim == 4:
                imgs = [img.cpu().numpy() for img in 图像列表]
            elif 图像列表.ndim == 3:
                imgs = [图像列表.cpu().numpy()]
            else:
                raise ValueError("输入图像格式不支持")
        elif isinstance(图像列表, (list, tuple)):
            imgs = []
            for img in 图像列表:
                if isinstance(img, torch.Tensor):
                    imgs.append(img.cpu().numpy())
                else:
                    imgs.append(np.array(img))
        else:
            imgs = [np.array(图像列表)]

        # 计算目标尺寸
        whs = [(im.shape[0], im.shape[1]) if im.ndim==3 else (im.shape[1], im.shape[2]) for im in imgs]
        if 自定义尺寸:
            target_w, target_h = int(自定义宽度), int(自定义高度)
        elif 以最大尺寸为准:
            target_h = max([h for h, w in whs])
            target_w = max([w for h, w in whs])
        elif 以最小尺寸为准:
            target_h = min([h for h, w in whs])
            target_w = min([w for h, w in whs])
        else:
            # 默认最大
            target_h = max([h for h, w in whs])
            target_w = max([w for h, w in whs])

        # 统一尺寸
        resized_imgs = []
        for im in imgs:
            arr = im.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.shape[-1] > 3:
                arr = arr[..., :3]
            h, w = arr.shape[:2]
            scale = min(target_w/w, target_h/h)
            new_w, new_h = int(w*scale), int(h*scale)
            arr_resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
            pad = np.zeros((target_h, target_w, 3), dtype=np.float32)
            pad[:new_h, :new_w, :] = arr_resized
            resized_imgs.append(torch.from_numpy(pad).contiguous())

        # 输出格式，严格参考LayerUtility节点，始终输出单个tensor
        imgs_out = [im.float() if isinstance(im, torch.Tensor) else torch.from_numpy(im).float() for im in resized_imgs]
        imgs_out = [im/255.0 if im.max()>1.0 else im for im in imgs_out]
        out_tensor = torch.stack(imgs_out, dim=0)  # (B,H,W,3)
        return (out_tensor,)

NODE_CLASS_MAPPINGS = {
    "图像列表（批次）统一尺寸": 图像列表批次统一尺寸,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "图像列表（批次）统一尺寸": "图像列表（批次）统一尺寸",
}
