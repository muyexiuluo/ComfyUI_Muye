import torch
import torch.nn.functional as F

class MaskConcatenate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "遮罩1": ("MASK", {"defaultInput": True}),
                "遮罩2": ("MASK", {"defaultInput": True}),
                "匹配尺寸": ("BOOLEAN", {"default": False}),
                "拼接方向": (["左右", "上下"], {"default": "左右"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("拼接遮罩",)
    FUNCTION = "concatenate_masks"
    CATEGORY = "Muye/Mask"

    def concatenate_masks(self, 遮罩1, 遮罩2, 匹配尺寸, 拼接方向):
        # 默认尺寸（用于异常情况）
        default_height, default_width = 512, 512

        # 规范化遮罩为 2D 张量
        def normalize_mask(mask):
            print(f"[MaskConcatenate] 处理遮罩，形状: {mask.shape}, 设备: {mask.device}")
            # 处理 4D 张量（[B, C, H, W]）
            if mask.dim() == 4:
                mask = mask.flatten(0, 1).max(dim=0).values
            # 处理 3D 张量（[C, H, W] 或 [B, H, W]）
            elif mask.dim() == 3:
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                else:
                    mask = mask.max(dim=0).values
            # 处理 2D 张量（[H, W]）
            elif mask.dim() == 2:
                pass
            else:
                raise ValueError(f"不支持的遮罩维度: {mask.shape}")
            # 确保输出为 2D 张量
            if mask.dim() != 2:
                raise ValueError(f"无法规范化遮罩为 2D，当前形状: {mask.shape}")
            return mask

        # 调整遮罩尺寸的函数
        def resize_mask(mask, target_height, target_width):
            if mask.shape != (target_height, target_width):
                print(f"[MaskConcatenate] 调整遮罩尺寸从 {mask.shape} 到 {target_height}x{target_width}, 设备: {mask.device}")
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                mask = F.interpolate(mask, size=(target_height, target_width), mode="bilinear", align_corners=False)
                mask = mask.squeeze(0).squeeze(0)  # [H, W]
            return mask

        # 规范化两个输入遮罩
        mask1 = normalize_mask(遮罩1)
        mask2 = normalize_mask(遮罩2)

        # 获取输入遮罩的设备（优先使用遮罩1的设备）
        device = mask1.device
        if mask2.device != device:
            print(f"[MaskConcatenate] 遮罩2设备不一致（{mask2.device}），移动到 {device}")
            mask2 = mask2.to(device)

        # 获取遮罩尺寸
        height1, width1 = mask1.shape
        height2, width2 = mask2.shape

        # 如果启用匹配尺寸，将小图拉伸到大图尺寸
        if 匹配尺寸:
            if height1 * width1 >= height2 * width2:
                # 遮罩1 是大图，调整遮罩2
                mask2 = resize_mask(mask2, height1, width1)
                height2, width2 = height1, width1
            else:
                # 遮罩2 是大图，调整遮罩1
                mask1 = resize_mask(mask1, height2, width2)
                height1, width1 = height2, width2
            print(f"[MaskConcatenate] 匹配尺寸后，遮罩1: {height1}x{width1}, 遮罩2: {height2}x{width2}, 设备: {device}")

        # 确定输出尺寸（与未匹配尺寸的拼接逻辑一致）
        output_height = height1 + height2 if 拼接方向 == "上下" else max(height1, height2)
        output_width = width1 + width2 if 拼接方向 == "左右" else max(width1, width2)
        print(f"[MaskConcatenate] 输出尺寸: {output_height}x{output_width}, 设备: {device}")

        # 执行拼接
        output = torch.zeros(output_height, output_width, device=device)
        if 拼接方向 == "左右":
            # 左右拼接（沿宽度轴）
            output[:height1, :width1] = mask1
            output[:height2, width1:width1 + width2] = mask2
        else:
            # 上下拼接（沿高度轴）
            output[:height1, :width1] = mask1
            output[height1:height1 + height2, :width2] = mask2

        # 确保输出为 2D 张量
        if output.dim() != 2:
            raise ValueError(f"拼接后的遮罩维度异常: {output.shape}")

        print(f"[MaskConcatenate] 拼接遮罩输出形状: {output.shape}, 设备: {output.device}")
        return (output,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "MaskConcatenate": MaskConcatenate
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskConcatenate": "遮罩拼接"
}