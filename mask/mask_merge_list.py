import torch
import torch.nn.functional as F

class MaskMergeList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "遮罩": ("MASK", {"defaultInput": True, "multiple": True}),
                "匹配尺寸最大": ("BOOLEAN", {"default": False}),
                "匹配尺寸最小": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("合并遮罩",)
    FUNCTION = "merge_masks"
    CATEGORY = "Muye/Mask"

    def merge_masks(self, 遮罩, 匹配尺寸最大, 匹配尺寸最小):
        # Default size for empty mask
        default_height, default_width = 512, 512
        output_height, output_width = default_height, default_width

        # Validate boolean inputs
        if 匹配尺寸最大 and 匹配尺寸最小:
            raise ValueError("不能同时启用'匹配尺寸最大'和'匹配尺寸最小'")

        # Handle empty input
        if 遮罩 is None or (isinstance(遮罩, list) and len(遮罩) == 0):
            print("[MaskMergeList] 无输入遮罩，返回默认空遮罩 (512x512)")
            return (torch.zeros((default_height, default_width), device="cuda" if torch.cuda.is_available() else "cpu"),)

        # Helper function to normalize mask to 2D
        def normalize_mask(mask):
            print(f"[MaskMergeList] 处理遮罩，形状: {mask.shape}")

            # Handle 4D tensors (batch format, e.g., [B, C, H, W])
            if mask.dim() == 4:
                mask = mask.flatten(0, 1).max(dim=0).values

            # Handle 3D tensors (e.g., [C, H, W])
            elif mask.dim() == 3:
                if mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                else:
                    mask = mask.max(dim=0).values

            # Handle 2D tensors (e.g., [H, W])
            elif mask.dim() == 2:
                pass

            else:
                raise ValueError(f"不支持的遮罩维度: {mask.shape}")

            # Ensure 2D tensor
            if mask.dim() != 2:
                raise ValueError(f"无法规范化遮罩为 2D，当前形状: {mask.shape}")

            return mask

        # Normalize all masks
        if not isinstance(遮罩, list):
            遮罩 = [遮罩]

        # Process single mask case
        if len(遮罩) == 1:
            normalized_mask = normalize_mask(遮罩[0])
            print(f"[MaskMergeList] 单遮罩输出形状: {normalized_mask.shape}")
            return (normalized_mask,)

        # Process multiple masks
        normalized_masks = [normalize_mask(mask) for mask in 遮罩]

        # Get dimensions of all masks
        dimensions = [(mask.shape[-2], mask.shape[-1]) for mask in normalized_masks]
        output_height, output_width = dimensions[0]

        # Check if dimensions are consistent
        if not all(dim == dimensions[0] for dim in dimensions):
            if 匹配尺寸最大:
                # Resize to largest dimensions
                output_height = max(h for h, _ in dimensions)
                output_width = max(w for _, w in dimensions)
                print(f"[MaskMergeList] 缩放所有遮罩到最大尺寸: {output_height}x{output_width}")
            elif 匹配尺寸最小:
                # Resize to smallest dimensions
                output_height = min(h for h, _ in dimensions)
                output_width = min(w for _, w in dimensions)
                print(f"[MaskMergeList] 缩放所有遮罩到最小尺寸: {output_height}x{output_width}")
            else:
                # Default: throw error if dimensions differ
                raise ValueError(f"所有遮罩尺寸必须一致。当前尺寸: {dimensions}")

        # Resize masks if necessary
        resized_masks = []
        for mask in normalized_masks:
            if (mask.shape[-2], mask.shape[-1]) != (output_height, output_width):
                # Use bilinear interpolation for resizing
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                mask = F.interpolate(mask, size=(output_height, output_width), mode="bilinear", align_corners=False)
                mask = mask.squeeze(0).squeeze(0)  # [H, W]
            resized_masks.append(mask)

        # Move all masks to the same device (CUDA if available, else CPU)
        device = resized_masks[0].device
        resized_masks = [mask.to(device) for mask in resized_masks]

        # Initialize merged mask with the first mask
        merged = resized_masks[0].clone()

        # Perform union (OR) by taking the maximum value across all masks
        for mask in resized_masks[1:]:
            merged = torch.maximum(merged, mask)

        # Ensure output is 2D
        if merged.dim() != 2:
            raise ValueError(f"合并后的遮罩维度异常: {merged.shape}")

        print(f"[MaskMergeList] 合并遮罩输出形状: {merged.shape}")
        return (merged,)

# Register the node
NODE_CLASS_MAPPINGS = {
    "MaskMergeList": MaskMergeList
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskMergeList": "遮罩合并（列表）"
}