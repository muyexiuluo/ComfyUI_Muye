class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any = AnyType("*")
import torch
import comfy.model_management
import re


最大分辨率 = 8192

class MuyeSizeSelector:
    def __init__(self):
        self.设备 = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "分辨率预设": (
                    (
                        "3840x2160 - 16:9",
                        "2160x3840 - 9:16",
                        "2560x1440 - 16:9",
                        "1440x2560 - 9:16",
                        "2304x1280 - 16:9",
                        "1280x2304 - 9:16",
                        "1920x1080 - 16:9",
                        "1080x1920 - 9:16",
                        "1344x768 - 16:9",
                        "768x1344 - 9:16",
                        # 7:4 和 4:7
                        "2240x1280 - 7:4",
                        "1280x2240 - 4:7",
                        "1792x1024 - 7:4",
                        "1024x1792 - 4:7",
                        "1400x800 - 7:4",
                        "800x1400 - 4:7",
                        # 4:3 和 3:4
                        "2560x1920 - 4:3",
                        "1920x2560 - 3:4",
                        "2048x1536 - 4:3",
                        "1536x2048 - 3:4",
                        "1920x1440 - 4:3",
                        "1440x1920 - 3:4",
                        "1600x1200 - 4:3",
                        "1200x1600 - 3:4",
                        "1280x960 - 4:3",
                        "960x1280 - 3:4",
                        "1024x768 - 4:3",
                        "768x1024 - 3:4",
                        # 3:2 和 2:3
                        "2400x1600 - 3:2",
                        "1600x2400 - 2:3",
                        "1920x1280 - 3:2",
                        "1280x1920 - 2:3",
                        "1800x1200 - 3:2",
                        "1200x1800 - 2:3",
                        "1536x1024 - 3:2",
                        "1024x1536 - 2:3",
                        "1440x960 - 3:2",
                        "960x1440 - 2:3",
                        "1216x832 - 3:2",
                        "832x1216 - 2:3",
                        # 1:1
                        "2048x2048 - 1:1",
                        "1024x1024 - 1:1",
                        "768x768 - 1:1",
                        "512x512 - 1:1",
                    ),
                    {"default": "832x1216 - 2:3"}
                ),
                "批次大小": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "长边覆盖": ("INT", {"default": 0, "min": 0, "max": 最大分辨率, "step": 8}),
                "比例覆盖": ("STRING", {"default": "无"}),
                "帧数": ("INT", {"default": 0, "min": 0, "max": 1024}),
                "因数": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
            },
            "optional": {
                # 允许任意类型输入，内部自动转换
                "宽": (any,),
                "高": (any,),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT", "INT")
    RETURN_NAMES = ("潜在", "宽度", "高度", "帧数")
    FUNCTION = "execute"
    CATEGORY = "Muye"

    def parse_ratio(self, ratio_str: str) -> tuple:
        """解析比例字符串，支持多种分隔符（如1:1, 2 3 1, 2.39/1等）"""
        if ratio_str == "无":
            return None
        # 替换常见分隔符为冒号
        ratio_str = re.sub(r'[\s/\\]', ':', ratio_str)
        # 提取数字
        numbers = re.findall(r'\d*\.?\d+', ratio_str)
        if len(numbers) >= 2:
            width_ratio = float(numbers[0])
            height_ratio = float(numbers[1])
            return width_ratio, height_ratio
        return None

    def adjust_to_factor(self, value: int, factor: int) -> int:
        """将值调整为因数的倍数"""
        if factor <= 0:
            return value
        return ((value + factor - 1) // factor) * factor

    def execute(self, 宽 = 0, 高 = 0, 分辨率预设: str = "832x1216 - 2:3", 批次大小: int = 1, 长边覆盖: int = 0, 比例覆盖: str = "无", 帧数: int = 0, 因数: int = 0) -> tuple:
        # 兼容任意类型输入
        def parse_num(val):
            if isinstance(val, (int, float)):
                return val
            try:
                return float(val)
            except Exception:
                return 0

        宽数 = parse_num(宽)
        高数 = parse_num(高)

        if 宽数 > 0 and 高数 > 0:
            宽度 = min(self.adjust_to_factor(宽数, 因数), 最大分辨率)
            高度 = min(self.adjust_to_factor(高数, 因数), 最大分辨率)
        else:
            # 解析分辨率预设
            宽度, 高度 = 分辨率预设.split(" ")[0].split("x")
            宽度 = int(宽度)
            高度 = int(高度)

            # 检查比例覆盖
            ratio = self.parse_ratio(比例覆盖)
            if ratio:
                width_ratio, height_ratio = ratio
                if 长边覆盖 > 0:
                    # 根据宽高比决定长边
                    if width_ratio >= height_ratio:  # 例如 3:2，宽度是长边
                        宽度 = 长边覆盖
                        高度 = int(宽度 * height_ratio / width_ratio)
                    else:  # 例如 2:3，高度是长边
                        高度 = 长边覆盖
                        宽度 = int(高度 * width_ratio / height_ratio)
                else:
                    # 无长边覆盖时，使用预设尺寸按比例缩放
                    if 宽度 / 高度 > width_ratio / height_ratio:
                        宽度 = int(高度 * width_ratio / height_ratio)
                    else:
                        高度 = int(宽度 * height_ratio / width_ratio)
            else:
                # 无比例覆盖，使用预设尺寸
                宽度 = 长边覆盖 if 长边覆盖 > 0 else 宽度
                高度 = 长边覆盖 if 长边覆盖 > 0 else 高度

            # 确保宽度和高度不超过最大分辨率
            宽度 = min(宽度, 最大分辨率)
            高度 = min(高度, 最大分辨率)

            # 调整为因数的倍数
            宽度 = self.adjust_to_factor(宽度, 因数)
            高度 = self.adjust_to_factor(高度, 因数)

        # 根据帧数决定潜在张量形状
        if 帧数 > 0:
            # 视频模型：包含帧数维度
            潜在 = torch.zeros([批次大小, 帧数, 16, 高度 // 8, 宽度 // 8], device=self.设备)
        else:
            # 图像模型：不包含帧数维度
            潜在 = torch.zeros([批次大小, 16, 高度 // 8, 宽度 // 8], device=self.设备)
            帧数 = 0  # 确保输出帧数为 0

        # 调试日志：记录输出尺寸
        print(f"潜在 - 尺寸: {潜在.shape}, 宽度: {宽度}, 高度: {高度}, 帧数: {帧数}")

        return ({"samples": 潜在}, 宽度, 高度, 帧数)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "MuyeSizeSelector": MuyeSizeSelector
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuyeSizeSelector": "尺寸选择器"
}

## print("MuyeSizeSelector node defined successfully")