import torch
import numpy as np
from PIL import Image, ImageCms
import os
import sys
import random
from skimage.color import rgb2hsv, hsv2rgb

# 确保 ComfyUI_Muye 插件的路径在 sys.path 中
custom_nodes_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if custom_nodes_path not in sys.path:
    sys.path.append(custom_nodes_path)

# 输入标准化函数，转换为张量并规范化维度
def normalize_input(input_data, debug_name="输入"):
    """将任意输入转换为标准张量，格式为 (batch, height, width, channels)"""
    # 处理 PIL 图像
    if isinstance(input_data, Image.Image):
        array = np.array(input_data).astype(np.float32) / 255.0
        if input_data.mode == "L":
            array = array[..., np.newaxis]  # 灰度图 (height, width, 1)
        tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)  # (1, channels, height, width)
        return tensor.permute(0, 2, 3, 1)  # (1, height, width, channels)

    # 处理 PIL 图像列表
    if isinstance(input_data, list) and all(isinstance(item, Image.Image) for item in input_data):
        tensors = []
        for img in input_data:
            array = np.array(img).astype(np.float32) / 255.0
            if img.mode == "L":
                array = array[..., np.newaxis]
            tensor = torch.from_numpy(array).permute(2, 0, 1)  # (channels, height, width)
            tensors.append(tensor.permute(1, 2, 0))  # (height, width, channels)
        return torch.stack(tensors, dim=0)  # (batch, height, width, channels)

    # 处理 numpy 数组
    if isinstance(input_data, np.ndarray):
        array = input_data.astype(np.float32) / 255.0 if input_data.max() > 1.0 else input_data.astype(np.float32)
        if array.ndim == 2:  # (height, width)
            array = array[..., np.newaxis]  # (height, width, 1)
        elif array.ndim == 3:  # (height, width, channels) 或 (batch, height, width)
            if array.shape[-1] in [1, 3, 4]:  # (height, width, channels)
                array = array[np.newaxis, ...]  # (1, height, width, channels)
            else:  # (batch, height, width)
                array = array[..., np.newaxis]  # (batch, height, width, 1)
        elif array.ndim == 4:  # (batch, height, width, channels) 或 (batch, channels, height, width)
            if array.shape[1] in [1, 3, 4]:  # (batch, channels, height, width)
                array = array.permute(0, 2, 3, 1)
        else:
            raise ValueError(f"不支持的 numpy 数组维度：{array.shape}")
        return torch.from_numpy(array)

    # 处理 PyTorch 张量
    if isinstance(input_data, torch.Tensor):
        tensor = input_data.cpu().detach().float()
        if tensor.ndim == 2:  # (height, width)
            tensor = tensor.unsqueeze(-1).unsqueeze(0)  # (1, height, width, 1)
        elif tensor.ndim == 3:  # (channels, height, width) 或 (height, width, channels)
            if tensor.shape[0] in [1, 3, 4]:  # (channels, height, width)
                tensor = tensor.permute(1, 2, 0).unsqueeze(0)  # (1, height, width, channels)
            else:  # (height, width, channels)
                tensor = tensor.unsqueeze(0)  # (1, height, width, channels)
        elif tensor.ndim == 4:  # (batch, channels, height, width) 或 (batch, height, width, channels)
            if tensor.shape[1] in [1, 3, 4]:  # (batch, channels, height, width)
                tensor = tensor.permute(0, 2, 3, 1)  # (batch, height, width, channels)
        elif tensor.ndim == 2 and tensor.shape[1] < 5:  # 特殊情况，如 (2000, 4)
            tensor = tensor.unsqueeze(0).unsqueeze(2)  # (1, height, width, channels)
        else:
            raise ValueError(f"不支持的张量维度：{tensor.shape}")
        return torch.clamp(tensor, 0, 1)

    # 尝试将其他类型转换为张量
    try:
        array = np.array(input_data, dtype=np.float32)
        return normalize_input(array, debug_name)
    except Exception as e:
        raise ValueError(f"无法处理 {debug_name} 的输入类型：{type(input_data)}，错误：{str(e)}")

# 工具函数：张量转 PIL 图像
def tensor2pil(tensor):
    """将张量转换为 PIL 图像，假设输入为 (batch, height, width, channels)"""
    if tensor.ndim == 4:
        tensor = tensor[0]  # 取第一张
    tensor = tensor.cpu().numpy()
    tensor = np.clip(tensor * 255, 0, 255).astype(np.uint8)
    mode = 'RGBA' if tensor.shape[-1] == 4 else 'RGB' if tensor.shape[-1] == 3 else 'L'
    return Image.fromarray(tensor, mode=mode)

# 工具函数：PIL 图像转张量
def pil2tensor(image):
    """将 PIL 图像转换为张量，输出为 (batch, height, width, channels)"""
    array = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array)[np.newaxis, ...]  # (1, height, width, channels)
    return tensor

# 张量混合函数
def blend_tensors(tensor_a, tensor_b, mode, blend_factor=1.0, color_space='Lab', output_format='RGB'):
    """在张量级别实现图像混合，确保输出为 (batch, height, width, 3) 或 (batch, height, width, 4)"""
    # 确保张量格式为 (batch, height, width, channels)
    if tensor_a.shape[1:3] != tensor_b.shape[1:3]:  # 尺寸不一致
        tensor_b = torch.nn.functional.interpolate(tensor_b.permute(0, 3, 1, 2), size=tensor_a.shape[1:3], mode='bilinear', align_corners=False)
        tensor_b = tensor_b.permute(0, 2, 3, 1)

    # 保存前景的 alpha 通道（如果存在）
    alpha_a = None
    if tensor_a.shape[-1] == 4:
        alpha_a = tensor_a[:, :, :, 3:4]  # (batch, height, width, 1)
    else:
        alpha_a = torch.ones(tensor_a.shape[:-1] + (1,), dtype=tensor_a.dtype, device=tensor_a.device)  # 全不透明

    # 保存背景的 alpha 通道（如果存在）
    alpha_b = None
    if tensor_b.shape[-1] == 4:
        alpha_b = tensor_b[:, :, :, 3:4]  # (batch, height, width, 1)
    else:
        alpha_b = torch.ones(tensor_b.shape[:-1] + (1,), dtype=tensor_b.dtype, device=tensor_b.device)  # 全不透明

    # 提取 RGB 通道用于混合
    tensor_a_rgb = tensor_a[:, :, :, :3] if tensor_a.shape[-1] in [3, 4] else tensor_a.repeat(1, 1, 1, 3 // tensor_a.shape[-1])
    tensor_b_rgb = tensor_b[:, :, :, :3] if tensor_b.shape[-1] in [3, 4] else tensor_b.repeat(1, 1, 1, 3 // tensor_b.shape[-1])

    # 初始化输出
    batch_size = max(tensor_a.shape[0], tensor_b.shape[0])
    output_tensors = []

    for i in range(batch_size):
        img_a_rgb = tensor_a_rgb[i:i+1] if i < tensor_a.shape[0] else tensor_a_rgb[-1:]
        img_b_rgb = tensor_b_rgb[i:i+1] if i < tensor_b.shape[0] else tensor_b_rgb[-1:]
        alpha_a_i = alpha_a[i:i+1] if i < tensor_a.shape[0] else alpha_a[-1:]
        alpha_b_i = alpha_b[i:i+1] if i < tensor_b.shape[0] else alpha_b[-1:]

        if mode in ["色相", "饱和度", "颜色", "明度"] and color_space == 'Lab':
            try:
                # 转换为 PIL 图像
                pil_a = tensor2pil(img_a_rgb)
                pil_b = tensor2pil(img_b_rgb)
                # 确保 RGB
                pil_a = pil_a.convert('RGB')
                pil_b = pil_b.convert('RGB')
                # 转换为 Lab
                lab_a = ImageCms.profileToProfile(pil_a, ImageCms.createProfile('sRGB'), ImageCms.createProfile('LAB'), outputMode='LAB')
                lab_b = ImageCms.profileToProfile(pil_b, ImageCms.createProfile('sRGB'), ImageCms.createProfile('LAB'), outputMode='LAB')
                # 分离 Lab 通道
                l_a, a_a, b_a = lab_a.split()
                l_b, a_b, b_b = lab_b.split()
                # 混合
                if mode == "色相":
                    img_out = Image.merge('LAB', (l_a, a_b, b_b))  # 保留前景明度，使用背景色相和饱和度
                elif mode == "饱和度":
                    img_out = Image.merge('LAB', (l_a, a_a, b_a))  # 简化处理
                elif mode == "颜色":
                    img_out = Image.merge('LAB', (l_a, a_b, b_b))  # 色相+饱和度
                elif mode == "明度":
                    img_out = Image.merge('LAB', (l_b, a_a, b_a))  # 使用背景明度
                # 转换回 RGB
                img_out = ImageCms.profileToProfile(img_out, ImageCms.createProfile('LAB'), ImageCms.createProfile('sRGB'), outputMode='RGB')
                # 应用混合系数
                if blend_factor != 1:
                    img_out = Image.blend(pil_a, img_out, blend_factor)
                # 转换为张量
                result_rgb = pil2tensor(img_out)[:, :, :, :3]  # (1, height, width, 3)
            except Exception as e:
                raise ValueError(f"Lab 色彩空间转换失败，建议切换到 HSV 模式或检查 Pillow 版本：{str(e)}")
        else:
            # 其他混合模式（包括 HSV）
            img_a_t = img_a_rgb.permute(0, 3, 1, 2)  # (1, channels, height, width)
            img_b_t = img_b_rgb.permute(0, 3, 1, 2)
            result = img_a_t.clone()
            if mode == "正常":
                result = img_b_t * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "溶解":
                mask = torch.rand(img_a_t.shape[2:], device=img_a_t.device) < blend_factor
                mask = mask.unsqueeze(0).unsqueeze(0).expand_as(img_a_t)
                result = torch.where(mask, img_b_t, img_a_t)
            elif mode == "变暗":
                result = torch.min(img_a_t, img_b_t) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "正片叠底":
                result = img_a_t * img_b_t * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "颜色加深":
                result = 1 - torch.clamp((1 - img_a_t) / (img_b_t + 1e-6), 0, 1)
                result = result * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "线性加深":
                result = torch.clamp(img_a_t + img_b_t - 1, 0, 1) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "深色":
                sum_a = img_a_t.sum(dim=1, keepdim=True)
                sum_b = img_b_t.sum(dim=1, keepdim=True)
                result = torch.where(sum_a < sum_b, img_a_t, img_b_t) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "变亮":
                result = torch.max(img_a_t, img_b_t) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "滤色":
                result = (1 - (1 - img_a_t) * (1 - img_b_t)) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "颜色减淡":
                result = torch.clamp(img_a_t / (1 - img_b_t + 1e-6), 0, 1) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "线性减淡":
                result = torch.clamp(img_a_t + img_b_t, 0, 1) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "浅色":
                sum_a = img_a_t.sum(dim=1, keepdim=True)
                sum_b = img_b_t.sum(dim=1, keepdim=True)
                result = torch.where(sum_a > sum_b, img_a_t, img_b_t) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "叠加":
                result = torch.where(img_a_t < 0.5, 2 * img_a_t * img_b_t, 1 - 2 * (1 - img_a_t) * (1 - img_b_t))
                result = result * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "柔光":
                result = torch.where(img_b_t < 0.5, img_a_t * (2 * img_b_t), 1 - (1 - img_a_t) * (2 * (1 - img_b_t)))
                result = result * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "强光":
                result = torch.where(img_b_t < 0.5, 2 * img_a_t * img_b_t, 1 - 2 * (1 - img_a_t) * (1 - img_b_t))
                result = result * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "亮光":
                result = torch.where(img_b_t < 0.5, img_a_t / (1 - 2 * img_b_t + 1e-6), img_a_t / (2 * (img_b_t - 0.5) + 1e-6))
                result = torch.clamp(result, 0, 1) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "线性光":
                result = torch.clamp(img_a_t + 2 * img_b_t - 1, 0, 1) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "点光":
                result = torch.where(img_b_t < 0.5, torch.clamp(img_a_t + 2 * img_b_t - 1, 0, 1),
                                     torch.clamp(img_a_t + 2 * (img_b_t - 0.5), 0, 1))
                result = result * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "硬混":
                result = torch.where(img_b_t < 0.5, torch.zeros_like(img_a_t), torch.ones_like(img_a_t))
                result = result * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "差值":
                result = torch.abs(img_a_t - img_b_t) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "排除":
                result = (img_a_t + img_b_t - 2 * img_a_t * img_b_t) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "减去":
                result = torch.clamp(img_a_t - img_b_t, 0, 1) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode == "除":
                result = torch.clamp(img_a_t / (img_b_t + 1e-6), 0, 1) * blend_factor + img_a_t * (1 - blend_factor)
            elif mode in ["色相", "饱和度", "颜色", "明度"] and color_space == 'HSV':
                img_a_np = img_a_t.cpu().numpy()[0].transpose(1, 2, 0)  # (height, width, 3)
                img_b_np = img_b_t.cpu().numpy()[0].transpose(1, 2, 0)
                hsv_a = rgb2hsv(img_a_np)
                hsv_b = rgb2hsv(img_b_np)
                if mode == "色相":
                    hsv_a[:, :, 0] = hsv_b[:, :, 0]
                elif mode == "饱和度":
                    hsv_a[:, :, 1] = hsv_b[:, :, 1]
                elif mode == "颜色":
                    hsv_a[:, :, 0:2] = hsv_b[:, :, 0:2]
                elif mode == "明度":
                    hsv_a[:, :, 2] = hsv_b[:, :, 2]
                result_np = hsv2rgb(hsv_a)
                result = torch.from_numpy(result_np).permute(2, 0, 1).unsqueeze(0).float()
                if blend_factor != 1:
                    result = result * blend_factor + img_a_t * (1 - blend_factor)
            result_rgb = result.permute(0, 2, 3, 1)  # (1, height, width, 3)

        # 应用 alpha 混合
        effective_alpha = alpha_a_i * blend_factor  # 考虑混合系数
        result = result_rgb * effective_alpha + img_b_rgb * (1 - effective_alpha)

        # 根据输出格式决定是否包含 alpha 通道
        if output_format == 'RGBA':
            result = torch.cat([result, alpha_a_i], dim=-1)  # (1, height, width, 4)

        output_tensors.append(result)

    # 堆叠批次
    output_tensor = torch.cat(output_tensors, dim=0)

    # 确保输出格式正确
    expected_channels = 4 if output_format == 'RGBA' else 3
    if output_tensor.shape[-1] != expected_channels:
        raise ValueError(f"输出张量通道数不正确：{output_tensor.shape}，期望 (batch, height, width, {expected_channels})")
    if output_tensor.dtype != torch.float32:
        output_tensor = output_tensor.float()
    if output_tensor.min() < 0 or output_tensor.max() > 1:
        output_tensor = torch.clamp(output_tensor, 0, 1)

    return output_tensor

# 节点定义
class MuyeImageBlendingMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "前景图像": ("IMAGE", {}),
                "背景图像": ("IMAGE", {}),
                "混合模式": ([
                    "正常", "溶解", "变暗", "正片叠底", "颜色加深", "线性加深", "深色",
                    "变亮", "滤色", "颜色减淡", "线性减淡", "浅色",
                    "叠加", "柔光", "强光", "亮光", "线性光", "点光", "硬混",
                    "差值", "排除", "减去", "除",
                    "色相", "饱和度", "颜色", "明度"
                ], {"default": "色相"}),
                "系数": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "数量差异补齐": (["无", "循环", "首张补齐", "尾张补齐", "随机补齐"], {"default": "无"}),
                "色彩空间": (["Lab", "HSV"], {"default": "Lab"}),
                "输出格式": (["RGB", "RGBA"], {"default": "RGB"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("混合图像",)
    FUNCTION = "blend"
    CATEGORY = "Muye/图像"

    def blend(self, 前景图像, 背景图像, 混合模式, 系数, 数量差异补齐, 色彩空间, 输出格式):
        # 标准化输入为张量
        try:
            tensor_a = normalize_input(前景图像, "前景图像")
            tensor_b = normalize_input(背景图像, "背景图像")
        except Exception as e:
            raise ValueError(f"输入标准化失败：{str(e)}")

        # 获取批次大小
        batch_size_a = tensor_a.shape[0]
        batch_size_b = tensor_b.shape[0]
        batch_size = max(batch_size_a, batch_size_b)

        # 处理数量差异
        if batch_size_a != batch_size_b:
            if 数量差异补齐 == "无":
                pass
            elif 数量差异补齐 == "循环":
                if batch_size_a < batch_size:
                    indices = torch.arange(batch_size) % batch_size_a
                    tensor_a = tensor_a[indices]
                else:
                    indices = torch.arange(batch_size) % batch_size_b
                    tensor_b = tensor_b[indices]
            elif 数量差异补齐 == "首张补齐":
                if batch_size_a < batch_size:
                    tensor_a = tensor_a.repeat((batch_size + batch_size_a - 1) // batch_size_a, 1, 1, 1)[:batch_size]
                else:
                    tensor_b = tensor_b.repeat((batch_size + batch_size_b - 1) // batch_size_b, 1, 1, 1)[:batch_size]
            elif 数量差异补齐 == "尾张补齐":
                if batch_size_a < batch_size:
                    tensor_a = torch.cat([tensor_a, tensor_a[-1:].repeat(batch_size - batch_size_a, 1, 1, 1)], dim=0)
                else:
                    tensor_b = torch.cat([tensor_b, tensor_b[-1:].repeat(batch_size - batch_size_b, 1, 1, 1)], dim=0)
            elif 数量差异补齐 == "随机补齐":
                if batch_size_a < batch_size:
                    indices = torch.randint(0, batch_size_a, (batch_size - batch_size_a,))
                    tensor_a = torch.cat([tensor_a, tensor_a[indices]], dim=0)
                else:
                    indices = torch.randint(0, batch_size_b, (batch_size - batch_size_b,))
                    tensor_b = torch.cat([tensor_b, tensor_b[indices]], dim=0)

        # 执行混合
        output_tensor = blend_tensors(tensor_a, tensor_b, 混合模式, 系数, 色彩空间, 输出格式)

        # 验证输出
        expected_channels = 4 if 输出格式 == 'RGBA' else 3
        if output_tensor.shape[-1] != expected_channels:
            raise ValueError(f"输出张量通道数不正确：{output_tensor.shape}，期望 (batch, height, width, {expected_channels})")
        if output_tensor.dtype != torch.float32:
            output_tensor = output_tensor.float()

        return (output_tensor,)

# 注册节点
NODE_CLASS_MAPPINGS = {
    "MuyeImageBlendingMode": MuyeImageBlendingMode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuyeImageBlendingMode": "图像混合模式"
}