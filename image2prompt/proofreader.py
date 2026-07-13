import torch
import numpy as np
import re
from PIL import Image

CAPTION_MODEL = "Muye_CaptionModel"


# ────────────────────────────────────────────────────────────
def comfy_image_to_pil(image_tensor):
    if isinstance(image_tensor, torch.Tensor):
        images = image_tensor
    else:
        images = torch.from_numpy(np.array(image_tensor))
    pil_list = []
    if images.ndim == 3:
        images = images.unsqueeze(0)
    for i in range(images.shape[0]):
        img = images[i].cpu().float()
        if img.max() <= 1.0:
            img = img * 255.0
        img = img.clamp(0, 255).byte()
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        elif img.shape[-1] == 1:
            img = img.repeat(1, 1, 3)
        pil = Image.fromarray(img.numpy(), mode="RGB")
        pil_list.append(pil)
    return pil_list









def proofread_with_model(model, processor, pil_img, 候选提示词列表, arch_type,
                         校对规则,
                         最大生成长度=2048, 温度=0.1, TopP采样=0.9, 重复惩罚=1.05):
    """
    让模型看着原图，对比多个候选提示词，进行交叉校对。
    去除幻觉、错误描述，合并正确信息，输出最终校对结果。
    校对规则由用户自定义输入，完全控制校对行为。
    """
    # 构建候选提示词的编号列表
    candidates_text = ""
    for idx, text in enumerate(候选提示词列表, 1):
        if text and text.strip():
            candidates_text += f"【来源{idx}】\n{text.strip()}\n\n"

    # system prompt：轻量引导角色
    system_prompt = (
        "你是一个智能助手，请严格按照用户的指令执行任务。\n"
        "如果用户上传了图片，请结合图片内容辅助完成任务。\n"
        "直接输出结果，不要任何前言后语。"
    )

    # 把用户校对规则和候选提示词都放在 user prompt 里，模型遵从度更高
    user_text = (
        f"{校对规则}\n\n"
        f"以下是多个来源对同一张图片的描述，请仔细查看原图后按上述规则进行校对：\n\n"
        f"{candidates_text}"
    )

    text_prompt = None
    if arch_type == "llava":
        # LLaVA 需要用简单字符串 messages，<image> 直接放在用户消息里
        user_prompt = f"<image>\n{user_text}"
        llava_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text_prompt = processor.apply_chat_template(
            llava_messages, add_generation_prompt=True, tokenize=False,
        )
        inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt")
    else:
        # Qwen 系列用多模态 messages 格式
        user_content = [
            {"type": "image"},
            {"type": "text", "text": user_text},
        ]
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        text_prompt = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        inputs = processor(
            text=[text_prompt], images=[pil_img],
            return_tensors="pt", padding=True,
        )

    model_inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=最大生成长度,
            temperature=温度,
            top_p=TopP采样,
            repetition_penalty=重复惩罚,
            do_sample=True if 温度 > 0.0 else False,
        )

    input_ids = model_inputs["input_ids"]
    generated_tokens = output_ids[0][input_ids.shape[1]:]
    result = processor.decode(
        generated_tokens, skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    result = result.strip()
    while result.startswith('\n'):
        result = result[1:]

    return result


# ════════════════════════════════════════════════════════════
class 提示词校对:
    """
    木叶节点 - 提示词校对
    接收多个反推节点的输出，让模型看着原图进行交叉对比校对，
    去除幻觉和错误描述，输出准确的最终提示词。
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "模型": (CAPTION_MODEL,),
                "校对规则": ("STRING", {
                    "default": "仔细观察图片，对比多个来源的描述，去除幻觉和错误内容，输出准确的描述。",
                    "multiline": True,
                    "tooltip": "自定义校对规则，模型会严格按照此规则执行校对任务。你可以指定风格、语言、详细程度、NSFW处理方式等任何要求。",
                }),
                "最大生成长度": ("INT", {
                    "default": 2048, "min": 64, "max": 8192, "step": 64,
                    "tooltip": "校对输出最大长度",
                }),
                "温度": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 2.0, "step": 0.01,
                    "tooltip": "温度控制创造性，0=最确定(复读机)，0.1=推荐值(平衡准确和多样性)，越高越有变化但可能不准确",
                }),
                "TopP采样": ("FLOAT", {
                    "default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "重复惩罚": ("FLOAT", {
                    "default": 1.05, "min": 1.0, "max": 2.0, "step": 0.01,
                }),
            },
            "optional": {
                "提示词1": ("STRING", {"tooltip": "第一个反推节点的提示词输出"}),
                "提示词2": ("STRING", {"tooltip": "第二个反推节点的提示词输出（可选）"}),
                "提示词3": ("STRING", {"tooltip": "第三个反推节点的提示词输出（可选）"}),
                "提示词4": ("STRING", {"tooltip": "第四个反推节点的提示词输出（可选）"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("校对后提示词",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "proofread"
    CATEGORY = "Muye/图像反推"

    def proofread(
        self,
        图像,
        模型,
        校对规则,
        提示词1,
        最大生成长度=2048,
        温度=0.1,
        TopP采样=0.9,
        重复惩罚=1.05,
        提示词2="",
        提示词3="",
        提示词4="",
    ):
        # 检查模型
        if not hasattr(模型, "model") or not hasattr(模型, "processor") or not hasattr(模型, "arch_type"):
            return (["错误：未正确连接模型"],)
        if 模型.model is None:
            return (["错误：模型加载失败"],)

        model = 模型.model
        processor = 模型.processor
        arch_type = 模型.arch_type

        print(f"[木叶·提示词校对] 使用模型: {模型.model_name} ({arch_type})")
        print(f"[木叶·提示词校对] 校对规则: {校对规则[:80]}{'...' if len(校对规则) > 80 else ''}")

        # 收集有效的候选提示词（兼容字符串和列表输入）
        def _to_str(x):
            """将输入转为字符串，兼容list/Tensor等类型"""
            if isinstance(x, list):
                for item in x:
                    if isinstance(item, str) and item.strip():
                        return item.strip()
                return ""
            elif isinstance(x, str):
                return x.strip()
            else:
                return str(x).strip()

        candidates = []
        提示词1_str = _to_str(提示词1)
        提示词2_str = _to_str(提示词2)
        提示词3_str = _to_str(提示词3)
        提示词4_str = _to_str(提示词4)

        if 提示词1_str:
            candidates.append(提示词1_str)
        if 提示词2_str:
            candidates.append(提示词2_str)
        if 提示词3_str:
            candidates.append(提示词3_str)
        if 提示词4_str:
            candidates.append(提示词4_str)

        if len(candidates) < 1:
            return (["错误：请至少提供一个提示词来源"],)

        if len(candidates) == 1:
            print(f"[木叶·提示词校对] 收到 1 个提示词来源，进行单来源校对")
        else:
            print(f"[木叶·提示词校对] 收到 {len(candidates)} 个提示词来源，开始校对")

        # 转换图片
        pil_images = comfy_image_to_pil(图像)
        results = []

        for pil_img in pil_images:
            try:
                result = proofread_with_model(
                    model, processor, pil_img, candidates, arch_type,
                    校对规则,
                    最大生成长度, 温度, TopP采样, 重复惩罚,
                )
                results.append(result)
                print(f"[木叶·提示词校对] 校对完成, 长度: {len(result)}")
            except Exception as e:
                import traceback
                results.append(f"校对失败: {str(e)}")
                print(f"[木叶·提示词校对] 失败:\n{traceback.format_exc()}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return (results,)


NODE_CLASS_MAPPINGS = {
    "提示词校对": 提示词校对,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "提示词校对": "🦞 提示词校对",
}