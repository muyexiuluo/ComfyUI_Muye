import re



class TextSplitByDelimiterEnhanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入来源": (["文本框", "外部列表"], {"default": "文本框"}),
                "输入文本": ("STRING", {"default": "", "multiline": True}),
                "分隔符": ("STRING", {"default": "<think>"}),
                "使用正则表达式": ("BOOLEAN", {"default": False}),
                "输出段落索引": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "输出列表索引": ("INT", {"default": 1, "min": 0, "max": 100, "step": 1}),
                "总列表段落选择": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            },
            "optional": {
                "外部文本列表": ("STRING", {"forceInput": True, "is_list": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("总列表", "选中列表的完整分割", "选中段落")
    OUTPUT_IS_LIST = (True, True, False)
    FUNCTION = "split_text"
    CATEGORY = "Muye/文本"

    def split_text(self, 输入来源, 输入文本, 分隔符, 使用正则表达式, 输出段落索引, 输出列表索引, 总列表段落选择, 外部文本列表=None):
        # 初始化文本列表
        texts_to_process = []

        # 根据输入来源选择文本
        if 输入来源 == "文本框":
            if not 输入文本 or not 分隔符:
                return ([], [], "")
            texts_to_process = [输入文本]
        else:  # 外部列表
            if not 外部文本列表 or not 分隔符:
                return ([], [], "")
            texts_to_process = 外部文本列表

        # 分割所有文本，保存每个文本的分割结果，同时生成总列表
        total_segments = []  # 所有段落的合并列表
        split_results = []  # 每个文本的分割结果
        for text in texts_to_process:
            if 使用正则表达式:
                split_result = re.split(分隔符, text)
            else:
                split_result = text.split(分隔符)
            split_result = [item.strip() for item in split_result if item.strip()]
            split_results.append(split_result)
            total_segments.extend(split_result)

        # 根据“总列表段落选择”调整总列表输出
        if 总列表段落选择 == 0:
            final_total_segments = total_segments  # 输出所有段落
        else:
            # 从每个文本的分割结果中取第 N 段
            segment_idx = 总列表段落选择 - 1  # 转换为 0-based 索引
            final_total_segments = []
            for split_result in split_results:
                if 0 <= segment_idx < len(split_result):
                    final_total_segments.append(split_result[segment_idx])

        # 选中列表的完整分割
        if 输出列表索引 == 0:
            selected_segments = []  # 当输出列表索引为 0 时，返回空列表
        else:
            selected_list_index = max(0, min(输出列表索引 - 1, len(texts_to_process) - 1))  # 转换为 0-based 索引并防止越界
            selected_text = texts_to_process[selected_list_index]
            if 使用正则表达式:
                selected_segments = re.split(分隔符, selected_text)
            else:
                selected_segments = selected_text.split(分隔符)
            selected_segments = [item.strip() for item in selected_segments if item.strip()]

        # 选中段落
        if 输出段落索引 == 0:
            selected_segment = ""  # 当输出段落索引为 0 时，返回空字符串
        else:
            if 输入来源 == "文本框":
                # 单个文本：直接从总列表中选择段落
                idx = 输出段落索引 - 1  # 转换为 0-based 索引
                selected_segment = total_segments[idx] if 0 <= idx < len(total_segments) else ""
            else:
                # 列表输入：从选中列表的分割结果中选择段落
                idx = 输出段落索引 - 1  # 转换为 0-based 索引
                selected_segment = selected_segments[idx] if 0 <= idx < len(selected_segments) else ""

        # 返回结果
        return (final_total_segments, selected_segments, selected_segment)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "TextSplitByDelimiterEnhanced": TextSplitByDelimiterEnhanced
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextSplitByDelimiterEnhanced": "文本按分隔符分割"
}

## print("TextSplitByDelimiterEnhanced node defined successfully")