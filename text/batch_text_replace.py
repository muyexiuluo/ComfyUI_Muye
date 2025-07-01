import re



class BatchTextReplace:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入文本": ("STRING", {"forceInput": True, "is_list": True}),
                "查找文本1": ("STRING", {"default": ""}),
                "替换为1": ("STRING", {"default": ""}),
                "查找文本2": ("STRING", {"default": ""}),
                "替换为2": ("STRING", {"default": ""}),
                "查找文本3": ("STRING", {"default": ""}),
                "替换为3": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("替换后列表",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "replace_text"
    CATEGORY = "Muye/文本"

    def replace_text(self, 输入文本, 查找文本1, 替换为1, 查找文本2, 替换为2, 查找文本3, 替换为3):
        # 如果输入是字符串，转换为单元素列表
        if isinstance(输入文本, str):
            texts_to_process = [输入文本]
        else:
            texts_to_process = 输入文本

        # 检查输入是否为空
        if not texts_to_process:
            return ([],)

        # 批量替换
        replaced_texts = []
        for text in texts_to_process:
            current_text = text

            # 定义三组替换规则
            replacements = [
                (查找文本1, 替换为1),
                (查找文本2, 替换为2),
                (查找文本3, 替换为3),
            ]

            # 按顺序应用每一组替换规则
            for find_text, replace_text in replacements:
                if not find_text:  # 如果查找文本为空，跳过
                    continue

                # 尝试判断是否为正则表达式
                try:
                    # 尝试编译正则表达式
                    re.compile(find_text)
                    # 如果编译成功，使用正则表达式替换
                    try:
                        current_text = re.sub(find_text, replace_text, current_text)
                    except re.error:
                        # 如果正则表达式执行失败，跳过
                        continue
                except re.error:
                    # 如果编译失败，说明是普通字符串，使用普通替换
                    current_text = current_text.replace(find_text, replace_text)

            replaced_texts.append(current_text)

        return (replaced_texts,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "BatchTextReplace": BatchTextReplace
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchTextReplace": "批量文本替换"
}

## print("BatchTextReplace node defined successfully")