class TextListSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "文本": ("STRING", {
                    "forceInput": True,
                    "is_list": True,
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("输出",)
    OUTPUT_IS_LIST = (True,)  # 修正为 True
    FUNCTION = "split_list"
    CATEGORY = "Muye/文本"
    OUTPUT_NODE = True

    def split_list(self, 文本=None):
        # 调试日志：打印原始输入
        print(f"[TextListSplit] 原始输入类型: {type(文本)}, 内容: {str(文本)[:100]}...")

        # 初始化输出列表
        items = []

        # 处理输入
        try:
            if 文本 is None or 文本 == "":
                print("[TextListSplit] 警告: 输入为 None 或空")
                return ([],)  # 返回空列表
            
            if isinstance(文本, str):
                # 字符串输入：按换行符、逗号或分号分割
                if "\n" in 文本:
                    items = [item.strip() for item in 文本.split("\n") if item.strip()]
                elif "," in 文本:
                    items = [item.strip() for item in 文本.split(",") if item.strip()]
                elif ";" in 文本:
                    items = [item.strip() for item in 文本.split(";") if item.strip()]
                else:
                    items = [文本.strip()] if 文本.strip() else []
            
            elif isinstance(文本, (list, tuple)):
                # 列表输入：展平嵌套列表
                def flatten(lst):
                    result = []
                    for item in lst:
                        if isinstance(item, (list, tuple)):
                            result.extend(flatten(item))
                        elif isinstance(item, str) and item.strip():
                            result.append(item.strip())
                        else:
                            # 非字符串类型转换为字符串
                            item_str = str(item).strip()
                            if item_str:
                                result.append(item_str)
                    return result
                items = flatten(文本)
            
            else:
                # 其他类型：转换为字符串
                item_str = str(文本).strip()
                items = [item_str] if item_str else []
        
        except Exception as e:
            print(f"[TextListSplit] 处理输入时发生错误: {str(e)}")
            return ([],)  # 返回空列表

        # 检查处理结果
        if not items:
            print("[TextListSplit] 警告: 处理后列表为空")
            return ([],)  # 返回空列表

        # 调试日志：打印处理结果
        print(f"[TextListSplit] 处理后列表长度: {len(items)}, 前3项: {[item[:50] + '...' for item in items[:3]]}")

        # 按批次输出，每次一个完整文本项
        outputs = []
        for idx, item in enumerate(items):
            outputs.append((item,))
            print(f"[TextListSplit] 输出批次 {idx + 1}/{len(items)}: {item[:50]}...")

        return (items,)  # 返回字符串列表作为唯一输出

# 节点映射
NODE_CLASS_MAPPINGS = {
    "TextListSplit": TextListSplit
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextListSplit": "文本列表拆分"
}
