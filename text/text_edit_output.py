
class MuyeTextEditOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "编辑文本": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "rows": 5,
                    "label": "编辑文本",
                    "tooltip": "手动输入文本"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("文本",)
    OUTPUT_IS_LIST = (True,)
    OUTPUT_NODE = True
    FUNCTION = "process_text"
    CATEGORY = "Muye/文本"

    def process_text(self, 编辑文本=""):
        # 直接整体输出编辑文本，不做分割
        text = 编辑文本 if 编辑文本 else ""
        return ([text],)

NODE_CLASS_MAPPINGS = {
    "MuyeTextEditOutput": MuyeTextEditOutput
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuyeTextEditOutput": "文本"
}