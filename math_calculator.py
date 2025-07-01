class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
any = AnyType("*")
import re

# 调试日志
## print("Loading MuyeMathCalculator from muye_math_calculator.py")

class MuyeMathCalculator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "表达式": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "输入数学表达式，如 a+b*c 或 1*2+3"
                }),
            },
            "optional": {
                # 允许任意类型输入，内部自动转换
                "a": (any,),
                "b": (any,),
                "c": (any,),
                "d": (any,),
                "e": (any,),
            }
        }

    RETURN_TYPES = ("INT", "FLOAT")
    RETURN_NAMES = ("整数", "浮点")
    FUNCTION = "calculate"
    CATEGORY = "Muye"
    OUTPUT_NODE = True

    def calculate(self, 表达式, a=None, b=None, c=None, d=None, e=None):
        available_ports = ['a', 'b', 'c', 'd', 'e']
        def parse_num(val):
            if isinstance(val, (int, float)):
                return val
            try:
                return float(val)
            except Exception:
                return 0
        port_values = {p: parse_num(locals()[p]) for p in available_ports}
        connected_ports = [p for p in available_ports if locals()[p] is not None]

        if not 表达式.strip():
            return 0, 0.0

        try:
            expr = 表达式.strip()
            if connected_ports:
                used_vars = set(re.findall(r'[a-e]', expr))
                invalid_vars = used_vars - set(connected_ports)
                if invalid_vars:
                    raise ValueError(f"表达式包含未连接的变量：{invalid_vars}")
                safe_dict = {p: port_values[p] for p in connected_ports}
                if not re.match(r'^[\w\s\+\-\*/\(\)]+$', expr):
                    raise ValueError("表达式包含非法字符")
                result = eval(expr, {"__builtins__": {}}, safe_dict)
            else:
                if re.search(r'[a-e]', expr):
                    raise ValueError("无端口连接时，表达式不能包含变量 a-e")
                if not re.match(r'^[\d\s\+\-\*/\(\)]+$', expr):
                    raise ValueError("直接模式仅允许数字和运算符")
                result = eval(expr, {"__builtins__": {}}, {})

            float_result = float(result)
            int_result = int(float_result)
            return int_result, float_result

        except Exception as e:
            print(f"计算错误: {str(e)}")
            return 0, 0.0

# 节点映射
NODE_CLASS_MAPPINGS = {
    "MuyeMathCalculator": MuyeMathCalculator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MuyeMathCalculator": "数学表达式计算"
}
