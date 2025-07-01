# -*- coding: utf-8 -*-
import os
import re
from PIL import Image, UnidentifiedImageError
import torch
import numpy as np
import folder_paths
import torchaudio
from PIL import Image, UnidentifiedImageError
try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False

class MuyeFileReader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "路径": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "dynamicPrompts": True,
                    "tooltip": r"输入文件或文件夹的完整路径（如 F:\\ComfyUI_windows_portable\\ComfyUI\\output\\娜然）",
                }),
                "文件名称输出序号": ("STRING", {
                    "default": "0",
                    "multiline": False,
                    "tooltip": "输入序号，如'1,3,5'或'1-10'，0表示全部",
                    "label": "文件名称输出序号"
                }),
                "去掉文件名称后缀": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "启用时，输出的文件名称将移除扩展名（如 .png、.txt 等）",
                    "label": "去掉文件名称后缀"
                }),
                "图像输出序号": ("STRING", {
                    "default": "0",
                    "multiline": False,
                    "tooltip": "输入序号，如'1,3,5'或'1-10'，0表示全部",
                    "label": "图像输出序号"
                }),
                "文本输出序号": ("STRING", {
                    "default": "0",
                    "multiline": False,
                    "tooltip": "输入序号，如'1,3,5'或'1-10'，0表示全部",
                    "label": "文本输出序号"
                }),
                "输出排序方式": (["文件名称", "修改日期", "文件大小", "图像分辨率", "文件类型"], {
                    "default": "文件名称",
                    "label": "输出排序方式"
                }),
                "排序规则": (["正序", "倒序"], {
                    "default": "正序",
                    "label": "排序规则"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "INT", "IMAGE", "STRING")
    RETURN_NAMES = ("文件名称列表", "文件总数", "图片", "文本")
    OUTPUT_IS_LIST = (True, False, True, True)
    FUNCTION = "read_files"
    CATEGORY = "Muye/文件"

    def resolve_path(self, path):
        if path is None:
            path = ""
        elif isinstance(path, (list, tuple)) and len(path) > 0:
            path = path[0]
        elif isinstance(path, dict) and 'path' in path:
            path = path['path']
        elif not isinstance(path, str):
            path = ""

        if not path or path.strip() == "":
            return None

        # 去掉首尾引号
        path = path.strip()
        if (path.startswith('"') and path.endswith('"')) or (path.startswith("'") and path.endswith("'")):
            path = path[1:-1].strip()

        path = os.path.normpath(path)
        if os.path.exists(path):
            return os.path.abspath(path)
        
        for base_dir in [folder_paths.get_input_directory(), folder_paths.get_output_directory()]:
            full_path = os.path.join(base_dir, os.path.basename(path))
            if os.path.exists(full_path):
                return os.path.abspath(full_path)
        
        return None

    def parse_indices(self, index_str, max_index):
        if index_str is None:
            index_str = "0"
        if not index_str or index_str.strip() == "0":
            return list(range(max_index))
        
        indices = set()
        index_str = index_str.replace('，', ',').replace(' ', ',')
        parts = index_str.split(',')
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if '-' in part:
                try:
                    start, end = map(int, part.split('-'))
                    start = max(1, start)
                    end = min(max_index, end)
                    if start <= end:
                        indices.update(range(start - 1, end))
                except ValueError:
                    continue
            else:
                try:
                    idx = int(part)
                    if 1 <= idx <= max_index:
                        indices.add(idx - 1)
                except ValueError:
                    continue
        
        return sorted(list(indices))

    def natural_sort_key(self, name):
        """将文件名分解为非数字和数字部分，用于自然排序"""
        def convert(text):
            return int(text) if text.isdigit() else text.lower()
        return [convert(c) for c in re.split(r'(\d+)', name)]

    def read_files(self, 路径, 文件名称输出序号, 去掉文件名称后缀, 图像输出序号, 文本输出序号, 输出排序方式, 排序规则):
        file_names = []
        file_count = 0
        image_tensors = []
        text_contents = []
        video_output = None
        audio_output = None

        resolved_path = self.resolve_path(路径)
        if not resolved_path or not os.path.exists(resolved_path):
            return (file_names, file_count, image_tensors, text_contents, video_output, audio_output)
        路径 = resolved_path

        image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff", ".tif", ".webp"}
        text_extensions = {".txt", ".csv", ".json", ".xml", ".yaml", ".yml", ".md"}

        files_info = []
        if os.path.isfile(路径):
            file_name = os.path.basename(路径)
            file_path = 路径
            file_ext = os.path.splitext(file_name)[1].lower()
            file_size = os.path.getsize(file_path)
            file_mtime = os.path.getmtime(file_path)
            resolution = (0, 0)
            if file_ext in image_extensions:
                try:
                    with Image.open(file_path) as img:
                        resolution = img.size
                except (OSError, UnidentifiedImageError):
                    pass
            files_info = [{
                "name": file_name,
                "path": file_path,
                "ext": file_ext,
                "size": file_size,
                "mtime": file_mtime,
                "resolution": resolution
            }]
            file_names = [file_name]
            file_count = 1
        elif os.path.isdir(路径):
            try:
                for f in os.listdir(路径):
                    file_path = os.path.join(路径, f)
                    if os.path.isfile(file_path):
                        file_ext = os.path.splitext(f)[1].lower()
                        file_size = os.path.getsize(file_path)
                        file_mtime = os.path.getmtime(file_path)
                        resolution = (0, 0)
                        if file_ext in image_extensions:
                            try:
                                with Image.open(file_path) as img:
                                    resolution = img.size
                            except (OSError, UnidentifiedImageError):
                                pass
                        files_info.append({
                            "name": f,
                            "path": file_path,
                            "ext": file_ext,
                            "size": file_size,
                            "mtime": file_mtime,
                            "resolution": resolution
                        })
                file_names = [info["name"] for info in files_info]
                file_count = len(file_names)
            except OSError:
                return (file_names, file_count, image_tensors, text_contents, video_output, audio_output)

        image_files = [f for f in files_info if f["ext"] in image_extensions]
        text_files = [f for f in files_info if f["ext"] in text_extensions]
        # video_files = [f for f in files_info if f["ext"] in video_extensions]
        # audio_files = [f for f in files_info if f["ext"] in audio_extensions]

        def get_sort_key(info):
            if 输出排序方式 == "文件名称":
                return self.natural_sort_key(info["name"])
            elif 输出排序方式 == "修改日期":
                return info["mtime"]
            elif 输出排序方式 == "文件大小":
                return info["size"]
            elif 输出排序方式 == "图像分辨率":
                return info["resolution"][0] * info["resolution"][1] if info["resolution"] != (0, 0) else 0
            elif 输出排序方式 == "文件类型":
                return info["ext"]
            return self.natural_sort_key(info["name"])

        # for file_list in [files_info, image_files, text_files, video_files, audio_files]:
        for file_list in [files_info, image_files, text_files]:
            file_list.sort(key=get_sort_key, reverse=(排序规则 == "倒序"))

        name_indices = self.parse_indices(文件名称输出序号, len(files_info))
        image_indices = self.parse_indices(图像输出序号, len(image_files))
        text_indices = self.parse_indices(文本输出序号, len(text_files))

        file_names = [files_info[i]["name"] for i in name_indices]
        if 去掉文件名称后缀:
            file_names = [os.path.splitext(name)[0] for name in file_names]

        for i in image_indices:
            try:
                info = image_files[i]
                image = Image.open(info["path"]).convert("RGB")
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]
                image_tensors.append(image_tensor)
            except (OSError, UnidentifiedImageError):
                pass

        for i in text_indices:
            try:
                info = text_files[i]
                with open(info["path"], "r", encoding="utf-8") as f:
                    text_contents.append(f.read())
            except OSError:
                pass

        return (file_names, file_count, image_tensors, text_contents)

    @classmethod
    def VALIDATE_INPUTS(cls, 路径, 文件名称输出序号, 去掉文件名称后缀, 图像输出序号, 文本输出序号, 输出排序方式, 排序规则):
        if 路径 is not None and not isinstance(路径, str):
            return f"路径 must be a string or None, got type={type(路径)}, value={路径!r}"

        inputs = [
            (文件名称输出序号, "文件名称输出序号"),
            (图像输出序号, "图像输出序号"),
            (文本输出序号, "文本输出序号")
        ]
        for input_value, input_name in inputs:
            if input_value is not None and not isinstance(input_value, str):
                return f"{input_name} must be a string or None, got type={type(input_value)}, value={input_value!r}"

        if not isinstance(去掉文件名称后缀, bool):
            return f"去掉文件名称后缀 must be a boolean, got type={type(去掉文件名称后缀)}, value={去掉文件名称后缀!r}"

        for index_str, name in inputs:
            if index_str is None:
                continue
            if index_str.strip() != "0":
                index_str = index_str.replace('，', ',').replace(' ', ',')
                parts = index_str.split(',')
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if '-' in part:
                        try:
                            start, end = map(int, part.split('-'))
                            if start < 1 or end < start:
                                return f"Invalid range in {name}: {part}"
                        except ValueError:
                            return f"Invalid range format in {name}: {part}"
                    else:
                        try:
                            idx = int(part)
                            if idx < 1:
                                return f"Invalid index in {name}: {idx}"
                        except ValueError:
                            return f"Invalid index format in {name}: {part}"
        
        return True

NODE_CLASS_MAPPINGS = {"MuyeFileReader": MuyeFileReader}
NODE_DISPLAY_NAME_MAPPINGS = {"MuyeFileReader": "文件夹读取器"}