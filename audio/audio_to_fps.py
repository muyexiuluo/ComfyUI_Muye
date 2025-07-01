class AudioToFPS:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "音频": ("AUDIO", {"defaultInput": True, "label": "音频输入"}),
                "帧率": ("INT", {"default": 25, "min": 1, "max": 240, "label": "帧率"}),
                "因数": ("INT", {"default": 8, "min": 1, "max": 10000, "label": "因数"}),
                "加1帧": ("BOOLEAN", {"default": True, "label": "加1帧"}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("音频时长(s)", "FPS")
    FUNCTION = "audio_to_fps"
    CATEGORY = "Muye/音频"

    def audio_to_fps(self, 音频, 帧率, 因数, 加1帧):
        # 音频输入为 dict，包含 waveform 和 sample_rate
        waveform = None
        sample_rate = None
        if isinstance(音频, dict):
            waveform = 音频.get("waveform", None)
            sample_rate = 音频.get("sample_rate", None)
        if waveform is None or sample_rate is None:
            return (0, 0)
        # 计算时长（以秒为单位，四舍五入为整数秒，只输出整数）
        if hasattr(waveform, 'shape'):
            duration = int(round(waveform.shape[-1] / float(sample_rate)))
        else:
            duration = 0
        # 计算 FPS，只输出整数
        fps = int(duration * 帧率)
        # 使 fps 可被因数整除
        if 因数 > 1:
            fps = (fps // 因数) * 因数
        if 加1帧:
            fps += 1
        return (int(duration), int(fps))

NODE_CLASS_MAPPINGS = {
    "AudioToFPS": AudioToFPS
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioToFPS": "音频到FPS转换"
}
