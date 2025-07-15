import cv2
import numpy as np
import mediapipe as mp

class FaceSelector:
    def __init__(self):
        try:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True)
            self.mp_drawing = mp.solutions.drawing_utils
            print("成功加载MediaPipe模型")
            self.model_loaded = True
        except Exception as e:
            print("未检测到MediaPipe模型")
            self.model_loaded = False

    def detect_faces(self, image, min_size=50):
        if not getattr(self, 'model_loaded', True):
            print("未检测到MediaPipe模型")
            return []
        results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        faces = []
        if results.detections:
            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x1 = int(bboxC.xmin * iw)
                y1 = int(bboxC.ymin * ih)
                w = int(bboxC.width * iw)
                h = int(bboxC.height * ih)
                if w >= min_size and h >= min_size:
                    faces.append({
                        'box': [x1, y1, w, h],
                        'score': detection.score[0],
                        'gender': self._estimate_gender(image[y1:y1+h, x1:x1+w])
                    })
        print(f"检测到有效人脸数量: {len(faces)}")
        for i, f in enumerate(faces):
            print(f"人脸{i}: box={f['box']}")
        return faces

    def _estimate_gender(self, face_img):
        # 这里可以接入更复杂的性别识别模型，暂时用亮度简单区分
        if face_img.size == 0:
            return 'unknown'
        mean = np.mean(face_img)
        return 'male' if mean < 120 else 'female'

    def sort_faces(self, faces, method='area'):
        if method == 'area':
            return sorted(faces, key=lambda f: f['box'][2]*f['box'][3], reverse=True)
        elif method == 'left':
            return sorted(faces, key=lambda f: f['box'][0])
        return faces

    def crop_face(self, image, face, crop_scale=2.0, rotate=True):
        x, y, w, h = face['box']
        cx, cy = x + w//2, y + h//2
        size = int(max(w, h) * crop_scale)
        nx, ny = max(cx - size//2, 0), max(cy - size//2, 0)
        ex, ey = min(cx + size//2, image.shape[1]), min(cy + size//2, image.shape[0])
        crop_img = image[ny:ey, nx:ex].copy()
        angle = 0
        if rotate:
            angle = self._get_face_angle(image, face)
            print(f" 旋转角度: {angle}")
            crop_img = self._rotate_image(crop_img, angle)
        crop_data = {
            'box': [nx, ny, ex-nx, ey-ny],
            'angle': angle
        }
        return crop_img, crop_data

    def _get_face_angle(self, image, face):
        # 用face mesh检测眼睛位置，计算旋转角度
        x, y, w, h = face['box']
        roi = image[y:y+h, x:x+w]
        results = self.face_mesh.process(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            left_eye = lm[33]
            right_eye = lm[263]
            dx = (right_eye.x - left_eye.x) * w
            dy = (right_eye.y - left_eye.y) * h
            angle = np.degrees(np.arctan2(dy, dx))
            return angle
        return 0

    def _rotate_image(self, image, angle):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
        return cv2.warpAffine(image, M, (w, h))

# ComfyUI节点实现
import comfy.sd

class 面部选择器:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "区分男女": (["不区分", "男", "女"], {"default": "不区分"}),
                "人物排序": (["像素占比", "从左向右"], {"default": "从左向右"}),
                "输出索引": ("INT", {"default": 1, "min": 0, "max": 100}),
                "最小尺寸": ("INT", {"default": 50, "min": 0, "max": 5000}),
                "置信度阈值": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "裁剪系数": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0}),
                "是否旋转面部": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FACE_CROP_DATA", "MASK")
    RETURN_NAMES = ("裁剪图像", "裁剪数据", "裁剪遮罩")
    FUNCTION = "run"
    CATEGORY = "Muye/面部"

    def __init__(self):
        self.selector = FaceSelector()

    def run(self, 图像, 区分男女, 人物排序, 输出索引, 最小尺寸, 置信度阈值, 裁剪系数, 是否旋转面部):
        import numpy as np
        # 保留原始原图
        if hasattr(图像, 'cpu') and hasattr(图像, 'numpy'):
            orig_img = 图像.cpu().numpy()
            if orig_img.ndim == 3 and orig_img.shape[0] in [1, 3, 4]:
                orig_img = np.transpose(orig_img, (1, 2, 0))
            if orig_img.ndim == 4 and orig_img.shape[0] == 1:
                orig_img = orig_img[0]
            orig_img = orig_img.copy()
            if orig_img.max() <= 1.0:
                orig_img = (orig_img * 255).clip(0, 255).astype(np.uint8)
            else:
                orig_img = orig_img.astype(np.uint8)
        else:
            orig_img = np.array(图像)
            if orig_img.ndim == 4 and orig_img.shape[0] == 1:
                orig_img = orig_img[0]

        # ...删除原有通道相关调试...

        # 用于人脸检测的img（保证MediaPipe能识别）
        img = orig_img
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        faces = self.selector.detect_faces(img, min_size=最小尺寸)
        # 置信度过滤
        faces = [f for f in faces if f.get('score', 1.0) >= 置信度阈值]
        # 性别过滤（当前未集成性别识别模型，选项无效，仅做提示）
        if 区分男女 != "不区分":
            print("当前未集成性别识别模型，‘区分男女’选项无效，始终不区分性别。")
        # 排序
        if 人物排序 == "像素占比":
            faces = self.selector.sort_faces(faces, method='area')
        else:
            faces = self.selector.sort_faces(faces, method='left')

        import torch
        def np2torch(img_np, batch=False):
            arr = np.array(img_np)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            tensor = torch.from_numpy(arr).contiguous()
            if batch:
                return tensor  # (H, W, 3)
            else:
                return tensor.unsqueeze(0)  # (1, H, W, 3)

        if 输出索引 == 0 and faces:
            # 输出所有人脸的裁剪图像/数据/mask列表，图像为(H, W, 3)的torch.Tensor
            crop_imgs, crop_datas, mask_crops = [], [], []
            for idx, face in enumerate(faces):
                if 是否旋转面部:
                    angle = self.selector._get_face_angle(orig_img, face)
                    h0, w0 = orig_img.shape[:2]
                    center = (w0/2, h0/2)
                    M = cv2.getRotationMatrix2D(center, angle, 1)
                    rotated_img = cv2.warpAffine(orig_img, M, (w0, h0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    x, y, w, h = face['box']
                    box_pts = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]], dtype=np.float32)
                    ones = np.ones((4,1), dtype=np.float32)
                    box_pts_homo = np.concatenate([box_pts, ones], axis=1)
                    new_pts = (M @ box_pts_homo.T).T
                    nx, ny = new_pts[:,0].min(), new_pts[:,1].min()
                    ex, ey = new_pts[:,0].max(), new_pts[:,1].max()
                    nx, ny, ex, ey = int(round(nx)), int(round(ny)), int(round(ex)), int(round(ey))
                    cx, cy = (nx+ex)//2, (ny+ey)//2
                    size = int(max(ex-nx, ey-ny) * 裁剪系数)
                    nnx, nny = max(cx-size//2, 0), max(cy-size//2, 0)
                    nex, ney = min(cx+size//2, w0), min(cy+size//2, h0)
                    crop_img = rotated_img[nny:ney, nnx:nex].copy()
                    mask = np.zeros((h0, w0), dtype=np.uint8)
                    cv2.fillConvexPoly(mask, np.int32(new_pts), 255)
                    mask_crop = mask[nny:ney, nnx:nex].copy()
                    crop_data = {
                        'box': [nnx, nny, nex-nnx, ney-nny],
                        'angle': angle,
                        'center': center,
                        'rotated': True
                    }
                else:
                    crop_img, crop_data = self.selector.crop_face(orig_img, face, crop_scale=裁剪系数, rotate=False)
                    mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
                    x, y, w, h = crop_data['box']
                    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                    mask_crop = mask[y:y+h, x:x+w].copy()
                    crop_data['angle'] = 0
                    crop_data['center'] = (orig_img.shape[1]/2, orig_img.shape[0]/2)
                    crop_data['rotated'] = False
                print(f"裁剪坐标: {crop_data['box']}, 旋转角度: {crop_data['angle']}")
                if crop_img.ndim == 2:
                    crop_img = np.stack([crop_img]*3, axis=-1)
                elif crop_img.ndim == 3 and crop_img.shape[2] == 1:
                    crop_img = np.repeat(crop_img, 3, axis=2)
                elif crop_img.ndim == 3 and crop_img.shape[2] > 3:
                    crop_img = crop_img[:, :, :3]
                crop_imgs.append(np2torch(crop_img, batch=True))
                crop_datas.append(crop_data)
                mask_crops.append(mask_crop)
            return (crop_imgs, crop_datas, mask_crops)
        elif not faces:
            mask = np.ones(orig_img.shape[:2], dtype=np.uint8) * 255
            mask_crop = mask.copy()
            return (np2torch(orig_img), {"box": None, "angle": 0}, mask_crop)
        else:
            idx = min(输出索引-1, len(faces)-1)
            if 是否旋转面部:
                angle = self.selector._get_face_angle(orig_img, faces[idx])
                h0, w0 = orig_img.shape[:2]
                center = (w0/2, h0/2)
                M = cv2.getRotationMatrix2D(center, angle, 1)
                rotated_img = cv2.warpAffine(orig_img, M, (w0, h0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                x, y, w, h = faces[idx]['box']
                box_pts = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]], dtype=np.float32)
                ones = np.ones((4,1), dtype=np.float32)
                box_pts_homo = np.concatenate([box_pts, ones], axis=1)
                new_pts = (M @ box_pts_homo.T).T
                nx, ny = new_pts[:,0].min(), new_pts[:,1].min()
                ex, ey = new_pts[:,0].max(), new_pts[:,1].max()
                nx, ny, ex, ey = int(round(nx)), int(round(ny)), int(round(ex)), int(round(ey))
                cx, cy = (nx+ex)//2, (ny+ey)//2
                size = int(max(ex-nx, ey-ny) * 裁剪系数)
                nnx, nny = max(cx-size//2, 0), max(cy-size//2, 0)
                nex, ney = min(cx+size//2, w0), min(cy+size//2, h0)
                crop_img = rotated_img[nny:ney, nnx:nex].copy()
                mask = np.zeros((h0, w0), dtype=np.uint8)
                cv2.fillConvexPoly(mask, np.int32(new_pts), 255)
                mask_crop = mask[nny:ney, nnx:nex].copy()
                crop_data = {
                    'box': [nnx, nny, nex-nnx, ney-nny],
                    'angle': angle,
                    'center': center,
                    'rotated': True
                }
            else:
                crop_img, crop_data = self.selector.crop_face(orig_img, faces[idx], crop_scale=裁剪系数, rotate=False)
                mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
                x, y, w, h = crop_data['box']
                cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                mask_crop = mask[y:y+h, x:x+w].copy()
                crop_data['angle'] = 0
                crop_data['center'] = (orig_img.shape[1]/2, orig_img.shape[0]/2)
                crop_data['rotated'] = False
            print(f"裁剪坐标: {crop_data['box']}, 旋转角度: {crop_data['angle']}")
            if crop_img.ndim == 2:
                crop_img = np.stack([crop_img]*3, axis=-1)
            elif crop_img.ndim == 3 and crop_img.shape[2] == 1:
                crop_img = np.repeat(crop_img, 3, axis=2)
            elif crop_img.ndim == 3 and crop_img.shape[2] > 3:
                crop_img = crop_img[:, :, :3]
            return (np2torch(crop_img), crop_data, mask_crop)

class 面部粘贴:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "原图": ("IMAGE",),
                "裁剪图像": ("IMAGE",),
                "裁剪数据": ("FACE_CROP_DATA",),
                "裁剪遮罩": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("输出图像",)
    FUNCTION = "run"
    CATEGORY = "Muye/面部"

    def run(self, 原图, 裁剪图像, 裁剪数据, 裁剪遮罩):
        # 支持批量粘贴
        import torch
        import numpy as np
        import collections.abc
        def is_list(obj):
            return isinstance(obj, (list, tuple))
        import torch
        import numpy as np
        def np2torch(img_np):
            arr = np.array(img_np)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            tensor = torch.from_numpy(arr).contiguous()
            return tensor.unsqueeze(0)  # (1, H, W, 3)

        def to_numpy(img):
            # 兼容各种输入格式，确保输出为(H, W, 3) uint8，值域0~255，且内容不全黑
            import numbers
            if isinstance(img, torch.Tensor):
                arr = img.detach().cpu().numpy()
                if arr.ndim == 4 and arr.shape[0] == 1:
                    arr = arr[0]
            else:
                arr = np.array(img)
            # squeeze多余的1维
            while arr.ndim > 3:
                arr = arr.squeeze(0)
            # 单通道转3通道
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            # 类型和值域归一
            if not (arr.dtype == np.uint8 or arr.dtype == np.float32 or arr.dtype == np.float64):
                arr = arr.astype(np.float32)
            # 处理全0、负值、极端异常
            if arr.max() == arr.min():
                arr = np.zeros_like(arr) + 128
            # 归一化到0~1
            if arr.max() > 1.0:
                arr = arr / arr.max()
            arr = np.clip(arr, 0, 1)
            arr = (arr * 255).round().astype(np.uint8)
            # 再次保证shape
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            return arr

        # 批量粘贴
        if is_list(裁剪图像) and is_list(裁剪数据) and is_list(裁剪遮罩):
            paste_img = to_numpy(原图).copy()
            for face_img, crop_data, mask in zip(裁剪图像, 裁剪数据, 裁剪遮罩):
                if not crop_data or not crop_data.get("box"):
                    continue
                x, y, w, h = crop_data["box"]
                angle = crop_data.get("angle", 0)
                center = crop_data.get("center", None)
                rotated = crop_data.get("rotated", False)
                face_img_np = to_numpy(face_img).copy()
                mask_np = mask.copy()
                H, W = paste_img.shape[:2]
                if rotated and angle != 0 and center is not None:
                    restored_face = np.zeros_like(paste_img)
                    restored_mask = np.zeros((H, W), dtype=np.uint8)
                    face_resized = cv2.resize(face_img_np, (w, h))
                    mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    restored_face[y:y+h, x:x+w] = face_resized
                    restored_mask[y:y+h, x:x+w] = mask_resized
                    M = cv2.getRotationMatrix2D(center, -angle, 1)
                    inv_face = cv2.warpAffine(restored_face, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    inv_mask = cv2.warpAffine(restored_mask, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
                    mask_bool = (inv_mask > 127).astype(np.uint8)
                    paste_img = paste_img * (1 - mask_bool[..., None]) + inv_face * mask_bool[..., None]
                else:
                    x1, y1 = max(x, 0), max(y, 0)
                    x2, y2 = min(x+w, W), min(y+h, H)
                    target_w, target_h = x2-x1, y2-y1
                    if target_w > 0 and target_h > 0:
                        face_resized = cv2.resize(face_img_np, (target_w, target_h))
                        mask_resized = cv2.resize(mask_np, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                        roi = paste_img[y1:y2, x1:x2]
                        mask_bool = (mask_resized > 127).astype(np.uint8)
                        for c in range(3):
                            roi[...,c] = roi[...,c] * (1-mask_bool) + face_resized[...,c] * mask_bool
                        paste_img[y1:y2, x1:x2] = roi
            return (np2torch(paste_img),)
        # 单张粘贴
        if not 裁剪数据 or not 裁剪数据.get("box"):
            return (裁剪图像,)
        x, y, w, h = 裁剪数据["box"]
        angle = 裁剪数据.get("angle", 0)
        center = 裁剪数据.get("center", None)
        rotated = 裁剪数据.get("rotated", False)
        paste_img = to_numpy(原图).copy()  # (H, W, 3)
        face_img = to_numpy(裁剪图像).copy()  # (h, w, 3)
        mask = 裁剪遮罩.copy()  # (h, w)
        H, W = paste_img.shape[:2]
        if rotated and angle != 0 and center is not None:
            restored_face = np.zeros_like(paste_img)
            restored_mask = np.zeros((H, W), dtype=np.uint8)
            face_resized = cv2.resize(face_img, (w, h))
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            restored_face[y:y+h, x:x+w] = face_resized
            restored_mask[y:y+h, x:x+w] = mask_resized
            M = cv2.getRotationMatrix2D(center, -angle, 1)
            inv_face = cv2.warpAffine(restored_face, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            inv_mask = cv2.warpAffine(restored_mask, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            mask_bool = (inv_mask > 127).astype(np.uint8)
            paste_img = paste_img * (1 - mask_bool[..., None]) + inv_face * mask_bool[..., None]
        else:
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x+w, W), min(y+h, H)
            target_w, target_h = x2-x1, y2-y1
            if target_w > 0 and target_h > 0:
                face_resized = cv2.resize(face_img, (target_w, target_h))
                mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                roi = paste_img[y1:y2, x1:x2]
                mask_bool = (mask_resized > 127).astype(np.uint8)
                for c in range(3):
                    roi[...,c] = roi[...,c] * (1-mask_bool) + face_resized[...,c] * mask_bool
                paste_img[y1:y2, x1:x2] = roi
        return (np2torch(paste_img),)

# 节点注册导出
NODE_CLASS_MAPPINGS = {
    "面部选择器": 面部选择器,
    "面部粘贴": 面部粘贴,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "面部选择器": "面部选择器",
    "面部粘贴": "面部粘贴",
}
