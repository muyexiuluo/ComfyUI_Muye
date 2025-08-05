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
        return faces

    def _estimate_gender(self, face_img):
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
            crop_img = self._rotate_image(crop_img, angle)
        crop_data = {
            'box': [nx, ny, ex-nx, ey-ny],
            'angle': angle
        }
        return crop_img, crop_data

    def _get_face_angle(self, image, face):
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

import comfy.sd

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def extract_mask_faces(mask, min_size=50, min_area=100):
    mask_bin = (mask > 127).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    faces = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= min_size and h >= min_size and w*h >= min_area:
            faces.append({'box': [x, y, w, h]})
    return faces

class 面部选择器高级:
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
            },
            "optional": {
                "辅助遮罩": ("MASK", ),
                "deepface_model": ([
                    "VGG-Face",
                    "ArcFace",
                    "Facenet",
                    "Facenet512",
                    "OpenFace",
                    "DeepFace",
                    "DeepID",
                    "Dlib",
                    "SFace"
                ], {"default": "VGG-Face"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "FACE_CROP_DATA", "MASK")
    RETURN_NAMES = ("裁剪图像", "裁剪数据", "裁剪遮罩")
    FUNCTION = "run"
    CATEGORY = "Muye/面部"

    def __init__(self):
        self.selector = FaceSelector()

    def run(self, 图像, 区分男女, 人物排序, 输出索引, 最小尺寸, 置信度阈值, 裁剪系数, 是否旋转面部, 辅助遮罩=None, deepface_model="VGG-Face"):
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

        img = orig_img
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        faces_model = self.selector.detect_faces(img, min_size=最小尺寸)
        faces_model = [f for f in faces_model if f.get('score', 1.0) >= 置信度阈值]
        print(f"[高级面部选择器] MediaPipe检测到人脸数量: {len(faces_model)}")

        # 先排序
        if 人物排序 == "像素占比":
            faces_model = self.selector.sort_faces(faces_model, method='area')
        else:
            faces_model = self.selector.sort_faces(faces_model, method='left')

        # DeepFace性别识别（始终执行，先不做性别过滤）
        deepface_available = False
        try:
            from deepface import DeepFace
            deepface_has_model_arg = False
            import inspect
            if 'model_name' in inspect.signature(DeepFace.analyze).parameters:
                deepface_has_model_arg = True
            print(f"[高级面部选择器] 检测到DeepFace，支持模型选择: {deepface_has_model_arg}，当前模型: {deepface_model}")
            deepface_available = True
        except ImportError:
            print("[高级面部选择器] 未安装deepface，性别识别将被忽略")
        except Exception as e:
            print(f"[高级面部选择器] DeepFace加载异常: {e}")
        if deepface_available:
            for f in faces_model:
                x, y, w, h = f['box']
                face_img = img[y:y+h, x:x+w]
                try:
                    analyze_sig = inspect.signature(DeepFace.analyze)
                    if 'model_name' in analyze_sig.parameters:
                        result = DeepFace.analyze(face_img, actions=['gender'], enforce_detection=False, model_name=deepface_model)
                    elif 'model' in analyze_sig.parameters:
                        result = DeepFace.analyze(face_img, actions=['gender'], enforce_detection=False, model=deepface_model)
                    else:
                        result = DeepFace.analyze(face_img, actions=['gender'], enforce_detection=False)
                    if isinstance(result, list):
                        if result:
                            gender = result[0].get('gender', 'unknown')
                        else:
                            gender = 'unknown'
                    elif isinstance(result, dict):
                        gender = result.get('gender', 'unknown')
                    else:
                        gender = 'unknown'
                    if isinstance(gender, dict):
                        gender_str = max(gender.items(), key=lambda x: x[1])[0].lower()
                    else:
                        gender_str = str(gender).lower()
                    if gender_str.startswith('m'):
                        f['gender'] = 'male'
                    elif gender_str.startswith('w') or gender_str.startswith('f'):
                        f['gender'] = 'female'
                    else:
                        f['gender'] = 'unknown'
                except Exception as e:
                    print(f"[DeepFace] 性别识别失败: {e}")

        # 性别过滤在排序和识别后执行
        faces_model_filtered = faces_model
        if 区分男女 == "男":
            faces_model_filtered = [f for f in faces_model if f.get('gender', 'unknown') == 'male']
        elif 区分男女 == "女":
            faces_model_filtered = [f for f in faces_model if f.get('gender', 'unknown') == 'female']
        else:
            faces_model_filtered = faces_model
        faces_mask = []
        if 辅助遮罩 is not None:
            mask = 辅助遮罩
            import torch
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.array(mask)
            # squeeze到2维 (H, W)
            while mask_np.ndim > 2:
                mask_np = np.squeeze(mask_np, axis=0)
            # 归一化到0~255 uint8
            if mask_np.max() <= 1.0:
                mask_np = (mask_np * 255).round().astype(np.uint8)
            else:
                mask_np = mask_np.astype(np.uint8)
            if mask_np.ndim != 2:
                raise ValueError(f"辅助遮罩格式错误，shape={mask_np.shape}，请提供单通道mask")
            faces_mask = extract_mask_faces(mask_np, min_size=最小尺寸)
            print(f"[高级面部选择器] 遮罩检测到人脸数量: {len(faces_mask)}")
        else:
            print(f"[高级面部选择器] 未提供辅助遮罩")

        # 融合：每个模型box与mask区域配对，配对后mask不再参与，未配对mask补充输出
        used_mask_idx = set()
        faces_final = []
        pair_count = 0
        only_model_count = 0
        only_mask_count = 0
        # 1. 先尝试配对
        for i, f in enumerate(faces_model_filtered):
            best_iou = 0
            best_idx = -1
            for j, m in enumerate(faces_mask):
                if j in used_mask_idx:
                    continue
                iou = compute_iou(f['box'], m['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_iou > 0.5 and best_idx >= 0:
                faces_final.append({**f, 'box': faces_mask[best_idx]['box'], 'iou': best_iou, 'source': 'pair'})
                used_mask_idx.add(best_idx)
                pair_count += 1
            else:
                faces_final.append({**f, 'iou': best_iou, 'source': 'model_only'})
                only_model_count += 1
        # 2. 补充未被配对的mask区域
        for j, m in enumerate(faces_mask):
            if j not in used_mask_idx:
                faces_final.append({'box': m['box'], 'score': 1.0, 'gender': 'unknown', 'iou': 0, 'source': 'mask_only'})
                only_mask_count += 1

        # 打印最终输出信息
        if 输出索引 == 0:
            out_idx_str = "所有"
        else:
            out_idx_str = str(输出索引)
        sort_str = "像素占比" if 人物排序 == "像素占比" else "从左向右"
        gender_str = f"，{区分男女}" if 区分男女 != "不区分" else ""
        if faces_final:
            if 输出索引 == 0:
                sizes = [f['box'][2:4] for f in faces_final]
                size_str = ", ".join([f"{w}x{h}" for w, h in sizes])
            else:
                idx = min(输出索引-1, len(faces_final)-1)
                w, h = faces_final[idx]['box'][2:4]
                size_str = f"{w}x{h}"
        else:
            size_str = "无输出"
        print(f"[高级面部选择器] 最终人脸输出：{out_idx_str}，{sort_str}{gender_str}，像素大小：{size_str}")

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
                return tensor
            else:
                return tensor.unsqueeze(0)

        # 输出逻辑同原版
        if 输出索引 == 0 and faces_final:
            crop_imgs, crop_datas, mask_crops = [], [], []
            for idx, face in enumerate(faces_final):
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
                    if face.get('mask_contour') is not None:
                        cv2.drawContours(mask, [face['mask_contour']], -1, 255, -1)
                    else:
                        x, y, w, h = crop_data['box']
                        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                    x, y, w, h = crop_data['box']
                    mask_crop = mask[y:y+h, x:x+w].copy()
                    crop_data['angle'] = 0
                    crop_data['center'] = (orig_img.shape[1]/2, orig_img.shape[0]/2)
                    crop_data['rotated'] = False
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
        elif not faces_final:
            mask = np.ones(orig_img.shape[:2], dtype=np.uint8) * 255
            mask_crop = mask.copy()
            return (np2torch(orig_img), {"box": None, "angle": 0}, mask_crop)
        else:
            idx = min(输出索引-1, len(faces_final)-1)
            face = faces_final[idx]
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
                if face.get('mask_contour') is not None:
                    cv2.drawContours(mask, [face['mask_contour']], -1, 255, -1)
                else:
                    x, y, w, h = crop_data['box']
                    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                x, y, w, h = crop_data['box']
                mask_crop = mask[y:y+h, x:x+w].copy()
                crop_data['angle'] = 0
                crop_data['center'] = (orig_img.shape[1]/2, orig_img.shape[0]/2)
                crop_data['rotated'] = False
            if crop_img.ndim == 2:
                crop_img = np.stack([crop_img]*3, axis=-1)
            elif crop_img.ndim == 3 and crop_img.shape[2] == 1:
                crop_img = np.repeat(crop_img, 3, axis=2)
            elif crop_img.ndim == 3 and crop_img.shape[2] > 3:
                crop_img = crop_img[:, :, :3]
            return (np2torch(crop_img), crop_data, mask_crop)

# 节点注册导出
NODE_CLASS_MAPPINGS = {
    "面部选择器高级": 面部选择器高级,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "面部选择器高级": "面部选择器（高级）",
}
