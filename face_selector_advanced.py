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
    return faces, contours


class 面部选择器高级:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "区分男女": (["不区分", "男", "女"], {"default": "不区分"}),
                "人物排序": (["像素占比", "从左向右"], {"default": "从左向右"}),
                # 修改为字符串类型，允许多序号输入
                "输出索引": ("STRING", {"default": "1", "multiline": False, "tooltip": "输入0输出全部，或如1 3、2,6、2，6、2。6等，支持空格/中英文逗号/句号分隔多个序号"}),
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
        import torch
        import re
        # 1. 预处理原图
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

        # 2. MediaPipe检测+置信度过滤+最小尺寸过滤
        faces_model = self.selector.detect_faces(img, min_size=最小尺寸)
        faces_model = [f for f in faces_model if f.get('score', 1.0) >= 置信度阈值]
        faces_model = [f for f in faces_model if f['box'][2] >= 最小尺寸 and f['box'][3] >= 最小尺寸]
        print(f"[高级面部选择器] MediaPipe检测到人脸数量: {len(faces_model)}")

        # 3. 遮罩box提取
        faces_mask = []
        if 辅助遮罩 is not None:
            mask = 辅助遮罩
            if isinstance(mask, torch.Tensor):
                mask_np = mask.detach().cpu().numpy()
            else:
                mask_np = np.array(mask)
            while mask_np.ndim > 2:
                mask_np = np.squeeze(mask_np, axis=0)
            if mask_np.max() <= 1.0:
                mask_np = (mask_np * 255).round().astype(np.uint8)
            else:
                mask_np = mask_np.astype(np.uint8)
            if mask_np.ndim != 2:
                raise ValueError(f"辅助遮罩格式错误，shape={mask_np.shape}，请提供单通道mask")
            # 先检测人脸中心点
            face_centers = []
            for f in faces_model:
                x, y, w, h = f['box']
                face_centers.append((x + w//2, y + h//2))
            # 获取所有遮罩大连通区域
            _, contours = extract_mask_faces(mask_np, min_size=最小尺寸)
            try:
                from skimage.segmentation import watershed
                from scipy import ndimage as ndi
            except ImportError:
                watershed = None
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                region_centers = [pt for pt in face_centers if x <= pt[0] <= x+w and y <= pt[1] <= y+h]
                if len(region_centers) <= 1 or watershed is None:
                    if w >= 最小尺寸 and h >= 最小尺寸 and w*h >= 100:
                        faces_mask.append({'box': [x, y, w, h]})
                else:
                    # 最短距离分割（watershed）
                    mask_region = np.zeros_like(mask_np)
                    cv2.drawContours(mask_region, [cnt], -1, 255, -1)
                    mask_region = (mask_region > 127).astype(np.uint8)
                    markers = np.zeros_like(mask_region, dtype=np.int32)
                    for i, pt in enumerate(region_centers):
                        cx, cy = int(pt[0]), int(pt[1])
                        if 0 <= cy < markers.shape[0] and 0 <= cx < markers.shape[1]:
                            markers[cy, cx] = i+1
                    distance = ndi.distance_transform_edt(mask_region)
                    labels = watershed(-distance, markers, mask=mask_region)
                    found_label = set()
                    for i in range(1, len(region_centers)+1):
                        mask_i = (labels == i).astype(np.uint8)
                        if np.sum(mask_i) == 0:
                            continue
                        found_label.add(i)
                        sub_contours, _ = cv2.findContours(mask_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for scnt in sub_contours:
                            xx, yy, ww, hh = cv2.boundingRect(scnt)
                            if ww >= 5 and hh >= 5 and ww*hh >= 10:
                                faces_mask.append({'box': [xx, yy, ww, hh]})
                    # 兜底：如有中心点未分出mask，则补一个小box
                    for i, pt in enumerate(region_centers):
                        if (i+1) not in found_label:
                            cx, cy = int(pt[0]), int(pt[1])
                            r = max(最小尺寸//2, 10)
                            xx = max(cx - r, 0)
                            yy = max(cy - r, 0)
                            ww = min(r*2, mask_np.shape[1]-xx)
                            hh = min(r*2, mask_np.shape[0]-yy)
                            faces_mask.append({'box': [xx, yy, ww, hh]})
            print(f"[高级面部选择器] 遮罩检测到人脸数量: {len(faces_mask)}")
        else:
            print(f"[高级面部选择器] 未提供辅助遮罩")

        # 4. IoU配对，重合用mask box，不重合补mask box，所有box都用原始box
        used_mask_idx = set()
        faces_final = []
        for i, f in enumerate(faces_model):
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
                box = faces_mask[best_idx]['box']
                used_mask_idx.add(best_idx)
            else:
                box = f['box']
            faces_final.append({**f, 'box': box})
        # 补充未配对的mask box
        for j, m in enumerate(faces_mask):
            if j not in used_mask_idx:
                faces_final.append({'box': m['box'], 'score': 1.0, 'gender': 'unknown'})

        # 5. 性别识别（DeepFace优先）
        deepface_available = False
        try:
            from deepface import DeepFace
            deepface_has_model_arg = False
            import inspect
            if 'model_name' in inspect.signature(DeepFace.analyze).parameters:
                deepface_has_model_arg = True
            deepface_available = True
        except ImportError:
            pass
        except Exception as e:
            pass
        if deepface_available:
            for f in faces_final:
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
                    pass
        else:
            for f in faces_final:
                if 'gender' not in f:
                    x, y, w, h = f['box']
                    face_img = img[y:y+h, x:x+w]
                    mean = np.mean(face_img) if face_img.size > 0 else 0
                    f['gender'] = 'male' if mean < 120 else 'female'

        # 6. 性别过滤
        # 先保存一份未过滤的列表以便回退
        faces_before_gender = list(faces_final)
        if 区分男女 == "男":
            faces_final = [f for f in faces_final if f.get('gender', 'unknown') == 'male']
        elif 区分男女 == "女":
            faces_final = [f for f in faces_final if f.get('gender', 'unknown') == 'female']
        # 如果按性别过滤后为空，则回退到不区分性别的列表（避免输出像素条）
        if (区分男女 in ("男", "女")) and (not faces_final):
            # 检查是否多数为 unknown，如果是也回退
            faces_final = faces_before_gender
            print(f"[高级面部选择器] 按性别过滤后无匹配，已回退到不区分性别的检测结果，共{len(faces_final)}个候选")
        # 7. 排序
        if 人物排序 == "像素占比":
            faces_final = self.selector.sort_faces(faces_final, method='area')
        else:
            faces_final = self.selector.sort_faces(faces_final, method='left')

        # 8. 解析输出索引
        idx_str = str(输出索引).strip()
        idx_raw_list = re.split(r'[\s,，。\.]+', idx_str)
        idx_raw_list = [s for s in idx_raw_list if s]
        idx_list = [int(s) for s in idx_raw_list if s.isdigit()]

        # 9. 输出
        def np2torch(img_np, batch=False):
            arr = np.array(img_np)
            if arr.ndim == 2:
                arr = np.stack([arr]*3, axis=-1)
            if arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.repeat(arr, 3, axis=2)
            if arr.ndim == 3 and arr.shape[2] > 3:
                arr = arr[:, :, :3]
            while arr.ndim > 3:
                arr = arr[0]
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            tensor = torch.from_numpy(arr).contiguous()
            if batch:
                return tensor.unsqueeze(0)
            else:
                return tensor

        def expand_box(box, scale, img_shape):
            x, y, w, h = box
            cx, cy = x + w//2, y + h//2
            size = int(max(w, h) * scale)
            nx, ny = max(cx - size//2, 0), max(cy - size//2, 0)
            ex, ey = min(cx + size//2, img_shape[1]), min(cy + size//2, img_shape[0])
            return [nx, ny, ex-nx, ey-ny]

        if not faces_final:
            mask = np.ones(orig_img.shape[:2], dtype=np.uint8) * 255
            mask_crop = mask.copy()
            return (np2torch(orig_img), {"box": None, "angle": 0}, mask_crop)

        # 10. 旋转逻辑（如需）
        # 为避免对小crop做边界镜像而产生缝隙，旋转在原图上执行，然后再裁剪出扩张后的box，
        # 这样扩张像素来自原图而不是crop的镜像填充，粘贴回去时更自然。
        def rotate_box_from_full(img_full, box, angle):
            # img_full: 原始整图；box: [x,y,w,h] 已经是扩张后的box（在原图坐标系）
            H, W = img_full.shape[:2]
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0
            M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
            # 在整图上旋转，使用常数填充(0)，并同时生成有效像素mask
            rotated_full = cv2.warpAffine(img_full, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            full_mask = np.ones((H, W), dtype=np.uint8) * 255
            rotated_mask_full = cv2.warpAffine(full_mask, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            x, y, w, h = box
            x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
            x0 = max(x0, 0); y0 = max(y0, 0); x1 = min(x1, W); y1 = min(y1, H)
            crop = rotated_full[y0:y1, x0:x1].copy()
            crop_mask = rotated_mask_full[y0:y1, x0:x1].copy()
            # 返回裁剪图和对应的mask，调用处需要使用mask_crop
            return crop, crop_mask

        # 判定逻辑：只要有0（且不是10、20等），输出全部
        if 0 in idx_list:
            if not faces_final:
                mask = np.ones(orig_img.shape[:2], dtype=np.uint8) * 255
                mask_crop = mask.copy()
                return (np2torch(orig_img), {"box": None, "angle": 0}, mask_crop)
            crop_imgs, crop_datas, mask_crops = [], [], []
            for face in faces_final:
                x, y, w, h = expand_box(face['box'], 裁剪系数, orig_img.shape)
                crop_img = orig_img[y:y+h, x:x+w].copy()
                mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                mask_crop = mask[y:y+h, x:x+w].copy()
                center = (x + w/2, y + h/2)
                crop_data = {'box': [x, y, w, h], 'angle': 0, 'center': center, 'rotated': False}
                if 是否旋转面部:
                    try:
                        angle = self.selector._get_face_angle(img, face)
                    except Exception:
                        angle = 0
                    # 在原图上旋转再裁剪，保证扩张像素来自原图
                    crop_img, crop_mask = rotate_box_from_full(orig_img, [x, y, w, h], angle)
                    # 如果rotate_box_from_full返回的mask为空，保留之前的mask_crop
                    if crop_mask is not None and crop_mask.size:
                        mask_crop = crop_mask
                    crop_data['angle'] = angle
                    crop_data['rotated'] = True if angle != 0 else False
                if crop_img.ndim == 2:
                    crop_img = np.stack([crop_img]*3, axis=-1)
                elif crop_img.ndim == 3 and crop_img.shape[2] == 1:
                    crop_img = np.repeat(crop_img, 3, axis=2)
                elif crop_img.ndim == 3 and crop_img.shape[2] > 3:
                    crop_img = crop_img[:, :, :3]
                crop_imgs.append(np2torch(crop_img, batch=False))
                crop_datas.append(crop_data)
                mask_crops.append(mask_crop)
            return (tuple(crop_imgs), tuple(crop_datas), tuple(mask_crops))

        # 多序号输出
        valid_idxs = [idx for idx in idx_list if 1 <= idx <= len(faces_final)]
        # 如果用户提供了索引但所有索引都超出已检测到的人脸数量，退化为输出当前检测到的所有人脸（基于已过滤的 faces_final）
        if idx_list and (not valid_idxs):
            valid_idxs = list(range(1, len(faces_final) + 1))
        if len(valid_idxs) == 1:
            idx = valid_idxs[0] - 1
            face = faces_final[idx]
            x, y, w, h = expand_box(face['box'], 裁剪系数, orig_img.shape)
            crop_img = orig_img[y:y+h, x:x+w].copy()
            mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            mask_crop = mask[y:y+h, x:x+w].copy()
            center = (x + w/2, y + h/2)
            crop_data = {'box': [x, y, w, h], 'angle': 0, 'center': center, 'rotated': False}
            if 是否旋转面部:
                try:
                    angle = self.selector._get_face_angle(img, face)
                except Exception:
                    angle = 0
                crop_img, crop_mask = rotate_box_from_full(orig_img, [x, y, w, h], angle)
                if crop_mask is not None and crop_mask.size:
                    mask_crop = crop_mask
                crop_data['angle'] = angle
                crop_data['rotated'] = True if angle != 0 else False
            if crop_img.ndim == 2:
                crop_img = np.stack([crop_img]*3, axis=-1)
            elif crop_img.ndim == 3 and crop_img.shape[2] == 1:
                crop_img = np.repeat(crop_img, 3, axis=2)
            elif crop_img.ndim == 3 and crop_img.shape[2] > 3:
                crop_img = crop_img[:, :, :3]
            return (np2torch(crop_img, batch=True), crop_data, mask_crop)
        elif len(valid_idxs) > 1:
            crop_imgs, crop_datas, mask_crops = [], [], []
            for idx in valid_idxs:
                i = idx - 1
                face = faces_final[i]
                x, y, w, h = expand_box(face['box'], 裁剪系数, orig_img.shape)
                crop_img = orig_img[y:y+h, x:x+w].copy()
                mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                mask_crop = mask[y:y+h, x:x+w].copy()
                center = (x + w/2, y + h/2)
                crop_data = {'box': [x, y, w, h], 'angle': 0, 'center': center, 'rotated': False}
                if 是否旋转面部:
                    try:
                        angle = self.selector._get_face_angle(img, face)
                    except Exception:
                        angle = 0
                    crop_img, crop_mask = rotate_box_from_full(orig_img, [x, y, w, h], angle)
                    if crop_mask is not None and crop_mask.size:
                        mask_crop = crop_mask
                    crop_data['angle'] = angle
                    crop_data['rotated'] = True if angle != 0 else False
                if crop_img.ndim == 2:
                    crop_img = np.stack([crop_img]*3, axis=-1)
                elif crop_img.ndim == 3 and crop_img.shape[2] == 1:
                    crop_img = np.repeat(crop_img, 3, axis=2)
                elif crop_img.ndim == 3 and crop_img.shape[2] > 3:
                    crop_img = crop_img[:, :, :3]
                crop_imgs.append(np2torch(crop_img, batch=False))
                crop_datas.append(crop_data)
                mask_crops.append(mask_crop)
            return (tuple(crop_imgs), tuple(crop_datas), tuple(mask_crops))
        else:
            mask = np.ones(orig_img.shape[:2], dtype=np.uint8) * 255
            mask_crop = mask.copy()
            return (np2torch(orig_img), {"box": None, "angle": 0}, mask_crop)

# 节点注册导出
NODE_CLASS_MAPPINGS = {
    "面部选择器高级": 面部选择器高级,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "面部选择器高级": "面部选择器（高级）",
}
