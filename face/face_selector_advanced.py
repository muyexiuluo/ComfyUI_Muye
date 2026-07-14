import cv2
import numpy as np

# UniFace 替代 MediaPipe + DeepFace
try:
    from uniface import RetinaFace, FaceAnalyzer
    from uniface.attribute import AgeGender
except ImportError:
    RetinaFace = None
    print("[面部选择器] ❌ 未检测到 UniFace 库，请安装：pip install uniface")

class FaceAnalyzerUni:
    """用UniFace封装人脸检测+性别识别+角度计算"""
    def __init__(self):
        self.detector = None
        self.age_gender = None
        self.initialized = False
        try:
            if RetinaFace is None:
                print("[面部选择器] ❌ UniFace 未安装，请运行：pip install uniface")
                return
            self.detector = RetinaFace()
            self.age_gender = AgeGender()
            self.initialized = True
        except Exception as e:
            print(f"[面部选择器] ❌ UniFace 初始化失败: {e}")

    def detect_faces(self, image, min_size=50):
        """UniFace bbox 格式是 [x1,y1,x2,y2]，转换为 [x,y,w,h]"""
        if not self.initialized or self.detector is None:
            return []
        faces = []
        try:
            detections = self.detector.detect(image)
            for d in detections:
                bbox = np.array(d.bbox)
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox.astype(int)
                    bw = x2 - x1
                    bh = y2 - y1
                    if bw >= min_size and bh >= min_size:
                        gender = 'unknown'
                        try:
                            result = self.age_gender.predict(image, d)
                            if result.gender == 1:
                                gender = 'male'
                            elif result.gender == 0:
                                gender = 'female'
                        except Exception:
                            pass
                        landmarks = np.array(d.landmarks) if d.landmarks is not None else None
                        faces.append({
                            'box': [x1, y1, bw, bh],
                            'score': float(d.confidence),
                            'gender': gender,
                            'landmarks': landmarks
                        })
            print(f"[面部选择器] 检测到的脸性别: {[f['gender'] for f in faces]}")
        except Exception as e:
            print(f"[面部选择器] 检测异常: {e}")
        return faces

    def get_face_angle(self, image, face):
        """用RetinaFace的5点关键点算旋转角度
        landmarks[0]=左眼, landmarks[1]=右眼
        ±5度以内算水平，不转
        """
        if not self.initialized:
            return 0
        landmarks = face.get('landmarks')
        if landmarks is not None and len(landmarks) >= 5:
            left_eye = (float(landmarks[0, 0]), float(landmarks[0, 1]))
            right_eye = (float(landmarks[1, 0]), float(landmarks[1, 1]))
            dx = right_eye[0] - left_eye[0]
            dy = right_eye[1] - left_eye[1]
            if abs(dx) >= 5:
                angle = np.degrees(np.arctan2(dy, dx))
                if abs(angle) < 5:
                    return 0
                # dy>0:右眼偏低→cv2逆时针转=扶正
                # dy<0:右眼偏高→cv2顺时针转=扶正
                return angle
        return 0

    def sort_faces(self, faces, method='area'):
        if method == 'area':
            return sorted(faces, key=lambda f: f['box'][2]*f['box'][3], reverse=True)
        elif method == 'left':
            return sorted(faces, key=lambda f: f['box'][0])
        return faces

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


class 面部选择器:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "区分男女": (["不区分", "男", "女"], {"default": "不区分"}),
                "人物排序": (["像素占比", "从左向右"], {"default": "从左向右"}),
                "输出索引": ("STRING", {"default": "1", "multiline": False, "tooltip": "输入0输出全部，或如1 3、2,6、2，6、2。6等，支持空格/中英文逗号/句号分隔多个序号"}),
                "最小尺寸": ("INT", {"default": 50, "min": 0, "max": 5000}),
                "置信度阈值": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "裁剪系数": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 10.0}),
                "是否旋转面部": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "辅助遮罩": ("MASK", ),
            }
        }

    RETURN_TYPES = ("IMAGE", "FACE_CROP_DATA", "MASK")
    RETURN_NAMES = ("裁剪图像", "裁剪数据", "裁剪遮罩")
    FUNCTION = "run"
    CATEGORY = "Muye/面部"

    def __init__(self):
        self.analyzer = FaceAnalyzerUni()

    def run(self, 图像, 区分男女, 人物排序, 输出索引, 最小尺寸, 置信度阈值, 裁剪系数, 是否旋转面部, 辅助遮罩=None):
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

        # 2. UniFace检测+置信度过滤+最小尺寸过滤
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        faces_model = self.analyzer.detect_faces(img_bgr, min_size=最小尺寸)
        faces_model = [f for f in faces_model if f.get('score', 1.0) >= 置信度阈值]
        faces_model = [f for f in faces_model if f['box'][2] >= 最小尺寸 and f['box'][3] >= 最小尺寸]
        print(f"[面部选择器] UniFace检测到人脸数量: {len(faces_model)}")

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
            face_centers = []
            for f in faces_model:
                x, y, w, h = f['box']
                face_centers.append((x + w//2, y + h//2))
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
                    for i, pt in enumerate(region_centers):
                        if (i+1) not in found_label:
                            cx, cy = int(pt[0]), int(pt[1])
                            r = max(最小尺寸//2, 10)
                            xx = max(cx - r, 0)
                            yy = max(cy - r, 0)
                            ww = min(r*2, mask_np.shape[1]-xx)
                            hh = min(r*2, mask_np.shape[0]-yy)
                            faces_mask.append({'box': [xx, yy, ww, hh]})
            print(f"[面部选择器] 遮罩检测到人脸数量: {len(faces_mask)}")

        # 4. IoU配对
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
        for j, m in enumerate(faces_mask):
            if j not in used_mask_idx:
                faces_final.append({'box': m['box'], 'score': 1.0, 'gender': 'unknown', 'landmarks': None})

        # 5. 性别识别已在 detect_faces 时完成
        faces_before_gender = list(faces_final)

        # 6. 性别过滤
        if 区分男女 == "男":
            faces_final = [f for f in faces_final if f.get('gender', 'unknown') == 'male']
        elif 区分男女 == "女":
            faces_final = [f for f in faces_final if f.get('gender', 'unknown') == 'female']
        if (区分男女 in ("男", "女")) and (not faces_final):
            faces_final = faces_before_gender
            print(f"[面部选择器] 按性别过滤后无匹配，已回退到不区分性别的检测结果，共{len(faces_final)}个候选")

        # 7. 排序
        if 人物排序 == "像素占比":
            faces_final = self.analyzer.sort_faces(faces_final, method='area')
        else:
            faces_final = self.analyzer.sort_faces(faces_final, method='left')

        # 8. 解析输出索引
        idx_str = str(输出索引).strip()
        idx_raw_list = re.split(r'[\s,，。\.]+', idx_str)
        idx_raw_list = [s for s in idx_raw_list if s]
        idx_list = [int(s) for s in idx_raw_list if s.isdigit()]

        # 9. 输出辅助函数
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
            return (np2torch(orig_img, batch=True), {"box": None, "angle": 0}, mask_crop)

        # 10. 旋转逻辑：以人脸box中心为旋转中心
        def rotate_box_from_full(img_full, face_orig_box, angle):
            H, W = img_full.shape[:2]
            h0, w0 = H, W
            fx, fy, fw, fh = face_orig_box
            center = (fx + fw/2, fy + fh/2)
            M = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_img = cv2.warpAffine(img_full, M, (w0, h0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
            ccx = int(fx + fw/2)
            ccy = int(fy + fh/2)
            size = int(max(fw, fh) * 裁剪系数)
            nnx, nny = max(ccx - size//2, 0), max(ccy - size//2, 0)
            nex, ney = min(ccx + size//2, w0), min(ccy + size//2, h0)
            crop_img = rotated_img[nny:ney, nnx:nex].copy()
            mask_crop_full = np.zeros((h0, w0), dtype=np.uint8)
            cv2.rectangle(mask_crop_full, (nnx, nny), (nex, ney), 255, -1)
            mask_crop = mask_crop_full[nny:ney, nnx:nex].copy()
            crop_data = {
                'box': [nnx, nny, nex-nnx, ney-nny],
                'angle': angle,
                'center': center,
                'rotated': True
            }
            return crop_img, mask_crop, crop_data

        # 排序方式名称
        sort_label = "从左至右" if 人物排序 == "从左向右" else "像素占比"
        total_faces = len(faces_final)

        # 判定逻辑：0 = 输出全部
        if 0 in idx_list:
            if not faces_final:
                mask = np.ones(orig_img.shape[:2], dtype=np.uint8) * 255
                mask_crop = mask.copy()
                return (np2torch(orig_img, batch=True), {"box": None, "angle": 0}, mask_crop)
            crop_imgs, crop_datas, mask_crops = [], [], []
            rotated_info = []
            for idx_num, face in enumerate(faces_final):
                if 是否旋转面部:
                    try:
                        angle = self.analyzer.get_face_angle(img, face)
                    except Exception:
                        angle = 0
                    crop_img, mask_crop, crop_data = rotate_box_from_full(orig_img, face['box'], angle)
                    if angle != 0:
                        rotated_info.append(f"#{idx_num+1} {angle:+.1f}°")
                else:
                    x, y, w, h = expand_box(face['box'], 裁剪系数, orig_img.shape)
                    crop_img = orig_img[y:y+h, x:x+w].copy()
                    mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                    mask_crop = mask[y:y+h, x:x+w].copy()
                    center = (x + w/2, y + h/2)
                    crop_data = {'box': [x, y, w, h], 'angle': 0, 'center': center, 'rotated': False}
                if crop_img.ndim == 2:
                    crop_img = np.stack([crop_img]*3, axis=-1)
                elif crop_img.ndim == 3 and crop_img.shape[2] == 1:
                    crop_img = np.repeat(crop_img, 3, axis=2)
                elif crop_img.ndim == 3 and crop_img.shape[2] > 3:
                    crop_img = crop_img[:, :, :3]
                crop_imgs.append(np2torch(crop_img, batch=False))
                crop_datas.append(crop_data)
                mask_crops.append(mask_crop)
            # 输出日志
            output_info = f"[面部选择器] 输出 {total_faces}/{total_faces} {sort_label} 全部"
            if rotated_info:
                output_info += f" | 旋转: {', '.join(rotated_info)}"
            print(output_info)
            return (tuple(crop_imgs), tuple(crop_datas), tuple(mask_crops))

        # 多序号输出
        valid_idxs = [idx for idx in idx_list if 1 <= idx <= len(faces_final)]
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
            angle_log = ""
            if 是否旋转面部:
                try:
                    angle = self.analyzer.get_face_angle(img, face)
                except Exception:
                    angle = 0
                crop_img, mask_crop, crop_data = rotate_box_from_full(orig_img, face['box'], angle)
                if angle != 0:
                    angle_log = f" | 旋转 #{valid_idxs[0]} {angle:+.1f}°"
            if crop_img.ndim == 2:
                crop_img = np.stack([crop_img]*3, axis=-1)
            elif crop_img.ndim == 3 and crop_img.shape[2] == 1:
                crop_img = np.repeat(crop_img, 3, axis=2)
            elif crop_img.ndim == 3 and crop_img.shape[2] > 3:
                crop_img = crop_img[:, :, :3]
            print(f"[面部选择器] 输出 {len(valid_idxs)}/{total_faces} {sort_label} {valid_idxs}{angle_log}")
            return (np2torch(crop_img, batch=True), crop_data, mask_crop)
        elif len(valid_idxs) > 1:
            crop_imgs, crop_datas, mask_crops = [], [], []
            rotated_info = []
            for idx in valid_idxs:
                i = idx - 1
                face = faces_final[i]
                if 是否旋转面部:
                    try:
                        angle = self.analyzer.get_face_angle(img, face)
                    except Exception:
                        angle = 0
                    crop_img, mask_crop, crop_data = rotate_box_from_full(orig_img, face['box'], angle)
                    if angle != 0:
                        rotated_info.append(f"#{idx} {angle:+.1f}°")
                else:
                    x, y, w, h = expand_box(face['box'], 裁剪系数, orig_img.shape)
                    crop_img = orig_img[y:y+h, x:x+w].copy()
                    mask = np.zeros(orig_img.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
                    mask_crop = mask[y:y+h, x:x+w].copy()
                    center = (x + w/2, y + h/2)
                    crop_data = {'box': [x, y, w, h], 'angle': 0, 'center': center, 'rotated': False}
                if crop_img.ndim == 2:
                    crop_img = np.stack([crop_img]*3, axis=-1)
                elif crop_img.ndim == 3 and crop_img.shape[2] == 1:
                    crop_img = np.repeat(crop_img, 3, axis=2)
                elif crop_img.ndim == 3 and crop_img.shape[2] > 3:
                    crop_img = crop_img[:, :, :3]
                crop_imgs.append(np2torch(crop_img, batch=False))
                crop_datas.append(crop_data)
                mask_crops.append(mask_crop)
            output_info = f"[面部选择器] 输出 {len(valid_idxs)}/{total_faces} {sort_label} {valid_idxs}"
            if rotated_info:
                output_info += f" | 旋转: {', '.join(rotated_info)}"
            print(output_info)
            return (tuple(crop_imgs), tuple(crop_datas), tuple(mask_crops))
        else:
            mask = np.ones(orig_img.shape[:2], dtype=np.uint8) * 255
            mask_crop = mask.copy()
            return (np2torch(orig_img, batch=True), {"box": None, "angle": 0}, mask_crop)

# 节点注册导出（名称改为"面部选择器"，原高级节点的class名也改）
NODE_CLASS_MAPPINGS = {
    "面部选择器": 面部选择器,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "面部选择器": "面部选择器",
}