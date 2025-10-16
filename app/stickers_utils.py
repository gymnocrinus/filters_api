import cv2
import numpy as np
from .face_mesh_utils import get_face_landmarks, landmark_to_pixel

def _overlay_with_alpha(base_bgr, sticker_rgba, x, y, w, h):
    st = cv2.resize(sticker_rgba, (w, h), interpolation=cv2.INTER_AREA)
    if st.shape[2] == 4:
        alpha = st[:, :, 3] / 255.0
        alpha = np.stack([alpha, alpha, alpha], axis=2)
        roi = base_bgr[y:y+h, x:x+w, :]
        rgb = st[:, :, :3]
        blended = (alpha * rgb + (1 - alpha) * roi).astype(np.uint8)
        base_bgr[y:y+h, x:x+w, :] = blended
    else:
        base_bgr[y:y+h, x:x+w, :] = st[:, :, :3]
    return base_bgr

def place_sticker(image_path: str, sticker_path: str, anchor_idxs=(10, 338)):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Input image not readable.")
    h, w = img.shape[:2]

    # MediaPipe RGB ister
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    lms = get_face_landmarks(rgb)
    if not lms:
        return img  # yüz bulunamazsa orijinali döndür

    # Alın hizası için 10 ve 338 iyi anchor noktalarıdır
    x1, y1 = landmark_to_pixel(lms[anchor_idxs[0]], w, h)
    x2, y2 = landmark_to_pixel(lms[anchor_idxs[1]], w, h)
    face_w = abs(x2 - x1)

    st = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
    if st is None:
        raise ValueError(f"Sticker not found: {sticker_path}")

    sw = int(face_w * 1.5)
    sh = int(sw * st.shape[0] / st.shape[1])

    x = max(0, x1 - int(face_w * 0.25))
    y = max(0, y1 - sh)

    # Taşmaları sınırla
    if x + sw > w: sw = w - x
    if y + sh > h: sh = h - y
    if sw <= 0 or sh <= 0:
        return img

    return _overlay_with_alpha(img, st, x, y, sw, sh)
