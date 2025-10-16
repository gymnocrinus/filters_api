import os
import cv2
import numpy as np
from .face_mesh_utils import get_face_landmarks

# Sticker PNG cache
_STICKER_CACHE = {}
_DEFAULT_DIRS = [
    os.getenv("STICKERS_DIR") or "",
    "app/stickers",
    "stickers",
    "assets/stickers",
]

def _load_sticker_rgba(sticker_name: str):
    """sticker_name: 'crown' -> crown.png; returns RGBA or None"""
    if sticker_name in _STICKER_CACHE:
        return _STICKER_CACHE[sticker_name]
    file_name = f"{sticker_name}.png"
    for d in _DEFAULT_DIRS:
        if not d:
            continue
        path = os.path.join(d, file_name)
        if os.path.exists(path):
            rgba = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if rgba is not None and rgba.ndim == 3 and rgba.shape[2] in (3, 4):
                _STICKER_CACHE[sticker_name] = rgba
                return rgba
    return None

def _overlay_roi_alpha(base_bgr: np.ndarray, sticker_rgba: np.ndarray, x: int, y: int, w: int, h: int):
    """ROI üzerinde alfa blend (ara dev tampon yok)."""
    st = cv2.resize(sticker_rgba, (w, h), interpolation=cv2.INTER_AREA)
    if st.shape[2] == 4:
        roi = base_bgr[y:y+h, x:x+w, :]
        rgb = st[:, :, :3]
        alpha = (st[:, :, 3:4].astype(np.float32) / 255.0)
        blended = (alpha * rgb.astype(np.float32) + (1.0 - alpha) * roi.astype(np.float32)).astype(np.uint8)
        base_bgr[y:y+h, x:x+w, :] = blended
    else:
        base_bgr[y:y+h, x:x+w, :] = st[:, :, :3]
    return base_bgr

def _face_bbox(landmarks, w: int, h: int):
    xs = [int(l.x * w) for l in landmarks]
    ys = [int(l.y * h) for l in landmarks]
    x1, y1 = max(0, min(xs)), max(0, min(ys))
    x2, y2 = min(w - 1, max(xs)), min(h - 1, max(ys))
    return x1, y1, x2, y2

def place_sticker(img_bgr: np.ndarray, sticker_name: str = "crown") -> np.ndarray:
    """Yüz landmark'larına göre sticker yerleştirir; BGR döner."""
    h, w = img_bgr.shape[:2]
    st = _load_sticker_rgba(sticker_name)
    if st is None:
        return img_bgr

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    landmarks = get_face_landmarks(rgb)
    if not landmarks:
        return img_bgr

    x1, y1, x2, y2 = _face_bbox(landmarks, w, h)
    face_w = max(1, x2 - x1)
    sw = int(face_w * 1.5)
    sh = max(1, int(sw * st.shape[0] / st.shape[1]))

    x = max(0, x1 - int(face_w * 0.25))
    y = y1 - sh

    if x < 0:
        sw += x; x = 0
    if y < 0:
        sh += y; y = 0
    if x + sw > w:
        sw = w - x
    if y + sh > h:
        sh = h - y
    if sw <= 0 or sh <= 0:
        return img_bgr

    return _overlay_roi_alpha(img_bgr, st, x, y, sw, sh)
