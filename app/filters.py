import os
import cv2
import numpy as np
from typing import Optional
from rembg import remove, new_session
from .stickers_utils import place_sticker

# rembg oturumu (tek sefer)
SESSION = new_session()

# Büyük görsellerde pik RAM'i düşürmek için uzun kenarı sınırla
MAX_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "2048"))

def _safe_out_path(image_path: str, suffix: str, force_ext: Optional[str] = None) -> str:
    base, ext = os.path.splitext(image_path)
    if force_ext:
        return f"{base}{suffix}{force_ext}"
    return f"{base}{suffix}{ext or '.jpg'}"

def _downscale_if_needed(img: np.ndarray, max_side: int = MAX_SIDE) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

# -------- 1) Beauty
def apply_beauty(image_path: str, intensity: float = 0.8) -> str:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("apply_beauty: input image not readable")
    img = _downscale_if_needed(img)

    d = max(5, int(9 * max(0.1, float(intensity))))
    if d % 2 == 0:
        d += 1
    sigma = 75 + 50 * intensity
    smooth = cv2.bilateralFilter(img, d, sigma, sigma)

    out_path = _safe_out_path(image_path, "_beauty", ".jpg")
    cv2.imwrite(out_path, smooth)
    return out_path

# -------- 2) Background Blur (downscale + rembg aynı boyutta + copyTo)
def apply_background_blur(image_path: str, blur_strength: float = 0.5) -> str:
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("apply_background_blur: input image not readable")
    bgr = _downscale_if_needed(bgr)

    ok, enc = cv2.imencode(".png", bgr)  # rembg'ye bgr'nin aynısını ver
    if not ok:
        raise ValueError("apply_background_blur: encode failed before rembg")
    rgba_bytes = remove(enc.tobytes(), session=SESSION)
    rgba = cv2.imdecode(np.frombuffer(rgba_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

    k = max(3, int(25 * max(0.0, min(1.0, blur_strength))) | 1)
    bg_blurred = cv2.GaussianBlur(bgr, (k, k), 0)

    if rgba is None or rgba.ndim < 3 or rgba.shape[2] < 4:
        out_path = _safe_out_path(image_path, "_bgblur", ".jpg")
        cv2.imwrite(out_path, bg_blurred)
        return out_path

    mask = rgba[:, :, 3]
    if mask.shape[:2] != bgr.shape[:2]:
        mask = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    comp = bg_blurred.copy()
    cv2.copyTo(bgr, mask, comp)

    out_path = _safe_out_path(image_path, "_bgblur", ".jpg")
    cv2.imwrite(out_path, comp)
    return out_path

# -------- 3) LUT / Renk filtreleri
def _apply_sepia(bgr: np.ndarray) -> np.ndarray:
    kernel = np.array([[0.131, 0.534, 0.272],
                       [0.168, 0.686, 0.349],
                       [0.189, 0.769, 0.393]], dtype=np.float32)
    out = cv2.transform(bgr, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)

def apply_lut(image_path: str, filter_type: str = "cool") -> str:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("apply_lut: input image not readable")
    img = _downscale_if_needed(img)

    ft = (filter_type or "cool").lower()
    if ft == "cool":
        b, g, r = cv2.split(img)
        b = cv2.addWeighted(b, 1.1, b, 0, 0)
        r = cv2.addWeighted(r, 0.9, r, 0, 0)
        out = cv2.merge([b, g, r])
    elif ft == "warm":
        b, g, r = cv2.split(img)
        r = cv2.addWeighted(r, 1.1, r, 0, 0)
        b = cv2.addWeighted(b, 0.9, b, 0, 0)
        out = cv2.merge([b, g, r])
    elif ft == "sepia":
        out = _apply_sepia(img)
    elif ft in {"bw", "b&w", "mono", "grayscale"}:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        out = img

    out_path = _safe_out_path(image_path, f"_{ft}", ".jpg")
    cv2.imwrite(out_path, out)
    return out_path

# -------- 4) Sticker
def apply_sticker(image_path: str, sticker_name: str = "crown") -> str:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("apply_sticker: input image not readable")
    img = _downscale_if_needed(img)

    composed = place_sticker(img, sticker_name)
    out_path = _safe_out_path(image_path, f"_{sticker_name}", ".png")
    cv2.imwrite(out_path, composed, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    return out_path

# -------- 5) Face Morph (placeholder)
def apply_face_morph(image_path: str, eye_size: float = 0.5, smile: float = 0.5, chin: float = 0.5) -> str:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("apply_face_morph: input image not readable")
    img = _downscale_if_needed(img)
    out_path = _safe_out_path(image_path, "_morph", ".jpg")
    cv2.imwrite(out_path, img)
    return out_path
