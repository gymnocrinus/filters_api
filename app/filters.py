import os
import cv2
import numpy as np
from rembg import remove

# Landmark tabanlı yerleştirmeyi yapan yardımcı modül
from .stickers_utils import place_sticker


def _safe_out_path(image_path: str, suffix: str, force_ext: str = None) -> str:
    base, ext = os.path.splitext(image_path)
    if force_ext:
        return f"{base}{suffix}{force_ext}"
    return f"{base}{suffix}{ext or '.jpg'}"


# -------------------------------
# 1) Beauty / Skin Smoothing
# -------------------------------
def apply_beauty(image_path: str, intensity: float = 0.8) -> str:
    """
    intensity: 0.0–1.0  (varsayılan 0.8)
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("apply_beauty: input image not readable")

    # d (kernel çapı) tek sayı olmalı; çok küçük değerlerde etkisiz kalmasın diye min 5
    d = max(5, int(9 * max(0.1, float(intensity))))
    if d % 2 == 0:
        d += 1

    # Sigma değerlerini de intensity ile biraz yükseltelim
    smooth = cv2.bilateralFilter(img, d, 75 + 50 * intensity, 75 + 50 * intensity)

    out_path = _safe_out_path(image_path, "_beauty", ".jpg")
    cv2.imwrite(out_path, smooth)
    return out_path


# -------------------------------
# 2) Background Blur (Portre tarzı)
# -------------------------------
def apply_background_blur(image_path: str, blur_strength: float = 0.5) -> str:
    """
    blur_strength: 0.0–1.0  (arka plan ne kadar bulanık)
    Ön planı keskin tutar, yalnızca arka planı bulanıklaştırır.
    """
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("apply_background_blur: input image not readable")

    # rembg ile alfa maskeyi çıkar (RGBA döner)
    rgba_bytes = remove(open(image_path, "rb").read())
    rgba = cv2.imdecode(np.frombuffer(rgba_bytes, np.uint8), cv2.IMREAD_UNCHANGED)

    # Maskeyi çıkaramazsak tüm kareyi blur'layıp döneriz (fallback)
    if rgba is None or rgba.shape[2] < 4:
        k = max(3, int(25 * max(0.0, min(1.0, blur_strength))) | 1)
        blurred_full = cv2.GaussianBlur(bgr, (k, k), 0)
        out_path = _safe_out_path(image_path, "_bgblur", ".jpg")
        cv2.imwrite(out_path, blurred_full)
        return out_path

    alpha = (rgba[:, :, 3].astype(np.float32) / 255.0)  # 0–1
    alpha3 = np.dstack([alpha, alpha, alpha])

    k = max(3, int(25 * max(0.0, min(1.0, blur_strength))) | 1)
    bg_blurred = cv2.GaussianBlur(bgr, (k, k), 0)

    # Ön planı keskin, arka planı blur kompoziti
    comp = (alpha3 * bgr + (1.0 - alpha3) * bg_blurred).astype(np.uint8)

    out_path = _safe_out_path(image_path, "_bgblur", ".jpg")
    cv2.imwrite(out_path, comp)
    return out_path


# -------------------------------
# 3) LUT / Renk Efektleri
# -------------------------------
def apply_lut(image_path: str, filter_type: str = "cool") -> str:
    """
    filter_type: "cool" | "warm" | "cinematic"
    - Eğer app/lut_filters/{filter_type}.png  (256x1 veya 1x256, 3 kanallı) bulunursa onu LUT olarak dener.
    - Aksi halde OpenCV colormap fallback uygular.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("apply_lut: input image not readable")

    lut_path = f"app/lut_filters/{filter_type}.png"
    result = None

    if os.path.exists(lut_path):
        lut_img = cv2.imread(lut_path, cv2.IMREAD_COLOR)
        # 1D LUT formu: 256x1x3 veya 1x256x3
        if lut_img is not None and (
            (lut_img.shape[0] == 256 and lut_img.shape[2] == 3 and lut_img.shape[1] in (1, 256)) or
            (lut_img.shape[1] == 256 and lut_img.shape[2] == 3 and lut_img.shape[0] in (1, 256))
        ):
            lut = lut_img.reshape((256, 1, 3))
            result = cv2.LUT(img, lut)

    if result is None:
        # Fallback: Colormap + ufak grading
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if filter_type == "cool":
            cm = cv2.COLORMAP_OCEAN
        elif filter_type == "warm":
            cm = cv2.COLORMAP_AUTUMN
        else:  # "cinematic" varsayılan
            cm = cv2.COLORMAP_CIVIDIS

        mapped = cv2.applyColorMap(gray, cm)

        if filter_type == "cinematic":
            # basit kontrast/teal grading
            mapped = cv2.convertScaleAbs(mapped, alpha=1.15, beta=0)
            b, g, r = cv2.split(mapped)
            b = cv2.addWeighted(b, 1.08, g, 0.0, 0)
            r = cv2.addWeighted(r, 0.95, g, 0.0, 0)
            mapped = cv2.merge([b, g, r])

        result = mapped

    out_path = _safe_out_path(image_path, f"_lut_{filter_type}", ".jpg")
    cv2.imwrite(out_path, result)
    return out_path


# -------------------------------
# 4) Sticker / Overlay (landmark tabanlı)
# -------------------------------
def apply_sticker(image_path: str, sticker_name: str = "crown") -> str:
    """
    sticker_name: app/stickers/{sticker_name}.png dosyasını arar.
    Yerleştirme ve ölçekleme place_sticker() ile otomatik yapılır.
    """
    sticker_path = f"app/stickers/{sticker_name}.png"
    result_bgr = place_sticker(image_path, sticker_path)  # BGR döner

    out_path = _safe_out_path(image_path, f"_sticker_{sticker_name}", ".png")
    cv2.imwrite(out_path, result_bgr)
    return out_path


# -------------------------------
# 5) Face Morph (placeholder)
# -------------------------------
def apply_face_morph(image_path: str, eye_size: float = 0.5, smile: float = 0.5, chin: float = 0.5) -> str:
    """
    Not: Morph örneği placeholder. Landmark tabanlı warping eklenecek alan.
    Parametreler: 0.0–1.0 aralığında beklenir.
    """
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("apply_face_morph: input image not readable")

    # TODO: eye_size / smile / chin ile Delaunay + piecewise affine morph
    # Şimdilik orijinali yazıyoruz:
    out_path = _safe_out_path(image_path, "_morph", ".jpg")
    cv2.imwrite(out_path, img)
    return out_path
