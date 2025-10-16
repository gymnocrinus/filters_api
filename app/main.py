from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, RedirectResponse
from app.filters import (
    apply_beauty,
    apply_background_blur,
    apply_lut,
    apply_sticker,
    apply_face_morph
)
import shutil
import os

app = FastAPI(title="Render Advanced Filter API")


# Root -> /docs redirection
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


# Health check endpoint
@app.get("/healthz", tags=["meta"])
async def healthz():
    return {"status": "ok"}

def save_temp_file(file: UploadFile):
    path = f"temp_{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return path

def cleanup_file(path):
    if os.path.exists(path):
        os.remove(path)

# ------------------ Beauty Filter ------------------
@app.post("/beauty-filter")
async def beauty_filter(
    file: UploadFile = File(...),
    intensity: float = Form(0.8)
):
    path = save_temp_file(file)
    output = apply_beauty(path, intensity)
    cleanup_file(path)
    return FileResponse(output, media_type="image/jpeg")

# ------------------ Background Blur ------------------
@app.post("/background-blur")
async def background_blur(
    file: UploadFile = File(...),
    blur_strength: float = Form(0.5)
):
    path = save_temp_file(file)
    output = apply_background_blur(path, blur_strength)
    cleanup_file(path)
    return FileResponse(output, media_type="image/png")

# ------------------ LUT Filter ------------------
@app.post("/lut-filter")
async def lut_filter(
    file: UploadFile = File(...),
    filter_type: str = Form("cool")
):
    path = save_temp_file(file)
    output = apply_lut(path, filter_type)
    cleanup_file(path)
    return FileResponse(output, media_type="image/jpeg")

# ------------------ Sticker ------------------
@app.post("/sticker")
async def sticker(
    file: UploadFile = File(...),
    sticker_name: str = Form("crown")
):
    path = save_temp_file(file)
    output = apply_sticker(path, sticker_name)
    cleanup_file(path)
    return FileResponse(output, media_type="image/png")

# ------------------ Face Morph ------------------
@app.post("/face-morph")
async def face_morph(
    file: UploadFile = File(...),
    eye_size: float = Form(0.5),
    smile: float = Form(0.5),
    chin: float = Form(0.5)
):
    path = save_temp_file(file)
    output = apply_face_morph(path, eye_size, smile, chin)
    cleanup_file(path)
    return FileResponse(output, media_type="image/jpeg")
