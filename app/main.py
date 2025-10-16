import os
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Request
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse

# Aynı paket içindeki modülden relative import:
from .filters import (
    apply_beauty,
    apply_background_blur,
    apply_lut,
    apply_sticker,
    apply_face_morph,
)

app = FastAPI(title="Render Advanced Filter API")

# ---------- Lifecycle ----------
@app.on_event("startup")
async def _startup():
    app.state.sem = asyncio.Semaphore(int(os.getenv("MAX_CONCURRENCY", "2")))

# ---------- Helpers ----------
def save_temp_file(file: UploadFile) -> str:
    path = f"temp_{file.filename}"
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return path

def cleanup_file(path: str) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except Exception:
            pass

# ---------- Root & Health ----------
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

@app.get("/healthz")
async def healthz():
    return JSONResponse({"status": "ok"})

# ---------- Beauty ----------
@app.post("/beauty-filter")
async def beauty_filter(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    intensity: float = Form(0.8),
):
    async with request.app.state.sem:
        path = save_temp_file(file)
        try:
            output = apply_beauty(path, intensity)
        finally:
            cleanup_file(path)
        background_tasks.add_task(cleanup_file, output)
        return FileResponse(output, media_type="image/jpeg")

# ---------- Background Blur ----------
@app.post("/background-blur")
async def background_blur(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    blur_strength: float = Form(0.5),
):
    async with request.app.state.sem:
        path = save_temp_file(file)
        try:
            output = apply_background_blur(path, blur_strength)
        finally:
            cleanup_file(path)
        background_tasks.add_task(cleanup_file, output)
        return FileResponse(output, media_type="image/jpeg")  # JPEG

# ---------- LUT ----------
@app.post("/lut-filter")
async def lut_filter(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    filter_type: str = Form("cool"),
):
    async with request.app.state.sem:
        path = save_temp_file(file)
        try:
            output = apply_lut(path, filter_type)
        finally:
            cleanup_file(path)
        background_tasks.add_task(cleanup_file, output)
        return FileResponse(output, media_type="image/jpeg")

# ---------- Sticker ----------
@app.post("/sticker")
async def sticker(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    sticker_name: str = Form("crown"),
):
    async with request.app.state.sem:
        path = save_temp_file(file)
        try:
            output = apply_sticker(path, sticker_name)
        finally:
            cleanup_file(path)
        background_tasks.add_task(cleanup_file, output)
        # Sticker PNG döndürüyoruz (alfa kanalı korunabilir)
        return FileResponse(output, media_type="image/png")

# ---------- Face Morph (placeholder) ----------
@app.post("/face-morph")
async def face_morph(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    eye_size: float = Form(0.5),
    smile: float = Form(0.5),
    chin: float = Form(0.5),
):
    async with request.app.state.sem:
        path = save_temp_file(file)
        try:
            output = apply_face_morph(path, eye_size, smile, chin)
        finally:
            cleanup_file(path)
        background_tasks.add_task(cleanup_file, output)
        return FileResponse(output, media_type="image/jpeg")
