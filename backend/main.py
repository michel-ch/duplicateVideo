"""FastAPI application entry point."""

import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# Suppress noisy polling endpoints from uvicorn access logs
class _QuietPollFilter(logging.Filter):
    """Filter out high-frequency polling endpoints from access logs."""
    _QUIET_PATHS = ("/api/stats", "/api/gpu-status")

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in self._QUIET_PATHS)

logging.getLogger("uvicorn.access").addFilter(_QuietPollFilter())

from models.database import init_db
from api.scan import router as scan_router
from api.duplicates import router as duplicates_router
from api.actions import router as actions_router
from services.gpu_detector import detect_gpu, get_gpu_info


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and probe GPU on startup."""
    await init_db()
    # Eagerly detect GPU capabilities so the first scan doesn't pay the cost
    gpu = detect_gpu()
    if gpu.available:
        print(f"[STARTUP] GPU acceleration enabled — {gpu.gpu_name}")
        print(f"[STARTUP] VRAM: {gpu.vram_total_mb} MB | CUVID decoders: {gpu.cuvid_decoders}")
    else:
        print("[STARTUP] No NVIDIA GPU detected — running in CPU-only mode")
    yield


app = FastAPI(
    title="Duplicate Video Detector",
    description="Scan directories and detect duplicate or near-duplicate videos",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(scan_router, prefix="/api", tags=["Scan"])
app.include_router(duplicates_router, prefix="/api", tags=["Duplicates"])
app.include_router(actions_router, prefix="/api", tags=["Actions"])

# Serve thumbnails as static files
from config import settings
thumbnails_dir = os.path.join(os.getcwd(), settings.THUMBNAILS_DIR)
os.makedirs(thumbnails_dir, exist_ok=True)
app.mount("/thumbnails", StaticFiles(directory=thumbnails_dir), name="thumbnails")


@app.get("/")
async def root():
    return {"message": "Duplicate Video Detector API", "version": "1.0.0"}


@app.get("/api/gpu-status")
async def gpu_status():
    """Return detected GPU capabilities and acceleration status."""
    gpu = get_gpu_info()
    return {
        "gpu_available": gpu.available,
        "gpu_name": gpu.gpu_name,
        "driver_version": gpu.driver_version,
        "vram_total_mb": gpu.vram_total_mb,
        "vram_free_mb": gpu.vram_free_mb,
        "hwaccel_supported": gpu.hwaccel_supported,
        "cuvid_decoders": gpu.cuvid_decoders,
        "cuda_filters": gpu.cuda_filters,
        "nvenc_encoders": gpu.nvenc_encoders,
        "acceleration_active": gpu.available and gpu.hwaccel_supported,
    }

