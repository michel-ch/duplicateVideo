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

from datetime import datetime, timezone
from sqlalchemy import select

from models.database import init_db, async_session, ScanJob
from api.scan import router as scan_router
from api.duplicates import router as duplicates_router
from api.actions import router as actions_router
from services.gpu_detector import detect_gpu, get_gpu_info


_ORPHAN_ACTIVE_STATUSES = (
    "pending", "scanning", "metadata", "hashing", "comparing", "paused",
)


async def _recover_orphaned_scans() -> None:
    """Mark scans that were active when the server died as 'stopped'.

    On a clean shutdown via /stop or the user clicking through the UI,
    scans transition into a terminal state cleanly.  But if the process
    was killed (Ctrl-C, OOM, power loss), any scan still in an active
    state has no running pipeline behind it — the FileCache rows it
    committed are intact, but the ScanJob row itself looks alive forever.
    Flip these to 'stopped' so the UI doesn't show a fake spinner and
    so new scans can leave the 'queued' state.
    """
    async with async_session() as db:
        result = await db.execute(
            select(ScanJob).where(ScanJob.status.in_(_ORPHAN_ACTIVE_STATUSES))
        )
        orphans = result.scalars().all()
        if not orphans:
            return
        now = datetime.now(timezone.utc)
        for s in orphans:
            s.status = "stopped"
            s.completed_at = now
            s.current_stage = "Server restarted while scan was in progress"
            s.current_file = None
        await db.commit()
        print(f"[STARTUP] Recovered {len(orphans)} orphaned scan(s) → stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and probe GPU on startup."""
    await init_db()
    await _recover_orphaned_scans()
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


# Serve frontend static files (production build)
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="frontend-assets")

    from fastapi.responses import FileResponse

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_dist, "index.html"))

    @app.get("/{full_path:path}")
    async def serve_frontend_fallback(full_path: str):
        """SPA fallback - serve index.html for client-side routes."""
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
else:
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

