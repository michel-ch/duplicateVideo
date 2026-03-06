# Duplicate Video Detector

![Dashboard](docs/dashboard.png)

A full-stack web application that scans directories, detects duplicate or near-duplicate videos, compares their quality, and helps you clean up lower-quality copies.

## Architecture

- **Backend**: Python + FastAPI + SQLAlchemy (SQLite) + FFmpeg
- **Frontend**: React + TypeScript + Vite
- **Real-time**: WebSocket for live scan progress

## Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **FFmpeg** (must be installed and in PATH)

### Installing FFmpeg

**Windows:**

```bash
# Using chocolatey
choco install ffmpeg

# Or download from https://ffmpeg.org/download.html
# Add ffmpeg/bin to your system PATH
```

**Mac:**

```bash
brew install ffmpeg
```

**Linux:**

```bash
sudo apt install ffmpeg
```

## Quick Start

### Option A: One-Click (Windows)

Double-click `start.bat` — starts both servers and opens the app.

- Backend: http://localhost:9000
- Frontend: http://localhost:3000

### Option B: Manual

**Backend:**

```bash
cd backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn main:app --reload --host 0.0.0.0 --port 9000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev -- --port 3000
```

Open **http://localhost:3000** in your browser.

## Features

### Duplicate Detection Pipeline

4-stage progressive filtering:

1. **Duration pre-filter** — groups videos by approximate duration (±2s or 5% tolerance)
2. **Perceptual hashing** — extracts key frames, computes pHash, compares via Hamming distance
3. **Audio fingerprinting** — RMS energy cross-correlation as fallback for re-encodes
4. **Quality scoring** — weighted analysis of resolution (40%), bitrate (25%), codec (15%), file size (10%), FPS (10%)

### Scan Queue

- Queue multiple directory scans — they run sequentially, one at a time
- Real-time progress via WebSocket with pause/resume/stop controls
- Cancel queued scans before they start
- GPU-accelerated processing when NVIDIA CUDA is available

### Duplicate Review

- Side-by-side comparison with full metadata
- Auto-selects best quality file to keep
- Status workflow: **Pending** → **In Queue** (after review) → **Resolved**
- Filters persist when navigating between list and comparison views

### Deletion & Cleanup

- **Deletion Queue** — batch process reviewed duplicates (permanent delete by default)
- **Auto-clean** — one-click cleanup of all lower-quality duplicates
- **History** — full deletion log with undo (restore from trash) and clear history

### Configuration

- Adjustable similarity thresholds and quality scoring weights
- Configurable detection parameters (key frames, hash threshold, duration tolerance)
- Video extensions and protected paths

## API Documentation

Once the backend is running, visit **http://localhost:9000/docs** for the interactive Swagger UI.

## Project Structure

```
├── backend/
│   ├── main.py                    # FastAPI app entry point
│   ├── config.py                  # Settings & configuration
│   ├── models/
│   │   ├── database.py            # SQLAlchemy models & DB setup
│   │   └── schemas.py             # Pydantic schemas
│   ├── services/
│   │   ├── scanner.py             # Video file discovery
│   │   ├── metadata.py            # FFprobe metadata extraction
│   │   ├── hasher.py              # Perceptual hashing
│   │   ├── audio_fingerprint.py   # Audio fingerprint extraction
│   │   ├── comparator.py          # Duplicate detection pipeline
│   │   ├── quality_scorer.py      # Quality scoring & ranking
│   │   ├── file_manager.py        # Deletion & trash operations
│   │   ├── scan_control.py        # Pause/resume/stop signals
│   │   └── gpu_detector.py        # NVIDIA GPU detection
│   ├── api/
│   │   ├── scan.py               # Scan endpoints + queue logic
│   │   ├── duplicates.py         # Duplicate group endpoints
│   │   ├── actions.py            # Delete/clean/stats/history endpoints
│   │   └── websocket.py          # WebSocket connection manager
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx               # Root with routing & sidebar
│   │   ├── pages/                # Dashboard, DuplicatesList, ComparisonView, etc.
│   │   ├── components/           # VideoCard, ProgressTracker, ConfirmationModal, etc.
│   │   ├── hooks/                # useWebSocket, useScanProgress
│   │   ├── services/api.ts       # API client
│   │   └── types/index.ts        # TypeScript interfaces
│   └── vite.config.ts
├── start.bat                      # Windows one-click launcher
└── README.md
```
