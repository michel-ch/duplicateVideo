# ---- Stage 1: Build frontend ----
FROM node:20-alpine AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Backend with NVIDIA CUDA support ----
FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python, FFmpeg (with NVIDIA support), and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/

# Copy built frontend into backend static serving directory
COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Create directories for runtime data (DATABASE_URL is ./data/... resolved
# from WORKDIR /app/backend, and docker-compose mounts ./data here too).
RUN mkdir -p /app/backend/thumbnails /app/backend/data

# Set working directory to backend for uvicorn
WORKDIR /app/backend

ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

EXPOSE 9000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
