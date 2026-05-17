# Deployment

Two supported deployment modes: native (a uvicorn process serving the built SPA on a single port) and Docker (CUDA-enabled container).

## Native single-port

Build the frontend, then run the backend. The backend auto-detects `frontend/dist` and serves it.

```bash
# Build frontend
cd frontend
npm ci
npm run build      # produces frontend/dist/

# Run backend
cd ../backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 9000
```

Open http://localhost:9000 — the same port serves API, thumbnails, and the SPA. Hard-refresh on `/duplicates/42` works because of the SPA fallback in [`backend/main.py`](../backend/main.py) lines 82–88.

For production: drop `--reload`, run behind a reverse proxy (see below), and use a process manager (systemd, NSSM, supervisord).

## Docker

[`Dockerfile`](../Dockerfile) is a two-stage build:

1. **`node:20-alpine`** builds the frontend (`npm ci && npm run build`).
2. **`nvidia/cuda:12.2.2-runtime-ubuntu22.04`** installs Python 3.11, FFmpeg (with NVIDIA support from the Ubuntu repo), copies the backend source, and copies `dist/` into `frontend/dist/` for static serving. Single port, single process.

Build and run:

```bash
docker compose up --build
```

[`docker-compose.yml`](../docker-compose.yml) declares:

- **NVIDIA runtime** + device bindings + `deploy.resources.reservations.devices` for GPU access.
- **Volume mounts** for the SQLite DB (`./data:/app/backend/data`) and thumbnails (`./thumbnails:/app/backend/thumbnails`) so they survive container restarts.
- **`DATABASE_URL`** override pointing inside the data volume.

To mount your media for scanning, uncomment the example line:

```yaml
volumes:
  - /path/to/your/videos:/media:ro
```

Then enter `/media` (or a subdirectory) as the scan path in the UI.

### Without GPU

Remove the NVIDIA-specific lines from `docker-compose.yml` (the `runtime`, `devices`, `deploy.resources` blocks, and the env vars). Pin a non-CUDA base image in the `Dockerfile`:

```dockerfile
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libgl1-mesa-glx libglib2.0-0 ...
```

The application will detect the missing GPU at startup and run in CPU mode automatically.

## Reverse proxy

For serving on port 80/443, put nginx (or Caddy / Traefik) in front. **WebSocket support is required** — the scan progress stream needs the standard upgrade dance:

```nginx
server {
    listen 443 ssl;
    server_name viddup.example.com;

    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host       $host;
        proxy_read_timeout 600s;     # long scans send WS messages over hours
    }
}
```

The `proxy_read_timeout` of 600s is important — a default 60s read timeout will silently drop idle WS connections, and the frontend will keep reconnecting. The throttled progress messages (≥ 2% increments) can occasionally be more than 60s apart on huge scans.

## Persistence

| Path | Purpose | Survives container restart? |
|---|---|---|
| `backend/duplicate_detector.db` | SQLite DB | Yes via `./data` volume in compose |
| `backend/thumbnails/` | Generated JPEGs | Yes via `./thumbnails` volume |
| Trash folders (`<scan_root>/.duplicate_trash/`) | Recoverable deleted files | Lives in user's filesystem; not container-managed |

If you delete the DB, the thumbnails become orphaned (no row references them). They'll be cleaned up implicitly only when their `scan{id}_thumb_{idx}.jpg` filename collides with a new scan's id+idx, which is unreliable. Periodic cleanup of orphans is left to the user.

## Authentication

There is **none**. This is a single-user local app. If exposing on the network:

- Bind only to localhost (`--host 127.0.0.1`) and reach via SSH tunnel, **or**
- Add basic auth at the reverse proxy layer, **or**
- Run on a private network only.

The application has the ability to delete files anywhere on the filesystem the process can reach. **Do not expose it publicly.**

## Logs

uvicorn logs to stdout. The `_QuietPollFilter` in `main.py` drops `/api/stats` and `/api/gpu-status` lines (Dashboard polls them every 3s). Other logs include:

- `[STARTUP] GPU acceleration enabled — <name>` (or CPU-only message)
- `[GPU] GPU: ... | Driver: ... | VRAM: ...` (one-time summary)
- `Audio FP error: <path>: <error>` per-file failures
- `Error processing video: <error>` from metadata stage
- `[GPU/CPU] Frame extraction error for <path>: <error>`
- `[GPU/CPU] Thumbnail extraction error: <error>`

Per-file errors don't abort the scan — they're logged and the scan continues with whatever videos succeeded.

## Updating

1. Stop the running container/process.
2. Pull or copy in new code.
3. **Frontend**: `npm ci && npm run build`.
4. **Backend**: re-install requirements if `requirements.txt` changed.
5. **DB**: if any model changed, delete `duplicate_detector.db` (no migrations).
6. Restart.

For Docker:

```bash
docker compose down
docker compose up --build -d
```

## Health monitoring

There's no `/health` endpoint, but `GET /api/stats` is a good liveness check — it touches the DB and returns quickly.

```bash
curl -fs http://localhost:9000/api/stats > /dev/null && echo "alive"
```

For deeper monitoring, `GET /api/gpu-status` returns the cached probe — useful for confirming the GPU stayed visible after a driver event.
