"""WebSocket handler for real-time scan progress updates."""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio


class ConnectionManager:
    """Manages WebSocket connections for scan progress updates."""

    def __init__(self):
        self.active_connections: Dict[int, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, scan_id: int):
        await websocket.accept()
        if scan_id not in self.active_connections:
            self.active_connections[scan_id] = set()
        self.active_connections[scan_id].add(websocket)

    def disconnect(self, websocket: WebSocket, scan_id: int):
        if scan_id in self.active_connections:
            self.active_connections[scan_id].discard(websocket)
            if not self.active_connections[scan_id]:
                del self.active_connections[scan_id]

    async def send_progress(self, scan_id: int, data: dict):
        """Broadcast progress data to all clients watching a scan."""
        if scan_id not in self.active_connections:
            return

        disconnected = set()
        for ws in self.active_connections[scan_id]:
            try:
                await ws.send_json(data)
            except Exception:
                disconnected.add(ws)

        for ws in disconnected:
            self.active_connections[scan_id].discard(ws)

    async def send_complete(self, scan_id: int, data: dict):
        """Send completion message and close connections."""
        data["type"] = "complete"
        await self.send_progress(scan_id, data)


manager = ConnectionManager()
