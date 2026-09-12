"""
VoiceGuard Real-Time WebSocket Streaming Gateway
Provides low-latency audio chunk streaming and live telemetry push to connected security dashboards.
"""

import json
import logging
from typing import Dict, Set
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger("VoiceGuard.WebSocket")
router = APIRouter(tags=["WebSocket"])

class ConnectionManager:
    """Manages active WebSocket telemetry connections keyed by call_id."""
    def __init__(self):
        self.active_connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, call_id: str, websocket: WebSocket):
        await websocket.accept()
        if call_id not in self.active_connections:
            self.active_connections[call_id] = set()
        self.active_connections[call_id].add(websocket)
        logger.info(f"Dashboard client connected to call stream: {call_id}")

    def disconnect(self, call_id: str, websocket: WebSocket):
        if call_id in self.active_connections:
            self.active_connections[call_id].discard(websocket)
            if not self.active_connections[call_id]:
                del self.active_connections[call_id]
        logger.info(f"Dashboard client disconnected from: {call_id}")

    async def broadcast_telemetry(self, call_id: str, message: dict):
        if call_id in self.active_connections:
            dead_sockets = set()
            for connection in self.active_connections[call_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception:
                    dead_sockets.add(connection)
            for dead in dead_sockets:
                self.active_connections[call_id].discard(dead)

manager = ConnectionManager()

@router.websocket("/ws/calls/{call_id}/telemetry")
async def websocket_call_telemetry(websocket: WebSocket, call_id: str):
    """
    Subscribes the frontend cybersecurity dashboard to real-time risk scores,
    waveform metrics, and defensive trigger notifications for a specific call.
    """
    await manager.connect(call_id, websocket)
    try:
        while True:
            # Client can ping or send heartbeat messages
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        manager.disconnect(call_id, websocket)