from __future__ import annotations

import argparse
import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .ai_config import configure_playable_ai
from .protocol import PROTOCOL_VERSION, ProtocolError, decode_message, encode_message, room_code
from .room import Room, RoomManager


manager = RoomManager()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    from tools.install_runtime_assets import ensure_runtime_assets

    ensure_runtime_assets()
    configure_playable_ai()
    cleanup = asyncio.create_task(manager.cleanup(), name="room-cleanup")
    try:
        yield
    finally:
        cleanup.cancel()
        try:
            await cleanup
        except asyncio.CancelledError:
            pass


app = FastAPI(title="Glorton Online Server", version=str(PROTOCOL_VERSION), lifespan=lifespan)


@app.get("/")
@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "protocol": PROTOCOL_VERSION,
        "rooms": len(manager.rooms),
        "server_time_ms": round(time.time() * 1000),
    }


async def _send(websocket: WebSocket, message: dict[str, Any]) -> None:
    await websocket.send_text(encode_message(message))


async def _error(websocket: WebSocket, code: str, detail: str = "") -> None:
    await _send(websocket, {"op": "error", "code": code, "detail": detail[:200]})


@app.websocket("/ws")
async def websocket_room(websocket: WebSocket) -> None:
    await websocket.accept()
    room: Room | None = None
    token = ""
    slot = -1

    async def send(message: dict[str, Any]) -> None:
        await _send(websocket, message)

    try:
        first = decode_message(await websocket.receive_text())
        op = first["op"]
        if int(first.get("protocol", -1)) != PROTOCOL_VERSION:
            await _error(websocket, "protocol_mismatch", str(PROTOCOL_VERSION))
            await websocket.close(code=1002)
            return
        if op == "create":
            room, token, slot = manager.create(str(first.get("name", "玩家")))
            room.add_connection(token, slot, send)
        elif op == "join":
            room = manager.get(room_code(first.get("room")))
            token, slot = room.join(str(first.get("name", "玩家")))
            room.add_connection(token, slot, send)
        elif op == "resume":
            room = manager.get(room_code(first.get("room")))
            token = str(first.get("token", ""))
            slot = room.resume(token, send)
        else:
            await _error(websocket, "handshake_required")
            await websocket.close(code=1002)
            return

        await send(
            {
                "op": "welcome",
                "protocol": PROTOCOL_VERSION,
                "room": room.code,
                "slot": slot,
                "token": token,
                "reconnect_seconds": 25,
            }
        )
        await room.broadcast(room.public_state())
        if room.phase == "playing":
            await room.send_snapshot(slot)

        while True:
            message = decode_message(await websocket.receive_text())
            connection = room.connections.get(token)
            if connection is None:
                break
            if not connection.accept_message():
                await _error(websocket, "rate_limited")
                await websocket.close(code=1008)
                break
            op = message["op"]
            try:
                if op == "lobby_update":
                    await room.update_lobby(slot, message)
                    if room.can_start():
                        await room.start_match()
                elif op == "settings_update":
                    await room.update_settings(slot, message)
                elif op == "input":
                    room.queue_input(slot, message)
                elif op == "desync":
                    await room.send_snapshot(slot)
                elif op == "ping":
                    await send(
                        {
                            "op": "pong",
                            "id": int(message.get("id", 0)),
                            "client_time_ms": int(message.get("client_time_ms", 0)),
                            "server_time_ms": round(time.time() * 1000),
                            "server_tick": room.server_tick,
                        }
                    )
                elif op == "leave":
                    break
                else:
                    await _error(websocket, "unknown_op", op)
            except (ProtocolError, TypeError, ValueError) as exc:
                await _error(websocket, str(exc))
    except WebSocketDisconnect:
        pass
    except (ProtocolError, TypeError, ValueError) as exc:
        try:
            await _error(websocket, str(exc))
        except Exception:
            pass
    finally:
        if room is not None and token:
            room.disconnect(token)
            await room.broadcast(room.public_state())


def main() -> None:
    parser = argparse.ArgumentParser(description="Glorton 40Hz 权威联机服务器")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8765")))
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args()
    import uvicorn

    uvicorn.run("online.server:app", host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
