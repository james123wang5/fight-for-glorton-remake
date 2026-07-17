from __future__ import annotations

import argparse
import asyncio
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import pygame
from websockets.asyncio.client import ClientConnection, connect

from src.runtime import (
    FighterInput,
    RuntimeApp,
    Stage,
    WINDOW_SIZE,
    detect_display_metrics,
    recommended_window_size,
)
from src.simulation import INPUT_FIELDS

from .ai_config import configure_playable_ai
from .protocol import FIGHTERS, PROTOCOL_VERSION, STAGES, TICK_MS, decode_message, encode_message


ROOT = Path(__file__).resolve().parents[1]
FIGHTER_LABELS = {
    "SBLPlayer": "Strawberry",
    "PeachPlayer": "Peach",
    "TrashPlayer": "Trash",
    "CoffeePlayer": "Coffee",
    "DefaultPlayer": "Ball",
    "AuberginePlayer": "Aubergine",
}
COLORS = ((196, 32, 40), (55, 98, 190), (78, 154, 39), (225, 145, 29))


class OnlineGameClient:
    def __init__(self, server_url: str, player_name: str) -> None:
        self.server_url = server_url
        self.player_name = player_name[:16] or "Player"
        self.websocket: ClientConnection | None = None
        self.receive_task: asyncio.Task[None] | None = None
        self.connect_task: asyncio.Task[None] | None = None
        self.mode = "gateway"
        self.status = "C: create room   J: join room"
        self.join_code = ""
        self.room = ""
        self.token = ""
        self.local_slot = -1
        self.room_state: dict[str, Any] = {}
        self.messages: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.pending_frames: dict[int, dict[str, Any]] = {}
        self.runtime: RuntimeApp | None = None
        self.local_input: FighterInput | None = None
        self.server_tick = 0
        self.input_seq = 0
        self.next_input_at = 0.0
        self.last_frame_at = time.monotonic()
        self.last_ping_at = 0.0
        self.ping_id = 0
        self.ping_sent: dict[int, float] = {}
        self.latency_ms = 0.0
        self.jitter_ms = 0.0
        self.corrections = 0
        self.reconnecting = False
        self.selected_target = 0
        self.prediction_offset = pygame.Vector2()
        self.last_local_controls = {field: False for field in INPUT_FIELDS}
        self.finished_message = ""
        self.running = True

    async def send(self, message: Mapping[str, Any]) -> None:
        if self.websocket is None:
            return
        try:
            await self.websocket.send(encode_message(message))
        except Exception:
            self._begin_reconnect()

    def _start_connect(self, action: str) -> None:
        if self.connect_task is not None and not self.connect_task.done():
            return
        self.mode = "connecting"
        self.status = "Connecting..."
        self.connect_task = asyncio.create_task(self._connect(action), name="online-connect")

    async def _connect(self, action: str) -> None:
        try:
            websocket = await connect(
                self.server_url,
                open_timeout=8,
                ping_interval=10,
                ping_timeout=10,
                max_size=2_000_000,
            )
            self.websocket = websocket
            if action == "create":
                hello = {"op": "create", "protocol": PROTOCOL_VERSION, "name": self.player_name}
            elif action == "join":
                hello = {
                    "op": "join",
                    "protocol": PROTOCOL_VERSION,
                    "name": self.player_name,
                    "room": self.join_code,
                }
            else:
                hello = {
                    "op": "resume",
                    "protocol": PROTOCOL_VERSION,
                    "room": self.room,
                    "token": self.token,
                }
            await websocket.send(encode_message(hello))
            self.receive_task = asyncio.create_task(self._receive(websocket), name="online-receive")
            self.reconnecting = False
        except Exception as exc:
            self.websocket = None
            self.mode = "gateway" if action != "resume" else self.mode
            self.status = f"Connection failed: {type(exc).__name__}"

    async def _receive(self, websocket: ClientConnection) -> None:
        try:
            async for payload in websocket:
                await self.messages.put(decode_message(payload))
        except Exception as exc:
            if self.running:
                self.status = f"Disconnected: {type(exc).__name__}"
        finally:
            if self.websocket is websocket:
                self.websocket = None
            if self.running and self.room and self.token and self.mode in {"lobby", "battle"}:
                self._begin_reconnect()

    def _begin_reconnect(self) -> None:
        if self.reconnecting or not self.room or not self.token:
            return
        self.reconnecting = True
        self.status = "Reconnecting..."
        self.connect_task = asyncio.create_task(self._reconnect_loop(), name="online-reconnect")

    async def _reconnect_loop(self) -> None:
        deadline = time.monotonic() + 24.0
        while self.running and time.monotonic() < deadline:
            await self._connect("resume")
            if self.websocket is not None:
                self.reconnecting = False
                return
            await asyncio.sleep(1.0)
        self.reconnecting = False
        self.mode = "gateway"
        self.status = "Reconnect timeout; create or join again"

    async def _handle_message(self, message: dict[str, Any]) -> None:
        op = message.get("op")
        if op == "welcome":
            self.room = str(message["room"])
            self.token = str(message["token"])
            self.local_slot = int(message["slot"])
            if self.mode != "battle":
                self.mode = "lobby"
            self.selected_target = self.local_slot
            self.status = "Connected"
        elif op == "room_state":
            self.room_state = message
            self.server_tick = int(message.get("server_tick", self.server_tick))
            if message.get("paused_for_reconnect"):
                self.status = "Opponent disconnected; waiting up to 25 seconds"
            elif self.mode == "lobby":
                self.status = "Both players press READY to start"
        elif op == "match_start":
            self._start_runtime(message)
        elif op == "frame":
            tick = int(message.get("tick", 0))
            if self.runtime is not None and tick > self.runtime.simulation.tick_index:
                self.pending_frames[tick] = message
            self.server_tick = max(self.server_tick, tick)
        elif op == "snapshot":
            if self.runtime is not None:
                try:
                    self.runtime.simulation.restore_snapshot(message["snapshot"])
                    self.pending_frames = {
                        tick: value
                        for tick, value in self.pending_frames.items()
                        if tick > self.runtime.simulation.tick_index
                    }
                    self.corrections += 1
                except (KeyError, TypeError, ValueError) as exc:
                    self.status = f"Correction failed: {exc}"
        elif op == "pong":
            ping_id = int(message.get("id", -1))
            started = self.ping_sent.pop(ping_id, None)
            if started is not None:
                sample = (time.monotonic() - started) * 1000.0
                previous = self.latency_ms
                self.latency_ms = sample if previous <= 0 else previous * 0.8 + sample * 0.2
                self.jitter_ms = self.jitter_ms * 0.8 + abs(sample - previous) * 0.2 if previous else 0.0
        elif op == "match_end":
            snapshot = message.get("snapshot")
            if self.runtime is not None and isinstance(snapshot, Mapping):
                try:
                    self.runtime.simulation.restore_snapshot(snapshot)
                except ValueError:
                    pass
            winner = message.get("winner")
            self.finished_message = f"Winner: {winner}" if winner else f"Match ended: {message.get('reason')}"
            self.mode = "finished"
        elif op == "error":
            self.status = f"Server: {message.get('code')} {message.get('detail', '')}".strip()

    def _start_runtime(self, message: Mapping[str, Any]) -> None:
        seed = int(message["seed"])
        config = dict(message["match_config"])
        runtime = RuntimeApp(random_seed=seed)
        runtime.audio = None
        runtime.match_config = config
        runtime.manifest["match"]["limit_mode"] = "stock"
        runtime.manifest["match"]["starting_lives"] = int(config.get("limit_value", 5))
        if not bool(config.get("items", True)):
            runtime.manifest["items"]["frequency"] = 0
        runtime.stage = Stage(runtime.manifest, str(config.get("stage", "Mogadishu")))
        runtime.app_state = "battle"
        runtime.simulation.reset(seed)
        initial = message.get("initial_snapshot")
        if isinstance(initial, Mapping):
            runtime.simulation.restore_snapshot(initial)
        runtime.display_metrics = detect_display_metrics(pygame.display.get_surface())
        self.runtime = runtime
        self.local_input = runtime._create_inputs()[0]
        self.pending_frames.clear()
        self.server_tick = runtime.simulation.tick_index
        self.next_input_at = time.monotonic()
        self.last_frame_at = time.monotonic()
        self.prediction_offset.update(0, 0)
        self.mode = "battle"
        self.status = "Online battle"

    async def _process_messages(self) -> None:
        while not self.messages.empty():
            await self._handle_message(self.messages.get_nowait())

    async def _send_controls(self) -> None:
        if self.mode != "battle" or self.local_input is None or self.websocket is None:
            return
        now = time.monotonic()
        if now < self.next_input_at:
            return
        keys = pygame.key.get_pressed()
        controls = self.local_input.controls(keys)
        self.last_local_controls = controls
        self.input_seq += 1
        await self.send(
            {
                "op": "input",
                "seq": self.input_seq,
                "tick": self.server_tick + 2,
                "controls": controls,
            }
        )
        self.next_input_at = max(self.next_input_at + TICK_MS / 1000.0, now)

    async def _send_ping(self) -> None:
        now = time.monotonic()
        if self.websocket is None or now - self.last_ping_at < 1.0:
            return
        self.last_ping_at = now
        self.ping_id += 1
        self.ping_sent[self.ping_id] = now
        await self.send(
            {
                "op": "ping",
                "id": self.ping_id,
                "client_time_ms": round(time.time() * 1000),
            }
        )

    async def _advance_confirmed_frames(self) -> None:
        if self.runtime is None:
            return
        advanced = 0
        while advanced < 8:
            next_tick = self.runtime.simulation.tick_index + 1
            frame = self.pending_frames.pop(next_tick, None)
            if frame is None:
                break
            controls = frame.get("controls", [])
            self.runtime.simulation._advance_once(
                controls,
                advance_clock=True,
                authoritative_inputs=False,
            )
            self.last_frame_at = time.monotonic()
            self.prediction_offset *= 0.45
            expected = frame.get("digest")
            if expected and self.runtime.simulation.state_digest() != expected:
                await self.send(
                    {
                        "op": "desync",
                        "tick": self.runtime.simulation.tick_index,
                        "digest": self.runtime.simulation.state_digest(),
                    }
                )
            advanced += 1

    def _update_prediction(self, elapsed_ms: int) -> None:
        if self.runtime is None or self.local_slot >= len(self.runtime.fighters):
            return
        direction = int(bool(self.last_local_controls.get("right"))) - int(
            bool(self.last_local_controls.get("left"))
        )
        target = direction * min(12.0, max(0.0, self.latency_ms * 0.04 + 3.0))
        factor = 1.0 - math.pow(0.5, max(0, elapsed_ms) / 35.0)
        self.prediction_offset.x += (target - self.prediction_offset.x) * factor
        self.prediction_offset.y *= 1.0 - factor * 0.7

    async def _lobby_patch(self, slot: int, **patch: Any) -> None:
        await self.send({"op": "lobby_update", "slot": slot, "patch": patch})

    async def _settings_patch(self, **patch: Any) -> None:
        await self.send({"op": "settings_update", "patch": patch})

    def _slot(self, index: int) -> dict[str, Any]:
        slots = self.room_state.get("slots", [])
        return dict(slots[index]) if 0 <= index < len(slots) else {}

    async def _handle_gateway_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_c:
            self._start_connect("create")
        elif event.key == pygame.K_j:
            self.join_code = ""
            self.mode = "join_code"
            self.status = "Type the 6-character room code, then Enter"

    async def _handle_join_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return
        if event.key == pygame.K_ESCAPE:
            self.mode = "gateway"
        elif event.key == pygame.K_BACKSPACE:
            self.join_code = self.join_code[:-1]
        elif event.key in {pygame.K_RETURN, pygame.K_KP_ENTER} and len(self.join_code) == 6:
            self._start_connect("join")
        else:
            text = event.unicode.upper()
            if text and text in "ABCDEFGHJKLMNPQRSTUVWXYZ23456789" and len(self.join_code) < 6:
                self.join_code += text

    async def _handle_lobby_event(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN or self.local_slot < 0:
            return
        own = self._slot(self.local_slot)
        is_host = self.local_slot == int(self.room_state.get("host_slot", -1))
        target = self.selected_target if is_host and self.selected_target >= 2 else self.local_slot
        target_slot = self._slot(target)
        if event.key in {pygame.K_RETURN, pygame.K_KP_ENTER}:
            await self._lobby_patch(self.local_slot, ready=not bool(own.get("ready")))
            return
        if event.key == pygame.K_LEFT:
            current = FIGHTERS.index(str(target_slot.get("fighter", "PeachPlayer")))
            await self._lobby_patch(target, fighter=FIGHTERS[(current - 1) % len(FIGHTERS)])
        elif event.key == pygame.K_RIGHT:
            current = FIGHTERS.index(str(target_slot.get("fighter", "PeachPlayer")))
            await self._lobby_patch(target, fighter=FIGHTERS[(current + 1) % len(FIGHTERS)])
        elif event.key == pygame.K_UP:
            await self._lobby_patch(target, color=(int(target_slot.get("color", 0)) + 1) % 4)
        elif event.key == pygame.K_DOWN:
            await self._lobby_patch(target, color=(int(target_slot.get("color", 0)) - 1) % 4)
        elif is_host:
            settings = self.room_state.get("settings", {})
            if event.key == pygame.K_LEFTBRACKET:
                index = STAGES.index(str(settings.get("stage", "Mogadishu")))
                await self._settings_patch(stage=STAGES[(index - 1) % len(STAGES)])
            elif event.key == pygame.K_RIGHTBRACKET:
                index = STAGES.index(str(settings.get("stage", "Mogadishu")))
                await self._settings_patch(stage=STAGES[(index + 1) % len(STAGES)])
            elif event.key == pygame.K_i:
                await self._settings_patch(items=not bool(settings.get("items", True)))
            elif event.key in {pygame.K_MINUS, pygame.K_KP_MINUS}:
                await self._settings_patch(stock=max(1, int(settings.get("stock", 5)) - 1))
            elif event.key in {pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS}:
                await self._settings_patch(stock=min(20, int(settings.get("stock", 5)) + 1))
            elif event.key in {pygame.K_F3, pygame.K_F4}:
                target = 2 if event.key == pygame.K_F3 else 3
                slot = self._slot(target)
                self.selected_target = target
                await self._lobby_patch(target, enabled=not bool(slot.get("enabled")))
            elif event.key == pygame.K_1:
                self.selected_target = self.local_slot
            elif event.key in {pygame.K_3, pygame.K_4}:
                self.selected_target = 2 if event.key == pygame.K_3 else 3
            elif event.key == pygame.K_COMMA and self.selected_target >= 2:
                slot = self._slot(self.selected_target)
                await self._lobby_patch(self.selected_target, level=max(1, int(slot.get("level", 20)) - 1))
            elif event.key == pygame.K_PERIOD and self.selected_target >= 2:
                slot = self._slot(self.selected_target)
                await self._lobby_patch(self.selected_target, level=min(22, int(slot.get("level", 20)) + 1))

    async def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.mode == "battle":
                self.running = False
            elif self.mode == "finished":
                self.running = False
            elif self.mode not in {"gateway", "join_code"}:
                await self.send({"op": "leave"})
                self.mode = "gateway"
                self.room = ""
                self.token = ""
            return
        if self.mode == "gateway":
            await self._handle_gateway_event(event)
        elif self.mode == "join_code":
            await self._handle_join_event(event)
        elif self.mode == "lobby":
            await self._handle_lobby_event(event)
        elif self.mode == "battle" and self.local_input is not None:
            if event.type == pygame.KEYDOWN:
                self.local_input.keydown(event.key)
            elif event.type == pygame.KEYUP:
                self.local_input.keyup(event.key)

    @staticmethod
    def _font(size: int, bold: bool = False) -> pygame.font.Font:
        path = ROOT / "assets" / "fonts" / "2_Futura Md BT.ttf"
        font = pygame.font.Font(str(path) if path.is_file() else None, size)
        font.set_bold(bold)
        return font

    def _draw_center(self, screen: pygame.Surface, text: str, y: int, size: int, color=(245, 245, 245)) -> None:
        image = self._font(size, True).render(text, True, color)
        screen.blit(image, (screen.get_width() // 2 - image.get_width() // 2, y))

    def _draw_gateway(self, screen: pygame.Surface) -> None:
        screen.fill((14, 17, 22))
        self._draw_center(screen, "GLORTON ONLINE", 95, 52)
        if self.mode == "join_code":
            self._draw_center(screen, "ROOM CODE", 205, 24, (170, 180, 195))
            self._draw_center(screen, self.join_code.ljust(6, "_"), 245, 48, (255, 210, 70))
        else:
            self._draw_center(screen, "C   CREATE ROOM", 210, 28, (110, 220, 150))
            self._draw_center(screen, "J   JOIN ROOM", 255, 28, (100, 165, 255))
        self._draw_center(screen, self.status, screen.get_height() - 52, 17, (180, 190, 205))

    def _draw_lobby(self, screen: pygame.Surface) -> None:
        screen.fill((18, 21, 27))
        title = self._font(34, True).render(f"ROOM  {self.room}", True, (255, 215, 72))
        screen.blit(title, (24, 18))
        settings = self.room_state.get("settings", {})
        details = self._font(18).render(
            f"Stage: {settings.get('stage')}   Stock: {settings.get('stock')}   Items: {'ON' if settings.get('items') else 'OFF'}",
            True,
            (210, 218, 228),
        )
        screen.blit(details, (24, 66))
        width = screen.get_width()
        panel_w = (width - 60) // 4
        for index in range(4):
            slot = self._slot(index)
            rect = pygame.Rect(18 + index * (panel_w + 8), 115, panel_w, 250)
            enabled = bool(slot.get("enabled"))
            color_index = int(slot.get("color", index)) % 4
            base = COLORS[color_index] if enabled else (65, 68, 75)
            pygame.draw.rect(screen, base, rect, border_radius=14)
            pygame.draw.rect(screen, (245, 245, 245), rect, 3 if index == self.local_slot else 1, 14)
            if index == self.selected_target:
                pygame.draw.rect(screen, (255, 220, 70), rect.inflate(-8, -8), 2, 10)
            name = str(slot.get("name", f"P{index + 1}"))
            kind = "HUMAN" if slot.get("kind") == "human" else "AI"
            fighter = FIGHTER_LABELS.get(str(slot.get("fighter")), "OFF") if enabled else "OFF"
            lines = [f"P{index + 1}  {kind}", name, fighter]
            if slot.get("kind") == "ai" and enabled:
                lines.append(f"LEVEL {slot.get('level', 20)}")
            if slot.get("kind") == "human":
                lines.append("READY" if slot.get("ready") else "NOT READY")
                lines.append("ONLINE" if slot.get("connected") else "DISCONNECTED")
            for row, line in enumerate(lines):
                image = self._font(20 if row < 3 else 17, row in {0, 2}).render(line, True, (255, 255, 255))
                screen.blit(image, (rect.centerx - image.get_width() // 2, rect.y + 22 + row * 38))
        instructions = [
            "Your fighter: LEFT/RIGHT    Color: UP/DOWN    ENTER: READY",
        ]
        if self.local_slot == int(self.room_state.get("host_slot", -1)):
            instructions += [
                "Host: [ ] stage   - + stock   I items   F3/F4 add CPU",
                "1 select yourself; 3/4 select CPU; arrows edit; ,/. AI level",
            ]
        for row, line in enumerate(instructions):
            image = self._font(16).render(line, True, (190, 200, 214))
            screen.blit(image, (22, 385 + row * 25))
        status = self._font(16).render(self.status, True, (120, 210, 255))
        screen.blit(status, (22, screen.get_height() - 28))

    def _draw_battle(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        if self.runtime is None:
            screen.fill((0, 0, 0))
            return
        elapsed_since_tick = (time.monotonic() - self.last_frame_at) * 1000
        self.runtime.accumulator = max(0, min(TICK_MS - 1, int(elapsed_since_tick)))
        fighter = (
            self.runtime.fighters[self.local_slot]
            if 0 <= self.local_slot < len(self.runtime.fighters)
            else None
        )
        if fighter is not None:
            original = pygame.Vector2(fighter.pos)
            original_prev = pygame.Vector2(fighter.prev_pos)
            fighter.pos += self.prediction_offset
            fighter.prev_pos += self.prediction_offset
            self.runtime._draw_output(screen, font)
            fighter.pos.update(original)
            fighter.prev_pos.update(original_prev)
        else:
            self.runtime._draw_output(screen, font)
        overlay = self._font(15, True).render(
            f"ROOM {self.room}   PING {self.latency_ms:.0f} ms   JITTER {self.jitter_ms:.0f} ms   CORR {self.corrections}",
            True,
            (255, 255, 255),
            (0, 0, 0),
        )
        screen.blit(overlay, (8, 8))
        if self.room_state.get("paused_for_reconnect") or self.reconnecting:
            shade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
            shade.fill((0, 0, 0, 155))
            screen.blit(shade, (0, 0))
            self._draw_center(screen, self.status, screen.get_height() // 2 - 25, 28)

    async def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("The Fight for Glorton - Online")
        screen = pygame.display.set_mode(recommended_window_size(WINDOW_SIZE), pygame.RESIZABLE)
        clock = pygame.time.Clock()
        debug_font = pygame.font.SysFont("menlo", 14)
        while self.running:
            elapsed = clock.tick(60)
            for event in pygame.event.get():
                await self._handle_event(event)
            await self._process_messages()
            await self._send_controls()
            await self._send_ping()
            await self._advance_confirmed_frames()
            self._update_prediction(elapsed)
            if self.mode in {"gateway", "join_code", "connecting"}:
                self._draw_gateway(screen)
            elif self.mode == "lobby":
                self._draw_lobby(screen)
            elif self.mode in {"battle", "finished"}:
                self._draw_battle(screen, debug_font)
                if self.mode == "finished":
                    shade = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
                    shade.fill((0, 0, 0, 145))
                    screen.blit(shade, (0, 0))
                    self._draw_center(screen, self.finished_message, screen.get_height() // 2 - 30, 34)
                    self._draw_center(screen, "ESC to close", screen.get_height() // 2 + 25, 18)
            pygame.display.flip()
            await asyncio.sleep(0)
        await self.send({"op": "leave"})
        if self.websocket is not None:
            await self.websocket.close()
        if self.receive_task is not None:
            self.receive_task.cancel()
        pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Glorton 电脑版联机客户端")
    parser.add_argument(
        "--server",
        default=os.environ.get("GLORTON_ONLINE_SERVER", "ws://127.0.0.1:8765/ws"),
        help="ws:// or wss:// server URL",
    )
    parser.add_argument("--name", default=os.environ.get("GLORTON_PLAYER_NAME", "Player"))
    args = parser.parse_args()
    from tools.install_runtime_assets import ensure_runtime_assets

    installed = ensure_runtime_assets()
    if installed:
        os.environ.setdefault("GLORTON_ASSET_SCALE", "1")
    configure_playable_ai()
    asyncio.run(OnlineGameClient(args.server, args.name).run())


if __name__ == "__main__":
    main()
