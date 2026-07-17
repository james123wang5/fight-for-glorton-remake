from __future__ import annotations

import asyncio
import copy
import secrets
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Awaitable, Callable, Mapping

from src.simulation import BattleSimulation, INPUT_FIELDS

from .protocol import (
    EDGE_FIELDS,
    FIGHTERS,
    PROTOCOL_VERSION,
    STAGES,
    TICK_MS,
    clean_name,
    controls_without_edges,
    normalize_controls,
    validate_lobby_patch,
    validate_settings_patch,
)


Send = Callable[[dict[str, Any]], Awaitable[None]]
ROOM_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
RECONNECT_SECONDS = 25.0
EMPTY_ROOM_TTL_SECONDS = 180.0
FINISHED_ROOM_TTL_SECONDS = 90.0


@dataclass
class PlayerSlot:
    index: int
    kind: str
    name: str
    fighter: str
    color: int
    level: int = 20
    enabled: bool = True
    ready: bool = False
    connected: bool = False
    token: str = ""
    disconnected_at: float | None = None

    def public(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("token", None)
        value.pop("disconnected_at", None)
        return value


@dataclass
class Connection:
    token: str
    slot: int
    send: Send
    messages_this_second: int = 0
    rate_window_started: float = field(default_factory=time.monotonic)

    def accept_message(self) -> bool:
        now = time.monotonic()
        if now - self.rate_window_started >= 1.0:
            self.rate_window_started = now
            self.messages_this_second = 0
        self.messages_this_second += 1
        return self.messages_this_second <= 120


def _blank_controls() -> dict[str, bool]:
    return {field: False for field in INPUT_FIELDS}


class Room:
    def __init__(self, code: str, host_name: str) -> None:
        token = secrets.token_urlsafe(24)
        self.code = code
        self.created_at = time.monotonic()
        self.updated_at = self.created_at
        self.phase = "lobby"
        self.host_token = token
        self.slots = [
            PlayerSlot(0, "human", clean_name(host_name), "PeachPlayer", 0, connected=True, token=token),
            PlayerSlot(1, "human", "等待玩家", "PeachPlayer", 1, enabled=False),
            PlayerSlot(2, "ai", "CP3", "TrashPlayer", 2, level=20, enabled=False, connected=True),
            PlayerSlot(3, "ai", "CP4", "CoffeePlayer", 3, level=20, enabled=False, connected=True),
        ]
        self.settings: dict[str, Any] = {"stage": "Mogadishu", "stock": 5, "items": True}
        self.connections: dict[str, Connection] = {}
        self.inputs: dict[int, dict[int, dict[str, bool]]] = {0: {}, 1: {}}
        self.last_inputs: dict[int, dict[str, bool]] = {0: _blank_controls(), 1: _blank_controls()}
        self.last_input_seq: dict[int, int] = {0: -1, 1: -1}
        self.acks: dict[int, int] = {0: -1, 1: -1}
        self.simulation: BattleSimulation | None = None
        self.match_task: asyncio.Task[None] | None = None
        self.seed = 0
        self.server_tick = 0
        self.paused_for_reconnect = False
        self.finished_at: float | None = None

    @property
    def host_slot(self) -> int:
        return 0

    def add_connection(self, token: str, slot: int, send: Send) -> None:
        self.connections[token] = Connection(token, slot, send)
        player = self.slots[slot]
        player.connected = True
        player.disconnected_at = None
        self.updated_at = time.monotonic()

    def join(self, player_name: str) -> tuple[str, int]:
        if self.phase != "lobby":
            raise ValueError("match_already_started")
        slot = self.slots[1]
        if slot.enabled and slot.token:
            raise ValueError("room_full")
        token = secrets.token_urlsafe(24)
        slot.name = clean_name(player_name)
        slot.enabled = True
        slot.ready = False
        slot.connected = True
        slot.token = token
        slot.disconnected_at = None
        self.updated_at = time.monotonic()
        return token, 1

    def resume(self, token: str, send: Send) -> int:
        for slot in self.slots[:2]:
            if slot.token == token:
                self.add_connection(token, slot.index, send)
                return slot.index
        raise ValueError("invalid_resume_token")

    def disconnect(self, token: str) -> None:
        connection = self.connections.pop(token, None)
        if connection is None:
            return
        slot = self.slots[connection.slot]
        slot.connected = False
        slot.disconnected_at = time.monotonic()
        slot.ready = False if self.phase == "lobby" else slot.ready
        self.updated_at = time.monotonic()

    def public_state(self) -> dict[str, Any]:
        return {
            "op": "room_state",
            "protocol": PROTOCOL_VERSION,
            "room": self.code,
            "phase": self.phase,
            "host_slot": self.host_slot,
            "settings": copy.deepcopy(self.settings),
            "slots": [slot.public() for slot in self.slots],
            "server_tick": self.server_tick,
            "paused_for_reconnect": self.paused_for_reconnect,
        }

    async def broadcast(self, message: dict[str, Any]) -> None:
        stale: list[str] = []
        for token, connection in tuple(self.connections.items()):
            try:
                await connection.send(message)
            except Exception:
                stale.append(token)
        for token in stale:
            self.disconnect(token)

    async def send_to(self, slot: int, message: dict[str, Any]) -> None:
        for connection in tuple(self.connections.values()):
            if connection.slot == slot:
                await connection.send(message)

    async def update_lobby(self, actor_slot: int, message: Mapping[str, Any]) -> None:
        if self.phase != "lobby":
            raise ValueError("not_in_lobby")
        target = int(message.get("slot", actor_slot))
        if target < 0 or target >= len(self.slots):
            raise ValueError("invalid_slot")
        patch = validate_lobby_patch(message.get("patch"))
        if target < 2:
            if target != actor_slot:
                raise ValueError("cannot_edit_other_human")
            patch = {key: value for key, value in patch.items() if key in {"fighter", "color", "ready"}}
        else:
            if actor_slot != self.host_slot:
                raise ValueError("host_only")
            patch.pop("ready", None)
        slot = self.slots[target]
        if "fighter" in patch or "color" in patch or "level" in patch or "enabled" in patch:
            for human in self.slots[:2]:
                human.ready = False
        for key, value in patch.items():
            setattr(slot, key, value)
        if target >= 2:
            slot.connected = slot.enabled
            slot.ready = slot.enabled
            # Keep active player indices stable: P4 cannot exist while P3 is
            # absent because RuntimeApp compacts disabled player configs.
            if target == 3 and slot.enabled:
                self.slots[2].enabled = True
                self.slots[2].connected = True
                self.slots[2].ready = True
            elif target == 2 and not slot.enabled:
                self.slots[3].enabled = False
                self.slots[3].connected = False
                self.slots[3].ready = False
        self.updated_at = time.monotonic()
        await self.broadcast(self.public_state())

    async def update_settings(self, actor_slot: int, message: Mapping[str, Any]) -> None:
        if actor_slot != self.host_slot:
            raise ValueError("host_only")
        if self.phase != "lobby":
            raise ValueError("not_in_lobby")
        self.settings.update(validate_settings_patch(message.get("patch")))
        for human in self.slots[:2]:
            human.ready = False
        self.updated_at = time.monotonic()
        await self.broadcast(self.public_state())

    def can_start(self) -> bool:
        humans = self.slots[:2]
        return bool(
            self.phase == "lobby"
            and all(slot.enabled and slot.connected and slot.ready and slot.fighter for slot in humans)
        )

    def match_config(self) -> dict[str, Any]:
        players = []
        for slot in self.slots:
            if not slot.enabled:
                continue
            players.append(
                {
                    "fighter": slot.fighter,
                    "color": slot.color,
                    "computer": slot.kind == "ai",
                    "enabled": True,
                    "level": slot.level,
                    "team_index": slot.index,
                    "network_slot": slot.index,
                }
            )
        return {
            "mode": "online",
            "type": "vsmode",
            "stage": self.settings["stage"],
            "selected_stage": self.settings["stage"],
            "limit_mode": "stock",
            "limit_value": self.settings["stock"],
            "items": self.settings["items"],
            "players": players,
            "fighters": [item["fighter"] for item in players],
            "colors": [item["color"] for item in players],
        }

    async def start_match(self) -> None:
        if not self.can_start() or self.match_task is not None:
            return
        unsupported = [
            slot.index + 1
            for slot in self.slots[2:]
            if slot.enabled and slot.level > 20 and self.settings["stage"] != "Mogadishu"
        ]
        if unsupported:
            for human in self.slots[:2]:
                human.ready = False
            await self.broadcast(
                {
                    "op": "error",
                    "code": "trained_ai_stage_mismatch",
                    "detail": "21/22 AI is currently trained for Mogadishu; choose Mogadishu or level 1-20",
                }
            )
            await self.broadcast(self.public_state())
            return
        self.phase = "starting"
        self.seed = secrets.randbits(31)
        config = self.match_config()
        self.simulation = BattleSimulation.headless(seed=self.seed, match_config=config)
        self.simulation.runtime.app_state = "battle"
        if not bool(self.settings.get("items", True)):
            self.simulation.runtime.manifest["items"]["frequency"] = 0
        self.server_tick = 0
        initial = self.simulation.snapshot()
        await self.broadcast(
            {
                "op": "match_start",
                "room": self.code,
                "seed": self.seed,
                "tick_ms": TICK_MS,
                "input_delay_ticks": 2,
                "match_config": config,
                "initial_snapshot": initial,
            }
        )
        self.phase = "playing"
        await self.broadcast(self.public_state())
        self.match_task = asyncio.create_task(self._run_match(), name=f"room-{self.code}")

    def queue_input(self, slot: int, message: Mapping[str, Any]) -> None:
        if self.phase != "playing" or slot not in {0, 1}:
            return
        seq = int(message.get("seq", -1))
        if seq <= self.last_input_seq[slot]:
            return
        target_tick = int(message.get("tick", self.server_tick + 2))
        target_tick = max(self.server_tick, min(self.server_tick + 8, target_tick))
        self.last_input_seq[slot] = seq
        self.inputs[slot][target_tick] = normalize_controls(message.get("controls"))
        for old_tick in [value for value in self.inputs[slot] if value < self.server_tick]:
            self.inputs[slot].pop(old_tick, None)

    def _controls_for_tick(self, slot: int, tick: int) -> dict[str, bool]:
        queued = self.inputs[slot].pop(tick, None)
        if queued is None:
            return controls_without_edges(self.last_inputs[slot])
        self.last_inputs[slot] = queued
        self.acks[slot] = self.last_input_seq[slot]
        return queued

    def _humans_connected(self) -> bool:
        return all(slot.connected for slot in self.slots[:2] if slot.enabled)

    async def _run_match(self) -> None:
        assert self.simulation is not None
        loop = asyncio.get_running_loop()
        deadline = loop.time()
        try:
            while self.phase == "playing":
                now = loop.time()
                if not self._humans_connected():
                    if not self.paused_for_reconnect:
                        self.paused_for_reconnect = True
                        await self.broadcast(self.public_state())
                    expired = any(
                        slot.enabled
                        and not slot.connected
                        and slot.disconnected_at is not None
                        and time.monotonic() - slot.disconnected_at > RECONNECT_SECONDS
                        for slot in self.slots[:2]
                    )
                    if expired:
                        self.phase = "finished"
                        self.finished_at = time.monotonic()
                        await self.broadcast({"op": "match_end", "reason": "reconnect_timeout"})
                        break
                    deadline = now + TICK_MS / 1000
                    await asyncio.sleep(0.05)
                    continue
                if self.paused_for_reconnect:
                    self.paused_for_reconnect = False
                    await self.broadcast(self.public_state())
                    await self.broadcast({"op": "snapshot", "snapshot": self.simulation.snapshot()})
                    deadline = now

                controls = [_blank_controls() for _ in self.simulation.runtime.fighters]
                for human_slot in (0, 1):
                    if human_slot < len(controls):
                        controls[human_slot] = self._controls_for_tick(human_slot, self.server_tick)
                self.simulation.step_fast(controls, advance_clock=True)
                self.server_tick = self.simulation.tick_index
                frame: dict[str, Any] = {
                    "op": "frame",
                    "tick": self.server_tick,
                    "controls": copy.deepcopy(self.simulation._previous_controls),
                    "acks": {str(key): value for key, value in self.acks.items()},
                }
                if self.server_tick % 20 == 0:
                    frame["digest"] = self.simulation.state_digest()
                await self.broadcast(frame)
                if self.simulation.runtime.match_state == "game_set":
                    self.phase = "finished"
                    self.finished_at = time.monotonic()
                    winner = self.simulation.runtime.match_winner
                    await self.broadcast(
                        {
                            "op": "match_end",
                            "reason": "game_set",
                            "winner": winner.name if winner is not None else None,
                            "snapshot": self.simulation.snapshot(),
                        }
                    )
                    break
                deadline += TICK_MS / 1000
                delay = deadline - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif delay < -0.25:
                    deadline = loop.time()
        finally:
            self.match_task = None

    async def send_snapshot(self, slot: int) -> None:
        if self.simulation is None:
            return
        await self.send_to(slot, {"op": "snapshot", "snapshot": self.simulation.snapshot()})

    def expired(self, now: float) -> bool:
        if self.phase == "finished" and self.finished_at is not None:
            return now - self.finished_at > FINISHED_ROOM_TTL_SECONDS
        if not self.connections:
            return now - self.updated_at > EMPTY_ROOM_TTL_SECONDS
        return False


class RoomManager:
    def __init__(self) -> None:
        self.rooms: dict[str, Room] = {}

    def _new_code(self) -> str:
        for _ in range(100):
            code = "".join(secrets.choice(ROOM_ALPHABET) for _ in range(6))
            if code not in self.rooms:
                return code
        raise RuntimeError("room_code_exhausted")

    def create(self, player_name: str) -> tuple[Room, str, int]:
        room = Room(self._new_code(), player_name)
        self.rooms[room.code] = room
        return room, room.host_token, 0

    def get(self, code: str) -> Room:
        try:
            return self.rooms[code]
        except KeyError as exc:
            raise ValueError("room_not_found") from exc

    async def cleanup(self) -> None:
        while True:
            await asyncio.sleep(15)
            now = time.monotonic()
            expired = [code for code, room in self.rooms.items() if room.expired(now)]
            for code in expired:
                room = self.rooms.pop(code)
                if room.match_task is not None:
                    room.match_task.cancel()
