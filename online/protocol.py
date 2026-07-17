from __future__ import annotations

import json
import re
from typing import Any, Mapping

from src.simulation import INPUT_FIELDS


PROTOCOL_VERSION = 1
TICK_MS = 25
MAX_CLIENT_MESSAGE_BYTES = 16_384
MAX_SERVER_MESSAGE_BYTES = 2_000_000
ROOM_CODE_RE = re.compile(r"^[A-Z2-9]{6}$")
EDGE_FIELDS = frozenset(
    {
        "jump_pressed",
        "punch_pressed",
        "special_pressed",
        "shield_pressed",
        "shield_released",
    }
)
HOLD_FIELDS = tuple(field for field in INPUT_FIELDS if field not in EDGE_FIELDS)

FIGHTERS = (
    "SBLPlayer",
    "PeachPlayer",
    "TrashPlayer",
    "CoffeePlayer",
    "DefaultPlayer",
    "AuberginePlayer",
)
STAGES = ("Rooftop", "Mogadishu", "B52", "Space")


class ProtocolError(ValueError):
    pass


def encode_message(message: Mapping[str, Any]) -> str:
    payload = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
    if len(payload.encode("utf-8")) > MAX_SERVER_MESSAGE_BYTES:
        raise ProtocolError("message_too_large")
    return payload


def decode_message(payload: str | bytes) -> dict[str, Any]:
    if isinstance(payload, bytes):
        if len(payload) > MAX_CLIENT_MESSAGE_BYTES:
            raise ProtocolError("message_too_large")
        try:
            payload = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ProtocolError("invalid_utf8") from exc
    if len(payload.encode("utf-8")) > MAX_CLIENT_MESSAGE_BYTES:
        raise ProtocolError("message_too_large")
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProtocolError("invalid_json") from exc
    if not isinstance(value, dict):
        raise ProtocolError("message_must_be_object")
    op = value.get("op")
    if not isinstance(op, str) or not op:
        raise ProtocolError("missing_op")
    return value


def room_code(value: Any) -> str:
    code = str(value or "").strip().upper()
    if not ROOM_CODE_RE.fullmatch(code):
        raise ProtocolError("invalid_room_code")
    return code


def clean_name(value: Any) -> str:
    name = " ".join(str(value or "玩家").strip().split())
    return name[:16] or "玩家"


def normalize_controls(value: Any) -> dict[str, bool]:
    source = value if isinstance(value, Mapping) else {}
    return {field: bool(source.get(field, False)) for field in INPUT_FIELDS}


def controls_without_edges(value: Mapping[str, Any]) -> dict[str, bool]:
    return {
        field: bool(value.get(field, False)) if field not in EDGE_FIELDS else False
        for field in INPUT_FIELDS
    }


def validate_lobby_patch(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError("invalid_lobby_patch")
    patch: dict[str, Any] = {}
    if "fighter" in value:
        fighter = str(value["fighter"])
        if fighter not in FIGHTERS:
            raise ProtocolError("invalid_fighter")
        patch["fighter"] = fighter
    if "color" in value:
        patch["color"] = max(0, min(3, int(value["color"])))
    if "level" in value:
        patch["level"] = max(1, min(22, int(value["level"])))
    if "enabled" in value:
        patch["enabled"] = bool(value["enabled"])
    if "ready" in value:
        patch["ready"] = bool(value["ready"])
    return patch


def validate_settings_patch(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError("invalid_settings_patch")
    patch: dict[str, Any] = {}
    if "stage" in value:
        stage = str(value["stage"])
        if stage not in STAGES:
            raise ProtocolError("invalid_stage")
        patch["stage"] = stage
    if "stock" in value:
        patch["stock"] = max(1, min(20, int(value["stock"])))
    if "items" in value:
        patch["items"] = bool(value["items"])
    return patch
