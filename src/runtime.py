from __future__ import annotations

import json
import math
import random
import webbrowser
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pygame

try:
    from .assets import LazySurfaceMap, LazySurfaceSequence, SURFACE_CACHE
    from .audio import AudioManager
    from .menu import MainMenu, MatchResults, MenuAction
except ImportError:
    from assets import LazySurfaceMap, LazySurfaceSequence, SURFACE_CACHE
    from audio import AudioManager
    from menu import MainMenu, MatchResults, MenuAction


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "assets/manifests/glorton_manifest.json"
TICK_MS = 25
ANIMATION_FPS = 30
PLAYER_REFLEX_MS = 50
WINDOW_SIZE = (1280, 760)
PANEL_WIDTH = 300
STARTING_LIVES = 5
ACTIVE_SPAWNS = ("SpawnP1", "SpawnP2", "SpawnP3", "SpawnP4")
RESPAWN_INVINCIBLE_MS = 3000
DEATH_EFFECT_MS = 900
CAMERA_TRICK_MS = 1000
BODY_HALF_WIDTH = 10
BODY_HEIGHT = 42
FOOT_RADIUS = 10
CAMERA_PADDING = 50
CAMERA_LERP = 5
CAMERA_RATIO = 600 / 400


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit("Manifest missing. Run: ../venv/bin/python tools/build_manifest.py")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@dataclass
class Platform:
    name: str
    rect: pygame.Rect
    moving: bool
    prev_rect: pygame.Rect | None = None


@dataclass
class StageFrame:
    image_path: Path
    offset: pygame.Vector2
    render_scale: float = 1.0


@dataclass
class SpriteAssetFrame:
    image_path: Path
    offset: pygame.Vector2
    render_scale: float = 1.0

    @property
    def image(self) -> pygame.Surface:
        return SURFACE_CACHE.get(self.image_path)


@dataclass
class Bullet:
    pos: pygame.Vector2
    xinc: float
    image: pygame.Surface
    sender: "PeachFighter"
    offset: pygame.Vector2
    source_scale: float = 1.0
    age: int = 0
    life: int = 3000
    alive: bool = True

    def fixed_tick(self, stage: "Stage") -> None:
        self.prev_pos = pygame.Vector2(self.pos)
        self.pos.x += self.xinc
        self.age += TICK_MS
        if self.age > self.life or not stage.bounds.collidepoint(self.pos.x, self.pos.y):
            self.alive = False
            return
        if stage.projectile_hits_fixed(self.hitbox()):
            self.alive = False

    def render_pos(self, alpha: float) -> pygame.Vector2:
        start = getattr(self, "prev_pos", self.pos)
        return start.lerp(self.pos, max(0.0, min(1.0, alpha)))

    def hitbox(self) -> pygame.Rect:
        source_scale = max(1.0, self.source_scale)
        width = self.image.get_width() / source_scale
        height = self.image.get_height() / source_scale
        left = self.offset.x * 0.5
        right = (self.offset.x + width) * 0.5
        if self.xinc < 0:
            left, right = -right, -left
        top = self.offset.y * 0.5
        bottom = (self.offset.y + height) * 0.5
        return pygame.Rect(
            math.floor(self.pos.x + left),
            math.floor(self.pos.y + top),
            max(1, math.ceil(self.pos.x + right) - math.floor(self.pos.x + left)),
            max(1, math.ceil(self.pos.y + bottom) - math.floor(self.pos.y + top)),
        )

    def draw_center_offset(self) -> pygame.Vector2:
        source_scale = max(1.0, self.source_scale)
        center = pygame.Vector2(
            (self.offset.x + self.image.get_width() / source_scale / 2) * 0.5,
            (self.offset.y + self.image.get_height() / source_scale / 2) * 0.5,
        )
        if self.xinc < 0:
            center.x *= -1
        return center


@dataclass
class RocketProjectile:
    pos: pygame.Vector2
    xinc: float
    yinc: float
    rotation: float
    image: pygame.Surface
    sender: "PeachFighter"
    offset: pygame.Vector2
    source_scale: float = 1.0
    mirrored: bool = False
    age: int = 0
    life: int = 3000
    alive: bool = True

    def fixed_tick(self, stage: "Stage") -> None:
        self.prev_pos = pygame.Vector2(self.pos)
        self.pos.x += self.xinc
        self.yinc += 0.5
        self.pos.y += self.yinc
        self.rotation = math.atan2(self.yinc, self.xinc) * 180 / math.pi + 90
        self.age += TICK_MS
        if self.age > self.life:
            self.alive = False

    def render_pos(self, alpha: float) -> pygame.Vector2:
        start = getattr(self, "prev_pos", self.pos)
        return start.lerp(self.pos, max(0.0, min(1.0, alpha)))

    def hitbox(self) -> pygame.Rect:
        source_scale = max(1.0, self.source_scale)
        width = self.image.get_width() / source_scale
        height = self.image.get_height() / source_scale
        corners = [
            pygame.Vector2(self.offset),
            pygame.Vector2(self.offset.x + width, self.offset.y),
            pygame.Vector2(self.offset.x, self.offset.y + height),
            pygame.Vector2(self.offset.x + width, self.offset.y + height),
        ]
        if self.mirrored:
            for point in corners:
                point.x *= -1
        corners = [point.rotate(self.rotation) for point in corners]
        left = self.pos.x + min(point.x for point in corners)
        top = self.pos.y + min(point.y for point in corners)
        right = self.pos.x + max(point.x for point in corners)
        bottom = self.pos.y + max(point.y for point in corners)
        return pygame.Rect(
            math.floor(left),
            math.floor(top),
            max(1, math.ceil(right) - math.floor(left)),
            max(1, math.ceil(bottom) - math.floor(top)),
        )

    def draw_center_offset(self) -> pygame.Vector2:
        source_scale = max(1.0, self.source_scale)
        center = self.offset + pygame.Vector2(
            self.image.get_width() / source_scale / 2,
            self.image.get_height() / source_scale / 2,
        )
        if self.mirrored:
            center.x *= -1
        return center.rotate(self.rotation)


@dataclass
class SpecialProjectile:
    kind: str
    pos: pygame.Vector2
    xinc: float
    yinc: float
    rotation: float
    frames: list[SpriteAssetFrame]
    sender: "PeachFighter"
    config: dict[str, object]
    variant: int = 1
    display_scale: float = 1.0
    facing: int = 1
    age: int = 0
    alive: bool = True
    kos: int = 0
    osd_score_event: str = ""
    osd_score_age_ms: int = 0

    @property
    def name(self) -> str:
        return self.kind

    @property
    def weight(self) -> float:
        return self.sender.weight

    @property
    def frame(self) -> SpriteAssetFrame | None:
        if not self.frames:
            return None
        if self.kind == "Garbage":
            index = max(0, min(len(self.frames) - 1, self.variant - 1))
        else:
            frame_no = int(self.age * ANIMATION_FPS / 1000) + 1
            playback = self.config.get("playback", {})
            stop_at = int(playback.get("stop_at", 0))
            if stop_at > 0:
                frame_no = min(stop_at, frame_no)
            else:
                loop_at = int(playback.get("loop_at", len(self.frames)))
                loop_from = int(playback.get("loop_from", 1))
                if frame_no > loop_at:
                    loop_length = max(1, loop_at - loop_from + 1)
                    frame_no = loop_from + ((frame_no - loop_at - 1) % loop_length)
            index = max(0, min(len(self.frames) - 1, frame_no - 1))
        return self.frames[index]

    def fixed_tick(self, stage: "Stage") -> None:
        self.prev_pos = pygame.Vector2(self.pos)
        self.pos.x += self.xinc
        if self.kind == "EnergyBall":
            degrees_per_x = float(self.config.get("sine_degrees_per_x", 10))
            amplitude = float(self.config.get("sine_amplitude", 2))
            self.pos.y += math.sin(self.pos.x * math.pi / 180 * degrees_per_x) * amplitude
        else:
            self.yinc += float(self.config.get("gravity_per_tick", 0))
            self.pos.y += self.yinc
        self.rotation += float(self.config.get("rotation_per_tick", 0))
        self.age += TICK_MS
        if self.age > int(self.config.get("life_ms", 0)):
            self.alive = False
            return
        if stage.projectile_hits_fixed(self.hitbox()):
            if bool(self.config.get("bounce_on_wall", False)):
                self._bounce_x()
            else:
                self.alive = False
        if not stage.bounds.collidepoint(self.pos.x, self.pos.y):
            if bool(self.config.get("bounce_on_wall", False)):
                # EnergyBall.DoCommon calls HitWall() for Bounds as well as
                # fixed geometry. Its HitWall reverses x even at top/bottom.
                self._bounce_x()
            else:
                self.alive = False

    def _bounce_x(self) -> None:
        self.xinc *= -1
        self.pos.x += 15 if self.xinc > 0 else -15

    def render_pos(self, alpha: float) -> pygame.Vector2:
        start = getattr(self, "prev_pos", self.pos)
        return start.lerp(self.pos, max(0.0, min(1.0, alpha)))

    def hitbox(self) -> pygame.Rect:
        frame = self.frame
        if frame is None:
            return pygame.Rect(round(self.pos.x), round(self.pos.y), 1, 1)
        source_scale = max(1.0, frame.render_scale)
        width = frame.image.get_width() / source_scale * self.display_scale
        height = frame.image.get_height() / source_scale * self.display_scale
        offset = frame.offset * self.display_scale
        corners = [
            pygame.Vector2(offset),
            pygame.Vector2(offset.x + width, offset.y),
            pygame.Vector2(offset.x, offset.y + height),
            pygame.Vector2(offset.x + width, offset.y + height),
        ]
        if self.facing < 0:
            for point in corners:
                point.x *= -1
        if self.rotation:
            corners = [point.rotate(self.rotation) for point in corners]
        left = self.pos.x + min(point.x for point in corners)
        top = self.pos.y + min(point.y for point in corners)
        right = self.pos.x + max(point.x for point in corners)
        bottom = self.pos.y + max(point.y for point in corners)
        return pygame.Rect(
            math.floor(left),
            math.floor(top),
            max(1, math.ceil(right) - math.floor(left)),
            max(1, math.ceil(bottom) - math.floor(top)),
        )

    def draw_center_offset(self) -> pygame.Vector2:
        frame = self.frame
        if frame is None:
            return pygame.Vector2()
        source_scale = max(1.0, frame.render_scale)
        center = frame.offset * self.display_scale + pygame.Vector2(
            frame.image.get_width() / source_scale * self.display_scale / 2,
            frame.image.get_height() / source_scale * self.display_scale / 2,
        )
        if self.facing < 0:
            center.x *= -1
        return center.rotate(self.rotation)


@dataclass
class StageItem:
    kind: str
    pos: pygame.Vector2
    frames: list[SpriteAssetFrame]
    frame_labels: dict[str, int] | None = None
    source_scale: float = 1.0
    life_ms: int = 20000
    age_ms: int = 0
    state: int = 1
    alive: bool = True
    sender: "PeachFighter | None" = None
    xinc: float = 0.0
    yinc: float = 0.0
    rotation: float = 0.0
    active_ms: int = 0
    active_platform: Platform | None = None
    active_offset: pygame.Vector2 | None = None
    influenced: set[int] | None = None

    def display_frame(self) -> SpriteAssetFrame | None:
        if not self.frames:
            return None
        labels = self.frame_labels or {}
        label = "active" if self.state == 0 else "airbone" if self.state == 2 else "spawn"
        frame_no = labels.get(label)
        if frame_no is not None and 1 <= frame_no <= len(self.frames):
            frame = self.frames[frame_no - 1]
            if frame.image.get_bounding_rect().w > 0 and frame.image.get_bounding_rect().h > 0:
                return frame
        return next(
            (
                frame
                for frame in self.frames
                if frame.image.get_bounding_rect().w > 0 and frame.image.get_bounding_rect().h > 0
            ),
            self.frames[0],
        )

    @property
    def visible(self) -> bool:
        if not self.alive or self.state == 3:
            return False
        if self.state == 1 and self.age_ms + 3000 > self.life_ms and self.age_ms % 2 == 0:
            return False
        return True

    def fixed_tick(self, stage: "Stage") -> None:
        self.age_ms += TICK_MS
        if self.state == 3:
            return
        if self.state == 0:
            self.active_ms += TICK_MS
            if self.active_platform is not None and self.active_offset is not None:
                self.pos.x = self.active_platform.rect.x + self.active_offset.x
                self.pos.y = self.active_platform.rect.y + self.active_offset.y
            return
        if self.state == 1:
            if self.age_ms > self.life_ms:
                self.alive = False
            return
        if self.state == 2:
            self.prev_pos = pygame.Vector2(self.pos)
            self.pos.x += self.xinc
            if self.yinc > 6:
                self.yinc = 6
            else:
                self.yinc += 0.6
            self.pos.y += self.yinc
            if not stage.bounds.collidepoint(self.pos.x, self.pos.y):
                self.alive = False
    def hitbox_at(self, pos: pygame.Vector2) -> pygame.Rect:
        asset = self.display_frame()
        if asset is None:
            return pygame.Rect(round(pos.x), round(pos.y), 1, 1)
        scale = max(1.0, asset.render_scale)
        width = asset.image.get_width() / scale
        height = asset.image.get_height() / scale
        corners = [
            pygame.Vector2(asset.offset),
            pygame.Vector2(asset.offset.x + width, asset.offset.y),
            pygame.Vector2(asset.offset.x, asset.offset.y + height),
            pygame.Vector2(asset.offset.x + width, asset.offset.y + height),
        ]
        if self.rotation:
            corners = [point.rotate(self.rotation) for point in corners]
        left = pos.x + min(point.x for point in corners)
        top = pos.y + min(point.y for point in corners)
        right = pos.x + max(point.x for point in corners)
        bottom = pos.y + max(point.y for point in corners)
        return pygame.Rect(
            math.floor(left),
            math.floor(top),
            max(1, math.ceil(right) - math.floor(left)),
            max(1, math.ceil(bottom) - math.floor(top)),
        )

    def hitbox(self) -> pygame.Rect:
        return self.hitbox_at(self.pos)

    def throw(self, power: float, angle: float) -> None:
        if angle < 0:
            angle = 180 + angle
        radians = angle * math.pi / 180
        self.xinc = math.cos(radians) * power
        self.yinc = -math.sin(radians) * power
        self.state = 2
        self.active_platform = None
        self.active_offset = None
        self.active_ms = 0


@dataclass
class DeathEvent:
    pos: pygame.Vector2
    death_type: str
    fighter_name: str
    killer_name: str | None


@dataclass
class DeathEffect:
    pos: pygame.Vector2
    death_type: str
    frames: list[SpriteAssetFrame]
    age_ms: int = 0
    life_ms: int = DEATH_EFFECT_MS
    frame_rate: int = ANIMATION_FPS

    @property
    def frame_index(self) -> int:
        if not self.frames:
            return 0
        return min(len(self.frames) - 1, int(self.age_ms * self.frame_rate / 1000))

    @property
    def alive(self) -> bool:
        if self.frames:
            return self.frame_index < max(0, len(self.frames) - 1)
        return self.age_ms <= self.life_ms


@dataclass
class SpawnEffect:
    pos: pygame.Vector2
    frames: list[SpriteAssetFrame]
    kind: int
    age_ms: int = 0
    frame_rate: int = ANIMATION_FPS

    @property
    def frame_index(self) -> int:
        if not self.frames:
            return 0
        return min(len(self.frames) - 1, int(self.age_ms * self.frame_rate / 1000))

    @property
    def alive(self) -> bool:
        return self.frame_index < max(0, len(self.frames) - 1)


@dataclass
class HitEffect:
    pos: pygame.Vector2
    frames: list[SpriteAssetFrame]
    rotation: float
    scale: float
    source_scale: float = 1.0
    root_layer: bool = False
    age_ms: int = 0
    frame_rate: int = ANIMATION_FPS

    @property
    def frame_index(self) -> int:
        if not self.frames:
            return 0
        return min(len(self.frames) - 1, int(self.age_ms * self.frame_rate / 1000))

    @property
    def alive(self) -> bool:
        return bool(self.frames) and self.frame_index < max(0, len(self.frames) - 1)


@dataclass
class ExplosionEffect:
    pos: pygame.Vector2
    size: int
    sender: "PeachFighter | None"
    frames: list[SpriteAssetFrame]
    wave_frames: list[SpriteAssetFrame]
    matter_frames: list[SpriteAssetFrame]
    matter_offsets: list[pygame.Vector2]
    age_ms: int = 0
    frame_rate: int = ANIMATION_FPS
    influenced: set[int] | None = None

    @property
    def square_size(self) -> float:
        return self.size * 10 * (self.size * 10) + 400

    @property
    def frame_index(self) -> int:
        if not self.frames:
            return 0
        return min(len(self.frames) - 1, int(self.age_ms * self.frame_rate / 1000))

    @property
    def wave_frame_index(self) -> int:
        if not self.wave_frames:
            return 0
        return min(len(self.wave_frames) - 1, int(self.age_ms * self.frame_rate / 1000))

    @property
    def matter_frame_index(self) -> int:
        if not self.matter_frames:
            return 0
        return min(len(self.matter_frames) - 1, int(self.age_ms * self.frame_rate / 1000))

    @property
    def damage_active(self) -> bool:
        return bool(self.matter_frames) and self.matter_frame_index < len(self.matter_frames) - 1

    @property
    def alive(self) -> bool:
        return bool(self.wave_frames) and self.wave_frame_index < max(0, len(self.wave_frames) - 1)


@dataclass
class FighterInput:
    left_keys: set[int]
    right_keys: set[int]
    up_keys: set[int]
    down_keys: set[int]
    jump_keys: set[int]
    punch_keys: set[int]
    special_keys: set[int]
    shield_keys: set[int]
    held_left: bool = False
    held_right: bool = False
    held_up: bool = False
    horizontal_direction: str = "stop"
    up_hold_ms: int = 0
    up_trace: bool = False
    pending_jump_pressed: bool = False
    pending_punch_pressed: bool = False
    pending_special_pressed: bool = False
    pending_shield_pressed: bool = False
    pending_shield_released: bool = False

    def keydown(self, key: int) -> str | None:
        if key in self.left_keys:
            self.held_left = True
            self.horizontal_direction = "left"
            return "left"
        if key in self.right_keys:
            self.held_right = True
            self.horizontal_direction = "right"
            return "right"
        if key in self.up_keys:
            self.held_up = True
            self.up_hold_ms = 0
            self.up_trace = True
            return None
        if key in self.jump_keys:
            self.pending_jump_pressed = True
            return None
        if key in self.punch_keys:
            self.pending_punch_pressed = True
            return None
        if key in self.special_keys:
            self.pending_special_pressed = True
            return None
        if key in self.shield_keys:
            self.pending_shield_pressed = True
            return None
        return None

    def keyup(self, key: int) -> str | None:
        if key in self.left_keys:
            self.held_left = False
            if self.held_right:
                return None
            self.horizontal_direction = "stop"
            return "stop"
        if key in self.right_keys:
            self.held_right = False
            if self.held_left:
                return None
            self.horizontal_direction = "stop"
            return "stop"
        if key in self.up_keys:
            self.held_up = False
            self.up_hold_ms = 0
            self.up_trace = False
            self.pending_jump_pressed = True
            return None
        if key in self.down_keys:
            return "stop"
        if key in self.shield_keys:
            self.pending_shield_released = True
            return None
        return None

    def controls(self, keys: pygame.key.ScancodeWrapper) -> dict[str, bool]:
        jump_pressed = self.pending_jump_pressed
        if self.held_up:
            self.up_hold_ms += TICK_MS
            if self.up_hold_ms < PLAYER_REFLEX_MS:
                self.up_trace = True
            else:
                self.up_trace = False
                jump_pressed = True
        controls = {
            "left": self.horizontal_direction == "left",
            "right": self.horizontal_direction == "right",
            "up_trace": self.up_trace,
            "down": any(keys[key] for key in self.down_keys),
            "jump_pressed": jump_pressed,
            "punch_pressed": self.pending_punch_pressed,
            "special_pressed": self.pending_special_pressed,
            "shield_pressed": self.pending_shield_pressed,
            "shield_released": self.pending_shield_released,
        }
        if self.up_trace and (self.pending_punch_pressed or self.pending_special_pressed):
            self.up_trace = False
        self.pending_jump_pressed = False
        self.pending_punch_pressed = False
        self.pending_special_pressed = False
        self.pending_shield_pressed = False
        self.pending_shield_released = False
        return controls


class Stage:
    def __init__(self, manifest: dict, stage_name: str | None = None) -> None:
        stages = manifest.get("stages", {})
        selected = stage_name or str(manifest.get("stage", {}).get("name", "Rooftop"))
        self.data = stages.get(selected, manifest["stage"])
        self.name = str(self.data.get("name", selected))
        self.bounds = pygame.Rect(
            self.data["bounds"]["x"],
            self.data["bounds"]["y"],
            self.data["bounds"]["w"],
            self.data["bounds"]["h"],
        )
        self.bounds_cam = pygame.Rect(
            self.data["bounds_cam"]["x"],
            self.data["bounds_cam"]["y"],
            self.data["bounds_cam"]["w"],
            self.data["bounds_cam"]["h"],
        )
        view_bounds = self.data.get("view_bounds", self.data["bounds_cam"])
        self.view_bounds = pygame.Rect(
            view_bounds["x"],
            view_bounds["y"],
            view_bounds["w"],
            view_bounds["h"],
        )
        self.background = pygame.image.load(str(ROOT / self.data["background"]))
        self.background_scale = max(1.0, float(self.data.get("background_size", {}).get("render_scale", 1)))
        background_offset = self.data.get("background_offset", {"x": 0.0, "y": 0.0})
        self.background_offset = pygame.Vector2(
            float(background_offset.get("x", 0.0)),
            float(background_offset.get("y", 0.0)),
        )
        background_animation = self.data.get("background_animation", {})
        background_size = self.data.get("background_size", {})
        self.background_canvas_size = pygame.Vector2(
            float(background_animation.get("canvas_size", {}).get("w", background_size.get("w", self.background.get_width() / self.background_scale))),
            float(background_animation.get("canvas_size", {}).get("h", background_size.get("h", self.background.get_height() / self.background_scale))),
        )
        self.background_layers: list[tuple[int, int, list[StageFrame]]] = []
        for layer in background_animation.get("layers", []):
            frames = [
                StageFrame(
                    image_path=ROOT / item["image"],
                    offset=pygame.Vector2(float(item["offset"]["x"]), float(item["offset"]["y"])),
                    render_scale=max(1.0, float(item.get("render_scale", 1))),
                )
                for item in layer.get("frames", [])
            ]
            if frames:
                self.background_layers.append(
                    (
                        int(layer.get("frame_rate", ANIMATION_FPS)),
                        max(1, int(layer.get("loop_from", 1))),
                        frames,
                    )
                )
        self._background_layer_cache: dict[tuple[int, int], pygame.Surface] = {}
        self.background_object_layers: list[dict[str, object]] = []
        for layer in background_animation.get("object_layers", []):
            sprite = layer.get("sprite", {})
            image_path = sprite.get("image")
            if not image_path:
                continue
            self.background_object_layers.append(
                {
                    "frame_rate": int(layer.get("frame_rate", ANIMATION_FPS)),
                    "surface": pygame.image.load(str(ROOT / str(image_path))),
                    "render_scale": max(1.0, float(sprite.get("render_scale", 1))),
                    "offset": pygame.Vector2(
                        float(sprite.get("offset", {}).get("x", 0.0)),
                        float(sprite.get("offset", {}).get("y", 0.0)),
                    ),
                    "logical_size": pygame.Vector2(
                        float(sprite.get("logical_size", {}).get("w", 0.0)),
                        float(sprite.get("logical_size", {}).get("h", 0.0)),
                    ),
                    "frames": [frame.get("matrices", []) for frame in layer.get("frames", [])],
                }
            )
        foreground = self.data.get("foreground")
        self.foreground = pygame.image.load(str(ROOT / foreground)) if foreground else None
        self.foreground_scale = max(1.0, float(self.data.get("foreground_size", {}).get("render_scale", 1)))
        foreground_offset = self.data.get("foreground_offset", {"x": 0.0, "y": 0.0})
        self.foreground_offset = pygame.Vector2(float(foreground_offset["x"]), float(foreground_offset["y"]))
        dynamic = self.data.get("dynamic_layer", {})
        self.dynamic_frame_rate = int(dynamic.get("frame_rate", ANIMATION_FPS))
        self.dynamic_above_foreground = bool(self.data.get("dynamic_above_foreground", False))
        self.dynamic_frames = [
            StageFrame(
                image_path=ROOT / item["image"],
                offset=pygame.Vector2(float(item["offset"]["x"]), float(item["offset"]["y"])),
                render_scale=max(1.0, float(item.get("render_scale", 1))),
            )
            for item in dynamic.get("frames", [])
        ]
        self._dynamic_cache: OrderedDict[int, pygame.Surface] = OrderedDict()
        helicopter = self.data.get("helicopter", {})
        self.helicopter_frame_rate = int(helicopter.get("frame_rate", ANIMATION_FPS))
        self.helicopter_frames = [
            StageFrame(
                image_path=ROOT / item["image"],
                offset=pygame.Vector2(float(item["offset"]["x"]), float(item["offset"]["y"])),
                render_scale=max(1.0, float(item.get("render_scale", 1))),
            )
            for item in helicopter.get("frames", [])
        ]
        self._helicopter_cache: OrderedDict[int, pygame.Surface] = OrderedDict()
        moving_platforms = self.data.get("moving_platforms", {})
        self.moving_platform_frame_rate = int(moving_platforms.get("frame_rate", ANIMATION_FPS))
        self.moving_platform_timelines = {
            name: [self._rect_from_data(frame["rect"]) for frame in frames]
            for name, frames in moving_platforms.get("platforms", {}).items()
        }
        self.platforms = self._build_platforms()
        self.boom_rects = self._helper_rects("Boom")
        self.killer_rects = self._helper_rects("Killer")
        self.set_time(0)

    def _build_platforms(self) -> list[Platform]:
        platforms: list[Platform] = []
        for obj in self.data["objects"]:
            name = obj["name"]
            if not (name.startswith("Fixed") or name.startswith("Moving")):
                continue
            rect = obj["estimated_rect"]
            if rect["w"] <= 0 or rect["h"] <= 0:
                continue
            platform_rect = self._rect_from_data(rect)
            platforms.append(
                Platform(
                    name=name,
                    rect=platform_rect,
                    moving=name.startswith("Moving"),
                    prev_rect=platform_rect.copy(),
                )
            )
        return platforms

    def _helper_rects(self, prefix: str) -> list[pygame.Rect]:
        return [
            self._rect_from_data(obj["estimated_rect"])
            for obj in self.data.get("objects", [])
            if str(obj.get("name", "")).startswith(prefix)
            and float(obj.get("estimated_rect", {}).get("w", 0)) > 0
            and float(obj.get("estimated_rect", {}).get("h", 0)) > 0
        ]

    def _rect_from_data(self, rect: dict[str, float]) -> pygame.Rect:
        return pygame.Rect(
            round(rect["x"]),
            round(rect["y"]),
            max(1, round(rect["w"])),
            max(1, round(rect["h"])),
        )

    def set_time(self, elapsed_ms: int) -> None:
        for platform in self.platforms:
            platform.prev_rect = platform.rect.copy()
            timeline = self.moving_platform_timelines.get(platform.name)
            if timeline:
                frame_index = int(elapsed_ms * self.moving_platform_frame_rate / 1000) % len(timeline)
                platform.rect = timeline[frame_index].copy()

    def helicopter_surface(self, frame_index: int) -> pygame.Surface:
        if frame_index in self._helicopter_cache:
            surface = self._helicopter_cache.pop(frame_index)
            self._helicopter_cache[frame_index] = surface
            return surface
        surface = pygame.image.load(str(self.helicopter_frames[frame_index].image_path))
        self._helicopter_cache[frame_index] = surface
        while len(self._helicopter_cache) > 8:
            self._helicopter_cache.popitem(last=False)
        return surface

    def dynamic_surface(self, frame_index: int) -> pygame.Surface:
        if frame_index in self._dynamic_cache:
            surface = self._dynamic_cache.pop(frame_index)
            self._dynamic_cache[frame_index] = surface
            return surface
        surface = pygame.image.load(str(self.dynamic_frames[frame_index].image_path))
        self._dynamic_cache[frame_index] = surface
        while len(self._dynamic_cache) > 8:
            self._dynamic_cache.popitem(last=False)
        return surface

    def background_layer_surface(self, layer_index: int, frame_index: int) -> pygame.Surface:
        key = (layer_index, frame_index)
        cached = self._background_layer_cache.get(key)
        if cached is not None:
            return cached
        frame = self.background_layers[layer_index][2][frame_index]
        surface = pygame.image.load(str(frame.image_path))
        if len(self._background_layer_cache) > 12:
            self._background_layer_cache.clear()
        self._background_layer_cache[key] = surface
        return surface

    def background_frame_key(self, elapsed_ms: int) -> tuple[int, ...]:
        raster_frames = tuple(
            self._background_frame_index(elapsed_ms, frame_rate, loop_from, len(frames))
            for frame_rate, loop_from, frames in self.background_layers
        )
        object_frames = tuple(
            int(elapsed_ms * int(layer["frame_rate"]) / 1000) % len(layer["frames"])
            for layer in self.background_object_layers
            if layer["frames"]
        )
        return raster_frames + object_frames

    @staticmethod
    def _background_frame_index(elapsed_ms: int, frame_rate: int, loop_from: int, length: int) -> int:
        if length <= 1:
            return 0
        elapsed_frame = max(0, int(elapsed_ms * frame_rate / 1000))
        loop_index = max(0, min(length - 1, loop_from - 1))
        if elapsed_frame < loop_index:
            return elapsed_frame
        return loop_index + (elapsed_frame - loop_index) % max(1, length - loop_index)

    def spawn_point(self, name: str = "SpawnP1") -> pygame.Vector2:
        for obj in self.data["objects"]:
            if obj["name"] == name:
                matrix = obj["matrix"]
                return pygame.Vector2(float(matrix["x"]), float(matrix["y"]))
        return pygame.Vector2(self.bounds_cam.centerx, self.bounds_cam.centery)

    def death_respawn_point(self) -> pygame.Vector2:
        # Fighter.Die() respawns at _x = random(400) + 100, _y = 0.
        return pygame.Vector2(random.randrange(400) + 100, 0)

    def item_spawn_point(self) -> pygame.Vector2:
        zones = {
            obj["name"]: obj
            for obj in self.data["objects"]
            if obj["name"].startswith("SpawnH")
        }
        if not zones:
            return pygame.Vector2(self.bounds_cam.centerx, self.bounds_cam.top)
        zone_index = random.randrange(4) + 1
        zone = zones.get(f"SpawnH{zone_index}")
        if zone is None:
            zone = next(iter(zones.values()))
        rect = zone["estimated_rect"]
        width = float(rect["w"])
        if width <= 0:
            width = float(zone.get("source_size", {}).get("w", 0))
        width = max(1.0, width)
        matrix = zone["matrix"]
        return pygame.Vector2(float(matrix["x"]) + random.randrange(max(1, int(width))), float(matrix["y"]))

    def find_floor_crossing(
        self,
        x: float,
        old_y: float,
        new_y: float,
        foot_radius: float,
        ignored: Platform | None = None,
    ) -> Platform | None:
        if new_y < old_y:
            return None
        candidates: list[Platform] = []
        for platform in self.platforms:
            if platform is ignored:
                continue
            top = platform.rect.top
            if old_y <= top <= new_y and platform.rect.left - foot_radius <= x <= platform.rect.right + foot_radius:
                candidates.append(platform)
        if not candidates:
            return None
        return min(candidates, key=lambda item: item.rect.top)

    def find_ceiling_crossing(
        self,
        x: float,
        old_head_y: float,
        new_head_y: float,
        half_width: float,
        ignored: Platform | None = None,
    ) -> Platform | None:
        if new_head_y > old_head_y:
            return None
        candidates: list[Platform] = []
        for platform in self.platforms:
            if platform is ignored or platform.moving:
                continue
            bottom = platform.rect.bottom
            if new_head_y <= bottom <= old_head_y and platform.rect.left - half_width <= x <= platform.rect.right + half_width:
                candidates.append(platform)
        if not candidates:
            return None
        return max(candidates, key=lambda item: item.rect.bottom)

    def find_side_crossing(
        self,
        old_x: float,
        new_x: float,
        top: float,
        bottom: float,
        half_width: float,
        ignored: Platform | None = None,
        include_moving: bool = True,
    ) -> Platform | None:
        if new_x == old_x:
            return None
        candidates: list[Platform] = []
        moving_right = new_x > old_x
        old_side = old_x + half_width if moving_right else old_x - half_width
        new_side = new_x + half_width if moving_right else new_x - half_width
        for platform in self.platforms:
            if platform is ignored or (platform.moving and not include_moving):
                continue
            if bottom <= platform.rect.top + 1 or top >= platform.rect.bottom - 1:
                continue
            wall_x = platform.rect.left if moving_right else platform.rect.right
            if min(old_side, new_side) <= wall_x <= max(old_side, new_side):
                candidates.append(platform)
        if not candidates:
            return None
        if moving_right:
            return min(candidates, key=lambda item: item.rect.left)
        return max(candidates, key=lambda item: item.rect.right)

    def projectile_hits_fixed(self, projectile_box: pygame.Rect) -> bool:
        # Projectile.as calls CollisionDetect(this, true, true, false, false):
        # platforms are checked, moving platforms are not.
        return any(not platform.moving and projectile_box.colliderect(platform.rect) for platform in self.platforms)

    def item_hits_platform(self, item_box: pygame.Rect) -> bool:
        return any(item_box.colliderect(platform.rect) for platform in self.platforms)

    def item_hit_platform(self, item_box: pygame.Rect) -> Platform | None:
        return next((platform for platform in self.platforms if item_box.colliderect(platform.rect)), None)

    def ai_helper_type(self, fighter_box: pygame.Rect) -> str | None:
        for obj in self.data.get("objects", []):
            name = str(obj.get("name", ""))
            if not name.startswith("AI_"):
                continue
            rect = self._rect_from_data(obj["estimated_rect"])
            if fighter_box.colliderect(rect):
                return name[3:4]
        return None


class PeachFighter:
    def __init__(
        self,
        manifest: dict,
        spawn: pygame.Vector2,
        name: str = "P1",
        color_index: int = 0,
        fighter_name: str = "PeachPlayer",
        team_index: int = 0,
    ) -> None:
        fighter = manifest.get("fighters", {}).get(fighter_name, manifest["fighter"])
        self.name = name
        self.team_index = max(0, min(3, int(team_index)))
        self.draw_depth = 0
        self.fighter_name = str(fighter.get("name", fighter_name))
        self.character_name = str(fighter.get("character_name", "PeachLock"))
        self.special_kind = str(fighter.get("special_kind", "peach_weapons"))
        self.spawn_pos = pygame.Vector2(spawn)
        self.fighter_data = fighter
        self.projectile_image_path = manifest["projectiles"]["Bullet"]["image"]
        self.rocket_image_path = manifest["projectiles"]["Rocket"]["image"]
        bullet_offset = manifest["projectiles"]["Bullet"].get("offset", {"x": 0, "y": 0})
        rocket_offset = manifest["projectiles"]["Rocket"].get("offset", {"x": 0, "y": 0})
        self.projectile_offset = pygame.Vector2(float(bullet_offset["x"]), float(bullet_offset["y"]))
        self.rocket_offset = pygame.Vector2(float(rocket_offset["x"]), float(rocket_offset["y"]))
        self.projectile_render_scale = max(1.0, float(manifest["projectiles"]["Bullet"].get("render_scale", 1)))
        self.rocket_render_scale = max(1.0, float(manifest["projectiles"]["Rocket"].get("render_scale", 1)))
        self.projectile_data = manifest.get("projectiles", {})
        self.special_projectile_cache: dict[str, list[SpriteAssetFrame]] = {}
        self.frames = LazySurfaceSequence(ROOT / item["raw"] for item in fighter["frames"])
        self.color_frame = max(1, min(4, int(color_index) + 1))
        state_animations = fighter.get("color_state_animations", {}).get(
            str(self.color_frame),
            fighter.get("state_animations", {}),
        )
        self.animations = self._load_animation_set(state_animations)
        self.fired_animations = self._load_animation_set(
            {
                name: animation
                for name, animation in state_animations.items()
                if animation.get("fired_frames")
            },
            frames_key="fired_frames",
        )
        self.held_item_animations = {
            item_name: self._load_animation_set(item_colors.get(str(self.color_frame), {}))
            for item_name, item_colors in fighter.get("held_item_state_animations", {}).items()
        }
        self.garbage_variant_animations = {
            int(variant): self._load_animation_set(variant_colors.get(str(self.color_frame), {}))
            for variant, variant_colors in fighter.get("garbage_variant_state_animations", {}).items()
        }
        self.state_offsets = {}
        labels_by_frame = {item["frame"]: item["name"] for item in fighter["timeline"]["labels"]}
        for place in fighter["timeline"]["named_places"]:
            label = labels_by_frame.get(place["frame"], place["name"])
            if label == "still" and place["name"] == "stil":
                label = "still"
            self.state_offsets[label] = pygame.Vector2(place["matrix"]["x"], place["matrix"]["y"])
        for label, animation in fighter.get("state_animations", {}).items():
            state_offset = animation.get("state_offset")
            if state_offset is not None:
                self.state_offsets[label] = pygame.Vector2(
                    float(state_offset["x"]),
                    float(state_offset["y"]),
                )
        still_animation = self.animations.get("still", {})
        still_metadata = still_animation.get("metadata", {}).get(1, {})
        still_offset = still_metadata.get("offset", {"x": 0.0, "y": -BODY_HEIGHT})
        still_size = still_metadata.get(
            "logical_size",
            {"w": BODY_HALF_WIDTH * 2, "h": BODY_HEIGHT},
        )
        still_state_offset = self.state_offsets.get("still", pygame.Vector2())
        collision_left = still_state_offset.x + float(still_offset["x"])
        collision_top = still_state_offset.y + float(still_offset["y"])
        collision_right = collision_left + float(still_size["w"])
        self.body_half_width = max(1, math.ceil(max(abs(collision_left), abs(collision_right))))
        self.body_height = max(1, math.ceil(abs(min(0.0, collision_top))))
        self.foot_radius = self.body_half_width
        self.label_to_frame = {item["name"]: item["frame"] - 1 for item in fighter["timeline"]["labels"]}
        self.weight = float(fighter["weight"])
        self.limit_mode = str(manifest.get("match", {}).get("limit_mode", "stock"))
        self.spawn_invincible_decay = int(
            manifest.get("match", {}).get("respawn_invincible_decrement_per_tick", TICK_MS)
        )
        self.speed = float(fighter["speed"])
        self.gravity = float(fighter["gravity_per_tick"])
        self.max_fall = float(fighter["max_fall_yinc"])
        self.move_xinc = float(fighter["base_move_xinc"])
        self.jump_yinc = float(fighter["jump_yinc"])
        self.pos = pygame.Vector2(spawn)
        self.prev_pos = pygame.Vector2(spawn)
        self.xinc = 0.0
        self.yinc = 0.0
        self.state = "stop"
        self.jumpstate = 0
        self.on_ground = False
        self.has_control = True
        self.ctrl_loss = 0
        self.paralized = 0
        self.electrocuted_ms = 0
        self.damage_amnt = 0
        self.combo = 2
        self.last_sender: PeachFighter | None = None
        self.time_ko = 0
        self.lives = int(manifest.get("match", {}).get("starting_lives", STARTING_LIVES))
        self.deaths = 0
        self.sds = 0
        self.kos = 0
        self.death_order: int | None = None
        self.dead = False
        self.last_death_type = ""
        self.spawn_invincible_ms = 0
        self.invincible = False
        self.blinky_cos = 0
        self.blinking = False
        self.facing = 1
        self.attack_facing = 1
        self.current_label = "still"
        self.animation_frame = 0
        self.animation_time_ms = 0
        self.current_attack = ""
        self.attack_frame = 0
        self.attack_done = False
        self.attack_pending_finish = ""
        self.bullet_shot = False
        self.garbage_variant = 1
        self.hit_targets: set[int] = set()
        self.pending_sounds: list[str] = []
        self.pending_stop_sounds: set[str] = set()
        self.move_queue = ""
        self.throw_victim: PeachFighter | None = None
        # Fighter.SpecUpOk starts undefined in AVM1 and is first enabled by
        # landing or Damage().
        self.spec_up_ok = False
        self.ground_platform: Platform | None = None
        self.go_through_platform: Platform | None = None
        self.shielded = False
        self.shield_size = 100.0
        self.pending_puffs: list[tuple[pygame.Vector2, float]] = []
        self.last_throw_puff_pos: pygame.Vector2 | None = None
        self.current_item = ""
        self.current_item_obj: StageItem | None = None
        self.last_collision = "-"
        self.dead_reason = ""
        self.out_of_camera = False
        self.spawn_reveal_ms = 0
        self.spawn_reveal_frame = 10
        self.spawn_age_ms = 0
        self.spawn_effect_kind = 1
        self.spawn_visual_alpha = 100.0
        self.spawn_white_offset = 0.0
        self.spawn_fighter_visible = True
        self.osd_damage_age_ms = 1000
        self.osd_score_event = ""
        self.osd_score_age_ms = 0
        self.pending_death_event: DeathEvent | None = None
        self.intro_visible = True
        self.resolve_attack_this_tick = False
        self.animate_this_tick = False
        self.pending_stage_boom = False

    @staticmethod
    def _load_animation_set(state_animations: dict, frames_key: str = "frames") -> dict[str, dict]:
        animations = {}
        for name, animation in state_animations.items():
            frame_items = animation.get(frames_key, [])
            if not frame_items:
                continue
            frames_by_number = LazySurfaceMap(
                {item["frame"]: ROOT / item["image"] for item in frame_items}
            )
            metadata_by_number = {item["frame"]: item for item in frame_items}
            animations[name] = {
                "frame_count": len(frame_items),
                "frames": frames_by_number,
                "metadata": metadata_by_number,
                "playback": animation.get("playback", {}),
            }
        return animations

    def move(self, direction: str) -> None:
        if not self.has_control or self.ctrl_loss > 0:
            return
        if self.current_attack:
            if self.on_ground and direction != "stop":
                self.move_queue = direction
                return
            if direction == "stop":
                self.move_queue = "stop"
                return
        if direction == "right":
            self.xinc = self.move_xinc
            self.state = "goright"
        elif direction == "left":
            self.xinc = -self.move_xinc
            self.state = "goleft"
        elif direction == "stop":
            if self.current_label == "dodge" and self.shielded:
                return
            if self.on_ground:
                self.xinc = 0.0
            self.state = "stop"
        elif direction == "down":
            if self.on_ground:
                self.xinc = 0.0
                self.state = "crouch"
            else:
                self.yinc = 8.0

    def jump(self) -> None:
        if not self.has_control or self.ctrl_loss > 0 or self.state == "ko" or self.shielded:
            return
        if self.current_attack:
            return
        self.go_through_platform = None
        if self.state == "thrown":
            self.state = "stop"
        if self.jumpstate == 0:
            self.jumpstate = 1
            self.yinc = self.jump_yinc
            self.pending_puffs.append((pygame.Vector2(self.pos), random.randrange(360)))
        elif self.jumpstate == 1 and self.yinc >= -3:
            self.jumpstate = 2
            self.yinc = self.jump_yinc
            self.pending_puffs.append((pygame.Vector2(self.pos), random.randrange(360)))

    def attack(self, attack_type: str, extra: str) -> None:
        if self.state == "ko" and self.time_ko > 10 * self.damage_amnt:
            self.state = "stop"
            self._animate_attack("koAttack")
            return
        if self.current_attack or not self.has_control or self.ctrl_loss > 0 or self.state == "thrown" or self.shielded:
            return
        if attack_type == "punch":
            if extra == "none":
                if self.on_ground:
                    if abs(self.xinc) <= 0:
                        self._animate_attack("punchGround")
                    else:
                        self._animate_attack("punchRun")
                        self.pending_puffs.append((pygame.Vector2(self.pos.x, self.pos.y - 5), random.randrange(360)))
                else:
                    self._animate_attack("punchAir")
                    self.yinc -= 0 if self.yinc <= -5 else 5
            elif extra == "up" and self.on_ground and abs(self.xinc) <= 0:
                self._animate_attack("punchUp")
        elif attack_type == "special":
            if extra == "none":
                if self.on_ground:
                    self.state = "stop"
                    self.xinc = 0.0
                    self._animate_attack("specialGround")
                else:
                    self._animate_attack("specialAir")
            elif extra == "up":
                if not self.spec_up_ok:
                    return
                self._animate_attack("specialUp")
                self.spec_up_ok = False

    def _animate_attack(self, label: str) -> None:
        self.current_attack = label
        self.attack_facing = self.facing
        self.current_label = label
        self.animation_frame = 0
        self.animation_time_ms = 0
        self.attack_frame = 0
        self.attack_done = False
        self.attack_pending_finish = ""
        self.bullet_shot = False
        if self.special_kind == "garbage" and label in {"specialGround", "specialAir"}:
            # The embedded Garbage clip executes
            # gotoAndStop(random(_totalframes + 1)) when the attack starts.
            # Flash frame zero leaves the clip on frame one, so frame one has
            # twice the probability of the other five source variants.
            self.garbage_variant = max(1, random.randrange(7))
        self.hit_targets.clear()
        if self.current_item and self.current_item_obj is not None and label.startswith("punch"):
            self._use_held_item(label)

    def _use_held_item(self, attack_label: str) -> None:
        item = self.current_item_obj
        if item is None:
            return
        hand = self.attack_contact_point(1) or pygame.Vector2(self.pos)
        angle = 10
        power = 8
        if attack_label == "punchUp":
            angle = 90
            power = 15
            hand.x = self.pos.x
        else:
            # Mine/Grenade.Use adds Sender[Attack]._x once inside the faced
            # local expression and once again unscaled in world space.
            hand.x += self.state_offsets.get(attack_label, pygame.Vector2()).x
        item.pos = pygame.Vector2(hand)
        item.sender = self
        item.throw(power, angle * self.attack_facing)
        self.current_item = ""
        self.current_item_obj = None

    def _update_shield(self) -> None:
        self.shield_size = min(100.0, self.shield_size)
        if self.shield_size < 0:
            self.shield_size = 0.0
            self.shielded = False
        if self.shielded:
            self.shield_size -= 0.2
        else:
            self.shield_size += 0.1
            self.shield_size = min(100.0, self.shield_size)

    def fixed_tick(
        self,
        stage: Stage,
        controls: dict[str, bool],
        bullets: list[Bullet] | None = None,
        rockets: list[RocketProjectile] | None = None,
        special_projectiles: list[SpecialProjectile] | None = None,
    ) -> None:
        self.resolve_attack_this_tick = False
        self.animate_this_tick = False
        if self.osd_score_event:
            self.osd_score_age_ms += TICK_MS
            if self.osd_score_age_ms >= 32 * 1000 / ANIMATION_FPS:
                self.osd_score_event = ""
                self.osd_score_age_ms = 0
        if self.dead:
            return
        self.prev_pos = pygame.Vector2(self.pos)
        if controls.get("shield_released"):
            self.shielded = False
        if controls.get("shield_pressed") and self.xinc == 0:
            self.shielded = True
        self._update_shield()
        if not self.shielded and self.current_label == "dodge":
            self.xinc = 0.0
            self.state = "stop"
        self.osd_damage_age_ms = min(1000, self.osd_damage_age_ms + TICK_MS)
        if self.state == "ko" or self.blinking:
            self.blinky_cos += 1
        if self.spawn_invincible_ms > 0:
            self.spawn_invincible_ms = max(0, self.spawn_invincible_ms - self.spawn_invincible_decay)
        else:
            # Fighter.DoFighter clears these in the `else` branch on the tick
            # after SpInvTme reaches zero, not on the decrementing tick itself.
            self.invincible = False
            self.blinking = False
        if self.state == "spawn":
            self.spawn_age_ms += TICK_MS
            spawn_frame = int(self.spawn_age_ms * ANIMATION_FPS / 1000) + 1
            if spawn_frame > self.spawn_reveal_frame:
                self.spawn_fighter_visible = True
                if self.spawn_white_offset > 5:
                    self.spawn_white_offset /= 2
                    if self.spawn_visual_alpha < 100:
                        self.spawn_visual_alpha *= 4
                else:
                    self.blinking = True
                    self.has_control = True
            self.out_of_camera = False
            if not self.invincible:
                self.state = "stop"
                self.move("stop")
            if self.state == "spawn":
                self.current_label = "still"
                self._advance_animation(self.current_label)
                return
        if self.paralized > 0:
            self.paralized = max(0, self.paralized - TICK_MS)
            self.electrocuted_ms = max(0, self.electrocuted_ms - TICK_MS)
            old_label = self.current_label
            if self.current_attack:
                self._advance_current_attack(allow_special_effects=False)
            else:
                self.current_label = "electrocuted" if self.electrocuted_ms > 0 else "takingHit"
            self._advance_animation(old_label)
            self.animate_this_tick = True
            return
        if self.ctrl_loss > 0:
            self.ctrl_loss = max(0, self.ctrl_loss - TICK_MS)
        if self.state == "ko":
            if self.time_ko > self.damage_amnt:
                self.state = "stop"
                self.has_control = True
            self.time_ko += TICK_MS

        extra = "up" if controls.get("up_trace") else "none"
        if self.current_attack:
            if controls.get("down"):
                self.move("down")
        else:
            if controls.get("down"):
                self.move("down")

            if controls.get("punch_pressed"):
                if extra == "up":
                    self.attack("punch", "up")
                self.attack("punch", "none")
            elif controls.get("special_pressed"):
                if extra == "up":
                    self.attack("special", "up")
                self.attack("special", "none")

            if not self.current_attack and controls.get("jump_pressed"):
                self.jump()

        if self.state == "crouch" and self.on_ground and self.ground_platform and self.ground_platform.moving:
            self.go_through_platform = self.ground_platform
            self.ground_platform = None
            self.on_ground = False
            self.yinc = 5.0
            self.xinc = 0.0
            self.last_collision = "gothrough"

        if self.state == "stop" and not self.on_ground:
            self.xinc *= 0.7
        elif self.state == "stop" and self.on_ground:
            self.xinc = 0.0
        if abs(self.xinc) < 0.5:
            self.xinc = 0.0

        self._carry_with_moving_platform()
        old_x = self.pos.x
        old_y = self.pos.y
        self._move_with_stage(stage, old_x, old_y)
        body = self.body_rect()
        if any(body.colliderect(rect) for rect in stage.killer_rects):
            self.pos.x += 30
            self.yinc = 0.0
        if any(body.colliderect(rect) for rect in stage.boom_rects):
            self.pending_stage_boom = True
        self._update_thrown_puffs()

        self._check_bounds(stage)
        if self.dead:
            return
        self.resolve_attack_this_tick = bool(self.current_attack or self.state == "thrown")
        old_label = self.current_label
        if self.current_attack:
            self._update_attack(
                bullets if bullets is not None else [],
                rockets if rockets is not None else [],
                special_projectiles if special_projectiles is not None else [],
                defer_finish=True,
            )
        else:
            self._animate()
        self._advance_animation(old_label)
        self.animate_this_tick = True

    def _carry_with_moving_platform(self) -> None:
        if not self.on_ground or self.ground_platform is None or not self.ground_platform.moving:
            return
        platform = self.ground_platform
        previous = platform.prev_rect or platform.rect
        dx = platform.rect.x - previous.x
        if dx != 0 and dx < 10:
            self.pos.x += dx
        self.pos.y = platform.rect.top

    def _update_thrown_puffs(self) -> None:
        if self.state != "thrown" or self.ctrl_loss <= 0 or self.last_throw_puff_pos is None:
            return
        distance_sq = (self.last_throw_puff_pos.x - self.pos.x) ** 2 + (self.last_throw_puff_pos.y - self.pos.y) ** 2
        if distance_sq > 400:
            self.pending_puffs.append((pygame.Vector2(self.pos), random.randrange(360)))
            self.last_throw_puff_pos = pygame.Vector2(self.pos)

    def _move_with_stage(self, stage: Stage, old_x: float, old_y: float) -> None:
        ignored = self.go_through_platform
        if self.last_collision != "gothrough":
            self.last_collision = "-"

        self.pos.x += self.xinc
        predicted_y = old_y + self.yinc
        body_after_x = self.body_rect_at(self.pos.x, predicted_y)
        side_hit = stage.find_side_crossing(
            old_x,
            self.pos.x,
            body_after_x.top,
            body_after_x.bottom,
            self.body_half_width,
            ignored=ignored,
            include_moving=not (self.state == "thrown" and self.yinc < 0),
        )
        if side_hit is not None:
            if self.pos.x > old_x:
                self.pos.x = side_hit.rect.left - self.body_half_width
            else:
                self.pos.x = side_hit.rect.right + self.body_half_width
            if self.state == "thrown":
                self.xinc *= -1
                self.facing *= -1
                self.last_collision = "wall-bounce"
            else:
                self.last_collision = "wall"

        old_head_y = old_y - self.body_height
        self.pos.y += self.yinc
        new_head_y = self.pos.y - self.body_height
        landed = stage.find_floor_crossing(
            self.pos.x,
            old_y,
            self.pos.y,
            foot_radius=self.foot_radius,
            ignored=ignored,
        )
        ceiling = stage.find_ceiling_crossing(
            self.pos.x,
            old_head_y,
            new_head_y,
            self.body_half_width,
            ignored=ignored,
        )

        if landed and self.yinc >= 0:
            self._land_on_platform(landed)
        elif ceiling and self.yinc < 0:
            self.pos.y = ceiling.rect.bottom + self.visual_bounds().height + 5
            self.yinc = 0.0
            self.on_ground = False
            self.ground_platform = None
            self.last_collision = "ceiling"
        else:
            self.on_ground = False
            self.ground_platform = None
            if self.yinc < self.max_fall:
                self.yinc += self.gravity
            elif self.yinc != 8:
                self.yinc = self.max_fall

        if self.go_through_platform is not None and not self.body_rect().colliderect(self.go_through_platform.rect):
            self.go_through_platform = None

    def _land_on_platform(self, platform: Platform) -> None:
        was_thrown = self.state == "thrown"
        self.pos.y = platform.rect.top
        self.yinc = 0.0
        self.jumpstate = 0
        self.on_ground = True
        self.ground_platform = platform
        self.spec_up_ok = True
        self.last_sender = None
        self.combo = 2
        self.last_collision = "floor:" + platform.name
        if was_thrown:
            self.pending_sounds.append("hit_ground")
            self.state = "ko"
            self.time_ko = 0
            self.xinc = 0.0
            self.has_control = False
            self.last_throw_puff_pos = None

    def _update_attack(
        self,
        bullets: list[Bullet],
        rockets: list[RocketProjectile],
        special_projectiles: list[SpecialProjectile],
        defer_finish: bool = False,
    ) -> None:
        self._advance_current_attack(
            allow_special_effects=True,
            bullets=bullets,
            rockets=rockets,
            special_projectiles=special_projectiles,
            defer_finish=defer_finish,
        )

    def _advance_current_attack(
        self,
        allow_special_effects: bool,
        bullets: list[Bullet] | None = None,
        rockets: list[RocketProjectile] | None = None,
        special_projectiles: list[SpecialProjectile] | None = None,
        defer_finish: bool = False,
    ) -> None:
        self.current_label = self.current_attack
        current_frame = self._timeline_frame(self.current_attack)
        if allow_special_effects:
            if self.current_attack in {"specialGround", "specialAir"}:
                if self.special_kind == "peach_weapons":
                    self._parse_bullet_attack(bullets if bullets is not None else [], current_frame)
                elif self.special_kind in {"pencil", "poop", "garbage", "electric"}:
                    self._parse_special_projectile(
                        special_projectiles if special_projectiles is not None else [],
                        current_frame,
                    )
                elif self.special_kind == "kamehameha":
                    if current_frame >= 5 and not self.bullet_shot:
                        self.bullet_shot = True
                        self.pending_sounds.append("kamehameha")
                    elif current_frame < 5:
                        self.bullet_shot = False
            elif self.current_attack == "specialBackThrow":
                self._update_back_throw(current_frame)
            elif self.current_attack == "specialUp":
                self._apply_special_up_motion(current_frame)
                if self.special_kind == "peach_weapons" and current_frame > 14:
                    self._parse_rocket_attack(rockets if rockets is not None else [], current_frame)
                elif self.special_kind == "garbage":
                    self._parse_special_projectile(
                        special_projectiles if special_projectiles is not None else [],
                        current_frame,
                    )
        animation = self.animations.get(self.current_attack)
        total = animation["frame_count"] if animation else 1
        if current_frame >= total:
            if self.current_attack == "specialBackThrow":
                self._finish_back_throw()
            if defer_finish:
                self.attack_frame = current_frame
                self.attack_pending_finish = self.current_attack
            else:
                self._finish_attack()
        else:
            self.attack_frame = current_frame

    def _apply_special_up_motion(self, current_frame: int) -> None:
        motion = self.fighter_data.get("special_up_motion", {})
        if self.special_kind == "pencil" and current_frame < 3:
            self.pending_sounds.append("woosh")
        elif self.special_kind in {"kamehameha", "poop"} and current_frame <= 2:
            self.pending_sounds.append("woosh")
        elif self.special_kind == "electric" and 10 <= current_frame < 12:
            self.pending_sounds.append("thunder")
        slow_before = int(motion.get("slow_before", 0))
        slow_through = int(motion.get("slow_through", 0))
        if (slow_before and current_frame < slow_before) or (
            slow_through and current_frame <= slow_through
        ):
            self.yinc *= float(motion.get("slow_factor", 1.0))
        rise_from = int(motion.get("rise_from", 0))
        rise_through = int(motion.get("rise_through", 0))
        if rise_from and current_frame >= rise_from and (
            not rise_through or current_frame <= rise_through
        ):
            self.yinc = float(motion.get("rise_yinc", self.yinc))

    def _start_back_throw(self, victim: "PeachFighter") -> None:
        self.current_attack = "specialBackThrow"
        self.current_label = "specialBackThrow"
        self.animation_frame = 0
        self.animation_time_ms = 0
        self.attack_frame = 0
        self.attack_done = False
        self.attack_pending_finish = ""
        self.bullet_shot = False
        self.hit_targets.clear()
        self.throw_victim = victim
        self.has_control = False

    def _update_back_throw(self, current_frame: int) -> None:
        if self.throw_victim is None or self.throw_victim.dead:
            return
        animation = self.animations.get("specialBackThrow")
        total = animation["frame_count"] if animation else 1
        if current_frame >= total:
            return
        angle = current_frame * math.pi * 0.8 / max(1, total) - math.pi
        self.throw_victim.pos.x = self.attack_facing * math.cos(angle) * 25 + self.pos.x
        self.throw_victim.pos.y = math.sin(angle) * 25 + self.pos.y
        self.throw_victim.has_control = False

    def _finish_back_throw(self) -> None:
        victim = self.throw_victim
        self.throw_victim = None
        self.has_control = True
        if victim is None or victim.dead:
            return
        victim.paralized = 0
        victim.has_control = True
        victim.damage(15, self)
        victim.throw_impulse(5, 30 * self.attack_facing, self)

    def _finish_attack(self) -> None:
        self.attack_pending_finish = ""
        self.current_attack = ""
        self.attack_frame = 0
        self.attack_done = True
        self.has_control = True
        self.throw_victim = None
        if self.move_queue:
            queued = self.move_queue
            self.move_queue = ""
            self.move(queued)
        if self.jumpstate > 0:
            self.jumpstate = 3
        self._animate()

    def finish_pending_attack(self) -> None:
        pending = self.attack_pending_finish
        self.attack_pending_finish = ""
        if not pending or self.current_attack != pending:
            return
        old_label = self.current_label
        self._finish_attack()
        if self.current_label != old_label:
            self.animation_frame = 0
            self.animation_time_ms = 0

    def finish_post_collision_tick(self) -> None:
        if not self.animate_this_tick:
            self.attack_pending_finish = ""
            return
        if self.xinc > 0:
            self.facing = -1 if self.state == "thrown" else 1
        elif self.xinc < 0:
            self.facing = 1 if self.state == "thrown" else -1
        if self.current_attack:
            self.attack_facing = self.facing
        self.finish_pending_attack()

    def _parse_bullet_attack(self, bullets: list[Bullet], current_frame: int) -> None:
        if current_frame < self.fighter_data["attacks"][self.current_attack]["spawn_frame_min"] or self.bullet_shot:
            return
        bullet_info = self.fighter_data["state_animations"][self.current_attack]["timeline"]["named_places"]
        bullet_place = next((item for item in bullet_info if item["name"] == "bullet"), None)
        if not bullet_place:
            return
        state_place = next(
            item for item in self.fighter_data["timeline"]["named_places"] if item["name"] == self.current_attack
        )
        bullet_matrix = bullet_place["matrix"]
        state_matrix = state_place["matrix"]
        facing = self.attack_facing if self.current_attack else self.facing
        local_x = -54 if facing < 0 else bullet_matrix["x"]
        bullet_x = self.pos.x + state_matrix["x"] + local_x + 5
        bullet_y = self.pos.y + state_matrix["y"] + bullet_matrix["y"] - 7
        bullet_image = pygame.image.load(str(ROOT / self.projectile_image_path))
        bullets.append(
            Bullet(
                pygame.Vector2(bullet_x, bullet_y),
                20 * facing,
                bullet_image,
                self,
                pygame.Vector2(self.projectile_offset),
                source_scale=self.projectile_render_scale,
            )
        )
        self.bullet_shot = True
        self.attack_done = True
        self.pending_sounds.append("gun")

    def _parse_rocket_attack(self, rockets: list[RocketProjectile], current_frame: int) -> None:
        if current_frame <= 14 or self.bullet_shot:
            return
        self.yinc = -5
        rocket_info = self.fighter_data["state_animations"][self.current_attack]["timeline"]["named_places"]
        rocket_place = next((item for item in rocket_info if item["name"] == "rocket"), None)
        if not rocket_place:
            return
        state_place = next(
            item for item in self.fighter_data["timeline"]["named_places"] if item["name"] == self.current_attack
        )
        rocket_matrix = rocket_place["matrix"]
        state_matrix = state_place["matrix"]
        facing = self.attack_facing if self.current_attack else self.facing
        local_x = -12 if facing < 0 else rocket_matrix["x"]
        base_rotation = math.degrees(math.atan2(rocket_matrix["rotate_skew0"], rocket_matrix["scale_x"]))
        rotation = -base_rotation if facing < 0 else base_rotation
        xinc = math.cos((rotation - 90) * math.pi / 180) * 15
        yinc = math.sin((rotation - 90) * math.pi / 180) * 15
        rocket_x = self.pos.x + state_matrix["x"] + local_x + 5
        rocket_y = self.pos.y + state_matrix["y"] + rocket_matrix["y"] - 7
        rocket_image = pygame.image.load(str(ROOT / self.rocket_image_path))
        rockets.append(
            RocketProjectile(
                pygame.Vector2(rocket_x, rocket_y),
                xinc,
                yinc,
                rotation,
                rocket_image,
                self,
                pygame.Vector2(self.rocket_offset),
                source_scale=self.rocket_render_scale,
                mirrored=facing < 0,
            )
        )
        self.bullet_shot = True
        self.pending_sounds.append("rocket")

    def _special_projectile_frames(self, kind: str) -> list[SpriteAssetFrame]:
        cached = self.special_projectile_cache.get(kind)
        if cached is not None:
            return cached
        frames = []
        for item in self.projectile_data.get(kind, {}).get("frames", []):
            offset = item.get("offset", {"x": 0.0, "y": 0.0})
            frames.append(
                SpriteAssetFrame(
                    image_path=ROOT / item.get("image", item["raw"]),
                    offset=pygame.Vector2(float(offset["x"]), float(offset["y"])),
                    render_scale=max(1.0, float(item.get("render_scale", 1))),
                )
            )
        self.special_projectile_cache[kind] = frames
        return frames

    def _special_place(self, place_name: str, current_frame: int) -> dict[str, object] | None:
        animation = self.fighter_data.get("state_animations", {}).get(self.current_attack, {})
        places = [
            item
            for item in animation.get("timeline", {}).get("named_places", [])
            if item.get("name") == place_name and int(item.get("frame", 1)) <= current_frame
        ]
        if not places:
            return None
        return max(places, key=lambda item: int(item.get("frame", 1)))

    def _parse_special_projectile(
        self,
        projectiles: list[SpecialProjectile],
        current_frame: int,
    ) -> None:
        profile = self.fighter_data.get("attacks", {}).get(self.current_attack, {})
        spawn_kind = str(profile.get("spawns", ""))
        if spawn_kind == "GarbageBurst":
            if current_frame < int(profile.get("spawn_frame_min", 11)) or self.bullet_shot:
                return
            config = dict(self.projectile_data.get("Garbage", {}))
            frames = self._special_projectile_frames("Garbage")
            origin = pygame.Vector2(self.pos.x, self.pos.y - self.visual_bounds().height / 2)
            for index in range(20):
                angle = index * 18 * math.pi / 180
                source_variant = random.randrange(6)
                projectiles.append(
                    SpecialProjectile(
                        kind="Garbage",
                        pos=pygame.Vector2(origin),
                        xinc=math.cos(angle) * 15,
                        yinc=math.sin(angle) * 15,
                        rotation=0.0,
                        frames=frames,
                        sender=self,
                        config=config,
                        variant=max(1, source_variant),
                        facing=self.attack_facing,
                    )
                )
            self.bullet_shot = True
            self.pending_sounds.append("boom")
            return

        if spawn_kind not in {"Pencil", "Poop", "Garbage", "EnergyBall"}:
            return
        animation = self.animations.get(self.current_attack)
        total = int(animation["frame_count"]) if animation else 1
        if profile.get("spawn_at_end"):
            ready = current_frame >= total
        else:
            ready = current_frame >= int(profile.get("spawn_frame_min", 1))
        if not ready:
            self.bullet_shot = False
            return
        if self.bullet_shot:
            return

        place_names = {
            "Pencil": "pencil",
            "Poop": "poop",
            "Garbage": "garbage",
            "EnergyBall": "hand",
        }
        mirrored_x = {"Pencil": -28.0, "Poop": -7.0, "Garbage": -12.0, "EnergyBall": -10.0}
        y_adjust = {"Pencil": -5.0, "Poop": -5.0, "Garbage": -7.0, "EnergyBall": -7.0}
        display_scale = {"Pencil": 0.8, "Poop": 0.4, "Garbage": 1.0, "EnergyBall": 0.5}
        place = self._special_place(place_names[spawn_kind], current_frame)
        if place is None:
            return
        matrix = place["matrix"]
        state_offset = self.state_offsets.get(self.current_attack, pygame.Vector2())
        facing = self.attack_facing if self.current_attack else self.facing
        local_x = mirrored_x[spawn_kind] if facing < 0 else float(matrix["x"])
        spawn = pygame.Vector2(
            self.pos.x + state_offset.x + local_x + 5,
            self.pos.y + state_offset.y + float(matrix["y"]) + y_adjust[spawn_kind],
        )
        config = dict(self.projectile_data.get(spawn_kind, {}))
        variant = 1
        if spawn_kind == "Garbage":
            variant = self.garbage_variant
        projectiles.append(
            SpecialProjectile(
                kind=spawn_kind,
                pos=spawn,
                xinc=float(config.get("xinc", 0)) * facing,
                yinc=float(config.get("yinc", 0)),
                rotation=0.0,
                frames=self._special_projectile_frames(spawn_kind),
                sender=self,
                config=config,
                variant=variant,
                display_scale=display_scale[spawn_kind],
                facing=facing,
            )
        )
        self.bullet_shot = True
        self.attack_done = True
        if spawn_kind == "Poop":
            self.pending_sounds.append("fart")

    def body_rect_at(self, x: float, y: float) -> pygame.Rect:
        return pygame.Rect(
            round(x - self.body_half_width),
            round(y - self.body_height),
            self.body_half_width * 2,
            self.body_height,
        )

    def body_rect(self) -> pygame.Rect:
        return self.body_rect_at(self.pos.x, self.pos.y)

    def hurtbox(self) -> pygame.Rect:
        return self.body_rect()

    def render_pos(self, alpha: float) -> pygame.Vector2:
        return self.prev_pos.lerp(self.pos, max(0.0, min(1.0, alpha)))

    def visual_bounds_at(self, x: float, y: float) -> pygame.Rect:
        image = self.current_image()
        offset = self.current_draw_offset(image)
        scale = self.current_render_scale()
        alpha_bounds = image.get_bounding_rect(min_alpha=1)
        left = x + offset.x + alpha_bounds.left / scale
        top = y + offset.y + alpha_bounds.top / scale
        right = x + offset.x + alpha_bounds.right / scale
        bottom = y + offset.y + alpha_bounds.bottom / scale
        return pygame.Rect(
            math.floor(left),
            math.floor(top),
            max(1, math.ceil(right) - math.floor(left)),
            max(1, math.ceil(bottom) - math.floor(top)),
        )

    def visual_bounds(self) -> pygame.Rect:
        return self.visual_bounds_at(self.pos.x, self.pos.y)

    def out_of_camera_proxy_pos(self, stage: Stage, camera_view: list[float] | None = None) -> pygame.Vector2:
        proxy = pygame.Vector2(self.pos)
        if self.pos.x < stage.bounds_cam.left:
            proxy.x = stage.bounds_cam.left + 50
            proxy.y = self.pos.y
        if self.pos.x > stage.bounds_cam.right:
            proxy.x = stage.bounds_cam.right - 50
            proxy.y = self.pos.y
        elif self.pos.y < stage.bounds_cam.top:
            proxy.x = self.pos.x
            proxy.y = stage.bounds_cam.top + 50
        elif self.pos.y > stage.bounds_cam.bottom:
            proxy.x = self.pos.x
            proxy.y = stage.bounds_cam.bottom - 50

        if camera_view is not None:
            cam_x, _cam_y, cam_w, _cam_h = camera_view
            image = self.current_image()
            visual_w = max(1, image.get_width() / self.current_render_scale())
            if self.pos.x > cam_x + cam_w - visual_w:
                proxy.x = cam_x + cam_w - 50
            if self.pos.x < cam_x:
                proxy.x = cam_x + 50
        proxy.y += 10
        return proxy

    def out_of_camera_proxy_visible(
        self,
        stage: Stage,
        camera_view: list[float] | None,
        target_view: list[float] | None,
    ) -> bool:
        if camera_view is None:
            return True
        cam_x, _cam_y, cam_w, _cam_h = camera_view
        target_x, target_y, target_w, target_h = target_view or camera_view
        visual = self.visual_bounds()
        visible = True
        if self.pos.x < stage.bounds_cam.left:
            visible = not (self.pos.x > cam_x)
        if self.pos.x > stage.bounds_cam.right:
            visible = not (self.pos.x < cam_x + cam_w - visual.w)
        elif self.pos.y < stage.bounds_cam.top:
            visible = not (self.pos.y < target_y + visual.h)
        elif self.pos.y > stage.bounds_cam.bottom:
            visible = not (self.pos.y > target_y + target_h - visual.h)
        return visible

    def is_visible_in_camera(self, camera_view: list[float] | None) -> bool:
        if camera_view is None:
            return True
        cam_x, cam_y, cam_w, cam_h = camera_view
        visual = self.visual_bounds()
        return visual.right >= cam_x and visual.left <= cam_x + cam_w and visual.bottom >= cam_y and visual.top <= cam_y + cam_h

    def camera_focus_bounds(self, stage: Stage, camera_view: list[float] | None, alpha: float) -> pygame.Rect:
        render_pos = self.render_pos(alpha)
        # VCamera follows Fighter.HitTester, which is a duplicate frozen on
        # outer frame 1. Attack, jump and the tall intro rope never resize it.
        return self.body_rect_at(render_pos.x, render_pos.y)

    def attack_hitbox(self) -> pygame.Rect | None:
        if not self.current_attack:
            return None
        frame = self.attack_frame or self._timeline_frame(self.current_attack)
        if frame < 1:
            return None
        return self.visual_bounds()

    def attack_contact_point(self, frame: int) -> pygame.Vector2 | None:
        animation = self.fighter_data["state_animations"].get(self.current_attack)
        if not animation:
            return None
        places = [
            item
            for item in animation["timeline"].get("named_places", [])
            if item["name"] == "hand" and int(item["frame"]) <= frame
        ]
        if not places:
            return None
        place = max(places, key=lambda item: int(item["frame"]))
        matrix = place["matrix"]
        state_offset = self.state_offsets.get(self.current_attack, pygame.Vector2())
        facing = self.attack_facing if self.current_attack else self.facing
        local_x = float(matrix["x"]) + state_offset.x
        local_y = float(matrix["y"]) + state_offset.y
        hand_x = self.pos.x + local_x * facing
        hand_y = self.pos.y + local_y
        return pygame.Vector2(hand_x, hand_y)

    def held_item_pos(self) -> pygame.Vector2 | None:
        label = self.current_attack or self.current_label
        animation = self.fighter_data["state_animations"].get(label)
        if not animation:
            return None
        frame = self._timeline_frame(label)
        places = [
            item
            for item in animation["timeline"].get("named_places", [])
            if item["name"] == "hand" and int(item["frame"]) <= frame
        ]
        if not places:
            return None
        place = max(places, key=lambda item: int(item["frame"]))
        matrix = place["matrix"]
        state_offset = self.state_offsets.get(label, pygame.Vector2())
        facing = self.attack_facing if self.current_attack else self.facing
        return pygame.Vector2(
            self.pos.x + (float(matrix["x"]) + state_offset.x) * facing,
            self.pos.y + float(matrix["y"]) + state_offset.y,
        )

    def front_attack_hitbox(self) -> pygame.Rect | None:
        hitbox = self.attack_hitbox()
        if hitbox is None:
            return None
        facing = self.attack_facing if self.current_attack else self.facing
        center_x = round(self.pos.x)
        if facing >= 0:
            left = max(hitbox.left, center_x + 4)
            return pygame.Rect(left, hitbox.top, max(1, hitbox.right - left), hitbox.h)
        right = min(hitbox.right, center_x - 4)
        return pygame.Rect(hitbox.left, hitbox.top, max(1, right - hitbox.left), hitbox.h)

    def back_grab_hitbox(self) -> pygame.Rect:
        return self.body_rect().inflate(10, 2)

    def can_back_throw_now(self) -> bool:
        return self.current_attack in {"punchGround", "punchRun"}

    def damage(self, amount: int, sender: "PeachFighter | None", force: bool = False) -> bool:
        if self.dead:
            return False
        if (self.state == "ko" and not force) or self.state == "spawn":
            return False
        # Fighter.Damage exits before shield depletion while the fighter is in
        # Peach's moving dodge state (`Shielded && dodge != undefined`).
        if self.shielded and self.current_label == "dodge":
            return False
        if self.shielded:
            if self.shield_size > 0:
                self.shield_size -= amount / 5
                return False
            self.shielded = False
        if sender is not None:
            self.last_sender = sender
        self.current_attack = ""
        self.attack_done = True
        self.attack_pending_finish = ""
        self.throw_victim = None
        self.spec_up_ok = True
        self.pending_stop_sounds.add("electric")
        self.electrocuted_ms = 0
        self.damage_amnt += math.floor(amount * (self.combo / 2))
        if self.last_sender is not None:
            self.combo += 1
        if self.damage_amnt > 999:
            self.damage_amnt = 999
            self.throw_impulse(30, 45, sender)
        self.osd_damage_age_ms = 0
        self.paralized = amount
        if sender is not None:
            sender.paralized = amount * 5
            if self.draw_depth > sender.draw_depth:
                self.draw_depth, sender.draw_depth = sender.draw_depth, self.draw_depth
        return True

    def throw_impulse(self, power: float, angle: float, sender: "PeachFighter | None", force: bool = False) -> bool:
        if self.dead:
            return False
        if (self.state == "ko" and not force) or self.state == "spawn":
            return False
        if self.shielded and self.current_label == "dodge":
            return False
        if self.shielded and not force:
            return False
        if sender is not None:
            self.last_sender = sender
        if angle < 0:
            angle = 180 + angle
        sender_weight = sender.weight if sender is not None else 1
        radians = angle * math.pi / 180
        power = power + power * sender_weight - power * self.weight + self.damage_amnt / 100 * math.log(power) * power + self.combo / 4
        power = max(0, power)
        self.xinc = math.cos(radians) * power
        self.yinc = -math.sin(radians) * power
        self.ctrl_loss = power**2 * 5
        if power > 10:
            self.pending_sounds.append("thrown")
        if power > 10 and self.ctrl_loss > 100:
            self.last_throw_puff_pos = pygame.Vector2(self.pos)
        else:
            self.last_throw_puff_pos = None
        self.state = "thrown"
        self.current_label = "thrown"
        return True

    def _check_bounds(self, stage: Stage) -> None:
        self.out_of_camera = not stage.bounds_cam.collidepoint(self.pos.x, self.pos.y)
        if stage.bounds.collidepoint(self.pos.x, self.pos.y):
            self.dead_reason = ""
            return
        if self.pos.y > stage.bounds.bottom:
            self.dead_reason = "bot"
        elif self.pos.y < stage.bounds.top:
            self.dead_reason = "top"
        elif self.pos.x < stage.bounds.left:
            self.dead_reason = "lef"
        elif self.pos.x > stage.bounds.right:
            self.dead_reason = "rig"
        self.die(self.dead_reason, stage)

    def die(self, death_type: str, stage: Stage) -> None:
        if self.dead:
            return
        killer = self.last_sender
        death_pos = pygame.Vector2(self.pos)
        if killer is not None:
            killer.kos += 1
            killer.osd_score_event = "plus"
            killer.osd_score_age_ms = 0
        else:
            self.sds += 1
        if self.limit_mode == "time":
            self.osd_score_event = "minus"
            self.osd_score_age_ms = 0
        self.deaths += 1
        stock_match = self.limit_mode == "stock"
        if stock_match:
            self.lives -= 1
        self.last_death_type = death_type
        self.damage_amnt = 0
        self.osd_damage_age_ms = 0
        self.current_attack = ""
        self.attack_done = True
        self.attack_pending_finish = ""
        self.bullet_shot = False
        self.hit_targets.clear()
        self.move_queue = ""
        self.throw_victim = None
        self.attack_facing = self.facing
        self.xinc = 0.0
        self.yinc = 0.0
        self.ctrl_loss = 0
        self.paralized = 0
        self.electrocuted_ms = 0
        self.time_ko = 0
        self.shielded = False
        self.last_throw_puff_pos = None
        self.jumpstate = 0
        self.on_ground = False
        self.ground_platform = None
        self.go_through_platform = None
        self.last_collision = "-"
        self.out_of_camera = False
        self.spawn_reveal_ms = 0
        self.pending_death_event = DeathEvent(
            pos=death_pos,
            death_type=death_type,
            fighter_name=self.name,
            killer_name=killer.name if killer is not None else None,
        )
        if stock_match and self.lives <= 0:
            self.prev_pos = pygame.Vector2(self.pos)
            self.dead = True
            self.has_control = False
            self.state = "dead"
            self.current_label = "ko"
            return
        self.respawn(from_death=True, stage=stage)

    def respawn(self, from_death: bool = False, stage: Stage | None = None) -> None:
        if from_death:
            self.pos = stage.death_respawn_point() if stage is not None else pygame.Vector2(random.randrange(400) + 100, 0)
        else:
            self.pos = pygame.Vector2(self.spawn_pos)
        self.prev_pos = pygame.Vector2(self.pos)
        self.xinc = 0.0
        self.yinc = 0.0
        self.state = "spawn" if from_death else "stop"
        self.current_label = "still"
        self.animation_frame = 0
        self.animation_time_ms = 0
        self.move_queue = ""
        self.throw_victim = None
        self.shielded = False
        self.pending_puffs.clear()
        self.last_throw_puff_pos = None
        self.attack_facing = self.facing
        self.has_control = not from_death
        self.ground_platform = None
        self.go_through_platform = None
        self.last_collision = "-"
        self.spawn_invincible_ms = RESPAWN_INVINCIBLE_MS if from_death else 0
        self.invincible = from_death
        self.spawn_reveal_ms = 0
        self.spawn_age_ms = 0
        self.spawn_effect_kind = random.randrange(2) + 1 if from_death else 1
        self.spawn_reveal_frame = 10 * self.spawn_effect_kind
        self.spawn_visual_alpha = 1.0 if from_death else 100.0
        self.spawn_white_offset = 255.0 if from_death else 0.0
        self.spawn_fighter_visible = not from_death
        self.blinking = False
        self.blinky_cos = 0
        if from_death:
            self.facing = 1
            self.attack_facing = 1
            self.on_ground = True
            self.yinc = 0.0
        self.dead_reason = ""
        self.out_of_camera = False

    def start_intro_spawn(self) -> None:
        self.pos = pygame.Vector2(self.spawn_pos)
        self.prev_pos = pygame.Vector2(self.pos)
        self.respawn(from_death=False)
        self.current_label = "spawn"
        self.animation_frame = 0
        self.animation_time_ms = 0
        # FightTimer only performs gotoAndStop("spawn"). Fighter.State and
        # HasControl keep their constructor values while GameOn is false.
        self.has_control = True
        self.intro_visible = True

    def advance_intro_tick(self) -> None:
        if not self.intro_visible or self.current_label != "spawn":
            return
        # FightTimer stops the fighter's outer clip on "spawn", while the
        # nested spawn MovieClip keeps playing on Flash's 30 fps timeline.
        self.animation_time_ms += TICK_MS
        self.animation_frame = self._timeline_frame("spawn") - 1

    def advance_pre_game_tick(self, controls: dict[str, bool]) -> None:
        """Run KeyCombi/timeline work while source DoFighter is GameOn-gated."""
        if self.dead:
            return
        if controls.get("shield_released"):
            self.shielded = False
        if controls.get("shield_pressed") and self.xinc == 0:
            self.shielded = True
        if controls.get("down"):
            self.move("down")

        extra = "up" if controls.get("up_trace") else "none"
        if controls.get("punch_pressed"):
            if extra == "up":
                self.attack("punch", "up")
            self.attack("punch", "none")
        elif controls.get("special_pressed"):
            if extra == "up":
                self.attack("special", "up")
            self.attack("special", "none")
        if controls.get("jump_pressed"):
            self.jump()

        # MovieClip children continue to play even though DoFighter returns
        # before physics, collision and ParseSAttacks while GameOn is false.
        if self.current_attack:
            old_label = self.current_label
            self._advance_current_attack(allow_special_effects=False)
            if self.current_attack:
                self._advance_animation(old_label)

    def finish_intro_spawn(self) -> None:
        self.has_control = True
        self.spawn_fighter_visible = True
        if self.current_attack:
            self.current_label = self.current_attack
        else:
            self._animate()

    def _animate(self) -> None:
        if self.state == "thrown":
            self.current_label = "thrown"
            return
        if self.state in {"ko", "dead"}:
            self.current_label = "ko"
            return
        if self.on_ground:
            if self.state in {"goright", "goleft"}:
                if self.shielded:
                    dodge = self.animations.get("dodge")
                    if dodge is None:
                        self.current_label = "run"
                    elif self.current_label == "dodge" and self._timeline_frame("dodge") >= int(dodge["frame_count"]):
                        self.xinc = 0.0
                        self.state = "stop"
                        self.current_label = "still"
                    else:
                        self.current_label = "dodge"
                else:
                    self.current_label = "run"
            elif self.state == "crouch":
                self.current_label = "crouch"
            else:
                self.current_label = "still"
        else:
            if self.jumpstate == 1:
                self.current_label = "jump1"
            elif self.jumpstate == 2:
                self.current_label = "jump2"
            else:
                self.current_label = "falling"

    def _advance_animation(self, old_label: str) -> None:
        if old_label != self.current_label:
            self.animation_frame = 0
            self.animation_time_ms = 0
        else:
            self.animation_time_ms += TICK_MS
            self.animation_frame = self._timeline_frame(self.current_label) - 1

    def _timeline_frame(self, label: str, render_offset_ms: float = 0.0) -> int:
        animation = self.animations.get(label)
        total = animation["frame_count"] if animation else 1
        frame = int((self.animation_time_ms + render_offset_ms) * ANIMATION_FPS / 1000) + 1
        if self.current_attack == label:
            return min(total, frame)
        playback = animation.get("playback", {}) if animation else {}
        stop_at = int(playback.get("stop_at", 0))
        if stop_at > 0:
            return min(stop_at, frame)
        loop_at = int(playback.get("loop_at", total))
        loop_from = int(playback.get("loop_from", 0))
        if loop_from > 0 and frame > loop_at:
            loop_length = max(1, loop_at - loop_from + 1)
            return loop_from + ((frame - loop_at - 1) % loop_length)
        return ((frame - 1) % total) + 1

    def _visual_animation(self, label: str) -> dict | None:
        if self.bullet_shot and label in self.fired_animations:
            return self.fired_animations[label]
        held = self.held_item_animations.get(self.current_item.lower()) if self.current_item else None
        if held and label in held:
            return held[label]
        garbage = self.garbage_variant_animations.get(self.garbage_variant)
        if self.special_kind == "garbage" and garbage and label in garbage:
            return garbage[label]
        return self.animations.get(label)

    def current_image(self, render_offset_ms: float = 0.0) -> pygame.Surface:
        animation = self._visual_animation(self.current_label)
        if animation:
            frame_no = self._current_frame_number(animation, render_offset_ms)
            image = animation["frames"][frame_no]
        else:
            index = self.label_to_frame.get(self.current_label, 0)
            image = self.frames[index]
        if self.bullet_shot and self.current_label not in self.fired_animations:
            if self.current_label in {"specialGround", "specialAir"}:
                for place_name in ("bullet", "pencil", "poop", "garbage"):
                    image = self._hide_embedded_place(image, place_name, render_offset_ms)
            elif self.current_label == "specialUp" and self.special_kind == "peach_weapons":
                image = self._hide_embedded_place(image, "rocket", render_offset_ms)
        facing = self.attack_facing if self.current_attack else self.facing
        if facing < 0:
            return pygame.transform.flip(image, True, False)
        return image

    def current_draw_offset(self, image: pygame.Surface, render_offset_ms: float = 0.0) -> pygame.Vector2:
        animation = self._visual_animation(self.current_label)
        if not animation:
            bbox = image.get_bounding_rect()
            return pygame.Vector2(-bbox.centerx, -bbox.bottom)
        frame_no = self._current_frame_number(animation, render_offset_ms)
        metadata = animation["metadata"][frame_no]
        state_offset = self.state_offsets.get(self.current_label, pygame.Vector2())
        local_offset = metadata.get("offset", {"x": 0.0, "y": 0.0})
        logical_size = metadata.get("logical_size", {})
        width = float(logical_size.get("w", image.get_width() / self.current_render_scale(render_offset_ms)))
        x = state_offset.x + float(local_offset["x"])
        facing = self.attack_facing if self.current_attack else self.facing
        if facing < 0:
            x = -x - width
        return pygame.Vector2(x, state_offset.y + float(local_offset["y"]))

    def _current_frame_number(self, animation: dict | None = None, render_offset_ms: float = 0.0) -> int:
        animation = animation or self.animations.get(self.current_label)
        if not animation:
            return 1
        return max(
            1,
            min(int(animation["frame_count"]), self._timeline_frame(self.current_label, render_offset_ms)),
        )

    def current_render_scale(self, render_offset_ms: float = 0.0) -> float:
        animation = self._visual_animation(self.current_label)
        if not animation:
            return 1.0
        metadata = animation["metadata"][self._current_frame_number(animation, render_offset_ms)]
        return max(1.0, float(metadata.get("render_scale", 1)))

    def _hide_embedded_place(
        self,
        image: pygame.Surface,
        place_name: str,
        render_offset_ms: float = 0.0,
    ) -> pygame.Surface:
        cleaned = image.copy()
        timeline = self.fighter_data["state_animations"][self.current_label]["timeline"]["named_places"]
        place = next((item for item in timeline if item["name"] == place_name), None)
        if place is None:
            return cleaned
        matrix = place["matrix"]
        animation = self._visual_animation(self.current_label)
        metadata = (
            animation["metadata"][self._current_frame_number(animation, render_offset_ms)]
            if animation
            else {}
        )
        local_offset = metadata.get("offset", {"x": 0.0, "y": 0.0})
        scale = self.current_render_scale(render_offset_ms)
        x = max(0, round((float(matrix["x"]) - float(local_offset["x"]) - 3) * scale))
        y = max(0, round((float(matrix["y"]) - float(local_offset["y"]) - 6) * scale))
        w = max(1, cleaned.get_width() - x)
        h = min(cleaned.get_height() - y, round(18 * scale))
        cleaned.fill((0, 0, 0, 0), pygame.Rect(x, y, w, h))
        return cleaned


class AIController:
    def __init__(
        self,
        player: PeachFighter,
        stage: Stage,
        level: int = 7,
        force_victim: bool = False,
    ) -> None:
        self.player = player
        self.stage = stage
        self.level = max(1, int(level))
        self.force_victim = force_victim
        self.victim: PeachFighter | None = None
        self.action_delay_ms = 0.0
        self.queued_action: tuple[str, str] | None = None

    def pick_fighter(self, fighters: list[PeachFighter]) -> None:
        candidates: list[tuple[float, int]] = []
        for index, fighter in enumerate(fighters):
            if fighter is self.player:
                continue
            distance = math.hypot(self.player.pos.x - fighter.pos.x, self.player.pos.y - fighter.pos.y)
            points = -distance / 5 + fighter.lives * 2 + random.randrange(100) - 50
            candidates.append((points, index))
        if not candidates:
            return
        # AIControl.PickFighter initializes _loc6_ to zero instead of the
        # first candidate's Player index. This only matters when P1 itself is
        # CPU and the first candidate remains the highest-scoring one.
        best_points = candidates[0][0]
        selected_index = 0
        for points, player_index in candidates:
            if points > best_points:
                selected_index = player_index
                best_points = points
        self.victim = fighters[selected_index]

    def act(self, action: str, extra: str) -> None:
        if self.action_delay_ms != 0:
            return
        self.action_delay_ms = 500 / self.level + random.randrange(200)
        self.queued_action = (action, extra)

    def check_action(self) -> None:
        if self.action_delay_ms > 0:
            self.action_delay_ms -= TICK_MS
            return
        if self.queued_action is None:
            self.action_delay_ms = 0
            return
        action, extra = self.queued_action
        self.player.attack(action, extra)
        self.action_delay_ms = 0
        self.queued_action = None

    def move_to(self, destination_x: float, destination_y: float, randomize: bool = False) -> None:
        if abs(destination_x - self.player.pos.x) < 30:
            self.player.move("stop")
            return
        if randomize:
            destination_x += random.randrange(40) - 20
        if destination_y - self.player.pos.y > 40 and self.player.on_ground:
            self.player.move("down")
            return
        if destination_x > self.player.pos.x:
            self.player.move("right")
        elif destination_x < self.player.pos.x:
            self.player.move("left")
        if self.victim is not None and destination_y < self.player.pos.y and self.victim.on_ground:
            self.player.jump()
        if self.stage.ai_helper_type(self.player.body_rect()) == "J":
            self.player.jump()

    def attack(self) -> None:
        victim = self.victim
        if victim is None:
            return
        distance_sq = (victim.pos.x - self.player.pos.x) ** 2 + (victim.pos.y - self.player.pos.y) ** 2
        if distance_sq < 900:
            if self.player.pos.y < victim.pos.y and not self.player.on_ground and self.player.yinc > 0:
                self.act("punch", "none")
            roll = random.randrange(max(1, round(160 / self.level)))
            if roll == 0:
                self.act("punch", "none")
            elif roll == 3:
                self.act("punch", "up")
            elif roll == 4:
                self.player.shielded = not self.player.shielded
            else:
                if self.player.shielded and roll > 12:
                    self.player.shielded = False
                self.move_to(victim.pos.x, victim.pos.y)
            return
        roll = random.randrange(max(1, round(720 / self.level)))
        self.player.shielded = False
        if roll == 0:
            if abs(victim.pos.y - self.player.pos.y) <= 30:
                self.act("special", "none")
        elif roll == 1:
            self.act("special", "up")
        else:
            self.move_to(victim.pos.x, victim.pos.y)

    def fixed_tick(self, fighters: list[PeachFighter]) -> None:
        if self.player.dead:
            return
        if (random.randrange(200) < 100 and not self.force_victim) or self.victim not in fighters:
            self.pick_fighter(fighters)
        if self.victim is None:
            return
        self.move_to(self.victim.pos.x, self.victim.pos.y)
        self.attack()
        self.check_action()


class RuntimeApp:
    def __init__(self) -> None:
        self.manifest = load_manifest()
        self.stage = Stage(self.manifest)
        self.bullets: list[Bullet] = []
        self.rockets: list[RocketProjectile] = []
        self.special_projectiles: list[SpecialProjectile] = []
        self.spawn_frames = self._load_spawn_frames()
        self.puff_frames = self._load_effect_assets("Puff")
        self.puff_source_scale = self._effect_source_scale("Puff")
        self.player_death_frames = self._load_effect_assets("PlayerDeath")
        self.punch_damage_frames = self._load_effect_assets("PunchDamage")
        punch_damage_data = self.manifest.get("effects", {}).get("PunchDamage", {})
        punch_damage_items = punch_damage_data.get("frames", [])
        self.punch_damage_source_scale = max(
            1.0,
            float(punch_damage_items[0].get("render_scale", 1)) if punch_damage_items else 1.0,
        )
        self.pos_indicator_frames = self._load_effect_assets("PosIndicator")
        self.far_indicator_frames = self._load_effect_assets("FarIndicator")
        self.item_indicator_frames = self._load_effect_assets("ItemIndicator")
        self.item_indicator_source_scale = self._effect_source_scale("ItemIndicator")
        self.boom_star_frames = self._load_effect_assets("BoomStar")
        self.boom_wave_frames = self._load_effect_assets("BoomWave")
        self.boom_matter_frames = self._load_effect_assets("BoomMatter")
        self.shield_frames = self._load_effect_frames("Shield")
        self.shield_source_scale = self._effect_source_scale("Shield")
        self.osd_bigicon_frames = self._load_ui_assets("OSDBigIcon")
        bigicon_data = self.manifest.get("ui", {}).get("OSDBigIcon", {})
        self.osd_bigicon_by_character = {
            str(item["name"]): int(item["frame"]) - 1
            for item in bigicon_data.get("timeline", {}).get("labels", [])
        }
        self.osd_damage_frames = self._load_ui_frames("OSDDamage")
        self.osd_score_upper_frames = self._load_ui_assets("OSDScoreUpper")
        self._ui_font_cache: dict[tuple[str, int, bool], pygame.font.Font] = {}
        life_data = self.manifest.get("ui", {}).get("OSDLifeGraphic", {})
        self.osd_life_frames = self._load_sprite_assets(
            {"frames": life_data.get("peach_color_frames", life_data.get("frames", []))}
        )
        self.osd_life_frames_by_character = {
            character_name: self._load_sprite_assets({"frames": frames})
            for character_name, frames in life_data.get("character_color_frames", {}).items()
        }
        self.item_frames = self._load_item_frames()
        self.item_source_scales = {
            kind: self._item_source_scale(kind) for kind in self.manifest.get("items", {}).get("classes", [])
        }
        self.item_frame_labels = {
            kind: {
                str(label["name"]): int(label["frame"])
                for label in self.manifest.get("items", {}).get(kind, {}).get("timeline", {}).get("labels", [])
            }
            for kind in self.manifest.get("items", {}).get("classes", [])
        }
        self.accumulator = 0
        self.ai_controllers: dict[int, AIController] = {}
        self.endurance_level = 0
        self.killed_players = 0
        self.inputs: list[FighterInput] = self._create_inputs()
        self.show_debug = False
        self.camera_view: list[float] | None = None
        self.camera_target_view: list[float] | None = None
        self.stage_time_ms = 0
        self._stage_surface_cache: dict[tuple[object, ...], pygame.Surface] = {}
        self._surface_bounds_cache: dict[int, pygame.Rect] = {}
        self._stage_backdrop_cache_key: tuple[object, ...] | None = None
        self._stage_backdrop_cache: pygame.Surface | None = None
        self._stage_foreground_cache_key: tuple[object, ...] | None = None
        self._stage_foreground_cache: pygame.Surface | None = None
        self._stage_scene_cache_key: tuple[object, ...] | None = None
        self._stage_scene_cache: pygame.Surface | None = None
        self._sky_surface_cache: dict[tuple[int, int], pygame.Surface] = {}
        self.death_effects: list[DeathEffect] = []
        self.camera_tricks: list[DeathEffect] = []
        self.camera_shake_start_ms = 0
        self.camera_shake_until_ms = 0
        self.spawn_effects: list[SpawnEffect] = []
        self.hit_effects: list[HitEffect] = []
        self.explosions: list[ExplosionEffect] = []
        self.items: list[StageItem] = []
        self.item_gen_timer_ms = 0
        self.ready_set = 5
        self.ready_timer_ms = 0
        self.ready_text = "5"
        self.match_state = "playing"
        self.match_winner: PeachFighter | None = None
        self.num_dead = 0
        self.app_state = "menu"
        self.menu: MainMenu | None = None
        self.results: MatchResults | None = None
        self.match_config: dict[str, object] = {}
        self.menu_music_started = False
        self.paused = False
        self.match_end_elapsed_ms = 0
        self.match_time_remaining_ms = 0.0
        self.game_set_audio_played = False
        self.audio: AudioManager | None = None
        self.pause_overlay = pygame.image.load(str(ROOT / "assets/menu/sprites/DefineSprite_1066/1.png"))
        loading_root = ROOT / "assets/menu/sprites/DefineSprite_821_LoadingMC"
        self.match_loading_frames = LazySurfaceSequence(
            loading_root / f"{frame}.png" for frame in range(1, 6)
        )
        self.match_loading_elapsed_ms = 0
        self.fight_timer_accumulator_ms = 0
        self.game_time_seconds = 0
        self.countdown_focus_indices: list[int] = []
        self.fighters: list[PeachFighter] = []
        self.player: PeachFighter
        self._reset_match()

    def _create_inputs(self) -> list[FighterInput]:
        menu = getattr(self, "menu", None)
        defaults = [
            [pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s, pygame.K_j, pygame.K_k, pygame.K_LSHIFT],
            [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_KP0, pygame.K_KP1, pygame.K_KP2],
            [0] * 7,
            [0] * 7,
        ]
        configured = menu.control_keys if menu is not None else defaults

        def key_set(value: int) -> set[int]:
            return {int(value)} if value else set()

        inputs = []
        for index in range(4):
            keys = configured[index] if index < len(configured) else defaults[index]
            inputs.append(
                FighterInput(
                    left_keys=key_set(keys[0]),
                    right_keys=key_set(keys[1]),
                    up_keys=key_set(keys[2]),
                    down_keys=key_set(keys[3]),
                    jump_keys=set(),
                    punch_keys=key_set(keys[4]),
                    special_keys=key_set(keys[5]),
                    shield_keys=key_set(keys[6]),
                )
            )
        return inputs

    def _load_spawn_frames(self) -> dict[int, list[SpriteAssetFrame]]:
        effects = self.manifest.get("effects", {})
        frames: dict[int, list[SpriteAssetFrame]] = {}
        for kind, name in ((1, "Spawn1"), (2, "Spawn2")):
            frames[kind] = self._load_sprite_assets(effects.get(name, {}))
        return frames

    def _load_sprite_assets(self, data: dict) -> list[SpriteAssetFrame]:
        assets: list[SpriteAssetFrame] = []
        for item in data.get("frames", []):
            offset = item.get("offset", {"x": 0.0, "y": 0.0})
            assets.append(
                SpriteAssetFrame(
                    image_path=ROOT / item.get("image", item["raw"]),
                    offset=pygame.Vector2(float(offset["x"]), float(offset["y"])),
                    render_scale=max(1.0, float(item.get("render_scale", 1))),
                )
            )
        return assets

    def _load_effect_assets(self, name: str) -> list[SpriteAssetFrame]:
        return self._load_sprite_assets(self.manifest.get("effects", {}).get(name, {}))

    def _load_ui_assets(self, name: str) -> list[SpriteAssetFrame]:
        return self._load_sprite_assets(self.manifest.get("ui", {}).get(name, {}))

    def _load_effect_frames(self, name: str) -> list[pygame.Surface]:
        return LazySurfaceSequence(
            ROOT / item["raw"]
            for item in self.manifest.get("effects", {}).get(name, {}).get("frames", [])
        )

    def _effect_source_scale(self, name: str) -> float:
        frames = self.manifest.get("effects", {}).get(name, {}).get("frames", [])
        return max(1.0, float(frames[0].get("render_scale", 1)) if frames else 1.0)

    def _item_source_scale(self, name: str) -> float:
        frames = self.manifest.get("items", {}).get(name, {}).get("frames", [])
        return max(1.0, float(frames[0].get("render_scale", 1)) if frames else 1.0)

    def _load_ui_frames(self, name: str) -> list[pygame.Surface]:
        return LazySurfaceSequence(
            ROOT / item["raw"]
            for item in self.manifest.get("ui", {}).get(name, {}).get("frames", [])
        )

    def _load_item_frames(self) -> dict[str, list[SpriteAssetFrame]]:
        items = self.manifest.get("items", {})
        loaded: dict[str, list[SpriteAssetFrame]] = {}
        for kind in items.get("classes", []):
            frames = [
                SpriteAssetFrame(
                    image_path=ROOT / item["image"],
                    offset=pygame.Vector2(float(item["offset"]["x"]), float(item["offset"]["y"])),
                    render_scale=max(1.0, float(item.get("render_scale", 1))),
                )
                for item in items.get(kind, {}).get("frames", [])
            ]
            loaded[kind] = frames
        return loaded

    def _handle_keydown(self, key: int) -> None:
        for index, player_input in enumerate(self.inputs):
            if index >= len(self.fighters) or index in self.ai_controllers:
                continue
            fighter = self.fighters[index]
            move = player_input.keydown(key)
            if move is not None:
                fighter.move(move)
            if key in player_input.punch_keys:
                if player_input.up_trace:
                    fighter.attack("punch", "up")
                    player_input.up_trace = False
                fighter.attack("punch", "none")
                player_input.pending_punch_pressed = False
            elif key in player_input.special_keys:
                if player_input.up_trace:
                    fighter.attack("special", "up")
                    player_input.up_trace = False
                fighter.attack("special", "none")
                player_input.pending_special_pressed = False
            elif key in player_input.jump_keys:
                fighter.jump()
                player_input.pending_jump_pressed = False
            elif key in player_input.shield_keys:
                if fighter.xinc == 0:
                    fighter.shielded = True
                player_input.pending_shield_pressed = False

    def _handle_keyup(self, key: int) -> None:
        for index, player_input in enumerate(self.inputs):
            if index >= len(self.fighters) or index in self.ai_controllers:
                continue
            fighter = self.fighters[index]
            move = player_input.keyup(key)
            if move is not None:
                fighter.move(move)
            if key in player_input.up_keys:
                fighter.jump()
                player_input.pending_jump_pressed = False
            elif key in player_input.shield_keys:
                fighter.shielded = False
                player_input.pending_shield_released = False

    def run(self) -> None:
        pygame.init()
        self.audio = AudioManager(ROOT)
        screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
        pygame.display.set_caption("The Fight for Glorton")
        self.menu = MainMenu(ROOT, self.manifest)
        self.results = MatchResults(ROOT, self.manifest)
        self.audio.set_muted(not self.menu.sound_on)
        last_sound_on = self.menu.sound_on
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("menlo", 14)
        running = True
        while running:
            elapsed = clock.tick(60)
            if last_sound_on and not self.menu.sound_on:
                self.audio.stop_all()
                self.menu_music_started = False
            self.audio.set_muted(not self.menu.sound_on)
            last_sound_on = self.menu.sound_on
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    continue
                if self.app_state == "menu":
                    action = self.menu.handle_event(event, screen.get_size())
                    if action is not None:
                        running = self._handle_menu_action(action)
                    continue
                if self.app_state == "results":
                    action = self.results.handle_event(event, screen.get_size())
                    if action is not None:
                        running = self._handle_menu_action(action)
                    continue
                if self.paused:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        self.paused = False
                    elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                        ref_pos = MainMenu._screen_to_reference(event.pos, screen.get_size())
                        button = self.manifest.get("ui", {}).get("layout", {}).get(
                            "pause_end_button",
                            {"x": 251.8, "y": 202.7, "w": 99.0, "h": 23.25},
                        )
                        if ref_pos is not None and pygame.Rect(
                            button["x"], button["y"], button["w"], button["h"]
                        ).collidepoint(ref_pos):
                            self.paused = False
                            alive = [fighter for fighter in self.fighters if not fighter.dead]
                            self._begin_game_set(alive[0] if len(alive) == 1 else None)
                    continue
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p and self.match_state in {"countdown", "playing"}:
                        self.paused = True
                    elif event.key == pygame.K_F1:
                        self.show_debug = not self.show_debug
                    elif event.key == pygame.K_r:
                        self._reset_match()
                    elif self.match_state not in {"loading", "countdown", "playing"}:
                        continue
                    else:
                        self._handle_keydown(event.key)
                elif event.type == pygame.KEYUP:
                    if self.match_state in {"loading", "countdown", "playing"}:
                        self._handle_keyup(event.key)

            if self.app_state == "menu":
                self.accumulator = 0
                action = self.menu.update(elapsed)
                if action is not None:
                    running = self._handle_menu_action(action)
                self._sync_menu_music()
                self.menu.draw(screen)
            elif self.app_state == "results":
                self.accumulator = 0
                self.results.update(elapsed)
                self.results.draw(screen)
            else:
                self._advance_battle_time(elapsed)
                keys = pygame.key.get_pressed()
                while not self.paused and self.accumulator >= TICK_MS:
                    self.stage.set_time(self.stage_time_ms)
                    if self.match_state in {"loading", "countdown", "playing"}:
                        controls = [player_input.controls(keys) for player_input in self.inputs]
                        if self.match_state in {"loading", "countdown"}:
                            self._fixed_tick_countdown(controls)
                        else:
                            self._fixed_tick_match(controls)
                    self.accumulator -= TICK_MS
                if self.match_state == "game_set" and not self.game_set_audio_played:
                    self._play_game_set_audio()
                if self.match_state == "game_set" and not self.paused:
                    self.match_end_elapsed_ms += elapsed
                    if self.match_end_elapsed_ms >= self._pre_end_duration_ms():
                        self.results.start(
                            self.fighters,
                            self.match_winner,
                            str(self.match_config.get("limit_mode", "stock")),
                            str(self.match_config.get("type", "vsmode")),
                            self.killed_players,
                            self.game_time_seconds,
                        )
                        self.app_state = "results"
                        self.accumulator = 0
                if self.app_state == "results":
                    self.results.draw(screen)
                else:
                    self._draw(screen, font)
                    if self.paused:
                        self._draw_pause_overlay(screen)
            pygame.display.flip()
        self.audio.stop_all()
        pygame.quit()

    def _handle_menu_action(self, action: MenuAction) -> bool:
        if action.kind == "quit":
            return False
        if action.kind == "return_main":
            self.app_state = "menu"
            self.paused = False
            self.accumulator = 0
            self.menu.return_to_main()
            if self.audio is not None:
                self.audio.stop_all()
                self.audio.play_loop("menu_music", "music")
                self.menu_music_started = self.menu.sound_on
            return True
        if action.kind == "open_url":
            url = str((action.payload or {}).get("url", "http://www.armorgames.com/"))
            webbrowser.open(url)
            return True
        if action.kind != "start_game":
            return True
        self.match_config = dict(action.payload or {})
        self.manifest["match"]["limit_mode"] = str(self.match_config.get("limit_mode", "stock"))
        if self.match_config.get("limit_mode") == "stock":
            self.manifest["match"]["starting_lives"] = int(self.match_config.get("limit_value", STARTING_LIVES))
        else:
            self.manifest["match"]["starting_lives"] = -1
        selected_stage = str(self.match_config.get("selected_stage", self.match_config.get("stage", "Rooftop")))
        self.stage = Stage(self.manifest, selected_stage)
        self.app_state = "battle"
        self.paused = False
        self.match_end_elapsed_ms = 0
        self.accumulator = 0
        self._reset_match()
        if self.audio is not None:
            self.audio.stop_all()
            self.menu_music_started = False
            self.audio.play_loop(selected_stage.lower(), "music")
            if selected_stage == "Rooftop":
                self.audio.play_loop("helicopter", "ambience")
            elif selected_stage == "B52":
                self.audio.play_loop("jet_engine", "ambience")
        return True

    def _sync_menu_music(self) -> None:
        if self.audio is None or self.app_state != "menu" or not self.menu.sound_on:
            return
        if self.menu.scene in {"preloader", "sponsor_intro", "opening"}:
            return
        if not self.menu_music_started:
            self.audio.play_loop("menu_music", "music")
            self.menu_music_started = True

    def _draw_pause_overlay(self, screen: pygame.Surface) -> None:
        rect = MainMenu._screen_rect(screen.get_size())
        overlay = self._quality_scale(self.pause_overlay, rect.size)
        screen.blit(overlay, rect)

    def _reset_match(self) -> None:
        player_configs = [
            dict(item)
            for item in self.match_config.get("players", [])
            if item.get("enabled", True) and item.get("fighter")
        ]
        if not player_configs:
            selected_colors = list(self.match_config.get("colors", [0, 1]))
            selected_fighters = list(self.match_config.get("fighters", ["PeachPlayer", "PeachPlayer"]))
            player_configs = [
                {
                    "fighter": fighter_name,
                    "color": selected_colors[index] if index < len(selected_colors) else index,
                    "computer": False,
                    "level": 7,
                    "team_index": index,
                }
                for index, fighter_name in enumerate(selected_fighters)
            ]
        self.endurance_level = 0
        self.killed_players = 0
        if str(self.match_config.get("type", "")) == "endurance" and len(player_configs) == 1:
            human_color = int(player_configs[0].get("color", 0))
            player_configs.append(
                {
                    "fighter": random.choice(tuple(self.manifest.get("fighters", {"PeachPlayer": {}}))),
                    "color": 1 if human_color == 0 else 0,
                    "computer": True,
                    "level": 1,
                    "endurance": True,
                }
            )
            self.endurance_level = 1
        player_configs = player_configs[:4]
        self.fighters = []
        self.ai_controllers = {}
        for index, config in enumerate(player_configs):
            spawn_name = ACTIVE_SPAWNS[index]
            fighter = PeachFighter(
                self.manifest,
                self.stage.spawn_point(spawn_name),
                f"P{index + 1}",
                int(config.get("color", index)),
                str(config.get("fighter", "PeachPlayer")),
                int(config.get("team_index", index)),
            )
            fighter.draw_depth = index
            if config.get("endurance"):
                fighter.lives = 1
            self.fighters.append(fighter)
            if bool(config.get("computer", False)):
                self.ai_controllers[index] = AIController(
                    fighter,
                    self.stage,
                    int(config.get("level", 7)),
                    force_victim=bool(config.get("endurance", False)),
                )
        if not self.fighters:
            self.fighters = [
                PeachFighter(self.manifest, self.stage.spawn_point("SpawnP1"), "P1", 0, "PeachPlayer")
            ]
        self.player = self.fighters[0]
        if 1 in self.ai_controllers and self.ai_controllers[1].force_victim:
            self.ai_controllers[1].victim = self.player
        for fighter in self.fighters:
            fighter.intro_visible = False
        self.bullets.clear()
        self.rockets.clear()
        self.special_projectiles.clear()
        self.inputs = self._create_inputs()
        self.match_state = "playing"
        self.match_winner = None
        self.num_dead = 0
        self.match_end_elapsed_ms = 0
        self.match_time_remaining_ms = (
            float(self.match_config.get("limit_value", 0)) * 1000
            if self.manifest.get("match", {}).get("limit_mode") == "time"
            else 0.0
        )
        self.game_set_audio_played = False
        self.death_effects.clear()
        self.camera_tricks.clear()
        self.spawn_effects.clear()
        self.hit_effects.clear()
        self.explosions.clear()
        self.items.clear()
        self.item_gen_timer_ms = 0
        self.camera_shake_start_ms = 0
        self.camera_shake_until_ms = 0
        self.ready_set = 5
        self.ready_timer_ms = 0
        self.ready_text = ""
        self.countdown_focus_indices = []
        self.match_state = "loading"
        self.match_loading_elapsed_ms = 0
        self.fight_timer_accumulator_ms = 0
        self.game_time_seconds = 0
        self.camera_view = None
        self.camera_target_view = None
        self.stage_time_ms = 0
        self.stage.set_time(self.stage_time_ms)
        self._stage_surface_cache.clear()
        # Surface object ids may be reused after a stage or fighter set is
        # replaced. Alpha bounds cached by id therefore belong to one match.
        self._surface_bounds_cache.clear()
        self._stage_backdrop_cache_key = None
        self._stage_backdrop_cache = None
        self._stage_foreground_cache_key = None
        self._stage_foreground_cache = None
        self._stage_scene_cache_key = None
        self._stage_scene_cache = None

    def _fixed_tick_countdown(self, player_controls: list[dict[str, bool]] | None = None) -> None:
        controls_by_player = player_controls or [{} for _ in self.fighters]
        source_computers = list(self.ai_controllers.values())
        for index, fighter in enumerate(self.fighters):
            controls = (
                controls_by_player[index]
                if index < len(controls_by_player) and index not in self.ai_controllers
                else {}
            )
            fighter.advance_pre_game_tick(controls)
            fighter.advance_intro_tick()
            if index < len(source_computers):
                source_computers[index].fixed_tick(self.fighters)
        if self.match_state == "countdown":
            self._step_camera()

    def _start_match_countdown(self) -> None:
        self.match_state = "countdown"
        # FightTimer case 5 calls VCam.init(): Ratio is captured from 600/400,
        # then x, y, Width and Height are all reset to zero before /5 tracking.
        self.camera_view = [0.0, 0.0, 0.0, 0.0]
        self.camera_target_view = None
        self._stage_scene_cache_key = None
        self._stage_scene_cache = None
        self.ready_set = 5
        self.ready_timer_ms = 0
        self.ready_text = "5"
        self._apply_ready_step()

    def _advance_fight_timer(self, elapsed_ms: int) -> None:
        if self.match_state == "game_set":
            return
        self.fight_timer_accumulator_ms += max(0, elapsed_ms)
        while self.fight_timer_accumulator_ms >= 1000 and self.match_state != "game_set":
            self.fight_timer_accumulator_ms -= 1000
            self._fight_timer_tick()

    def _fight_timer_tick(self) -> None:
        if self.match_state == "loading":
            self._start_match_countdown()
        elif self.match_state == "countdown":
            self.ready_set -= 1
            self._apply_ready_step()
        if self.paused:
            return
        self.game_time_seconds += 1
        if self.match_state != "playing" or self.manifest.get("match", {}).get("limit_mode") != "time":
            return
        self.match_time_remaining_ms = max(0.0, self.match_time_remaining_ms - 1000)
        if self.match_time_remaining_ms > 0:
            return
        self._begin_game_set(
            max(
                self.fighters,
                key=lambda fighter: fighter.kos - fighter.deaths,
                default=None,
            )
        )

    def _apply_ready_step(self) -> None:
        if self.ready_set == 5:
            self._show_intro_player(0)
            self.countdown_focus_indices = [0]
            self.ready_text = "5"
        elif self.ready_set == 4:
            if len(self.fighters) > 2:
                self._show_intro_player(1)
                self.countdown_focus_indices = [1]
            else:
                self.countdown_focus_indices = [0]
            self.ready_text = "4"
        elif self.ready_set == 3:
            if len(self.fighters) == 2:
                self._show_intro_player(1)
                self.countdown_focus_indices = [1]
            elif len(self.fighters) > 2:
                self._show_intro_player(2)
                self.countdown_focus_indices = [2]
            self.ready_text = "3"
        elif self.ready_set == 2:
            if len(self.fighters) == 4:
                self._show_intro_player(3)
                self.countdown_focus_indices = [3]
            elif len(self.fighters) == 2:
                self.countdown_focus_indices = [1]
            self.ready_text = "2"
        elif self.ready_set == 1:
            self.countdown_focus_indices = list(range(len(self.fighters)))
            self.ready_text = "1"
        elif self.ready_set == 0:
            self.countdown_focus_indices = list(range(len(self.fighters)))
            self.ready_text = "GO!"
        elif self.ready_set <= -1:
            self.match_state = "playing"
            self.ready_text = ""
            self.countdown_focus_indices = []
            for fighter in self.fighters:
                fighter.intro_visible = True
                if not fighter.dead:
                    fighter.finish_intro_spawn()

    def _show_intro_player(self, index: int) -> None:
        if index < 0 or index >= len(self.fighters):
            return
        fighter = self.fighters[index]
        if fighter.intro_visible:
            return
        fighter.start_intro_spawn()

    def _start_spawn_effect(self, fighter: PeachFighter) -> None:
        # In the original, Spawn1/Spawn2 is attached to the player and removed
        # when the player leaves State=="spawn"; keep it tied to fighter state
        # instead of playing a detached duplicate effect.
        return

    def _fixed_tick_match(self, player_controls: list[dict[str, bool]]) -> None:
        self._fixed_tick_items()
        # DoMain indexes the dense Computers[] array with the fighter-loop
        # index. With P1 human, Computers[0] (P2) therefore acts after P1 and
        # before P2 is simulated, rather than after its own fighter tick.
        source_computers = list(self.ai_controllers.values())
        for index, fighter in enumerate(self.fighters):
            controls = (
                player_controls[index]
                if index < len(player_controls) and index not in self.ai_controllers
                else {}
            )
            fighter.fixed_tick(
                self.stage,
                controls,
                self.bullets,
                self.rockets,
                self.special_projectiles,
            )
            if fighter.pending_stage_boom:
                self._start_explosion(fighter.pos, None, 4)
                fighter.pending_stage_boom = False
            self._collect_fighter_sounds(fighter)
            self._collect_puffs(fighter)
            self._collect_death_event(fighter)
            if fighter.resolve_attack_this_tick:
                self._resolve_melee_hits(fighter)
            fighter.finish_post_collision_tick()
            if index < len(source_computers):
                source_computers[index].fixed_tick(self.fighters)
        for bullet in self.bullets:
            bullet.fixed_tick(self.stage)
        for rocket in self.rockets:
            rocket.fixed_tick(self.stage)
        for projectile in self.special_projectiles:
            projectile.fixed_tick(self.stage)
        self._resolve_bullet_hits()
        self._resolve_rocket_hits()
        self._resolve_special_projectile_hits()
        self.bullets = [bullet for bullet in self.bullets if bullet.alive]
        self.rockets = [rocket for rocket in self.rockets if rocket.alive]
        self.special_projectiles = [projectile for projectile in self.special_projectiles if projectile.alive]
        self._resolve_item_collisions()
        self._tick_explosions()
        for fighter in self.fighters:
            self._collect_fighter_sound_stops(fighter)
        self._tick_death_effects()
        self._tick_spawn_effects()
        self._update_match_state()
        self._step_camera()

    def _fixed_tick_items(self) -> None:
        self.item_gen_timer_ms += TICK_MS
        frequency = int(self.manifest.get("items", {}).get("frequency", 0))
        if frequency > 0 and self.item_gen_timer_ms >= 10000 / frequency:
            self.item_gen_timer_ms = 0
            classes = self.manifest.get("items", {}).get("classes", [])
            roll_max = len(classes) + int(50 / frequency)
            roll = random.randrange(max(1, roll_max))
            if roll < len(classes):
                kind = classes[roll]
                frames = self.item_frames.get(kind, [])
                if frames:
                    pos = self.stage.item_spawn_point()
                    pos.y -= 2
                    self.items.append(
                        StageItem(
                            kind=kind,
                            pos=pos,
                            frames=frames,
                            frame_labels=self.item_frame_labels.get(kind, {}),
                            source_scale=self.item_source_scales.get(kind, 1.0),
                            life_ms=int(self.manifest.get("items", {}).get(kind, {}).get("life_ms", 20000)),
                        )
                    )
        for item in self.items:
            item.fixed_tick(self.stage)
        self.items = [item for item in self.items if item.alive]

    def _resolve_item_collisions(self) -> None:
        for item in self.items:
            if item.state != 2 or not item.alive:
                continue
            platform = self.stage.item_hit_platform(item.hitbox())
            if platform is not None:
                if item.kind == "Grenade":
                    self._start_explosion(item.pos, item.sender, 4)
                    item.alive = False
                    continue
                elif item.kind == "Mine":
                    previous = getattr(item, "prev_pos", item.pos)
                    previous_box = item.hitbox_at(previous)
                    current_box = item.hitbox()
                    floor_hit = (
                        item.yinc > 0
                        and previous_box.bottom <= platform.rect.top
                        and current_box.bottom >= platform.rect.top
                        and platform.rect.left <= current_box.centerx <= platform.rect.right
                    )
                    right_hit = (
                        not floor_hit
                        and previous_box.right <= platform.rect.left
                        and current_box.right >= platform.rect.left
                    )
                    left_hit = (
                        not floor_hit
                        and previous_box.left >= platform.rect.right
                        and current_box.left <= platform.rect.right
                    )
                    if right_hit:
                        item.pos.x = platform.rect.left
                        item.rotation = -90.0
                    elif left_hit:
                        item.pos.x = platform.rect.right
                        item.rotation = 90.0
                    elif floor_hit:
                        item.pos.y = platform.rect.top
                        item.rotation = 0.0
                    else:
                        platform = None
                    if platform is not None:
                        item.state = 0
                        item.active_platform = platform
                        item.active_offset = pygame.Vector2(item.pos.x - platform.rect.x, item.pos.y - platform.rect.y)
                        self._play_sound("mine_activate")
            if item.kind != "Grenade":
                if item.kind == "Mine":
                    item_box = item.hitbox()
                    if item.influenced is None:
                        item.influenced = set()
                    for index, fighter in enumerate(self._fighters()):
                        if fighter is item.sender or fighter.dead or index in item.influenced:
                            continue
                        if item_box.colliderect(fighter.visual_bounds()):
                            fighter.throw_impulse(2, 45, item.sender)
                            fighter.damage(20, item.sender)
                            bounce_angle = 5 if item.xinc <= 0 else -5
                            item.pos.x -= item.xinc
                            item.throw(abs(item.xinc) * 0.6, bounce_angle)
                            item.influenced.add(index)
                            break
                continue
            item_box = item.hitbox()
            for fighter in self._fighters():
                if fighter is item.sender or fighter.dead:
                    continue
                if item_box.colliderect(fighter.visual_bounds()):
                    self._start_explosion(item.pos, item.sender, 4)
                    item.alive = False
                    break
        for item in self.items:
            if item.alive and item.kind == "Grenade" and item.state == 2:
                item.rotation += 10
        for item in self.items:
            if item.kind != "Mine" or item.state != 0 or item.active_ms < 1000:
                continue
            item_box = item.hitbox()
            for fighter in self._fighters():
                if fighter.dead:
                    continue
                if item_box.colliderect(fighter.visual_bounds()):
                    self._start_explosion(item.pos, item.sender, 3)
                    item.alive = False
                    break
        self.items = [item for item in self.items if item.alive]

    def _start_explosion(self, pos: pygame.Vector2, sender: PeachFighter | None, size: int) -> None:
        self._play_sound("boom")
        matter_offsets = [
            pygame.Vector2(random.randrange(max(1, size * 10)) - size * 5, random.randrange(max(1, size * 10)) - size * 5)
            for _ in range(size)
        ]
        self.explosions.append(
            ExplosionEffect(
                pos=pygame.Vector2(pos),
                size=size,
                sender=sender,
                frames=self.boom_star_frames,
                wave_frames=self.boom_wave_frames,
                matter_frames=self.boom_matter_frames,
                matter_offsets=matter_offsets,
                influenced=set(),
            )
        )
        self.camera_shake_start_ms = self.stage_time_ms - 500
        self.camera_shake_until_ms = self.stage_time_ms + 500

    def _tick_explosions(self) -> None:
        for explosion in self.explosions:
            if explosion.influenced is None:
                explosion.influenced = set()
            if explosion.damage_active:
                for index, fighter in enumerate(self._fighters()):
                    if fighter.dead or index in explosion.influenced:
                        continue
                    distance_sq = (fighter.pos.x - explosion.pos.x) ** 2 + (fighter.pos.y - explosion.pos.y) ** 2
                    if distance_sq <= explosion.square_size:
                        fighter.damage(explosion.size * 5, explosion.sender, force=True)
                        direction = -1 if fighter.pos.x <= explosion.pos.x else 1
                        fighter.throw_impulse(explosion.size * 2, 45 * direction, explosion.sender, force=True)
                        explosion.influenced.add(index)
            explosion.age_ms += TICK_MS
        self.explosions = [explosion for explosion in self.explosions if explosion.alive]

    def _collect_death_event(self, fighter: PeachFighter) -> None:
        event = fighter.pending_death_event
        if event is None:
            return
        fighter.pending_death_event = None
        self._play_sound("thunder")
        self._play_sound("boom")
        self.death_effects.append(
            DeathEffect(
                pygame.Vector2(event.pos),
                event.death_type,
                self.player_death_frames,
                life_ms=max(DEATH_EFFECT_MS, round(len(self.player_death_frames) / ANIMATION_FPS * 1000)),
            )
        )
        self.camera_tricks.append(
            DeathEffect(pygame.Vector2(event.pos.x * 0.95, event.pos.y * 0.95), event.death_type, [], life_ms=CAMERA_TRICK_MS)
        )
        if not fighter.dead:
            self._start_spawn_effect(fighter)
        self.camera_shake_start_ms = self.stage_time_ms + 500
        self.camera_shake_until_ms = self.camera_shake_start_ms + 1000

    def _collect_puffs(self, fighter: PeachFighter) -> None:
        if not fighter.pending_puffs:
            return
        if self.puff_frames:
            for pos, rotation in fighter.pending_puffs:
                self.hit_effects.append(
                    HitEffect(
                        pygame.Vector2(pos),
                        self.puff_frames,
                        rotation=rotation,
                        scale=1.0,
                        source_scale=self.puff_source_scale,
                        root_layer=True,
                    )
                )
        fighter.pending_puffs.clear()

    def _collect_fighter_sounds(self, fighter: PeachFighter) -> None:
        for sound_name in fighter.pending_sounds:
            self._play_sound(sound_name)
        fighter.pending_sounds.clear()

    def _collect_fighter_sound_stops(self, fighter: PeachFighter) -> None:
        if self.audio is not None:
            for sound_name in fighter.pending_stop_sounds:
                self.audio.stop(sound_name)
        fighter.pending_stop_sounds.clear()

    def _play_sound(self, name: str) -> None:
        if self.audio is not None:
            self.audio.play(name)

    def _play_game_set_audio(self) -> None:
        self.game_set_audio_played = True
        if self.audio is None:
            return
        self.audio.stop_all()
        self.audio.play("thunder")
        self.audio.play("boom")

    def _begin_game_set(self, winner: PeachFighter | None) -> None:
        if self.match_state == "game_set":
            return
        self.match_state = "game_set"
        self.match_winner = winner
        self.match_end_elapsed_ms = 0
        self.game_set_audio_played = False
        self.accumulator = 0

    def _advance_battle_time(self, elapsed_ms: int) -> None:
        elapsed_ms = max(0, elapsed_ms)
        if self.match_state == "loading":
            self.match_loading_elapsed_ms += elapsed_ms
        self._advance_fight_timer(elapsed_ms)
        if self.match_state == "game_set":
            return
        if self.paused:
            return
        self.stage_time_ms += elapsed_ms
        self.accumulator += elapsed_ms

    def _pre_end_duration_ms(self) -> float:
        data = self.manifest.get("results", {})
        frames = int(data.get("pre_end_stop_frame", 100)) - int(data.get("pre_end_start_frame", 52)) + 1
        return frames * 1000 / max(1.0, float(data.get("frame_rate", 30)))

    def _tick_death_effects(self) -> None:
        for effect in self.death_effects:
            effect.age_ms += TICK_MS
        for trick in self.camera_tricks:
            trick.age_ms += TICK_MS
        self.death_effects = [effect for effect in self.death_effects if effect.alive]
        self.camera_tricks = [trick for trick in self.camera_tricks if trick.alive]

    def _tick_spawn_effects(self) -> None:
        for effect in self.spawn_effects:
            effect.age_ms += TICK_MS
        self.spawn_effects = [effect for effect in self.spawn_effects if effect.alive]
        for effect in self.hit_effects:
            effect.age_ms += TICK_MS
        self.hit_effects = [effect for effect in self.hit_effects if effect.alive]

    def _update_match_state(self) -> None:
        if self.manifest.get("match", {}).get("limit_mode", "stock") == "time":
            return
        if str(self.match_config.get("type", "")) == "endurance":
            if not self.fighters or self.fighters[0].dead:
                self._begin_game_set(None)
                return
            if len(self.fighters) > 1 and self.fighters[1].dead:
                self._replace_endurance_opponent()
            return
        for fighter in self.fighters:
            if fighter.dead and fighter.death_order is None:
                fighter.death_order = self.num_dead
                self.num_dead += 1
        alive = [fighter for fighter in self.fighters if not fighter.dead]
        if len(self.fighters) == 1:
            if not alive:
                self._begin_game_set(None)
            return
        if len(alive) <= 1:
            self._begin_game_set(alive[0] if alive else None)

    def _replace_endurance_opponent(self) -> None:
        if len(self.fighters) < 2:
            return
        self.endurance_level += 1
        self.killed_players += 1
        human = self.fighters[0]
        color = 1 if human.color_frame == 1 else 0
        opponent = PeachFighter(
            self.manifest,
            pygame.Vector2(self.stage.spawn_point("SpawnP2").x, 0),
            "P2",
            color,
            random.choice(tuple(self.manifest.get("fighters", {"PeachPlayer": {}}))),
            1,
        )
        opponent.draw_depth = self.fighters[1].draw_depth
        opponent.lives = 1
        opponent.intro_visible = True
        opponent.has_control = True
        opponent.state = "stop"
        opponent.current_label = "still"
        self.fighters[1] = opponent
        self.ai_controllers[1] = AIController(
            opponent,
            self.stage,
            self.endurance_level,
            force_victim=True,
        )
        self.ai_controllers[1].victim = human

    def _fighters(self) -> list[PeachFighter]:
        return self.fighters

    def _resolve_bullet_hits(self) -> None:
        for bullet in self.bullets:
            if not bullet.alive:
                continue
            bullet_box = bullet.hitbox()
            for fighter in self._fighters():
                if fighter is bullet.sender or fighter.dead or fighter.invincible or fighter.state == "ko":
                    continue
                if bullet_box.colliderect(fighter.visual_bounds()):
                    self._play_sound("headshot")
                    fighter.damage(10, bullet.sender)
                    fighter.throw_impulse(3, 45 * (-1 if bullet.xinc <= 0 else 1), bullet.sender)
                    bullet.alive = False
                    break

    def _resolve_rocket_hits(self) -> None:
        for rocket in self.rockets:
            if not rocket.alive:
                continue
            rocket_box = rocket.hitbox()
            for fighter in self._fighters():
                if fighter is rocket.sender or fighter.dead or fighter.invincible or fighter.state == "ko":
                    continue
                if rocket_box.colliderect(fighter.visual_bounds()):
                    self._start_explosion(rocket.pos, rocket.sender, 5)
                    rocket.alive = False
                    break

    def _resolve_special_projectile_hits(self) -> None:
        for projectile in self.special_projectiles:
            if not projectile.alive:
                continue
            projectile_box = projectile.hitbox()
            for fighter in self._fighters():
                if (
                    fighter is projectile.sender
                    or fighter.dead
                    or fighter.invincible
                    or fighter.state == "ko"
                ):
                    continue
                if not projectile_box.colliderect(fighter.visual_bounds()):
                    continue
                config = projectile.config
                sound = {"Pencil": "punch", "Poop": "fart", "Garbage": "punch"}.get(projectile.kind)
                if sound:
                    self._play_sound(sound)
                damage_applied = fighter.damage(int(config.get("damage", 0)), projectile.sender)
                if damage_applied:
                    electrocuted_ms = int(config.get("electrocuted_ms", 0))
                    if electrocuted_ms:
                        fighter.state = "electrocuted"
                        fighter.electrocuted_ms = electrocuted_ms
                        fighter.paralized = electrocuted_ms
                    direction = -1 if projectile.xinc <= 0 else 1
                    throw_sender = projectile if projectile.kind == "EnergyBall" else projectile.sender
                    fighter.throw_impulse(
                        float(config.get("throw_power", 0)),
                        float(config.get("throw_angle", 45)) * direction,
                        throw_sender,
                    )
                projectile.alive = False
                break

    def _resolve_melee_hits(self, only_attacker: PeachFighter | None = None) -> None:
        for attacker in self._fighters():
            if only_attacker is not None and attacker is not only_attacker:
                continue
            profile = attacker.fighter_data.get("attacks", {}).get(attacker.current_attack)
            if not profile:
                continue
            if attacker.dead:
                continue
            hitbox = attacker.attack_hitbox()
            if hitbox is None:
                continue
            for target in self._fighters():
                if target is attacker or target.dead or target.invincible or target.state in {"spawn", "ko"}:
                    continue
                if target.state in {"thrown", "ko"} and attacker.attack_done:
                    continue
                attack_facing = attacker.attack_facing if attacker.current_attack else attacker.facing
                target_box = target.visual_bounds()
                if not hitbox.colliderect(target_box):
                    continue
                if attacker.current_attack in {"punchGround", "punchRun"} and attack_facing * (attacker.pos.x - target.pos.x) > 0:
                    if attacker.can_back_throw_now() and abs(attacker.pos.x - target.pos.x) <= 20:
                        attacker._start_back_throw(target)
                        break
                    continue
                current_frame = attacker.attack_frame or attacker._timeline_frame(attacker.current_attack)
                active_min = int(profile.get("active_frame_min", 0))
                active_max = int(profile.get("active_frame_max", 0))
                if active_min and current_frame < active_min:
                    continue
                if active_max and current_frame > active_max:
                    continue
                minimum_gap = float(profile.get("minimum_vertical_gap", -math.inf))
                if minimum_gap != -math.inf and attacker.pos.y - target.pos.y < minimum_gap:
                    continue
                maximum_y = profile.get("maximum_target_y_offset")
                if maximum_y is not None and target.pos.y > attacker.pos.y + float(maximum_y):
                    continue

                damage = profile.get("damage")
                power = profile.get("throw_power")
                angle = profile.get("angle")
                if profile.get("kind") == "kamehameha":
                    distance = abs(target.pos.x - attacker.pos.x)
                    if not (
                        float(profile.get("distance_min", 20)) < distance
                        < float(profile.get("distance_max", 240))
                    ):
                        continue
                    factor = 1.0 - distance / 230.0
                    if factor < 0.2:
                        continue
                    damage = math.floor(10 * factor + 0.5)
                    power = 5 * factor
                    angle = 45
                    self._play_sound("boom")
                if damage is None or power is None or angle is None:
                    continue
                hit_pos = attacker.attack_contact_point(attacker.attack_frame or 1) or pygame.Vector2(hitbox.centerx, hitbox.centery)
                if not attacker.current_attack.startswith("special"):
                    self._play_sound("punch")
                elif int(profile.get("electrocuted_ms", 0)):
                    self._play_sound("electric")
                elif attacker.current_attack == "specialUp" and attacker.special_kind in {
                    "pencil",
                    "kamehameha",
                    "poop",
                }:
                    self._play_sound("punch")
                damage_applied = target.damage(int(damage), attacker)
                if damage_applied:
                    electrocuted_ms = int(profile.get("electrocuted_ms", 0))
                    if electrocuted_ms:
                        target.state = "electrocuted"
                        target.electrocuted_ms = electrocuted_ms
                        target.paralized = electrocuted_ms
                    target.throw_impulse(float(power), float(angle) * attack_facing, attacker)
                if damage_applied and not attacker.current_attack.startswith("special"):
                    self._start_punch_damage(hit_pos, int(damage))
                attacker.attack_done = True
                break
            self._resolve_item_hits(attacker, hitbox)

    def _resolve_item_hits(self, attacker: PeachFighter, hitbox: pygame.Rect) -> None:
        if attacker.current_attack not in {"punchGround", "punchUp"}:
            return
        if attacker.current_item:
            return
        for item in self.items:
            if item.state != 1 or not item.alive:
                continue
            if not hitbox.colliderect(item.hitbox()):
                continue
            item.state = 3
            item.sender = attacker
            attacker.current_item = item.kind.lower()
            attacker.current_item_obj = item
            break

    def _start_punch_damage(self, pos: pygame.Vector2, amount: int) -> None:
        if not self.punch_damage_frames:
            return
        self.hit_effects.append(
            HitEffect(
                pygame.Vector2(pos),
                self.punch_damage_frames,
                rotation=random.randrange(40) - 20,
                scale=max(0.5, math.log(max(2, amount)) * 0.5),
                source_scale=self.punch_damage_source_scale,
            )
        )

    def _render_alpha(self) -> float:
        return max(0.0, min(1.0, self.accumulator / TICK_MS))

    def _camera(self, viewport: pygame.Rect, alpha: float = 1.0) -> tuple[pygame.Vector2, float]:
        if self.camera_view is None:
            self.camera_view = self._camera_target(viewport, alpha)

        cam_x, cam_y, cam_w, cam_h = self._clamp_camera_view(self.camera_view)
        self.camera_view = [cam_x, cam_y, cam_w, cam_h]
        viewport_w = max(1, viewport.w)
        viewport_h = max(1, viewport.h)
        zoom = min(viewport_w / max(1, cam_w), viewport_h / max(1, cam_h))
        return pygame.Vector2(cam_x, cam_y), zoom

    def _step_camera(self) -> None:
        target = self._camera_target(pygame.Rect(0, 0, 600, 400), 1.0)
        self.camera_target_view = list(target)
        if self.camera_view is None:
            self.camera_view = target
        else:
            for index, value in enumerate(target):
                self.camera_view[index] += (value - self.camera_view[index]) / CAMERA_LERP
        self.camera_view = self._clamp_camera_view(self.camera_view)

    def _camera_shake_offset(self) -> pygame.Vector2:
        if self.camera_shake_until_ms <= 0:
            return pygame.Vector2()
        now = self.stage_time_ms
        if now > self.camera_shake_until_ms:
            self.camera_shake_until_ms = 0
            return pygame.Vector2()
        elapsed = now - self.camera_shake_start_ms
        strength = max(0.0, (self.camera_shake_until_ms - now) / 7)
        return pygame.Vector2(math.cos(elapsed) * strength, math.sin(elapsed) * strength)

    def _camera_target(self, viewport: pygame.Rect, alpha: float = 1.0) -> list[float]:
        focus_rects = []
        indexed_fighters = list(enumerate(self._fighters()))
        if self.match_state == "countdown" and self.countdown_focus_indices:
            indexed_fighters = [
                (index, self.fighters[index])
                for index in self.countdown_focus_indices
                if 0 <= index < len(self.fighters)
            ]
        for _, fighter in indexed_fighters:
            if fighter.dead or not fighter.intro_visible:
                continue
            focus_rects.append(fighter.camera_focus_bounds(self.stage, self.camera_view, alpha))
        for trick in self.camera_tricks:
            focus_rects.append(pygame.Rect(round(trick.pos.x - 1), round(trick.pos.y - 1), 2, 2))
        if not focus_rects:
            focus_rects = [self.player.hurtbox()]

        left = min(rect.left for rect in focus_rects) - CAMERA_PADDING
        top = min(rect.top for rect in focus_rects) - CAMERA_PADDING
        right = max(rect.right for rect in focus_rects) + CAMERA_PADDING
        bottom = max(rect.bottom for rect in focus_rects) + CAMERA_PADDING
        shake = self._camera_shake_offset()
        left += shake.x
        top += shake.y
        width = max(1.0, right - left)
        height = max(1.0, bottom - top)

        if width / height > CAMERA_RATIO:
            height = width / CAMERA_RATIO
            top -= height / 4
        else:
            width = height * CAMERA_RATIO

        return [left, top, width, height]

    def _clamp_camera_view(self, view: list[float]) -> list[float]:
        x, y, width, height = view
        bounds = self.stage.bounds_cam
        if y + height > bounds.bottom:
            y = bounds.bottom - height
        if y < bounds.top:
            y = bounds.top
        if x < bounds.left:
            x = bounds.left
        if x + width > bounds.right:
            x = bounds.right - width
        return [x, y, width, height]

    def _clamp_render_camera(self, x: float, y: float, width: float, height: float) -> tuple[float, float]:
        bounds = self.stage.view_bounds
        if width <= bounds.w:
            x = max(bounds.left, min(x, bounds.right - width))
        else:
            x = bounds.centerx - width / 2
        if height <= bounds.h:
            y = max(bounds.top, min(y, bounds.bottom - height))
        else:
            y = bounds.centery - height / 2
        return x, y

    def _world_to_screen(self, point: pygame.Vector2, cam: pygame.Vector2, zoom: float, viewport: pygame.Rect) -> pygame.Vector2:
        return pygame.Vector2(viewport.x + (point.x - cam.x) * zoom, viewport.y + (point.y - cam.y) * zoom)

    def _scale_stage_surface(
        self,
        surface: pygame.Surface,
        width: int,
        height: int,
        method: str = "auto",
        cache: bool = True,
    ) -> pygame.Surface:
        width = max(1, width)
        height = max(1, height)
        if surface.get_width() == width and surface.get_height() == height:
            return surface
        key = (id(surface), width, height, method)
        if cache and key in self._stage_surface_cache:
            return self._stage_surface_cache[key]
        if method == "smooth" or (method == "auto" and (width < surface.get_width() or height < surface.get_height())):
            scaled = self._quality_scale(surface, (width, height))
        else:
            scaled = pygame.transform.scale(surface, (width, height))
        if cache:
            if len(self._stage_surface_cache) > 12:
                self._stage_surface_cache.clear()
            self._stage_surface_cache[key] = scaled
        return scaled

    def _quality_scale(self, surface: pygame.Surface, size: tuple[int, int]) -> pygame.Surface:
        quality = getattr(getattr(self, "menu", None), "quality", "MEDIUM")
        if quality == "LOW":
            return pygame.transform.scale(surface, size)
        return pygame.transform.smoothscale(surface, size)

    def _draw_sky_backdrop(self, screen: pygame.Surface, viewport: pygame.Rect) -> None:
        if viewport.w <= 0 or viewport.h <= 0:
            return
        cache_key = viewport.size
        backdrop = self._sky_surface_cache.get(cache_key)
        if backdrop is None:
            backdrop = pygame.Surface(viewport.size)
            for y in range(viewport.h):
                t = y / max(1, viewport.h - 1)
                blue = round(4 + 34 * t)
                pygame.draw.line(backdrop, (0, 0, blue), (0, y), (viewport.w, y))
            self._sky_surface_cache = {cache_key: backdrop}
        screen.blit(backdrop, viewport.topleft)

    def _draw_background(self, screen: pygame.Surface, cam: pygame.Vector2, zoom: float, viewport: pygame.Rect) -> None:
        background = self.stage.background
        source_scale = self.stage.background_scale
        logical_width = self.stage.background_canvas_size.x
        logical_height = self.stage.background_canvas_size.y
        base_scale = min(viewport.w / 600, viewport.h / 400)
        camera_scale = zoom / max(0.001, base_scale)
        world_width = logical_width / max(0.001, camera_scale) + logical_width
        world_height = logical_height / max(0.001, camera_scale) + logical_height
        root_pos = pygame.Vector2(cam.x / 2, cam.y / 2)
        scale_x = world_width / max(0.001, logical_width)
        scale_y = world_height / max(0.001, logical_height)
        background_width = background.get_width() / source_scale
        background_height = background.get_height() / source_scale
        self._draw_world_surface_clipped(
            screen,
            background,
            root_pos
            + pygame.Vector2(
                self.stage.background_offset.x * scale_x,
                self.stage.background_offset.y * scale_y,
            ),
            cam,
            zoom,
            viewport,
            source_scale=source_scale,
            world_size=(background_width * scale_x, background_height * scale_y),
        )
        self._draw_background_objects(screen, cam, zoom, viewport, root_pos, scale_x, scale_y)
        for layer_index, (frame_rate, loop_from, frames) in enumerate(self.stage.background_layers):
            frame_index = self.stage._background_frame_index(
                self.stage_time_ms,
                frame_rate,
                loop_from,
                len(frames),
            )
            frame = frames[frame_index]
            layer = self.stage.background_layer_surface(layer_index, frame_index)
            layer_width = layer.get_width() / frame.render_scale
            layer_height = layer.get_height() / frame.render_scale
            self._draw_world_surface_clipped(
                screen,
                layer,
                root_pos + pygame.Vector2(frame.offset.x * scale_x, frame.offset.y * scale_y),
                cam,
                zoom,
                viewport,
                source_scale=frame.render_scale,
                world_size=(layer_width * scale_x, layer_height * scale_y),
            )

    def _draw_background_objects(
        self,
        screen: pygame.Surface,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
        root_pos: pygame.Vector2,
        background_scale_x: float,
        background_scale_y: float,
    ) -> None:
        for layer in self.stage.background_object_layers:
            frames = layer["frames"]
            if not frames:
                continue
            frame_index = int(self.stage_time_ms * int(layer["frame_rate"]) / 1000) % len(frames)
            surface = layer["surface"]
            source_scale = float(layer["render_scale"])
            offset = layer["offset"]
            logical_size = layer["logical_size"]
            local_center = offset + logical_size / 2
            transformed_cache: dict[tuple[float, ...], pygame.Surface] = {}
            for matrix_data in frames[frame_index]:
                a, b, c, d, tx, ty = (float(value) for value in matrix_data)
                root_center = pygame.Vector2(
                    a * local_center.x + b * local_center.y + tx,
                    c * local_center.x + d * local_center.y + ty,
                )
                world_center = root_pos + pygame.Vector2(
                    root_center.x * background_scale_x,
                    root_center.y * background_scale_y,
                )
                screen_center = self._world_to_screen(world_center, cam, zoom, viewport)
                if not viewport.inflate(80, 80).collidepoint(screen_center):
                    continue
                scale_x = math.hypot(a, c)
                scale_y = math.hypot(b, d)
                angle = math.degrees(math.atan2(-c, a))
                screen_scale_x = scale_x * background_scale_x * zoom / source_scale
                screen_scale_y = scale_y * background_scale_y * zoom / source_scale
                cache_key = (
                    round(a, 6),
                    round(b, 6),
                    round(c, 6),
                    round(d, 6),
                    round(screen_scale_x, 5),
                    round(screen_scale_y, 5),
                )
                transformed = transformed_cache.get(cache_key)
                if transformed is None:
                    uniform_scale = max(0.001, math.sqrt(screen_scale_x * screen_scale_y))
                    transformed = pygame.transform.rotozoom(surface, angle, uniform_scale)
                    ratio_x = screen_scale_x / uniform_scale
                    ratio_y = screen_scale_y / uniform_scale
                    target_size = (
                        max(1, round(transformed.get_width() * ratio_x)),
                        max(1, round(transformed.get_height() * ratio_y)),
                    )
                    if target_size != transformed.get_size():
                        transformed = self._quality_scale(transformed, target_size)
                    transformed_cache[cache_key] = transformed
                screen.blit(
                    transformed,
                    (
                        round(screen_center.x - transformed.get_width() / 2),
                        round(screen_center.y - transformed.get_height() / 2),
                    ),
                )

    def _draw_world_surface_clipped(
        self,
        screen: pygame.Surface,
        surface: pygame.Surface,
        world_pos: pygame.Vector2,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
        source_scale: float = 1.0,
        world_size: tuple[float, float] | None = None,
    ) -> None:
        source_scale = max(1.0, source_scale)
        logical_source_w = surface.get_width() / source_scale
        logical_source_h = surface.get_height() / source_scale
        world_w, world_h = world_size or (logical_source_w, logical_source_h)
        visible_left = max(world_pos.x, cam.x)
        visible_top = max(world_pos.y, cam.y)
        visible_right = min(world_pos.x + world_w, cam.x + viewport.w / zoom)
        visible_bottom = min(world_pos.y + world_h, cam.y + viewport.h / zoom)
        if visible_right <= visible_left or visible_bottom <= visible_top:
            return

        world_scale_x = world_w / max(0.001, logical_source_w)
        world_scale_y = world_h / max(0.001, logical_source_h)
        source_left = math.floor((visible_left - world_pos.x) / world_scale_x * source_scale)
        source_top = math.floor((visible_top - world_pos.y) / world_scale_y * source_scale)
        source_right = math.ceil((visible_right - world_pos.x) / world_scale_x * source_scale)
        source_bottom = math.ceil((visible_bottom - world_pos.y) / world_scale_y * source_scale)
        source_rect = pygame.Rect(source_left, source_top, source_right - source_left, source_bottom - source_top)
        source_rect = source_rect.clip(surface.get_rect())
        alpha_bounds = self._surface_bounds_cache.get(id(surface))
        if alpha_bounds is None:
            alpha_bounds = surface.get_bounding_rect(min_alpha=1)
            self._surface_bounds_cache[id(surface)] = alpha_bounds
        source_rect = source_rect.clip(alpha_bounds)
        if source_rect.w <= 0 or source_rect.h <= 0:
            return

        target_w = max(1, round(source_rect.w / source_scale * world_scale_x * zoom))
        target_h = max(1, round(source_rect.h / source_scale * world_scale_y * zoom))
        cache_key = (
            "clip",
            id(surface),
            source_rect.x,
            source_rect.y,
            source_rect.w,
            source_rect.h,
            target_w,
            target_h,
        )
        scaled = self._stage_surface_cache.get(cache_key)
        if scaled is None:
            crop = surface.subsurface(source_rect)
            scaled = self._quality_scale(crop, (target_w, target_h))
            if len(self._stage_surface_cache) > 12:
                self._stage_surface_cache.clear()
            self._stage_surface_cache[cache_key] = scaled
        sample_world = pygame.Vector2(
            world_pos.x + source_rect.x / source_scale * world_scale_x,
            world_pos.y + source_rect.y / source_scale * world_scale_y,
        )
        pos = self._world_to_screen(sample_world, cam, zoom, viewport)
        screen.blit(scaled, (round(pos.x), round(pos.y)))

    def _draw_world_surface(
        self,
        screen: pygame.Surface,
        surface: pygame.Surface,
        world_pos: pygame.Vector2,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
        cache: bool = True,
        source_scale: float = 1.0,
    ) -> None:
        scaled = self._scale_stage_surface(
            surface,
            round(surface.get_width() * zoom / max(1.0, source_scale)),
            round(surface.get_height() * zoom / max(1.0, source_scale)),
            cache=cache,
        )
        pos = self._world_to_screen(world_pos, cam, zoom, viewport)
        screen.blit(scaled, (round(pos.x), round(pos.y)))

    def _draw_helicopter(self, screen: pygame.Surface, cam: pygame.Vector2, zoom: float, viewport: pygame.Rect) -> None:
        if not self.stage.helicopter_frames:
            return
        frame_index = int(self.stage_time_ms * self.stage.helicopter_frame_rate / 1000) % len(self.stage.helicopter_frames)
        frame = self.stage.helicopter_frames[frame_index]
        image = self.stage.helicopter_surface(frame_index)
        self._draw_world_surface(
            screen,
            image,
            frame.offset,
            cam,
            zoom,
            viewport,
            cache=True,
            source_scale=frame.render_scale,
        )

    def _draw_dynamic_stage(self, screen: pygame.Surface, cam: pygame.Vector2, zoom: float, viewport: pygame.Rect) -> None:
        if not self.stage.dynamic_frames:
            return
        frame_index = int(self.stage_time_ms * self.stage.dynamic_frame_rate / 1000) % len(self.stage.dynamic_frames)
        frame = self.stage.dynamic_frames[frame_index]
        self._draw_world_surface(
            screen,
            self.stage.dynamic_surface(frame_index),
            frame.offset,
            cam,
            zoom,
            viewport,
            cache=True,
            source_scale=frame.render_scale,
        )

    def _draw_world_rect(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        rect: pygame.Rect,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
        color: tuple[int, int, int],
        label: str,
        width: int = 2,
    ) -> None:
        screen_rect = pygame.Rect(
            round(viewport.x + (rect.x - cam.x) * zoom),
            round(viewport.y + (rect.y - cam.y) * zoom),
            max(1, round(rect.w * zoom)),
            max(1, round(rect.h * zoom)),
        )
        pygame.draw.rect(screen, color, screen_rect, width)
        screen.blit(font.render(label, True, color), (screen_rect.x, screen_rect.y - 14))

    def _stage_scene(
        self,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
    ) -> pygame.Surface:
        helicopter_frame = (
            int(self.stage_time_ms * self.stage.helicopter_frame_rate / 1000)
            % len(self.stage.helicopter_frames)
            if self.stage.helicopter_frames
            else 0
        )
        dynamic_frame = (
            int(self.stage_time_ms * self.stage.dynamic_frame_rate / 1000)
            % len(self.stage.dynamic_frames)
            if self.stage.dynamic_frames
            else 0
        )
        camera_key = (
            viewport.size,
            round(cam.x * zoom, 3),
            round(cam.y * zoom, 3),
            round(zoom, 6),
        )
        background_key = self.stage.background_frame_key(self.stage_time_ms)
        key = camera_key + (helicopter_frame, dynamic_frame) + background_key
        if self._stage_scene_cache_key == key and self._stage_scene_cache is not None:
            return self._stage_scene_cache

        local_viewport = pygame.Rect(0, 0, viewport.w, viewport.h)
        backdrop_key = camera_key + background_key
        if self._stage_backdrop_cache_key != backdrop_key or self._stage_backdrop_cache is None:
            backdrop = pygame.Surface(viewport.size).convert()
            self._draw_sky_backdrop(backdrop, local_viewport)
            self._draw_background(backdrop, cam, zoom, local_viewport)
            self._stage_backdrop_cache_key = backdrop_key
            self._stage_backdrop_cache = backdrop
        if self._stage_foreground_cache_key != camera_key or self._stage_foreground_cache is None:
            foreground = pygame.Surface(viewport.size, pygame.SRCALPHA)
            if self.stage.foreground is not None:
                self._draw_world_surface_clipped(
                    foreground,
                    self.stage.foreground,
                    self.stage.foreground_offset,
                    cam,
                    zoom,
                    local_viewport,
                    source_scale=self.stage.foreground_scale,
                )
            self._stage_foreground_cache_key = camera_key
            self._stage_foreground_cache = foreground

        scene = self._stage_backdrop_cache.copy()
        self._draw_helicopter(scene, cam, zoom, local_viewport)
        if not self.stage.dynamic_above_foreground:
            self._draw_dynamic_stage(scene, cam, zoom, local_viewport)
        scene.blit(self._stage_foreground_cache, (0, 0))
        if self.stage.dynamic_above_foreground:
            self._draw_dynamic_stage(scene, cam, zoom, local_viewport)
        self._stage_scene_cache_key = key
        self._stage_scene_cache = scene
        return scene

    def _draw(self, screen: pygame.Surface, font: pygame.font.Font) -> None:
        w, h = screen.get_size()
        panel_x = max(0, w - PANEL_WIDTH) if self.show_debug else w
        viewport = MainMenu._screen_rect((panel_x, h))
        alpha = self._render_alpha()
        cam, zoom = self._camera(viewport, alpha)

        screen.fill((18, 20, 24))
        screen.blit(self._stage_scene(cam, zoom, viewport), viewport.topleft)

        if self.show_debug:
            self._draw_world_rect(screen, font, self.stage.bounds_cam, cam, zoom, viewport, (120, 180, 255), "BoundsCam")
            self._draw_world_rect(screen, font, self.stage.bounds, cam, zoom, viewport, (255, 100, 100), "Bounds")
            self._draw_world_rect(screen, font, self.stage.view_bounds, cam, zoom, viewport, (170, 170, 170), "View")
            for platform in self.stage.platforms:
                color = (120, 255, 140) if platform.moving else (255, 220, 80)
                self._draw_world_rect(screen, font, platform.rect, cam, zoom, viewport, color, platform.name)

        for fighter in sorted(self.fighters, key=lambda item: item.draw_depth):
            self._draw_fighter(screen, font, fighter, cam, zoom, viewport, alpha)

        # PunchDamage is attached inside the attack clip, above its fighter but
        # below the root OSD. Root effects are attached after OSD initialization.
        self._draw_hit_effects(screen, cam, zoom, viewport, root_layer=False)
        self._draw_osd(screen, font, viewport)
        self._draw_far_indicators(screen, font, cam, zoom, viewport, alpha)
        self._draw_spawn_effects(screen, cam, zoom, viewport)
        self._draw_active_spawn_clouds(screen, cam, zoom, viewport)
        self._draw_death_effects(screen, cam, zoom, viewport)
        self._draw_explosions(screen, cam, zoom, viewport)
        self._draw_items(screen, cam, zoom, viewport)

        for bullet in self.bullets:
            bullet_scale = zoom / max(1.0, bullet.source_scale)
            bullet_source = pygame.transform.flip(bullet.image, True, False) if bullet.xinc < 0 else bullet.image
            bullet_img = self._quality_scale(
                bullet_source,
                (
                    max(1, round(bullet_source.get_width() * bullet_scale * 0.5)),
                    max(1, round(bullet_source.get_height() * bullet_scale * 0.5)),
                ),
            )
            bullet_pos = self._world_to_screen(
                bullet.render_pos(alpha) + bullet.draw_center_offset(),
                cam,
                zoom,
                viewport,
            )
            screen.blit(
                bullet_img,
                (round(bullet_pos.x - bullet_img.get_width() / 2), round(bullet_pos.y - bullet_img.get_height() / 2)),
            )
        for rocket in self.rockets:
            rocket_scale = zoom / max(1.0, rocket.source_scale)
            rocket_source = pygame.transform.flip(rocket.image, True, False) if rocket.mirrored else rocket.image
            rocket_img = self._quality_scale(
                rocket_source,
                (
                    max(1, round(rocket_source.get_width() * rocket_scale)),
                    max(1, round(rocket_source.get_height() * rocket_scale)),
                ),
            )
            rocket_img = pygame.transform.rotate(rocket_img, -rocket.rotation)
            rocket_pos = self._world_to_screen(
                rocket.render_pos(alpha) + rocket.draw_center_offset(),
                cam,
                zoom,
                viewport,
            )
            screen.blit(
                rocket_img,
                (round(rocket_pos.x - rocket_img.get_width() / 2), round(rocket_pos.y - rocket_img.get_height() / 2)),
            )
        for projectile in self.special_projectiles:
            asset = projectile.frame
            if asset is None:
                continue
            source = pygame.transform.flip(asset.image, True, False) if projectile.facing < 0 else asset.image
            projectile_scale = zoom / asset.render_scale * projectile.display_scale
            image = self._quality_scale(
                source,
                (
                    max(1, round(source.get_width() * projectile_scale)),
                    max(1, round(source.get_height() * projectile_scale)),
                ),
            )
            if projectile.rotation:
                image = pygame.transform.rotate(image, -projectile.rotation)
            projectile_pos = self._world_to_screen(
                projectile.render_pos(alpha) + projectile.draw_center_offset(),
                cam,
                zoom,
                viewport,
            )
            screen.blit(
                image,
                (
                    round(projectile_pos.x - image.get_width() / 2),
                    round(projectile_pos.y - image.get_height() / 2),
                ),
            )

        self._draw_hit_effects(screen, cam, zoom, viewport, root_layer=True)
        for fighter in self.fighters:
            self._draw_out_of_camera_indicator(screen, font, fighter, cam, zoom, viewport, alpha)

        if self.match_state == "game_set":
            self._draw_match_overlay(screen, font, viewport)
        elif self.match_state == "loading":
            self._draw_match_loading(screen, viewport)
        elif self.match_state == "countdown" and self.ready_text:
            self._draw_ready_overlay(screen, font, viewport)

        if not self.show_debug:
            return

        pygame.draw.rect(screen, (32, 35, 42), (panel_x, 0, PANEL_WIDTH, h))
        winner = self.match_winner.name if self.match_winner is not None else "-"
        lines = [
            "Playable Runtime",
            "P1 A/D move  W jump",
            "P1 J punch  K special  Shift shield",
            "P2 Arrows move/jump",
            "P2 KP0 punch  KP1 special  KP2 shield",
            "R: reset  F1: debug",
            "",
            "Two-Fighter Scene",
            "P1 and P2 playable",
            f"Match: {self.match_state}",
            f"Winner: {winner}",
            f"Tick: {TICK_MS} ms",
            f"State: {self.player.state}",
            f"Frame: {self.player.current_label}",
            f"Anim: {self.player.animation_frame}",
            f"Attack: {self.player.current_attack or '-'}",
            f"Bullets: {len(self.bullets)}",
            f"Specials: {len(self.special_projectiles)}",
            f"Effects: {len(self.death_effects)}",
            "",
            "Stocks",
            *[
                f"{fighter.name}: {fighter.lives} stock  {fighter.damage_amnt}%"
                + ("  OUT" if fighter.dead else "")
                for fighter in self.fighters
            ],
            "",
            f"Pos: {self.player.pos.x:.2f}, {self.player.pos.y:.2f}",
            f"xInc/yInc: {self.player.xinc:.2f}, {self.player.yinc:.2f}",
            f"jumpstate: {self.player.jumpstate}",
            f"onGround: {self.player.on_ground}",
            f"OutCam: {self.player.out_of_camera}",
            f"Ground: {self.player.ground_platform.name if self.player.ground_platform else '-'}",
            f"Collision: {self.player.last_collision}",
            f"Last death: {self.player.last_death_type or '-'}",
        ]
        for i, line in enumerate(lines):
            color = (255, 220, 160) if line in {"Playable Runtime", "Two-Fighter Scene", "Stocks"} else (230, 230, 230)
            screen.blit(font.render(line, True, color), (panel_x + 18, 18 + i * 22))

    def _draw_match_overlay(self, screen: pygame.Surface, font: pygame.font.Font, viewport: pygame.Rect) -> None:
        self._draw_big_text(screen, "GAME SET!", viewport)

    def _draw_match_loading(self, screen: pygame.Surface, viewport: pygame.Rect) -> None:
        if not self.match_loading_frames:
            screen.fill((0, 0, 0), viewport)
            return
        frame_index = int(self.match_loading_elapsed_ms * ANIMATION_FPS / 1000) % len(self.match_loading_frames)
        image = self._quality_scale(self.match_loading_frames[frame_index], viewport.size)
        screen.blit(image, viewport)

    def _draw_ready_overlay(self, screen: pygame.Surface, font: pygame.font.Font, viewport: pygame.Rect) -> None:
        self._draw_big_text(screen, self.ready_text, viewport)

    def _draw_big_text(self, screen: pygame.Surface, value: str, viewport: pygame.Rect) -> None:
        layout = self.manifest.get("ui", {}).get("layout", {})
        reference = layout.get("reference_size", {"w": 600, "h": 400})
        base_scale = min(viewport.w / float(reference.get("w", 600)), viewport.h / float(reference.get("h", 400)))
        data = layout.get("big_text", {"x": 2, "y": 152, "w": 600, "h": 100, "font_size": 80})
        big_font = self._ui_font(
            str(data.get("font", "Futura Md BT")),
            max(12, round(float(data.get("font_size", 80)) * base_scale)),
            True,
        )
        text = big_font.render(value, True, (255, 204, 0))
        shadow = big_font.render(value, True, (0, 0, 0))
        field_x = viewport.x + float(data.get("x", 2)) * base_scale
        field_w = float(data.get("w", 600)) * base_scale
        x = field_x + (field_w - text.get_width()) / 2
        y = viewport.y + float(data.get("y", 152)) * base_scale
        glow = max(2, round(4 * base_scale))
        for dx, dy in ((-glow, 0), (glow, 0), (0, -glow), (0, glow)):
            screen.blit(shadow, (round(x + dx), round(y + dy)))
        screen.blit(text, (round(x), round(y)))

    def _draw_spawn_effects(
        self,
        screen: pygame.Surface,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
    ) -> None:
        for effect in self.spawn_effects:
            if not effect.frames:
                continue
            asset = effect.frames[effect.frame_index]
            image = asset.image
            asset_scale = zoom / asset.render_scale
            scaled = self._quality_scale(
                image,
                (
                    max(1, round(image.get_width() * asset_scale)),
                    max(1, round(image.get_height() * asset_scale)),
                ),
            )
            pos = self._world_to_screen(effect.pos, cam, zoom, viewport)
            draw_x = round(pos.x + asset.offset.x * zoom)
            draw_y = round(pos.y + asset.offset.y * zoom)
            screen.blit(scaled, (draw_x, draw_y))

    def _draw_active_spawn_clouds(
        self,
        screen: pygame.Surface,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
    ) -> None:
        for fighter in self.fighters:
            if fighter.dead or not fighter.intro_visible or fighter.state != "spawn":
                continue
            frames = self.spawn_frames.get(fighter.spawn_effect_kind, [])
            if not frames:
                continue
            frame_index = min(len(frames) - 1, int(fighter.spawn_age_ms * ANIMATION_FPS / 1000))
            asset = frames[frame_index]
            image = asset.image
            asset_scale = zoom / asset.render_scale
            scaled = self._quality_scale(
                image,
                (
                    max(1, round(image.get_width() * asset_scale)),
                    max(1, round(image.get_height() * asset_scale)),
                ),
            )
            pos = self._world_to_screen(fighter.render_pos(1.0), cam, zoom, viewport)
            screen.blit(
                scaled,
                (
                    round(pos.x + asset.offset.x * zoom),
                    round(pos.y + asset.offset.y * zoom),
                ),
            )

    def _draw_out_of_camera_indicator(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        fighter: PeachFighter,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
        alpha: float,
    ) -> None:
        if fighter.dead or not fighter.intro_visible or not fighter.out_of_camera:
            return
        if not fighter.out_of_camera_proxy_visible(self.stage, self.camera_view, self.camera_target_view):
            return
        proxy = fighter.out_of_camera_proxy_pos(self.stage, self.camera_view)
        render_offset_ms = alpha * TICK_MS
        image = fighter.current_image(render_offset_ms)
        source_scale = fighter.current_render_scale(render_offset_ms)
        scaled = self._quality_scale(
            image,
            (
                max(1, round(image.get_width() * zoom / source_scale)),
                max(1, round(image.get_height() * zoom / source_scale)),
            ),
        )
        proxy_screen = self._world_to_screen(proxy, cam, zoom, viewport)
        indicator_asset = self._team_frame(self.pos_indicator_frames, fighter)
        if indicator_asset is None:
            return
        angle = math.degrees(math.atan2(proxy.y - fighter.pos.y, proxy.x - fighter.pos.x))
        indicator = indicator_asset.image
        indicator_scale = zoom / indicator_asset.render_scale
        indicator_scaled = self._quality_scale(
            indicator,
            (
                max(1, round(indicator.get_width() * indicator_scale)),
                max(1, round(indicator.get_height() * indicator_scale)),
            ),
        )
        indicator_scaled = pygame.transform.rotate(indicator_scaled, -angle)
        top_left = pygame.Vector2(indicator_asset.offset.x * zoom, indicator_asset.offset.y * zoom)
        corners = [
            top_left,
            top_left + pygame.Vector2(indicator.get_width() * indicator_scale, 0),
            top_left + pygame.Vector2(0, indicator.get_height() * indicator_scale),
            top_left
            + pygame.Vector2(
                indicator.get_width() * indicator_scale,
                indicator.get_height() * indicator_scale,
            ),
        ]
        rotated_corners = [corner.rotate(angle) for corner in corners]
        screen.blit(
            indicator_scaled,
            (
                round(proxy_screen.x + min(corner.x for corner in rotated_corners)),
                round(proxy_screen.y + min(corner.y for corner in rotated_corners)),
            ),
        )
        # OSD.OutBounds attaches PosIndicator first and PlayerVis afterwards,
        # so the proxy fighter is above the directional ring in Flash depth.
        offset = fighter.current_draw_offset(image, render_offset_ms)
        screen.blit(
            scaled,
            (round(proxy_screen.x + offset.x * zoom), round(proxy_screen.y + offset.y * zoom)),
        )

    def _draw_death_effects(
        self,
        screen: pygame.Surface,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
    ) -> None:
        rotations = {"lef": 90, "rig": -90, "bot": 0, "top": 180}
        for effect in self.death_effects:
            center = self._world_to_screen(effect.pos, cam, zoom, viewport)
            if effect.frames:
                asset = effect.frames[effect.frame_index]
                image = asset.image
                asset_scale = zoom / asset.render_scale
                scaled = self._quality_scale(
                    image,
                    (
                        max(1, round(image.get_width() * asset_scale)),
                        max(1, round(image.get_height() * asset_scale)),
                    ),
                )
                rotation = rotations.get(effect.death_type, 0)
                rotated = pygame.transform.rotate(scaled, -rotation)
                top_left = pygame.Vector2(asset.offset.x * zoom, asset.offset.y * zoom)
                corners = [
                    top_left,
                    top_left + pygame.Vector2(scaled.get_width(), 0),
                    top_left + pygame.Vector2(0, scaled.get_height()),
                    top_left + pygame.Vector2(scaled.get_width(), scaled.get_height()),
                ]
                rotated_corners = [corner.rotate(rotation) for corner in corners]
                draw_offset = pygame.Vector2(
                    min(corner.x for corner in rotated_corners),
                    min(corner.y for corner in rotated_corners),
                )
                screen.blit(
                    rotated,
                    (round(center.x + draw_offset.x), round(center.y + draw_offset.y)),
                )
                continue
            t = max(0.0, min(1.0, effect.age_ms / max(1, effect.life_ms)))
            radius = max(6, round((18 + 55 * t) * zoom))
            alpha = max(0, round(220 * (1 - t)))
            fallback = pygame.Surface((radius * 2 + 8, radius * 2 + 8), pygame.SRCALPHA)
            fallback_center = pygame.Vector2(fallback.get_width() / 2, fallback.get_height() / 2)
            pygame.draw.circle(fallback, (255, 245, 180, alpha), fallback_center, radius, max(1, round(4 * zoom)))
            screen.blit(
                fallback,
                (round(center.x - fallback_center.x), round(center.y - fallback_center.y)),
            )

    def _draw_hit_effects(
        self,
        screen: pygame.Surface,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
        root_layer: bool | None = None,
    ) -> None:
        for effect in self.hit_effects:
            if root_layer is not None and effect.root_layer != root_layer:
                continue
            self._draw_registered_asset(
                screen,
                effect.frames[effect.frame_index],
                effect.pos,
                cam,
                zoom,
                viewport,
                effect.scale,
                effect.scale,
                effect.rotation,
            )

    def _draw_registered_asset(
        self,
        screen: pygame.Surface,
        asset: SpriteAssetFrame,
        world_origin: pygame.Vector2,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        rotation: float = 0.0,
    ) -> None:
        image = asset.image
        source_scale = max(1.0, asset.render_scale)
        logical_width = image.get_width() / source_scale
        logical_height = image.get_height() / source_scale
        scaled = self._quality_scale(
            image,
            (
                max(1, round(logical_width * scale_x * zoom)),
                max(1, round(logical_height * scale_y * zoom)),
            ),
        )
        local_center = pygame.Vector2(
            (asset.offset.x + logical_width / 2) * scale_x,
            (asset.offset.y + logical_height / 2) * scale_y,
        )
        if rotation:
            scaled = pygame.transform.rotate(scaled, -rotation)
            local_center = local_center.rotate(rotation)
        center = self._world_to_screen(world_origin + local_center, cam, zoom, viewport)
        screen.blit(
            scaled,
            (round(center.x - scaled.get_width() / 2), round(center.y - scaled.get_height() / 2)),
        )

    def _draw_items(
        self,
        screen: pygame.Surface,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
    ) -> None:
        for item in self.items:
            if not item.visible:
                continue
            asset = self._item_draw_frame(item)
            if asset is None:
                continue
            image = asset.image
            source_scale = max(1.0, asset.render_scale)
            scaled = self._quality_scale(
                image,
                (
                    max(1, round(image.get_width() * zoom / source_scale)),
                    max(1, round(image.get_height() * zoom / source_scale)),
                ),
            )
            logical_center = asset.offset + pygame.Vector2(
                image.get_width() / source_scale / 2,
                image.get_height() / source_scale / 2,
            )
            if item.rotation:
                scaled = pygame.transform.rotate(scaled, -item.rotation)
                logical_center = logical_center.rotate(item.rotation)
            center = self._world_to_screen(item.pos + logical_center, cam, zoom, viewport)
            screen.blit(
                scaled,
                (round(center.x - scaled.get_width() / 2), round(center.y - scaled.get_height() / 2)),
            )
            if item.state == 1 and self.item_indicator_frames:
                indicator = self.item_indicator_frames[
                    int(item.age_ms * ANIMATION_FPS / 1000) % len(self.item_indicator_frames)
                ]
                self._draw_registered_asset(
                    screen,
                    indicator,
                    item.pos,
                    cam,
                    zoom,
                    viewport,
                )

    def _first_visible_frame(self, frames: list[pygame.Surface]) -> pygame.Surface | None:
        for frame in frames:
            if frame.get_bounding_rect().w > 0 and frame.get_bounding_rect().h > 0:
                return frame
        return frames[0] if frames else None

    def _item_draw_frame(self, item: StageItem) -> SpriteAssetFrame | None:
        return item.display_frame()

    def _draw_explosions(
        self,
        screen: pygame.Surface,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
    ) -> None:
        for explosion in self.explosions:
            if explosion.matter_frames:
                matter = explosion.matter_frames[explosion.matter_frame_index]
                base = explosion.matter_frames[0]
                base_width = base.image.get_width() / max(1.0, base.render_scale)
                base_height = base.image.get_height() / max(1.0, base.render_scale)
                for offset in explosion.matter_offsets:
                    self._draw_registered_asset(
                        screen,
                        matter,
                        explosion.pos + offset,
                        cam,
                        zoom,
                        viewport,
                        explosion.size / max(0.001, base_width),
                        explosion.size / max(0.001, base_height),
                    )
            if explosion.wave_frames:
                wave = explosion.wave_frames[explosion.wave_frame_index]
                base = explosion.wave_frames[0]
                base_width = base.image.get_width() / max(1.0, base.render_scale)
                base_height = base.image.get_height() / max(1.0, base.render_scale)
                self._draw_registered_asset(
                    screen,
                    wave,
                    explosion.pos,
                    cam,
                    zoom,
                    viewport,
                    explosion.size * 2 / max(0.001, base_width),
                    explosion.size * 2 / max(0.001, base_height),
                )
            if explosion.frames:
                star = explosion.frames[explosion.frame_index]
                base = explosion.frames[0]
                base_width = base.image.get_width() / max(1.0, base.render_scale)
                base_height = base.image.get_height() / max(1.0, base.render_scale)
                self._draw_registered_asset(
                    screen,
                    star,
                    explosion.pos,
                    cam,
                    zoom,
                    viewport,
                    explosion.size * 50 / max(0.001, base_width),
                    explosion.size * 50 / max(0.001, base_height),
                )

    def _draw_far_indicators(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
        alpha: float,
    ) -> None:
        base_scale = min(viewport.w / 600, viewport.h / 400)
        camera_scale = zoom / max(0.001, base_scale)
        threshold = float(self.manifest.get("ui", {}).get("layout", {}).get("far_indicator_scale_threshold", 1.5))
        if camera_scale >= threshold:
            return
        for index, fighter in enumerate(self.fighters, start=1):
            if fighter.dead or not fighter.intro_visible:
                continue
            asset = self._team_frame(self.far_indicator_frames, fighter)
            if asset is None:
                continue
            render_pos = fighter.render_pos(alpha)
            pos = self._world_to_screen(render_pos, cam, zoom, viewport)
            arrow = asset.image.copy()
            # FarIndicator embeds the default dynamic-field value "CP" above
            # the arrow. OSD.AddDamageGraphic replaces it with P1/P2 at run
            # time, so clear the complete 17-logical-pixel text raster first.
            text_height = min(arrow.get_height(), round(17 * asset.render_scale))
            arrow.fill((0, 0, 0, 0), pygame.Rect(0, 0, arrow.get_width(), text_height))
            asset_scale = base_scale / asset.render_scale
            scaled = self._quality_scale(
                arrow,
                (
                    max(1, round(arrow.get_width() * asset_scale)),
                    max(1, round(arrow.get_height() * asset_scale)),
                ),
            )
            screen.blit(
                scaled,
                (
                    round(pos.x + asset.offset.x * base_scale),
                    round(pos.y + asset.offset.y * base_scale),
                ),
            )
            indicator_layout = self.manifest.get("ui", {}).get("layout", {}).get(
                "far_indicator",
                {"font": "Futura Md BT", "font_size": 20, "text_center_x": 1, "text_visible_top": -83},
            )
            team_colors = [(255, 0, 0), (51, 102, 255), (102, 204, 0), (255, 204, 0)]
            label_font = self._ui_font(
                str(indicator_layout.get("font", "Futura Md BT")),
                max(10, round(float(indicator_layout.get("font_size", 20)) * base_scale)),
                True,
            )
            label_text = "CP" if index - 1 in self.ai_controllers else f"P{index}"
            label = label_font.render(label_text, True, team_colors[fighter.team_index % len(team_colors)])
            glyph = label.get_bounding_rect(min_alpha=1)
            center_x = pos.x + float(indicator_layout.get("text_center_x", 1)) * base_scale
            visible_top = pos.y + float(indicator_layout.get("text_visible_top", -83)) * base_scale
            screen.blit(
                label,
                (
                    round(center_x - glyph.centerx),
                    round(visible_top - glyph.top),
                ),
            )

    def _team_frame(self, frames: list[SpriteAssetFrame], fighter: PeachFighter) -> SpriteAssetFrame | None:
        if not frames:
            return None
        return frames[min(fighter.team_index, len(frames) - 1)]

    def _draw_osd(self, screen: pygame.Surface, font: pygame.font.Font, viewport: pygame.Rect) -> None:
        layout = self.manifest.get("ui", {}).get("layout", {})
        reference = layout.get("reference_size", {"w": 600, "h": 400})
        reference_w = float(reference.get("w", 600))
        reference_h = float(reference.get("h", 400))
        base_scale = min(viewport.w / reference_w, viewport.h / reference_h)
        origin = layout.get("damage_origin", {"x": 60, "y": 340})
        spacing = float(layout.get("damage_spacing", 150))
        limit_mode = self.manifest.get("match", {}).get("limit_mode", "stock")
        for index, fighter in enumerate(self.fighters[:4]):
            x = viewport.x + (float(origin["x"]) + spacing * index) * base_scale
            y = viewport.y + float(origin["y"]) * base_scale
            if self.osd_bigicon_frames:
                frame_index = self.osd_bigicon_by_character.get(fighter.character_name, 2)
                asset = self.osd_bigicon_frames[min(max(0, frame_index), len(self.osd_bigicon_frames) - 1)]
                tinted = self._team_tinted_osd_icon(asset.image, fighter.team_index)
                asset_scale = base_scale / asset.render_scale
                scaled = self._quality_scale(
                    tinted,
                    (
                        max(1, round(tinted.get_width() * asset_scale)),
                        max(1, round(tinted.get_height() * asset_scale)),
                    ),
                )
                screen.blit(
                    scaled,
                    (
                        round(x + (-2.15 + asset.offset.x) * base_scale),
                        round(y + (-0.6 + asset.offset.y) * base_scale),
                    ),
                )
            life_frames = self.osd_life_frames_by_character.get(fighter.character_name, self.osd_life_frames)
            if limit_mode == "stock" and life_frames and fighter.lives > 0:
                asset = life_frames[min(fighter.color_frame - 1, len(life_frames) - 1)]
                life_image = asset.image
                asset_scale = base_scale / asset.render_scale
                life_scaled = self._quality_scale(
                    life_image,
                    (
                        max(1, round(life_image.get_width() * asset_scale)),
                        max(1, round(life_image.get_height() * asset_scale)),
                    ),
                )
                counter_data = layout.get(
                    "life_counter",
                    {
                        "counter_x": -38.2,
                        "counter_y": -48.75,
                        "more_icon_x": 18.25,
                        "more_text_x": 33.5,
                        "more_text_y": -2.85,
                        "font_size": 14,
                    },
                )
                counter_x = x + float(counter_data.get("counter_x", -38.2)) * base_scale
                counter_y = y + float(counter_data.get("counter_y", -48.75)) * base_scale
                if fighter.lives < 6:
                    icon_x = counter_x + asset.offset.x * base_scale
                    icon_y = counter_y + asset.offset.y * base_scale
                    for life in range(fighter.lives):
                        screen.blit(life_scaled, (round(icon_x + life * 17 * base_scale), round(icon_y)))
                else:
                    icon_x = counter_x + (float(counter_data.get("more_icon_x", 18.25)) + asset.offset.x) * base_scale
                    icon_y = counter_y + asset.offset.y * base_scale
                    screen.blit(life_scaled, (round(icon_x), round(icon_y)))
                    life_font = self._ui_font(
                        "Futura Md BT",
                        max(8, round(float(counter_data.get("font_size", 14)) * base_scale)),
                        True,
                    )
                    value = life_font.render(f"x{fighter.lives}", True, (255, 255, 255))
                    shadow = life_font.render(f"x{fighter.lives}", True, (0, 0, 0))
                    text_x = counter_x + float(counter_data.get("more_text_x", 33.5)) * base_scale
                    text_y = counter_y + float(counter_data.get("more_text_y", -2.85)) * base_scale
                    spread = max(1, round(base_scale))
                    for dx, dy in ((-spread, 0), (spread, 0), (0, -spread), (0, spread)):
                        screen.blit(shadow, (round(text_x + dx), round(text_y + dy)))
                    screen.blit(value, (round(text_x), round(text_y)))
            self._draw_osd_score_event(screen, fighter, x, y, base_scale, layout)
            self._draw_osd_damage(screen, fighter, x, y, base_scale, layout, limit_mode)
        if limit_mode == "time":
            self._draw_match_timer(screen, viewport, base_scale, layout)

    def _draw_osd_damage(
        self,
        screen: pygame.Surface,
        fighter: PeachFighter,
        x: float,
        y: float,
        base_scale: float,
        layout: dict[str, object],
        limit_mode: str,
    ) -> None:
        if fighter.dead or (limit_mode == "stock" and fighter.lives <= 0):
            return
        pulse = layout.get(
            "damage_pulse",
            [
                {"scale": 1.0, "x": 2.25, "y": -2.8, "brightness": 255},
                {"scale": 1.4526215, "x": 4.25, "y": -4.8, "brightness": 51},
                {"scale": 1.311203, "x": 0.75, "y": -4.3, "brightness": 102},
                {"scale": 1.1697388, "x": -2.75, "y": -3.8, "brightness": 153},
                {"scale": 1.1343842, "x": -0.75, "y": -1.8, "brightness": 179},
                {"scale": 1.0990143, "x": 1.25, "y": 0.2, "brightness": 204},
                {"scale": 1.04953, "x": 3.25, "y": -1.8, "brightness": 230},
                {"scale": 1.0, "x": 5.25, "y": -3.8, "brightness": 255},
            ],
        )
        if fighter.osd_damage_age_ms < 7 * (1000 / ANIMATION_FPS):
            pulse_index = min(7, 1 + int(fighter.osd_damage_age_ms * ANIMATION_FPS / 1000))
        else:
            pulse_index = 0
        state = pulse[pulse_index]
        text_scale = float(state["scale"])
        damage_x = float(state["x"])
        damage_y = float(state["y"])
        brightness = int(state["brightness"])
        font_data = layout.get("damage_font", {"name": "Arial", "size": 23, "bold": True})
        damage_font = self._ui_font(
            str(font_data.get("name", "Arial")),
            max(8, round(float(font_data.get("size", 23)) * text_scale * base_scale)),
            bool(font_data.get("bold", True)),
        )
        value = f"{fighter.damage_amnt}%"
        color = (brightness, brightness, brightness)
        text = damage_font.render(value, True, color)
        shadow = damage_font.render(value, True, (0, 0, 0))
        field = layout.get("damage_field", {"right": 35.4, "top": -14.85})
        field_right = x + (damage_x + text_scale * float(field.get("right", 35.4))) * base_scale
        text_x = field_right - text.get_width()
        text_y = y + (damage_y + text_scale * float(field.get("top", -14.85))) * base_scale
        glow_data = layout.get("damage_glow", {"blur_x": 2.0})
        glow = max(1, round(float(glow_data.get("blur_x", 2.0)) * 0.5 * base_scale))
        for dx, dy in (
            (-glow, -glow),
            (0, -glow),
            (glow, -glow),
            (-glow, 0),
            (glow, 0),
            (-glow, glow),
            (0, glow),
            (glow, glow),
        ):
            screen.blit(shadow, (round(text_x + dx), round(text_y + dy)))
        screen.blit(text, (round(text_x), round(text_y)))

    def _ui_font(self, name: str, size: int, bold: bool) -> pygame.font.Font:
        key = (name.lower(), max(1, int(size)), bool(bold))
        cache = getattr(self, "_ui_font_cache", None)
        if cache is None:
            cache = {}
            self._ui_font_cache = cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        if name.lower() in {"futura md bt", "futura lt bt"}:
            embedded_path = ROOT / "assets/fonts/2_Futura Md BT.ttf"
        else:
            embedded_path = None
        if embedded_path is not None and embedded_path.is_file():
            cached = pygame.font.Font(str(embedded_path), key[1])
            cached.set_bold(key[2])
        else:
            path = pygame.font.match_font(name, bold=bold)
            cached = pygame.font.Font(path, key[1]) if path else pygame.font.SysFont(name, key[1], bold=bold)
        cache[key] = cached
        return cached

    def _draw_osd_score_event(
        self,
        screen: pygame.Surface,
        fighter: PeachFighter,
        x: float,
        y: float,
        base_scale: float,
        layout: dict[str, object],
    ) -> None:
        if not fighter.osd_score_event or not self.osd_score_upper_frames:
            return
        start_frame = 2 if fighter.osd_score_event == "plus" else 35
        frame_no = start_frame + int(fighter.osd_score_age_ms * ANIMATION_FPS / 1000)
        frame_no = min(start_frame + 31, frame_no)
        asset = self.osd_score_upper_frames[min(frame_no - 1, len(self.osd_score_upper_frames) - 1)]
        asset_scale = base_scale / asset.render_scale
        image = self._quality_scale(
            asset.image,
            (
                max(1, round(asset.image.get_width() * asset_scale)),
                max(1, round(asset.image.get_height() * asset_scale)),
            ),
        )
        position = layout.get("score_upper", {"x": 8.35, "y": -43.2})
        screen.blit(
            image,
            (
                round(x + (float(position.get("x", 8.35)) + asset.offset.x) * base_scale),
                round(y + (float(position.get("y", -43.2)) + asset.offset.y) * base_scale),
            ),
        )

    def _draw_match_timer(
        self,
        screen: pygame.Surface,
        viewport: pygame.Rect,
        base_scale: float,
        layout: dict[str, object],
    ) -> None:
        timer_data = layout.get("timer", {"x": 306.8, "y": 10, "font_size": 31})
        seconds = max(0, math.ceil(self.match_time_remaining_ms / 1000))
        value = f"{seconds // 60:02d}:{seconds % 60:02d}"
        color = (204, 0, 0) if seconds < 30 else (255, 255, 255)
        timer_font = self._ui_font(
            "Futura Md BT",
            max(8, round(float(timer_data.get("font_size", 31)) * base_scale)),
            True,
        )
        image = timer_font.render(value, True, color)
        shadow = timer_font.render(value, True, (0, 0, 0))
        x = viewport.x + float(timer_data.get("x", 306.8)) * base_scale - image.get_width() / 2
        y = viewport.y + float(timer_data.get("y", 10)) * base_scale
        spread = max(1, round(2 * base_scale))
        for dx, dy in ((-spread, 0), (spread, 0), (0, -spread), (0, spread)):
            screen.blit(shadow, (round(x + dx), round(y + dy)))
        screen.blit(image, (round(x), round(y)))

    def _team_tinted_osd_icon(self, image: pygame.Surface, index: int) -> pygame.Surface:
        tinted = image.copy()
        if index % 4 == 0:
            tinted.fill((255, 0, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
            tinted.fill((255, 0, 0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        elif index % 4 == 1:
            tinted.fill((0, 0, 255, 255), special_flags=pygame.BLEND_RGBA_MULT)
        elif index % 4 == 2:
            tinted.fill((0, 255, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
        else:
            tinted.fill((255, 128, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
            tinted.fill((40, 40, 0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        # MovieClip._alpha multiplies existing per-pixel alpha. Surface.set_alpha
        # would instead give the transparent crop rectangle a global opacity.
        tinted.fill((255, 255, 255, 128), special_flags=pygame.BLEND_RGBA_MULT)
        return tinted

    def _draw_fighter(
        self,
        screen: pygame.Surface,
        font: pygame.font.Font,
        fighter: PeachFighter,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
        alpha: float,
    ) -> None:
        if fighter.dead or not fighter.intro_visible:
            return
        if fighter.state == "spawn" and not fighter.spawn_fighter_visible:
            return
        render_offset_ms = alpha * TICK_MS
        image = fighter.current_image(render_offset_ms)
        color_offset = 0
        if fighter.state == "spawn" and fighter.spawn_white_offset > 0:
            color_offset = round(fighter.spawn_white_offset)
        elif fighter.state == "ko" or fighter.blinking:
            color_offset = round(math.cos(fighter.blinky_cos) * 100)
        if color_offset != 0:
            image = image.copy()
            amount = min(255, abs(color_offset))
            blend = pygame.BLEND_RGBA_ADD if color_offset > 0 else pygame.BLEND_RGBA_SUB
            image.fill((amount, amount, amount, 0), special_flags=blend)
        if fighter.state == "spawn":
            image = image.copy()
            image.set_alpha(max(0, min(255, round(fighter.spawn_visual_alpha / 100 * 255))))
        source_scale = fighter.current_render_scale(render_offset_ms)
        scaled = self._quality_scale(
            image,
            (
                max(1, round(image.get_width() * zoom / source_scale)),
                max(1, round(image.get_height() * zoom / source_scale)),
            ),
        )
        render_pos = fighter.render_pos(alpha)
        foot = self._world_to_screen(render_pos, cam, zoom, viewport)
        offset = fighter.current_draw_offset(image, render_offset_ms)
        draw_pos = (round(foot.x + offset.x * zoom), round(foot.y + offset.y * zoom))
        screen.blit(scaled, draw_pos)
        self._draw_shield(screen, fighter, cam, zoom, viewport, alpha)
        if self.show_debug:
            pygame.draw.circle(screen, (255, 80, 80), (round(foot.x), round(foot.y)), 4)
            hurtbox = fighter.body_rect_at(render_pos.x, render_pos.y)
            hurt = pygame.Rect(
                round(viewport.x + (hurtbox.x - cam.x) * zoom),
                round(viewport.y + (hurtbox.y - cam.y) * zoom),
                max(1, round(hurtbox.w * zoom)),
                max(1, round(hurtbox.h * zoom)),
            )
            pygame.draw.rect(screen, (255, 100, 100), hurt, 1)
            hitbox = fighter.attack_hitbox()
            if hitbox is not None:
                hit = pygame.Rect(
                    round(viewport.x + (hitbox.x - cam.x) * zoom),
                    round(viewport.y + (hitbox.y - cam.y) * zoom),
                    max(1, round(hitbox.w * zoom)),
                    max(1, round(hitbox.h * zoom)),
                )
                pygame.draw.rect(screen, (255, 220, 80), hit, 1)

    def _draw_shield(
        self,
        screen: pygame.Surface,
        fighter: PeachFighter,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
        alpha: float,
    ) -> None:
        if not fighter.shielded or fighter.xinc != 0 or fighter.shield_size <= 0 or not self.shield_frames:
            return
        frame_index = max(0, min(len(self.shield_frames) - 1, fighter.color_frame - 1))
        image = self.shield_frames[frame_index]
        scale = zoom * fighter.shield_size / 100 / self.shield_source_scale
        scaled = self._quality_scale(
            image,
            (max(1, round(image.get_width() * scale)), max(1, round(image.get_height() * scale))),
        )
        visual = fighter.visual_bounds_at(fighter.render_pos(alpha).x, fighter.render_pos(alpha).y)
        center = self._world_to_screen(pygame.Vector2(visual.centerx, visual.centery), cam, zoom, viewport)
        screen.blit(scaled, (round(center.x - scaled.get_width() / 2), round(center.y - scaled.get_height() / 2)))

    def _draw_held_item(
        self,
        screen: pygame.Surface,
        fighter: PeachFighter,
        cam: pygame.Vector2,
        zoom: float,
        viewport: pygame.Rect,
    ) -> None:
        item = fighter.current_item_obj
        if item is None or not fighter.current_item:
            return
        hand_pos = fighter.held_item_pos()
        if hand_pos is None:
            return
        asset = self._item_draw_frame(item)
        if asset is None:
            return
        image = asset.image
        scale = zoom / max(1.0, asset.render_scale)
        scaled = self._quality_scale(
            image,
            (max(1, round(image.get_width() * scale)), max(1, round(image.get_height() * scale))),
        )
        logical_center = asset.offset + pygame.Vector2(
            image.get_width() / max(1.0, asset.render_scale) / 2,
            image.get_height() / max(1.0, asset.render_scale) / 2,
        )
        pos = self._world_to_screen(hand_pos + logical_center, cam, zoom, viewport)
        screen.blit(scaled, (round(pos.x - scaled.get_width() / 2), round(pos.y - scaled.get_height() / 2)))


def main() -> None:
    RuntimeApp().run()


if __name__ == "__main__":
    main()
