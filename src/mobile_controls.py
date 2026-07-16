from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import pygame


CONTROL_NAMES = (
    "left",
    "right",
    "up_trace",
    "down",
    "jump_pressed",
    "punch_pressed",
    "special_pressed",
    "shield_pressed",
    "shield_released",
)


@dataclass(frozen=True)
class MobileControlLayout:
    stick_center: pygame.Vector2
    stick_radius: float
    punch_center: pygame.Vector2
    special_center: pygame.Vector2
    shield_center: pygame.Vector2
    button_radius: float
    pause_rect: pygame.Rect

    @classmethod
    def for_size(cls, size: tuple[int, int]) -> "MobileControlLayout":
        width, height = size
        short = min(width, height)
        stick_radius = max(64.0, min(132.0, short * 0.165))
        button_radius = max(43.0, min(88.0, short * 0.105))
        edge_x = max(stick_radius + 30, width * 0.115)
        base_y = height - max(stick_radius + 24, height * 0.155)
        right_edge = width - max(button_radius + 30, width * 0.075)
        return cls(
            stick_center=pygame.Vector2(edge_x, base_y),
            stick_radius=stick_radius,
            punch_center=pygame.Vector2(right_edge, base_y - button_radius * 0.9),
            special_center=pygame.Vector2(right_edge - button_radius * 2.05, base_y + button_radius * 0.15),
            shield_center=pygame.Vector2(right_edge - button_radius * 0.65, base_y + button_radius * 1.55),
            button_radius=button_radius,
            pause_rect=pygame.Rect(width - max(96, round(short * 0.13)), 20, max(72, round(short * 0.1)), max(54, round(short * 0.075))),
        )


class MobileControls:
    """Multi-touch adapter kept completely outside the desktop control path."""

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = (
            os.environ.get("GLORTON_MOBILE") == "1" or sys.platform == "emscripten"
            if enabled is None
            else bool(enabled)
        )
        self._finger_roles: dict[int, str] = {}
        self._ui_finger: int | None = None
        self._stick = pygame.Vector2()
        self._stick_up = False
        self._pending_jump = False
        self._pending_punch = False
        self._pending_special = False
        self._pending_shield_press = False
        self._pending_shield_release = False
        self._shield_held = False
        self._pause_toggle = False
        self._last_touch_ms = -10_000
        self._font_cache: dict[int, pygame.font.Font] = {}

    @staticmethod
    def _finger_id(event: pygame.event.Event) -> int:
        return int(getattr(event, "finger_id", getattr(event, "touch_id", 0)))

    @staticmethod
    def _finger_pos(event: pygame.event.Event, size: tuple[int, int]) -> pygame.Vector2:
        return pygame.Vector2(float(event.x) * size[0], float(event.y) * size[1])

    @staticmethod
    def _inside_circle(pos: pygame.Vector2, center: pygame.Vector2, radius: float) -> bool:
        return pos.distance_squared_to(center) <= radius * radius

    def reset(self) -> None:
        self._finger_roles.clear()
        self._ui_finger = None
        self._stick.update(0, 0)
        self._stick_up = False
        self._pending_jump = False
        self._pending_punch = False
        self._pending_special = False
        self._pending_shield_press = False
        self._pending_shield_release = self._shield_held
        self._shield_held = False
        self._pause_toggle = False

    def handle_battle_event(self, event: pygame.event.Event, size: tuple[int, int]) -> bool:
        """Consume a touch event and update battle controls."""

        if not self.enabled or event.type not in {pygame.FINGERDOWN, pygame.FINGERMOTION, pygame.FINGERUP}:
            return False
        self._last_touch_ms = pygame.time.get_ticks()
        finger_id = self._finger_id(event)
        pos = self._finger_pos(event, size)
        layout = MobileControlLayout.for_size(size)

        if event.type == pygame.FINGERDOWN:
            if layout.pause_rect.collidepoint(pos):
                self._finger_roles[finger_id] = "pause"
                self._pause_toggle = True
                return True
            role = ""
            if self._inside_circle(pos, layout.punch_center, layout.button_radius * 1.2):
                role = "punch"
                self._pending_punch = True
            elif self._inside_circle(pos, layout.special_center, layout.button_radius * 1.2):
                role = "special"
                self._pending_special = True
            elif self._inside_circle(pos, layout.shield_center, layout.button_radius * 1.2):
                role = "shield"
                if not self._shield_held:
                    self._pending_shield_press = True
                self._shield_held = True
            elif pos.x <= size[0] * 0.48 and "stick" not in self._finger_roles.values():
                role = "stick"
                self._update_stick(pos, layout)
            if role:
                self._finger_roles[finger_id] = role
            return True

        role = self._finger_roles.get(finger_id)
        if event.type == pygame.FINGERMOTION:
            if role == "stick":
                self._update_stick(pos, layout)
            return True

        self._finger_roles.pop(finger_id, None)
        if role == "stick":
            self._stick.update(0, 0)
            self._stick_up = False
        elif role == "shield" and "shield" not in self._finger_roles.values():
            self._shield_held = False
            self._pending_shield_release = True
        return True

    def _update_stick(self, pos: pygame.Vector2, layout: MobileControlLayout) -> None:
        offset = pos - layout.stick_center
        if offset.length_squared() > layout.stick_radius * layout.stick_radius:
            offset.scale_to_length(layout.stick_radius)
        self._stick = offset / layout.stick_radius
        now_up = self._stick.y < -0.36
        if now_up and not self._stick_up:
            # Jump is an edge, while up_trace shares that edge with a simultaneously
            # pressed attack so the original uppercut/rocket combination survives.
            self._pending_jump = True
        self._stick_up = now_up

    def controls(self) -> dict[str, bool]:
        """Return one 40 Hz input sample and consume button/jump edges."""

        if not self.enabled:
            return {name: False for name in CONTROL_NAMES}
        result = {
            "left": self._stick.x < -0.27,
            "right": self._stick.x > 0.27,
            "up_trace": self._pending_jump,
            "down": self._stick.y > 0.36,
            "jump_pressed": self._pending_jump,
            "punch_pressed": self._pending_punch,
            "special_pressed": self._pending_special,
            "shield_pressed": self._pending_shield_press,
            "shield_released": self._pending_shield_release,
        }
        self._pending_jump = False
        self._pending_punch = False
        self._pending_special = False
        self._pending_shield_press = False
        self._pending_shield_release = False
        return result

    def merge_into(self, controls: list[dict[str, bool]], player_index: int | None) -> None:
        if not self.enabled or player_index is None or not 0 <= player_index < len(controls):
            return
        mobile = self.controls()
        target = controls[player_index]
        for name in CONTROL_NAMES:
            target[name] = bool(target.get(name) or mobile[name])

    def take_pause_toggle(self) -> bool:
        value = self._pause_toggle
        self._pause_toggle = False
        return value

    def post_mouse_event(self, event: pygame.event.Event, size: tuple[int, int]) -> bool:
        """Translate the primary menu touch into the menu's existing mouse path."""

        if not self.enabled or event.type not in {pygame.FINGERDOWN, pygame.FINGERMOTION, pygame.FINGERUP}:
            return False
        self._last_touch_ms = pygame.time.get_ticks()
        finger_id = self._finger_id(event)
        pos = tuple(round(value) for value in self._finger_pos(event, size))
        translated: pygame.event.Event | None = None
        if event.type == pygame.FINGERDOWN and self._ui_finger is None:
            self._ui_finger = finger_id
            translated = pygame.event.Event(
                pygame.MOUSEBUTTONDOWN,
                pos=pos,
                button=1,
                touch=True,
                glorton_translated=True,
            )
        elif event.type == pygame.FINGERMOTION and self._ui_finger == finger_id:
            translated = pygame.event.Event(
                pygame.MOUSEMOTION,
                pos=pos,
                rel=(0, 0),
                buttons=(1, 0, 0),
                touch=True,
                glorton_translated=True,
            )
        elif event.type == pygame.FINGERUP and self._ui_finger == finger_id:
            self._ui_finger = None
            translated = pygame.event.Event(
                pygame.MOUSEBUTTONUP,
                pos=pos,
                button=1,
                touch=True,
                glorton_translated=True,
            )
        if translated is not None:
            pygame.event.post(translated)
        return True

    def ignore_synthetic_mouse(self, event: pygame.event.Event) -> bool:
        if not self.enabled or event.type not in {pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION}:
            return False
        # Keep the one mouse event we post from the primary finger, but drop
        # SDL/browser compatibility mouse events generated for that same
        # touch.  Without the explicit marker, a menu tap can toggle a player
        # twice (human -> CPU -> human) or fire GO twice on mobile browsers.
        if getattr(event, "glorton_translated", False):
            return False
        if getattr(event, "touch", False):
            return True
        return pygame.time.get_ticks() - self._last_touch_ms < 600

    def draw(self, screen: pygame.Surface, *, paused: bool = False) -> None:
        if not self.enabled:
            return
        layout = MobileControlLayout.for_size(screen.get_size())
        layer = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        line = max(3, round(layout.button_radius * 0.07))

        if not paused:
            pygame.draw.circle(layer, (8, 12, 18, 82), layout.stick_center, layout.stick_radius)
            pygame.draw.circle(layer, (235, 242, 255, 95), layout.stick_center, layout.stick_radius, line)
            knob = layout.stick_center + self._stick * layout.stick_radius
            pygame.draw.circle(layer, (220, 232, 250, 150), knob, layout.stick_radius * 0.38)
            self._draw_button(layer, layout.special_center, layout.button_radius, "SP", (245, 159, 40), line)
            self._draw_button(layer, layout.punch_center, layout.button_radius, "ATK", (220, 65, 62), line)
            self._draw_button(layer, layout.shield_center, layout.button_radius, "DEF", (55, 146, 224), line)

        pygame.draw.rect(layer, (8, 12, 18, 90), layout.pause_rect, border_radius=12)
        pygame.draw.rect(layer, (235, 242, 255, 105), layout.pause_rect, max(2, line - 1), border_radius=12)
        center = layout.pause_rect.center
        if paused:
            points = [
                (center[0] - 8, center[1] - 15),
                (center[0] - 8, center[1] + 15),
                (center[0] + 17, center[1]),
            ]
            pygame.draw.polygon(layer, (255, 255, 255, 185), points)
        else:
            bar_w = max(5, round(layout.pause_rect.w * 0.1))
            bar_h = round(layout.pause_rect.h * 0.46)
            for dx in (-bar_w, bar_w):
                pygame.draw.rect(layer, (255, 255, 255, 185), (center[0] + dx - bar_w // 2, center[1] - bar_h // 2, bar_w, bar_h), border_radius=2)
        screen.blit(layer, (0, 0))

    def _draw_button(
        self,
        layer: pygame.Surface,
        center: pygame.Vector2,
        radius: float,
        label: str,
        color: tuple[int, int, int],
        line: int,
    ) -> None:
        active = label == "DEF" and self._shield_held
        alpha = 180 if active else 112
        pygame.draw.circle(layer, (*color, alpha), center, radius)
        pygame.draw.circle(layer, (255, 255, 255, 145), center, radius, line)
        font_size = max(18, round(radius * (0.47 if len(label) == 3 else 0.58)))
        font = self._font_cache.get(font_size)
        if font is None:
            font = pygame.font.Font(None, font_size)
            self._font_cache[font_size] = font
        text = font.render(label, True, (255, 255, 255))
        layer.blit(text, text.get_rect(center=(round(center.x), round(center.y))))
