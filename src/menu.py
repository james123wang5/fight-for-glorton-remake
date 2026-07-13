from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import pygame

try:
    from .assets import LazySurfaceSequence
except ImportError:
    from assets import LazySurfaceSequence


REFERENCE_SIZE = (600, 400)
EXPORT_SCALE = 4
CANVAS_SIZE = (REFERENCE_SIZE[0] * EXPORT_SCALE, REFERENCE_SIZE[1] * EXPORT_SCALE)


@dataclass(frozen=True)
class MenuAction:
    kind: str
    payload: dict[str, object] | None = None


@dataclass(frozen=True)
class MenuButton:
    symbol_id: int
    center: tuple[float, float]
    action: str


@dataclass(frozen=True)
class FloatRect:
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0

    def collidepoint(self, point: tuple[float, float]) -> bool:
        px, py = point
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def move(self, offset: tuple[float, float]) -> FloatRect:
        dx, dy = offset
        return FloatRect(self.x + dx, self.y + dy, self.w, self.h)

    def copy(self) -> FloatRect:
        return FloatRect(self.x, self.y, self.w, self.h)


class MainMenu:
    """Recreates main timeline frames 1-50 and their AS2 navigation."""

    FRAME_BY_SCENE = {
        "intro": 42,
        "main": 43,
        "single": 44,
        "multi": 45,
        "player_select": 46,
        "stage_select": 47,
        "game_load": 48,
        "controls": 49,
        "options": 50,
    }
    BUTTONS = {
        "main": (
            MenuButton(969, (302.0, 214.2), "single"),
            MenuButton(965, (302.0, 243.2), "multi"),
            MenuButton(974, (302.0, 272.2), "controls"),
            MenuButton(979, (302.0, 301.2), "options"),
            MenuButton(983, (302.0, 330.3), "more_games"),
        ),
        "single": (
            MenuButton(1004, (300.0, 162.35), "quick"),
            MenuButton(994, (300.0, 196.75), "endurance"),
            MenuButton(999, (300.0, 231.15), "oneonone"),
            MenuButton(988, (300.0, 301.75), "main"),
        ),
        "multi": (
            MenuButton(1010, (300.0, 167.45), "vsmode"),
            MenuButton(988, (300.0, 301.75), "main"),
        ),
    }
    FIGHTERS = (
        ("SBLPlayer", 110),
        ("PeachPlayer", 422),
        ("TrashPlayer", 326),
        ("CoffeePlayer", 230),
        ("DefaultPlayer", 560),
        ("AuberginePlayer", 10),
    )
    STAGES = (
        ("Rooftop", 713, (30.0, 90.0)),
        ("Mogadishu", 830, (30.0 + 550.0 / 3.0, 90.0)),
        ("B52", 874, (30.0 + 1100.0 / 3.0, 90.0)),
        ("Space", 900, (30.0, 210.0)),
    )
    TEAM_COLORS = ((194, 24, 31), (48, 91, 181), (40, 153, 65), (220, 125, 30))
    CONTROL_NAMES = ("Left", "Right", "Up/Jump", "Down/Crouch", "Punch", "Special Attack", "Shield")
    HOVER_TEXT = {
        965: "Play with your friends or ennemies! Up to 4 people can play simultaneously!",
        969: "No friends? Play by yourself and win challenges. ",
        974: "Change the keyboard controls.",
        979: "Change image quality or mute sounds!",
        994: "How long can you last against a never ending flow of enemies?",
        999: "Play against only one fighter.",
        1004: "Want to play right now? Just choose a fighter and you're ready to fight!",
        1010: "The classical mode: every man for himself!",
    }
    # Pixel-aligned against original main timeline frames 43-45 and the
    # exported button glyphs. Keeping the extracted result avoids decoding
    # three 2400x1600 reference frames every time the game starts.
    BUTTON_HOVER_OFFSETS = {
        969: (18, 8),
        965: (13, 0),
        974: (18, 6),
        979: (18, 9),
        983: (13, 9),
        1004: (24, 9),
        994: (24, 8),
        999: (24, 8),
        988: (23, 10),
        1010: (24, 6),
    }

    def __init__(self, root: Path, manifest: dict | None = None) -> None:
        self.root = root
        self.futura_font_path = root / "assets/fonts/2_Futura Md BT.ttf"
        self._font_cache: dict[tuple[int, bool], pygame.font.Font] = {}
        self.frame_root = root / "assets/menu/main_frames"
        self.sprite_root = root / "assets/menu/sprites"
        self.button_root = root / "assets/menu/buttons"
        self.menu_background = pygame.image.load(str(root / "assets/menu/shapes/958.png")).convert_alpha()
        self.frames: OrderedDict[int, pygame.Surface] = OrderedDict()
        self.buttons = {
            symbol_id: pygame.image.load(
                str(self.button_root / f"DefineButton2_{symbol_id}" / "1.png")
            ).convert_alpha()
            for symbol_id in {
                button.symbol_id for buttons in self.BUTTONS.values() for button in buttons
            }
            | {1011, 1019}
        }
        self.button_hover_offsets = dict(self.BUTTON_HOVER_OFFSETS)
        self.fighter_box = self._load_sprite(793)
        self.player_boxes = [self._load_sprite(809, frame) for frame in range(1, 6)]
        self.player_coins = [self._load_sprite(816, frame) for frame in range(1, 5)]
        self.stage_box = [self._load_sprite(819, frame) for frame in range(1, 3)]
        self.limit_picker_frames = [self._load_sprite(1028, frame) for frame in range(1, 3)]
        self.limit_picker_background = pygame.image.load(
            str(root / "assets/menu/shapes/1020.png")
        ).convert_alpha()
        self.fighter_poses = {name: self._load_sprite(symbol_id) for name, symbol_id in self.FIGHTERS}
        self.manifest = manifest or json.loads(
            (root / "assets/manifests/glorton_manifest.json").read_text(encoding="utf-8")
        )
        self.menu_data = self.manifest.get("menu", {})
        self.sponsor_data = self.menu_data.get("sponsor_intro", {})
        sponsor_dir = root / str(
            self.sponsor_data.get("asset_dir", "assets/menu/sponsor_intro/2")
        )
        sponsor_count = int(self.sponsor_data.get("frame_count", 0))
        self.sponsor_intro_frames = LazySurfaceSequence(
            sponsor_dir / f"{frame_no}.png" for frame_no in range(1, sponsor_count + 1)
        )
        self.player_select_data = self.menu_data.get("player_select", {})
        self.root_button_rects: dict[tuple[int, int], FloatRect] = {}
        for frame_no, records in self.menu_data.get("root_buttons", {}).items():
            for record in records:
                rect = record["hit_rect"]
                self.root_button_rects[(int(frame_no), int(record["symbol_id"]))] = FloatRect(
                    float(rect["x"]),
                    float(rect["y"]),
                    float(rect["w"]),
                    float(rect["h"]),
                )

        self.selection_previews: dict[str, list[dict[str, object]]] = {}
        for fighter_name, preview in self.menu_data.get("selection_previews", {}).items():
            colors: list[dict[str, object]] = []
            for color_frame in range(1, 5):
                run = preview["colors"][str(color_frame)]
                state_offset = run.get("state_offset", {"x": 0.0, "y": 0.0})
                colors.append(
                    {
                        "frames": LazySurfaceSequence(root / item["image"] for item in run["frames"]),
                        "metadata": list(run["frames"]),
                        "state_offset": pygame.Vector2(
                            float(state_offset["x"]), float(state_offset["y"])
                        ),
                        "playback": dict(run.get("playback", {})),
                    }
                )
            self.selection_previews[fighter_name] = colors

        self.player_coin_backgrounds: list[tuple[pygame.Surface, pygame.Vector2]] = []
        coin_assets = self.player_select_data.get("coin_assets", {})
        for color_frame in range(1, 5):
            item = coin_assets.get(str(color_frame))
            if item:
                offset = item.get("offset", {"x": -13.0, "y": -13.0})
                self.player_coin_backgrounds.append(
                    (
                        pygame.image.load(str(root / item["image"])).convert_alpha(),
                        pygame.Vector2(float(offset["x"]), float(offset["y"])),
                    )
                )
            else:
                self.player_coin_backgrounds.append(
                    (self.player_coins[color_frame - 1], pygame.Vector2(-13.0, -13.0))
                )

        self.player_box_backgrounds: list[tuple[pygame.Surface, pygame.Vector2]] = []
        box_assets = self.player_select_data.get("player_box_backgrounds", {})
        for frame_no in range(1, 6):
            item = box_assets.get(str(frame_no))
            if item:
                offset = item.get("offset", {"x": 0.0, "y": 0.0})
                self.player_box_backgrounds.append(
                    (
                        pygame.image.load(str(root / item["image"])).convert_alpha(),
                        pygame.Vector2(float(offset["x"]), float(offset["y"])),
                    )
                )
            else:
                self.player_box_backgrounds.append((self.player_boxes[frame_no - 1], pygame.Vector2()))

        preloader = self.menu_data.get("preloader", {})
        self.preloader_ready = self._load_sprite(int(preloader.get("root_symbol_id", 913)), 2)
        self.option_checkboxes = [self._load_sprite(1048, frame) for frame in range(1, 3)]
        key_presser_root = self.sprite_root / "DefineSprite_1041_textless"
        self.key_presser_frames = LazySurfaceSequence(
            key_presser_root / f"{frame}.png" for frame in range(1, 21)
        )
        self.key_presser_masks = {
            name: LazySurfaceSequence(
                key_presser_root / f"{name}_masks/{frame}.png" for frame in range(1, 21)
            )
            for name in ("name", "key")
        }
        self.control_row_backgrounds = [self._load_sprite(850, frame) for frame in range(1, 3)]
        self.control_key_box = self._load_sprite(852)
        self.stage_thumbs = {name: self._load_sprite(symbol_id) for name, symbol_id, _ in self.STAGES}
        self.scene = "preloader"
        self.sponsor_frame = 1
        self.sponsor_elapsed_ms = 0.0
        self.opening_frame = int(self.menu_data.get("opening_start_frame", 3))
        self.opening_elapsed_ms = 0.0
        self.selection_animation_ms = 0
        self.mode = "multi"
        self.game_type = "vsmode"
        self.num_gamers = 4
        self.num_players = 4
        self.limit_mode = "stock"
        self.limit_value = 5
        self.selected_fighters: list[str | None] = []
        self.selected_colors: list[int | None] = []
        self.computer_players: list[bool] = []
        self.player_enabled: list[bool] = []
        self.player_levels: list[int] = []
        self.coin_positions: list[pygame.Vector2] = []
        self.coin_targets: list[pygame.Vector2 | None] = []
        self.dragging_player: int | None = None
        self.pressed_button: int | None = None
        self.drag_pos = pygame.Vector2()
        self.quality = "MEDIUM"
        self.sound_on = True
        self.listening_control: tuple[int, int] | None = None
        self.control_confirmation: tuple[int, int, str] | None = None
        self.control_confirmation_elapsed_ms = 0
        self.control_keys = [
            [pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s, pygame.K_j, pygame.K_k, pygame.K_LSHIFT],
            [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_KP0, pygame.K_KP1, pygame.K_KP2],
            [0] * 7,
            [0] * 7,
        ]
        self.settings_path = Path.home() / ".glorton_remake_settings.json"
        self._load_settings()

    def _load_settings(self) -> None:
        try:
            saved = json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return
        quality = str(saved.get("quality", self.quality)).upper()
        if quality in {"HIGH", "MEDIUM", "LOW"}:
            self.quality = quality
        self.sound_on = bool(saved.get("sound_on", self.sound_on))
        controls = saved.get("control_keys")
        if isinstance(controls, list) and len(controls) == 4:
            for player, row in enumerate(controls):
                if isinstance(row, list) and len(row) == 7:
                    self.control_keys[player] = [int(key) for key in row]

    def _save_settings(self) -> None:
        data = {
            "quality": self.quality,
            "sound_on": self.sound_on,
            "control_keys": self.control_keys,
        }
        try:
            self.settings_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _load_sprite(self, symbol_id: int, frame: int = 1) -> pygame.Surface:
        matches = []
        for directory in self.sprite_root.glob("DefineSprite_*"):
            id_part = directory.name.removeprefix("DefineSprite_").split("_", 1)[0]
            if id_part.isdigit() and int(id_part) == symbol_id:
                candidate = directory / f"{frame}.png"
                if candidate.is_file():
                    matches.append(candidate)
        if not matches:
            raise FileNotFoundError(f"Missing menu sprite {symbol_id}, frame {frame}")
        return pygame.image.load(str(matches[0])).convert_alpha()

    def _compute_button_hover_offsets(self) -> dict[int, tuple[int, int]]:
        offsets: dict[int, tuple[int, int]] = {}
        for scene, buttons in self.BUTTONS.items():
            frame_no = self.FRAME_BY_SCENE[scene]
            base = pygame.image.load(str(self.frame_root / f"{frame_no}.png")).convert_alpha()
            white = pygame.mask.from_threshold(base, (255, 255, 255, 255), (70, 70, 70, 255))
            for button in buttons:
                image = self.buttons[button.symbol_id]
                glyph = pygame.mask.from_surface(image, 30)
                expected_x = round(button.center[0] * EXPORT_SCALE - image.get_width() / 2)
                expected_y = round(button.center[1] * EXPORT_SCALE - image.get_height() / 2)
                best_score = -1
                best_offset = (0, 0)
                for dy in range(-24, 25):
                    for dx in range(-24, 25):
                        score = white.overlap_area(glyph, (expected_x + dx, expected_y + dy))
                        if score > best_score:
                            best_score = score
                            best_offset = (dx, dy)
                offsets[button.symbol_id] = best_offset
        return offsets

    def _frame(self, index: int) -> pygame.Surface:
        if index in self.frames:
            frame = self.frames.pop(index)
            self.frames[index] = frame
            return frame
        frame = pygame.image.load(str(self.frame_root / f"{index}.png")).convert_alpha()
        self.frames[index] = frame
        while len(self.frames) > 3:
            self.frames.popitem(last=False)
        return frame

    def reset_to_intro(self) -> None:
        self.scene = "preloader"
        self.pressed_button = None
        self.sponsor_frame = 1
        self.sponsor_elapsed_ms = 0.0
        self.opening_frame = int(self.menu_data.get("opening_start_frame", 3))
        self.opening_elapsed_ms = 0.0

    def return_to_main(self) -> None:
        self.scene = "main"
        self.pressed_button = None
        self.dragging_player = None
        self.listening_control = None
        self.control_confirmation = None

    def update(self, elapsed_ms: int) -> MenuAction | None:
        if self.scene == "preloader":
            return None
        if self.scene == "sponsor_intro":
            frame_rate = float(self.sponsor_data.get("frame_rate", 30))
            frame_count = len(self.sponsor_intro_frames)
            self.sponsor_elapsed_ms += max(0, elapsed_ms)
            elapsed_frames = int(self.sponsor_elapsed_ms * frame_rate / 1000.0)
            self.sponsor_frame = 1 + elapsed_frames
            if self.sponsor_frame > frame_count:
                self.scene = "opening"
                self.sponsor_frame = max(1, frame_count)
                self.opening_frame = int(self.menu_data.get("opening_start_frame", 3))
                self.opening_elapsed_ms = 0.0
            return None
        if self.scene == "opening":
            frame_rate = float(self.menu_data.get("frame_rate", 30))
            start_frame = int(self.menu_data.get("opening_start_frame", 3))
            stop_frame = int(self.menu_data.get("opening_stop_frame", 39))
            self.opening_elapsed_ms += max(0, elapsed_ms)
            elapsed_frames = int(self.opening_elapsed_ms * frame_rate / 1000.0)
            self.opening_frame = start_frame + elapsed_frames
            if self.opening_frame > stop_frame:
                self.scene = "intro"
                self.opening_frame = stop_frame
            return None
        if self.scene == "player_select":
            self.selection_animation_ms += max(0, elapsed_ms)
            frame_factor = 1.0 - (2.0 / 3.0) ** (max(0, elapsed_ms) * 30.0 / 1000.0)
            for index, target in enumerate(self.coin_targets):
                if target is None or index == self.dragging_player:
                    continue
                self.coin_positions[index] += (target - self.coin_positions[index]) * frame_factor
        if self.scene == "controls" and self.control_confirmation is not None:
            self.control_confirmation_elapsed_ms += max(0, elapsed_ms)
            if self.control_confirmation_elapsed_ms >= 19 * 1000 / 30:
                self.control_confirmation = None
        if self.scene == "game_load":
            return self._start_game_action()
        return None

    def _start_game_action(self) -> MenuAction:
        active = [
            index
            for index, fighter in enumerate(self.selected_fighters)
            if fighter is not None and self.player_enabled[index]
        ]
        players = [
            {
                "fighter": self.selected_fighters[index],
                "color": self.selected_colors[index] if self.selected_colors[index] is not None else 0,
                "computer": self.computer_players[index],
                "enabled": self.player_enabled[index],
                "level": self.player_levels[index],
                "team_index": index,
            }
            for index in active
        ]
        return MenuAction(
            "start_game",
            {
                "mode": self.mode,
                "type": self.game_type,
                "stage": getattr(self, "selected_stage", "Rooftop"),
                "selected_stage": getattr(self, "selected_stage", "Rooftop"),
                "players": players,
                "fighters": [self.selected_fighters[index] for index in active],
                "colors": [
                    self.selected_colors[index] if self.selected_colors[index] is not None else 0
                    for index in active
                ],
                "limit_mode": self.limit_mode,
                "limit_value": self.limit_value,
            },
        )

    def handle_event(self, event: pygame.event.Event, screen_size: tuple[int, int]) -> MenuAction | None:
        if self.scene == "preloader":
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                ref_pos = self._screen_to_reference(event.pos, screen_size)
                rect = self.menu_data.get("preloader", {}).get("play_rect", {})
                if ref_pos is not None and FloatRect(
                    float(rect.get("x", 271.48)),
                    float(rect.get("y", 220.49)),
                    float(rect.get("w", 69.48)),
                    float(rect.get("h", 22.98)),
                ).collidepoint(ref_pos):
                    if self.sponsor_intro_frames:
                        self.scene = "sponsor_intro"
                        self.sponsor_frame = 1
                        self.sponsor_elapsed_ms = 0.0
                    else:
                        self.scene = "opening"
                        self.opening_frame = int(self.menu_data.get("opening_start_frame", 3))
                        self.opening_elapsed_ms = 0.0
            return None
        if self.scene == "sponsor_intro":
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                ref_pos = self._screen_to_reference(event.pos, screen_size)
                rect = self.sponsor_data.get("armor_button_rect", {})
                active_start = int(self.sponsor_data.get("armor_button_active_start", 23))
                active_stop = int(self.sponsor_data.get("armor_button_active_stop", 81))
                if (
                    ref_pos is not None
                    and active_start <= self.sponsor_frame <= active_stop
                    and FloatRect(
                        float(rect.get("x", 194.05)),
                        float(rect.get("y", 110.5)),
                        float(rect.get("w", 212.0)),
                        float(rect.get("h", 212.0)),
                    ).collidepoint(ref_pos)
                ):
                    return MenuAction(
                        "open_url",
                        {"url": str(self.sponsor_data.get("url", "http://www.armorgames.com"))},
                    )
            return None
        if self.scene in {"opening", "intro"}:
            if self.scene == "intro" and event.type in {pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN}:
                self.scene = "main"
            return None

        if self.scene == "controls" and self.listening_control is not None and event.type == pygame.KEYDOWN:
            player, control = self.listening_control
            self.control_keys[player][control] = event.key
            self.control_confirmation = (player, control, self._key_name(event.key))
            self.control_confirmation_elapsed_ms = 0
            self.listening_control = None
            self._save_settings()
            return None

        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            if self.scene != "main":
                self.return_to_main()
                return None
            return MenuAction("quit")

        if event.type not in {pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION}:
            return None
        ref_pos = self._screen_to_reference(event.pos, screen_size)
        if ref_pos is None:
            return None

        if self.scene == "player_select":
            return self._handle_player_select_event(event, ref_pos)
        if self.scene == "stage_select":
            return self._handle_stage_select_event(event, ref_pos)
        if self.scene in self.BUTTONS and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.pressed_button = next(
                (
                    button.symbol_id
                    for button in self.BUTTONS[self.scene]
                    if self._button_rect(button).collidepoint(ref_pos)
                ),
                None,
            )
            return None
        if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
            return None
        pressed_button = self.pressed_button
        self.pressed_button = None
        if self.scene in self.BUTTONS:
            for button in self.BUTTONS[self.scene]:
                if button.symbol_id == pressed_button and self._button_rect(button).collidepoint(ref_pos):
                    return self._activate_button(button.action)
        elif self.scene == "controls":
            return self._handle_controls_click(ref_pos)
        elif self.scene == "options":
            return self._handle_options_click(ref_pos)
        return None

    def _activate_button(self, action: str) -> MenuAction | None:
        if action == "main":
            self.return_to_main()
        elif action == "single":
            self.mode = "single"
            self.scene = "single"
        elif action == "multi":
            self.mode = "multi"
            self.scene = "multi"
        elif action in {"controls", "options"}:
            self.scene = action
        elif action == "more_games":
            return MenuAction("open_url", {"url": "http://www.armorgames.com/"})
        elif action == "endurance":
            self._start_player_select("endurance", 1, 1)
        elif action == "oneonone":
            self._start_player_select("oneonone", 1, 2, random_computers=1)
        elif action == "quick":
            self._start_player_select("quick", 1, 4, random_computers=3)
        elif action == "vsmode":
            self._start_player_select("vsmode", 4, 4)
        return None

    def _start_player_select(
        self,
        game_type: str,
        gamers: int,
        players: int,
        random_computers: int = 0,
    ) -> None:
        self.game_type = game_type
        self.num_gamers = gamers
        self.num_players = players
        self.limit_mode = "stock"
        self.limit_value = 5
        self.selected_fighters = [None] * players
        self.selected_colors = [None] * players
        self.computer_players = [index >= gamers for index in range(players)]
        self.player_enabled = [True] * players
        default_level = int(self.player_select_data.get("default_ai_level", 7))
        self.player_levels = [default_level] * players
        self.coin_positions = [pygame.Vector2(index * 32 + 30, 50) for index in range(players)]
        self.coin_targets = [None] * players
        self.selection_animation_ms = 0
        fighter_names = [name for name, _ in self.FIGHTERS]
        for index in range(players - random_computers, players):
            if 0 <= index < players:
                selected = random.choice(fighter_names)
                self.selected_fighters[index] = selected
                target_index = fighter_names.index(selected)
                self.coin_targets[index] = pygame.Vector2(
                    target_index * 94 + 40 + random.randrange(60) - 30,
                    125 - (random.randrange(60) - 30),
                )
                self._resolve_duplicate_color(index)
        self.scene = "player_select"

    def _handle_player_select_event(
        self,
        event: pygame.event.Event,
        ref_pos: tuple[float, float],
    ) -> MenuAction | None:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index in range(self.num_players):
                if not self.player_enabled[index]:
                    continue
                coin_pos = self.coin_positions[index]
                coin_rect = FloatRect(coin_pos.x - 13, coin_pos.y - 13, 26, 26)
                if coin_rect.collidepoint(ref_pos):
                    self.dragging_player = index
                    self.drag_pos.update(ref_pos)
                    return None
        elif event.type == pygame.MOUSEMOTION and self.dragging_player is not None:
            self.drag_pos.update(ref_pos)
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging_player is not None:
                player_index = self.dragging_player
                self.dragging_player = None
                fighter_index: int | None = None
                for candidate in range(len(self.FIGHTERS)):
                    if ref_pos[0] > candidate * 94 + 6:
                        fighter_index = candidate
                if fighter_index is not None:
                    name = self.FIGHTERS[fighter_index][0]
                    self.selected_fighters[player_index] = name
                    self.coin_targets[player_index] = pygame.Vector2(
                        fighter_index * 94 + 40 + random.randrange(60) - 30,
                        125 - (random.randrange(60) - 30),
                    )
                    self._resolve_duplicate_color(player_index)
                return None
            if self._main_menu_rect().collidepoint(ref_pos):
                self.return_to_main()
            elif self._root_button_rect(46, 1019).collidepoint(ref_pos):
                if any(self.selected_fighters):
                    self.scene = "stage_select"
            elif self._player_select_rect("decrement").collidepoint(ref_pos):
                self._adjust_limit(-1)
            elif self._player_select_rect("increment").collidepoint(ref_pos):
                self._adjust_limit(1)
            elif (
                self._player_select_rect("toggle_stock").collidepoint(ref_pos)
                or self._player_select_rect("toggle_time").collidepoint(ref_pos)
            ):
                self.limit_mode = "time" if self.limit_mode == "stock" else "stock"
                self.limit_value = 300 if self.limit_mode == "time" else 5
            else:
                for index in range(self.num_players):
                    origin = self._player_box_origin(index)
                    if self.computer_players[index] and self.player_enabled[index]:
                        if self._player_local_rect("decrement").move(origin).collidepoint(ref_pos):
                            minimum = int(self.player_select_data.get("ai_level_min", 1))
                            self.player_levels[index] = max(minimum, self.player_levels[index] - 1)
                            return None
                        if self._player_local_rect("increment").move(origin).collidepoint(ref_pos):
                            maximum = int(self.player_select_data.get("ai_level_max", 20))
                            self.player_levels[index] = min(maximum, self.player_levels[index] + 1)
                            return None
                    if self._player_local_rect("toggle").move(origin).collidepoint(ref_pos):
                        self._toggle_player(index)
                        return None
        return None

    def _adjust_limit(self, direction: int) -> None:
        if self.limit_mode == "stock":
            if direction < 0:
                self.limit_value = max(1, self.limit_value - 1)
            else:
                self.limit_value += 1
            return

        minutes, seconds = divmod(max(0, self.limit_value), 60)
        if direction < 0:
            if minutes <= 0 and seconds <= 30:
                minutes = 0
                seconds = 0
            elif minutes >= 20:
                minutes -= 10
            elif minutes >= 10:
                minutes -= 5
            elif minutes > 2:
                minutes -= 1
            else:
                seconds -= 30
                if seconds < 0:
                    seconds = 30
                    minutes -= 1
        elif minutes >= 20:
            minutes += 10
        elif minutes >= 10:
            minutes += 5
        elif minutes > 2:
            minutes += 1
        else:
            seconds += 30
            if seconds >= 60:
                seconds = 0
                minutes += 1
        self.limit_value = max(0, minutes * 60 + seconds)

    def _toggle_player(self, index: int) -> None:
        if self.mode == "single":
            return
        if not self.computer_players[index] and self.player_enabled[index]:
            self.computer_players[index] = True
            return
        if self.computer_players[index] and self.player_enabled[index]:
            self.player_enabled[index] = False
            return
        self.computer_players[index] = False
        self.player_enabled[index] = True

    def _player_box_origin(self, index: int) -> tuple[float, float]:
        return (index * (550.0 / max(1, self.num_players)) + 25.0, 188.0)

    def _player_local_rect(self, name: str) -> FloatRect:
        if name == "toggle":
            data = self.player_select_data.get("toggle_rect", {})
        else:
            data = self.player_select_data.get("ai_level_rects", {}).get(name, {})
        return FloatRect(
            float(data.get("x", 0.0)),
            float(data.get("y", 0.0)),
            float(data.get("w", 0.0)),
            float(data.get("h", 0.0)),
        )

    def _player_select_rect(self, name: str) -> FloatRect:
        data = self.player_select_data.get("limit_rects", {}).get(name, {})
        return FloatRect(
            float(data.get("x", 0.0)),
            float(data.get("y", 0.0)),
            float(data.get("w", 0.0)),
            float(data.get("h", 0.0)),
        )

    def _resolve_duplicate_color(self, player_index: int) -> None:
        # AS2 SetColor recursively chooses the next unused color for duplicate fighters.
        chosen = self.selected_fighters[player_index]
        if chosen is None:
            return
        used = {
            color
            for index, (fighter, color) in enumerate(zip(self.selected_fighters, self.selected_colors))
            if index != player_index and fighter == chosen and color is not None
        }
        color = 0
        for _ in range(4):
            if color not in used:
                self.selected_colors[player_index] = color
                return
            color = (color + 1) % 4

    def _handle_stage_select_event(
        self,
        event: pygame.event.Event,
        ref_pos: tuple[float, float],
    ) -> MenuAction | None:
        if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
            return None
        if self._main_menu_rect().collidepoint(ref_pos):
            self.return_to_main()
            return None
        for name, _symbol_id, pos in self.STAGES:
            if FloatRect(pos[0] - 2.5, pos[1] - 2.5, 155.75, 105.0).collidepoint(ref_pos):
                self.selected_stage = name
                self.scene = "game_load"
                return self._start_game_action()
        return None

    def _handle_controls_click(self, ref_pos: tuple[float, float]) -> MenuAction | None:
        if self._main_menu_rect().collidepoint(ref_pos):
            self.return_to_main()
            return None
        if 150 <= ref_pos[1] < 325:
            player = min(3, max(0, int(ref_pos[0] // 150)))
            control = min(6, max(0, int((ref_pos[1] - 150) // 25)))
            self.control_confirmation = None
            self.listening_control = (player, control)
        return None

    def _handle_options_click(self, ref_pos: tuple[float, float]) -> MenuAction | None:
        choices = (("HIGH", 124.25), ("MEDIUM", 152.25), ("LOW", 179.25))
        for quality, y in choices:
            if FloatRect(99.85, y - 1.0, 19.7, 19.7).collidepoint(ref_pos):
                self.quality = quality
                return None
        if FloatRect(99.85, 230.15, 19.7, 19.7).collidepoint(ref_pos):
            self.sound_on = True
        elif FloatRect(99.85, 258.15, 19.7, 19.7).collidepoint(ref_pos):
            self.sound_on = False
        elif self._main_menu_rect().collidepoint(ref_pos):
            self._save_settings()
            self.return_to_main()
        return None

    def draw(self, screen: pygame.Surface) -> None:
        canvas = self._base_canvas()
        mouse_ref = self._screen_to_reference(pygame.mouse.get_pos(), screen.get_size())
        if self.scene in self.BUTTONS:
            self._draw_button_hover(canvas, mouse_ref)
        elif self.scene == "player_select":
            self._draw_player_select(canvas, mouse_ref)
        elif self.scene == "stage_select":
            self._draw_stage_select(canvas, mouse_ref)
        elif self.scene == "controls":
            self._draw_controls(canvas)
        elif self.scene == "options":
            self._draw_options(canvas)
        self._fit_canvas(screen, canvas)

    def _base_canvas(self) -> pygame.Surface:
        if self.scene == "preloader":
            canvas = self._frame(1).copy()
            position = self.menu_data.get("preloader", {}).get("root_pos", {"x": 211, "y": 204})
            canvas.blit(
                self.preloader_ready,
                (
                    round(float(position["x"]) * EXPORT_SCALE),
                    round(float(position["y"]) * EXPORT_SCALE),
                ),
            )
            return canvas
        if self.scene == "opening":
            return self._frame(self.opening_frame).copy()
        if self.scene == "sponsor_intro" and self.sponsor_intro_frames:
            frame_index = max(0, min(len(self.sponsor_intro_frames) - 1, self.sponsor_frame - 1))
            return self.sponsor_intro_frames[frame_index].copy()
        frame = self.FRAME_BY_SCENE.get(self.scene, 43)
        if self.scene == "controls":
            return pygame.transform.smoothscale(self.menu_background, CANVAS_SIZE)
        return self._frame(frame).copy()

    def _draw_button_hover(
        self,
        canvas: pygame.Surface,
        mouse_ref: tuple[float, float] | None,
    ) -> None:
        if mouse_ref is None:
            return
        for button in self.BUTTONS[self.scene]:
            if not self._button_rect(button).collidepoint(mouse_ref):
                continue
            image = self.buttons[button.symbol_id].copy()
            image.fill((255, 204, 0, 255), special_flags=pygame.BLEND_RGBA_MULT)
            offset_x, offset_y = self.button_hover_offsets.get(button.symbol_id, (0, 0))
            down_shift = EXPORT_SCALE if self.pressed_button == button.symbol_id else 0
            up_position = (
                round(button.center[0] * EXPORT_SCALE - image.get_width() / 2 + offset_x),
                round(button.center[1] * EXPORT_SCALE - image.get_height() / 2 + offset_y),
            )
            # DefineButton2 replaces its Up record with the Over/Down record.
            # Restore the exact root texture first so antialiased white Up
            # pixels do not remain underneath the yellow glyph.
            canvas.blit(
                self.menu_background,
                up_position,
                pygame.Rect(up_position, image.get_size()),
            )
            canvas.blit(
                image,
                (
                    up_position[0] + down_shift,
                    up_position[1] + down_shift,
                ),
            )
            description = self.HOVER_TEXT.get(button.symbol_id)
            if description:
                self._draw_text(canvas, description, 19, 353, 12, (255, 255, 255))

    def _draw_player_select(
        self,
        canvas: pygame.Surface,
        mouse_ref: tuple[float, float] | None,
    ) -> None:
        for index, (name, _symbol_id) in enumerate(self.FIGHTERS):
            x = (index * 94 + 6) * EXPORT_SCALE
            y = 70 * EXPORT_SCALE
            canvas.blit(self.fighter_box, (x, y))
            pose = self.fighter_poses[name]
            max_w = 78 * EXPORT_SCALE
            max_h = 96 * EXPORT_SCALE
            scale = min(max_w / pose.get_width(), max_h / pose.get_height(), 1.0)
            pose_draw = pygame.transform.smoothscale(
                pose,
                (max(1, round(pose.get_width() * scale)), max(1, round(pose.get_height() * scale))),
            )
            canvas.blit(
                pose_draw,
                (
                    round(x + 43.5 * EXPORT_SCALE - pose_draw.get_width() / 2),
                    round(y + 55 * EXPORT_SCALE - pose_draw.get_height() / 2),
                ),
            )
        for index in range(self.num_players):
            x, y = self._player_box_origin(index)
            enabled = self.player_enabled[index]
            color_index = min(index, 3)
            if enabled and self.computer_players[index]:
                box = self.player_boxes[color_index].copy()
                number_patch = pygame.transform.smoothscale(
                    box.subsurface(pygame.Rect(220, 64, 12, 104)).copy(),
                    (82, 104),
                )
                box.blit(number_patch, (345, 64))
                canvas.blit(box, (round(x * EXPORT_SCALE), round(y * EXPORT_SCALE)))
                self._draw_center_text(
                    canvas,
                    str(self.player_levels[index]),
                    x + 96,
                    y + 31,
                    18,
                    (255, 255, 255),
                )
            else:
                frame_index = color_index if enabled else 4
                box, box_offset = self.player_box_backgrounds[frame_index]
                canvas.blit(
                    box,
                    (
                        round((x + box_offset.x) * EXPORT_SCALE),
                        round((y + box_offset.y) * EXPORT_SCALE),
                    ),
                )
                if not enabled:
                    self._draw_text(canvas, "N/A", x + 7, y + 25, 28, (255, 255, 255))

            label = "CP" if self.computer_players[index] else f"P{index + 1}"
            if enabled:
                self._draw_text(canvas, label, x + 6, y + 19, 28, (255, 255, 255))
            selected = self.selected_fighters[index]
            if selected and enabled:
                self._draw_selection_preview(canvas, selected, index, x, y)

            if enabled:
                coin_x, coin_y = self.coin_positions[index]
                if self.dragging_player == index:
                    coin_x, coin_y = self.drag_pos
                coin, coin_offset = self.player_coin_backgrounds[color_index]
                canvas.blit(
                    coin,
                    (
                        round((coin_x + coin_offset.x) * EXPORT_SCALE),
                        round((coin_y + coin_offset.y) * EXPORT_SCALE),
                    ),
                )
                self._draw_center_text(canvas, label, coin_x, coin_y, 11, (255, 255, 255))

        self._draw_limit_picker(canvas)

        if mouse_ref and self._root_button_rect(46, 1019).collidepoint(mouse_ref):
            go = self.buttons[1019].copy()
            go.fill((255, 220, 80, 255), special_flags=pygame.BLEND_RGBA_MULT)
            canvas.blit(go, (round(534.85 * EXPORT_SCALE - go.get_width() / 2), round(36.8 * EXPORT_SCALE - go.get_height() / 2)))

    def _draw_selection_preview(
        self,
        canvas: pygame.Surface,
        fighter_name: str,
        player_index: int,
        box_x: float,
        box_y: float,
    ) -> None:
        colors = self.selection_previews.get(fighter_name)
        if not colors:
            return
        color = self.selected_colors[player_index] if self.selected_colors[player_index] is not None else 0
        run = colors[max(0, min(3, color))]
        frames = run["frames"]
        if not frames:
            return
        frame_no = int(self.selection_animation_ms * 30 / 1000) + 1
        playback = run["playback"]
        loop_from = int(playback.get("loop_from", 1))
        loop_at = int(playback.get("loop_at", len(frames)))
        if frame_no > loop_at:
            frame_no = loop_from + (
                (frame_no - loop_at - 1) % max(1, loop_at - loop_from + 1)
            )
        frame_index = max(0, min(len(frames) - 1, frame_no - 1))
        pose = frames[frame_index]
        metadata = run["metadata"][frame_index]
        local = metadata.get("offset", {"x": 0.0, "y": 0.0})
        offset = run["state_offset"] + pygame.Vector2(float(local["x"]), float(local["y"]))
        source_scale = max(1.0, float(metadata.get("render_scale", EXPORT_SCALE)))
        display_scale = float(self.player_select_data.get("preview", {}).get("scale", 3.5))
        raster_scale = display_scale * EXPORT_SCALE / source_scale
        pose_draw = pygame.transform.smoothscale(
            pose,
            (
                max(1, round(pose.get_width() * raster_scale)),
                max(1, round(pose.get_height() * raster_scale)),
            ),
        )
        preview_data = self.player_select_data.get("preview", {})
        anchor_x = float(preview_data.get("x", 70))
        anchor_y = float(preview_data.get("y", 170))
        canvas.blit(
            pose_draw,
            (
                round((box_x + anchor_x + offset.x * display_scale) * EXPORT_SCALE),
                round((box_y + anchor_y + offset.y * display_scale) * EXPORT_SCALE),
            ),
        )

    def _draw_limit_picker(self, canvas: pygame.Surface) -> None:
        # Root frame 46 already contains the exact default stock/5 state.
        if self.limit_mode == "stock" and self.limit_value == 5:
            return

        background_pos = (round(109.95 * EXPORT_SCALE), round(-3.0 * EXPORT_SCALE))
        value_rect = pygame.Rect(
            round(308.0 * EXPORT_SCALE),
            0,
            round(75.0 * EXPORT_SCALE),
            round(30.6 * EXPORT_SCALE),
        )
        if self.limit_mode == "time":
            canvas.blit(self.limit_picker_background, background_pos)
            canvas.blit(
                self.limit_picker_frames[1],
                (round(211.0 * EXPORT_SCALE), round(-1.0 * EXPORT_SCALE)),
            )
            value_text = self._format_time(self.limit_value)
        else:
            source_rect = value_rect.move(-background_pos[0], -background_pos[1])
            canvas.blit(self.limit_picker_background, value_rect, source_rect)
            value_text = str(self.limit_value)
        self._draw_center_text(
            canvas,
            value_text,
            345.5,
            14.8,
            23,
            (255, 255, 255),
        )

    def _draw_stage_select(
        self,
        canvas: pygame.Surface,
        mouse_ref: tuple[float, float] | None,
    ) -> None:
        for name, _symbol_id, pos in self.STAGES:
            x = round(pos[0] * EXPORT_SCALE)
            y = round(pos[1] * EXPORT_SCALE)
            image = self.stage_thumbs[name]
            hovered = bool(
                mouse_ref
                and FloatRect(pos[0] - 2.5, pos[1] - 2.5, 155.75, 105.0).collidepoint(mouse_ref)
            )
            canvas.blit(
                self.stage_box[1 if hovered else 0],
                (round((pos[0] - 2.5) * EXPORT_SCALE), round((pos[1] - 2.5) * EXPORT_SCALE)),
            )
            canvas.blit(image, (x, y))

    def _draw_controls(self, canvas: pygame.Surface) -> None:
        self._draw_text(canvas, "Controls Setup", 13, 42, 43, (255, 255, 255), bold=True)
        self._draw_text(
            canvas,
            "Got a Joystick or Game Controller? Click",
            13,
            82,
            13,
            (255, 255, 255),
            bold=True,
        )
        for player in range(4):
            self._draw_center_text(canvas, f"Player {player + 1}", player * 150 + 75, 132, 14, (255, 255, 255), bold=True)
            for control, name in enumerate(self.CONTROL_NAMES):
                x = player * 150
                y = 150 + control * 25
                row = self.control_row_backgrounds[control % 2]
                row = pygame.transform.smoothscale(row, (150 * EXPORT_SCALE, 25 * EXPORT_SCALE))
                canvas.blit(row, (x * EXPORT_SCALE, y * EXPORT_SCALE))
                canvas.blit(
                    self.control_key_box,
                    (
                        round((x + 96.8) * EXPORT_SCALE),
                        round((y + 1.6) * EXPORT_SCALE),
                    ),
                )
                self._draw_text(canvas, name, x + 3.4, y + 12.5, 12, (51, 51, 51))
                key_name = self._key_name(self.control_keys[player][control])
                self._draw_center_text(canvas, key_name, x + 121.15, y + 12.5, 12, (0, 0, 0))
        main_button = self.buttons[1011]
        main_scaled = pygame.transform.smoothscale(
            main_button,
            (
                max(1, round(main_button.get_width() * 0.54545593)),
                max(1, round(main_button.get_height() * 0.54545593)),
            ),
        )
        canvas.blit(
            main_scaled,
            (
                round(49 * EXPORT_SCALE - main_scaled.get_width() / 2),
                round(382.3 * EXPORT_SCALE - main_scaled.get_height() / 2),
            ),
        )
        if self.listening_control is not None or self.control_confirmation is not None:
            self._draw_key_presser(canvas)

    def _draw_key_presser(self, canvas: pygame.Surface) -> None:
        if self.listening_control is not None:
            player, control = self.listening_control
            key_text = ""
            frame_no = 1
        else:
            assert self.control_confirmation is not None
            player, control, key_text = self.control_confirmation
            frame_no = min(
                19,
                1 + int(self.control_confirmation_elapsed_ms * 30 / 1000),
            )
        canvas.blit(self.key_presser_frames[frame_no - 1], (0, 0))
        text_overlay = pygame.Surface(CANVAS_SIZE, pygame.SRCALPHA)
        text_overlay.fill((0, 0, 0, 0))
        self._draw_center_text(
            text_overlay,
            f"Player {player + 1}/{self.CONTROL_NAMES[control]}",
            294.0,
            229.15,
            16,
            (255, 255, 255),
        )
        if key_text:
            key_image = self._font(24).render(key_text, True, (255, 255, 102))
            shadow = self._font(24).render(key_text, True, (0, 0, 0))
            x = round(293.95 * EXPORT_SCALE - key_image.get_width() / 2)
            y = round(269.95 * EXPORT_SCALE - key_image.get_height() / 2)
            spread = 2 * EXPORT_SCALE
            for dx, dy in ((-spread, 0), (spread, 0), (0, -spread), (0, spread)):
                text_overlay.blit(shadow, (x + dx, y + dy))
            text_overlay.blit(key_image, (x, y))
        field_rects = {
            "name": pygame.Rect(724, 866, 904, 101),
            "key": pygame.Rect(720, 982, 912, 164),
        }
        if frame_no > 1:
            for name, rect in field_rects.items():
                field = text_overlay.subsurface(rect).copy()
                field.blit(
                    self.key_presser_masks[name][frame_no - 1],
                    (0, 0),
                    special_flags=pygame.BLEND_RGBA_MULT,
                )
                text_overlay.fill((0, 0, 0, 0), rect)
                text_overlay.blit(field, rect)
        canvas.blit(text_overlay, (0, 0))

    def _draw_options(self, canvas: pygame.Surface) -> None:
        for quality, y in (("HIGH", 124.25), ("MEDIUM", 152.25), ("LOW", 179.25)):
            self._draw_option_checkbox(canvas, 100.85, y, self.quality == quality)
        self._draw_option_checkbox(canvas, 100.85, 231.15, self.sound_on)
        self._draw_option_checkbox(canvas, 100.85, 259.15, not self.sound_on)

    def _draw_option_checkbox(
        self,
        canvas: pygame.Surface,
        x: float,
        y: float,
        selected: bool,
    ) -> None:
        dest = pygame.Rect(
            round((x - 1.0) * EXPORT_SCALE),
            round((y - 1.0) * EXPORT_SCALE),
            self.option_checkboxes[0].get_width(),
            self.option_checkboxes[0].get_height(),
        )
        canvas.blit(self.menu_background, dest, dest)
        canvas.blit(
            self.option_checkboxes[0 if selected else 1],
            dest.topleft,
        )

    def _button_rect(self, button: MenuButton) -> FloatRect:
        exact = self.root_button_rects.get((self.FRAME_BY_SCENE[self.scene], button.symbol_id))
        if exact is not None:
            return exact.copy()
        image = self.buttons[button.symbol_id]
        width = image.get_width() / EXPORT_SCALE
        height = max(18.0, image.get_height() / EXPORT_SCALE)
        return FloatRect(button.center[0] - width / 2, button.center[1] - height / 2, width, height)

    def _root_button_rect(self, frame_no: int, symbol_id: int) -> FloatRect:
        rect = self.root_button_rects.get((frame_no, symbol_id))
        return rect.copy() if rect is not None else FloatRect()

    def _main_menu_rect(self) -> FloatRect:
        frame_no = self.FRAME_BY_SCENE.get(self.scene, 43)
        symbol_id = 1052 if frame_no == 50 else 1011
        rect = self.root_button_rects.get((frame_no, symbol_id))
        return rect.copy() if rect is not None else FloatRect(0, 365, 105, 35)

    def _fit_canvas(self, screen: pygame.Surface, canvas: pygame.Surface) -> None:
        rect = self._screen_rect(screen.get_size())
        screen.fill((0, 0, 0))
        key = (id(canvas), rect.w, rect.h)
        scaled = pygame.transform.smoothscale(canvas, rect.size)
        screen.blit(scaled, rect)

    @staticmethod
    def _screen_rect(screen_size: tuple[int, int]) -> pygame.Rect:
        width, height = screen_size
        scale = min(width / REFERENCE_SIZE[0], height / REFERENCE_SIZE[1])
        draw_w = max(1, round(REFERENCE_SIZE[0] * scale))
        draw_h = max(1, round(REFERENCE_SIZE[1] * scale))
        return pygame.Rect((width - draw_w) // 2, (height - draw_h) // 2, draw_w, draw_h)

    @classmethod
    def _screen_to_reference(
        cls,
        pos: tuple[int, int],
        screen_size: tuple[int, int],
    ) -> tuple[float, float] | None:
        rect = cls._screen_rect(screen_size)
        if not rect.collidepoint(pos):
            return None
        return (
            (pos[0] - rect.x) * REFERENCE_SIZE[0] / rect.w,
            (pos[1] - rect.y) * REFERENCE_SIZE[1] / rect.h,
        )

    def _font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (max(8, size * EXPORT_SCALE), bool(bold))
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached
        if self.futura_font_path.is_file():
            cached = pygame.font.Font(str(self.futura_font_path), key[0])
            cached.set_bold(key[1])
        else:
            cached = pygame.font.SysFont("futura", key[0], bold=key[1])
        self._font_cache[key] = cached
        return cached

    def _draw_text(
        self,
        canvas: pygame.Surface,
        text: str,
        x: float,
        y: float,
        size: int,
        color: tuple[int, int, int],
        bold: bool = False,
    ) -> None:
        image = self._font(size, bold).render(text, True, color)
        canvas.blit(image, (round(x * EXPORT_SCALE), round(y * EXPORT_SCALE - image.get_height() / 2)))

    def _draw_center_text(
        self,
        canvas: pygame.Surface,
        text: str,
        x: float,
        y: float,
        size: int,
        color: tuple[int, int, int],
        bold: bool = False,
    ) -> None:
        image = self._font(size, bold).render(text, True, color)
        canvas.blit(
            image,
            (
                round(x * EXPORT_SCALE - image.get_width() / 2),
                round(y * EXPORT_SCALE - image.get_height() / 2),
            ),
        )

    @staticmethod
    def _format_time(seconds: int) -> str:
        if seconds <= 0:
            return "none"
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    @staticmethod
    def _key_name(key: int) -> str:
        if key == 0:
            return "N/A"
        name = pygame.key.name(key).upper()
        aliases = {
            "LEFT SHIFT": "SHIFT",
            "RIGHT SHIFT": "SHIFT",
            "LEFT CTRL": "CTRL",
            "RIGHT CTRL": "CTRL",
            "RETURN": "ENTER",
            "ESCAPE": "ESC",
            "PAGEUP": "PGUP",
            "PAGE UP": "PGUP",
            "PAGEDOWN": "PGDN",
            "PAGE DOWN": "PGDN",
            "INSERT": "INS",
            "DELETE": "DEL",
            "CAPSLOCK": "CAPS",
            "NUMLOCK": "NUMLCK",
            "SCROLLLOCK": "SCRLK",
            "PAUSE": "BREAK",
            "KEYPAD 0": "NUM 0",
            "KEYPAD 1": "NUM 1",
            "KEYPAD 2": "NUM 2",
            "[0]": "NUM 0",
            "[1]": "NUM 1",
            "[2]": "NUM 2",
        }
        return aliases.get(name, name)


class MatchResults:
    """Main timeline frames 101-170 plus the dynamic AS2 winner/stat fields."""

    def __init__(self, root: Path, manifest: dict) -> None:
        self.root = root
        self.data = manifest.get("results", {})
        self.frame_root = root / "assets/menu/end_frames"
        self.frames: OrderedDict[int, pygame.Surface] = OrderedDict()
        self.frame_data = {int(item["frame"]): item for item in self.data.get("frames", [])}
        self.elapsed_ms = 0
        self.entries: list[dict[str, object]] = []
        self.ranking: list[dict[str, object]] = []
        self.winner_player = 1
        self.limit_mode = "stock"
        self.game_type = "vsmode"
        self.killed_players = 0
        self.game_time_seconds = 0
        system_futura = Path("/System/Library/Fonts/Supplemental/Futura.ttc")
        self.futura_font = system_futura if system_futura.is_file() else None
        self.fighter_images: dict[str, LazySurfaceSequence] = {}
        self.fighter_offsets: dict[str, list[pygame.Vector2]] = {}
        for fighter_name, fighter_data in manifest.get("fighters", {"PeachPlayer": manifest["fighter"]}).items():
            image_paths = []
            offsets = []
            color_states = fighter_data.get("color_state_animations", {})
            for color_frame in range(1, 5):
                still = color_states.get(str(color_frame), fighter_data["state_animations"])["still"]
                first_frame = still["frames"][0]
                image_paths.append(root / first_frame["image"])
                state_offset = still.get("state_offset", {"x": 0, "y": 0})
                local_offset = first_frame.get("offset", {"x": 0, "y": 0})
                offsets.append(
                    pygame.Vector2(
                        float(state_offset["x"]) + float(local_offset["x"]),
                        float(state_offset["y"]) + float(local_offset["y"]),
                    )
                )
            self.fighter_images[fighter_name] = LazySurfaceSequence(image_paths)
            self.fighter_offsets[fighter_name] = offsets

    def _frame(self, index: int) -> pygame.Surface:
        if index in self.frames:
            frame = self.frames.pop(index)
            self.frames[index] = frame
            return frame
        frame = pygame.image.load(str(self.frame_root / f"{index}.png")).convert_alpha()
        self.frames[index] = frame
        while len(self.frames) > 3:
            self.frames.popitem(last=False)
        return frame

    @property
    def frame_number(self) -> int:
        start = int(self.data.get("start_frame", 101))
        stop = int(self.data.get("stop_frame", 170))
        rate = float(self.data.get("frame_rate", 30))
        return min(stop, start + int(self.elapsed_ms * rate / 1000))

    def start(
        self,
        fighters: list[object],
        winner: object | None,
        limit_mode: str = "stock",
        game_type: str = "vsmode",
        killed_players: int = 0,
        game_time_seconds: int = 0,
    ) -> None:
        self.elapsed_ms = 0
        self.limit_mode = limit_mode
        self.game_type = game_type
        self.killed_players = int(killed_players)
        self.game_time_seconds = int(game_time_seconds)
        self.entries = []
        for index, fighter in enumerate(fighters):
            self.entries.append(
                {
                    "player": index + 1,
                    "kos": int(getattr(fighter, "kos", 0)),
                    "sds": int(getattr(fighter, "sds", 0)),
                    "deaths": int(getattr(fighter, "deaths", 0)),
                    "lives": int(getattr(fighter, "lives", 0)),
                    "dead": bool(getattr(fighter, "dead", False)),
                    "death_order": getattr(fighter, "death_order", None),
                    "winner": fighter is winner,
                    "color_frame": int(getattr(fighter, "color_frame", index + 1)),
                    "fighter_name": str(getattr(fighter, "fighter_name", "PeachPlayer")),
                }
            )
        if self.game_type == "endurance":
            self.ranking = self.entries[:1]
        elif self.limit_mode == "time":
            self.ranking = sorted(
                self.entries,
                key=lambda item: int(item["kos"]) - int(item["deaths"]),
                reverse=True,
            )
        else:
            assigned_orders = [
                int(item["death_order"])
                for item in self.entries
                if item["death_order"] is not None
            ]
            next_death_order = max(assigned_orders, default=-1) + 1
            for entry in self.entries:
                if entry["death_order"] is None:
                    entry["death_order"] = next_death_order
            self.ranking = sorted(
                self.entries,
                key=lambda item: int(item["death_order"]),
                reverse=True,
            )
        for rank, entry in enumerate(self.ranking, start=1):
            entry["rank"] = rank
        winning = self.ranking[0] if self.ranking else None
        self.winner_player = int(winning["player"]) if winning else 1

    def update(self, elapsed_ms: int) -> None:
        self.elapsed_ms += max(0, elapsed_ms)

    def handle_event(self, event: pygame.event.Event, screen_size: tuple[int, int]) -> MenuAction | None:
        if event.type != pygame.MOUSEBUTTONUP or event.button != 1:
            return None
        if self.frame_number < int(self.data.get("stop_frame", 170)):
            return None
        ref_pos = MainMenu._screen_to_reference(event.pos, screen_size)
        if ref_pos is not None:
            button = self.data.get("main_button", {"x": 485.33, "y": 372.55, "w": 101.28, "h": 16.96})
            if pygame.Rect(button["x"], button["y"], button["w"], button["h"]).collidepoint(ref_pos):
                return MenuAction("return_main")
            button = self.data.get("more_games_button", {"x": 8.35, "y": 367.2, "w": 160.5, "h": 23.5})
            if pygame.Rect(button["x"], button["y"], button["w"], button["h"]).collidepoint(ref_pos):
                return MenuAction("open_url", {"url": str(self.data.get("more_games_url", "http://www.armorgames.com/"))})
        return None

    def draw(self, screen: pygame.Surface) -> None:
        frame_no = self.frame_number
        canvas = self._frame(frame_no).copy()
        if frame_no >= int(self.data.get("stop_frame", 170)):
            self._clear_dynamic_stats(canvas)
        if frame_no >= int(self.data.get("podium_start_frame", 143)):
            self._draw_podium_fighters(canvas)
        if frame_no >= int(self.data.get("stop_frame", 170)):
            self._draw_stats(canvas)
        self._draw_winner_text(canvas, frame_no)
        rect = MainMenu._screen_rect(screen.get_size())
        screen.fill((0, 0, 0))
        screen.blit(pygame.transform.smoothscale(canvas, rect.size), rect)

    def _draw_winner_text(self, canvas: pygame.Surface, frame_no: int) -> None:
        frame = self.frame_data.get(frame_no, {})
        pos = frame.get("winner_text_pos", {"x": 300, "y": 36.65})
        text_data = self.data.get("winner_text", {})
        font_size = max(8, round(float(text_data.get("font_size", 52)) * EXPORT_SCALE))
        font = pygame.font.Font(str(self.futura_font), font_size) if self.futura_font else pygame.font.SysFont(
            str(text_data.get("font", "futura")), font_size, bold=True
        )
        font.set_bold(True)
        color = tuple(text_data.get("color", [255, 255, 0]))
        winner_text = (
            f"{self.killed_players} KILLS!"
            if self.game_type == "endurance"
            else f"PLAYER {self.winner_player} WINS!"
        )
        image = font.render(winner_text, True, color)
        canvas.blit(
            image,
            (
                round(float(pos["x"]) * EXPORT_SCALE - image.get_width() / 2),
                round(float(pos["y"]) * EXPORT_SCALE - image.get_height() / 2),
            ),
        )

    def _clear_dynamic_stats(self, canvas: pygame.Surface) -> None:
        clean = self._frame(160)
        stats = self.data.get("stats", {})
        origin = stats.get("container", {"x": 310, "y": 151})
        legend = stats.get("legend", {"x": -255, "y": -52, "width": 105.45, "height": 135.7})
        areas = [
            pygame.Rect(
                round((origin["x"] + legend["x"]) * EXPORT_SCALE),
                round((origin["y"] + legend["y"]) * EXPORT_SCALE),
                round(legend["width"] * EXPORT_SCALE),
                round(legend["height"] * EXPORT_SCALE),
            )
        ]
        for column in stats.get("columns", [-149.25, -47.25, 54.75, 156.75]):
            areas.append(
                pygame.Rect(
                    round((origin["x"] + column) * EXPORT_SCALE),
                    round((origin["y"] - 79.65) * EXPORT_SCALE),
                    round(102 * EXPORT_SCALE),
                    round(163.3 * EXPORT_SCALE),
                )
            )
        for area in areas:
            canvas.blit(clean, area, area)

    def _draw_podium_fighters(self, canvas: pygame.Surface) -> None:
        if self.game_type == "endurance":
            if not self.entries:
                return
            entry = self.entries[0]
            color_index = max(0, min(3, int(entry.get("color_frame", 1)) - 1))
            fighter_name = str(entry.get("fighter_name", "PeachPlayer"))
            source = self.fighter_images.get(fighter_name, self.fighter_images["PeachPlayer"])[color_index]
            offset = self.fighter_offsets.get(fighter_name, self.fighter_offsets["PeachPlayer"])[color_index]
            fighter_scale = 4.0
            image = pygame.transform.smoothscale(
                source,
                (
                    max(1, round(source.get_width() * fighter_scale)),
                    max(1, round(source.get_height() * fighter_scale)),
                ),
            )
            canvas.blit(
                image,
                (
                    round(300 * EXPORT_SCALE + offset.x * fighter_scale * EXPORT_SCALE),
                    round(380 * EXPORT_SCALE + offset.y * fighter_scale * EXPORT_SCALE),
                ),
            )
            return
        slots = self.data.get("podium_slots", {})
        fighter_scale = float(self.data.get("fighter_scale", 2.5))
        for rank, entry in enumerate(self.ranking[:4], start=1):
            slot = slots.get(f"p{rank}")
            if not slot:
                continue
            color_index = max(0, min(3, int(entry.get("color_frame", 1)) - 1))
            fighter_name = str(entry.get("fighter_name", "PeachPlayer"))
            images = self.fighter_images.get(fighter_name, self.fighter_images["PeachPlayer"])
            offsets = self.fighter_offsets.get(fighter_name, self.fighter_offsets["PeachPlayer"])
            source = images[color_index]
            offset = offsets[color_index]
            image = pygame.transform.smoothscale(
                source,
                (
                    max(1, round(source.get_width() * fighter_scale)),
                    max(1, round(source.get_height() * fighter_scale)),
                ),
            )
            draw_x = float(slot["x"]) * EXPORT_SCALE + offset.x * fighter_scale * EXPORT_SCALE
            draw_y = float(slot["y"]) * EXPORT_SCALE + offset.y * fighter_scale * EXPORT_SCALE
            canvas.blit(image, (round(draw_x), round(draw_y)))

    def _draw_stats(self, canvas: pygame.Surface) -> None:
        stats = self.data.get("stats", {})
        origin = stats.get("container", {"x": 310, "y": 151})
        font_name = str(stats.get("font", "futura"))
        font_size = max(8, round(float(stats.get("font_size", 20)) * EXPORT_SCALE))
        font = pygame.font.Font(str(self.futura_font), font_size) if self.futura_font else pygame.font.SysFont(
            font_name, font_size, bold=True
        )
        font.set_bold(True)
        line_step = float(stats.get("line_step", 22))

        def draw_lines(lines: list[str], center_x: float, top_y: float, centered: bool) -> None:
            for line_no, line in enumerate(lines):
                if not line:
                    continue
                image = font.render(line, True, (255, 255, 255))
                x = center_x * EXPORT_SCALE - image.get_width() / 2 if centered else center_x * EXPORT_SCALE
                canvas.blit(image, (round(x), round((top_y + line_no * line_step) * EXPORT_SCALE)))

        if self.game_type == "endurance":
            legend_lines = ["Kills", "", "Deaths", "", "Time"]
        elif self.limit_mode == "time":
            legend_lines = ["Rank", "KOs", "Deaths", "", "Score"]
        else:
            legend_lines = ["Rank", "", "KOs", "", "Suicides"]
        draw_lines(legend_lines, origin["x"] - 255, origin["y"] - 52, False)
        rank_by_player = {int(item["player"]): int(item["rank"]) for item in self.ranking}
        columns = stats.get("columns", [-149.25, -47.25, 54.75, 156.75])
        for index, entry in enumerate(self.entries[:4]):
            if self.game_type == "endurance" and index > 0:
                continue
            center_x = origin["x"] + float(columns[index]) + 51
            if self.game_type == "endurance":
                seconds = self.game_time_seconds % 60
                minutes = (self.game_time_seconds - seconds) // 60
                lines = [
                    "P1",
                    str(self.killed_players),
                    "",
                    str(entry["deaths"]),
                    "",
                    f"{minutes}:{seconds:02d}",
                ]
            elif self.limit_mode == "time":
                lines = [
                    f"P{entry['player']}",
                    str(rank_by_player[int(entry["player"])]),
                    f"+{entry['kos']}",
                    f"-{entry['deaths']}",
                    "",
                    str(int(entry["kos"]) - int(entry["deaths"])),
                ]
            else:
                lines = [
                    f"P{entry['player']}",
                    str(rank_by_player[int(entry["player"])]),
                    "",
                    str(entry["kos"]),
                    "",
                    str(entry["sds"]),
                ]
            draw_lines(lines, center_x, origin["y"] - 79.65, True)
