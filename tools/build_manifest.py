from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT.parent / "glorton_peach"


SYMBOLS = {
    "peach_player": 538,
    "peach_pose": 422,
    "mine": 54,
    "grenade": 56,
    "bullet": 483,
    "rocket": 425,
    "pencil": 70,
    "poop": 259,
    "garbage": 361,
    "energy_ball": 594,
    "puff": 399,
    "spawn2": 541,
    "spawn1": 553,
    "player_death": 581,
    "punch_damage": 584,
    "rooftop": 721,
    "mogadishu": 827,
    "b52": 868,
    "space": 881,
    "camera_trick": 782,
    "pos_indicator": 787,
    "far_indicator": 842,
    "item_indicator": 791,
    "boom_star": 564,
    "boom_wave": 567,
    "boom_matter": 569,
    "shield": 847,
    "osd_bigicon": 738,
    "osd_life_graphic": 771,
    "osd_score_upper": 776,
    "osd_damage": 780,
    "rooftop_background": 727,
    "mogadishu_background": 833,
    "b52_background": 870,
    "space_background": 896,
    "platform_probe": 717,
    "spawn_h": 719,
    "spawn_point": 720,
}

HIDDEN_STAGE_PREFIXES = ("Fixed", "Moving", "Spawn", "AI_")
DYNAMIC_STAGE_CHARACTER_IDS = {711}
TEAM_COLOR_LABELS = {"red", "blue", "green", "orange"}
MENU_FIGHTERS = {
    "AuberginePlayer": 107,
    "SBLPlayer": 200,
    "CoffeePlayer": 295,
    "TrashPlayer": 402,
    "PeachPlayer": 538,
    "DefaultPlayer": 680,
}
MENU_FIGHTER_ORDER = (
    "SBLPlayer",
    "PeachPlayer",
    "TrashPlayer",
    "CoffeePlayer",
    "DefaultPlayer",
    "AuberginePlayer",
)
FIGHTER_CONFIGS = {
    "AuberginePlayer": {
        "slug": "aubergine",
        "character_name": "AubergineLock",
        "weight": 0.5,
        "speed": 0.6,
        "power": 0.4,
        "special_kind": "pencil",
    },
    "SBLPlayer": {
        "slug": "strawberry",
        "character_name": "StrawberryLock",
        "weight": 0.5,
        "speed": 0.5,
        "power": 0.5,
        "special_kind": "kamehameha",
    },
    "CoffeePlayer": {
        "slug": "coffee",
        "character_name": "CoffeeLock",
        "weight": 0.4,
        "speed": 0.7,
        "power": 0.4,
        "special_kind": "poop",
    },
    "TrashPlayer": {
        "slug": "trash",
        "character_name": "TrashLock",
        "weight": 0.6,
        "speed": 0.4,
        "power": 0.5,
        "special_kind": "garbage",
    },
    "PeachPlayer": {
        "slug": "peach",
        "character_name": "PeachLock",
        "weight": 0.4,
        "speed": 0.5,
        "power": 0.6,
        "special_kind": "peach_weapons",
    },
    "DefaultPlayer": {
        "slug": "default",
        "character_name": "BallLock",
        "weight": 0.5,
        "speed": 0.5,
        "power": 0.5,
        "special_kind": "electric",
    },
}


SOURCE_REFS = {
    "fighter": "raw_ffdec_export_scripts/scripts/__Packages/Fighter.as",
    "peach": "raw_ffdec_export_scripts/scripts/__Packages/PeachPlayer.as",
    "bullet": "raw_ffdec_export_scripts/scripts/__Packages/Bullet.as",
    "stage_init": "raw_ffdec_export_scripts/scripts/frame_51/DoAction.as",
    "xml": "raw_ffdec_xml/fight-for-glorton.xml",
}


def natural_key(path: Path) -> list[object]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", path.stem)]


def load_symbols(source: Path) -> dict[str, int]:
    path = source / "raw_ffdec_export/symbolClass/symbols.csv"
    symbols: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.reader(f, delimiter=";"):
            if len(row) >= 2:
                symbols[row[1].strip('"')] = int(row[0])
    return symbols


def find_sprite(root: ET.Element, sprite_id: int) -> ET.Element:
    sid = str(sprite_id)
    for item in root.iter("item"):
        if item.attrib.get("type") == "DefineSpriteTag" and item.attrib.get("spriteId") == sid:
            return item
    raise ValueError(f"DefineSpriteTag spriteId={sprite_id} not found")


def find_button(root: ET.Element, button_id: int) -> ET.Element:
    sid = str(button_id)
    for item in root.iter("item"):
        if item.attrib.get("type") == "DefineButton2Tag" and item.attrib.get("buttonId") == sid:
            return item
    raise ValueError(f"DefineButton2Tag buttonId={button_id} not found")


def extract_sprite_timeline(sprite: ET.Element) -> dict[str, object]:
    frame = 1
    labels: list[dict[str, object]] = []
    named_places: list[dict[str, object]] = []
    for item in sprite.iter("item"):
        typ = item.attrib.get("type")
        if typ == "FrameLabelTag":
            labels.append({"frame": frame, "name": item.attrib.get("name", "")})
        elif typ and typ.startswith("PlaceObject") and item.attrib.get("placeFlagHasName") == "true":
            matrix = item.find("matrix")
            named_places.append(
                {
                    "frame": frame,
                    "name": item.attrib.get("name", ""),
                    "character_id": int(item.attrib.get("characterId", "0")),
                    "depth": int(item.attrib.get("depth", "0")),
                    "matrix": matrix_to_dict(matrix),
                }
            )
        elif typ == "ShowFrameTag":
            frame += 1
    return {"frame_count": int(sprite.attrib.get("frameCount", "0")), "labels": labels, "named_places": named_places}


def matrix_to_dict(matrix: ET.Element | None) -> dict[str, float]:
    if matrix is None:
        return {"x": 0.0, "y": 0.0, "scale_x": 1.0, "scale_y": 1.0, "rotate_skew0": 0.0, "rotate_skew1": 0.0}
    has_scale = matrix.attrib.get("hasScale") == "true"
    has_rotate = matrix.attrib.get("hasRotate") == "true"
    return {
        "x": int(matrix.attrib.get("translateX", "0")) / 20.0,
        "y": int(matrix.attrib.get("translateY", "0")) / 20.0,
        "scale_x": float(matrix.attrib.get("scaleX", "1") or 1) if has_scale else 1.0,
        "scale_y": float(matrix.attrib.get("scaleY", "1") or 1) if has_scale else 1.0,
        "rotate_skew0": float(matrix.attrib.get("rotateSkew0", "0") or 0) if has_rotate else 0.0,
        "rotate_skew1": float(matrix.attrib.get("rotateSkew1", "0") or 0) if has_rotate else 0.0,
    }


def transformed_bounds(
    bounds: tuple[float, float, float, float],
    matrix: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = bounds
    a, b, c, d, tx, ty = matrix
    points = (
        (a * x0 + b * y0 + tx, c * x0 + d * y0 + ty),
        (a * x1 + b * y0 + tx, c * x1 + d * y0 + ty),
        (a * x0 + b * y1 + tx, c * x0 + d * y1 + ty),
        (a * x1 + b * y1 + tx, c * x1 + d * y1 + ty),
    )
    return (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )


def compose_matrices(
    parent: tuple[float, float, float, float, float, float],
    child: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    pa, pb, pc, pd, ptx, pty = parent
    ca, cb, cc, cd, ctx, cty = child
    return (
        pa * ca + pb * cc,
        pa * cb + pb * cd,
        pc * ca + pd * cc,
        pc * cb + pd * cd,
        pa * ctx + pb * cty + ptx,
        pc * ctx + pd * cty + pty,
    )


def button_hit_bounds(
    source: Path,
    root: ET.Element,
    button_id: int,
) -> tuple[float, float, float, float]:
    button = find_button(root, button_id)
    bounds_cache: dict[tuple[int, int, int], tuple[float, float, float, float] | None] = {}
    timeline_cache: dict[int, tuple[int, set[str]]] = {}
    rects = []
    characters = button.find("characters")
    for record in characters or []:
        if record.attrib.get("buttonStateHitTest") != "true":
            continue
        child = symbol_bounds(
            source,
            root,
            int(record.attrib.get("characterId", "0")),
            1,
            3,
            bounds_cache,
            timeline_cache,
        )
        if child is not None:
            rects.append(transformed_bounds(child, matrix_values(record.find("placeMatrix"))))
    if not rects:
        raise ValueError(f"button {button_id} has no renderable HitTest record")
    return (
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    )


def rect_dict(bounds: tuple[float, float, float, float]) -> dict[str, float]:
    x0, y0, x1, y1 = bounds
    return {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0}


def image_size(path: Path) -> dict[str, int]:
    with Image.open(path) as im:
        return {"w": im.width, "h": im.height}


def export_fighter_frames(source: Path, fighter_id: int, slug: str) -> list[dict[str, object]]:
    sprite_root = source / "raw_ffdec_export/sprites"
    prefix = f"DefineSprite_{fighter_id}_"
    candidates = sorted(
        path
        for path in sprite_root.iterdir()
        if path.is_dir() and (path.name == f"DefineSprite_{fighter_id}" or path.name.startswith(prefix))
    )
    if not candidates:
        return []
    src = candidates[0]
    dst = ROOT / f"assets/fighters/{slug}/timeline"
    dst.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, object]] = []
    for path in sorted(src.glob("*.png"), key=natural_key):
        frame_no = int(path.stem)
        with Image.open(path).convert("RGBA") as im:
            bbox = im.getbbox()
            raw_name = f"{frame_no:03d}_raw.png"
            shutil.copy2(path, dst / raw_name)
            cropped_name = f"{frame_no:03d}.png"
            if bbox:
                crop = im.crop(bbox)
                crop.save(dst / cropped_name)
            else:
                crop = im
                crop.save(dst / cropped_name)
                bbox = (0, 0, im.width, im.height)
            frames.append(
                {
                    "frame": frame_no,
                    "raw": f"assets/fighters/{slug}/timeline/{raw_name}",
                    "image": f"assets/fighters/{slug}/timeline/{cropped_name}",
                    "raw_size": {"w": im.width, "h": im.height},
                    "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2] - bbox[0], "h": bbox[3] - bbox[1]},
                    "cropped_size": {"w": crop.width, "h": crop.height},
                }
            )
    return frames


def export_peach_frames(source: Path) -> list[dict[str, object]]:
    return export_fighter_frames(source, SYMBOLS["peach_player"], "peach")


def export_sprite_frames(
    source: Path,
    sprite_id: int,
    dst: Path,
    raster_root: Path | None = None,
    render_scale: int = 1,
    root: ET.Element | None = None,
    color_frame: int = 3,
) -> list[dict[str, object]]:
    sprite_root = (raster_root / "sprites") if raster_root is not None else (source / "raw_ffdec_export/sprites")
    prefix = f"DefineSprite_{sprite_id}_"
    candidates = sorted(
        path
        for path in sprite_root.iterdir()
        if path.is_dir() and (path.name == f"DefineSprite_{sprite_id}" or path.name.startswith(prefix))
    )
    if not candidates:
        return []
    src = candidates[0]
    dst.mkdir(parents=True, exist_ok=True)
    for stale in dst.glob("*.png"):
        stale.unlink()
    frame_bounds: dict[int, tuple[float, float, float, float]] = {}
    union_bounds: tuple[float, float, float, float] | None = None
    if root is not None:
        bounds_cache: dict[tuple[int, int, int], tuple[float, float, float, float] | None] = {}
        timeline_cache: dict[int, tuple[int, set[str]]] = {}
        for path in sorted(src.glob("*.png"), key=natural_key):
            frame_no = int(path.stem)
            bounds = symbol_bounds(
                source,
                root,
                sprite_id,
                frame_no,
                color_frame,
                bounds_cache,
                timeline_cache,
            )
            if bounds is None:
                continue
            frame_bounds[frame_no] = bounds
            if union_bounds is None:
                union_bounds = bounds
            else:
                union_bounds = (
                    min(union_bounds[0], bounds[0]),
                    min(union_bounds[1], bounds[1]),
                    max(union_bounds[2], bounds[2]),
                    max(union_bounds[3], bounds[3]),
                )
    frames: list[dict[str, object]] = []
    scale = max(1, int(render_scale))
    for path in sorted(src.glob("*.png"), key=natural_key):
        frame_no = int(path.stem)
        with Image.open(path).convert("RGBA") as im:
            bbox = im.getbbox() or (0, 0, im.width, im.height)
            raw_name = f"{frame_no:03d}_raw.png"
            shutil.copy2(path, dst / raw_name)
            cropped_name = f"{frame_no:03d}.png"
            crop = im.crop(bbox)
            crop.save(dst / cropped_name)
            offset = None
            if union_bounds is not None:
                offset = {
                    "x": union_bounds[0] + bbox[0] / scale,
                    "y": union_bounds[1] + bbox[1] / scale,
                }
            elif frame_no in frame_bounds:
                offset = {"x": frame_bounds[frame_no][0], "y": frame_bounds[frame_no][1]}
            frames.append(
                {
                    "frame": frame_no,
                    "raw": str((dst / raw_name).relative_to(ROOT)),
                    "image": str((dst / cropped_name).relative_to(ROOT)),
                    "render_scale": scale,
                    **({"offset": offset} if offset is not None else {}),
                    "raw_size": {"w": im.width, "h": im.height},
                    "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2] - bbox[0], "h": bbox[3] - bbox[1]},
                    "cropped_size": {"w": crop.width, "h": crop.height},
                    "logical_size": {"w": crop.width / scale, "h": crop.height / scale},
                }
            )
    return frames


def extract_timeline_playback(source: Path, sprite_id: int) -> dict[str, int]:
    scripts_root = source / "raw_ffdec_export_scripts/scripts"
    candidates = sorted(
        path
        for path in scripts_root.glob("DefineSprite_*")
        if path.is_dir()
        and path.name.removeprefix("DefineSprite_").split("_", 1)[0].isdigit()
        and int(path.name.removeprefix("DefineSprite_").split("_", 1)[0]) == sprite_id
    )
    playback: dict[str, int] = {}
    if not candidates:
        return playback
    for script in candidates[0].glob("frame_*/DoAction*.as"):
        match = re.search(r"frame_(\d+)", script.parent.name)
        if not match:
            continue
        frame_no = int(match.group(1))
        text = script.read_text(encoding="utf-8", errors="replace")
        loop = re.search(r"gotoAndPlay\((\d+)\)", text)
        if loop:
            playback["loop_from"] = int(loop.group(1))
            playback["loop_at"] = frame_no
        if re.search(r"\bstop\(\)", text):
            playback["stop_at"] = frame_no
    return playback


def matrix_values(matrix: ET.Element | None) -> tuple[float, float, float, float, float, float]:
    if matrix is None:
        return 1.0, 0.0, 0.0, 1.0, 0.0, 0.0
    has_scale = matrix.attrib.get("hasScale") == "true"
    has_rotate = matrix.attrib.get("hasRotate") == "true"
    a = float(matrix.attrib.get("scaleX", "1") or 1) if has_scale else 1.0
    d = float(matrix.attrib.get("scaleY", "1") or 1) if has_scale else 1.0
    # SWF matrices apply RotateSkew1 to x' and RotateSkew0 to y'.
    b = float(matrix.attrib.get("rotateSkew1", "0") or 0) if has_rotate else 0.0
    c = float(matrix.attrib.get("rotateSkew0", "0") or 0) if has_rotate else 0.0
    tx = int(matrix.attrib.get("translateX", "0")) / 20.0
    ty = int(matrix.attrib.get("translateY", "0")) / 20.0
    return a, b, c, d, tx, ty


def sprite_subtags(root: ET.Element, sprite_id: int) -> list[ET.Element]:
    sprite = find_sprite(root, sprite_id)
    sub_tags = sprite.find("subTags")
    if sub_tags is None:
        return []
    return list(sub_tags)


def build_display_frames(root: ET.Element, sprite_id: int) -> list[list[dict[str, object]]]:
    display: dict[int, dict[str, object]] = {}
    frames: list[list[dict[str, object]]] = []
    frame_no = 1
    for tag in sprite_subtags(root, sprite_id):
        typ = tag.attrib.get("type")
        if typ and typ.startswith("PlaceObject"):
            depth = int(tag.attrib.get("depth", "0"))
            current = dict(display.get(depth, {}))
            character_id = int(tag.attrib.get("characterId", "0"))
            if tag.attrib.get("placeFlagHasCharacter") == "true" and character_id > 0:
                current["character_id"] = character_id
                current["start_frame"] = frame_no
            if tag.attrib.get("placeFlagHasName") == "true" and tag.attrib.get("name"):
                current["name"] = tag.attrib.get("name", "")
            if tag.find("matrix") is not None:
                current["matrix"] = matrix_values(tag.find("matrix"))
            if tag.find("colorTransform") is not None:
                current["color_transform"] = color_transform_values(tag.find("colorTransform"))
            display[depth] = current
        elif typ and typ.startswith("RemoveObject"):
            display.pop(int(tag.attrib.get("depth", "0")), None)
        elif typ == "ShowFrameTag":
            frames.append([{"depth": depth, **item} for depth, item in sorted(display.items())])
            frame_no += 1
    return frames


def build_main_display_frames(root: ET.Element) -> list[list[dict[str, object]]]:
    tags = root.find("tags")
    if tags is None:
        return []
    display: dict[int, dict[str, object]] = {}
    frames: list[list[dict[str, object]]] = []
    frame_no = 1
    for tag in tags:
        typ = tag.attrib.get("type")
        if typ and typ.startswith("PlaceObject"):
            depth = int(tag.attrib.get("depth", "0"))
            current = dict(display.get(depth, {}))
            character_id = int(tag.attrib.get("characterId", "0"))
            if tag.attrib.get("placeFlagHasCharacter") == "true" and character_id > 0:
                current["character_id"] = character_id
                current["start_frame"] = frame_no
            if tag.attrib.get("placeFlagHasName") == "true" and tag.attrib.get("name"):
                current["name"] = tag.attrib.get("name", "")
            if tag.find("matrix") is not None:
                current["matrix"] = matrix_values(tag.find("matrix"))
            display[depth] = current
        elif typ and typ.startswith("RemoveObject"):
            display.pop(int(tag.attrib.get("depth", "0")), None)
        elif typ == "ShowFrameTag":
            frames.append([{"depth": depth, **item} for depth, item in sorted(display.items())])
            frame_no += 1
    return frames


def color_transform_values(transform: ET.Element) -> dict[str, float]:
    return {
        "red_mult": int(transform.attrib.get("redMultTerm", "256")) / 256.0,
        "green_mult": int(transform.attrib.get("greenMultTerm", "256")) / 256.0,
        "blue_mult": int(transform.attrib.get("blueMultTerm", "256")) / 256.0,
        "alpha_mult": int(transform.attrib.get("alphaMultTerm", "256")) / 256.0,
        "red_add": int(transform.attrib.get("redAddTerm", "0")),
        "green_add": int(transform.attrib.get("greenAddTerm", "0")),
        "blue_add": int(transform.attrib.get("blueAddTerm", "0")),
        "alpha_add": int(transform.attrib.get("alphaAddTerm", "0")),
    }


def apply_color_transform(image: Image.Image, transform: dict[str, float] | None) -> Image.Image:
    if not transform:
        return image
    channels = image.split()
    channel_specs = (
        ("red", channels[0]),
        ("green", channels[1]),
        ("blue", channels[2]),
        ("alpha", channels[3]),
    )
    adjusted = []
    for name, channel in channel_specs:
        mult = transform[f"{name}_mult"]
        add = transform[f"{name}_add"]
        adjusted.append(channel.point(lambda value, mult=mult, add=add: max(0, min(255, round(value * mult + add)))))
    return Image.merge("RGBA", adjusted)


def symbol_raster_path(
    source: Path,
    character_id: int,
    frame_no: int,
    raster_root: Path | None = None,
) -> Path | None:
    sprite_root = (raster_root / "sprites") if raster_root is not None else (source / "raw_ffdec_export/sprites")
    prefix = f"DefineSprite_{character_id}_"
    sprite_dirs = sorted(
        path
        for path in sprite_root.iterdir()
        if path.is_dir() and (path.name == f"DefineSprite_{character_id}" or path.name.startswith(prefix))
    )
    if sprite_dirs:
        exact = sprite_dirs[0] / f"{frame_no}.png"
        return exact if exact.exists() else sprite_dirs[0] / "1.png"
    shape = (
        raster_root / f"shapes/{character_id}.png"
        if raster_root is not None
        else source / f"raw_ffdec_export/shapes/{character_id}.png"
    )
    if shape.exists():
        return shape
    image = (
        raster_root / f"images/{character_id}.png"
        if raster_root is not None
        else source / f"raw_ffdec_export/images/{character_id}.png"
    )
    if image.exists():
        return image
    if raster_root is not None:
        return symbol_raster_path(source, character_id, frame_no, None)
    return None


def transform_image_with_local_bounds(
    image: Image.Image,
    local_bounds: tuple[float, float, float, float],
    matrix: tuple[float, float, float, float, float, float],
    render_scale: int = 1,
) -> tuple[Image.Image, tuple[int, int]]:
    x0, y0, x1, y1 = local_bounds
    a, b, c, d, tx, ty = matrix
    points = [
        (a * x0 + b * y0 + tx, c * x0 + d * y0 + ty),
        (a * x1 + b * y0 + tx, c * x1 + d * y0 + ty),
        (a * x0 + b * y1 + tx, c * x0 + d * y1 + ty),
        (a * x1 + b * y1 + tx, c * x1 + d * y1 + ty),
    ]
    scale = max(1, int(render_scale))
    left = math.floor(min(point[0] for point in points) * scale)
    top = math.floor(min(point[1] for point in points) * scale)
    right = math.ceil(max(point[0] for point in points) * scale)
    bottom = math.ceil(max(point[1] for point in points) * scale)
    width = max(1, right - left)
    height = max(1, bottom - top)

    det = a * d - b * c
    if abs(det) < 1e-8:
        det = 1.0
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det
    scale_u = image.width / max(1e-8, x1 - x0)
    scale_v = image.height / max(1e-8, y1 - y0)
    coeffs = (
        inv_a * scale_u / scale,
        inv_b * scale_u / scale,
        (inv_a * (left / scale - tx) + inv_b * (top / scale - ty) - x0) * scale_u,
        inv_c * scale_v / scale,
        inv_d * scale_v / scale,
        (inv_c * (left / scale - tx) + inv_d * (top / scale - ty) - y0) * scale_v,
    )
    transformed = image.transform(
        (width, height),
        Image.Transform.AFFINE,
        coeffs,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0, 0),
    )
    return transformed, (left, top)


def export_rooftop_foreground(
    source: Path,
    root: ET.Element,
    dst: Path,
    canvas_size: tuple[int, int],
    render_scale: int = 1,
    raster_root: Path | None = None,
) -> dict[str, int]:
    placements = build_display_frames(root, SYMBOLS["rooftop"])[0]
    bounds_cache: dict[tuple[int, int, int], tuple[float, float, float, float] | None] = {}
    render_cache: dict[
        tuple[int, int, int, int],
        tuple[Image.Image, tuple[float, float, float, float]] | None,
    ] = {}
    timeline_cache: dict[int, tuple[int, set[str]]] = {}
    scale = max(1, int(render_scale))
    canvas = Image.new("RGBA", (canvas_size[0] * scale, canvas_size[1] * scale), (0, 0, 0, 0))
    for place in placements:
        character_id = int(place.get("character_id", 0))
        name = str(place.get("name", ""))
        if (
            character_id <= 0
            or character_id in DYNAMIC_STAGE_CHARACTER_IDS
            or any(name.startswith(prefix) for prefix in HIDDEN_STAGE_PREFIXES)
        ):
            continue
        rendered = render_symbol_frame(
            source,
            root,
            character_id,
            1,
            3,
            bounds_cache,
            render_cache,
            render_scale=scale,
            raster_root=raster_root,
            timeline_cache=timeline_cache,
        )
        if rendered is None:
            continue
        image, bounds = rendered
        image = apply_color_transform(image, place.get("color_transform"))
        transformed, pos = transform_image_with_local_bounds(
            image,
            bounds,
            place.get("matrix", (1, 0, 0, 1, 0, 0)),
            scale,
        )
        canvas.alpha_composite(transformed, pos)
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return {"w": canvas_size[0], "h": canvas_size[1], "render_scale": scale}


def export_rooftop_helicopter_frames(
    source: Path,
    root: ET.Element,
    dst: Path,
    render_scale: int = 1,
    raster_root: Path | None = None,
) -> dict[str, object]:
    placements_by_frame = build_display_frames(root, SYMBOLS["rooftop"])
    bounds_cache: dict[tuple[int, int, int], tuple[float, float, float, float] | None] = {}
    render_cache: dict[
        tuple[int, int, int, int],
        tuple[Image.Image, tuple[float, float, float, float]] | None,
    ] = {}
    timeline_cache: dict[int, tuple[int, set[str]]] = {}
    scale = max(1, int(render_scale))
    dst.mkdir(parents=True, exist_ok=True)
    frames: list[dict[str, object]] = []
    for frame_no, placements in enumerate(placements_by_frame, start=1):
        place = next(
            (
                item
                for item in placements
                if int(item.get("character_id", 0)) in DYNAMIC_STAGE_CHARACTER_IDS
            ),
            None,
        )
        if place is None:
            continue
        character_id = int(place["character_id"])
        rendered = render_symbol_frame(
            source,
            root,
            character_id,
            1,
            3,
            bounds_cache,
            render_cache,
            render_scale=scale,
            raster_root=raster_root,
            timeline_cache=timeline_cache,
        )
        if rendered is None:
            continue
        image, bounds = rendered
        image = apply_color_transform(image, place.get("color_transform"))
        transformed, pos = transform_image_with_local_bounds(
            image,
            bounds,
            place.get("matrix", (1, 0, 0, 1, 0, 0)),
            scale,
        )
        bbox = transformed.getbbox()
        if bbox is None:
            continue
        crop = transformed.crop(bbox)
        filename = f"{frame_no:03d}.png"
        crop.save(dst / filename)
        frames.append(
            {
                "frame": frame_no,
                "image": str((dst / filename).relative_to(ROOT)),
                "render_scale": scale,
                "offset": {"x": (pos[0] + bbox[0]) / scale, "y": (pos[1] + bbox[1]) / scale},
                "size": {"w": crop.width, "h": crop.height},
                "logical_size": {"w": crop.width / scale, "h": crop.height / scale},
            }
        )
    return {"frame_rate": 30, "frames": frames}


def export_rooftop_moving_platforms(root: ET.Element, platform_size: dict[str, int]) -> dict[str, object]:
    platforms: dict[str, list[dict[str, object]]] = {}
    for frame_no, placements in enumerate(build_display_frames(root, SYMBOLS["rooftop"]), start=1):
        for place in placements:
            name = str(place.get("name", ""))
            if not name.startswith("Moving"):
                continue
            character_id = int(place.get("character_id", 0))
            if character_id != SYMBOLS["platform_probe"]:
                continue
            a, _b, _c, d, x, y = place.get("matrix", (1, 0, 0, 1, 0, 0))
            platforms.setdefault(name, []).append(
                {
                    "frame": frame_no,
                    "depth": int(place.get("depth", 0)),
                    "rect": {
                        "x": x,
                        "y": y,
                        "w": platform_size["w"] * abs(a),
                        "h": platform_size["h"] * abs(d),
                    },
                }
            )
    return {"frame_rate": 30, "platforms": platforms}


def transformed_bbox(size: tuple[int, int], matrix: tuple[float, float, float, float, float, float]) -> tuple[int, int, int, int]:
    w, h = size
    a, b, c, d, tx, ty = matrix
    points = [
        (tx, ty),
        (a * w + tx, c * w + ty),
        (b * h + tx, d * h + ty),
        (a * w + b * h + tx, c * w + d * h + ty),
    ]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return math.floor(min(xs)), math.floor(min(ys)), math.ceil(max(xs)), math.ceil(max(ys))


def transform_image(
    image: Image.Image,
    matrix: tuple[float, float, float, float, float, float],
    local_offset: tuple[float, float] = (0.0, 0.0),
) -> tuple[Image.Image, tuple[int, int]]:
    a, b, c, d, tx, ty = matrix
    ox, oy = local_offset
    tx = tx + a * ox + b * oy
    ty = ty + c * ox + d * oy
    left, top, right, bottom = transformed_bbox(image.size, matrix)
    width = max(1, right - left)
    height = max(1, bottom - top)
    det = a * d - b * c
    if abs(det) < 1e-8:
        det = 1.0
    inv_a = d / det
    inv_b = -b / det
    inv_c = -c / det
    inv_d = a / det
    inv_tx = (b * (top - ty) - d * (left - tx)) / det
    inv_ty = (c * (left - tx) - a * (top - ty)) / det
    transformed = image.transform(
        (width, height),
        Image.Transform.AFFINE,
        (inv_a, inv_b, inv_tx, inv_c, inv_d, inv_ty),
        resample=Image.Resampling.NEAREST,
    )
    return transformed, (left, top)


def shape_bounds(root: ET.Element, character_id: int) -> tuple[float, float, float, float] | None:
    sid = str(character_id)
    for item in root.iter("item"):
        if item.attrib.get("shapeId") == sid:
            bounds = item.find("shapeBounds")
            if bounds is None:
                return None
            return (
                int(bounds.attrib.get("Xmin", "0")) / 20.0,
                int(bounds.attrib.get("Ymin", "0")) / 20.0,
                int(bounds.attrib.get("Xmax", "0")) / 20.0,
                int(bounds.attrib.get("Ymax", "0")) / 20.0,
            )
    return None


def sprite_timeline_info(
    root: ET.Element,
    character_id: int,
    cache: dict[int, tuple[int, set[str]]],
) -> tuple[int, set[str]]:
    if character_id in cache:
        return cache[character_id]
    try:
        timeline = extract_sprite_timeline(find_sprite(root, character_id))
    except ValueError:
        result = (1, set())
    else:
        result = (
            max(1, int(timeline["frame_count"])),
            {str(label["name"]) for label in timeline["labels"]},
        )
    cache[character_id] = result
    return result


def resolve_child_frame(
    root: ET.Element,
    name: str,
    character_id: int,
    color_frame: int,
    parent_frame: int,
    start_frame: int,
    timeline_cache: dict[int, tuple[int, set[str]]],
    hand_frame: int = 1,
    child_frame_overrides: dict[str, int] | None = None,
) -> int:
    frame_count, labels = sprite_timeline_info(root, character_id, timeline_cache)
    if labels and labels.issubset(TEAM_COLOR_LABELS):
        return min(color_frame, frame_count)
    if name in {"f1", "f2", "f3", "f4", "f5"}:
        return min(color_frame, frame_count)
    if child_frame_overrides and name in child_frame_overrides:
        return min(max(1, int(child_frame_overrides[name])), frame_count)
    if name == "hand":
        # The outer hand clip uses 1/2/3 for empty/mine/grenade. Its nested
        # team-color hand is handled by the labeled-timeline branch above.
        return min(max(1, hand_frame), frame_count)
    # Mine/Grenade frame 1 runs gotoAndStop("airbone"). When either symbol is
    # embedded in a hand clip, Flash executes that script immediately.
    if character_id == SYMBOLS["mine"]:
        return 8
    if character_id == SYMBOLS["grenade"]:
        return 10
    if frame_count > 1 and not labels:
        return ((max(1, parent_frame) - max(1, start_frame)) % frame_count) + 1
    return 1


def raster_symbol_frame(
    source: Path,
    root: ET.Element,
    character_id: int,
    frame_no: int,
    color_frame: int,
    bounds_cache: dict[tuple[int, int, int], tuple[float, float, float, float] | None],
    raster_root: Path | None = None,
    timeline_cache: dict[int, tuple[int, set[str]]] | None = None,
    hand_frame: int = 1,
    child_frame_overrides: dict[str, int] | None = None,
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    if timeline_cache is None:
        timeline_cache = {}
    path = symbol_raster_path(source, character_id, frame_no, raster_root)
    bounds = symbol_bounds(
        source,
        root,
        character_id,
        frame_no,
        color_frame,
        bounds_cache,
        timeline_cache,
        hand_frame,
        child_frame_overrides,
    )
    if path is None or bounds is None:
        return None
    with Image.open(path).convert("RGBA") as image:
        bbox = image.getbbox()
        if bbox is None:
            return None
        return image.crop(bbox), bounds


def render_symbol_frame(
    source: Path,
    root: ET.Element,
    character_id: int,
    frame_no: int,
    color_frame: int,
    bounds_cache: dict[tuple[int, int, int], tuple[float, float, float, float] | None],
    render_cache: dict[tuple[int, int, int, int], tuple[Image.Image, tuple[float, float, float, float]] | None],
    render_scale: int = 1,
    raster_root: Path | None = None,
    timeline_cache: dict[int, tuple[int, set[str]]] | None = None,
    hand_frame: int = 1,
    child_frame_overrides: dict[str, int] | None = None,
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    if timeline_cache is None:
        timeline_cache = {}
    key = (character_id, frame_no, color_frame, render_scale)
    if key in render_cache:
        cached = render_cache[key]
        if cached is None:
            return None
        return cached[0].copy(), cached[1]

    try:
        sprite = find_sprite(root, character_id)
        timeline = extract_sprite_timeline(sprite)
        label_names = {label["name"] for label in timeline["labels"]}
        if label_names and label_names.issubset(TEAM_COLOR_LABELS):
            rendered = raster_symbol_frame(
                source, root, character_id, frame_no, color_frame, bounds_cache, raster_root, timeline_cache, hand_frame,
                child_frame_overrides,
            )
            render_cache[key] = rendered
            if rendered is None:
                return None
            return rendered[0].copy(), rendered[1]
        frames = build_display_frames(root, character_id)
    except ValueError:
        frames = []

    if frames:
        placements = frames[min(max(frame_no, 1), len(frames)) - 1]
        rendered = render_display_frame_with_bounds(
            source,
            root,
            placements,
            color_frame,
            bounds_cache,
            render_cache,
            parent_frame=frame_no,
            render_scale=render_scale,
            raster_root=raster_root,
            timeline_cache=timeline_cache,
            hand_frame=hand_frame,
            child_frame_overrides=child_frame_overrides,
        )
        render_cache[key] = rendered
        if rendered is None:
            return None
        return rendered[0].copy(), rendered[1]

    rendered = raster_symbol_frame(
        source, root, character_id, frame_no, color_frame, bounds_cache, raster_root, timeline_cache, hand_frame,
        child_frame_overrides,
    )
    if rendered is None:
        render_cache[key] = None
        return None
    render_cache[key] = rendered
    return rendered[0].copy(), rendered[1]


def symbol_bounds(
    source: Path,
    root: ET.Element,
    character_id: int,
    frame_no: int,
    color_frame: int,
    cache: dict[tuple[int, int, int], tuple[float, float, float, float] | None],
    timeline_cache: dict[int, tuple[int, set[str]]] | None = None,
    hand_frame: int = 1,
    child_frame_overrides: dict[str, int] | None = None,
) -> tuple[float, float, float, float] | None:
    if timeline_cache is None:
        timeline_cache = {}
    key = (character_id, frame_no, color_frame)
    if key in cache:
        return cache[key]
    direct = shape_bounds(root, character_id)
    if direct is not None:
        cache[key] = direct
        return direct
    try:
        frames = build_display_frames(root, character_id)
    except ValueError:
        cache[key] = None
        return None
    if not frames:
        cache[key] = None
        return None
    placements = frames[min(max(frame_no, 1), len(frames)) - 1]
    rects = []
    for place in placements:
        cid = int(place.get("character_id", 0))
        if cid <= 0:
            continue
        child_frame = resolve_child_frame(
            root,
            str(place.get("name", "")),
            cid,
            color_frame,
            frame_no,
            int(place.get("start_frame", 1)),
            timeline_cache,
            hand_frame,
            child_frame_overrides,
        )
        child_bounds = symbol_bounds(
            source, root, cid, child_frame, color_frame, cache, timeline_cache, hand_frame,
            child_frame_overrides,
        )
        if child_bounds is None:
            continue
        x0, y0, x1, y1 = child_bounds
        matrix = place.get("matrix", (1, 0, 0, 1, 0, 0))
        a, b, c, d, tx, ty = matrix
        points = [
            (a * x0 + b * y0 + tx, c * x0 + d * y0 + ty),
            (a * x1 + b * y0 + tx, c * x1 + d * y0 + ty),
            (a * x0 + b * y1 + tx, c * x0 + d * y1 + ty),
            (a * x1 + b * y1 + tx, c * x1 + d * y1 + ty),
        ]
        rects.append((min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points)))
    if not rects:
        cache[key] = None
        return None
    result = (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )
    cache[key] = result
    return result


def render_display_frame_with_bounds(
    source: Path,
    root: ET.Element,
    placements: list[dict[str, object]],
    color_frame: int,
    bounds_cache: dict[tuple[int, int, int], tuple[float, float, float, float] | None] | None = None,
    render_cache: dict[tuple[int, int, int, int], tuple[Image.Image, tuple[float, float, float, float]] | None] | None = None,
    parent_frame: int = 1,
    render_scale: int = 1,
    raster_root: Path | None = None,
    timeline_cache: dict[int, tuple[int, set[str]]] | None = None,
    hand_frame: int = 1,
    hidden_names: set[str] | None = None,
    child_frame_overrides: dict[str, int] | None = None,
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    prepared = []
    if bounds_cache is None:
        bounds_cache = {}
    if render_cache is None:
        render_cache = {}
    if timeline_cache is None:
        timeline_cache = {}
    for place in placements:
        character_id = int(place.get("character_id", 0))
        name = str(place.get("name", ""))
        if character_id <= 0 or (hidden_names and name in hidden_names):
            continue
        frame_no = resolve_child_frame(
            root,
            name,
            character_id,
            color_frame,
            parent_frame,
            int(place.get("start_frame", 1)),
            timeline_cache,
            hand_frame,
            child_frame_overrides,
        )
        rendered = render_symbol_frame(
            source,
            root,
            character_id,
            frame_no,
            color_frame,
            bounds_cache,
            render_cache,
            render_scale,
            raster_root,
            timeline_cache,
            hand_frame,
            child_frame_overrides,
        )
        if rendered is None:
            continue
        image, bounds = rendered
        image = apply_color_transform(image, place.get("color_transform"))
        transformed, pos = transform_image_with_local_bounds(
            image,
            bounds,
            place.get("matrix", (1, 0, 0, 1, 0, 0)),
            render_scale,
        )
        bbox = transformed.getbbox()
        if bbox is None:
            continue
        prepared.append((transformed.crop(bbox), (pos[0] + bbox[0], pos[1] + bbox[1])))
    if not prepared:
        return None
    left = min(pos[0] for _, pos in prepared)
    top = min(pos[1] for _, pos in prepared)
    right = max(pos[0] + img.width for img, pos in prepared)
    bottom = max(pos[1] + img.height for img, pos in prepared)
    canvas = Image.new("RGBA", (max(1, right - left), max(1, bottom - top)), (0, 0, 0, 0))
    for image, pos in prepared:
        canvas.alpha_composite(image, (pos[0] - left, pos[1] - top))
    scale = max(1, int(render_scale))
    return canvas, (left / scale, top / scale, right / scale, bottom / scale)


def render_display_frame(source: Path, root: ET.Element, placements: list[dict[str, object]], color_frame: int) -> Image.Image:
    rendered = render_display_frame_with_bounds(source, root, placements, color_frame)
    if rendered is None:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    return rendered[0]


def export_composited_sprite_frames(
    source: Path,
    root: ET.Element,
    sprite_id: int,
    dst: Path,
    color_frame: int = 3,
    render_scale: int = 1,
    raster_root: Path | None = None,
    hand_frame: int = 1,
    hidden_names: set[str] | None = None,
    child_frame_overrides: dict[str, int] | None = None,
) -> list[dict[str, object]]:
    dst.mkdir(parents=True, exist_ok=True)
    for stale in dst.glob("*.png"):
        stale.unlink()
    frames: list[dict[str, object]] = []
    bounds_cache: dict[tuple[int, int, int], tuple[float, float, float, float] | None] = {}
    render_cache: dict[
        tuple[int, int, int, int],
        tuple[Image.Image, tuple[float, float, float, float]] | None,
    ] = {}
    timeline_cache: dict[int, tuple[int, set[str]]] = {}
    scale = max(1, int(render_scale))
    for frame_no, placements in enumerate(build_display_frames(root, sprite_id), start=1):
        rendered = render_display_frame_with_bounds(
            source,
            root,
            placements,
            color_frame,
            bounds_cache,
            render_cache,
            parent_frame=frame_no,
            render_scale=scale,
            raster_root=raster_root,
            timeline_cache=timeline_cache,
            hand_frame=hand_frame,
            hidden_names=hidden_names,
            child_frame_overrides=child_frame_overrides,
        )
        if rendered is None:
            image = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
            logical_bounds = (0.0, 0.0, 1.0 / scale, 1.0 / scale)
        else:
            image, logical_bounds = rendered
        bbox = image.getbbox() or (0, 0, image.width, image.height)
        raw_name = f"{frame_no:03d}_raw.png"
        image.save(dst / raw_name)
        cropped_name = f"{frame_no:03d}.png"
        crop = image.crop(bbox)
        crop.save(dst / cropped_name)
        offset_x = logical_bounds[0] + bbox[0] / scale
        offset_y = logical_bounds[1] + bbox[1] / scale
        frames.append(
            {
                "frame": frame_no,
                "raw": str((dst / raw_name).relative_to(ROOT)),
                "image": str((dst / cropped_name).relative_to(ROOT)),
                "render_scale": scale,
                "offset": {"x": offset_x, "y": offset_y},
                "raw_size": {"w": image.width, "h": image.height},
                "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2] - bbox[0], "h": bbox[3] - bbox[1]},
                "cropped_size": {"w": crop.width, "h": crop.height},
                "logical_size": {"w": crop.width / scale, "h": crop.height / scale},
            }
        )
    return frames


def fighter_attack_profiles(fighter_name: str) -> dict[str, dict[str, object]]:
    attacks: dict[str, dict[str, object]] = {
        "punchGround": {"damage": 5, "throw_power": 3, "angle": 45},
        "punchRun": {"damage": 5, "throw_power": 3, "angle": 45},
        "koAttack": {"damage": 5, "throw_power": 3, "angle": 45},
        "punchUp": {"damage": 10, "throw_power": 6, "angle": 90},
        "punchAir": {"damage": 10, "throw_power": 6, "angle": 80, "extra_yinc": -5},
    }
    if fighter_name == "PeachPlayer":
        attacks.update(
            {
                "specialGround": {
                    "damage": 10,
                    "throw_power": 3,
                    "angle": 45,
                    "spawns": "Bullet",
                    "spawn_frame_min": 5,
                },
                "specialAir": {
                    "damage": 10,
                    "throw_power": 3,
                    "angle": 45,
                    "spawns": "Bullet",
                    "spawn_frame_min": 5,
                },
                "specialUp": {"spawns": "Rocket", "spawn_frame_min": 15},
            }
        )
    elif fighter_name == "AuberginePlayer":
        attacks.update(
            {
                "specialGround": {"spawns": "Pencil", "spawn_frame_min": 9},
                "specialAir": {"spawns": "Pencil", "spawn_frame_min": 9},
                "specialUp": {
                    "damage": 20,
                    "throw_power": 4,
                    "angle": 85,
                    "minimum_vertical_gap": 50,
                },
            }
        )
    elif fighter_name == "SBLPlayer":
        attacks.update(
            {
                "specialGround": {
                    "kind": "kamehameha",
                    "active_frame_min": 9,
                    "active_frame_max": 14,
                    "distance_min": 20,
                    "distance_max": 240,
                },
                "specialAir": {
                    "kind": "kamehameha",
                    "active_frame_min": 9,
                    "active_frame_max": 14,
                    "distance_min": 20,
                    "distance_max": 240,
                },
                "specialUp": {"damage": 20, "throw_power": 4, "angle": 45},
            }
        )
    elif fighter_name == "CoffeePlayer":
        attacks.update(
            {
                "specialGround": {"spawns": "Poop", "spawn_frame_min": 9},
                "specialAir": {"spawns": "Poop", "spawn_frame_min": 9},
                "specialUp": {"damage": 20, "throw_power": 4, "angle": 45},
            }
        )
    elif fighter_name == "TrashPlayer":
        attacks.update(
            {
                "specialGround": {"spawns": "Garbage", "spawn_frame_min": 6},
                "specialAir": {"spawns": "Garbage", "spawn_frame_min": 6},
                "specialUp": {"spawns": "GarbageBurst", "spawn_frame_min": 11},
            }
        )
    elif fighter_name == "DefaultPlayer":
        attacks.update(
            {
                "specialGround": {
                    "damage": 10,
                    "throw_power": 2.5,
                    "angle": 45,
                    "electrocuted_ms": 100,
                    "spawns": "EnergyBall",
                    "spawn_at_end": True,
                },
                "specialAir": {
                    "damage": 10,
                    "throw_power": 2.5,
                    "angle": 45,
                    "electrocuted_ms": 100,
                    "spawns": "EnergyBall",
                    "spawn_at_end": True,
                },
                "specialUp": {
                    "damage": 20,
                    "throw_power": 5,
                    "angle": 45,
                    "active_frame_min": 11,
                    "maximum_target_y_offset": 5,
                    "electrocuted_ms": 300,
                },
            }
        )
    return attacks


def fighter_special_motion(fighter_name: str) -> dict[str, object]:
    if fighter_name == "PeachPlayer":
        return {
            "slow_before": 7,
            "slow_factor": 0.5,
            "rise_from": 7,
            "rise_through": 10,
            "rise_yinc": -6,
            "projectile_from": 15,
            "projectile_yinc": -5,
        }
    if fighter_name == "AuberginePlayer":
        return {"slow_through": 7, "slow_factor": 0.2, "rise_from": 8, "rise_yinc": -9}
    if fighter_name in {"SBLPlayer", "CoffeePlayer"}:
        return {"slow_through": 2, "slow_factor": 0.5, "rise_from": 3, "rise_yinc": -8}
    return {"slow_before": 10, "slow_factor": 0.5, "rise_from": 10, "rise_yinc": -6}


def export_fighter_bundle(
    source: Path,
    root: ET.Element,
    fighter_name: str,
    raster_root: Path,
) -> dict[str, object]:
    fighter_id = MENU_FIGHTERS[fighter_name]
    config = FIGHTER_CONFIGS[fighter_name]
    slug = str(config["slug"])
    timeline = extract_sprite_timeline(find_sprite(root, fighter_id))
    display_frames = build_display_frames(root, fighter_id)
    state_templates: dict[str, dict[str, object]] = {}
    for label_item in timeline["labels"]:
        label = str(label_item["name"])
        frame_no = int(label_item["frame"])
        placements = display_frames[frame_no - 1] if 0 < frame_no <= len(display_frames) else []
        if not placements:
            continue
        place = next((item for item in placements if item.get("name")), placements[0])
        sprite_id = int(place["character_id"])
        matrix = place.get("matrix", (1, 0, 0, 1, 0, 0))
        if isinstance(matrix, dict):
            state_offset = {"x": float(matrix["x"]), "y": float(matrix["y"])}
        else:
            state_offset = {"x": float(matrix[4]), "y": float(matrix[5])}
        state_templates[label] = {
            "symbol_id": sprite_id,
            "placed_name": str(place.get("name", "")),
            "timeline_frame": frame_no,
            "timeline": extract_sprite_timeline(find_sprite(root, sprite_id)),
            "playback": extract_timeline_playback(source, sprite_id),
            "state_offset": state_offset,
        }

    color_state_animations: dict[str, dict[str, dict[str, object]]] = {}
    for color_frame in range(1, 5):
        states = {}
        for label, template in state_templates.items():
            frames = export_composited_sprite_frames(
                source,
                root,
                int(template["symbol_id"]),
                ROOT / f"assets/fighters/{slug}/colors/{color_frame}/states/{label}",
                color_frame=color_frame,
                render_scale=4,
                raster_root=raster_root,
            )
            states[label] = {
                **template,
                "color_frame": color_frame,
                "frame_count": len(frames),
                "frames": frames,
            }
            fired_names = {
                str(place.get("name", ""))
                for place in template["timeline"].get("named_places", [])
                if str(place.get("name", "")) in {"bullet", "rocket", "pencil", "poop", "garbage"}
            }
            if fired_names:
                states[label]["fired_frames"] = export_composited_sprite_frames(
                    source,
                    root,
                    int(template["symbol_id"]),
                    ROOT / f"assets/fighters/{slug}/colors/{color_frame}/states/{label}_fired",
                    color_frame=color_frame,
                    render_scale=4,
                    raster_root=raster_root,
                    hidden_names=fired_names,
                )
                states[label]["fired_hidden_names"] = sorted(fired_names)
        color_state_animations[str(color_frame)] = states

    held_item_animations: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for item_name, hand_frame in (("mine", 2), ("grenade", 3)):
        item_colors = {}
        for color_frame in range(1, 5):
            states = {}
            for label, animation in color_state_animations[str(color_frame)].items():
                named_places = animation["timeline"].get("named_places", [])
                if not any(place.get("name") == "hand" for place in named_places):
                    continue
                frames = export_composited_sprite_frames(
                    source,
                    root,
                    int(animation["symbol_id"]),
                    ROOT / f"assets/fighters/{slug}/held/{item_name}/colors/{color_frame}/states/{label}",
                    color_frame=color_frame,
                    render_scale=4,
                    raster_root=raster_root,
                    hand_frame=hand_frame,
                )
                states[label] = {
                    **animation,
                    "held_item": item_name,
                    "hand_frame": hand_frame,
                    "frame_count": len(frames),
                    "frames": frames,
                }
            item_colors[str(color_frame)] = states
        held_item_animations[item_name] = item_colors

    garbage_variant_animations: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    if fighter_name == "TrashPlayer":
        for variant in range(1, 7):
            variant_colors = {}
            for color_frame in range(1, 5):
                states = {}
                for label in ("specialGround", "specialAir"):
                    animation = color_state_animations[str(color_frame)].get(label)
                    if animation is None:
                        continue
                    frames = export_composited_sprite_frames(
                        source,
                        root,
                        int(animation["symbol_id"]),
                        ROOT / f"assets/fighters/{slug}/garbage/{variant}/colors/{color_frame}/states/{label}",
                        color_frame=color_frame,
                        render_scale=4,
                        raster_root=raster_root,
                        child_frame_overrides={"garbage": variant},
                    )
                    states[label] = {
                        **animation,
                        "garbage_variant": variant,
                        "frame_count": len(frames),
                        "frames": frames,
                    }
                variant_colors[str(color_frame)] = states
            garbage_variant_animations[str(variant)] = variant_colors

    return {
        "name": fighter_name,
        "symbol_id": fighter_id,
        "class": fighter_name,
        "character_name": config["character_name"],
        "slug": slug,
        "weight": config["weight"],
        "speed": config["speed"],
        "power": config["power"],
        "base_move_xinc": 8 * float(config["speed"]),
        "jump_yinc": -9,
        "gravity_per_tick": 0.5,
        "max_fall_yinc": 6,
        "special_kind": config["special_kind"],
        "special_up_motion": fighter_special_motion(fighter_name),
        "timeline": timeline,
        "frames": export_fighter_frames(source, fighter_id, slug),
        "state_animations": color_state_animations["3"],
        "color_state_animations": color_state_animations,
        "held_item_state_animations": held_item_animations,
        "garbage_variant_state_animations": garbage_variant_animations,
        "attacks": fighter_attack_profiles(fighter_name),
        "source_class": f"raw_ffdec_export_scripts/scripts/__Packages/{fighter_name}.as",
    }


def copy_single(src: Path, dst: Path) -> dict[str, int]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return image_size(dst)


def build(source: Path) -> dict[str, object]:
    output_path = ROOT / "assets/manifests/glorton_manifest.json"
    try:
        previous_manifest = json.loads(output_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        previous_manifest = {}
    xml_path = source / SOURCE_REFS["xml"]
    root = ET.parse(xml_path).getroot()
    swf_meta = {
        "frame_count": int(root.attrib.get("frameCount", "0")),
        "frame_rate": float(root.attrib.get("frameRate", "0") or 0),
    }
    symbols = load_symbols(source)
    peach_timeline = extract_sprite_timeline(find_sprite(root, SYMBOLS["peach_player"]))
    rooftop_timeline = extract_sprite_timeline(find_sprite(root, SYMBOLS["rooftop"]))
    main_display_frames = build_main_display_frames(root)
    result_frame_data = []
    for frame_no in range(101, 171):
        placements = main_display_frames[frame_no - 1] if frame_no <= len(main_display_frames) else []
        winner = next((item for item in placements if item.get("name") == "WinMC"), None)
        winner_matrix = winner.get("matrix", (1, 0, 0, 1, 300, 36.65)) if winner else (1, 0, 0, 1, 300, 36.65)
        result_frame_data.append(
            {
                "frame": frame_no,
                "image": f"assets/menu/end_frames/{frame_no}.png",
                "winner_text_pos": {"x": winner_matrix[4], "y": winner_matrix[5]},
            }
        )
    podium_placements = main_display_frames[142] if len(main_display_frames) >= 143 else []
    podium_slots = {}
    for item in podium_placements:
        name = str(item.get("name", ""))
        if name not in {"p1", "p2", "p3", "p4"}:
            continue
        matrix = item.get("matrix", (1, 0, 0, 1, 0, 0))
        podium_slots[name] = {"x": matrix[4], "y": matrix[5]}
    peach_raster_root = ROOT / "assets/ffdec_zoom4"
    if not peach_raster_root.exists():
        raise SystemExit(
            "Missing 4x Peach source assets. Run tools/export_highres_peach.py before building the manifest."
        )

    background_source = symbol_raster_path(source, SYMBOLS["rooftop_background"], 1, peach_raster_root)
    if background_source is None:
        raise SystemExit("Missing 4x Rooftop background export.")
    copy_single(
        background_source,
        ROOT / "assets/stages/rooftop/background.png",
    )
    bg_size = {"w": 1287, "h": 638, "render_scale": 4}
    foreground_size = export_rooftop_foreground(
        source,
        root,
        ROOT / "assets/stages/rooftop/foreground.png",
        (bg_size["w"], 800),
        render_scale=4,
        raster_root=peach_raster_root,
    )
    helicopter = export_rooftop_helicopter_frames(
        source,
        root,
        ROOT / "assets/stages/rooftop/helicopter",
        render_scale=4,
        raster_root=peach_raster_root,
    )
    bullet_source = symbol_raster_path(source, SYMBOLS["bullet"], 1, peach_raster_root)
    rocket_source = symbol_raster_path(source, SYMBOLS["rocket"], 1, peach_raster_root)
    if bullet_source is None or rocket_source is None:
        raise SystemExit("Missing 4x Peach projectile exports.")
    bullet_bounds = symbol_bounds(source, root, SYMBOLS["bullet"], 1, 3, {})
    rocket_bounds = symbol_bounds(source, root, SYMBOLS["rocket"], 1, 3, {})
    if bullet_bounds is None or rocket_bounds is None:
        raise SystemExit("Missing Peach projectile bounds.")
    bullet_size = copy_single(
        bullet_source,
        ROOT / "assets/projectiles/bullet/bullet.png",
    )
    rocket_size = copy_single(
        rocket_source,
        ROOT / "assets/projectiles/rocket/rocket.png",
    )
    special_projectile_frames = {}
    special_projectile_bounds = {}
    for projectile_name, symbol_name, slug in (
        ("Pencil", "pencil", "pencil"),
        ("Poop", "poop", "poop"),
        ("Garbage", "garbage", "garbage"),
        ("EnergyBall", "energy_ball", "energy_ball"),
    ):
        symbol_id = SYMBOLS[symbol_name]
        special_projectile_frames[projectile_name] = export_sprite_frames(
            source,
            symbol_id,
            ROOT / f"assets/projectiles/{slug}",
            root=root,
            raster_root=peach_raster_root,
            render_scale=4,
        )
        bounds = symbol_bounds(source, root, symbol_id, 1, 3, {})
        if bounds is None:
            raise SystemExit(f"Missing {projectile_name} projectile bounds.")
        special_projectile_bounds[projectile_name] = bounds
    spawn1_frames = export_sprite_frames(
        source, SYMBOLS["spawn1"], ROOT / "assets/effects/spawn1", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    spawn2_frames = export_sprite_frames(
        source, SYMBOLS["spawn2"], ROOT / "assets/effects/spawn2", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    puff_frames = export_sprite_frames(
        source, SYMBOLS["puff"], ROOT / "assets/effects/puff", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    player_death_frames = export_sprite_frames(
        source,
        SYMBOLS["player_death"],
        ROOT / "assets/effects/player_death",
        root=root, raster_root=peach_raster_root, render_scale=4,
    )
    punch_damage_frames = export_sprite_frames(
        source,
        SYMBOLS["punch_damage"],
        ROOT / "assets/effects/punch_damage",
        raster_root=peach_raster_root,
        render_scale=4,
    )
    camera_trick_frames = export_sprite_frames(
        source, SYMBOLS["camera_trick"], ROOT / "assets/effects/camera_trick", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    pos_indicator_frames = export_sprite_frames(
        source, SYMBOLS["pos_indicator"], ROOT / "assets/effects/pos_indicator", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    far_indicator_frames = export_sprite_frames(
        source, SYMBOLS["far_indicator"], ROOT / "assets/effects/far_indicator", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    # symbol_bounds cannot include DefineEditText 835/837/839/841. The raw
    # 136x196 canvas starts at the full field/arrow union (-16, -90), and its
    # alpha crop starts at (19, 28) in the 4x export.
    for frame in far_indicator_frames:
        frame["offset"] = {"x": -11.25, "y": -83.0}
    shield_frames = export_sprite_frames(
        source, SYMBOLS["shield"], ROOT / "assets/effects/shield", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    osd_bigicon_frames = export_sprite_frames(
        source, SYMBOLS["osd_bigicon"], ROOT / "assets/ui/osd_bigicon", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    osd_bigicon_timeline = extract_sprite_timeline(find_sprite(root, SYMBOLS["osd_bigicon"]))
    osd_life_frames = export_sprite_frames(
        source, SYMBOLS["osd_life_graphic"], ROOT / "assets/ui/osd_life_graphic", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    osd_life_timeline = extract_sprite_timeline(find_sprite(root, SYMBOLS["osd_life_graphic"]))
    osd_life_frames_by_label = {
        str(item["name"]): int(item["frame"])
        for item in osd_life_timeline["labels"]
    }
    osd_life_color_exports = {}
    for color_frame in range(1, 5):
        osd_life_color_exports[str(color_frame)] = export_composited_sprite_frames(
            source,
            root,
            SYMBOLS["osd_life_graphic"],
            ROOT / f"assets/ui/osd_life_graphic/color_{color_frame}",
            color_frame=color_frame,
            render_scale=4,
            raster_root=peach_raster_root,
        )
    # OSD.UpdateLives first selects CharacterName on the outer timeline, then
    # Player.Colr on its nested Embed. Preserve both dimensions explicitly.
    osd_life_character_frames = {}
    for fighter_config in FIGHTER_CONFIGS.values():
        character_name = str(fighter_config["character_name"])
        frame_no = osd_life_frames_by_label[character_name]
        osd_life_character_frames[character_name] = [
            osd_life_color_exports[str(color_frame)][frame_no - 1]
            for color_frame in range(1, 5)
        ]
    osd_peach_life_frames = osd_life_character_frames["PeachLock"]
    osd_damage_frames = export_sprite_frames(
        source, SYMBOLS["osd_damage"], ROOT / "assets/ui/osd_damage", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    osd_score_upper_frames = export_sprite_frames(
        source, SYMBOLS["osd_score_upper"], ROOT / "assets/ui/osd_score_upper", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    osd_score_upper_timeline = extract_sprite_timeline(
        find_sprite(root, SYMBOLS["osd_score_upper"])
    )
    mine_frames = export_sprite_frames(
        source, SYMBOLS["mine"], ROOT / "assets/items/mine", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    grenade_frames = export_sprite_frames(
        source, SYMBOLS["grenade"], ROOT / "assets/items/grenade", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    item_indicator_frames = export_sprite_frames(
        source, SYMBOLS["item_indicator"], ROOT / "assets/effects/item_indicator", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    boom_star_frames = export_sprite_frames(
        source, SYMBOLS["boom_star"], ROOT / "assets/effects/boom_star", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    boom_wave_frames = export_sprite_frames(
        source, SYMBOLS["boom_wave"], ROOT / "assets/effects/boom_wave", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    boom_matter_frames = export_sprite_frames(
        source, SYMBOLS["boom_matter"], ROOT / "assets/effects/boom_matter", root=root,
        raster_root=peach_raster_root, render_scale=4,
    )
    peach_frames = export_peach_frames(source)
    peach_labels_by_frame = {item["frame"]: item["name"] for item in peach_timeline["labels"]}
    peach_display_frames = build_display_frames(root, SYMBOLS["peach_player"])
    peach_state_animations = {}
    for place in peach_timeline["named_places"]:
        label = peach_labels_by_frame.get(place["frame"], place["name"])
        if label == "still" and place["name"] == "stil":
            label = "still"
        sprite_id = int(place["character_id"])
        state_frames = export_composited_sprite_frames(
            source,
            root,
            sprite_id,
            ROOT / f"assets/fighters/peach/states/{label}",
            color_frame=3,
            render_scale=4,
            raster_root=peach_raster_root,
        )
        state_timeline = extract_sprite_timeline(find_sprite(root, sprite_id))
        fired_names = {
            str(item.get("name", ""))
            for item in state_timeline.get("named_places", [])
            if str(item.get("name", "")) in {"bullet", "rocket", "pencil", "poop", "garbage"}
        }
        peach_state_animations[label] = {
            "symbol_id": sprite_id,
            "placed_name": place["name"],
            "timeline_frame": place["frame"],
            "frame_count": len(state_frames),
            "timeline": state_timeline,
            "playback": extract_timeline_playback(source, sprite_id),
            "state_offset": {"x": place["matrix"]["x"], "y": place["matrix"]["y"]},
            "frames": state_frames,
        }
        if fired_names:
            peach_state_animations[label]["fired_frames"] = export_composited_sprite_frames(
                source,
                root,
                sprite_id,
                ROOT / f"assets/fighters/peach/states/{label}_fired",
                color_frame=3,
                render_scale=4,
                raster_root=peach_raster_root,
                hidden_names=fired_names,
            )
            peach_state_animations[label]["fired_hidden_names"] = sorted(fired_names)

    # Peach's outer frame 22 ("spawn") contains an unnamed 57-frame clip, so
    # it is absent from named_places even though the original countdown uses it.
    for label_item in peach_timeline["labels"]:
        label = str(label_item["name"])
        if label in peach_state_animations:
            continue
        frame_index = int(label_item["frame"]) - 1
        if not (0 <= frame_index < len(peach_display_frames)) or not peach_display_frames[frame_index]:
            continue
        place = peach_display_frames[frame_index][0]
        sprite_id = int(place["character_id"])
        state_frames = export_composited_sprite_frames(
            source,
            root,
            sprite_id,
            ROOT / f"assets/fighters/peach/states/{label}",
            color_frame=3,
            render_scale=4,
            raster_root=peach_raster_root,
        )
        matrix = place["matrix"]
        peach_state_animations[label] = {
            "symbol_id": sprite_id,
            "placed_name": place.get("name", ""),
            "timeline_frame": int(label_item["frame"]),
            "timeline": extract_sprite_timeline(find_sprite(root, sprite_id)),
            "frame_count": len(state_frames),
            "playback": extract_timeline_playback(source, sprite_id),
            "state_offset": {"x": matrix[4], "y": matrix[5]},
            "frames": state_frames,
        }

    # SelectFighter.SetColor stores red/blue/green/orange as frames 1-4 on
    # every nested body-part timeline. Export each state with one immutable
    # color frame so a running/turning fighter cannot mix character variants.
    peach_color_state_animations = {"3": peach_state_animations}
    for color_frame in (1, 2, 4):
        color_animations = {}
        for label, animation in peach_state_animations.items():
            state_frames = export_composited_sprite_frames(
                source,
                root,
                int(animation["symbol_id"]),
                ROOT / f"assets/fighters/peach/colors/{color_frame}/states/{label}",
                color_frame=color_frame,
                render_scale=4,
                raster_root=peach_raster_root,
            )
            color_animations[label] = {
                **animation,
                "color_frame": color_frame,
                "frame_count": len(state_frames),
                "frames": state_frames,
            }
            fired_names = set(animation.get("fired_hidden_names", []))
            if fired_names:
                color_animations[label]["fired_frames"] = export_composited_sprite_frames(
                    source,
                    root,
                    int(animation["symbol_id"]),
                    ROOT / f"assets/fighters/peach/colors/{color_frame}/states/{label}_fired",
                    color_frame=color_frame,
                    render_scale=4,
                    raster_root=peach_raster_root,
                    hidden_names=fired_names,
                )
                color_animations[label]["fired_hidden_names"] = sorted(fired_names)
        peach_color_state_animations[str(color_frame)] = color_animations

    peach_held_item_animations = {}
    for item_name, hand_frame in (("mine", 2), ("grenade", 3)):
        item_colors = {}
        for color_frame in range(1, 5):
            held_animations = {}
            for label, animation in peach_color_state_animations[str(color_frame)].items():
                state_frames = export_composited_sprite_frames(
                    source,
                    root,
                    int(animation["symbol_id"]),
                    ROOT / f"assets/fighters/peach/held/{item_name}/colors/{color_frame}/states/{label}",
                    color_frame=color_frame,
                    render_scale=4,
                    raster_root=peach_raster_root,
                    hand_frame=hand_frame,
                )
                held_animations[label] = {
                    **animation,
                    "held_item": item_name,
                    "hand_frame": hand_frame,
                    "frame_count": len(state_frames),
                    "frames": state_frames,
                }
            item_colors[str(color_frame)] = held_animations
        peach_held_item_animations[item_name] = item_colors

    peach_config = FIGHTER_CONFIGS["PeachPlayer"]
    fighter_bundles = {
        "PeachPlayer": {
            "name": "PeachPlayer",
            "symbol_id": SYMBOLS["peach_player"],
            "class": "PeachPlayer",
            "character_name": peach_config["character_name"],
            "slug": peach_config["slug"],
            "weight": peach_config["weight"],
            "speed": peach_config["speed"],
            "power": peach_config["power"],
            "base_move_xinc": 8 * float(peach_config["speed"]),
            "jump_yinc": -9,
            "gravity_per_tick": 0.5,
            "max_fall_yinc": 6,
            "special_kind": peach_config["special_kind"],
            "special_up_motion": fighter_special_motion("PeachPlayer"),
            "timeline": peach_timeline,
            "frames": peach_frames,
            "state_animations": peach_state_animations,
            "color_state_animations": peach_color_state_animations,
            "held_item_state_animations": peach_held_item_animations,
            "attacks": fighter_attack_profiles("PeachPlayer"),
            "source_class": "raw_ffdec_export_scripts/scripts/__Packages/PeachPlayer.as",
        }
    }
    for fighter_name in MENU_FIGHTER_ORDER:
        if fighter_name == "PeachPlayer":
            continue
        fighter_bundles[fighter_name] = export_fighter_bundle(
            source,
            root,
            fighter_name,
            peach_raster_root,
        )

    menu_selection_previews = {}
    for fighter_name, fighter_id in MENU_FIGHTERS.items():
        fighter_timeline = extract_sprite_timeline(find_sprite(root, fighter_id))
        labels = {int(item["frame"]): str(item["name"]) for item in fighter_timeline["labels"]}
        run_place = next(
            place
            for place in fighter_timeline["named_places"]
            if labels.get(int(place["frame"]), str(place["name"])) == "run"
        )
        run_id = int(run_place["character_id"])
        colors = {}
        for color_frame in range(1, 5):
            animation = fighter_bundles[fighter_name]["color_state_animations"][str(color_frame)]["run"]
            frames = animation["frames"]
            playback = animation.get("playback", {})
            timeline = animation["timeline"]
            colors[str(color_frame)] = {
                "frame_count": len(frames),
                "timeline": timeline,
                "playback": playback,
                "state_offset": {
                    "x": float(run_place["matrix"]["x"]),
                    "y": float(run_place["matrix"]["y"]),
                },
                "frames": frames,
            }
        menu_selection_previews[fighter_name] = {
            "fighter_symbol_id": fighter_id,
            "run_symbol_id": run_id,
            "colors": colors,
        }

    menu_raster_root = ROOT / "assets/menu"
    menu_coin_assets = {}
    coin_dst = ROOT / "assets/menu/player_coins"
    coin_dst.mkdir(parents=True, exist_ok=True)
    coin_frames = build_display_frames(root, 816)
    for color_frame in range(1, 5):
        placements = [
            place
            for place in coin_frames[color_frame - 1]
            if int(place.get("depth", 0)) != 3
        ]
        rendered = render_display_frame_with_bounds(
            source,
            root,
            placements,
            color_frame,
            render_scale=4,
            raster_root=menu_raster_root,
        )
        if rendered is None:
            raise SystemExit(f"Could not render PlayerCoin color {color_frame} without PNum text.")
        image, bounds = rendered
        path = coin_dst / f"{color_frame}.png"
        image.save(path)
        menu_coin_assets[str(color_frame)] = {
            "image": f"assets/menu/player_coins/{color_frame}.png",
            "render_scale": 4,
            "offset": {"x": bounds[0], "y": bounds[1]},
            "logical_size": {"w": bounds[2] - bounds[0], "h": bounds[3] - bounds[1]},
        }

    menu_player_box_assets = {}
    box_dst = ROOT / "assets/menu/player_boxes"
    box_dst.mkdir(parents=True, exist_ok=True)
    box_frames = build_display_frames(root, 809)
    for frame_no, placements in enumerate(box_frames, start=1):
        background_places = [
            place
            for place in placements
            if int(place.get("depth", 0)) in {1, 2}
        ]
        rendered = render_display_frame_with_bounds(
            source,
            root,
            background_places,
            min(frame_no, 4),
            render_scale=4,
            raster_root=menu_raster_root,
        )
        if rendered is None:
            raise SystemExit(f"Could not render PlayerPoseBox background frame {frame_no}.")
        image, bounds = rendered
        path = box_dst / f"{frame_no}.png"
        image.save(path)
        menu_player_box_assets[str(frame_no)] = {
            "image": f"assets/menu/player_boxes/{frame_no}.png",
            "render_scale": 4,
            "offset": {"x": bounds[0], "y": bounds[1]},
            "logical_size": {"w": bounds[2] - bounds[0], "h": bounds[3] - bounds[1]},
        }

    pose_places = build_display_frames(root, 809)[0]
    toggle_place = next(place for place in pose_places if place.get("name") == "Toggle")
    ai_place = next(place for place in pose_places if place.get("name") == "AISetter")
    ai_buttons = {
        int(place["character_id"]): place
        for place in build_display_frames(root, int(ai_place["character_id"]))[0]
        if int(place.get("character_id", 0)) in {799, 800}
    }
    player_toggle_rect = rect_dict(
        transformed_bounds(
            button_hit_bounds(source, root, 803),
            toggle_place.get("matrix", (1, 0, 0, 1, 0, 0)),
        )
    )
    player_ai_rects = {}
    for direction, button_id in (("decrement", 799), ("increment", 800)):
        matrix = compose_matrices(
            ai_place.get("matrix", (1, 0, 0, 1, 0, 0)),
            ai_buttons[button_id].get("matrix", (1, 0, 0, 1, 0, 0)),
        )
        player_ai_rects[direction] = rect_dict(
            transformed_bounds(button_hit_bounds(source, root, button_id), matrix)
        )

    limit_root = next(
        place for place in main_display_frames[45] if int(place.get("character_id", 0)) == 1029
    )
    limit_container = next(
        place for place in build_display_frames(root, 1029)[0] if int(place.get("character_id", 0)) == 1028
    )
    limit_parent_matrix = compose_matrices(
        limit_root.get("matrix", (1, 0, 0, 1, 0, 0)),
        limit_container.get("matrix", (1, 0, 0, 1, 0, 0)),
    )
    limit_places = build_display_frames(root, 1028)[0]
    limit_rects = {}
    for key, button_id in (
        ("decrement", 1022),
        ("increment", 1023),
        ("toggle_stock", 1025),
        ("toggle_time", 1026),
    ):
        place = next(item for item in limit_places if int(item.get("character_id", 0)) == button_id)
        matrix = compose_matrices(
            limit_parent_matrix,
            place.get("matrix", (1, 0, 0, 1, 0, 0)),
        )
        limit_rects[key] = rect_dict(
            transformed_bounds(button_hit_bounds(source, root, button_id), matrix)
        )

    menu_root_buttons = {}
    for frame_no in range(43, 51):
        frame_buttons = []
        for place in main_display_frames[frame_no - 1]:
            button_id = int(place.get("character_id", 0))
            try:
                local_hit = button_hit_bounds(source, root, button_id)
            except ValueError:
                continue
            matrix = place.get("matrix", (1, 0, 0, 1, 0, 0))
            frame_buttons.append(
                {
                    "symbol_id": button_id,
                    "matrix": {
                        "a": matrix[0],
                        "b": matrix[1],
                        "c": matrix[2],
                        "d": matrix[3],
                        "x": matrix[4],
                        "y": matrix[5],
                    },
                    "hit_rect": rect_dict(transformed_bounds(local_hit, matrix)),
                }
            )
        menu_root_buttons[str(frame_no)] = frame_buttons

    preloader_root = next(
        place for place in main_display_frames[0] if int(place.get("character_id", 0)) == 913
    )
    preloader_matrix = preloader_root.get("matrix", (1, 0, 0, 1, 0, 0))
    preloader_ready = build_display_frames(root, 913)[1]
    play_place = next(place for place in preloader_ready if place.get("name") == "play_pb")
    play_matrix = play_place.get("matrix", (1, 0, 0, 1, 0, 0))
    play_bounds = symbol_bounds(source, root, 912, 1, 3, {})
    if play_bounds is None:
        raise SystemExit("Missing preloader play button bounds.")
    nested_play_bounds = transformed_bounds(play_bounds, play_matrix)
    preloader_play_rect = rect_dict(transformed_bounds(nested_play_bounds, preloader_matrix))

    sponsor_root = next(
        place for place in main_display_frames[1] if int(place.get("character_id", 0)) == 928
    )
    sponsor_frames = build_display_frames(root, 928)
    sponsor_button_frames = [
        frame_no
        for frame_no, frame in enumerate(sponsor_frames, start=1)
        if any(int(place.get("character_id", 0)) == 921 for place in frame)
    ]
    sponsor_button_place = next(
        place
        for place in sponsor_frames[sponsor_button_frames[0] - 1]
        if int(place.get("character_id", 0)) == 921
    )
    sponsor_button_matrix = compose_matrices(
        sponsor_root.get("matrix", (1, 0, 0, 1, 0, 0)),
        sponsor_button_place.get("matrix", (1, 0, 0, 1, 0, 0)),
    )
    sponsor_button_rect = rect_dict(
        transformed_bounds(button_hit_bounds(source, root, 921), sponsor_button_matrix)
    )

    platform_probe_size = image_size(source / "raw_ffdec_export/sprites/DefineSprite_717/1.png")
    moving_platforms = export_rooftop_moving_platforms(root, platform_probe_size)
    spawn_h_size = image_size(source / "raw_ffdec_export/sprites/DefineSprite_719/1.png")
    spawn_point_size = image_size(source / "raw_ffdec_export/sprites/DefineSprite_720/1.png")
    symbol_sizes = {
        "717": platform_probe_size,
        "719": spawn_h_size,
        "720": spawn_point_size,
    }

    rooftop_objects = []
    for obj in rooftop_timeline["named_places"]:
        cid = str(obj["character_id"])
        size = symbol_sizes.get(cid, {"w": 0, "h": 0})
        matrix = obj["matrix"]
        rooftop_objects.append(
            {
                **obj,
                "source_size": size,
                "estimated_rect": {
                    "x": matrix["x"],
                    "y": matrix["y"],
                    "w": size["w"] * abs(matrix["scale_x"]),
                    "h": size["h"] * abs(matrix["scale_y"]),
                },
            }
        )

    manifest = {
        "swf": swf_meta,
        "match": {
            "limit_mode": "stock",
            "starting_lives": 5,
            "respawn_invincible_ms": 3000,
            "respawn_invincible_decrement_per_tick": 25,
            "ready_sequence": [5, 4, 3, 2, 1, 0, -1],
        },
        "results": {
            "pre_end_start_frame": 52,
            "pre_end_stop_frame": 100,
            "start_frame": 101,
            "podium_start_frame": 143,
            "stop_frame": 170,
            "frame_rate": 30,
            "frames": result_frame_data,
            "winner_text": {
                "local_x": -243.7,
                "local_y": -31.2,
                "width": 491.4,
                "height": 66.35,
                "font": "Futura Md BT",
                "font_size": 52,
                "color": [255, 255, 0],
            },
            "podium_slots": podium_slots,
            "fighter_scale": 2.5,
            "stats": {
                "container": {"x": 310, "y": 151},
                "legend": {"x": -255, "y": -52, "width": 105.45, "height": 135.7},
                "columns": [-149.25, -47.25, 54.75, 156.75],
                "font": "Futura Md BT",
                "font_size": 20,
                "line_step": 22,
            },
            "main_button": {"x": 485.3279094815, "y": 372.5525283325, "w": 101.27883985, "h": 16.9560380265},
            "more_games_button": {"x": 8.35, "y": 367.2, "w": 160.5, "h": 23.5},
            "more_games_url": "http://www.armorgames.com/",
        },
        "menu": {
            "frame_rate": 30,
            "opening_start_frame": 3,
            "opening_stop_frame": 39,
            "intro_frame": 42,
            "root_buttons": menu_root_buttons,
            "preloader": {
                "root_symbol_id": 913,
                "play_symbol_id": 912,
                "root_pos": {"x": preloader_matrix[4], "y": preloader_matrix[5]},
                "play_rect": preloader_play_rect,
            },
            "sponsor_intro": {
                "root_frame": 2,
                "root_symbol_id": 928,
                "frame_rate": 30,
                "frame_count": len(sponsor_frames),
                "asset_dir": "assets/menu/sponsor_intro/2",
                "armor_button_id": 921,
                "armor_button_active_start": sponsor_button_frames[0],
                "armor_button_active_stop": sponsor_button_frames[-1],
                "armor_button_rect": sponsor_button_rect,
                "url": "http://www.armorgames.com",
                "target": "blank",
            },
            "selection_previews": menu_selection_previews,
            "player_select": {
                "fighters": list(MENU_FIGHTER_ORDER),
                "fighter_box": {"first_x": 6, "step_x": 94, "y": 70},
                "player_box": {"first_x": 25, "span_x": 550, "y": 188},
                "coin": {"first_x": 30, "step_x": 32, "target_x": 40, "target_y": 125, "random_range": 60},
                "preview": {"x": 70, "y": 170, "scale": 3.5, "state": "run"},
                "default_ai_level": 7,
                "ai_level_min": 1,
                "ai_level_max": 20,
                "toggle_rect": player_toggle_rect,
                "ai_level_rects": player_ai_rects,
                "limit_rects": limit_rects,
                "coin_assets": menu_coin_assets,
                "player_box_backgrounds": menu_player_box_assets,
            },
        },
        "source_project": str(source),
        "source_refs": SOURCE_REFS,
        "symbols": {k: SYMBOLS[k] for k in sorted(SYMBOLS)},
        "symbol_class_count": len(symbols),
        "stage": {
            "name": "Rooftop",
            "symbol_id": SYMBOLS["rooftop"],
            "background": "assets/stages/rooftop/background.png",
            "background_size": bg_size,
            "background_offset": {"x": -247.94991248800002, "y": -106.03146821499998},
            "foreground": "assets/stages/rooftop/foreground.png",
            "foreground_size": foreground_size,
            "helicopter": helicopter,
            "moving_platforms": moving_platforms,
            "view_bounds": {"x": 0, "y": -120, "w": 1287, "h": 920},
            "bounds_cam": {"x": 100, "y": -100, "w": 950, "h": 450},
            "bounds": {"x": -50, "y": -200, "w": 1200, "h": 700},
            "objects": rooftop_objects,
            "sounds_later": ["Rooftop", "Helicopter"],
        },
        "fighter": fighter_bundles["PeachPlayer"],
        "fighters": fighter_bundles,
        "projectiles": {
            "Bullet": {
                "symbol_id": SYMBOLS["bullet"],
                "image": "assets/projectiles/bullet/bullet.png",
                "image_size": bullet_size,
                "render_scale": 4,
                "offset": {"x": bullet_bounds[0], "y": bullet_bounds[1]},
                "life_ms": 3000,
                "xinc": 20,
                "damage": 10,
                "throw_power": 3,
                "throw_angle": 45,
            },
            "Rocket": {
                "symbol_id": SYMBOLS["rocket"],
                "image": "assets/projectiles/rocket/rocket.png",
                "image_size": rocket_size,
                "render_scale": 4,
                "offset": {"x": rocket_bounds[0], "y": rocket_bounds[1]},
                "life_ms": 3000,
                "explosion_size": 5,
            },
            "Pencil": {
                "symbol_id": SYMBOLS["pencil"],
                "frames": special_projectile_frames["Pencil"],
                "offset": {"x": special_projectile_bounds["Pencil"][0], "y": special_projectile_bounds["Pencil"][1]},
                "life_ms": 2500,
                "xinc": 7,
                "damage": 20,
                "throw_power": 3.5,
                "throw_angle": 50,
                "rotation_per_tick": 15,
            },
            "Poop": {
                "symbol_id": SYMBOLS["poop"],
                "frames": special_projectile_frames["Poop"],
                "offset": {"x": special_projectile_bounds["Poop"][0], "y": special_projectile_bounds["Poop"][1]},
                # Persistent child DefineSprite_400 jumps 28 -> 19.
                "playback": {"loop_from": 19, "loop_at": 28},
                "life_ms": 3000,
                "xinc": 7,
                "damage": 15,
                "throw_power": 3,
                "throw_angle": 45,
            },
            "Garbage": {
                "symbol_id": SYMBOLS["garbage"],
                "frames": special_projectile_frames["Garbage"],
                "offset": {"x": special_projectile_bounds["Garbage"][0], "y": special_projectile_bounds["Garbage"][1]},
                "life_ms": 500,
                "xinc": 8,
                "yinc": -2,
                "gravity_per_tick": 0.5,
                "damage": 6,
                "throw_power": 3,
                "throw_angle": 45,
                "rotation_per_tick": 20,
            },
            "EnergyBall": {
                "symbol_id": SYMBOLS["energy_ball"],
                "frames": special_projectile_frames["EnergyBall"],
                "offset": {"x": special_projectile_bounds["EnergyBall"][0], "y": special_projectile_bounds["EnergyBall"][1]},
                "life_ms": 2000,
                "xinc": 5,
                "damage": 7,
                "throw_power": 2.5,
                "throw_angle": 45,
                "electrocuted_ms": 100,
                "sine_degrees_per_x": 10,
                "sine_amplitude": 2,
                "bounce_on_wall": True,
            },
        },
        "items": {
            "frequency": 5,
            "classes": ["Mine", "Grenade"],
            "Mine": {
                "symbol_id": SYMBOLS["mine"],
                "class": "Mine",
                "life_ms": 20000,
                "timeline": extract_sprite_timeline(find_sprite(root, SYMBOLS["mine"])),
                "frames": mine_frames,
            },
            "Grenade": {
                "symbol_id": SYMBOLS["grenade"],
                "class": "Grenade",
                "life_ms": 20000,
                "timeline": extract_sprite_timeline(find_sprite(root, SYMBOLS["grenade"])),
                "frames": grenade_frames,
            },
        },
        "effects": {
            "Spawn1": {
                "symbol_id": SYMBOLS["spawn1"],
                "frame_rate": 30,
                "reveal_frame": 10,
                "frames": spawn1_frames,
            },
            "Spawn2": {
                "symbol_id": SYMBOLS["spawn2"],
                "frame_rate": 30,
                "reveal_frame": 20,
                "frames": spawn2_frames,
            },
            "Puff": {
                "symbol_id": SYMBOLS["puff"],
                "frame_rate": 30,
                "frames": puff_frames,
            },
            "PlayerDeath": {
                "symbol_id": SYMBOLS["player_death"],
                "frame_rate": 30,
                "frames": player_death_frames,
            },
            "PunchDamage": {
                "symbol_id": SYMBOLS["punch_damage"],
                "frame_rate": 30,
                "frames": punch_damage_frames,
            },
            "CameraTrick": {
                "symbol_id": SYMBOLS["camera_trick"],
                "frame_rate": 30,
                "frames": camera_trick_frames,
            },
            "PosIndicator": {
                "symbol_id": SYMBOLS["pos_indicator"],
                "frame_rate": 30,
                "frames": pos_indicator_frames,
            },
            "FarIndicator": {
                "symbol_id": SYMBOLS["far_indicator"],
                "frame_rate": 30,
                "frames": far_indicator_frames,
            },
            "Shield": {
                "symbol_id": SYMBOLS["shield"],
                "frame_rate": 30,
                "frames": shield_frames,
            },
            "ItemIndicator": {
                "symbol_id": SYMBOLS["item_indicator"],
                "frame_rate": 30,
                "frames": item_indicator_frames,
            },
            "BoomStar": {
                "symbol_id": SYMBOLS["boom_star"],
                "frame_rate": 30,
                "frames": boom_star_frames,
            },
            "BoomWave": {
                "symbol_id": SYMBOLS["boom_wave"],
                "frame_rate": 30,
                "frames": boom_wave_frames,
            },
            "BoomMatter": {
                "symbol_id": SYMBOLS["boom_matter"],
                "frame_rate": 30,
                "frames": boom_matter_frames,
            },
        },
        "ui": {
            "layout": {
                "reference_size": {"w": 600, "h": 400},
                "damage_origin": {"x": 60, "y": 340},
                "damage_spacing": 150,
                "damage_font": {"name": "Arial", "size": 23, "bold": True},
                # DefineSprite 779 / DefineEditText 778. The right/top values
                # include the field placement and its RECT edge in logical px.
                "damage_field": {"right": 35.4, "top": -14.85, "align": "right"},
                "damage_glow": {"blur_x": 2.0, "blur_y": 2.0, "strength": 3.328125},
                # DefineSprite 780 depth-9 matrices and colour transforms.
                "damage_pulse": [
                    {"scale": 1.0, "x": 2.25, "y": -2.8, "brightness": 255},
                    {"scale": 1.4526215, "x": 4.25, "y": -4.8, "brightness": 51},
                    {"scale": 1.311203, "x": 0.75, "y": -4.3, "brightness": 102},
                    {"scale": 1.1697388, "x": -2.75, "y": -3.8, "brightness": 153},
                    {"scale": 1.1343842, "x": -0.75, "y": -1.8, "brightness": 179},
                    {"scale": 1.0990143, "x": 1.25, "y": 0.2, "brightness": 204},
                    {"scale": 1.04953, "x": 3.25, "y": -1.8, "brightness": 230},
                    {"scale": 1.0, "x": 5.25, "y": -3.8, "brightness": 255},
                ],
                "life_counter": {
                    "counter_x": -38.2,
                    "counter_y": -48.75,
                    "more_icon_x": 18.25,
                    "more_text_x": 33.5,
                    "more_text_y": -2.85,
                    "font_size": 14,
                },
                "timer": {"x": 306.8, "y": 10, "font_size": 31},
                "big_text": {"x": 2, "y": 152, "w": 600, "h": 100, "font_size": 80},
                "score_upper": {"x": 8.35, "y": -43.2},
                "pause_end_button": {"x": 251.8, "y": 202.7, "w": 99.0, "h": 23.25},
                "far_indicator_scale_threshold": 1.5,
                "far_indicator": {
                    "font": "Futura Md BT",
                    "font_size": 20,
                    "text_center_x": 1,
                    "text_visible_top": -83,
                    "source_canvas": {"x": -16, "y": -90, "w": 34, "h": 49},
                    "team_colors": [
                        [255, 0, 0],
                        [51, 102, 255],
                        [102, 204, 0],
                        [255, 204, 0],
                    ],
                },
            },
            "OSDBigIcon": {
                "symbol_id": SYMBOLS["osd_bigicon"],
                "peach_frame": 3,
                "timeline": osd_bigicon_timeline,
                "frames": osd_bigicon_frames,
            },
            "OSDDamage": {
                "symbol_id": SYMBOLS["osd_damage"],
                "frame_rate": 30,
                "frames": osd_damage_frames,
            },
            "OSDScoreUpper": {
                "symbol_id": SYMBOLS["osd_score_upper"],
                "frame_rate": 30,
                "timeline": osd_score_upper_timeline,
                "frames": osd_score_upper_frames,
            },
            "OSDLifeGraphic": {
                "symbol_id": SYMBOLS["osd_life_graphic"],
                "peach_frame": 3,
                "timeline": osd_life_timeline,
                "frames": osd_life_frames,
                "peach_color_frames": osd_peach_life_frames,
                "character_color_frames": osd_life_character_frames,
            },
        },
    }
    # build_stages.py owns the expensive nested stage extraction. Keep its
    # authoritative result during incremental fighter/menu rebuilds.
    if isinstance(previous_manifest.get("stages"), dict):
        manifest["stages"] = previous_manifest["stages"]
    out = output_path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def main() -> None:
    source = Path(sys.argv[1]).expanduser().resolve() if len(sys.argv) > 1 else DEFAULT_SOURCE
    manifest = build(source)
    print(f"wrote assets/manifests/glorton_manifest.json")
    print(f"peach labels: {len(manifest['fighter']['timeline']['labels'])}")
    print(f"rooftop named objects: {len(manifest['stage']['objects'])}")


if __name__ == "__main__":
    main()
