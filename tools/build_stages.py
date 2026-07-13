from __future__ import annotations

import gc
import json
import math
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

from build_manifest import (
    DEFAULT_SOURCE,
    HIDDEN_STAGE_PREFIXES,
    ROOT,
    SYMBOLS,
    apply_color_transform,
    build_display_frames,
    compose_matrices,
    extract_timeline_playback,
    extract_sprite_timeline,
    find_sprite,
    matrix_values,
    render_display_frame_with_bounds,
    raster_symbol_frame,
    sprite_subtags,
    symbol_bounds,
    symbol_raster_path,
    transform_image_with_local_bounds,
    transformed_bounds,
)


STAGE_CONFIGS = {
    "Rooftop": {
        "symbol": "rooftop",
        "background": "rooftop_background",
        "slug": "rooftop",
        "bounds_cam": (100, -100, 950, 450),
        "bounds": (-50, -200, 1200, 700),
        "sounds": ["Rooftop", "Helicopter"],
        "dynamic_ids": {711},
    },
    "Mogadishu": {
        "symbol": "mogadishu",
        "background": "mogadishu_background",
        "slug": "mogadishu",
        "bounds_cam": (-900, -100, 2600, 500),
        "bounds": (-1000, -200, 2700, 700),
        "sounds": ["Mogadishu"],
        "dynamic_ids": set(),
    },
    "B52": {
        "symbol": "b52",
        "background": "b52_background",
        "slug": "b52",
        "bounds_cam": (50, -200, 950, 550),
        "bounds": (-50, -300, 1200, 700),
        "sounds": ["B52", "JetEngine"],
        "dynamic_ids": {860, 861, 862, 864, 865},
    },
    "Space": {
        "symbol": "space",
        "background": "space_background",
        "slug": "space",
        "bounds_cam": (-800, -400, 1500, 1000),
        "bounds": (-900, -500, 1700, 1200),
        "sounds": ["Space"],
        "dynamic_ids": set(),
    },
}


def matrix_tuple(data: dict[str, float]) -> tuple[float, float, float, float, float, float]:
    return (
        float(data["scale_x"]),
        float(data["rotate_skew1"]),
        float(data["rotate_skew0"]),
        float(data["scale_y"]),
        float(data["x"]),
        float(data["y"]),
    )


def helper_name(name: str) -> bool:
    return any(name.startswith(prefix) for prefix in HIDDEN_STAGE_PREFIXES) or name.startswith(("Boom", "Killer"))


def save_layer(
    source: Path,
    root: ET.Element,
    placements: list[dict[str, object]],
    destination: Path,
    parent_frame: int,
    bounds_cache: dict,
    render_cache: dict,
    timeline_cache: dict,
) -> dict[str, object] | None:
    rendered = render_display_frame_with_bounds(
        source,
        root,
        placements,
        color_frame=3,
        bounds_cache=bounds_cache,
        render_cache=render_cache,
        parent_frame=parent_frame,
        render_scale=4,
        raster_root=ROOT / "assets/ffdec_zoom4",
        timeline_cache=timeline_cache,
    )
    if rendered is None:
        return None
    image, bounds = rendered
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    return {
        "frame": parent_frame,
        "image": str(destination.relative_to(ROOT)),
        "render_scale": 4,
        "offset": {"x": bounds[0], "y": bounds[1]},
        "logical_size": {"w": image.width / 4, "h": image.height / 4},
    }


def _timeline_frame(
    source: Path,
    root: ET.Element,
    character_id: int,
    age: int,
    display_cache: dict[int, list[list[dict[str, object]]]],
    playback_cache: dict[int, dict[str, int]],
) -> tuple[int, list[dict[str, object]]] | None:
    try:
        if character_id not in display_cache:
            display_cache[character_id] = build_display_frames(root, character_id)
        frames = display_cache[character_id]
    except ValueError:
        return None
    if not frames:
        return None
    playback = playback_cache.setdefault(character_id, extract_timeline_playback(source, character_id))
    age = max(1, int(age))
    if "stop_at" in playback:
        frame_no = min(age, int(playback["stop_at"]))
    elif "loop_at" in playback:
        loop_at = min(len(frames), int(playback["loop_at"]))
        loop_from = max(1, min(loop_at, int(playback.get("loop_from", 1))))
        if age <= loop_at:
            frame_no = age
        else:
            frame_no = loop_from + (age - loop_at - 1) % max(1, loop_at - loop_from + 1)
    else:
        frame_no = (age - 1) % len(frames) + 1
    return frame_no, frames[min(len(frames), frame_no) - 1]


def _static_timeline_tree(
    source: Path,
    root: ET.Element,
    character_id: int,
    display_cache: dict[int, list[list[dict[str, object]]]],
    playback_cache: dict[int, dict[str, int]],
    result_cache: dict[int, bool],
    visiting: set[int] | None = None,
) -> bool:
    if character_id in result_cache:
        return result_cache[character_id]
    visiting = set() if visiting is None else set(visiting)
    if character_id in visiting:
        return True
    visiting.add(character_id)
    timeline = _timeline_frame(source, root, character_id, 1, display_cache, playback_cache)
    if timeline is None:
        result_cache[character_id] = True
        return True
    frames = display_cache[character_id]
    if len(frames) != 1:
        result_cache[character_id] = False
        return False
    result = all(
        _static_timeline_tree(
            source,
            root,
            int(place.get("character_id", 0)),
            display_cache,
            playback_cache,
            result_cache,
            visiting,
        )
        for place in frames[0]
        if int(place.get("character_id", 0)) > 0
    )
    result_cache[character_id] = result
    return result


def render_symbol_at_age(
    source: Path,
    root: ET.Element,
    character_id: int,
    age: int,
    bounds_cache: dict,
    raster_cache: dict,
    timeline_cache: dict,
    display_cache: dict[int, list[list[dict[str, object]]]],
    playback_cache: dict[int, dict[str, int]],
    static_tree_cache: dict[int, bool],
    static_render_cache: dict[int, tuple[Image.Image, tuple[float, float, float, float]] | None],
) -> tuple[Image.Image, tuple[float, float, float, float]] | None:
    is_static = _static_timeline_tree(
        source,
        root,
        character_id,
        display_cache,
        playback_cache,
        static_tree_cache,
    )
    if is_static and character_id in static_render_cache:
        cached = static_render_cache[character_id]
        return None if cached is None else (cached[0].copy(), cached[1])

    timeline = _timeline_frame(source, root, character_id, age, display_cache, playback_cache)
    if timeline is None:
        rendered = raster_symbol_frame(
            source,
            root,
            character_id,
            1,
            3,
            bounds_cache,
            ROOT / "assets/ffdec_zoom4",
            timeline_cache,
        )
        if is_static:
            static_render_cache[character_id] = rendered
        return None if rendered is None else (rendered[0].copy(), rendered[1])

    _frame_no, placements = timeline
    prepared: list[tuple[Image.Image, tuple[int, int]]] = []
    for place in placements:
        child_id = int(place.get("character_id", 0))
        if child_id <= 0:
            continue
        child_age = max(1, int(age) - int(place.get("start_frame", 1)) + 1)
        child = render_symbol_at_age(
            source,
            root,
            child_id,
            child_age,
            bounds_cache,
            raster_cache,
            timeline_cache,
            display_cache,
            playback_cache,
            static_tree_cache,
            static_render_cache,
        )
        if child is None:
            continue
        image, bounds = child
        image = apply_color_transform(image, place.get("color_transform"))
        transformed, pos = transform_image_with_local_bounds(
            image,
            bounds,
            place.get("matrix", (1, 0, 0, 1, 0, 0)),
            4,
        )
        bbox = transformed.getbbox()
        if bbox is not None:
            prepared.append((transformed.crop(bbox), (pos[0] + bbox[0], pos[1] + bbox[1])))
    if not prepared:
        if is_static:
            static_render_cache[character_id] = None
        return None
    left = min(pos[0] for _, pos in prepared)
    top = min(pos[1] for _, pos in prepared)
    right = max(pos[0] + image.width for image, pos in prepared)
    bottom = max(pos[1] + image.height for image, pos in prepared)
    canvas = Image.new("RGBA", (max(1, right - left), max(1, bottom - top)), (0, 0, 0, 0))
    for image, pos in prepared:
        canvas.alpha_composite(image, (pos[0] - left, pos[1] - top))
    rendered = canvas, (left / 4, top / 4, right / 4, bottom / 4)
    if is_static:
        static_render_cache[character_id] = rendered
    return rendered


def animation_period(
    source: Path,
    root: ET.Element,
    character_id: int,
    display_cache: dict[int, list[list[dict[str, object]]]],
    playback_cache: dict[int, dict[str, int]],
    seen: set[int] | None = None,
) -> tuple[int, bool]:
    seen = set() if seen is None else set(seen)
    if character_id in seen:
        return 1, False
    seen.add(character_id)
    timeline = _timeline_frame(source, root, character_id, 1, display_cache, playback_cache)
    if timeline is None:
        return 1, False
    frames = display_cache[character_id]
    playback = playback_cache[character_id]
    stopped = "stop_at" in playback
    period = 1 if stopped else len(frames)
    has_stop = stopped
    child_ids = {
        int(place.get("character_id", 0))
        for frame in frames
        for place in frame
        if int(place.get("character_id", 0)) > 0
    }
    for child_id in child_ids:
        child_period, child_stop = animation_period(
            source,
            root,
            child_id,
            display_cache,
            playback_cache,
            seen,
        )
        period = math.lcm(period, child_period)
        has_stop = has_stop or child_stop
    return max(1, period), has_stop


def export_visual_layers(
    source: Path,
    root: ET.Element,
    stage_name: str,
    stage_id: int,
    slug: str,
    dynamic_ids: set[int],
) -> tuple[dict[str, object], dict[str, object]]:
    display_frames = build_display_frames(root, stage_id)
    hidden = lambda place: helper_name(str(place.get("name", "")))
    full_animation = False
    static_placements = [
        place
        for place in display_frames[0]
        if not hidden(place) and int(place.get("character_id", 0)) not in dynamic_ids
    ]
    bounds_cache: dict = {}
    render_cache: dict = {}
    timeline_cache: dict = {}
    if full_animation:
        foreground_path = ROOT / f"assets/stages/{slug}/foreground.png"
        foreground_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(foreground_path)
        static = {
            "frame": 1,
            "image": str(foreground_path.relative_to(ROOT)),
            "render_scale": 4,
            "offset": {"x": 0.0, "y": 0.0},
            "logical_size": {"w": 0.25, "h": 0.25},
        }
    else:
        static = save_layer(
            source,
            root,
            static_placements,
            ROOT / f"assets/stages/{slug}/foreground.png",
            1,
            bounds_cache,
            render_cache,
            timeline_cache,
        )
    if static is None:
        raise SystemExit(f"{stage_name} has no visible foreground")

    dynamic_root = ROOT / f"assets/stages/{slug}/dynamic"
    dynamic_root.mkdir(parents=True, exist_ok=True)
    for stale in dynamic_root.glob("*.png"):
        stale.unlink()
    dynamic_frames = []
    if dynamic_ids:
        for frame_no, placements in enumerate(display_frames, start=1):
            selected = [
                place
                for place in placements
                if not hidden(place)
                and (full_animation or int(place.get("character_id", 0)) in dynamic_ids)
            ]
            frame = save_layer(
                source,
                root,
                selected,
                dynamic_root / f"{frame_no:03d}.png",
                frame_no,
                bounds_cache,
                render_cache,
                timeline_cache,
            )
            if frame is not None:
                dynamic_frames.append(frame)
            render_cache.clear()
            if frame_no % 10 == 0 or frame_no == len(display_frames):
                gc.collect()
                print(f"rendered {stage_name} frame {frame_no}/{len(display_frames)}", flush=True)
    return static, {"frame_rate": 30, "frames": dynamic_frames}


def stage_objects(source: Path, root: ET.Element, stage_id: int) -> list[dict[str, object]]:
    timeline = extract_sprite_timeline(find_sprite(root, stage_id))
    bounds_cache: dict = {}
    timeline_cache: dict = {}
    objects = []
    for obj in timeline["named_places"]:
        character_id = int(obj["character_id"])
        local_bounds = symbol_bounds(
            source,
            root,
            character_id,
            1,
            3,
            bounds_cache,
            timeline_cache,
        )
        if local_bounds is None:
            estimated = (float(obj["matrix"]["x"]), float(obj["matrix"]["y"]), 0.0, 0.0)
            rect = {"x": estimated[0], "y": estimated[1], "w": 0.0, "h": 0.0}
            source_size = {"w": 0.0, "h": 0.0}
        else:
            world_bounds = transformed_bounds(local_bounds, matrix_tuple(obj["matrix"]))
            rect = {
                "x": world_bounds[0],
                "y": world_bounds[1],
                "w": world_bounds[2] - world_bounds[0],
                "h": world_bounds[3] - world_bounds[1],
            }
            source_size = {
                "w": local_bounds[2] - local_bounds[0],
                "h": local_bounds[3] - local_bounds[1],
            }
        objects.append({**obj, "source_size": source_size, "estimated_rect": rect})
    return objects


def moving_platforms(source: Path, root: ET.Element, stage_id: int) -> dict[str, object]:
    bounds_cache: dict = {}
    timeline_cache: dict = {}
    platforms: dict[str, list[dict[str, object]]] = {}
    for frame_no, placements in enumerate(build_display_frames(root, stage_id), start=1):
        for place in placements:
            name = str(place.get("name", ""))
            if not name.startswith("Moving"):
                continue
            character_id = int(place.get("character_id", 0))
            local_bounds = symbol_bounds(
                source,
                root,
                character_id,
                1,
                3,
                bounds_cache,
                timeline_cache,
            )
            if local_bounds is None:
                continue
            world_bounds = transformed_bounds(local_bounds, place.get("matrix", (1, 0, 0, 1, 0, 0)))
            platforms.setdefault(name, []).append(
                {
                    "frame": frame_no,
                    "rect": {
                        "x": world_bounds[0],
                        "y": world_bounds[1],
                        "w": world_bounds[2] - world_bounds[0],
                        "h": world_bounds[3] - world_bounds[1],
                    },
                }
            )
    return {"frame_rate": 30, "platforms": platforms}


def export_background(
    source: Path,
    root: ET.Element,
    symbol_id: int,
    slug: str,
) -> tuple[str, dict[str, object], dict[str, float]]:
    path = symbol_raster_path(source, symbol_id, 1, ROOT / "assets/ffdec_zoom4")
    if path is None:
        raise SystemExit(f"missing high-resolution background symbol {symbol_id}")
    bounds = symbol_bounds(source, root, symbol_id, 1, 3, {}, {})
    if bounds is None:
        raise SystemExit(f"background symbol {symbol_id} has no source bounds")
    destination = ROOT / f"assets/stages/{slug}/background.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
    with Image.open(destination) as image:
        size = {"w": image.width / 4, "h": image.height / 4, "render_scale": 4}
    return (
        str(destination.relative_to(ROOT)),
        size,
        {"x": bounds[0], "y": bounds[1]},
    )


def export_space_base(
    source: Path,
    root: ET.Element,
    symbol_id: int,
    slug: str,
) -> tuple[str, dict[str, object], dict[str, float]]:
    root_bounds = symbol_bounds(source, root, symbol_id, 1, 3, {}, {})
    if root_bounds is None:
        raise SystemExit("Space background has no source bounds")
    top_places = build_display_frames(root, symbol_id)[0]
    field_place = next(place for place in top_places if int(place.get("character_id", 0)) == 885)
    field_places = build_display_frames(root, 885)[0]
    base_place = next(place for place in field_places if int(place.get("character_id", 0)) == 882)
    matrix = compose_matrices(
        field_place.get("matrix", (1, 0, 0, 1, 0, 0)),
        base_place.get("matrix", (1, 0, 0, 1, 0, 0)),
    )
    rendered = raster_symbol_frame(
        source,
        root,
        882,
        1,
        3,
        {},
        ROOT / "assets/ffdec_zoom4",
        {},
    )
    if rendered is None:
        raise SystemExit("Space radial background shape 882 is missing")
    image, bounds = rendered
    transformed, pos = transform_image_with_local_bounds(image, bounds, matrix, 4)
    bbox = transformed.getbbox()
    if bbox is None:
        raise SystemExit("Space radial background shape 882 is empty")
    transformed = transformed.crop(bbox)
    # This is a smooth vector gradient. Two source pixels per logical unit are
    # enough for native-quality filtering and avoid a 190 MB runtime surface.
    image_2x = transformed.resize(
        (max(1, transformed.width // 2), max(1, transformed.height // 2)),
        Image.Resampling.LANCZOS,
    )
    destination = ROOT / f"assets/stages/{slug}/background.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    image_2x.save(destination)
    left = (pos[0] + bbox[0]) / 4
    top = (pos[1] + bbox[1]) / 4
    return (
        str(destination.relative_to(ROOT)),
        {
            "w": root_bounds[2] - root_bounds[0],
            "h": root_bounds[3] - root_bounds[1],
            "render_scale": 2,
        },
        {"x": left, "y": top},
    )


def export_space_star_layer(
    source: Path,
    root: ET.Element,
    top_place: dict[str, object],
    slug: str,
) -> dict[str, object]:
    source_path = symbol_raster_path(source, 884, 1, ROOT / "assets/ffdec_zoom4")
    bounds = symbol_bounds(source, root, 884, 1, 3, {}, {})
    if source_path is None or bounds is None:
        raise SystemExit("Space star sprite 884 is missing")
    destination = ROOT / f"assets/stages/{slug}/star.png"
    shutil.copy2(source_path, destination)
    top_matrix = top_place.get("matrix", (1, 0, 0, 1, 0, 0))
    frames = []
    for frame_no, placements in enumerate(build_display_frames(root, 885), start=1):
        matrices = [
            list(compose_matrices(top_matrix, place.get("matrix", (1, 0, 0, 1, 0, 0))))
            for place in placements
            if int(place.get("character_id", 0)) == 884
        ]
        frames.append({"frame": frame_no, "matrices": matrices})
    return {
        "frame_rate": 30,
        "sprite": {
            "image": str(destination.relative_to(ROOT)),
            "render_scale": 4,
            "offset": {"x": bounds[0], "y": bounds[1]},
            "logical_size": {"w": bounds[2] - bounds[0], "h": bounds[3] - bounds[1]},
        },
        "frames": frames,
    }


def export_background_animation(
    source: Path,
    root: ET.Element,
    symbol_id: int,
    slug: str,
    canvas_size: dict[str, object],
) -> dict[str, object]:
    placements = build_display_frames(root, symbol_id)[0]
    stage_root = ROOT / f"assets/stages/{slug}"
    for stale_layer in stage_root.glob("background_layer_*"):
        if stale_layer.is_dir():
            shutil.rmtree(stale_layer)
    bounds_cache: dict = {}
    render_cache: dict = {}
    timeline_cache: dict = {}
    display_cache: dict[int, list[list[dict[str, object]]]] = {}
    playback_cache: dict[int, dict[str, int]] = {}
    static_tree_cache: dict[int, bool] = {}
    static_render_cache: dict[int, tuple[Image.Image, tuple[float, float, float, float]] | None] = {}
    layers = []
    object_layers = []
    for layer_index, place in enumerate(placements):
        character_id = int(place.get("character_id", 0))
        try:
            frame_count = int(extract_sprite_timeline(find_sprite(root, character_id))["frame_count"])
        except ValueError:
            frame_count = 1
        if frame_count <= 1:
            continue
        if slug == "space" and character_id == 885:
            object_layers.append(export_space_star_layer(source, root, place, slug))
            continue
        destination = ROOT / f"assets/stages/{slug}/background_layer_{layer_index + 1}"
        destination.mkdir(parents=True, exist_ok=True)
        for stale in destination.glob("*.png"):
            stale.unlink()
        period, has_stop = animation_period(
            source,
            root,
            character_id,
            display_cache,
            playback_cache,
        )
        total_frames = period * (2 if has_stop else 1)
        loop_from = period + 1 if has_stop else 1
        frames = []
        for frame_no in range(1, total_frames + 1):
            child_age = max(1, frame_no - int(place.get("start_frame", 1)) + 1)
            child = render_symbol_at_age(
                source,
                root,
                character_id,
                child_age,
                bounds_cache,
                render_cache,
                timeline_cache,
                display_cache,
                playback_cache,
                static_tree_cache,
                static_render_cache,
            )
            if child is None:
                continue
            image, bounds = child
            transformed, pos = transform_image_with_local_bounds(
                image,
                bounds,
                place.get("matrix", (1, 0, 0, 1, 0, 0)),
                4,
            )
            bbox = transformed.getbbox()
            if bbox is None:
                continue
            cropped = transformed.crop(bbox)
            path = destination / f"{frame_no:03d}.png"
            cropped.save(path)
            left = pos[0] + bbox[0]
            top = pos[1] + bbox[1]
            frames.append(
                {
                    "frame": frame_no,
                    "image": str(path.relative_to(ROOT)),
                    "render_scale": 4,
                    "offset": {"x": left / 4, "y": top / 4},
                    "logical_size": {"w": cropped.width / 4, "h": cropped.height / 4},
                }
            )
        layers.append(
            {
                "frame_rate": 30,
                "loop_from": loop_from,
                "period": period,
                "frames": frames,
            }
        )
    return {
        "canvas_size": {"w": canvas_size["w"], "h": canvas_size["h"]},
        "layers": layers,
        "object_layers": object_layers,
    }


def build() -> None:
    source = DEFAULT_SOURCE
    root = ET.parse(source / "raw_ffdec_xml/fight-for-glorton.xml").getroot()
    manifest_path = ROOT / "assets/manifests/glorton_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages = {}
    for stage_name, config in STAGE_CONFIGS.items():
        stage_id = SYMBOLS[str(config["symbol"])]
        slug = str(config["slug"])
        if stage_name == "Rooftop":
            existing = dict(manifest["stage"])
            static = {
                "image": existing["foreground"],
                "render_scale": existing.get("foreground_size", {}).get("render_scale", 4),
                "offset": existing.get("foreground_offset", {"x": 0, "y": 0}),
            }
            dynamic = existing.get("dynamic_layer", {"frame_rate": 30, "frames": []})
            background = existing["background"]
            background_size = existing["background_size"]
            source_background_bounds = symbol_bounds(
                source,
                root,
                SYMBOLS[str(config["background"])],
                1,
                3,
                {},
                {},
            )
            if source_background_bounds is None:
                raise SystemExit("Rooftop background has no source bounds")
            background_offset = {
                "x": source_background_bounds[0],
                "y": source_background_bounds[1],
            }
        else:
            static, dynamic = export_visual_layers(
                source,
                root,
                stage_name,
                stage_id,
                slug,
                set(config["dynamic_ids"]),
            )
            if stage_name == "Space":
                background, background_size, background_offset = export_space_base(
                    source,
                    root,
                    SYMBOLS[str(config["background"])],
                    slug,
                )
            else:
                background, background_size, background_offset = export_background(
                    source,
                    root,
                    SYMBOLS[str(config["background"])],
                    slug,
                )
        background_animation = export_background_animation(
            source,
            root,
            SYMBOLS[str(config["background"])],
            slug,
            background_size,
        )
        bounds_cam = config["bounds_cam"]
        bounds = config["bounds"]
        stage = {
            "name": stage_name,
            "symbol_id": stage_id,
            "background": background,
            "background_size": background_size,
            "background_offset": background_offset,
            "background_animation": background_animation,
            "foreground": static["image"],
            "foreground_size": {**static.get("logical_size", {}), "render_scale": static["render_scale"]},
            "foreground_offset": static["offset"],
            "dynamic_layer": dynamic,
            "dynamic_above_foreground": stage_name == "B52",
            "moving_platforms": moving_platforms(source, root, stage_id),
            "view_bounds": {"x": bounds[0], "y": bounds[1], "w": bounds[2], "h": bounds[3]},
            "bounds_cam": {"x": bounds_cam[0], "y": bounds_cam[1], "w": bounds_cam[2], "h": bounds_cam[3]},
            "bounds": {"x": bounds[0], "y": bounds[1], "w": bounds[2], "h": bounds[3]},
            "objects": stage_objects(source, root, stage_id),
            "sounds": list(config["sounds"]),
        }
        if stage_name == "Rooftop":
            stage["helicopter"] = manifest["stage"].get("helicopter", {})
        stages[stage_name] = stage
        print(f"built {stage_name}: {len(stage['objects'])} objects")
    manifest["stages"] = stages
    manifest["stage"] = stages["Rooftop"]
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    build()
