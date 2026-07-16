from __future__ import annotations

import argparse
import html
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

from PIL import Image, ImageChops, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_MANIFEST = ROOT / "assets/manifests/glorton_manifest.json"
DEFAULT_SCENARIOS = ROOT / "tools/visual_scenarios.json"
REFERENCE_SIZE = (600, 400)
KO_FIXTURE_MARKER = "VisualKoListener"
KO_FIXTURE_APPEND = r'''
var VisualKoListener = new Object();
VisualKoListener.onKeyDown = function()
{
   if(Key.getCode() == 85)
   {
      var VisualKoIndex = 0;
      while(VisualKoIndex < Players.length)
      {
         if(Players[VisualKoIndex] != undefined)
         {
            Players[VisualKoIndex].State = "stop";
            Players[VisualKoIndex].AnimateAttack("koAttack");
         }
         VisualKoIndex++;
      }
   }
};
Key.addListener(VisualKoListener);
'''.strip()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def coverage_cases(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    fighters = list(manifest.get("fighters", {}))
    stages = list(manifest.get("stages", {}))
    items = list(manifest.get("items", {}).get("classes", []))
    attacks = (
        list(manifest["fighters"][fighters[0]].get("attacks", {}))
        if fighters
        else []
    )
    result: list[dict[str, Any]] = []
    for scene in (
        "preloader",
        "sponsor_intro",
        "opening",
        "main",
        "player_select",
        "stage_select",
        "options",
        "controls",
    ):
        result.append({"id": f"menu/{scene}", "category": "menu", "scene": scene})
    for value in ("loading", "5", "4", "3", "2", "1", "go"):
        result.append({"id": f"countdown/{value}", "category": "countdown", "value": value})
    for fighter in fighters:
        for color in range(4):
            result.append(
                {
                    "id": f"roster/{fighter}/color_{color + 1}",
                    "category": "roster",
                    "fighter": fighter,
                    "color": color,
                }
            )
        for attack in attacks:
            result.append(
                {
                    "id": f"attacks/{fighter}/{attack}",
                    "category": "attack",
                    "fighter": fighter,
                    "attack": attack,
                }
            )
        result.append(
            {
                "id": f"damage/{fighter}",
                "category": "damage",
                "fighter": fighter,
            }
        )
    for item in items:
        result.append({"id": f"items/{item}", "category": "item", "item": item})
    for stage in stages:
        result.append({"id": f"stages/{stage}", "category": "stage", "stage": stage})
    return result


def image_files(root: Path) -> set[str]:
    if not root.is_dir():
        return set()
    return {path.relative_to(root).as_posix() for path in root.rglob("*.png")}


def command_coverage(args: argparse.Namespace) -> int:
    manifest = load_json(args.manifest)
    cases = coverage_cases(manifest)
    original_files = image_files(args.original)
    remake_files = image_files(args.remake)
    diff_files = image_files(args.diff)
    counts: dict[str, int] = {}
    for case in cases:
        prefix = case["id"] + "/"
        original_count = sum(name.startswith(prefix) for name in original_files)
        remake_count = sum(name.startswith(prefix) for name in remake_files)
        diff_count = sum(name.startswith(prefix) for name in diff_files)
        case["frames"] = {
            "original": original_count,
            "remake": remake_count,
            "diff": diff_count,
        }
        case["status"] = (
            "compared"
            if diff_count
            else "paired"
            if original_count and remake_count
            else "remake-only"
            if remake_count
            else "original-only"
            if original_count
            else "missing"
        )
        counts[case["status"]] = counts.get(case["status"], 0) + 1
    report = {
        "schema": "glorton-visual-coverage-v1",
        "reference_size": list(REFERENCE_SIZE),
        "case_count": len(cases),
        "status_counts": counts,
        "cases": cases,
    }
    write_json(args.output, report)
    print(f"visual coverage: {len(cases)} cases -> {args.output}")
    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0


def prepare_runner(args: argparse.Namespace) -> int:
    destination = args.output
    destination.mkdir(parents=True, exist_ok=True)
    scenarios = load_json(args.scenarios)
    harness = ROOT / "tools/visual_harness/index.html"
    shutil.copy2(harness, destination / "index.html")
    shutil.copy2(args.swf, destination / "fight-for-glorton.swf")
    target = destination / "ruffle"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(args.ruffle, target)
    package = load_json(target / "package.json") if (target / "package.json").is_file() else {}
    write_json(
        args.baseline_metadata,
        {
            "schema": "glorton-visual-run-v1",
            "engine": "original-swf-via-ruffle",
            "ruffle_version": package.get("version", "unknown"),
            "swf": str(args.swf),
            "seed": scenarios.get("canonical_seed"),
            "seed_scope": "metadata-only; AVM1 random() is not externally seedable in Ruffle",
            "tick_ms": scenarios.get("tick_ms", 25),
            "reference_size": scenarios.get("reference_size", list(REFERENCE_SIZE)),
            "scenarios": str(args.scenarios),
            "extreme_state_fixture": {
                "file": "fight-for-glorton-ko.swf",
                "trigger_key": "U",
                "scope": "frame-51 listener calls the original AnimateAttack('koAttack') timeline",
                "known_side_effect": "FFDec AS2 bulk recompilation can leave unrelated HUD values undefined",
            },
        },
    )
    print(f"original runner ready: {destination / 'index.html'}")
    return 0


def prepare_ko_fixture(args: argparse.Namespace) -> int:
    """Build an instrumented SWF copy for the source-unreachable koAttack."""

    if args.work.exists():
        shutil.rmtree(args.work)
    shutil.copytree(args.scripts, args.work)
    frame_script = args.work / "frame_51/DoAction.as"
    source = frame_script.read_text(encoding="utf-8")
    if KO_FIXTURE_MARKER not in source:
        frame_script.write_text(source.rstrip() + "\n" + KO_FIXTURE_APPEND + "\n", encoding="utf-8")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "sh",
            str(args.ffdec),
            "-onerror",
            "abort",
            "-importScript",
            str(args.swf),
            str(args.output),
            str(args.work),
        ],
        check=True,
    )
    print(f"KO source-timeline fixture ready: {args.output}")
    return 0


def _save_surface(surface: Any, path: Path) -> None:
    import pygame

    path.parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surface, str(path))


def _case_frame(root: Path, case_id: str, frame: int) -> Path:
    return root / case_id / f"frame_{frame:06d}.png"


def _configure_battle(runtime: Any, stage_name: str, fighter_name: str, color: int, seed: int) -> None:
    import pygame

    from src.runtime import Stage

    runtime.match_config = {
        "type": "vsmode",
        "selected_stage": stage_name,
        "limit_mode": "stock",
        "limit_value": 5,
        "players": [
            {"fighter": fighter_name, "color": color, "computer": False, "enabled": True},
            {"fighter": "SBLPlayer", "color": (color + 1) % 4, "computer": False, "enabled": True},
        ],
    }
    runtime.manifest["match"]["limit_mode"] = "stock"
    runtime.manifest["match"]["starting_lives"] = 5
    runtime.stage = Stage(runtime.manifest, stage_name)
    runtime.simulation.reset(seed)
    runtime.match_state = "playing"
    runtime.ready_text = ""
    runtime.accumulator = 0
    runtime.stage_time_ms = 0
    platform = max(runtime.stage.platforms, key=lambda item: item.rect.w)
    center = float(platform.rect.centerx)
    for index, fighter in enumerate(runtime.fighters):
        fighter.intro_visible = True
        fighter.has_control = True
        fighter.dead = False
        fighter.state = "stop"
        fighter.current_label = "still"
        fighter.current_attack = ""
        fighter.pos = pygame.Vector2(center + index * 28, platform.rect.top)
        fighter.prev_pos = pygame.Vector2(fighter.pos)
        fighter.xinc = 0
        fighter.yinc = 0
        fighter.on_ground = True
        fighter.ground_platform = platform
    runtime.camera_view = None
    runtime.camera_target_view = None


def _draw_battle(runtime: Any, surface: Any, font: Any) -> None:
    surface.fill((0, 0, 0))
    runtime._draw_output(surface, font)


def _scripted_controls(
    events: Iterable[Mapping[str, Any]],
    tick: int,
    player_count: int = 2,
) -> list[dict[str, bool]]:
    controls = [{} for _ in range(player_count)]
    for event in events:
        if int(event.get("tick", -1)) != tick:
            continue
        player = int(event.get("player", 0))
        if not 0 <= player < player_count:
            raise ValueError(f"Visual input script refers to missing player {player}")
        for button in event.get("buttons", []):
            controls[player][str(button)] = True
    return controls


def _record_attack(
    runtime: Any,
    surface: Any,
    font: Any,
    output: Path,
    fighter_name: str,
    attack: str,
    seed: int,
    input_script: Iterable[Mapping[str, Any]],
) -> None:
    _configure_battle(runtime, "Rooftop", fighter_name, 0, seed)
    fighter = runtime.fighters[0]
    target = runtime.fighters[1]
    target.pos.x = fighter.pos.x + 18
    target.prev_pos.update(target.pos)
    if attack in {"punchAir", "specialAir"}:
        fighter.pos.y -= 70
        fighter.prev_pos.update(fighter.pos)
        fighter.on_ground = False
        fighter.ground_platform = None
        fighter.yinc = 0
    if attack == "specialUp":
        fighter.spec_up_ok = True
    if attack == "koAttack":
        # The original Fighter.DoCommon ko timeout makes this exported label
        # unreachable before Fighter.Attack can select it. Capture the source
        # timeline explicitly as an extreme-state proof instead of falsely
        # labelling a normal ground punch as koAttack.
        fighter.state = "stop"
        fighter._animate_attack("koAttack")
    animation = fighter.animations.get(attack, {})
    frame_count = int(animation.get("frame_count", 30))
    tick_count = min(100, max(12, math.ceil(frame_count * 40 / 30) + 6))
    case_id = f"attacks/{fighter_name}/{attack}"
    runtime.simulation.start_recording(
        {
            "visual_case": case_id,
            "state_setup": "source-unreachable-state-injection" if attack == "koAttack" else "input",
        }
    )
    _draw_battle(runtime, surface, font)
    _save_surface(surface, _case_frame(output, case_id, 0))
    for tick in range(tick_count):
        runtime.simulation.step(_scripted_controls(input_script, tick))
        _draw_battle(runtime, surface, font)
        _save_surface(surface, _case_frame(output, case_id, tick + 1))
    recording = runtime.simulation.stop_recording()
    write_json(output / case_id / "recording.json", recording)


def _record_damage(
    runtime: Any,
    surface: Any,
    font: Any,
    output: Path,
    fighter_name: str,
    seed: int,
    *,
    full: bool,
    input_script: Iterable[Mapping[str, Any]],
) -> None:
    """Record the named fighter receiving a real SBL ground punch."""

    _configure_battle(runtime, "Rooftop", fighter_name, 0, seed)
    target, attacker = runtime.fighters[:2]
    attacker.pos.x = target.pos.x + 18
    attacker.prev_pos.update(attacker.pos)
    attacker.facing = -1
    attacker.attack_facing = -1
    case_id = f"damage/{fighter_name}"
    runtime.simulation.start_recording({"visual_case": case_id})
    tick_count = 28 if full else 16
    for tick in range(tick_count):
        runtime.simulation.step(_scripted_controls(input_script, tick))
        _draw_battle(runtime, surface, font)
        _save_surface(surface, _case_frame(output, case_id, tick))
    recording = runtime.simulation.stop_recording()
    write_json(output / case_id / "recording.json", recording)


def command_record_remake(args: argparse.Namespace) -> int:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    import pygame

    from src.menu import MainMenu
    from src.runtime import RuntimeApp, StageItem

    pygame.init()
    pygame.display.set_mode((1, 1), pygame.HIDDEN)
    manifest = load_json(args.manifest)
    scenarios = load_json(args.scenarios)
    seed = int(args.seed if args.seed is not None else scenarios["canonical_seed"])
    if int(scenarios.get("tick_ms", 25)) != 25:
        raise ValueError("The remake currently records at the source 25 ms fixed tick")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "run.json",
        {
            "schema": "glorton-visual-run-v1",
            "engine": "remake",
            "seed": seed,
            "tick_ms": 25,
            "quality": args.quality,
            "suite": args.suite,
            "scenarios": str(args.scenarios),
        },
    )
    runtime = RuntimeApp(random_seed=seed)
    runtime.audio = None
    runtime.menu = MainMenu(ROOT, runtime.manifest)
    runtime.menu.quality = "MEDIUM"
    runtime.menu.sound_on = True
    runtime.menu.control_keys = [
        [pygame.K_a, pygame.K_d, pygame.K_w, pygame.K_s, pygame.K_j, pygame.K_k, pygame.K_LSHIFT],
        [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN, pygame.K_KP0, pygame.K_KP1, pygame.K_KP2],
        [0] * 7,
        [0] * 7,
    ]
    surface = pygame.Surface(REFERENCE_SIZE).convert()
    font = pygame.font.SysFont("menlo", 14)
    full = args.suite == "full"

    menu_frames = {
        "preloader": [1],
        "sponsor_intro": list(range(1, 82)) if full else [1, 41, 81],
        "opening": list(range(3, 40)) if full else [3, 20, 39],
        "main": [1],
        "options": [1],
        "controls": [1],
    }
    for scene, frames in menu_frames.items():
        runtime.menu.scene = scene
        for frame in frames:
            if scene == "sponsor_intro":
                runtime.menu.sponsor_frame = frame
            elif scene == "opening":
                runtime.menu.opening_frame = frame
            runtime.menu.draw(surface)
            _save_surface(surface, _case_frame(output, f"menu/{scene}", frame))
    runtime.menu._start_player_select("vsmode", 4, 4)
    runtime.menu.draw(surface)
    _save_surface(surface, _case_frame(output, "menu/player_select", 1))
    runtime.menu.scene = "stage_select"
    runtime.menu.draw(surface)
    _save_surface(surface, _case_frame(output, "menu/stage_select", 1))
    runtime.menu.quality = args.quality

    _configure_battle(runtime, "Rooftop", "PeachPlayer", 0, seed)
    runtime.simulation.reset(seed)
    last_value = None
    for tick in range(281):
        runtime.simulation.step([{}, {}])
        value = "loading" if runtime.match_state == "loading" else runtime.ready_text.lower().replace("!", "")
        if full or value != last_value:
            _draw_battle(runtime, surface, font)
            _save_surface(surface, _case_frame(output, f"countdown/{value or 'done'}", tick))
        last_value = value

    fighters = list(manifest.get("fighters", {}))
    roster_fighters = fighters if full else fighters[:1]
    roster_colors = range(4) if full else range(1)
    for fighter_name in roster_fighters:
        for color in roster_colors:
            _configure_battle(runtime, "Rooftop", fighter_name, color, seed)
            _draw_battle(runtime, surface, font)
            _save_surface(surface, _case_frame(output, f"roster/{fighter_name}/color_{color + 1}", 0))

    attacks = list(manifest["fighters"][fighters[0]].get("attacks", {}))
    attack_fighters = fighters if full else fighters[:1]
    attack_names = attacks if full else ["punchGround", "specialGround"]
    for fighter_name in attack_fighters:
        for attack in attack_names:
            input_script = scenarios.get("attack_inputs", {}).get(attack)
            if input_script is None:
                raise ValueError(f"Missing fixed input script for attack {attack}")
            _record_attack(
                runtime,
                surface,
                font,
                output,
                fighter_name,
                attack,
                seed,
                input_script,
            )

    damage_fighters = fighters if full else fighters[:1]
    for fighter_name in damage_fighters:
        _record_damage(
            runtime,
            surface,
            font,
            output,
            fighter_name,
            seed,
            full=full,
            input_script=scenarios.get("damage_input", []),
        )

    for stage_name in manifest.get("stages", {}):
        _configure_battle(runtime, stage_name, "PeachPlayer", 0, seed)
        ticks = 40 if full else 1
        for tick in range(ticks):
            if tick:
                runtime.simulation.step([{}, {}])
            _draw_battle(runtime, surface, font)
            _save_surface(surface, _case_frame(output, f"stages/{stage_name}", tick))

    for kind in manifest.get("items", {}).get("classes", []):
        _configure_battle(runtime, "Rooftop", "PeachPlayer", 0, seed)
        fighter = runtime.fighters[0]
        item = StageItem(
            kind=kind,
            pos=fighter.pos.copy(),
            frames=runtime.item_frames[kind],
            frame_labels=runtime.item_frame_labels.get(kind, {}),
            source_scale=runtime.item_source_scales.get(kind, 1.0),
        )
        runtime.items.append(item)
        for tick in range(16 if full else 2):
            if tick:
                runtime.simulation.step([{"punch_pressed": tick in {1, 8}}, {}])
            _draw_battle(runtime, surface, font)
            _save_surface(surface, _case_frame(output, f"items/{kind}", tick))

    command_coverage(
        argparse.Namespace(
            manifest=args.manifest,
            original=output.parent / "original",
            remake=output,
            diff=output.parent / "diff/frames",
            output=output.parent / "coverage.json",
        )
    )
    pygame.quit()
    print(f"remake {args.suite} baseline recorded: {output}")
    return 0


def command_normalize_original(args: argparse.Namespace) -> int:
    """Crop Retina Chrome captures to the source 600x400 SWF viewport."""

    source_files = sorted(args.input.rglob("*.png"))
    ratio = float(args.pixel_ratio)
    crop_width = round(REFERENCE_SIZE[0] * ratio)
    crop_height = round(REFERENCE_SIZE[1] * ratio)
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    updated = 0
    for source in source_files:
        relative = source.relative_to(args.input)
        destination = args.output / relative
        if destination.is_file() and destination.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            continue
        image = Image.open(source).convert("RGBA")
        if image.width < crop_width or image.height < crop_height:
            raise ValueError(
                f"{source} is {image.size}, smaller than the {crop_width}x{crop_height} "
                "Retina source crop"
            )
        normalized = image.crop((0, 0, crop_width, crop_height)).resize(
            REFERENCE_SIZE,
            resampling,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        normalized.save(destination)
        updated += 1
    print(
        f"normalized {updated} updated / {len(source_files)} original frames -> {args.output}"
    )
    return 0


def _mean_error(first: Image.Image, second: Image.Image) -> float:
    difference = ImageChops.difference(first.convert("RGB"), second.convert("RGB"))
    histogram = difference.histogram()
    total = sum(value * (index % 256) for index, value in enumerate(histogram))
    return total / max(1, first.width * first.height * 3)


def _alignment_hint(reference: Image.Image, actual: Image.Image, radius: int = 3) -> dict[str, Any]:
    candidates: list[tuple[float, int, int]] = []
    width, height = reference.size
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            left_a = max(0, dx)
            top_a = max(0, dy)
            left_b = max(0, -dx)
            top_b = max(0, -dy)
            crop_w = width - abs(dx)
            crop_h = height - abs(dy)
            if crop_w <= 0 or crop_h <= 0:
                continue
            first = reference.crop((left_a, top_a, left_a + crop_w, top_a + crop_h))
            second = actual.crop((left_b, top_b, left_b + crop_w, top_b + crop_h))
            candidates.append((_mean_error(first, second), dx, dy))
    error, dx, dy = min(candidates)
    return {"dx": dx, "dy": dy, "mean_error": round(error, 6)}


def compare_pair(reference_path: Path, actual_path: Path, diff_path: Path) -> dict[str, Any]:
    reference = Image.open(reference_path).convert("RGBA")
    actual = Image.open(actual_path).convert("RGBA")
    result: dict[str, Any] = {"size": list(reference.size), "actual_size": list(actual.size)}
    if actual.size != reference.size:
        result.update(status="size-mismatch", changed_ratio=1.0, mean_error=255.0, max_error=255)
        return result
    difference = ImageChops.difference(reference, actual).convert("RGB")
    gray = difference.convert("L")
    histogram = gray.histogram()
    unchanged = histogram[0]
    pixels = reference.width * reference.height
    changed = pixels - unchanged
    rgb_histogram = difference.histogram()
    mean_error = sum(value * (index % 256) for index, value in enumerate(rgb_histogram)) / max(1, pixels * 3)
    extrema = difference.getextrema()
    max_error = max(channel[1] for channel in extrema)
    heat = ImageEnhance.Contrast(difference).enhance(4.0)
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    heat.save(diff_path)
    alignment = (
        _alignment_hint(reference, actual)
        if mean_error <= 32.0
        else {"status": "skipped-high-error", "mean_error": round(mean_error, 6)}
    )
    result.update(
        status="different" if changed else "identical",
        changed_pixels=changed,
        changed_ratio=round(changed / max(1, pixels), 8),
        mean_error=round(mean_error, 6),
        max_error=max_error,
        bbox=list(difference.getbbox()) if difference.getbbox() else None,
        alignment_hint=alignment,
    )
    return result


def command_compare(args: argparse.Namespace) -> int:
    reference_files = image_files(args.reference)
    actual_files = image_files(args.actual)
    paired = sorted(reference_files & actual_files)
    records = []
    for relative in paired:
        metrics = compare_pair(
            args.reference / relative,
            args.actual / relative,
            args.output / "frames" / relative,
        )
        records.append({"frame": relative, **metrics})
    records.sort(key=lambda item: (item.get("changed_ratio", 1.0), item.get("mean_error", 255.0)), reverse=True)
    ratios = [float(item.get("changed_ratio", 1.0)) for item in records]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in records:
        case_id = Path(item["frame"]).parent.as_posix()
        grouped.setdefault(case_id, []).append(item)
    case_summaries = []
    for case_id, items in grouped.items():
        case_summaries.append(
            {
                "case": case_id,
                "frames": len(items),
                "mean_changed_ratio": round(
                    statistics.fmean(float(item.get("changed_ratio", 1.0)) for item in items),
                    8,
                ),
                "mean_error": round(
                    statistics.fmean(float(item.get("mean_error", 255.0)) for item in items),
                    6,
                ),
                "max_error": max(int(item.get("max_error", 255)) for item in items),
            }
        )
    case_summaries.sort(key=lambda item: (item["mean_error"], item["case"]))
    reference_run = args.reference / "run.json"
    actual_run = args.actual / "run.json"
    report = {
        "schema": "glorton-frame-diff-v1",
        "reference": str(args.reference),
        "actual": str(args.actual),
        "paired_frames": len(paired),
        "missing_actual": sorted(reference_files - actual_files),
        "missing_reference": sorted(actual_files - reference_files),
        "reference_metadata": load_json(reference_run) if reference_run.is_file() else None,
        "actual_metadata": load_json(actual_run) if actual_run.is_file() else None,
        "caveats": [
            "The original SWF is executed through Ruffle, not Adobe Flash Player.",
            "Same-name frame metrics are authoritative only after input, camera and temporal phase are aligned.",
            "koAttack reference frames use an instrumented SWF source-timeline fixture because the original normal-input branch is unreachable.",
        ],
        "summary": {
            "mean_changed_ratio": round(statistics.fmean(ratios), 8) if ratios else None,
            "max_changed_ratio": round(max(ratios), 8) if ratios else None,
            "identical_frames": sum(item.get("status") == "identical" for item in records),
        },
        "cases": case_summaries,
        "frames": records,
    }
    write_json(args.output / "report.json", report)
    rows = []
    for item in records[:200]:
        relative = html.escape(item["frame"])
        rows.append(
            "<tr>"
            f"<td>{relative}</td><td>{item.get('changed_ratio', 1):.4%}</td>"
            f"<td>{item.get('mean_error', 255):.3f}</td>"
            f"<td>{html.escape(str(item.get('alignment_hint')))}</td>"
            f"<td><img loading='lazy' src='frames/{relative}' width='300'></td>"
            "</tr>"
        )
    case_rows = []
    for item in case_summaries:
        case_rows.append(
            "<tr>"
            f"<td>{html.escape(item['case'])}</td><td>{item['frames']}</td>"
            f"<td>{item['mean_changed_ratio']:.4%}</td><td>{item['mean_error']:.3f}</td>"
            "</tr>"
        )
    page = (
        "<!doctype html><meta charset='utf-8'><title>Glorton frame diff</title>"
        "<style>body{font:14px system-ui;background:#16181d;color:#eee}table{border-collapse:collapse}"
        "td,th{padding:6px;border:1px solid #444;vertical-align:top}img{image-rendering:auto}</style>"
        f"<h1>Glorton frame diff</h1><p>Paired frames: {len(paired)}</p>"
        "<p>Reference: original SWF via Ruffle. Same-name metrics require matching input, camera and phase. "
        "koAttack uses the explicitly instrumented source-timeline fixture.</p>"
        "<h2>Case summary</h2><table><tr><th>Case</th><th>Frames</th>"
        "<th>Changed</th><th>Mean error</th></tr>"
        + "".join(case_rows)
        + "</table><h2>Frame heatmaps</h2>"
        "<table><tr><th>Frame</th><th>Changed</th><th>Mean error</th><th>Alignment</th><th>Heatmap</th></tr>"
        + "".join(rows)
        + "</table>"
    )
    (args.output / "index.html").write_text(page, encoding="utf-8")
    print(f"compared {len(paired)} frames -> {args.output / 'report.json'}")
    if args.fail_ratio is not None and any(value > args.fail_ratio for value in ratios):
        return 2
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Glorton deterministic visual baseline and frame diff")
    commands = result.add_subparsers(dest="command", required=True)

    coverage = commands.add_parser("coverage", help="write the visual coverage matrix")
    coverage.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    coverage.add_argument("--original", type=Path, default=ROOT / "artifacts/visual/original")
    coverage.add_argument("--remake", type=Path, default=ROOT / "artifacts/visual/remake")
    coverage.add_argument("--diff", type=Path, default=ROOT / "artifacts/visual/diff/frames")
    coverage.add_argument("--output", type=Path, default=ROOT / "artifacts/visual/coverage.json")
    coverage.set_defaults(function=command_coverage)

    prepare = commands.add_parser("prepare-original", help="prepare the local Ruffle SWF runner")
    prepare.add_argument("--swf", type=Path, required=True)
    prepare.add_argument("--ruffle", type=Path, required=True)
    prepare.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    prepare.add_argument("--output", type=Path, default=ROOT / "artifacts/visual/original_runner")
    prepare.add_argument(
        "--baseline-metadata",
        type=Path,
        default=ROOT / "artifacts/visual/original/run.json",
    )
    prepare.set_defaults(function=prepare_runner)

    fixture = commands.add_parser(
        "prepare-ko-fixture",
        help="build an instrumented SWF copy for the unreachable koAttack timeline",
    )
    fixture.add_argument("--swf", type=Path, required=True)
    fixture.add_argument("--ffdec", type=Path, required=True)
    fixture.add_argument("--scripts", type=Path, required=True)
    fixture.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/visual/original_runner/fight-for-glorton-ko.swf",
    )
    fixture.add_argument(
        "--work",
        type=Path,
        default=ROOT / "artifacts/visual/ko_script",
    )
    fixture.set_defaults(function=prepare_ko_fixture)

    record = commands.add_parser("record-remake", help="record deterministic remake PNG sequences")
    record.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    record.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    record.add_argument("--output", type=Path, default=ROOT / "artifacts/visual/remake")
    record.add_argument("--seed", type=int)
    record.add_argument("--quality", choices=("LOW", "MEDIUM", "HIGH"), default="HIGH")
    record.add_argument("--suite", choices=("smoke", "full"), default="smoke")
    record.set_defaults(function=command_record_remake)

    normalize = commands.add_parser(
        "normalize-original",
        help="crop/downsample raw Retina Chrome frames to 600x400",
    )
    normalize.add_argument("--input", type=Path, default=ROOT / "artifacts/visual/original_raw")
    normalize.add_argument("--output", type=Path, default=ROOT / "artifacts/visual/original")
    normalize.add_argument("--pixel-ratio", type=float, default=2.0)
    normalize.set_defaults(function=command_normalize_original)

    compare = commands.add_parser("compare", help="compare matching reference/remake PNG frames")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--actual", type=Path, required=True)
    compare.add_argument("--output", type=Path, default=ROOT / "artifacts/visual/diff")
    compare.add_argument("--fail-ratio", type=float)
    compare.set_defaults(function=command_compare)
    return result


def main() -> int:
    args = parser().parse_args()
    return int(args.function(args))


if __name__ == "__main__":
    raise SystemExit(main())
