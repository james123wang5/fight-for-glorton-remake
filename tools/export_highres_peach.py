from __future__ import annotations

import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from build_manifest import (
    DEFAULT_SOURCE,
    MENU_FIGHTERS,
    ROOT,
    SYMBOLS,
    build_display_frames,
    extract_sprite_timeline,
    find_sprite,
    sprite_subtags,
)


SWF_PATH = ROOT.parent / "fight-for-glorton.swf"
FFDEC_JAR = ROOT.parent / "FFDec.app/Contents/Resources/ffdec-cli.jar"
OUTPUT = ROOT / "assets/ffdec_zoom4"


def find_java() -> Path:
    candidates = [
        Path("/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home/bin/java"),
        Path(shutil.which("java") or ""),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit("Java was not found; FFDec CLI requires a Java runtime.")


def dependencies(root: ET.Element, root_ids: list[int]) -> tuple[set[int], set[int]]:
    pending = list(root_ids)
    sprites: set[int] = set()
    shapes: set[int] = set()
    while pending:
        character_id = pending.pop()
        if character_id in sprites or character_id in shapes:
            continue
        try:
            tags = sprite_subtags(root, character_id)
        except ValueError:
            shapes.add(character_id)
            continue
        sprites.add(character_id)
        for tag in tags:
            if tag.attrib.get("placeFlagHasCharacter") != "true":
                continue
            child_id = int(tag.attrib.get("characterId", "0"))
            if child_id > 0:
                pending.append(child_id)
    return sprites, shapes


def fighter_state_dependencies(root: ET.Element) -> tuple[set[int], set[int]]:
    state_ids: list[int] = []
    for fighter_id in MENU_FIGHTERS.values():
        fighter_timeline = extract_sprite_timeline(find_sprite(root, fighter_id))
        fighter_frames = build_display_frames(root, fighter_id)
        for label in fighter_timeline["labels"]:
            frame_index = int(label["frame"]) - 1
            if 0 <= frame_index < len(fighter_frames):
                state_ids.extend(int(place["character_id"]) for place in fighter_frames[frame_index])
    return dependencies(root, state_ids)


def export(kind: str, ids: set[int], destination: Path) -> None:
    if not ids:
        return
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        str(find_java()),
        "-jar",
        str(FFDEC_JAR),
        "-config",
        "animateSubsprites=false",
        "-zoom",
        "4",
        "-selectid",
        ",".join(str(item) for item in sorted(ids)),
        "-format",
        f"{kind}:png",
        "-export",
        kind,
        str(destination),
        str(SWF_PATH),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    if not SWF_PATH.is_file() or not FFDEC_JAR.is_file():
        raise SystemExit("fight-for-glorton.swf or FFDec.app is missing beside the project folder.")
    root = ET.parse(DEFAULT_SOURCE / "raw_ffdec_xml/fight-for-glorton.xml").getroot()
    stages_only = "--stages-only" in sys.argv[1:]
    sprites: set[int] = set()
    shapes: set[int] = set()
    if not stages_only:
        sprites, shapes = fighter_state_dependencies(root)
    stage_roots = [
        SYMBOLS[name]
        for name in (
            "rooftop",
            "rooftop_background",
            "mogadishu",
            "mogadishu_background",
            "b52",
            "b52_background",
            "space",
            "space_background",
        )
    ]
    stage_sprites, stage_shapes = dependencies(root, stage_roots)
    for stage_name in ("rooftop", "mogadishu", "b52", "space"):
        stage_sprites.discard(SYMBOLS[stage_name])
    sprites.update(stage_sprites)
    shapes.update(stage_shapes)
    if not stages_only:
        combat_roots = [
            SYMBOLS[name]
            for name in (
                "mine",
                "grenade",
                "pencil",
                "poop",
                "garbage",
                "energy_ball",
                "puff",
                "spawn1",
                "spawn2",
                "player_death",
                "punch_damage",
                "camera_trick",
                "pos_indicator",
                "far_indicator",
                "item_indicator",
                "boom_star",
                "boom_wave",
                "boom_matter",
                "shield",
                "osd_bigicon",
                "osd_life_graphic",
                "osd_damage",
            )
        ]
        combat_sprites, combat_shapes = dependencies(root, combat_roots)
        sprites.update(combat_sprites)
        shapes.update(combat_shapes)
    export("sprite", sprites, OUTPUT / "sprites")
    export("shape", shapes, OUTPUT / "shapes")
    print(f"Exported {len(sprites)} sprites and {len(shapes)} shapes to {OUTPUT}")


if __name__ == "__main__":
    main()
