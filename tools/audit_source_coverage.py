from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CLASSES = ROOT.parent / "glorton_peach/raw_ffdec_export_scripts/scripts/__Packages"


@dataclass(frozen=True)
class Coverage:
    target: str
    token: str
    note: str


COVERAGE = {
    "AIControl": Coverage("src/runtime.py", "class AIController", "AI target scoring, movement and delayed attacks"),
    "AuberginePlayer": Coverage("tools/build_manifest.py", '"AuberginePlayer"', "selectable fighter bundle"),
    "Bullet": Coverage("src/runtime.py", "class Bullet", "Peach bullet lifecycle and hit response"),
    "CoffeePlayer": Coverage("tools/build_manifest.py", '"CoffeePlayer"', "selectable fighter bundle"),
    "DefaultPlayer": Coverage("tools/build_manifest.py", '"DefaultPlayer"', "selectable fighter bundle"),
    "EnergyBall": Coverage("src/runtime.py", 'self.kind == "EnergyBall"', "bounce, sine movement and projectile attribution"),
    "Explosion": Coverage("src/runtime.py", "class ExplosionEffect", "damage radius and three source timelines"),
    "Fighter": Coverage("src/runtime.py", "class PeachFighter", "shared fighter state machine for all six fighters"),
    "Garbage": Coverage("src/runtime.py", 'kind="Garbage"', "normal and 20-way radial variants"),
    "Grenade": Coverage("src/runtime.py", 'item.kind == "Grenade"', "pickup, throw and explosion"),
    "Item": Coverage("src/runtime.py", "class StageItem", "shared item state machine"),
    "ItemGen": Coverage("src/runtime.py", "def _fixed_tick_items", "source 5 Hz generator and probability"),
    "KeyCombi": Coverage("src/runtime.py", "class FighterInput", "keydown, keyup and key-hold traces"),
    "KeyCombiKey": Coverage("src/runtime.py", "up_hold_ms", "per-key hold duration"),
    "Mine": Coverage("src/runtime.py", 'item.kind == "Mine"', "pickup, sticking, activation and explosion"),
    "OSD": Coverage("src/runtime.py", "def _draw_osd", "damage, stock, score and far indicators"),
    "PeachPlayer": Coverage("tools/build_manifest.py", '"PeachPlayer"', "selectable fighter bundle"),
    "Pencil": Coverage("src/runtime.py", '"Pencil": "pencil"', "Aubergine projectile"),
    "PlayerControl": Coverage("src/runtime.py", "def _handle_keydown", "four local control slots and reflex window"),
    "Poop": Coverage("src/runtime.py", '"Poop": "poop"', "Coffee projectile"),
    "Projectile": Coverage("src/runtime.py", "class SpecialProjectile", "shared projectile lifecycle"),
    "Rocket": Coverage("src/runtime.py", "class RocketProjectile", "Peach rocket trajectory and explosion"),
    "SBLPlayer": Coverage("tools/build_manifest.py", '"SBLPlayer"', "selectable fighter bundle"),
    "Snd": Coverage("src/audio.py", "class AudioManager", "source sound registry and channels"),
    "TrashPlayer": Coverage("tools/build_manifest.py", '"TrashPlayer"', "selectable fighter bundle"),
    "VCamera": Coverage("src/runtime.py", "def _camera_target", "focus union, ratio, smoothing and bounds"),
    "World": Coverage("src/runtime.py", "class Stage", "platform, helper, hazard and attack collision world"),
}

UNREACHABLE = {
    "RedGuy": "Registered as symbol 700, but never placed by the root timeline or any selectable-fighter list.",
}


def audit() -> list[str]:
    errors: list[str] = []
    source_names = {path.stem for path in SOURCE_CLASSES.glob("*.as")}
    classified = set(COVERAGE) | set(UNREACHABLE)
    for name in sorted(source_names - classified):
        errors.append(f"unclassified source class: {name}")
    for name in sorted(classified - source_names):
        errors.append(f"classification has no source class: {name}")

    files: dict[Path, str] = {}
    for name, coverage in COVERAGE.items():
        target = ROOT / coverage.target
        if target not in files:
            files[target] = target.read_text(encoding="utf-8")
        if coverage.token not in files[target]:
            errors.append(f"{name}: missing token {coverage.token!r} in {coverage.target}")
    return errors


def main() -> None:
    errors = audit()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Covered runtime classes: {len(COVERAGE)}")
    for name, coverage in COVERAGE.items():
        print(f"  {name:18} -> {coverage.target}: {coverage.note}")
    print(f"Source-only unreachable classes: {len(UNREACHABLE)}")
    for name, note in UNREACHABLE.items():
        print(f"  {name:18} -> {note}")


if __name__ == "__main__":
    main()
