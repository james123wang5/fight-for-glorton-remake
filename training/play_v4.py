from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "training" / "checkpoints" / "peach_active_v4"


def _manifest(directory: Path) -> dict[str, Any]:
    path = directory / "champions.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_model(directory: Path, level: int, *, allow_candidate: bool) -> Path:
    entry = _manifest(directory).get("levels", {}).get(str(level), {})
    if entry.get("qualified"):
        champion = directory / str(entry.get("path", f"champion_level{level}_model.zip"))
        if champion.is_file():
            return champion.resolve()
    if allow_candidate:
        candidate = directory / f"candidate_level{level}_model.zip"
        if candidate.is_file():
            return candidate.resolve()
    qualifier = "。如果只想验收候选模型，加 --allow-candidate" if not allow_candidate else ""
    raise FileNotFoundError(f"找不到{level}级可用v4模型: {directory}{qualifier}")


def main() -> None:
    parser = argparse.ArgumentParser(description="启动游戏并试玩v4主动战斗21/22级AI")
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--level21", type=Path)
    parser.add_argument("--level22", type=Path)
    parser.add_argument(
        "--allow-candidate",
        action="store_true",
        help="允许加载未通过自动门槛的candidate，仅供人工验收",
    )
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve()
    try:
        level21 = (
            args.level21.expanduser().resolve()
            if args.level21 is not None
            else resolve_model(directory, 21, allow_candidate=args.allow_candidate)
        )
        level22 = (
            args.level22.expanduser().resolve()
            if args.level22 is not None
            else resolve_model(directory, 22, allow_candidate=args.allow_candidate)
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    for level, path in ((21, level21), (22, level22)):
        if not path.is_file():
            raise SystemExit(f"找不到{level}级v4模型: {path}")

    os.environ["GLORTON_AI21_MODEL"] = str(level21)
    os.environ["GLORTON_AI22_MODEL"] = str(level22)
    os.environ["GLORTON_AI21_HUMANIZED"] = "1"
    os.environ["GLORTON_AI_V4"] = "1"
    os.environ.pop("GLORTON_AI_TACTICAL", None)

    from src.runtime import main as run_game

    print(f"v4主动战斗21级: {level21}")
    print(f"v4主动战斗22级: {level22}")
    if args.allow_candidate:
        print("当前允许candidate：这只是人工验收，不代表已通过冠军门槛。")
    print("建议试玩: Peach vs Peach、STOCK 3、Mogadishu、关闭道具。")
    print("普通play.py、v2和v3启动器不受影响。")
    run_game()


if __name__ == "__main__":
    main()
