from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "training" / "checkpoints" / "peach_purpose_v5"


def _manifest(directory: Path) -> dict[str, Any]:
    path = directory / "champions.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def resolve_model(
    directory: Path,
    level: int,
    *,
    allow_candidate: bool,
    prefer_candidate: bool = False,
) -> Path:
    if prefer_candidate:
        preferred = []
        if level == 22:
            preferred.append(directory / "human_candidate_level22_model.zip")
        preferred.append(directory / f"candidate_level{level}_model.zip")
        for candidate in preferred:
            if candidate.is_file():
                return candidate.resolve()
    entry = _manifest(directory).get("levels", {}).get(str(level), {})
    if entry.get("qualified"):
        champion = directory / str(entry.get("path", f"champion_level{level}_model.zip"))
        if champion.is_file():
            return champion.resolve()
    if allow_candidate:
        candidate = directory / f"candidate_level{level}_model.zip"
        if candidate.is_file():
            return candidate.resolve()
    suffix = "；候选模型请显式加 --allow-candidate" if not allow_candidate else ""
    raise FileNotFoundError(f"找不到{level}级可用v5模型: {directory}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description="启动游戏并试玩v5目的驱动21/22级AI")
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--level21", type=Path)
    parser.add_argument("--level22", type=Path)
    parser.add_argument("--allow-candidate", action="store_true")
    parser.add_argument(
        "--prefer-candidate",
        action="store_true",
        help="优先试玩独立候选：22级先选真人模仿候选，不覆盖冠军",
    )
    parser.add_argument(
        "--record-human",
        action="store_true",
        help="保存真人对战的逐帧输入和状态，供后续离线模仿/强化训练",
    )
    parser.add_argument(
        "--human-replay-dir",
        type=Path,
        default=ROOT / "training" / "replays" / "human_v5",
    )
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve()
    try:
        level21 = (
            args.level21.expanduser().resolve()
            if args.level21 is not None
            else resolve_model(
                directory,
                21,
                allow_candidate=args.allow_candidate,
                prefer_candidate=args.prefer_candidate,
            )
        )
        level22 = (
            args.level22.expanduser().resolve()
            if args.level22 is not None
            else resolve_model(
                directory,
                22,
                allow_candidate=args.allow_candidate,
                prefer_candidate=args.prefer_candidate,
            )
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    for level, model in ((21, level21), (22, level22)):
        if not model.is_file():
            raise SystemExit(f"找不到{level}级v5模型: {model}")

    os.environ["GLORTON_AI21_MODEL"] = str(level21)
    os.environ["GLORTON_AI22_MODEL"] = str(level22)
    os.environ["GLORTON_AI21_HUMANIZED"] = "1"
    os.environ["GLORTON_AI_V5"] = "1"
    os.environ.pop("GLORTON_AI_V4", None)
    os.environ.pop("GLORTON_AI_TACTICAL", None)
    if args.record_human:
        replay_dir = args.human_replay_dir.expanduser().resolve()
        replay_dir.mkdir(parents=True, exist_ok=True)
        os.environ["GLORTON_HUMAN_REPLAY_DIR"] = str(replay_dir)

    from src.runtime import main as run_game

    print(f"v5目的驱动21级: {level21}")
    print(f"v5目的驱动22级: {level22}")
    if args.allow_candidate:
        print("当前允许candidate：仅用于人工验收，不代表已通过冠军门槛。")
    if args.prefer_candidate:
        print("当前优先候选模型：仅用于实战验收，冻结冠军未改写。")
    print("建议: Peach vs Peach、STOCK 3、Mogadishu、关闭道具。")
    print("重点观察：越墙、空中横移手刀、受击第一帧脱离和无效二段跳。")
    if args.record_human:
        print(f"真人学习素材录制已开启: {os.environ['GLORTON_HUMAN_REPLAY_DIR']}")
        print("每局结束自动保存；当前冠军不会在对局中被即时改写。")
    run_game()


if __name__ == "__main__":
    main()
