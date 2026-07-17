from __future__ import annotations

import argparse
import os
from pathlib import Path

from .roster_contract import FIGHTER_ORDER


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "training" / "checkpoints" / "roster_v6"
DEFAULT_PEACH_DIR = ROOT / "training" / "checkpoints" / "peach_purpose_v5"
TRAINABLE_ROSTER = tuple(name for name in FIGHTER_ORDER if name != "PeachPlayer")


def candidate_path(
    root: Path,
    fighter_name: str,
    *,
    stage_name: str,
    run_id: str,
    level: int,
) -> Path:
    role = fighter_name.removesuffix("Player").lower()
    return (
        root
        / role
        / stage_name.lower()
        / run_id
        / f"candidate_level{level}_model.zip"
    ).resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="启动普通游戏并按所选角色加载roster_v6的21/22级候选"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-id", default="roster_b1")
    parser.add_argument("--stage", choices=("Mogadishu",), default="Mogadishu")
    parser.add_argument(
        "--peach-directory",
        type=Path,
        default=DEFAULT_PEACH_DIR,
        help="桃子21/22级模型目录",
    )
    parser.add_argument(
        "--prefer-candidate",
        action="store_true",
        help="桃子也优先使用candidate；其他五个角色本轮本来就是candidate",
    )
    parser.add_argument(
        "--record-human",
        action="store_true",
        help="保存真人对战输入和状态，供后续离线学习",
    )
    parser.add_argument(
        "--human-replay-dir",
        type=Path,
        default=ROOT / "training" / "replays" / "human_roster_v6",
    )
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    peach_directory = args.peach_directory.expanduser().resolve()

    missing: list[Path] = []
    for fighter_name in TRAINABLE_ROSTER:
        for level in (21, 22):
            path = candidate_path(
                root,
                fighter_name,
                stage_name=args.stage,
                run_id=args.run_id,
                level=level,
            )
            if not path.is_file():
                missing.append(path)
    if missing:
        details = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"缺少全角色候选模型:\n{details}")

    peach_models: dict[int, Path] = {}
    for level in (21, 22):
        filename = (
            f"candidate_level{level}_model.zip"
            if args.prefer_candidate
            else f"champion_level{level}_model.zip"
        )
        model_path = (peach_directory / filename).resolve()
        if not model_path.is_file():
            raise SystemExit(f"缺少桃子{level}级模型: {model_path}")
        peach_models[level] = model_path

    # training.peach_env selects SDL's dummy driver when imported before model
    # paths are configured. This playable entry deliberately stays independent
    # of training preflight imports and always restores the native Mac drivers.
    os.environ.pop("SDL_VIDEODRIVER", None)
    os.environ.pop("SDL_AUDIODRIVER", None)
    os.environ["GLORTON_AI21_MODEL"] = str(peach_models[21])
    os.environ["GLORTON_AI22_MODEL"] = str(peach_models[22])
    os.environ["GLORTON_AI_ROSTER"] = "1"
    os.environ["GLORTON_AI_ROSTER_ROOT"] = str(root)
    os.environ["GLORTON_AI_ROSTER_RUN_ID"] = args.run_id
    os.environ["GLORTON_AI_ROSTER_STAGE"] = args.stage
    os.environ["GLORTON_AI_V5"] = "1"
    for name in (
        "GLORTON_AI_V4",
        "GLORTON_AI_TACTICAL",
        "GLORTON_AI_V5_WEB",
        "GLORTON_AUTOSTART_MATCH_JSON",
        "GLORTON_FORCE_WINDOW_FOCUS",
    ):
        os.environ.pop(name, None)
    if args.record_human:
        replay_dir = args.human_replay_dir.expanduser().resolve()
        replay_dir.mkdir(parents=True, exist_ok=True)
        os.environ["GLORTON_HUMAN_REPLAY_DIR"] = str(replay_dir)

    from src.runtime import main as run_game

    print("全六角色21/22级试玩入口已经加载。")
    print("会像普通游戏一样打开完整主菜单，不会自动开战。")
    print("训练地图: Mogadishu（楼房）")
    print("流程: MULTIPLAYER → 选择真人/CP → 选择任意角色 → CP等级设为21/22")
    print("桃子读取原v5模型；其他五个角色分别读取自己训练完成的roster_v6模型。")
    if args.record_human:
        print(f"真人学习录像已开启: {os.environ['GLORTON_HUMAN_REPLAY_DIR']}")
    run_game()


if __name__ == "__main__":
    main()
