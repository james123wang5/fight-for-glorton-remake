from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "training" / "checkpoints" / "peach_league_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="启动游戏并开放双模型21/22级联赛AI")
    parser.add_argument("--level21", type=Path, default=DEFAULT_DIR / "level21_model.zip")
    parser.add_argument("--level22", type=Path, default=DEFAULT_DIR / "level22_model.zip")
    args = parser.parse_args()
    level21 = args.level21.expanduser().resolve()
    level22 = args.level22.expanduser().resolve()
    for level, model in ((21, level21), (22, level22)):
        if not model.is_file():
            raise SystemExit(f"找不到{level}级联赛模型: {model}")
    os.environ["GLORTON_AI21_MODEL"] = str(level21)
    os.environ["GLORTON_AI21_HUMANIZED"] = "1"
    os.environ["GLORTON_AI22_MODEL"] = str(level22)

    from src.runtime import main as run_game

    print(f"21级联赛模型: {level21}")
    print(f"22级联赛模型: {level22}")
    print("试玩建议: Peach vs Peach、STOCK 3、Mogadishu，CPU选21或22。")
    print("普通play.py仍只开放原版1至20级，不受训练配置影响。")
    run_game()


if __name__ == "__main__":
    main()
