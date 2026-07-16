from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "training" / "checkpoints" / "peach_tactical_v3"


def _default_model(level: int) -> Path:
    best = DEFAULT_DIR / f"best_level{level}_model.zip"
    return best if best.is_file() else DEFAULT_DIR / f"level{level}_model.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="启动游戏并试玩v3战术21/22级AI")
    parser.add_argument("--level21", type=Path, default=_default_model(21))
    parser.add_argument("--level22", type=Path, default=_default_model(22))
    args = parser.parse_args()
    level21 = args.level21.expanduser().resolve()
    level22 = args.level22.expanduser().resolve()
    for level, model in ((21, level21), (22, level22)):
        if not model.is_file():
            raise SystemExit(
                f"找不到{level}级v3模型: {model}\n"
                "请先运行 python -m training.train_tactical 训练。"
            )
    os.environ["GLORTON_AI21_MODEL"] = str(level21)
    os.environ["GLORTON_AI22_MODEL"] = str(level22)
    os.environ["GLORTON_AI21_HUMANIZED"] = "1"
    os.environ["GLORTON_AI_TACTICAL"] = "1"

    from src.runtime import main as run_game

    print(f"v3战术21级: {level21}")
    print(f"v3战术22级: {level22}")
    print("建议试玩: Peach vs Peach、STOCK 3、Mogadishu。")
    print("本启动器优先读取best模型；普通play.py和v2启动器不受影响。")
    run_game()


if __name__ == "__main__":
    main()
