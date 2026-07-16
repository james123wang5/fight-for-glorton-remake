from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "training" / "checkpoints" / "peach_mogadishu_v1" / "final_model.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description="启动游戏并把训练模型开放为21级AI")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f"找不到训练模型: {model}")
    os.environ["GLORTON_AI21_MODEL"] = str(model)

    from src.runtime import main as run_game

    print(f"21级AI模型: {model}")
    print("试玩建议: Single Player -> One on One；双方选 Peach；CPU 调到 21；STOCK 3；地图 Mogadishu。")
    print("原版1至20级仍使用原来的规则AI。")
    run_game()


if __name__ == "__main__":
    main()
