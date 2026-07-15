from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .common import print_summary, run_policy_episodes
from .peach_env import PeachVsLevel20Env


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="评估PPO桃子对原版20级AI的胜率")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30260714)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    parser.add_argument("--record", type=int, default=0, help="保存前N局的确定性输入录像")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit(
            "训练依赖尚未安装。请运行: pip install -r requirements-training.txt"
        ) from exc

    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.exists() and checkpoint.suffix != ".zip":
        checkpoint = checkpoint.with_suffix(".zip")
    if not checkpoint.exists():
        raise SystemExit(f"找不到模型: {checkpoint}")
    model = PPO.load(str(checkpoint), device=args.device)

    def policy(observation: np.ndarray, _: PeachVsLevel20Env, __: int) -> np.ndarray:
        action, _state = model.predict(observation, deterministic=True)
        return np.asarray(action, dtype=np.int64)

    record_dir = ROOT / "training" / "replays" / checkpoint.stem
    results, wall_seconds = run_policy_episodes(
        policy,
        episodes=max(1, args.episodes),
        seed=args.seed,
        max_episode_seconds=max(1.0, args.max_seconds),
        record_count=max(0, args.record),
        record_dir=record_dir,
    )
    print_summary(results, wall_seconds, label=f"模型评估: {checkpoint.name}")
    if args.record > 0:
        print(f"录像目录: {record_dir}")


if __name__ == "__main__":
    main()
