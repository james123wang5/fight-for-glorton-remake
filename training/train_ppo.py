from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .peach_env import PeachVsLevel20Env


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="用PPO训练桃子挑战原版20级AI")
    parser.add_argument("--steps", type=int, default=1_000_000, help="AI决策步数；每步=50ms游戏时间")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--name", default="peach_mogadishu_v1")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
    except ImportError as exc:
        raise SystemExit(
            "训练依赖尚未安装。请运行: pip install -r requirements-training.txt"
        ) from exc

    set_random_seed(args.seed)
    output_dir = ROOT / "training" / "checkpoints" / args.name
    log_dir = ROOT / "training" / "logs" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(
        PeachVsLevel20Env(
            seed=args.seed,
            max_episode_seconds=max(1.0, args.max_seconds),
            randomize_spawns=True,
        ),
        info_keywords=("outcome", "agent_kos", "opponent_kos"),
    )
    callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every),
        save_path=str(output_dir),
        name_prefix="ppo_peach",
    )
    try:
        if args.resume is not None:
            model = PPO.load(
                str(args.resume),
                env=env,
                device=args.device,
                tensorboard_log=str(log_dir),
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=2.5e-4,
                n_steps=2048,
                batch_size=256,
                n_epochs=10,
                gamma=0.998,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs={"net_arch": [256, 256]},
                verbose=1,
                seed=args.seed,
                device=args.device,
                tensorboard_log=str(log_dir),
            )
        model.learn(
            total_timesteps=max(1, args.steps),
            callback=callback,
            reset_num_timesteps=args.resume is None,
            tb_log_name=args.name,
        )
        final_path = output_dir / "final_model"
        model.save(str(final_path))
        metadata = {
            "scenario": "peach-vs-level20-mogadishu-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "requested_steps": args.steps,
            "seed": args.seed,
            "max_episode_seconds": args.max_seconds,
            "frame_skip": 2,
            "physics_hz": 40,
            "policy_hz": 20,
            "device": args.device,
            "normal_game_modified": False,
        }
        (output_dir / "training_config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n训练完成: {final_path}.zip")
        print(f"评估命令: python -m training.evaluate --checkpoint {final_path}.zip --episodes 100")
    finally:
        env.close()


if __name__ == "__main__":
    main()
