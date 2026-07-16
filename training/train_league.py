from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .league_env import (
    LEAGUE_OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    PeachLeagueEnv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = ROOT / "training" / "checkpoints" / "peach_mogadishu_v1" / "final_model.zip"


def _transfer_legacy_policy(legacy: Any, learner: Any) -> dict[str, int]:
    """Copy the 142-input level-21 policy into the expanded 180-input policy."""

    source = legacy.policy.state_dict()
    target = learner.policy.state_dict()
    copied = 0
    expanded = 0
    for key, target_value in target.items():
        source_value = source.get(key)
        if source_value is None:
            continue
        if source_value.shape == target_value.shape:
            target[key] = source_value.detach().clone()
            copied += 1
            continue
        if (
            source_value.ndim == 2
            and target_value.ndim == 2
            and source_value.shape[0] == target_value.shape[0]
            and source_value.shape[1] < target_value.shape[1]
            and key.endswith(".0.weight")
        ):
            value = target_value.detach().clone()
            value.zero_()
            value[:, : source_value.shape[1]] = source_value
            target[key] = value
            expanded += 1
    learner.policy.load_state_dict(target)
    return {"exact_tensors": copied, "expanded_input_tensors": expanded}


def _new_model(PPO: Any, env: Any, *, seed: int, device: str, log_dir: Path) -> Any:
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.998,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.015,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=str(log_dir),
    )


def _evaluate_pair(
    model_a: Any,
    model_b: Any,
    *,
    episodes: int,
    seed: int,
    max_seconds: float,
) -> dict[str, Any]:
    env = PeachLeagueEnv(
        seed=seed,
        max_episode_seconds=max_seconds,
        items_probability=0.30,
        recovery_start_probability=0.12,
    )
    env.set_opponent_pool(
        primary=model_b,
        primary_weight=1.0,
        teacher_weight=0.0,
        probe_weight=0.0,
    )
    env.opponent_deterministic = True
    outcomes: dict[str, int] = {}
    attacks: dict[str, int] = {}
    try:
        for episode in range(max(1, episodes)):
            obs, _info = env.reset(seed=seed + episode)
            terminated = truncated = False
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action, _state = model_a.predict(obs, deterministic=True)
                obs, _reward, terminated, truncated, info = env.step(action)
            outcome = str(info.get("outcome", "unknown"))
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            for label, count in info.get("successful_attacks", {}).items():
                attacks[label] = attacks.get(label, 0) + int(count)
    finally:
        env.close()
    return {"outcomes": outcomes, "successful_attacks": attacks}


def _save_pair(model21: Any, model22: Any, output_dir: Path, *, round_no: int) -> None:
    model21.save(str(output_dir / "level21_model"))
    model22.save(str(output_dir / "level22_model"))
    model21.save(str(output_dir / f"round_{round_no:03d}_level21"))
    model22.save(str(output_dir / f"round_{round_no:03d}_level22"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="轮流训练21/22级桃子AI：楼房地图、自对战、真人式按键和技能组合"
    )
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument(
        "--steps-per-round",
        type=int,
        default=100_000,
        help="每轮每个模型的决策步数；10轮默认各训100万步",
    )
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--name", default="peach_league_v2")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    parser.add_argument("--items-probability", type=float, default=0.30)
    parser.add_argument("--teacher-weight", type=float, default=0.15)
    parser.add_argument("--probe-weight", type=float, default=0.10)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-max-seconds", type=float, default=90.0)
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="从目录中的level21_model.zip/level22_model.zip继续联赛",
    )
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
    except ImportError as exc:
        raise SystemExit(
            "训练依赖尚未安装。请运行: pip install -r requirements-training.txt"
        ) from exc

    teacher_path = args.teacher.expanduser().resolve()
    if not teacher_path.is_file():
        raise SystemExit(f"找不到冻结21级老师模型: {teacher_path}")
    if args.rounds < 1 or args.steps_per_round < 1:
        raise SystemExit("--rounds 和 --steps-per-round 必须大于0")

    set_random_seed(args.seed)
    output_dir = ROOT / "training" / "checkpoints" / args.name
    log_dir = ROOT / "training" / "logs" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    teacher = PPO.load(str(teacher_path), device="cpu")
    raw21 = PeachLeagueEnv(
        seed=args.seed,
        max_episode_seconds=max(1.0, args.max_seconds),
        items_probability=args.items_probability,
    )
    raw22 = PeachLeagueEnv(
        seed=args.seed + 1_000_000,
        max_episode_seconds=max(1.0, args.max_seconds),
        items_probability=args.items_probability,
    )
    env21 = Monitor(raw21, info_keywords=("outcome", "agent_kos", "opponent_kos"))
    env22 = Monitor(raw22, info_keywords=("outcome", "agent_kos", "opponent_kos"))

    transfer: dict[str, Any] = {}
    if args.resume_dir is not None:
        resume_dir = args.resume_dir.expanduser().resolve()
        model21 = PPO.load(
            str(resume_dir / "level21_model.zip"),
            env=env21,
            device=args.device,
            tensorboard_log=str(log_dir),
        )
        model22 = PPO.load(
            str(resume_dir / "level22_model.zip"),
            env=env22,
            device=args.device,
            tensorboard_log=str(log_dir),
        )
        transfer["mode"] = "resume"
        transfer["source"] = str(resume_dir)
    else:
        model21 = _new_model(PPO, env21, seed=args.seed + 21, device=args.device, log_dir=log_dir)
        model22 = _new_model(PPO, env22, seed=args.seed + 22, device=args.device, log_dir=log_dir)
        transfer["mode"] = "expanded_legacy_policy"
        transfer["level21"] = _transfer_legacy_policy(teacher, model21)
        transfer["level22"] = _transfer_legacy_policy(teacher, model22)

    peer_weight = max(0.0, 1.0 - args.teacher_weight - args.probe_weight)
    raw21.set_opponent_pool(
        primary=model22,
        teacher=teacher,
        primary_weight=peer_weight,
        teacher_weight=args.teacher_weight,
        probe_weight=args.probe_weight,
    )
    raw22.set_opponent_pool(
        primary=model21,
        teacher=teacher,
        primary_weight=peer_weight,
        teacher_weight=args.teacher_weight,
        probe_weight=args.probe_weight,
    )

    history: list[dict[str, Any]] = []
    interrupted = False
    try:
        for round_no in range(1, args.rounds + 1):
            print(f"\n===== 联赛第 {round_no}/{args.rounds} 轮：先更新21级 =====")
            model21.learn(
                total_timesteps=args.steps_per_round,
                reset_num_timesteps=False,
                tb_log_name=f"{args.name}_level21",
            )
            # The freshly updated 21 becomes 22's opponent immediately. This
            # alternating freeze is intentional: truly changing both policies
            # during one rollout makes PPO's target distribution unstable.
            print(f"\n===== 联赛第 {round_no}/{args.rounds} 轮：再更新22级 =====")
            model22.learn(
                total_timesteps=args.steps_per_round,
                reset_num_timesteps=False,
                tb_log_name=f"{args.name}_level22",
            )
            _save_pair(model21, model22, output_dir, round_no=round_no)
            evaluation = {
                "round": round_no,
                "level21_as_learner": _evaluate_pair(
                    model21,
                    model22,
                    episodes=args.eval_episodes,
                    seed=args.seed + round_no * 10_000,
                    max_seconds=args.eval_max_seconds,
                ),
                "level22_as_learner": _evaluate_pair(
                    model22,
                    model21,
                    episodes=args.eval_episodes,
                    seed=args.seed + round_no * 10_000 + 5_000,
                    max_seconds=args.eval_max_seconds,
                ),
            }
            history.append(evaluation)
            print(json.dumps(evaluation, ensure_ascii=False, indent=2))
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到中断，正在保存当前21/22级模型……")
    finally:
        model21.save(str(output_dir / "level21_model"))
        model22.save(str(output_dir / "level22_model"))
        env21.close()
        env22.close()

    metadata = {
        "scenario": OBSERVATION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interrupted": interrupted,
        "teacher": str(teacher_path),
        "rounds_requested": args.rounds,
        "steps_per_round_per_model": args.steps_per_round,
        "requested_steps_per_model": args.rounds * args.steps_per_round,
        "seed": args.seed,
        "max_episode_seconds": args.max_seconds,
        "observation_size": LEAGUE_OBSERVATION_SIZE,
        "physics_hz": 40,
        "policy_hz": 20,
        "reaction_delay_ms": 100,
        "direction_commitment_ms": 100,
        "shield_rearm_ms": 200,
        "items_episode_probability": args.items_probability,
        "opponent_weights": {
            "current_peer": peer_weight,
            "frozen_level21": args.teacher_weight,
            "idle_and_retreat_probes": args.probe_weight,
        },
        "transfer": transfer,
        "normal_game_modified": False,
        "evaluation_history": history,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\n21级模型: {output_dir / 'level21_model.zip'}")
    print(f"22级模型: {output_dir / 'level22_model.zip'}")
    print("训练可视化试玩: python -m training.play_league")


if __name__ == "__main__":
    main()
