from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .tactical_env import (
    TACTICAL_OBSERVATION_SIZE,
    TACTICAL_OBSERVATION_VERSION,
    TACTICAL_POLICY_HZ,
    TacticalPeachEnv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = ROOT / "training" / "checkpoints" / "peach_league_v2" / "level21_model.zip"
QUALITY_KEYS = (
    "action_change_rate",
    "projectiles_per_minute",
    "projectile_accuracy",
    "shield_activations_per_minute",
    "false_shield_rate",
    "shield_block_precision",
    "melee_hit_rate",
    "far_fraction",
)


def _new_model(MaskablePPO: Any, env: Any, *, seed: int, device: str, log_dir: Path) -> Any:
    # V2 used a comparatively large entropy bonus and selected a raw key state
    # twenty times per second. V3 selects a legal tactical intention at 10 Hz,
    # so a smaller entropy term is enough to explore without teaching constant
    # weapon/shield switching.
    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=2.0e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.998,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.002,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=str(log_dir),
    )


def _curriculum_strength(round_no: int, rounds: int) -> float:
    if rounds <= 1:
        return 0.70
    progress = (round_no - 1) / (rounds - 1)
    return 0.70 + (0.20 - 0.70) * progress


def _weighted_pool(
    *,
    peer: Any,
    teacher: Any,
    history: Iterable[tuple[Any, str]],
) -> list[tuple[Any, float, str]]:
    frozen = list(history)
    if not frozen:
        return [(peer, 0.75, "current_peer"), (teacher, 0.25, "frozen_v2_level21")]
    history_weight = 0.35 / len(frozen)
    return [
        (peer, 0.50, "current_peer"),
        (teacher, 0.15, "frozen_v2_level21"),
        *((policy, history_weight, name) for policy, name in frozen),
    ]


def _outcome_points(outcome: str) -> float:
    return {
        "win": 1.0,
        "timeout_win": 0.35,
        "loss": -1.0,
        "timeout_loss": -0.35,
        "draw": 0.0,
        "timeout_draw": 0.0,
    }.get(outcome, 0.0)


def _quality_score(outcome_score: float, quality: dict[str, float]) -> float:
    """One comparable number, with wins primary and visible spam expensive."""

    accuracy = min(0.60, quality["projectile_accuracy"])
    melee = min(0.60, quality["melee_hit_rate"])
    shield_precision = min(1.0, quality["shield_block_precision"])
    score = outcome_score
    score += 0.55 * accuracy
    score += 0.20 * melee
    score += 0.10 * shield_precision
    score -= 1.10 * max(0.0, quality["action_change_rate"] - 0.35)
    score -= 0.018 * max(0.0, quality["projectiles_per_minute"] - 18.0)
    score -= 0.018 * max(0.0, quality["shield_activations_per_minute"] - 12.0)
    score -= 0.25 * quality["false_shield_rate"]
    score -= 0.10 * quality["far_fraction"]
    return float(score)


def evaluate_model(
    model: Any,
    opponents: list[tuple[Any, str]],
    *,
    episodes: int,
    seed: int,
    max_seconds: float,
) -> dict[str, Any]:
    """Deterministic duel evaluation across peers, v2 and both spawn slots."""

    env = TacticalPeachEnv(
        seed=seed,
        max_episode_seconds=max_seconds,
        items_probability=0.0,
        curriculum_strength=0.0,
    )
    env.opponent_deterministic = True
    outcomes: Counter[str] = Counter()
    outcomes_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    metric_totals: Counter[str] = Counter()
    successful_attacks: Counter[str] = Counter()
    rewards: list[float] = []
    outcome_values: list[float] = []
    try:
        for episode in range(max(1, int(episodes))):
            opponent, source = opponents[episode % len(opponents)]
            env.set_tactical_opponent_pool([(opponent, 1.0, source)])
            obs, _info = env.reset(
                seed=seed + episode,
                options={
                    "curriculum": "duel",
                    "items_enabled": False,
                    "agent_slot": episode % 2,
                    "swap_spawns": bool((episode // 2) % 2),
                },
            )
            terminated = truncated = False
            reward_sum = 0.0
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action, _state = model.predict(
                    obs,
                    action_masks=env.action_masks(),
                    deterministic=True,
                )
                obs, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
            outcome = str(info.get("outcome", "unknown"))
            outcomes[outcome] += 1
            outcomes_by_source[source][outcome] += 1
            outcome_values.append(_outcome_points(outcome))
            rewards.append(reward_sum)
            quality = info.get("quality", {})
            for key in QUALITY_KEYS:
                metric_totals[key] += float(quality.get(key, 0.0))
            successful_attacks.update(
                {str(key): int(value) for key, value in info.get("successful_attacks", {}).items()}
            )
    finally:
        env.close()

    count = max(1, len(rewards))
    quality = {key: float(metric_totals[key] / count) for key in QUALITY_KEYS}
    outcome_score = float(np.mean(outcome_values)) if outcome_values else 0.0
    score = _quality_score(outcome_score, quality)
    behavior_gate = bool(
        quality["action_change_rate"] <= 0.55
        and quality["projectiles_per_minute"] <= 30.0
        and quality["shield_activations_per_minute"] <= 25.0
        and quality["false_shield_rate"] <= 0.75
    )
    return {
        "score": score,
        "outcome_score": outcome_score,
        "behavior_gate_passed": behavior_gate,
        "outcomes": dict(outcomes),
        "outcomes_by_source": {key: dict(value) for key, value in outcomes_by_source.items()},
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "quality": quality,
        "successful_attacks": dict(successful_attacks),
    }


def _save_latest(model21: Any, model22: Any, output_dir: Path) -> None:
    model21.save(str(output_dir / "level21_model"))
    model22.save(str(output_dir / "level22_model"))


def _save_round(model21: Any, model22: Any, output_dir: Path, round_no: int) -> tuple[Path, Path]:
    _save_latest(model21, model22, output_dir)
    path21 = output_dir / f"round_{round_no:03d}_level21.zip"
    path22 = output_dir / f"round_{round_no:03d}_level22.zip"
    model21.save(str(path21.with_suffix("")))
    model22.save(str(path22.with_suffix("")))
    return path21, path22


def _load_recent_history(
    MaskablePPO: Any,
    output_dir: Path,
    *,
    limit: int,
) -> list[tuple[Any, str]]:
    if limit <= 0:
        return []
    paths = sorted(output_dir.glob("round_*_level*.zip"))[-limit:]
    return [
        (MaskablePPO.load(str(path), device="cpu"), f"history:{path.stem}")
        for path in paths
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练更有章法的v3战术21/22级桃子AI（动作掩码、课程、历史对手与质量门槛）"
    )
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--steps-per-round", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--eval-max-seconds", type=float, default=90.0)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--items-probability", type=float, default=0.20)
    parser.add_argument("--history-size", type=int, default=6)
    parser.add_argument("--name", default="peach_tactical_v3")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    parser.add_argument(
        "--resume-dir",
        type=Path,
        help="从目录中的v3 level21_model.zip/level22_model.zip继续；不会读取或覆盖v2模型",
    )
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
    except ImportError as exc:
        raise SystemExit(
            "训练依赖尚未安装。请运行: .venv-train/bin/python -m pip install -r requirements-training.txt"
        ) from exc

    if args.rounds < 1 or args.steps_per_round < 1 or args.eval_episodes < 2:
        raise SystemExit("--rounds、--steps-per-round 必须大于0，--eval-episodes 至少为2")
    teacher_path = args.teacher.expanduser().resolve()
    if not teacher_path.is_file():
        raise SystemExit(f"找不到冻结的v2 21级老师模型: {teacher_path}")

    set_random_seed(args.seed)
    output_dir = ROOT / "training" / "checkpoints" / args.name
    log_dir = ROOT / "training" / "logs" / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    teacher = PPO.load(str(teacher_path), device="cpu")

    raw21 = TacticalPeachEnv(
        seed=args.seed,
        max_episode_seconds=max(1.0, args.max_seconds),
        items_probability=args.items_probability,
    )
    raw22 = TacticalPeachEnv(
        seed=args.seed + 1_000_000,
        max_episode_seconds=max(1.0, args.max_seconds),
        items_probability=args.items_probability,
    )
    monitor_keys = (
        "outcome",
        "curriculum",
        "projectile_accuracy",
        "false_shield_rate",
        "action_change_rate",
    )
    env21 = Monitor(raw21, info_keywords=monitor_keys)
    env22 = Monitor(raw22, info_keywords=monitor_keys)

    if args.resume_dir is not None:
        resume_dir = args.resume_dir.expanduser().resolve()
        model21 = MaskablePPO.load(
            str(resume_dir / "level21_model.zip"),
            env=env21,
            device=args.device,
            tensorboard_log=str(log_dir),
        )
        model22 = MaskablePPO.load(
            str(resume_dir / "level22_model.zip"),
            env=env22,
            device=args.device,
            tensorboard_log=str(log_dir),
        )
        resume_mode = str(resume_dir)
    else:
        model21 = _new_model(
            MaskablePPO, env21, seed=args.seed + 21, device=args.device, log_dir=log_dir
        )
        model22 = _new_model(
            MaskablePPO, env22, seed=args.seed + 22, device=args.device, log_dir=log_dir
        )
        resume_mode = "new_v3_policy"

    history_models = _load_recent_history(
        MaskablePPO,
        args.resume_dir.expanduser().resolve() if args.resume_dir is not None else output_dir,
        limit=args.history_size,
    )
    best_scores = {21: float("-inf"), 22: float("-inf")}
    best_gates = {21: False, 22: False}
    history: list[dict[str, Any]] = []
    interrupted = False
    completed_rounds = 0
    try:
        for round_no in range(1, args.rounds + 1):
            strength = _curriculum_strength(round_no, args.rounds)
            raw21.set_curriculum_strength(strength)
            raw22.set_curriculum_strength(strength)
            raw21.set_tactical_opponent_pool(
                _weighted_pool(peer=model22, teacher=teacher, history=history_models)
            )
            raw22.set_tactical_opponent_pool(
                _weighted_pool(peer=model21, teacher=teacher, history=history_models)
            )

            print(
                f"\n===== v3第 {round_no}/{args.rounds} 轮：训练21级 "
                f"(课程概率 {strength:.0%}) ====="
            )
            model21.learn(
                total_timesteps=args.steps_per_round,
                reset_num_timesteps=False,
                tb_log_name=f"{args.name}_level21",
            )
            # PPO rollout期间对手保持冻结；21更新完成后，它才成为22的新对手。
            raw22.set_tactical_opponent_pool(
                _weighted_pool(peer=model21, teacher=teacher, history=history_models)
            )
            print(f"\n===== v3第 {round_no}/{args.rounds} 轮：训练22级 =====")
            model22.learn(
                total_timesteps=args.steps_per_round,
                reset_num_timesteps=False,
                tb_log_name=f"{args.name}_level22",
            )

            path21, path22 = _save_round(model21, model22, output_dir, round_no)
            evaluation21 = evaluate_model(
                model21,
                [(model22, "current_level22"), (teacher, "frozen_v2_level21")],
                episodes=args.eval_episodes,
                seed=args.seed + round_no * 20_000,
                max_seconds=args.eval_max_seconds,
            )
            evaluation22 = evaluate_model(
                model22,
                [(model21, "current_level21"), (teacher, "frozen_v2_level21")],
                episodes=args.eval_episodes,
                seed=args.seed + round_no * 20_000 + 10_000,
                max_seconds=args.eval_max_seconds,
            )
            for level, model, evaluation in (
                (21, model21, evaluation21),
                (22, model22, evaluation22),
            ):
                candidate_key = (
                    bool(evaluation["behavior_gate_passed"]),
                    float(evaluation["score"]),
                )
                best_key = (best_gates[level], best_scores[level])
                if candidate_key > best_key:
                    best_scores[level] = float(evaluation["score"])
                    best_gates[level] = bool(evaluation["behavior_gate_passed"])
                    model.save(str(output_dir / f"best_level{level}_model"))
                    evaluation["became_best"] = True
                else:
                    evaluation["became_best"] = False
            report = {
                "round": round_no,
                "curriculum_strength": strength,
                "level21": evaluation21,
                "level22": evaluation22,
            }
            history.append(report)
            completed_rounds = round_no
            print(json.dumps(report, ensure_ascii=False, indent=2))

            # Reload snapshots as inference-only CPU policies. This prevents
            # catastrophic forgetting without mutating an opponent mid-rollout.
            history_models.extend(
                [
                    (MaskablePPO.load(str(path21), device="cpu"), f"history:{path21.stem}"),
                    (MaskablePPO.load(str(path22), device="cpu"), f"history:{path22.stem}"),
                ]
            )
            history_models = (
                history_models[-args.history_size :] if args.history_size > 0 else []
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到中断，正在保存v3当前模型……")
    finally:
        _save_latest(model21, model22, output_dir)
        env21.close()
        env22.close()

    metadata = {
        "scenario": TACTICAL_OBSERVATION_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "interrupted": interrupted,
        "completed_rounds": completed_rounds,
        "rounds_requested": args.rounds,
        "steps_per_round_per_model": args.steps_per_round,
        "teacher": str(teacher_path),
        "resume": resume_mode,
        "seed": args.seed,
        "observation_size": TACTICAL_OBSERVATION_SIZE,
        "action_space": [4, 9],
        "physics_hz": 40,
        "policy_hz": TACTICAL_POLICY_HZ,
        "reaction_delay_ms": 100,
        "movement_commitment_ms": 200,
        "combat_rearm_ms": 300,
        "items_episode_probability": args.items_probability,
        "history_size": args.history_size,
        "best_scores": {str(key): value for key, value in best_scores.items()},
        "best_behavior_gates": {str(key): value for key, value in best_gates.items()},
        "normal_game_modified": False,
        "v2_models_overwritten": False,
        "evaluation_history": history,
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nv3当前21级: {output_dir / 'level21_model.zip'}")
    print(f"v3当前22级: {output_dir / 'level22_model.zip'}")
    if completed_rounds:
        print(f"质量最佳21级: {output_dir / 'best_level21_model.zip'}")
        print(f"质量最佳22级: {output_dir / 'best_level22_model.zip'}")
    print("试玩: .venv-train/bin/python -m training.play_tactical")


if __name__ == "__main__":
    main()
