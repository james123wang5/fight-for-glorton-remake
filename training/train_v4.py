from __future__ import annotations

import argparse
import json
import shlex
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .v4_env import (
    V4_OBSERVATION_SIZE,
    V4_OBSERVATION_VERSION,
    V4_POLICY_HZ,
    V4PeachEnv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = ROOT / "training" / "checkpoints" / "peach_league_v2" / "level21_model.zip"
MELEE_LABELS = ("punchGround", "punchRun", "punchUp", "punchAir", "specialBackThrow")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_model(MaskablePPO: Any, env: Any, *, seed: int, device: str, log_dir: Path) -> Any:
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
        ent_coef=0.0015,
        policy_kwargs={"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        verbose=1,
        seed=seed,
        device=device,
        tensorboard_log=str(log_dir),
    )


def _save_model(model: Any, path: Path) -> None:
    model.save(str(path.with_suffix("")))


def _curriculum_strength(round_no: int, rounds: int) -> float:
    if rounds <= 1:
        return 0.35
    progress = (round_no - 1) / (rounds - 1)
    return 0.60 + (0.20 - 0.60) * progress


def _foundation_pool(teacher: Any) -> list[tuple[Any, float, str]]:
    return [
        ("active", 0.40, "active_probe"),
        ("melee", 0.25, "melee_probe"),
        (teacher, 0.20, "frozen_v2_level21"),
        ("retreat", 0.10, "retreat_probe"),
        ("idle", 0.05, "idle_probe"),
    ]


def _league_pool(
    *,
    peer: Any,
    teacher: Any,
    foundation: Any,
    history: Iterable[tuple[Any, str]],
) -> list[tuple[Any, float, str]]:
    frozen = list(history)
    foundation_weight = 0.10 if frozen else 0.20
    history_weight = 0.10 / len(frozen) if frozen else 0.0
    return [
        (peer, 0.35, "current_peer"),
        ("active", 0.20, "active_probe"),
        ("melee", 0.10, "melee_probe"),
        (teacher, 0.10, "frozen_v2_level21"),
        (foundation, foundation_weight, "shared_foundation"),
        *((policy, history_weight, name) for policy, name in frozen),
        ("retreat", 0.03, "retreat_probe"),
        ("idle", 0.02, "idle_probe"),
    ]


def _outcome_points(outcome: str) -> float:
    # A timeout is a failed evaluation even if the damage tiebreak calls it a
    # timeout_win. This prevents two defensive policies from promoting each other.
    if outcome == "win":
        return 1.0
    return -1.0


def behavior_gate(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Return the strict deployment gate and human-readable failure reasons."""

    quality = report["quality"]
    episodes = max(1, int(report.get("episodes", 1)))
    minimum_events = max(4, episodes // 3)
    failures: list[str] = []

    checks = (
        (float(report["decisive_finish_rate"]) >= 0.60, "决出胜负局不足60%"),
        (float(report["win_rate"]) >= 0.35, "胜率低于35%"),
        (float(quality["far_idle_fraction"]) <= 0.18, "远距离发呆超过18%"),
        (float(quality["wall_stall_fraction"]) <= 0.15, "撞墙停滞超过15%"),
        (float(quality["ground_crouches_per_minute"]) <= 1.0, "固定平台蹲守过多"),
        (float(quality["shield_hold_fraction"]) <= 0.08, "护盾持续时间超过8%"),
        (float(quality["false_shield_rate"]) <= 0.15, "无威胁开盾率超过15%"),
        (
            float(quality["shield_activations_per_minute"]) <= 8.0,
            "每分钟开盾次数超过8次",
        ),
        (int(report["resolved_projectiles"]) >= minimum_events, "可评估的子弹样本不足"),
        (float(quality["projectiles_per_minute"]) >= 0.5, "几乎不使用手枪"),
        (float(quality["projectiles_per_minute"]) <= 24.0, "手枪/火箭发射过频"),
        (float(quality["projectile_accuracy"]) >= 0.18, "投射物命中率低于18%"),
        (int(report["melee_opportunities"]) >= minimum_events, "可评估的近战机会不足"),
        (
            float(quality["melee_opportunity_use_rate"]) >= 0.25,
            "近战机会利用率低于25%",
        ),
    )
    failures.extend(reason for passed, reason in checks if not passed)
    return not failures, failures


def _quality_score(outcome_score: float, quality: Mapping[str, float]) -> float:
    # Match result dominates; behavior terms only rank candidates with similar results.
    score = 3.0 * outcome_score
    score += 0.30 * min(0.60, float(quality["projectile_accuracy"]))
    score += 0.25 * min(0.75, float(quality["melee_opportunity_use_rate"]))
    score -= 0.35 * float(quality["far_idle_fraction"])
    score -= 0.30 * float(quality["wall_stall_fraction"])
    score -= 0.20 * float(quality["shield_hold_fraction"])
    return float(score)


def evaluate_model(
    model: Any,
    opponents: Sequence[tuple[Any, str]],
    *,
    episodes: int,
    seed: int,
    max_seconds: float,
) -> dict[str, Any]:
    """Evaluate fixed opponents, both player slots and swapped spawn layouts."""

    env = V4PeachEnv(
        seed=seed,
        max_episode_seconds=max_seconds,
        items_probability=0.0,
        curriculum_strength=0.0,
    )
    env.opponent_deterministic = True
    outcomes: Counter[str] = Counter()
    outcomes_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    shot_totals: Counter[str] = Counter()
    shield_totals: Counter[str] = Counter()
    behavior_totals: Counter[str] = Counter()
    attack_totals: Counter[str] = Counter()
    successful_totals: Counter[str] = Counter()
    rewards: list[float] = []
    outcome_values: list[float] = []
    decision_steps = 0
    try:
        for episode in range(max(1, int(episodes))):
            opponent, source = opponents[episode % len(opponents)]
            env.set_tactical_opponent_pool([(opponent, 1.0, source)])
            observation, _ = env.reset(
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
                    observation,
                    action_masks=env.action_masks(),
                    deterministic=True,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
            outcome = str(info.get("outcome", "unknown"))
            outcomes[outcome] += 1
            outcomes_by_source[source][outcome] += 1
            outcome_values.append(_outcome_points(outcome))
            rewards.append(reward_sum)
            decision_steps += int(info.get("decision_steps", 0))
            shot_totals.update(info.get("shot_outcomes", {}))
            shield_totals.update(info.get("shield_metrics", {}))
            behavior_totals.update(info.get("behavior_counts", {}))
            attack_totals.update(info.get("attack_starts", {}))
            successful_totals.update(info.get("successful_attacks", {}))
    finally:
        env.close()

    count = max(1, sum(outcomes.values()))
    minutes = max(1.0 / 600.0, decision_steps / V4_POLICY_HZ / 60.0)
    fired = shot_totals["bullet_fired"] + shot_totals["rocket_fired"]
    hits = shot_totals["bullet_hit"] + shot_totals["rocket_hit"]
    blocked = shot_totals["bullet_blocked"] + shot_totals["rocket_blocked"]
    misses = shot_totals["bullet_miss"] + shot_totals["rocket_miss"]
    resolved = hits + blocked + misses
    activations = shield_totals["activations"]
    melee_starts = sum(attack_totals[label] for label in MELEE_LABELS)
    melee_hits = sum(successful_totals[label] for label in MELEE_LABELS)
    melee_opportunities = behavior_totals["melee_opportunities"]
    quality = {
        "action_change_rate": behavior_totals["accepted_action_changes"]
        / max(1, decision_steps - count),
        "projectiles_per_minute": fired / minutes,
        "projectile_accuracy": hits / max(1, resolved),
        "shield_activations_per_minute": activations / minutes,
        "false_shield_rate": shield_totals["false_activations"] / max(1, activations),
        "shield_block_precision": min(1.0, shield_totals["blocks"] / max(1, activations)),
        "melee_hit_rate": melee_hits / max(1, melee_starts),
        "far_idle_fraction": behavior_totals["far_idle_decisions"] / max(1, decision_steps),
        "wall_stall_fraction": behavior_totals["wall_stall_decisions"] / max(1, decision_steps),
        "ground_crouches_per_minute": behavior_totals["ground_crouches"] / minutes,
        "shield_hold_fraction": behavior_totals["shield_hold_decisions"]
        / max(1, decision_steps),
        "melee_opportunity_use_rate": behavior_totals["melee_opportunity_uses"]
        / max(1, melee_opportunities),
    }
    decisive = outcomes["win"] + outcomes["loss"]
    outcome_score = float(np.mean(outcome_values)) if outcome_values else -1.0
    report: dict[str, Any] = {
        "episodes": count,
        "score": _quality_score(outcome_score, quality),
        "outcome_score": outcome_score,
        "win_rate": outcomes["win"] / count,
        "decisive_finish_rate": decisive / count,
        "timeout_rate": 1.0 - decisive / count,
        "outcomes": dict(outcomes),
        "outcomes_by_source": {key: dict(value) for key, value in outcomes_by_source.items()},
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "quality": quality,
        "resolved_projectiles": int(resolved),
        "melee_opportunities": int(melee_opportunities),
        "shot_outcomes": dict(shot_totals),
        "shield_metrics": dict(shield_totals),
        "successful_attacks": dict(successful_totals),
    }
    passed, failures = behavior_gate(report)
    report["behavior_gate_passed"] = passed
    report["gate_failures"] = failures
    return report


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_history(MaskablePPO: Any, output_dir: Path, limit: int) -> list[tuple[Any, str]]:
    if limit <= 0:
        return []
    paths = sorted(output_dir.glob("champion_history_round_*_level*.zip"))[-limit:]
    return [
        (MaskablePPO.load(str(path), device="cpu"), f"champion_history:{path.stem}")
        for path in paths
    ]


def _promote(
    model: Any,
    *,
    level: int,
    round_no: int,
    evaluation: Mapping[str, Any],
    champions: dict[str, Any],
    output_dir: Path,
) -> bool:
    previous = champions.setdefault("levels", {}).get(str(level), {})
    previous_score = float(previous.get("score", float("-inf")))
    if not bool(evaluation["behavior_gate_passed"]) or float(evaluation["score"]) <= previous_score:
        return False
    champion_path = output_dir / f"champion_level{level}_model.zip"
    history_path = output_dir / f"champion_history_round_{round_no:03d}_level{level}.zip"
    _save_model(model, champion_path)
    _save_model(model, history_path)
    champions["levels"][str(level)] = {
        "qualified": True,
        "path": champion_path.name,
        "round": round_no,
        "score": float(evaluation["score"]),
        "evaluation": dict(evaluation),
    }
    champions["updated_utc"] = _utc_now()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练v4主动战斗21/22级桃子AI：共享基础、冻结对手和严格冠军门槛"
    )
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--foundation-steps", type=int, default=300_000)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--steps-per-round", type=int, default=75_000)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--eval-max-seconds", type=float, default=90.0)
    parser.add_argument("--history-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--name", default="peach_active_v4")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从同名目录的candidate继续；不会覆盖v2/v3目录",
    )
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
    except ImportError as exc:
        raise SystemExit(
            "训练依赖尚未安装。请运行: "
            ".venv-train/bin/python -m pip install -r requirements-training.txt"
        ) from exc

    if (
        args.foundation_steps < 1
        or args.rounds < 1
        or args.steps_per_round < 1
        or args.eval_episodes < 10
    ):
        raise SystemExit("训练步数和轮数必须大于0，--eval-episodes至少为10")
    teacher_path = args.teacher.expanduser().resolve()
    if not teacher_path.is_file():
        raise SystemExit(f"找不到冻结的v2 21级老师模型: {teacher_path}")

    output_dir = ROOT / "training" / "checkpoints" / args.name
    log_dir = ROOT / "training" / "logs" / args.name
    foundation_path = output_dir / "foundation_model.zip"
    candidate_paths = {
        21: output_dir / "candidate_level21_model.zip",
        22: output_dir / "candidate_level22_model.zip",
    }
    if not args.resume and any(path.is_file() for path in (foundation_path, *candidate_paths.values())):
        raise SystemExit(
            f"目录已有v4模型: {output_dir}\n"
            "要继续请加 --resume，要重练请换一个 --name。"
        )
    if args.resume and not foundation_path.is_file():
        raise SystemExit(f"无法继续：缺少共享基础模型 {foundation_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    teacher = PPO.load(str(teacher_path), device="cpu")

    if not args.resume:
        raw_foundation = V4PeachEnv(
            seed=args.seed,
            max_episode_seconds=max(1.0, args.max_seconds),
            items_probability=0.0,
            curriculum_strength=0.75,
        )
        raw_foundation.set_tactical_opponent_pool(_foundation_pool(teacher))
        foundation_env = Monitor(raw_foundation, info_keywords=("outcome", "curriculum"))
        foundation_model = _new_model(
            MaskablePPO,
            foundation_env,
            seed=args.seed + 4,
            device=args.device,
            log_dir=log_dir,
        )
        try:
            print(f"\n===== v4共享基础：{args.foundation_steps:,} 步 =====")
            foundation_model.learn(
                total_timesteps=args.foundation_steps,
                reset_num_timesteps=False,
                tb_log_name=f"{args.name}_foundation",
            )
        except KeyboardInterrupt:
            _save_model(foundation_model, foundation_path)
            foundation_env.close()
            print(f"\n已保存中断的共享基础: {foundation_path}")
            print("继续命令加 --resume。")
            return
        _save_model(foundation_model, foundation_path)
        foundation_env.close()
        del foundation_model

    foundation_policy = MaskablePPO.load(str(foundation_path), device="cpu")
    raw21 = V4PeachEnv(
        seed=args.seed + 21,
        max_episode_seconds=max(1.0, args.max_seconds),
        items_probability=0.0,
    )
    raw22 = V4PeachEnv(
        seed=args.seed + 22,
        max_episode_seconds=max(1.0, args.max_seconds),
        items_probability=0.0,
    )
    env21 = Monitor(raw21, info_keywords=("outcome", "curriculum"))
    env22 = Monitor(raw22, info_keywords=("outcome", "curriculum"))

    source21 = candidate_paths[21] if args.resume and candidate_paths[21].is_file() else foundation_path
    source22 = candidate_paths[22] if args.resume and candidate_paths[22].is_file() else foundation_path
    model21 = MaskablePPO.load(
        str(source21), env=env21, device=args.device, tensorboard_log=str(log_dir)
    )
    model22 = MaskablePPO.load(
        str(source22), env=env22, device=args.device, tensorboard_log=str(log_dir)
    )

    champions_path = output_dir / "champions.json"
    champions = _load_json(
        champions_path,
        {
            "scenario": V4_OBSERVATION_VERSION,
            "created_utc": _utc_now(),
            "levels": {},
        },
    )
    evaluation_history = _load_json(output_dir / "training_config.json", {}).get(
        "evaluation_history", []
    )
    history_models = _load_history(MaskablePPO, output_dir, args.history_size)
    completed_rounds = 0
    interrupted = False
    try:
        for round_no in range(1, args.rounds + 1):
            strength = _curriculum_strength(round_no, args.rounds)
            raw21.set_curriculum_strength(strength)
            raw22.set_curriculum_strength(strength)
            raw21.set_tactical_opponent_pool(
                _league_pool(
                    peer=model22,
                    teacher=teacher,
                    foundation=foundation_policy,
                    history=history_models,
                )
            )
            print(
                f"\n===== v4第 {round_no}/{args.rounds} 轮：更新21级 "
                f"(专项课 {strength:.0%}) ====="
            )
            model21.learn(
                total_timesteps=args.steps_per_round,
                reset_num_timesteps=False,
                tb_log_name=f"{args.name}_level21",
            )

            # 21 is fixed while 22 collects its rollout; neither policy changes
            # underneath the other's PPO batch.
            raw22.set_tactical_opponent_pool(
                _league_pool(
                    peer=model21,
                    teacher=teacher,
                    foundation=foundation_policy,
                    history=history_models,
                )
            )
            print(f"\n===== v4第 {round_no}/{args.rounds} 轮：更新22级 =====")
            model22.learn(
                total_timesteps=args.steps_per_round,
                reset_num_timesteps=False,
                tb_log_name=f"{args.name}_level22",
            )

            _save_model(model21, candidate_paths[21])
            _save_model(model22, candidate_paths[22])
            round_paths = {
                21: output_dir / f"round_{round_no:03d}_level21.zip",
                22: output_dir / f"round_{round_no:03d}_level22.zip",
            }
            _save_model(model21, round_paths[21])
            _save_model(model22, round_paths[22])

            suite21 = [
                (model22, "current_level22"),
                (foundation_policy, "shared_foundation"),
                ("active", "active_probe"),
                ("melee", "melee_probe"),
                (teacher, "frozen_v2_level21"),
            ]
            suite22 = [
                (model21, "current_level21"),
                (foundation_policy, "shared_foundation"),
                ("active", "active_probe"),
                ("melee", "melee_probe"),
                (teacher, "frozen_v2_level21"),
            ]
            evaluation21 = evaluate_model(
                model21,
                suite21,
                episodes=args.eval_episodes,
                seed=args.seed + round_no * 20_000,
                max_seconds=args.eval_max_seconds,
            )
            evaluation22 = evaluate_model(
                model22,
                suite22,
                episodes=args.eval_episodes,
                seed=args.seed + round_no * 20_000 + 10_000,
                max_seconds=args.eval_max_seconds,
            )
            for level, model, evaluation in (
                (21, model21, evaluation21),
                (22, model22, evaluation22),
            ):
                evaluation["promoted"] = _promote(
                    model,
                    level=level,
                    round_no=round_no,
                    evaluation=evaluation,
                    champions=champions,
                    output_dir=output_dir,
                )
            _write_json(champions_path, champions)
            report = {
                "round": round_no,
                "curriculum_strength": strength,
                "level21": evaluation21,
                "level22": evaluation22,
            }
            evaluation_history.append(report)
            completed_rounds = round_no
            print(json.dumps(report, ensure_ascii=False, indent=2))

            history_models = _load_history(MaskablePPO, output_dir, args.history_size)
            for level in (21, 22):
                entry = champions.get("levels", {}).get(str(level), {})
                champion_path = output_dir / str(entry.get("path", ""))
                if entry.get("qualified") and champion_path.is_file():
                    # A failed or weaker candidate cannot become the next
                    # generation's base once a qualified champion exists.
                    if not report[f"level{level}"]["promoted"]:
                        replacement = MaskablePPO.load(
                            str(champion_path),
                            env=env21 if level == 21 else env22,
                            device=args.device,
                            tensorboard_log=str(log_dir),
                        )
                        if level == 21:
                            model21 = replacement
                        else:
                            model22 = replacement
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到中断，正在保存v4候选模型……")
    finally:
        _save_model(model21, candidate_paths[21])
        _save_model(model22, candidate_paths[22])
        env21.close()
        env22.close()

    metadata = {
        "scenario": V4_OBSERVATION_VERSION,
        "created_utc": _utc_now(),
        "interrupted": interrupted,
        "completed_rounds_this_run": completed_rounds,
        "rounds_requested": args.rounds,
        "foundation_steps": args.foundation_steps,
        "steps_per_round_per_model": args.steps_per_round,
        "teacher": str(teacher_path),
        "seed": args.seed,
        "observation_size": V4_OBSERVATION_SIZE,
        "action_space": [4, 9],
        "physics_hz": 40,
        "policy_hz": V4_POLICY_HZ,
        "items_enabled": False,
        "history_size": args.history_size,
        "normal_game_modified": False,
        "v2_v3_models_overwritten": False,
        "promotion_requires_behavior_gate": True,
        "evaluation_history": evaluation_history,
    }
    _write_json(output_dir / "training_config.json", metadata)
    print(f"\nv4候选21级: {candidate_paths[21]}")
    print(f"v4候选22级: {candidate_paths[22]}")
    qualified = champions.get("levels", {})
    if all(qualified.get(str(level), {}).get("qualified") for level in (21, 22)):
        print("两个级别都已有通过门槛的冠军。试玩: .venv-train/bin/python -m training.play_v4")
    else:
        print(
            "尚未两者都通过冠军门槛。可先验收候选: "
            ".venv-train/bin/python -m training.play_v4 "
            f"--directory {shlex.quote(str(output_dir))} --allow-candidate"
        )


if __name__ == "__main__":
    main()
