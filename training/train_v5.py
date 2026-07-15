from __future__ import annotations

import argparse
import json
import shlex
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .v5_env import (
    V5_OBSERVATION_SIZE,
    V5_OBSERVATION_VERSION,
    V5_POLICY_HZ,
    V5PeachEnv,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEACHER = (
    ROOT
    / "training"
    / "checkpoints"
    / "peach_purpose_v5"
    / "teacher_level21_v2.zip"
)
SKILL_PHASES = (
    ("v5_navigation", "navigation_steps", 0.90),
    ("v5_air_chase", "air_steps", 0.35),
    ("v5_escape", "escape_steps", 0.65),
    ("v5_combo", "combo_steps", 0.30),
)
QUALITY_KEYS = (
    "projectile_accuracy",
    "projectiles_per_minute",
    "false_shield_rate",
    "shield_hold_fraction",
    "far_idle_fraction",
    "wall_stall_fraction",
    "plan_completion_rate",
    "purposeful_jump_rate",
    "jump_down_reversal_rate",
    "air_chase_opportunity_use_rate",
    "air_chase_hit_rate",
    "escape_success_rate",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_model(model: Any, path: Path) -> None:
    model.save(str(path.with_suffix("")))


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


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def evaluate_skill(
    model: Any,
    curriculum: str,
    *,
    episodes: int,
    seed: int,
    max_seconds: float,
) -> dict[str, Any]:
    env = V5PeachEnv(
        seed=seed,
        max_episode_seconds=max_seconds,
        items_probability=0.0,
        curriculum_strength=0.0,
        lesson_seconds=max_seconds,
    )
    successes = 0
    rewards: list[float] = []
    metrics: Counter[str] = Counter()
    try:
        for episode in range(max(1, episodes)):
            observation, _ = env.reset(
                seed=seed + episode,
                options={
                    "curriculum": curriculum,
                    "agent_slot": episode % 2,
                    "swap_spawns": bool((episode // 2) % 2),
                    "items_enabled": False,
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
            successes += int(bool(info.get("lesson_success")))
            rewards.append(reward_sum)
            for key, value in info.get("quality", {}).items():
                metrics[str(key)] += float(value)
    finally:
        env.close()
    count = max(1, len(rewards))
    return {
        "curriculum": curriculum,
        "episodes": count,
        "successes": successes,
        "success_rate": successes / count,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "quality": {key: value / count for key, value in metrics.items()},
    }


def _outcome_value(outcome: str) -> float:
    return 1.0 if outcome == "win" else -1.0


def evaluate_duel(
    model: Any,
    opponents: Sequence[tuple[Any, str]],
    *,
    episodes: int,
    seed: int,
    max_seconds: float,
) -> dict[str, Any]:
    env = V5PeachEnv(
        seed=seed,
        max_episode_seconds=max_seconds,
        items_probability=0.0,
        curriculum_strength=0.0,
    )
    env.opponent_deterministic = True
    outcomes: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    qualities: Counter[str] = Counter()
    shot_totals: Counter[str] = Counter()
    rewards: list[float] = []
    values: list[float] = []
    try:
        for episode in range(max(1, episodes)):
            opponent, source = opponents[episode % len(opponents)]
            env.set_tactical_opponent_pool([(opponent, 1.0, source)])
            observation, _ = env.reset(
                seed=seed + episode,
                options={
                    "curriculum": "duel",
                    "agent_slot": episode % 2,
                    "swap_spawns": bool((episode // 2) % 2),
                    "items_enabled": False,
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
            by_source[source][outcome] += 1
            values.append(_outcome_value(outcome))
            rewards.append(reward_sum)
            quality = info.get("quality", {})
            for key in QUALITY_KEYS:
                qualities[key] += float(quality.get(key, 0.0))
            shot_totals.update(info.get("shot_outcomes", {}))
    finally:
        env.close()
    count = max(1, sum(outcomes.values()))
    quality = {key: qualities[key] / count for key in QUALITY_KEYS}
    decisive = outcomes["win"] + outcomes["loss"]
    resolved = sum(
        shot_totals[key]
        for key in ("bullet_hit", "bullet_miss", "bullet_blocked", "rocket_hit", "rocket_miss", "rocket_blocked")
    )
    outcome_score = float(np.mean(values)) if values else -1.0
    score = (
        3.0 * outcome_score
        + 0.30 * min(0.6, quality["projectile_accuracy"])
        + 0.35 * min(0.8, quality["air_chase_hit_rate"])
        + 0.20 * min(1.0, quality["plan_completion_rate"])
        - 0.50 * quality["wall_stall_fraction"]
        - 0.30 * quality["jump_down_reversal_rate"]
    )
    report: dict[str, Any] = {
        "episodes": count,
        "score": float(score),
        "outcome_score": outcome_score,
        "win_rate": outcomes["win"] / count,
        "decisive_finish_rate": decisive / count,
        "timeout_rate": 1.0 - decisive / count,
        "outcomes": dict(outcomes),
        "outcomes_by_source": {key: dict(value) for key, value in by_source.items()},
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "quality": quality,
        "resolved_projectiles": int(resolved),
        "shot_outcomes": dict(shot_totals),
    }
    passed, failures = behavior_gate(report)
    report["behavior_gate_passed"] = passed
    report["gate_failures"] = failures
    return report


def behavior_gate(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    quality = report["quality"]
    episodes = max(1, int(report["episodes"]))
    failures: list[str] = []
    checks = (
        (float(report["decisive_finish_rate"]) >= 0.60, "决出胜负局不足60%"),
        (float(report["win_rate"]) >= 0.35, "胜率低于35%"),
        (float(quality["far_idle_fraction"]) <= 0.15, "远距离无目的停留超过15%"),
        (float(quality["wall_stall_fraction"]) <= 0.05, "墙体停滞超过5%"),
        (float(quality["false_shield_rate"]) <= 0.15, "假护盾率超过15%"),
        (float(quality["shield_hold_fraction"]) <= 0.08, "护盾占用超过8%"),
        (float(quality["plan_completion_rate"]) >= 0.30, "战术计划完成率低于30%"),
        (float(quality["purposeful_jump_rate"]) >= 0.90, "有目的跳跃低于90%"),
        (float(quality["jump_down_reversal_rate"]) <= 0.01, "无效跳跃后快落超过1%"),
        (
            float(quality["air_chase_opportunity_use_rate"]) >= 0.30,
            "空中追击机会利用率低于30%",
        ),
        (float(quality["air_chase_hit_rate"]) >= 0.20, "空中手刀追击命中率低于20%"),
        (int(report["resolved_projectiles"]) >= max(2, episodes // 4), "子弹样本不足"),
        (float(quality["projectiles_per_minute"]) <= 24.0, "每分钟投射物超过24发"),
        (float(quality["projectile_accuracy"]) >= 0.15, "投射物命中率低于15%"),
    )
    failures.extend(reason for passed, reason in checks if not passed)
    return not failures, failures


def _foundation_pool(teacher: Any) -> list[tuple[Any, float, str]]:
    return [
        ("active", 0.40, "active_probe"),
        ("melee", 0.30, "melee_probe"),
        (teacher, 0.20, "frozen_v2_level21"),
        ("retreat", 0.05, "retreat_probe"),
        ("idle", 0.05, "idle_probe"),
    ]


def _league_pool(
    *, peer: Any, teacher: Any, foundation: Any, history: Iterable[tuple[Any, str]]
) -> list[tuple[Any, float, str]]:
    frozen = list(history)
    history_weight = 0.10 / len(frozen) if frozen else 0.0
    foundation_weight = 0.15 + (0.10 if not frozen else 0.0)
    return [
        (peer, 0.35, "current_peer"),
        ("active", 0.15, "active_probe"),
        ("melee", 0.10, "melee_probe"),
        (teacher, 0.10, "frozen_v2_level21"),
        (foundation, foundation_weight, "purpose_foundation"),
        *((policy, history_weight, name) for policy, name in frozen),
        ("retreat", 0.03, "retreat_probe"),
        ("idle", 0.02, "idle_probe"),
    ]


def _load_history(MaskablePPO: Any, directory: Path, limit: int) -> list[tuple[Any, str]]:
    paths = sorted(directory.glob("champion_history_round_*_level*.zip"))[-max(0, limit) :]
    return [
        (MaskablePPO.load(str(path), device="cpu"), f"champion_history:{path.stem}")
        for path in paths
    ] if limit > 0 else []


def _promote(
    model: Any,
    *,
    level: int,
    round_no: int,
    evaluation: Mapping[str, Any],
    champions: dict[str, Any],
    directory: Path,
) -> bool:
    previous = champions.setdefault("levels", {}).get(str(level), {})
    if (
        not evaluation["behavior_gate_passed"]
        or float(evaluation["score"]) <= float(previous.get("score", float("-inf")))
    ):
        return False
    path = directory / f"champion_level{level}_model.zip"
    history = directory / f"champion_history_round_{round_no:03d}_level{level}.zip"
    _save_model(model, path)
    _save_model(model, history)
    champions["levels"][str(level)] = {
        "qualified": True,
        "path": path.name,
        "round": round_no,
        "score": float(evaluation["score"]),
        "evaluation": dict(evaluation),
    }
    champions["updated_utc"] = _now()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="训练v5目的驱动AI：技能课、共享基础、冻结21/22联赛与冠军门槛"
    )
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--navigation-steps", type=int, default=100_000)
    parser.add_argument("--air-steps", type=int, default=120_000)
    parser.add_argument("--escape-steps", type=int, default=80_000)
    parser.add_argument("--combo-steps", type=int, default=120_000)
    parser.add_argument("--mixed-steps", type=int, default=150_000)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--steps-per-round", type=int, default=75_000)
    parser.add_argument("--skill-eval-episodes", type=int, default=20)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--eval-max-seconds", type=float, default=90.0)
    parser.add_argument("--lesson-seconds", type=float, default=16.0)
    parser.add_argument("--history-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--name", default="peach_purpose_v5")
    parser.add_argument("--device", default="cpu", choices=("cpu", "mps", "auto"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-unqualified-foundation",
        action="store_true",
        help="仅供小步数冒烟：技能考试未过也继续分叉21/22",
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
    phase_steps = {
        "navigation_steps": args.navigation_steps,
        "air_steps": args.air_steps,
        "escape_steps": args.escape_steps,
        "combo_steps": args.combo_steps,
    }
    if any(value < 1 for value in (*phase_steps.values(), args.mixed_steps, args.rounds, args.steps_per_round)):
        raise SystemExit("所有训练步数和轮数必须大于0")
    if args.skill_eval_episodes < 4 or args.eval_episodes < 10:
        raise SystemExit("技能评估至少4局，对战评估至少10局")
    teacher_path = args.teacher.expanduser().resolve()
    if not teacher_path.is_file():
        raise SystemExit(f"找不到v2老师模型: {teacher_path}")

    directory = ROOT / "training" / "checkpoints" / args.name
    log_dir = ROOT / "training" / "logs" / args.name
    foundation_path = directory / "foundation_model.zip"
    candidate_paths = {
        21: directory / "candidate_level21_model.zip",
        22: directory / "candidate_level22_model.zip",
    }
    if not args.resume and any(path.is_file() for path in (foundation_path, *candidate_paths.values())):
        raise SystemExit(f"目录已有v5模型: {directory}\n继续请加 --resume，重练请换 --name。")
    if args.resume and not foundation_path.is_file():
        raise SystemExit(f"无法继续：缺少 {foundation_path}")
    directory.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(args.seed)
    teacher = PPO.load(str(teacher_path), device="cpu")

    raw_foundation = V5PeachEnv(
        seed=args.seed,
        max_episode_seconds=args.max_seconds,
        items_probability=0.0,
        lesson_seconds=args.lesson_seconds,
    )
    raw_foundation.set_tactical_opponent_pool(_foundation_pool(teacher))
    foundation_env = Monitor(
        raw_foundation, info_keywords=("outcome", "curriculum", "lesson_success")
    )
    if args.resume:
        foundation = MaskablePPO.load(
            str(foundation_path),
            env=foundation_env,
            device=args.device,
            tensorboard_log=str(log_dir),
        )
    else:
        foundation = _new_model(
            MaskablePPO,
            foundation_env,
            seed=args.seed + 5,
            device=args.device,
            log_dir=log_dir,
        )

    skill_exams_path = directory / "skill_exams.json"
    training_state_path = directory / "training_state.json"
    skill_reports: dict[str, Any] = (
        _load_json(skill_exams_path, {}) if args.resume else {}
    )
    training_state: dict[str, Any] = (
        _load_json(training_state_path, {}) if args.resume else {}
    )

    def begin_or_resume_phase(name: str, requested_steps: int) -> int:
        active = training_state.get("active_phase", {})
        if isinstance(active, Mapping) and active.get("name") == name:
            start = int(active.get("start_timesteps", foundation.num_timesteps))
            target = int(active.get("target_steps", requested_steps))
            completed = max(0, int(foundation.num_timesteps) - start)
            remaining = max(0, target - completed)
            print(
                f"续训 {name}: 已完成约 {completed:,}/{target:,} 步，"
                f"剩余 {remaining:,} 步"
            )
            return remaining
        training_state["active_phase"] = {
            "name": name,
            "start_timesteps": int(foundation.num_timesteps),
            "target_steps": int(requested_steps),
        }
        training_state["foundation_timesteps"] = int(foundation.num_timesteps)
        _write_json(training_state_path, training_state)
        return requested_steps

    def finish_phase(name: str) -> None:
        training_state["active_phase"] = None
        training_state["last_completed_phase"] = name
        training_state["foundation_timesteps"] = int(foundation.num_timesteps)
        _write_json(training_state_path, training_state)

    try:
        for phase_index, (curriculum, argument_name, required_rate) in enumerate(SKILL_PHASES):
            existing = skill_reports.get(curriculum, {})
            if args.resume and bool(existing.get("passed")):
                print(
                    f"\n===== v5技能课 {curriculum}: 已通过 "
                    f"({float(existing.get('success_rate', 0.0)):.1%})，跳过 ====="
                )
                continue
            raw_foundation.set_fixed_curriculum(curriculum)
            steps = phase_steps[argument_name]
            remaining = begin_or_resume_phase(curriculum, steps)
            print(f"\n===== v5技能课 {curriculum}: {remaining:,} 剩余步数 =====")
            if remaining > 0:
                foundation.learn(
                    total_timesteps=remaining,
                    reset_num_timesteps=False,
                    tb_log_name=f"{args.name}_{curriculum}",
                )
            _save_model(foundation, foundation_path)
            report = evaluate_skill(
                foundation,
                curriculum,
                episodes=args.skill_eval_episodes,
                seed=args.seed + 10_000 + phase_index * 1_000,
                max_seconds=args.lesson_seconds,
            )
            report["required_success_rate"] = required_rate
            report["passed"] = report["success_rate"] >= required_rate
            skill_reports[curriculum] = report
            _write_json(skill_exams_path, skill_reports)
            finish_phase(curriculum)
            print(json.dumps(report, ensure_ascii=False, indent=2))

        raw_foundation.set_fixed_curriculum(None)
        raw_foundation.set_curriculum_strength(0.70)
        if args.resume and bool(training_state.get("mixed_foundation_complete")):
            print("\n===== v5混合基础: 已完成，跳过 =====")
        else:
            remaining = begin_or_resume_phase("mixed_foundation", args.mixed_steps)
            print(f"\n===== v5混合基础: {remaining:,} 剩余步数 =====")
            if remaining > 0:
                foundation.learn(
                    total_timesteps=remaining,
                    reset_num_timesteps=False,
                    tb_log_name=f"{args.name}_mixed_foundation",
                )
            _save_model(foundation, foundation_path)
            training_state["mixed_foundation_complete"] = True
            finish_phase("mixed_foundation")
    except KeyboardInterrupt:
        _save_model(foundation, foundation_path)
        foundation_env.close()
        training_state["foundation_timesteps"] = int(foundation.num_timesteps)
        _write_json(skill_exams_path, skill_reports)
        _write_json(training_state_path, training_state)
        print(f"\n已保存v5共享基础: {foundation_path}\n继续时加 --resume。")
        return
    foundation_env.close()
    training_state["skills_complete"] = all(
        bool(skill_reports.get(name, {}).get("passed")) for name, _argument, _rate in SKILL_PHASES
    )
    training_state["foundation_timesteps"] = int(foundation.num_timesteps)
    _write_json(skill_exams_path, skill_reports)
    _write_json(training_state_path, training_state)
    failed_skills = [name for name, report in skill_reports.items() if not report["passed"]]
    if failed_skills and not args.allow_unqualified_foundation:
        print("\nv5共享基础未通过以下技能考试，未开始21/22自对战:")
        for name in failed_skills:
            report = skill_reports[name]
            print(f"- {name}: {report['success_rate']:.1%} < {report['required_success_rate']:.1%}")
        print("加 --resume 继续训技能课。冒烟检查才使用 --allow-unqualified-foundation。")
        return

    foundation_policy = MaskablePPO.load(str(foundation_path), device="cpu")
    raw21 = V5PeachEnv(seed=args.seed + 21, max_episode_seconds=args.max_seconds)
    raw22 = V5PeachEnv(seed=args.seed + 22, max_episode_seconds=args.max_seconds)
    env21 = Monitor(raw21, info_keywords=("outcome", "curriculum", "lesson_success"))
    env22 = Monitor(raw22, info_keywords=("outcome", "curriculum", "lesson_success"))
    source21 = candidate_paths[21] if args.resume and candidate_paths[21].is_file() else foundation_path
    source22 = candidate_paths[22] if args.resume and candidate_paths[22].is_file() else foundation_path
    model21 = MaskablePPO.load(str(source21), env=env21, device=args.device, tensorboard_log=str(log_dir))
    model22 = MaskablePPO.load(str(source22), env=env22, device=args.device, tensorboard_log=str(log_dir))
    champions_path = directory / "champions.json"
    champions = _load_json(
        champions_path,
        {"scenario": V5_OBSERVATION_VERSION, "created_utc": _now(), "levels": {}},
    )
    history_models = _load_history(MaskablePPO, directory, args.history_size)
    evaluation_history: list[dict[str, Any]] = []
    interrupted = False
    completed_rounds = 0
    try:
        for round_no in range(1, args.rounds + 1):
            strength = 0.45 if args.rounds == 1 else 0.45 - 0.25 * (round_no - 1) / (args.rounds - 1)
            raw21.set_curriculum_strength(strength)
            raw22.set_curriculum_strength(strength)
            raw21.set_tactical_opponent_pool(
                _league_pool(peer=model22, teacher=teacher, foundation=foundation_policy, history=history_models)
            )
            print(f"\n===== v5第 {round_no}/{args.rounds} 轮：更新21级 =====")
            model21.learn(
                total_timesteps=args.steps_per_round,
                reset_num_timesteps=False,
                tb_log_name=f"{args.name}_level21",
            )
            raw22.set_tactical_opponent_pool(
                _league_pool(peer=model21, teacher=teacher, foundation=foundation_policy, history=history_models)
            )
            print(f"\n===== v5第 {round_no}/{args.rounds} 轮：更新22级 =====")
            model22.learn(
                total_timesteps=args.steps_per_round,
                reset_num_timesteps=False,
                tb_log_name=f"{args.name}_level22",
            )
            for level, model in ((21, model21), (22, model22)):
                _save_model(model, candidate_paths[level])
                _save_model(model, directory / f"round_{round_no:03d}_level{level}.zip")
            suite21 = [
                (model22, "current_level22"),
                (foundation_policy, "purpose_foundation"),
                ("active", "active_probe"),
                ("melee", "melee_probe"),
                (teacher, "frozen_v2_level21"),
            ]
            suite22 = [
                (model21, "current_level21"),
                (foundation_policy, "purpose_foundation"),
                ("active", "active_probe"),
                ("melee", "melee_probe"),
                (teacher, "frozen_v2_level21"),
            ]
            evaluation21 = evaluate_duel(
                model21, suite21, episodes=args.eval_episodes,
                seed=args.seed + round_no * 20_000, max_seconds=args.eval_max_seconds,
            )
            evaluation22 = evaluate_duel(
                model22, suite22, episodes=args.eval_episodes,
                seed=args.seed + round_no * 20_000 + 10_000, max_seconds=args.eval_max_seconds,
            )
            for level, model, evaluation in (
                (21, model21, evaluation21), (22, model22, evaluation22)
            ):
                evaluation["promoted"] = _promote(
                    model, level=level, round_no=round_no,
                    evaluation=evaluation, champions=champions, directory=directory,
                )
            _write_json(champions_path, champions)
            report = {"round": round_no, "level21": evaluation21, "level22": evaluation22}
            evaluation_history.append(report)
            completed_rounds = round_no
            print(json.dumps(report, ensure_ascii=False, indent=2))
            history_models = _load_history(MaskablePPO, directory, args.history_size)
            for level in (21, 22):
                entry = champions.get("levels", {}).get(str(level), {})
                champion_path = directory / str(entry.get("path", ""))
                if entry.get("qualified") and champion_path.is_file() and not report[f"level{level}"]["promoted"]:
                    replacement = MaskablePPO.load(
                        str(champion_path), env=env21 if level == 21 else env22,
                        device=args.device, tensorboard_log=str(log_dir),
                    )
                    if level == 21:
                        model21 = replacement
                    else:
                        model22 = replacement
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到中断，正在保存v5候选模型……")
    finally:
        _save_model(model21, candidate_paths[21])
        _save_model(model22, candidate_paths[22])
        env21.close()
        env22.close()

    metadata = {
        "scenario": V5_OBSERVATION_VERSION,
        "created_utc": _now(),
        "interrupted": interrupted,
        "completed_rounds_this_run": completed_rounds,
        "observation_size": V5_OBSERVATION_SIZE,
        "action_space": {"discrete_purposes": 14},
        "physics_hz": 40,
        "policy_hz": V5_POLICY_HZ,
        "teacher": str(teacher_path),
        "skill_exams": skill_reports,
        "evaluation_history": evaluation_history,
        "normal_game_training_overrides": False,
        "v2_v3_v4_models_overwritten": False,
    }
    _write_json(directory / "training_config.json", metadata)
    print(f"\nv5候选21级: {candidate_paths[21]}\nv5候选22级: {candidate_paths[22]}")
    levels = champions.get("levels", {})
    if all(levels.get(str(level), {}).get("qualified") for level in (21, 22)):
        print("两个级别都有合格冠军。试玩: .venv-train/bin/python -m training.play_v5")
    else:
        quoted = shlex.quote(str(directory))
        print(
            "人工验收候选: .venv-train/bin/python -m training.play_v5 "
            f"--directory {quoted} --allow-candidate"
        )


if __name__ == "__main__":
    main()
