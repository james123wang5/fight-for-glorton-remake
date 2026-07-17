from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .preflight_roster import TRAINABLE_ROSTER, preflight_fighter
from .roster_env import RosterPurposeEnv
from .roster_jobs import TrainingScenario, claim_training_job
from .roster_observation import ROSTER_OBSERVATION_SIZE, ROSTER_OBSERVATION_VERSION
from .roster_transfer import initialize_roster_from_v5


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "training"
    / "checkpoints"
    / "peach_purpose_v5"
    / "champion_level22_model.zip"
)
QUALITY_KEYS = (
    "far_idle_fraction",
    "wall_stall_fraction",
    "false_shield_rate",
    "shield_hold_fraction",
    "melee_opportunity_use_rate",
    "plan_completion_rate",
    "purposeful_jump_rate",
    "jump_down_reversal_rate",
    "air_chase_opportunity_use_rate",
    "air_chase_hit_rate",
    "role_special_accuracy",
    "role_specials_per_minute",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _save_model(model: Any, path: Path) -> None:
    model.save(str(path.with_suffix("")))


def _set_training_pool(env: RosterPurposeEnv, entries: Sequence[tuple[Any, float, str]]) -> None:
    env.set_tactical_opponent_pool(entries)


def roster_behavior_gate(report: Mapping[str, Any]) -> tuple[bool, list[str]]:
    quality = report["quality"]
    failures: list[str] = []
    checks = (
        (float(report["decisive_finish_rate"]) >= 0.50, "决出胜负局不足50%"),
        (float(quality["far_idle_fraction"]) <= 0.18, "远距离无目的停留超过18%"),
        (float(quality["wall_stall_fraction"]) <= 0.08, "墙体停滞超过8%"),
        (float(quality["false_shield_rate"]) <= 0.20, "假护盾率超过20%"),
        (float(quality["shield_hold_fraction"]) <= 0.10, "护盾占用超过10%"),
        (float(quality["melee_opportunity_use_rate"]) >= 0.20, "近战机会利用率低于20%"),
        (float(quality["purposeful_jump_rate"]) >= 0.85, "有目的跳跃低于85%"),
        (float(quality["jump_down_reversal_rate"]) <= 0.02, "无效跳跃快落超过2%"),
        (float(quality["role_specials_per_minute"]) <= 24.0, "特殊技每分钟超过24次"),
    )
    failures.extend(reason for passed, reason in checks if not passed)
    return not failures, failures


def evaluate_roster_model(
    model: Any,
    *,
    fighter_name: str,
    stage_name: str,
    opponents: Sequence[tuple[Any, str]],
    episodes: int,
    seed: int,
    max_seconds: float,
) -> dict[str, Any]:
    env = RosterPurposeEnv(
        fighter_name=fighter_name,
        stage_name=stage_name,
        seed=seed,
        max_episode_seconds=max_seconds,
        items_probability=0.0,
        curriculum_strength=0.0,
    )
    env.opponent_deterministic = True
    outcomes: Counter[str] = Counter()
    qualities: Counter[str] = Counter()
    rewards: list[float] = []
    try:
        for episode in range(max(1, episodes)):
            opponent, source = opponents[episode % len(opponents)]
            _set_training_pool(env, [(opponent, 1.0, source)])
            observation, _info = env.reset(
                seed=seed + episode,
                options={
                    "curriculum": "duel",
                    "agent_slot": episode % 2,
                    "swap_spawns": bool((episode // 2) % 2),
                    "items_enabled": False,
                },
            )
            terminated = truncated = False
            total = 0.0
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action, _state = model.predict(
                    observation,
                    action_masks=env.action_masks(),
                    deterministic=True,
                )
                observation, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
            outcomes[str(info.get("outcome", "unknown"))] += 1
            rewards.append(total)
            for key in QUALITY_KEYS:
                qualities[key] += float(info.get("quality", {}).get(key, 0.0))
    finally:
        env.close()
    count = max(1, sum(outcomes.values()))
    decisive = outcomes["win"] + outcomes["loss"]
    quality = {key: qualities[key] / count for key in QUALITY_KEYS}
    report: dict[str, Any] = {
        "episodes": count,
        "win_rate": outcomes["win"] / count,
        "decisive_finish_rate": decisive / count,
        "timeout_rate": 1.0 - decisive / count,
        "outcomes": dict(outcomes),
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "quality": quality,
    }
    passed, failures = roster_behavior_gate(report)
    report["behavior_gate_passed"] = passed
    report["gate_failures"] = failures
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="训练单个非桃子角色的v6级别21/22候选")
    parser.add_argument("--fighter", required=True, choices=TRAINABLE_ROSTER)
    parser.add_argument("--stage", choices=("Mogadishu",), default="Mogadishu")
    parser.add_argument("--run-id", default="roster_b1")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--foundation-steps", type=int, default=160_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--steps-per-level", type=int, default=60_000)
    parser.add_argument("--eval-episodes", type=int, default=12)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--eval-max-seconds", type=float, default=60.0)
    parser.add_argument("--autosave-steps", type=int, default=10_000)
    parser.add_argument("--rollout-steps", type=int, default=1024)
    parser.add_argument("--device", choices=("cpu", "mps", "auto"), default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    for value, label in (
        (args.foundation_steps, "foundation-steps"),
        (args.rounds, "rounds"),
        (args.steps_per_level, "steps-per-level"),
        (args.eval_episodes, "eval-episodes"),
        (args.autosave_steps, "autosave-steps"),
        (args.rollout_steps, "rollout-steps"),
    ):
        if value < 1:
            raise SystemExit(f"--{label} 必须大于0")
    source_path = args.source.expanduser().resolve()
    if not source_path.is_file():
        raise SystemExit(f"找不到冻结桃子起始模型: {source_path}")

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
    except ImportError as exc:
        raise SystemExit(
            "缺少训练依赖，请使用 .venv-train/bin/python 运行"
        ) from exc

    scenario = TrainingScenario(
        args.fighter, args.fighter, args.stage, args.run_id, args.seed
    )
    with claim_training_job(scenario) as (directory, log_dir):
        if not args.skip_preflight:
            report = preflight_fighter(
                args.fighter,
                stage_name=args.stage,
                seed=args.seed,
                rollout_decisions=40,
            )
            _atomic_json(directory / "preflight.json", report)
            print(f"预检通过: {args.fighter} / {args.stage}")

        foundation_path = directory / "foundation_model.zip"
        candidate21_path = directory / "candidate_level21_model.zip"
        candidate22_path = directory / "candidate_level22_model.zip"
        state_path = directory / "training_state.json"
        config_path = directory / "training_config.json"
        existing_models = any(
            path.is_file()
            for path in (foundation_path, candidate21_path, candidate22_path)
        )
        if existing_models and not args.resume:
            raise SystemExit(
                f"目录已有候选: {directory}\n继续请加 --resume；重练请换 --run-id。"
            )
        if args.resume and not foundation_path.is_file():
            raise SystemExit(f"无法续训，缺少 {foundation_path}")

        config = {
            "schema": "glorton-roster-training-b1",
            "created_utc": _now(),
            "fighter_name": args.fighter,
            "stage_name": args.stage,
            "run_id": args.run_id,
            "seed": args.seed,
            "source": str(source_path),
            "observation_version": ROSTER_OBSERVATION_VERSION,
            "observation_size": ROSTER_OBSERVATION_SIZE,
            "action_count": 14,
            "normal_game_training_overrides": False,
        }
        if config_path.is_file():
            current = _load_json(config_path, {})
            stable_keys = (
                "fighter_name",
                "stage_name",
                "run_id",
                "seed",
                "source",
                "observation_version",
            )
            if any(current.get(key) != config.get(key) for key in stable_keys):
                raise SystemExit(f"续训配置不一致: {config_path}")
        else:
            _atomic_json(config_path, config)

        set_random_seed(args.seed)
        frozen_v5 = MaskablePPO.load(str(source_path), device="cpu")
        state: dict[str, Any] = _load_json(
            state_path,
            {"schema": "glorton-roster-state-b1", "completed_phases": [], "reports": []},
        )
        completed = set(str(item) for item in state.get("completed_phases", []))

        class AutosaveCallback(BaseCallback):
            def __init__(self, path: Path) -> None:
                super().__init__(verbose=0)
                self.path = path
                self.last_saved = 0

            def _on_training_start(self) -> None:
                self.last_saved = int(self.model.num_timesteps)

            def _on_step(self) -> bool:
                current = int(self.model.num_timesteps)
                if current - self.last_saved < args.autosave_steps:
                    return True
                _save_model(self.model, self.path)
                self.last_saved = current
                state["last_autosave_utc"] = _now()
                state["last_autosave_timesteps"] = current
                _atomic_json(state_path, state)
                print(f"自动保存 {self.path.name}: {current:,} 步")
                return True

        def learn_phase(model: Any, path: Path, phase: str, steps: int, log_name: str) -> None:
            nonlocal completed
            if phase in completed:
                print(f"跳过已完成阶段: {phase}")
                return
            active = state.get("active_phase")
            if isinstance(active, Mapping) and active.get("name") == phase:
                start = int(active["start_timesteps"])
                target = int(active["target_steps"])
                done = max(0, int(model.num_timesteps) - start)
                remaining = max(0, target - done)
            else:
                remaining = steps
                state["active_phase"] = {
                    "name": phase,
                    "start_timesteps": int(model.num_timesteps),
                    "target_steps": int(steps),
                }
                _atomic_json(state_path, state)
            print(f"===== {args.fighter} {phase}: 剩余 {remaining:,} 步 =====")
            try:
                if remaining:
                    model.learn(
                        total_timesteps=remaining,
                        reset_num_timesteps=False,
                        tb_log_name=log_name,
                        callback=AutosaveCallback(path),
                    )
            except BaseException:
                _save_model(model, path)
                state["interrupted_utc"] = _now()
                state["interrupted_timesteps"] = int(model.num_timesteps)
                _atomic_json(state_path, state)
                raise
            _save_model(model, path)
            completed.add(phase)
            state["completed_phases"] = sorted(completed)
            state["active_phase"] = None
            _atomic_json(state_path, state)

        raw_foundation = RosterPurposeEnv(
            fighter_name=args.fighter,
            stage_name=args.stage,
            seed=args.seed,
            max_episode_seconds=args.max_seconds,
            curriculum_strength=0.72,
        )
        _set_training_pool(
            raw_foundation,
            [
                ("active", 0.35, "active_probe"),
                ("melee", 0.30, "melee_probe"),
                (frozen_v5, 0.25, "frozen_peach_v5_purpose"),
                ("retreat", 0.06, "retreat_probe"),
                ("idle", 0.04, "idle_probe"),
            ],
        )
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
            foundation, transfer_report = initialize_roster_from_v5(
                MaskablePPO,
                source_path=source_path,
                env=foundation_env,
                seed=args.seed + 5,
                device=args.device,
                log_dir=log_dir,
                rollout_steps=args.rollout_steps,
            )
            _atomic_json(directory / "transfer_report.json", transfer_report)
            _save_model(foundation, foundation_path)
        try:
            learn_phase(
                foundation,
                foundation_path,
                "mixed_foundation",
                args.foundation_steps,
                f"{args.fighter}_{args.run_id}_foundation",
            )
        except KeyboardInterrupt:
            print(f"\n已保存 {args.fighter} 基础模型；原命令加 --resume 可续训。")
            return
        finally:
            foundation_env.close()

        foundation_policy = MaskablePPO.load(str(foundation_path), device="cpu")
        raw21 = RosterPurposeEnv(
            fighter_name=args.fighter,
            stage_name=args.stage,
            seed=args.seed + 21,
            max_episode_seconds=args.max_seconds,
        )
        raw22 = RosterPurposeEnv(
            fighter_name=args.fighter,
            stage_name=args.stage,
            seed=args.seed + 22,
            max_episode_seconds=args.max_seconds,
        )
        env21 = Monitor(raw21, info_keywords=("outcome", "curriculum", "lesson_success"))
        env22 = Monitor(raw22, info_keywords=("outcome", "curriculum", "lesson_success"))
        source21 = candidate21_path if args.resume and candidate21_path.is_file() else foundation_path
        source22 = candidate22_path if args.resume and candidate22_path.is_file() else foundation_path
        model21 = MaskablePPO.load(
            str(source21), env=env21, device=args.device, tensorboard_log=str(log_dir)
        )
        model22 = MaskablePPO.load(
            str(source22), env=env22, device=args.device, tensorboard_log=str(log_dir)
        )
        try:
            for round_no in range(1, args.rounds + 1):
                level21_phase = f"round_{round_no:02d}_level21"
                level22_phase = f"round_{round_no:02d}_level22"
                round_already_reported = any(
                    int(item.get("round", -1)) == round_no
                    for item in state.get("reports", [])
                )
                if (
                    level21_phase in completed
                    and level22_phase in completed
                    and round_already_reported
                ):
                    print(f"跳过已训练并评估的第 {round_no} 轮")
                    continue
                strength = 0.42 if args.rounds == 1 else 0.42 - 0.20 * (round_no - 1) / (args.rounds - 1)
                raw21.set_curriculum_strength(strength)
                raw22.set_curriculum_strength(strength)
                _set_training_pool(
                    raw21,
                    [
                        (model22, 0.45, "current_level22"),
                        (foundation_policy, 0.22, "role_foundation"),
                        ("active", 0.15, "active_probe"),
                        ("melee", 0.10, "melee_probe"),
                        (frozen_v5, 0.05, "frozen_peach_v5_purpose"),
                        ("retreat", 0.03, "retreat_probe"),
                    ],
                )
                learn_phase(
                    model21,
                    candidate21_path,
                    level21_phase,
                    args.steps_per_level,
                    f"{args.fighter}_{args.run_id}_level21",
                )
                _set_training_pool(
                    raw22,
                    [
                        (model21, 0.50, "current_level21"),
                        (foundation_policy, 0.20, "role_foundation"),
                        ("active", 0.14, "active_probe"),
                        ("melee", 0.09, "melee_probe"),
                        (frozen_v5, 0.04, "frozen_peach_v5_purpose"),
                        ("retreat", 0.03, "retreat_probe"),
                    ],
                )
                learn_phase(
                    model22,
                    candidate22_path,
                    level22_phase,
                    args.steps_per_level,
                    f"{args.fighter}_{args.run_id}_level22",
                )
                report21 = evaluate_roster_model(
                    model21,
                    fighter_name=args.fighter,
                    stage_name=args.stage,
                    opponents=((model22, "current_level22"), ("active", "active_probe"), ("melee", "melee_probe")),
                    episodes=args.eval_episodes,
                    seed=args.seed + round_no * 20_000,
                    max_seconds=args.eval_max_seconds,
                )
                report22 = evaluate_roster_model(
                    model22,
                    fighter_name=args.fighter,
                    stage_name=args.stage,
                    opponents=((model21, "current_level21"), ("active", "active_probe"), ("melee", "melee_probe")),
                    episodes=args.eval_episodes,
                    seed=args.seed + round_no * 20_000 + 10_000,
                    max_seconds=args.eval_max_seconds,
                )
                round_report = {
                    "round": round_no,
                    "created_utc": _now(),
                    "level21": report21,
                    "level22": report22,
                }
                reports = list(state.get("reports", []))
                reports = [item for item in reports if int(item.get("round", -1)) != round_no]
                reports.append(round_report)
                state["reports"] = reports
                state["last_completed_round"] = round_no
                _atomic_json(state_path, state)
                _atomic_json(directory / f"evaluation_round_{round_no:02d}.json", round_report)
                print(json.dumps(round_report, ensure_ascii=False, indent=2))
        except KeyboardInterrupt:
            print(f"\n已保存 {args.fighter} 21/22候选；原命令加 --resume 可续训。")
            return
        finally:
            env21.close()
            env22.close()

        final = {
            "schema": "glorton-roster-candidates-b1",
            "completed_utc": _now(),
            "fighter_name": args.fighter,
            "stage_name": args.stage,
            "level21": candidate21_path.name,
            "level22": candidate22_path.name,
            "reports": state.get("reports", []),
            "requires_human_review": True,
        }
        _atomic_json(directory / "candidates.json", final)
        print(f"\n{args.fighter} 训练完成")
        print(f"21级候选: {candidate21_path}")
        print(f"22级候选: {candidate22_path}")


if __name__ == "__main__":
    main()
