from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .roster_contract import FIGHTER_ORDER
from .roster_env import RosterPurposeEnv
from .roster_observation import ROSTER_OBSERVATION_SIZE
from .roster_options import role_combat_profile
from .v5_options import Purpose


TRAINABLE_ROSTER = tuple(name for name in FIGHTER_ORDER if name != "PeachPlayer")


def _open_probe_platform(env: RosterPurposeEnv) -> Any:
    candidates = [
        item
        for item in env.runtime.stage.platforms
        if not item.moving and 190.0 <= float(item.rect.w) <= 420.0
    ]
    if not candidates:
        raise RuntimeError(f"{env.roster_stage_name} has no open probe platform")
    return max(candidates, key=lambda item: item.rect.w)


def _place_ground_special_probe(env: RosterPurposeEnv) -> None:
    platform = _open_probe_platform(env)
    profile = role_combat_profile(env.agent)
    spacing = min(
        profile.ground_max_distance - 8.0,
        max(profile.ground_min_distance + 18.0, 115.0),
    )
    center = float(platform.rect.centerx)
    env._place_fighter(env.agent, platform, center - spacing / 2)
    env._place_fighter(env.opponent, platform, center + spacing / 2)
    env.agent.facing = 1
    env.opponent.facing = -1
    for controller in env.intent_controllers:
        controller.reset()


def _place_up_special_probe(env: RosterPurposeEnv) -> None:
    platform = _open_probe_platform(env)
    center = float(platform.rect.centerx)
    env._place_fighter(env.agent, platform, center - 28.0)
    env._place_fighter(env.opponent, platform, center + 28.0)
    env.opponent.pos.y -= 52.0
    env.opponent.prev_pos.update(env.opponent.pos)
    env.opponent.on_ground = False
    env.opponent.ground_platform = None
    env.opponent.yinc = 0.0
    env.agent.facing = 1
    env.opponent.facing = -1
    env.agent.spec_up_ok = True
    for controller in env.intent_controllers:
        controller.reset()


def _run_attack_probe(
    env: RosterPurposeEnv,
    *,
    purpose: Purpose,
    attack_label: str,
    placement: Any,
) -> dict[str, Any]:
    env.reset(
        seed=env.base_seed + int(purpose),
        options={"curriculum": "duel", "agent_slot": 0, "items_enabled": False},
    )
    placement(env)
    mask = env.action_masks()
    if not mask[purpose]:
        raise RuntimeError(
            f"{env.roster_fighter_name} preflight did not expose {purpose.name}"
        )
    before = int(env._attack_counts[attack_label])
    _observation, _reward, _terminated, _truncated, _info = env.step(int(purpose))
    after = int(env._attack_counts[attack_label])
    if after <= before:
        raise RuntimeError(
            f"{env.roster_fighter_name} {purpose.name} did not start {attack_label}"
        )
    return {"purpose": purpose.name, "attack": attack_label, "started": True}


def preflight_fighter(
    fighter_name: str,
    *,
    stage_name: str = "Mogadishu",
    seed: int = 20260716,
    rollout_decisions: int = 40,
) -> dict[str, Any]:
    env = RosterPurposeEnv(
        fighter_name=fighter_name,
        stage_name=stage_name,
        seed=seed,
        max_episode_seconds=max(8.0, rollout_decisions / 10.0 + 2.0),
        items_probability=0.0,
        curriculum_strength=0.0,
    )
    try:
        ground = _run_attack_probe(
            env,
            purpose=Purpose.AIMED_SHOT,
            attack_label="specialGround",
            placement=_place_ground_special_probe,
        )
        upward = _run_attack_probe(
            env,
            purpose=Purpose.ROCKET,
            attack_label="specialUp",
            placement=_place_up_special_probe,
        )
        observation, _info = env.reset(
            seed=seed + 100,
            options={"curriculum": "duel", "agent_slot": 0, "items_enabled": False},
        )
        legal_counts: list[int] = []
        rewards: list[float] = []
        for _decision in range(max(1, rollout_decisions)):
            if observation.shape != (ROSTER_OBSERVATION_SIZE,) or not np.isfinite(
                observation
            ).all():
                raise RuntimeError(f"{fighter_name} emitted invalid roster observation")
            mask = env.action_masks()
            legal = np.flatnonzero(mask)
            if not legal.size:
                raise RuntimeError(f"{fighter_name} emitted an empty action mask")
            legal_counts.append(int(legal.size))
            priorities = (
                Purpose.RECOVER,
                Purpose.NAVIGATE,
                Purpose.AIR_CHASE,
                Purpose.MELEE,
                Purpose.AIMED_SHOT,
                Purpose.ROCKET,
                Purpose.CHASE,
                Purpose.CONTINUE,
            )
            action = next((int(item) for item in priorities if mask[item]), int(legal[0]))
            observation, reward, terminated, truncated, _info = env.step(action)
            rewards.append(float(reward))
            if terminated or truncated:
                observation, _info = env.reset(
                    options={"curriculum": "duel", "agent_slot": 0, "items_enabled": False}
                )
        return {
            "fighter_name": fighter_name,
            "stage_name": stage_name,
            "observation_size": ROSTER_OBSERVATION_SIZE,
            "ground_special": ground,
            "up_special": upward,
            "rollout_decisions": len(rewards),
            "min_legal_actions": min(legal_counts),
            "max_legal_actions": max(legal_counts),
            "finite_rewards": bool(np.isfinite(np.asarray(rewards)).all()),
            "passed": True,
        }
    finally:
        env.close()


def preflight_roster(
    fighters: tuple[str, ...] = TRAINABLE_ROSTER,
    *,
    stage_name: str = "Mogadishu",
    seed: int = 20260716,
    rollout_decisions: int = 40,
) -> dict[str, Any]:
    reports = [
        preflight_fighter(
            fighter,
            stage_name=stage_name,
            seed=seed + index * 1000,
            rollout_decisions=rollout_decisions,
        )
        for index, fighter in enumerate(fighters)
    ]
    return {
        "schema": "glorton-roster-preflight-b1",
        "stage_name": stage_name,
        "fighters": reports,
        "passed": all(bool(report["passed"]) for report in reports),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="全角色阶段B训练前强制预检")
    parser.add_argument("--fighters", nargs="+", choices=TRAINABLE_ROSTER, default=TRAINABLE_ROSTER)
    parser.add_argument("--stage", choices=("Mogadishu",), default="Mogadishu")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--rollout-decisions", type=int, default=40)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = preflight_roster(
        tuple(args.fighters),
        stage_name=args.stage,
        seed=args.seed,
        rollout_decisions=args.rollout_decisions,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(args.output.resolve())
    else:
        print(text, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
