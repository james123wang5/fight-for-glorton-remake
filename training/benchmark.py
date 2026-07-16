from __future__ import annotations

import argparse

import numpy as np

from .common import print_summary, run_policy_episodes
from .peach_env import PeachVsLevel20Env


def random_policy(_: np.ndarray, env: PeachVsLevel20Env, __: int) -> np.ndarray:
    return env.action_space.sample()


def heuristic_policy(_: np.ndarray, env: PeachVsLevel20Env, __: int) -> np.ndarray:
    """Small non-learning sanity opponent; it is not meant to be the final AI."""

    agent = env.agent
    opponent = env.opponent
    bounds = env.runtime.stage.bounds
    dx = float(opponent.pos.x - agent.pos.x)
    dy = float(opponent.pos.y - agent.pos.y)
    toward_opponent = 2 if dx > 12 else 1 if dx < -12 else 0
    toward_center = 2 if agent.pos.x < bounds.centerx else 1

    # Recovery has priority: jump, then use Peach's up-special rocket motion.
    if env._edge_danger(agent) > 0.72 or agent.pos.y > bounds.centery + bounds.h * 0.22:
        combat = 2 if agent.spec_up_ok else 0
        return np.asarray([toward_center, 1, combat], dtype=np.int64)
    if abs(dx) < 42 and abs(dy) < 55:
        vertical = 1 if opponent.pos.y < agent.pos.y - 15 and agent.on_ground else 0
        return np.asarray([toward_opponent, vertical, 1], dtype=np.int64)
    if abs(dx) > 150 and abs(dy) < 80 and env._decision_steps % 12 == 0:
        return np.asarray([0, 0, 2], dtype=np.int64)
    if dy < -55 and agent.on_ground:
        return np.asarray([toward_opponent, 1, 0], dtype=np.int64)
    return np.asarray([toward_opponent, 0, 0], dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="桃子 vs 原版20级AI的训练前基准")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--max-seconds", type=float, default=180.0)
    parser.add_argument("--policy", choices=("heuristic", "random"), default="heuristic")
    args = parser.parse_args()
    policy = heuristic_policy if args.policy == "heuristic" else random_policy
    results, wall_seconds = run_policy_episodes(
        policy,
        episodes=max(1, args.episodes),
        seed=args.seed,
        max_episode_seconds=max(1.0, args.max_seconds),
    )
    print_summary(results, wall_seconds, label=f"基准策略: {args.policy}")


if __name__ == "__main__":
    main()
