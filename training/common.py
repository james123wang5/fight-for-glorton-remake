from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .peach_env import PeachVsLevel20Env


Policy = Callable[[np.ndarray, PeachVsLevel20Env, int], np.ndarray]


@dataclass
class EpisodeResult:
    episode: int
    seed: int
    outcome: str
    reward: float
    elapsed_seconds: float
    decision_steps: int
    agent_kos: int
    opponent_kos: int
    agent_damage: float
    opponent_damage: float


def run_policy_episodes(
    policy: Policy,
    *,
    episodes: int,
    seed: int,
    max_episode_seconds: float,
    record_count: int = 0,
    record_dir: Path | None = None,
) -> tuple[list[EpisodeResult], float]:
    env = PeachVsLevel20Env(
        seed=seed,
        max_episode_seconds=max_episode_seconds,
        randomize_spawns=False,
    )
    results: list[EpisodeResult] = []
    started = time.perf_counter()
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            observation, _ = env.reset(
                seed=episode_seed,
                options={"swap_spawns": bool(episode % 2)},
            )
            should_record = episode < record_count and record_dir is not None
            if should_record:
                env.start_recording({"evaluation_episode": episode + 1})
            terminated = truncated = False
            info: dict[str, Any] = {}
            while not (terminated or truncated):
                action = policy(observation, env, episode)
                observation, _, terminated, truncated, info = env.step(action)
            if should_record:
                record_dir.mkdir(parents=True, exist_ok=True)
                recording = env.stop_recording()
                target = record_dir / f"episode_{episode + 1:03d}_seed_{episode_seed}.json"
                target.write_text(
                    json.dumps(recording, ensure_ascii=False, separators=(",", ":")),
                    encoding="utf-8",
                )
            results.append(
                EpisodeResult(
                    episode=episode + 1,
                    seed=episode_seed,
                    outcome=str(info["outcome"]),
                    reward=float(info["episode_reward"]),
                    elapsed_seconds=float(info["elapsed_seconds"]),
                    decision_steps=int(info["decision_steps"]),
                    agent_kos=int(info["agent_kos"]),
                    opponent_kos=int(info["opponent_kos"]),
                    agent_damage=float(info["agent_damage"]),
                    opponent_damage=float(info["opponent_damage"]),
                )
            )
    finally:
        env.close()
    return results, time.perf_counter() - started


def print_summary(results: list[EpisodeResult], wall_seconds: float, *, label: str) -> None:
    if not results:
        print("没有完成任何对局。")
        return
    counts = Counter(result.outcome for result in results)
    wins = counts["win"] + counts["timeout_win"]
    losses = counts["loss"] + counts["timeout_loss"]
    draws = counts["draw"] + counts["timeout_draw"]
    total_steps = sum(result.decision_steps for result in results)
    average = lambda values: sum(values) / len(values)
    print(f"\n{label}")
    print(f"对局: {len(results)}  胜/负/平: {wins}/{losses}/{draws}  胜率: {wins / len(results):.1%}")
    print(
        "正常结束/超时: "
        f"{counts['win'] + counts['loss'] + counts['draw']}/"
        f"{counts['timeout_win'] + counts['timeout_loss'] + counts['timeout_draw']}"
    )
    print(f"平均奖励: {average([item.reward for item in results]):.3f}")
    print(
        "平均击杀: "
        f"AI {average([item.agent_kos for item in results]):.2f} / "
        f"20级 {average([item.opponent_kos for item in results]):.2f}"
    )
    print(f"平均虚拟对局时长: {average([item.elapsed_seconds for item in results]):.1f} 秒")
    print(f"运行速度: {total_steps / max(wall_seconds, 1e-9):.0f} 个AI决策/秒")
    print(f"实际耗时: {wall_seconds:.1f} 秒")
