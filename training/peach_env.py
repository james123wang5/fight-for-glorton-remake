from __future__ import annotations

import copy
import math
import os
from typing import Any, Mapping, Sequence

if not (os.environ.get("GLORTON_AI21_MODEL") or os.environ.get("GLORTON_AI22_MODEL")):
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.runtime import Stage
from src.simulation import BattleSimulation


MATCH_CONFIG: dict[str, Any] = {
    "type": "vsmode",
    "selected_stage": "Mogadishu",
    "stage": "Mogadishu",
    "limit_mode": "stock",
    "limit_value": 3,
    "players": [
        {
            "fighter": "PeachPlayer",
            "color": 0,
            "computer": False,
            "enabled": True,
            "level": 7,
            "team_index": 0,
        },
        {
            "fighter": "PeachPlayer",
            "color": 1,
            "computer": True,
            "enabled": True,
            "level": 20,
            "team_index": 1,
        },
    ],
}

STATE_LABELS = ("stop", "goright", "goleft", "crouch", "thrown", "ko", "spawn", "dead")
ATTACK_LABELS = (
    "",
    "punchGround",
    "punchRun",
    "punchUp",
    "punchAir",
    "specialGround",
    "specialAir",
    "specialUp",
    "specialBackThrow",
    "koAttack",
)


class PeachVsLevel20Env(gym.Env[np.ndarray, np.ndarray]):
    """Stage-one RL task: Peach (agent) versus the source-style level-20 AI.

    Action is ``MultiDiscrete([3, 3, 4])``:

    * horizontal: neutral / left / right
    * vertical: neutral / up / down
    * combat: none / punch / special / shield

    Up+punch and up+special are emitted on the same source tick, exposing the
    original uppercut and rocket-launcher combinations.  An uncombined up is a
    jump/double-jump press.  Each policy action lasts two 25 ms physics ticks,
    so the policy runs at 20 Hz while combat remains at its original 40 Hz.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: int = 0,
        max_episode_seconds: float = 180.0,
        frame_skip: int = 2,
        randomize_spawns: bool = True,
        ringout_reward: float = 1.0,
        result_reward: float = 0.5,
        edge_progress_scale: float = 0.15,
        damage_scale: float = 0.0005,
    ) -> None:
        super().__init__()
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least one")
        self.base_seed = int(seed)
        self.next_seed = int(seed)
        self.frame_skip = int(frame_skip)
        self.max_episode_seconds = float(max_episode_seconds)
        self.max_ticks = max(1, math.ceil(max_episode_seconds * 1000 / 25))
        self.randomize_spawns = bool(randomize_spawns)
        self.ringout_reward = float(ringout_reward)
        self.result_reward = float(result_reward)
        self.edge_progress_scale = float(edge_progress_scale)
        self.damage_scale = float(damage_scale)

        # The runtime owns a private manifest dictionary, so these overrides do
        # not mutate load_manifest() or the configuration used by play.py.
        self.simulation = BattleSimulation.headless(
            seed=self.base_seed,
            match_config=copy.deepcopy(MATCH_CONFIG),
        )
        self.runtime = self.simulation.runtime
        self.runtime.audio = None
        self._apply_training_overrides()

        self.action_space = spaces.MultiDiscrete(np.array([3, 3, 4], dtype=np.int64))
        self._episode_ticks = 0
        self._decision_steps = 0
        self._previous_combat = 0
        self._episode_seed = self.base_seed
        self._spawns_swapped = False
        self._episode_reward = 0.0
        self._reward_totals = self._empty_reward_components()

        self._reset_runtime(self.base_seed, swap_spawns=False)
        sample_observation = self._observation()
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=sample_observation.shape,
            dtype=np.float32,
        )

    @property
    def agent(self) -> Any:
        return self.runtime.fighters[0]

    @property
    def opponent(self) -> Any:
        return self.runtime.fighters[1]

    def _apply_training_overrides(self) -> None:
        self.runtime.match_config = copy.deepcopy(MATCH_CONFIG)
        self.runtime.manifest["match"]["limit_mode"] = "stock"
        self.runtime.manifest["match"]["starting_lives"] = 3
        self.runtime.manifest["items"]["frequency"] = 0
        if self.runtime.stage.name != "Mogadishu":
            self.runtime.stage = Stage(self.runtime.manifest, "Mogadishu")

    def _reset_runtime(self, seed: int, *, swap_spawns: bool) -> None:
        self._apply_training_overrides()
        self.simulation.reset(seed)
        self.runtime.items.clear()
        self.runtime.item_gen_timer_ms = 0

        first = self.runtime.stage.spawn_point("SpawnP2" if swap_spawns else "SpawnP1")
        second = self.runtime.stage.spawn_point("SpawnP1" if swap_spawns else "SpawnP2")
        for fighter, spawn in zip(self.runtime.fighters, (first, second), strict=True):
            fighter.spawn_pos.update(spawn)
            fighter.start_intro_spawn()

        # Skip the seven-second presentation countdown only inside training.
        # finish_intro_spawn is the same transition used by the real GO! flow.
        self.runtime.ready_set = -1
        self.runtime._apply_ready_step()
        self.runtime.stage_time_ms = 0
        self.runtime.game_time_seconds = 0
        self.runtime.fight_timer_accumulator_ms = 0
        self.simulation.tick_index = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is None:
            seed = self.next_seed
            self.next_seed += 1
        else:
            self.next_seed = int(seed) + 1
        super().reset(seed=int(seed))
        options = dict(options or {})
        if "swap_spawns" in options:
            swap_spawns = bool(options["swap_spawns"])
        else:
            # Consecutive episode seeds alternate sides exactly.  This gives
            # equal exposure without introducing another hidden random draw.
            swap_spawns = bool(self.randomize_spawns and int(seed) % 2)

        self._episode_seed = int(seed)
        self._spawns_swapped = swap_spawns
        self._episode_ticks = 0
        self._decision_steps = 0
        self._previous_combat = 0
        self._episode_reward = 0.0
        self._reward_totals = self._empty_reward_components()
        self._reset_runtime(self._episode_seed, swap_spawns=swap_spawns)
        self.action_space.seed(self._episode_seed)
        return self._observation(), self._info("ongoing")

    def step(
        self,
        action: np.ndarray | Sequence[int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action_array = np.asarray(action, dtype=np.int64)
        if action_array.shape != (3,) or not self.action_space.contains(action_array):
            raise ValueError(f"invalid action {action!r}; expected MultiDiscrete([3, 3, 4])")

        before = self._reward_state()
        horizontal, vertical, combat = (int(value) for value in action_array)
        for internal_tick in range(self.frame_skip):
            controls = self._controls_for_action(
                horizontal,
                vertical,
                combat,
                press_tick=internal_tick == 0,
            )
            self.simulation.step_fast([controls, {}])
            self._episode_ticks += 1
            if self.runtime.match_state == "game_set" or self._episode_ticks >= self.max_ticks:
                break

        self._decision_steps += 1
        self._previous_combat = combat
        terminated = self.runtime.match_state == "game_set"
        truncated = self._episode_ticks >= self.max_ticks and not terminated
        outcome = self._outcome(terminated=terminated, truncated=truncated)
        reward, components = self._reward(before, outcome=outcome, terminated=terminated)
        self._episode_reward += reward
        for key, value in components.items():
            self._reward_totals[key] += value

        return self._observation(), float(reward), terminated, truncated, self._info(outcome)

    def _controls_for_action(
        self,
        horizontal: int,
        vertical: int,
        combat: int,
        *,
        press_tick: bool,
    ) -> dict[str, bool]:
        controls = {
            "left": horizontal == 1,
            "right": horizontal == 2,
            "down": vertical == 2,
        }
        if not press_tick:
            return controls

        upper_attack = vertical == 1 and combat in {1, 2}
        controls["up_trace"] = upper_attack
        controls["jump_pressed"] = vertical == 1 and combat == 0
        controls["punch_pressed"] = combat == 1
        controls["special_pressed"] = combat == 2
        controls["shield_pressed"] = combat == 3 and self._previous_combat != 3
        controls["shield_released"] = combat != 3 and self._previous_combat == 3
        return controls

    @staticmethod
    def _empty_reward_components() -> dict[str, float]:
        return {"ringout": 0.0, "result": 0.0, "edge_progress": 0.0, "damage": 0.0}

    def _reward_state(self) -> dict[str, float | int]:
        return {
            "agent_lives": int(self.agent.lives),
            "opponent_lives": int(self.opponent.lives),
            "agent_damage": float(self.agent.damage_amnt),
            "opponent_damage": float(self.opponent.damage_amnt),
            "agent_danger": self._edge_danger(self.agent),
            "opponent_danger": self._edge_danger(self.opponent),
        }

    def _reward(
        self,
        before: Mapping[str, float | int],
        *,
        outcome: str,
        terminated: bool,
    ) -> tuple[float, dict[str, float]]:
        components = self._empty_reward_components()
        agent_lives_lost = max(0, int(before["agent_lives"]) - int(self.agent.lives))
        opponent_lives_lost = max(0, int(before["opponent_lives"]) - int(self.opponent.lives))
        components["ringout"] = self.ringout_reward * (
            opponent_lives_lost - agent_lives_lost
        )

        if agent_lives_lost == 0 and opponent_lives_lost == 0:
            damage_dealt = max(0.0, float(self.opponent.damage_amnt) - float(before["opponent_damage"]))
            damage_taken = max(0.0, float(self.agent.damage_amnt) - float(before["agent_damage"]))
            components["damage"] = self.damage_scale * (damage_dealt - damage_taken)
            edge_delta = (
                self._edge_danger(self.opponent)
                - float(before["opponent_danger"])
                - self._edge_danger(self.agent)
                + float(before["agent_danger"])
            )
            components["edge_progress"] = self.edge_progress_scale * edge_delta

        if terminated:
            if outcome == "win":
                components["result"] = self.result_reward
            elif outcome == "loss":
                components["result"] = -self.result_reward
        return sum(components.values()), components

    def _edge_danger(self, fighter: Any) -> float:
        bounds = self.runtime.stage.bounds
        x = abs((float(fighter.pos.x) - bounds.centerx) / max(1.0, bounds.w / 2))
        y = abs((float(fighter.pos.y) - bounds.centery) / max(1.0, bounds.h / 2))
        return float(max(x, y))

    def _outcome(self, *, terminated: bool, truncated: bool) -> str:
        if terminated:
            if self.runtime.match_winner is self.agent:
                return "win"
            if self.runtime.match_winner is self.opponent:
                return "loss"
            return "draw"
        if not truncated:
            return "ongoing"
        agent_score = (int(self.agent.lives), -float(self.agent.damage_amnt), int(self.agent.kos - self.agent.deaths))
        opponent_score = (
            int(self.opponent.lives),
            -float(self.opponent.damage_amnt),
            int(self.opponent.kos - self.opponent.deaths),
        )
        if agent_score > opponent_score:
            return "timeout_win"
        if agent_score < opponent_score:
            return "timeout_loss"
        return "timeout_draw"

    def _stats(self) -> dict[str, int | float]:
        return {
            "agent_lives": int(self.agent.lives),
            "opponent_lives": int(self.opponent.lives),
            "agent_damage": float(self.agent.damage_amnt),
            "opponent_damage": float(self.opponent.damage_amnt),
            "agent_kos": int(self.agent.kos),
            "opponent_kos": int(self.opponent.kos),
            "agent_deaths": int(self.agent.deaths),
            "opponent_deaths": int(self.opponent.deaths),
        }

    def _info(self, outcome: str) -> dict[str, Any]:
        info: dict[str, Any] = {
            "outcome": outcome,
            "seed": self._episode_seed,
            "stage": self.runtime.stage.name,
            "items_enabled": False,
            "spawns_swapped": self._spawns_swapped,
            "elapsed_ticks": self._episode_ticks,
            "elapsed_seconds": self._episode_ticks * self.simulation.tick_ms / 1000,
            "decision_steps": self._decision_steps,
            "episode_reward": self._episode_reward,
            "reward_components": dict(self._reward_totals),
        }
        info.update(self._stats())
        return info

    def _observation(
        self,
        *,
        agent: Any | None = None,
        opponent: Any | None = None,
        episode_ticks: int | None = None,
        max_ticks: int | None = None,
        spawns_swapped: bool | None = None,
    ) -> np.ndarray:
        agent = self.agent if agent is None else agent
        opponent = self.opponent if opponent is None else opponent
        episode_ticks = self._episode_ticks if episode_ticks is None else episode_ticks
        max_ticks = self.max_ticks if max_ticks is None else max_ticks
        spawns_swapped = self._spawns_swapped if spawns_swapped is None else spawns_swapped
        bounds = self.runtime.stage.bounds
        values: list[float] = [
            episode_ticks / max(1, max_ticks),
            math.sin(self.runtime.stage_time_ms * math.tau / 10_000),
            math.cos(self.runtime.stage_time_ms * math.tau / 10_000),
            1.0 if spawns_swapped else -1.0,
        ]
        values.extend(self._fighter_observation(agent))
        values.extend(self._fighter_observation(opponent))
        values.extend(
            [
                (opponent.pos.x - agent.pos.x) / max(1.0, bounds.w),
                (opponent.pos.y - agent.pos.y) / max(1.0, bounds.h),
                (opponent.xinc - agent.xinc) / 30.0,
                (opponent.yinc - agent.yinc) / 30.0,
                math.hypot(
                    opponent.pos.x - agent.pos.x,
                    opponent.pos.y - agent.pos.y,
                )
                / max(1.0, math.hypot(bounds.w, bounds.h)),
            ]
        )
        values.extend(self._projectile_observation(agent=agent, limit=3))
        return np.clip(np.asarray(values, dtype=np.float32), -5.0, 5.0)

    def _fighter_observation(self, fighter: Any) -> list[float]:
        bounds = self.runtime.stage.bounds
        state = str(fighter.state)
        attack = str(fighter.current_attack)
        values = [
            (fighter.pos.x - bounds.centerx) / max(1.0, bounds.w / 2),
            (fighter.pos.y - bounds.centery) / max(1.0, bounds.h / 2),
            fighter.xinc / 30.0,
            fighter.yinc / 30.0,
            fighter.damage_amnt / 300.0,
            fighter.lives / 3.0,
            fighter.shield_size / 100.0,
            float(fighter.facing),
            float(fighter.attack_facing),
            float(bool(fighter.on_ground)),
            float(bool(fighter.has_control)),
            float(bool(fighter.invincible)),
            float(bool(fighter.shielded)),
            float(bool(fighter.spec_up_ok)),
            float(bool(fighter.out_of_camera)),
            float(bool(fighter.dead)),
            fighter.ctrl_loss / 1000.0,
            fighter.paralized / 1000.0,
            fighter.electrocuted_ms / 1000.0,
            fighter.spawn_invincible_ms / 3000.0,
            self._edge_danger(fighter),
        ]
        values.extend(float(fighter.jumpstate == index) for index in range(3))
        values.extend(float(state == label) for label in STATE_LABELS)
        values.extend(float(attack == label) for label in ATTACK_LABELS)
        attack_frames = max(
            1,
            len(fighter.animations.get(attack, {}).get("frames", [])) if attack else 1,
        )
        values.append(float(fighter.attack_frame) / attack_frames)
        values.extend(
            [
                (fighter.pos.x - bounds.left) / max(1.0, bounds.w),
                (bounds.right - fighter.pos.x) / max(1.0, bounds.w),
                (fighter.pos.y - bounds.top) / max(1.0, bounds.h),
                (bounds.bottom - fighter.pos.y) / max(1.0, bounds.h),
            ]
        )
        values.extend(self._nearest_platform_observation(fighter))
        return values

    def _nearest_platform_observation(self, fighter: Any) -> list[float]:
        bounds = self.runtime.stage.bounds
        if not self.runtime.stage.platforms:
            return [0.0] * 6

        def distance(platform: Any) -> float:
            gap_x = max(platform.rect.left - fighter.pos.x, 0, fighter.pos.x - platform.rect.right)
            return math.hypot(gap_x, platform.rect.top - fighter.pos.y)

        platform = min(self.runtime.stage.platforms, key=distance)
        gap_x = max(platform.rect.left - fighter.pos.x, 0, fighter.pos.x - platform.rect.right)
        return [
            gap_x / max(1.0, bounds.w),
            (platform.rect.top - fighter.pos.y) / max(1.0, bounds.h),
            platform.rect.w / max(1.0, bounds.w),
            float(bool(platform.moving)),
            float(fighter.ground_platform is platform),
            (platform.rect.centerx - fighter.pos.x) / max(1.0, bounds.w),
        ]

    def _projectile_observation(self, *, agent: Any | None = None, limit: int) -> list[float]:
        agent = self.agent if agent is None else agent
        bounds = self.runtime.stage.bounds
        projectiles = [
            *( (projectile, 0) for projectile in self.runtime.bullets ),
            *( (projectile, 1) for projectile in self.runtime.rockets ),
            *( (projectile, 2) for projectile in self.runtime.special_projectiles ),
        ]
        projectiles.sort(
            key=lambda item: math.hypot(
                item[0].pos.x - agent.pos.x,
                item[0].pos.y - agent.pos.y,
            )
        )
        values: list[float] = []
        for projectile, kind_index in projectiles[:limit]:
            life = max(1, int(getattr(projectile, "life", getattr(projectile, "config", {}).get("life_ms", 3000))))
            values.extend(
                [
                    (projectile.pos.x - agent.pos.x) / max(1.0, bounds.w),
                    (projectile.pos.y - agent.pos.y) / max(1.0, bounds.h),
                    float(getattr(projectile, "xinc", 0.0)) / 30.0,
                    float(getattr(projectile, "yinc", 0.0)) / 30.0,
                    float(getattr(projectile, "sender", None) is agent),
                    float(getattr(projectile, "age", 0)) / life,
                    float(kind_index == 0),
                    float(kind_index == 1),
                    float(kind_index == 2),
                ]
            )
        values.extend([0.0] * ((limit - min(limit, len(projectiles))) * 9))
        return values

    def start_recording(self, metadata: Mapping[str, Any] | None = None) -> None:
        details = {
            "training_scenario": "peach-vs-level20-mogadishu-v1",
            "episode_seed": self._episode_seed,
            "spawns_swapped": self._spawns_swapped,
        }
        details.update(dict(metadata or {}))
        self.simulation.start_recording(details)

    def stop_recording(self) -> dict[str, Any]:
        return self.simulation.stop_recording()

    def close(self) -> None:
        # pygame is process-global; closing one vector/evaluation environment
        # must not invalidate another environment in the same process.
        return None


def encode_runtime_observation(
    runtime: Any,
    agent: Any,
    opponent: Any,
    *,
    episode_ticks: int,
    max_ticks: int = 7200,
    spawns_swapped: bool,
) -> np.ndarray:
    """Encode a live match from either fighter's trained-model perspective.

    This deliberately reuses the environment's exact encoder, so deploying a
    model as P2 cannot silently change the 142-value observation contract it
    saw while training as P1.
    """

    encoder = object.__new__(PeachVsLevel20Env)
    encoder.runtime = runtime
    return encoder._observation(
        agent=agent,
        opponent=opponent,
        episode_ticks=episode_ticks,
        max_ticks=max_ticks,
        spawns_swapped=spawns_swapped,
    )
