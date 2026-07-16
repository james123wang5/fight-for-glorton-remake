from __future__ import annotations

import copy
import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.runtime import Stage
from src.simulation import BattleSimulation

from .human_input import HumanInputAdapter
from .peach_env import MATCH_CONFIG, encode_runtime_observation


OBSERVATION_VERSION = "glorton-peach-league-v2"
BASE_OBSERVATION_SIZE = 142
LEAGUE_OBSERVATION_SIZE = 180
POLICY_HZ = 20
PHYSICS_HZ = 40


def _edge_danger(runtime: Any, fighter: Any) -> float:
    bounds = runtime.stage.bounds
    x = abs((float(fighter.pos.x) - bounds.centerx) / max(1.0, bounds.w / 2))
    y = abs((float(fighter.pos.y) - bounds.centery) / max(1.0, bounds.h / 2))
    return float(max(x, y))


def _platform_distance(runtime: Any, fighter: Any) -> float:
    bounds = runtime.stage.bounds
    diagonal = max(1.0, math.hypot(bounds.w, bounds.h))
    if not runtime.stage.platforms:
        return 1.0
    distances = []
    for platform in runtime.stage.platforms:
        gap_x = max(
            float(platform.rect.left) - float(fighter.pos.x),
            0.0,
            float(fighter.pos.x) - float(platform.rect.right),
        )
        gap_y = float(platform.rect.top) - float(fighter.pos.y)
        distances.append(math.hypot(gap_x, gap_y))
    return min(distances) / diagonal


def _is_offstage(runtime: Any, fighter: Any) -> bool:
    return bool(
        not fighter.on_ground
        and (
            _edge_danger(runtime, fighter) > 0.58
            or _platform_distance(runtime, fighter) > 0.09
            or fighter.out_of_camera
        )
    )


def encode_league_observation(
    runtime: Any,
    agent: Any,
    opponent: Any,
    input_adapter: HumanInputAdapter,
    *,
    episode_ticks: int,
    max_ticks: int = 7200,
    spawns_swapped: bool,
) -> np.ndarray:
    """The v2 policy observation: v1 combat state plus input/items/opportunities."""

    base = encode_runtime_observation(
        runtime,
        agent,
        opponent,
        episode_ticks=episode_ticks,
        max_ticks=max_ticks,
        spawns_swapped=spawns_swapped,
    )
    bounds = runtime.stage.bounds
    diagonal = max(1.0, math.hypot(bounds.w, bounds.h))
    extras: list[float] = input_adapter.features()

    items = [item for item in runtime.items if item.alive and item.state != 3]
    items.sort(key=lambda item: agent.pos.distance_to(item.pos))
    for item in items[:2]:
        kind = str(item.kind).lower()
        extras.extend(
            [
                (item.pos.x - agent.pos.x) / max(1.0, bounds.w),
                (item.pos.y - agent.pos.y) / max(1.0, bounds.h),
                float(kind == "mine"),
                float(kind == "grenade"),
                float(item.state) / 3.0,
                float(item.age_ms) / max(1.0, float(item.life_ms)),
                1.0,
            ]
        )
    extras.extend([0.0] * ((2 - min(2, len(items))) * 7))

    held = str(agent.current_item).lower()
    extras.extend([float(not held), float(held == "mine"), float(held == "grenade")])

    dx = float(opponent.pos.x - agent.pos.x)
    dy = float(opponent.pos.y - agent.pos.y)
    distance = math.hypot(dx, dy)
    target_is_behind = agent.facing * (agent.pos.x - opponent.pos.x) > 0
    extras.extend(
        [
            dx / max(1.0, bounds.w),
            dy / max(1.0, bounds.h),
            distance / diagonal,
            float(distance <= 24 and abs(dy) <= 30 and target_is_behind),
            float(dy < -18 and abs(dx) < 100),
            float(distance > 100 and abs(dy) < 70),
            float(_is_offstage(runtime, agent)),
            float(bool(agent.spec_up_ok)),
        ]
    )
    observation = np.concatenate((base, np.asarray(extras, dtype=np.float32)))
    if observation.shape != (LEAGUE_OBSERVATION_SIZE,):
        raise RuntimeError(f"v2 observation contract changed: {observation.shape}")
    return np.clip(observation, -5.0, 5.0).astype(np.float32, copy=False)


@dataclass
class AttackTrial:
    label: str
    started_tick: int
    context: dict[str, bool]
    hit: bool = False


class PeachLeagueEnv(gym.Env[np.ndarray, np.ndarray]):
    """Humanized Peach self-play environment used by both level 21 and 22.

    The opponent is selected per episode from the other live policy, the
    frozen original level-21 teacher, and small idle/retreat probes. The probe
    policies are not teachers; they exist to prove the learner can pursue an
    opponent that does not obligingly run toward it.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: int = 0,
        max_episode_seconds: float = 180.0,
        frame_skip: int = 2,
        items_probability: float = 0.30,
        reaction_delay_decisions: int = 2,
        recovery_start_probability: float = 0.12,
    ) -> None:
        super().__init__()
        if frame_skip != 2:
            raise ValueError("the human input adapter currently requires frame_skip=2")
        self.base_seed = int(seed)
        self.next_seed = int(seed)
        self.frame_skip = frame_skip
        self.max_episode_seconds = float(max_episode_seconds)
        self.max_ticks = max(1, math.ceil(max_episode_seconds * PHYSICS_HZ))
        self.items_probability = float(np.clip(items_probability, 0.0, 1.0))
        self.reaction_delay_decisions = max(0, int(reaction_delay_decisions))
        self.recovery_start_probability = float(
            np.clip(recovery_start_probability, 0.0, 1.0)
        )

        config = copy.deepcopy(MATCH_CONFIG)
        for player in config["players"]:
            player["computer"] = False
            player["level"] = 1
        self.simulation = BattleSimulation.headless(seed=self.base_seed, match_config=config)
        self.runtime = self.simulation.runtime
        self.runtime.audio = None
        self.match_config = config

        self.action_space = spaces.MultiDiscrete(np.asarray([3, 3, 4], dtype=np.int64))
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(LEAGUE_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self.adapters = [HumanInputAdapter(), HumanInputAdapter()]
        self.opponent_pool: list[tuple[Any, float, str]] = []
        self.opponent_policy: Any | None = None
        self.opponent_name = "neutral"
        self.opponent_deterministic = False
        self.agent_slot = 0
        self.opponent_slot = 1
        self._episode_seed = self.base_seed
        self._episode_ticks = 0
        self._decision_steps = 0
        self._episode_reward = 0.0
        self._items_enabled = False
        self._spawns_swapped_by_slot = [False, True]
        self._observation_buffers: list[deque[np.ndarray]] = [deque(), deque()]
        self._reward_totals = self._empty_reward_components()
        self._attack_counts: Counter[str] = Counter()
        self._successful_attacks: Counter[str] = Counter()
        self._active_melee: AttackTrial | None = None
        self._recent_specials: deque[AttackTrial] = deque(maxlen=8)
        self._last_successful_attack = ""
        self._far_stall_steps = 0
        self._was_recovering = False
        self._previous_attack = ""
        self._previous_rocket_ids: set[int] = set()
        self._reset_runtime(self.base_seed, swap_physical=False, items_enabled=False)

    @property
    def agent(self) -> Any:
        return self.runtime.fighters[self.agent_slot]

    @property
    def opponent(self) -> Any:
        return self.runtime.fighters[self.opponent_slot]

    def set_opponent_pool(
        self,
        *,
        primary: Any | None,
        teacher: Any | None = None,
        primary_weight: float = 0.75,
        teacher_weight: float = 0.15,
        probe_weight: float = 0.10,
    ) -> None:
        pool: list[tuple[Any, float, str]] = []
        if primary is not None and primary_weight > 0:
            pool.append((primary, float(primary_weight), "current_peer"))
        if teacher is not None and teacher_weight > 0:
            pool.append((teacher, float(teacher_weight), "frozen_level21"))
        if probe_weight > 0:
            pool.extend(
                [
                    ("idle", float(probe_weight) * 0.4, "idle_probe"),
                    ("retreat", float(probe_weight) * 0.6, "retreat_probe"),
                ]
            )
        self.opponent_pool = pool

    def _apply_training_overrides(self, *, items_enabled: bool) -> None:
        self.runtime.match_config = copy.deepcopy(self.match_config)
        self.runtime.manifest["match"]["limit_mode"] = "stock"
        self.runtime.manifest["match"]["starting_lives"] = 3
        self.runtime.manifest["items"]["frequency"] = 5 if items_enabled else 0
        if self.runtime.stage.name != "Mogadishu":
            self.runtime.stage = Stage(self.runtime.manifest, "Mogadishu")

    def _reset_runtime(
        self,
        seed: int,
        *,
        swap_physical: bool,
        items_enabled: bool,
    ) -> None:
        self._apply_training_overrides(items_enabled=items_enabled)
        self.simulation.reset(seed)
        self.runtime.items.clear()
        self.runtime.item_gen_timer_ms = 0
        names = ("SpawnP2", "SpawnP1") if swap_physical else ("SpawnP1", "SpawnP2")
        for fighter, name in zip(self.runtime.fighters, names, strict=True):
            spawn = self.runtime.stage.spawn_point(name)
            fighter.spawn_pos.update(spawn)
            fighter.start_intro_spawn()
        self.runtime.ready_set = -1
        self.runtime._apply_ready_step()
        self.runtime.stage_time_ms = 0
        self.runtime.game_time_seconds = 0
        self.runtime.fight_timer_accumulator_ms = 0
        self.simulation.tick_index = 0
        self._spawns_swapped_by_slot = [swap_physical, not swap_physical]

    def _select_opponent(self) -> None:
        if not self.opponent_pool:
            self.opponent_policy, self.opponent_name = None, "neutral"
            return
        weights = np.asarray([max(0.0, entry[1]) for entry in self.opponent_pool])
        if weights.sum() <= 0:
            self.opponent_policy, _, self.opponent_name = self.opponent_pool[0]
            return
        index = int(self.np_random.choice(len(self.opponent_pool), p=weights / weights.sum()))
        self.opponent_policy, _, self.opponent_name = self.opponent_pool[index]

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
        self._episode_seed = int(seed)
        self.agent_slot = int(options.get("agent_slot", int(seed) % 2))
        self.opponent_slot = 1 - self.agent_slot
        swap_physical = bool(options.get("swap_spawns", (int(seed) // 2) % 2))
        self._items_enabled = bool(
            options.get("items_enabled", self.np_random.random() < self.items_probability)
        )
        self._reset_runtime(
            self._episode_seed,
            swap_physical=swap_physical,
            items_enabled=self._items_enabled,
        )
        self._select_opponent()
        for adapter in self.adapters:
            adapter.reset()
        self._episode_ticks = 0
        self._decision_steps = 0
        self._episode_reward = 0.0
        self._reward_totals = self._empty_reward_components()
        self._attack_counts.clear()
        self._successful_attacks.clear()
        self._active_melee = None
        self._recent_specials.clear()
        self._last_successful_attack = ""
        self._far_stall_steps = 0
        self._was_recovering = False
        self._previous_attack = ""
        self._previous_rocket_ids = set()
        if self.np_random.random() < self.recovery_start_probability:
            self._place_recovery_start()

        self._observation_buffers = [deque(), deque()]
        for slot in range(2):
            current = self._current_observation(slot)
            self._observation_buffers[slot] = deque(
                [current.copy() for _ in range(self.reaction_delay_decisions + 1)],
                maxlen=self.reaction_delay_decisions + 1,
            )
        self.action_space.seed(self._episode_seed)
        return self._delayed_observation(self.agent_slot), self._info("ongoing")

    def _place_recovery_start(self) -> None:
        if not self.runtime.stage.platforms:
            return
        platform = max(self.runtime.stage.platforms, key=lambda item: item.rect.w)
        side = -1 if self.np_random.random() < 0.5 else 1
        fighter = self.agent
        fighter.pos.x = platform.rect.left - 65 if side < 0 else platform.rect.right + 65
        fighter.pos.y = platform.rect.top - 55
        fighter.prev_pos.update(fighter.pos)
        fighter.xinc = 2.5 * side
        fighter.yinc = 2.0
        fighter.on_ground = False
        fighter.ground_platform = None
        fighter.jumpstate = 1
        fighter.spec_up_ok = True
        fighter.state = "stop"

    def step(
        self,
        action: np.ndarray | Sequence[int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        learner_action = self._validate_action(action)
        opponent_action = self._opponent_action()
        decisions = [np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64)]
        decisions[self.agent_slot] = learner_action
        decisions[self.opponent_slot] = opponent_action
        control_pairs = [
            self.adapters[slot].begin_decision(decisions[slot]) for slot in range(2)
        ]
        before = self._reward_state()
        event_reward = 0.0
        for internal_tick in range(self.frame_skip):
            tick_before = self._event_state()
            controls = [control_pairs[slot][internal_tick] for slot in range(2)]
            self.simulation.step_fast(controls)
            self._episode_ticks += 1
            event_reward += self._track_attack_events(tick_before)
            if self.runtime.match_state == "game_set" or self._episode_ticks >= self.max_ticks:
                break

        self._decision_steps += 1
        for slot in range(2):
            self._observation_buffers[slot].append(self._current_observation(slot))
        terminated = self.runtime.match_state == "game_set"
        truncated = self._episode_ticks >= self.max_ticks and not terminated
        outcome = self._outcome(terminated=terminated, truncated=truncated)
        reward, components = self._reward(before, outcome=outcome, terminated=terminated)
        components["skill_timing"] += event_reward
        reward += event_reward
        self._episode_reward += reward
        for key, value in components.items():
            self._reward_totals[key] += value
        return (
            self._delayed_observation(self.agent_slot),
            float(reward),
            terminated,
            truncated,
            self._info(outcome),
        )

    def _validate_action(self, action: np.ndarray | Sequence[int]) -> np.ndarray:
        value = np.asarray(action, dtype=np.int64).reshape(-1)
        if value.shape != (3,) or not self.action_space.contains(value):
            raise ValueError(f"invalid action {action!r}; expected MultiDiscrete([3, 3, 4])")
        return value

    def _opponent_action(self) -> np.ndarray:
        policy = self.opponent_policy
        if policy is None or policy == "idle":
            return np.zeros(3, dtype=np.int64)
        if policy == "retreat":
            return self._retreat_probe_action()
        observation = self._delayed_observation(self.opponent_slot)
        expected = tuple(getattr(getattr(policy, "observation_space", None), "shape", ()) or ())
        if expected == (BASE_OBSERVATION_SIZE,):
            observation = observation[:BASE_OBSERVATION_SIZE]
        action, _state = policy.predict(
            observation,
            deterministic=self.opponent_deterministic,
        )
        return self._validate_action(action)

    def _retreat_probe_action(self) -> np.ndarray:
        fighter = self.opponent
        target = self.agent
        if _is_offstage(self.runtime, fighter):
            horizontal = 2 if fighter.pos.x < self.runtime.stage.bounds.centerx else 1
            vertical = 1
            combat = 2 if fighter.spec_up_ok and self._decision_steps % 5 == 0 else 0
            return np.asarray([horizontal, vertical, combat], dtype=np.int64)
        horizontal = 1 if target.pos.x > fighter.pos.x else 2
        # Periodic release makes every key press physically possible while the
        # probe continues to retreat instead of teaching an attack pattern.
        vertical = 1 if self._decision_steps % 37 == 0 else 0
        return np.asarray([horizontal, vertical, 0], dtype=np.int64)

    def _current_observation(self, slot: int) -> np.ndarray:
        other = 1 - slot
        return encode_league_observation(
            self.runtime,
            self.runtime.fighters[slot],
            self.runtime.fighters[other],
            self.adapters[slot],
            episode_ticks=min(self._episode_ticks, self.max_ticks),
            max_ticks=self.max_ticks,
            spawns_swapped=self._spawns_swapped_by_slot[slot],
        )

    def _delayed_observation(self, slot: int) -> np.ndarray:
        return self._observation_buffers[slot][0].copy()

    def _reward_state(self) -> dict[str, float | int | bool]:
        return {
            "agent_lives": int(self.agent.lives),
            "opponent_lives": int(self.opponent.lives),
            "agent_damage": float(self.agent.damage_amnt),
            "opponent_damage": float(self.opponent.damage_amnt),
            "agent_danger": _edge_danger(self.runtime, self.agent),
            "opponent_danger": _edge_danger(self.runtime, self.opponent),
            "distance": float(self.agent.pos.distance_to(self.opponent.pos)),
            "recovery_distance": _platform_distance(self.runtime, self.agent),
            "recovering": _is_offstage(self.runtime, self.agent),
        }

    def _event_state(self) -> dict[str, float | int]:
        return {
            "opponent_damage": float(self.opponent.damage_amnt),
            "opponent_lives": int(self.opponent.lives),
            "agent_kos": int(self.agent.kos),
        }

    def _attack_context(self) -> dict[str, bool]:
        dx = self.opponent.pos.x - self.agent.pos.x
        dy = self.opponent.pos.y - self.agent.pos.y
        distance = math.hypot(dx, dy)
        behind = self.agent.facing * (self.agent.pos.x - self.opponent.pos.x) > 0
        return {
            "behind": bool(distance <= 24 and abs(dy) <= 30 and behind),
            "above": bool(dy < -18 and abs(dx) < 100),
            "ranged": bool(distance > 100 and abs(dy) < 70),
            "airborne": bool(not self.agent.on_ground),
            "recovering": _is_offstage(self.runtime, self.agent),
        }

    def _track_attack_events(self, before: Mapping[str, float | int]) -> float:
        reward = 0.0
        current = str(self.agent.current_attack)
        if current and current != self._previous_attack:
            self._attack_counts[current] += 1
            trial = AttackTrial(current, self._episode_ticks, self._attack_context())
            if current.startswith("special"):
                self._recent_specials.append(trial)
            else:
                self._active_melee = trial
            if current == "specialBackThrow":
                # The grab replaces the punch that opened it. Attribute the
                # eventual 15-damage throw to the actual combo, not to a
                # phantom successful punch or a melee-whiff penalty.
                self._active_melee = None
                reward += 0.08
                self._attack_counts["timely_back_throw"] += 1

        damage_landed = (
            float(self.opponent.damage_amnt) > float(before["opponent_damage"])
            or int(self.opponent.lives) < int(before["opponent_lives"])
            or int(self.agent.kos) > int(before["agent_kos"])
        )
        attributed = self.opponent.last_sender is self.agent or int(self.agent.kos) > int(
            before["agent_kos"]
        )
        if damage_landed and attributed:
            trial = self._active_melee
            if trial is None or trial.hit:
                trial = next((item for item in reversed(self._recent_specials) if not item.hit), None)
            if trial is not None and not trial.hit:
                trial.hit = True
                self._successful_attacks[trial.label] += 1
                if self._last_successful_attack and self._last_successful_attack != trial.label:
                    reward += 0.015
                self._last_successful_attack = trial.label
                if trial.label == "punchUp" and trial.context["above"]:
                    reward += 0.02
                elif trial.label == "punchAir" and trial.context["airborne"]:
                    reward += 0.01
                elif trial.label in {"specialGround", "specialAir"} and trial.context["ranged"]:
                    reward += 0.015
                elif trial.label == "specialUp" and (
                    trial.context["recovering"] or trial.context["above"]
                ):
                    reward += 0.03
                elif trial.label == "specialBackThrow" and trial.context["behind"]:
                    reward += 0.03

        if self._active_melee is not None and self._previous_attack and not current:
            if not self._active_melee.hit:
                reward -= 0.006
                self._attack_counts["melee_whiff"] += 1
            self._active_melee = None
        while self._recent_specials and (
            self._episode_ticks - self._recent_specials[0].started_tick > 120
        ):
            self._recent_specials.popleft()
        new_rockets = [
            rocket
            for rocket in self.runtime.rockets
            if rocket.sender is self.agent and id(rocket) not in self._previous_rocket_ids
        ]
        if new_rockets:
            self._attack_counts["rockets_spawned"] += len(new_rockets)
        self._previous_rocket_ids = {id(rocket) for rocket in self.runtime.rockets}
        self._previous_attack = current
        return reward

    @staticmethod
    def _empty_reward_components() -> dict[str, float]:
        return {
            "ringout": 0.0,
            "result": 0.0,
            "edge_progress": 0.0,
            "damage": 0.0,
            "pursuit": 0.0,
            "recovery": 0.0,
            "skill_timing": 0.0,
        }

    def _reward(
        self,
        before: Mapping[str, float | int | bool],
        *,
        outcome: str,
        terminated: bool,
    ) -> tuple[float, dict[str, float]]:
        components = self._empty_reward_components()
        agent_lost = max(0, int(before["agent_lives"]) - int(self.agent.lives))
        opponent_lost = max(0, int(before["opponent_lives"]) - int(self.opponent.lives))
        components["ringout"] = 2.0 * (opponent_lost - agent_lost)
        if agent_lost == 0 and opponent_lost == 0:
            dealt = max(0.0, float(self.opponent.damage_amnt) - float(before["opponent_damage"]))
            taken = max(0.0, float(self.agent.damage_amnt) - float(before["agent_damage"]))
            components["damage"] = 0.0002 * (dealt - taken)
            edge_delta = (
                _edge_danger(self.runtime, self.opponent)
                - float(before["opponent_danger"])
                - _edge_danger(self.runtime, self.agent)
                + float(before["agent_danger"])
            )
            components["edge_progress"] = 0.10 * edge_delta

        distance_before = float(before["distance"])
        distance_after = float(self.agent.pos.distance_to(self.opponent.pos))
        if distance_before > 180 and not bool(before["recovering"]):
            closing = max(-20.0, min(20.0, distance_before - distance_after))
            components["pursuit"] = 0.0025 * closing
            if closing < 1.0:
                self._far_stall_steps += 1
                if self._far_stall_steps >= 20:
                    components["pursuit"] -= 0.002
            else:
                self._far_stall_steps = 0
        else:
            self._far_stall_steps = 0

        recovering_before = bool(before["recovering"])
        recovering_after = _is_offstage(self.runtime, self.agent)
        if recovering_before and agent_lost == 0:
            improvement = float(before["recovery_distance"]) - _platform_distance(
                self.runtime, self.agent
            )
            components["recovery"] += 0.20 * improvement
            if not recovering_after and self.agent.on_ground:
                components["recovery"] += 0.10
        self._was_recovering = recovering_after

        if terminated:
            if outcome == "win":
                components["result"] = 1.0
            elif outcome == "loss":
                components["result"] = -1.0
        return sum(components.values()), components

    def _outcome(self, *, terminated: bool, truncated: bool) -> str:
        if terminated:
            if self.runtime.match_winner is self.agent:
                return "win"
            if self.runtime.match_winner is self.opponent:
                return "loss"
            return "draw"
        if not truncated:
            return "ongoing"
        agent_score = (int(self.agent.lives), -float(self.agent.damage_amnt), int(self.agent.kos))
        opponent_score = (
            int(self.opponent.lives),
            -float(self.opponent.damage_amnt),
            int(self.opponent.kos),
        )
        if agent_score > opponent_score:
            return "timeout_win"
        if agent_score < opponent_score:
            return "timeout_loss"
        return "timeout_draw"

    def _info(self, outcome: str) -> dict[str, Any]:
        return {
            "outcome": outcome,
            "seed": self._episode_seed,
            "stage": self.runtime.stage.name,
            "agent_slot": self.agent_slot + 1,
            "opponent_source": self.opponent_name,
            "items_enabled": self._items_enabled,
            "elapsed_ticks": self._episode_ticks,
            "decision_steps": self._decision_steps,
            "episode_reward": self._episode_reward,
            "reward_components": dict(self._reward_totals),
            "attack_starts": dict(self._attack_counts),
            "successful_attacks": dict(self._successful_attacks),
            "agent_lives": int(self.agent.lives),
            "opponent_lives": int(self.opponent.lives),
            "agent_damage": float(self.agent.damage_amnt),
            "opponent_damage": float(self.opponent.damage_amnt),
            "agent_kos": int(self.agent.kos),
            "opponent_kos": int(self.opponent.kos),
        }

    def close(self) -> None:
        return None
