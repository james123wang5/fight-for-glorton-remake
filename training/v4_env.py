from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
from gymnasium import spaces

from .league_env import (
    BASE_OBSERVATION_SIZE,
    LEAGUE_OBSERVATION_SIZE,
    _edge_danger,
    _is_offstage,
    _platform_distance,
)
from .tactical_env import (
    TACTICAL_OBSERVATION_SIZE,
    ProjectileTrial,
    TacticalPeachEnv,
    _projectile_kind,
    encode_tactical_observation,
    tactical_context,
)
from .tactical_input import (
    COMBAT_LABELS,
    MOVEMENT_LABELS,
    TACTICAL_ACTION_NVECS,
    TacticalInputAdapter,
)


V4_OBSERVATION_VERSION = "glorton-peach-active-v4"
V4_OBSERVATION_SIZE = 252
V4_POLICY_HZ = 10
V4_FRAME_SKIP = 4


def wall_probe(runtime: Any, fighter: Any, direction: int, *, limit: float = 120.0) -> dict[str, Any]:
    """Describe the nearest source collision block in one horizontal direction."""

    direction = 1 if direction >= 0 else -1
    middle_y = float(fighter.pos.y) - float(fighter.body_height) * 0.5
    entries: list[tuple[float, Any]] = []
    for platform in runtime.stage.platforms:
        if not platform.rect.top <= middle_y < platform.rect.bottom:
            continue
        stop_x = (
            float(platform.rect.left - fighter.body_half_width)
            if direction > 0
            else float(platform.rect.right + fighter.body_half_width)
        )
        distance = (stop_x - float(fighter.pos.x)) * direction
        if -1.0 <= distance <= limit:
            entries.append((max(0.0, distance), platform))
    if not entries:
        return {
            "blocked": False,
            "distance": limit,
            "top_delta": 0.0,
            "height": 0.0,
            "moving": False,
            "platform": None,
        }
    distance, platform = min(entries, key=lambda item: item[0])
    return {
        "blocked": True,
        "distance": distance,
        "top_delta": float(platform.rect.top) - float(fighter.pos.y),
        "height": float(platform.rect.h),
        "moving": bool(platform.moving),
        "platform": platform,
    }


def route_distance(runtime: Any, fighter: Any, opponent: Any) -> float:
    dx = float(opponent.pos.x - fighter.pos.x)
    dy = float(opponent.pos.y - fighter.pos.y)
    direction = 1 if dx >= 0 else -1
    probe = wall_probe(runtime, fighter, direction)
    distance = abs(dx) + 0.75 * abs(dy)
    if probe["blocked"] and float(probe["distance"]) < min(120.0, abs(dx)):
        distance += 60.0 + min(100.0, abs(float(probe["top_delta"])))
    return distance


def _navigation_features(
    runtime: Any,
    fighter: Any,
    opponent: Any,
    adapter: TacticalInputAdapter,
    *,
    wall_stall_steps: int,
) -> list[float]:
    bounds = runtime.stage.bounds
    left = wall_probe(runtime, fighter, -1)
    right = wall_probe(runtime, fighter, 1)
    dx = float(opponent.pos.x - fighter.pos.x)
    dy = float(opponent.pos.y - fighter.pos.y)
    distance = math.hypot(dx, dy)
    diagonal = max(1.0, math.hypot(bounds.w, bounds.h))
    same_surface = bool(
        fighter.ground_platform is not None
        and fighter.ground_platform is opponent.ground_platform
    )
    values = [*adapter.v4_features()]
    for probe in (left, right):
        values.extend(
            [
                float(bool(probe["blocked"])),
                float(probe["distance"]) / 120.0,
                float(probe["top_delta"]) / max(1.0, bounds.h),
                float(probe["height"]) / max(1.0, bounds.h),
                float(bool(probe["moving"])),
            ]
        )
    values.extend(
        [
            float(same_surface),
            float(dy < -30.0),
            float(dy > 30.0),
            float(not bool(tactical_context(runtime, fighter, opponent)["clear_shot"])),
            route_distance(runtime, fighter, opponent) / diagonal,
            min(1.0, wall_stall_steps / 10.0),
            float(distance <= 75.0),
            float(75.0 < distance <= 180.0),
        ]
    )
    return values


def v4_combat_mask(runtime: Any, fighter: Any, opponent: Any, adapter: TacticalInputAdapter) -> list[bool]:
    context = tactical_context(runtime, fighter, opponent)
    ready = bool(
        not fighter.dead
        and fighter.has_control
        and fighter.ctrl_loss <= 0
        and fighter.state not in {"ko", "thrown", "spawn", "dead"}
        and not fighter.current_attack
    )
    if fighter.shielded:
        return [True, False, False, False, False, False, False, False, True]
    if not ready:
        return [True] + [False] * (len(COMBAT_LABELS) - 1)

    distance = float(context["distance"])
    dx = float(context["dx"])
    dy = float(context["dy"])
    moving_drop = bool(
        fighter.on_ground
        and fighter.ground_platform is not None
        and fighter.ground_platform.moving
        and dy > 30.0
    )
    airborne_drop = bool(not fighter.on_ground and fighter.yinc < fighter.max_fall)
    imminent_threat = float(context["threat_score"]) >= 0.20
    clear_aligned_shot = bool(
        75.0 <= distance <= 360.0
        and context["clear_shot"]
        and context["target_in_front"]
        and abs(float(context["lead_y"])) <= 32.0
        and adapter.shoot_rearm <= 0
    )
    return [
        True,
        bool(fighter.jumpstate < 2),
        moving_drop or airborne_drop,
        bool(distance <= 78.0 and abs(dy) <= 58.0 and context["target_in_front"]),
        bool(fighter.on_ground and dy < -8.0 and abs(dx) <= 95.0),
        bool(
            fighter.on_ground
            and distance <= 30.0
            and abs(dy) <= 35.0
            and context["target_behind"]
        ),
        clear_aligned_shot,
        bool(fighter.spec_up_ok and (context["offstage"] or context["rocket_opportunity"])),
        imminent_threat,
    ]


def v4_action_mask(
    runtime: Any,
    fighter: Any,
    opponent: Any,
    adapter: TacticalInputAdapter,
    *,
    curriculum: str = "duel",
    wall_stall_steps: int = 0,
    enforce_pursuit: bool = True,
) -> np.ndarray:
    movement = adapter.action_mask_prefix()
    context = tactical_context(runtime, fighter, opponent)
    direction = 1 if float(context["dx"]) >= 0 else -1
    obstacle = wall_probe(runtime, fighter, direction)
    should_chase = bool(
        enforce_pursuit
        and not context["offstage"]
        and float(context["threat_score"]) < 0.20
        and (float(context["distance"]) > 130.0 or obstacle["blocked"])
    )
    if should_chase and adapter.movement_lock <= 0:
        movement[0] = False
        movement[1] = True
        movement[2] = False

    combat = v4_combat_mask(runtime, fighter, opponent, adapter)
    if adapter.combat_cooldown > 0:
        combat = [True] + [False] * (len(COMBAT_LABELS) - 1)
    elif adapter.previous_combat in {1, 3, 4, 5, 6, 7} and not fighter.current_attack:
        combat = [True] + [False] * (len(COMBAT_LABELS) - 1)
    if adapter.shield_rearm > 0:
        combat[8] = False
    if adapter.shoot_rearm > 0:
        combat[6] = False
    if (
        (bool(obstacle["blocked"]) and float(obstacle["distance"]) <= 45.0)
        or wall_stall_steps >= 5
    ) and fighter.jumpstate < 2:
        combat[1] = True

    allowed_by_lesson: dict[str, set[int]] = {
        "aim_static": {0, 6},
        "aim_moving": {0, 6},
        "rocket": {0, 1, 7},
        "defense": {0, 1, 8},
        "melee": {0, 1, 3, 4, 5},
        "recovery": {0, 1, 7},
        "pursuit": {0, 1, 3, 4, 6},
        "navigation": {0, 1, 3, 4},
    }
    allowed = allowed_by_lesson.get(curriculum)
    if allowed is not None:
        combat = [value and index in allowed for index, value in enumerate(combat)]
        combat[0] = True
    return np.asarray([*movement, *combat], dtype=bool)


def encode_v4_observation(
    runtime: Any,
    fighter: Any,
    opponent: Any,
    adapter: TacticalInputAdapter,
    *,
    episode_ticks: int,
    max_ticks: int,
    spawns_swapped: bool,
    curriculum: str = "duel",
    wall_stall_steps: int = 0,
) -> np.ndarray:
    base = encode_tactical_observation(
        runtime,
        fighter,
        opponent,
        adapter,
        episode_ticks=episode_ticks,
        max_ticks=max_ticks,
        spawns_swapped=spawns_swapped,
        curriculum=curriculum,
    )
    extras = _navigation_features(
        runtime,
        fighter,
        opponent,
        adapter,
        wall_stall_steps=wall_stall_steps,
    )
    observation = np.concatenate((base, np.asarray(extras, dtype=np.float32)))
    if observation.shape != (V4_OBSERVATION_SIZE,):
        raise RuntimeError(f"v4 observation contract changed: {observation.shape}")
    return np.clip(observation, -5.0, 5.0).astype(np.float32, copy=False)


class V4PeachEnv(TacticalPeachEnv):
    """Active-combat v4: navigation, contextual defense and anti-timeout rewards."""

    def __init__(
        self,
        *,
        seed: int = 0,
        max_episode_seconds: float = 120.0,
        items_probability: float = 0.0,
        curriculum_strength: float = 0.60,
        lesson_seconds: float = 20.0,
    ) -> None:
        self._wall_stall_steps = [0, 0]
        self._ground_crouch_decisions = 0
        self._shield_hold_decisions = 0
        self._far_idle_decisions = 0
        self._wall_stall_decisions = 0
        self._melee_opportunities = 0
        self._melee_opportunity_uses = 0
        self._shield_blocked_this_activation = False
        super().__init__(
            seed=seed,
            max_episode_seconds=max_episode_seconds,
            items_probability=items_probability,
            curriculum_strength=curriculum_strength,
            lesson_seconds=lesson_seconds,
        )
        self.adapters = [self._new_adapter(), self._new_adapter()]
        self.action_space = spaces.MultiDiscrete(TACTICAL_ACTION_NVECS.copy())
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(V4_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    @staticmethod
    def _new_adapter() -> TacticalInputAdapter:
        return TacticalInputAdapter(
            frame_skip=V4_FRAME_SKIP,
            movement_commitment_decisions=2,
            combat_cooldown_decisions=2,
            shield_min_hold_decisions=1,
            shield_rearm_decisions=7,
            shield_max_hold_decisions=5,
            shoot_rearm_decisions=8,
        )

    def _choose_curriculum(self, options: Mapping[str, Any]) -> str:
        explicit = options.get("curriculum")
        if explicit is not None:
            return str(explicit)
        if self.np_random.random() >= self.curriculum_strength:
            return "duel"
        lessons = (
            "aim_static",
            "aim_moving",
            "rocket",
            "defense",
            "melee",
            "recovery",
            "pursuit",
            "navigation",
        )
        probabilities = np.asarray([0.12, 0.10, 0.06, 0.04, 0.24, 0.12, 0.18, 0.14])
        return str(self.np_random.choice(lessons, p=probabilities / probabilities.sum()))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._wall_stall_steps = [0, 0]
        self._ground_crouch_decisions = 0
        self._shield_hold_decisions = 0
        self._far_idle_decisions = 0
        self._wall_stall_decisions = 0
        self._melee_opportunities = 0
        self._melee_opportunity_uses = 0
        self._shield_blocked_this_activation = False
        observation, info = super().reset(seed=seed, options=options)
        return self._delayed_observation(self.agent_slot), self._info(str(info["outcome"]))

    def _setup_lesson(self) -> None:
        if self.curriculum in {"navigation", "pursuit"}:
            by_name = {platform.name: platform for platform in self.runtime.stage.platforms}
            base = by_name.get("Fixed1")
            wall = by_name.get("Fixed12" if self.np_random.random() < 0.5 else "Fixed13")
            if base is not None and wall is not None:
                self._place_fighter(self.agent, base, wall.rect.left - 70)
                self._place_fighter(self.opponent, base, wall.rect.right + 90)
                self.agent.facing = 1
                self.opponent.facing = -1
                return
        super()._setup_lesson()

    def _script_action(self) -> np.ndarray:
        if self.curriculum in {"navigation", "pursuit"}:
            phase = (self._decision_steps // 24) % 2
            return np.asarray([2 if phase == 0 else 0, 0], dtype=np.int64)
        return super()._script_action()

    def _action_mask_for_slot(self, slot: int) -> np.ndarray:
        relaxed_probe = bool(
            slot == self.opponent_slot
            and (
                self._episode_script
                or (
                    isinstance(self.opponent_policy, str)
                    and self.opponent_policy in {"idle", "retreat", "active", "melee"}
                )
            )
        )
        return v4_action_mask(
            self.runtime,
            self.runtime.fighters[slot],
            self.runtime.fighters[1 - slot],
            self.adapters[slot],
            curriculum=self.curriculum if slot == self.agent_slot else "duel",
            wall_stall_steps=self._wall_stall_steps[slot],
            enforce_pursuit=not relaxed_probe,
        )

    def _scripted_opponent_action(self, style: str) -> np.ndarray:
        fighter = self.opponent
        target = self.agent
        mask = self._action_mask_for_slot(self.opponent_slot)
        context = tactical_context(self.runtime, fighter, target)
        if style == "idle":
            return np.asarray([0, 0], dtype=np.int64)
        if style == "retreat":
            combat = 1 if self._decision_steps % 31 == 0 and mask[4 + 1] else 0
            return np.asarray([2, combat], dtype=np.int64)

        movement = 1
        combat = 0
        if context["offstage"]:
            desired_right = fighter.pos.x < self.runtime.stage.bounds.centerx
            target_right = target.pos.x >= fighter.pos.x
            movement = 1 if desired_right == target_right else 2
            combat = 7 if mask[4 + 7] else (1 if mask[4 + 1] else 0)
        elif mask[4 + 5]:
            combat = 5
        elif mask[4 + 4] and context["opponent_above"]:
            combat = 4
        elif mask[4 + 3]:
            combat = 3
        else:
            direction = 1 if float(context["dx"]) >= 0 else -1
            obstacle = wall_probe(self.runtime, fighter, direction)
            if (
                (obstacle["blocked"] and float(obstacle["distance"]) <= 45.0)
                or float(context["dy"]) < -35.0
            ) and mask[4 + 1]:
                combat = 1
            elif style == "active" and self._decision_steps % 8 == 0 and mask[4 + 6]:
                combat = 6
        return np.asarray([movement, combat], dtype=np.int64)

    def _opponent_action(self) -> np.ndarray:
        if self._episode_script:
            return self._script_action()
        policy = self.opponent_policy
        if policy is None:
            return np.asarray([0, 0], dtype=np.int64)
        if isinstance(policy, str):
            return self._scripted_opponent_action(policy)

        observation = self._delayed_observation(self.opponent_slot)
        expected = tuple(getattr(getattr(policy, "observation_space", None), "shape", ()) or ())
        if expected in {(BASE_OBSERVATION_SIZE,), (LEAGUE_OBSERVATION_SIZE,)}:
            action, _state = policy.predict(
                observation[: expected[0]],
                deterministic=self.opponent_deterministic,
            )
            return self._legacy_action_to_tactical(action, slot=self.opponent_slot)
        if expected == (TACTICAL_OBSERVATION_SIZE,):
            observation = observation[:TACTICAL_OBSERVATION_SIZE]
        elif expected != (V4_OBSERVATION_SIZE,):
            return np.asarray([0, 0], dtype=np.int64)
        mask = self._action_mask_for_slot(self.opponent_slot)
        try:
            action, _state = policy.predict(
                observation,
                action_masks=mask,
                deterministic=self.opponent_deterministic,
            )
        except TypeError:
            action, _state = policy.predict(observation, deterministic=self.opponent_deterministic)
        return self._validate_action(action)

    def _current_observation(self, slot: int) -> np.ndarray:
        return encode_v4_observation(
            self.runtime,
            self.runtime.fighters[slot],
            self.runtime.fighters[1 - slot],
            self.adapters[slot],
            episode_ticks=min(self._episode_ticks, self.max_ticks),
            max_ticks=self.max_ticks,
            spawns_swapped=self._spawns_swapped_by_slot[slot],
            curriculum=self.curriculum if slot == self.agent_slot else "duel",
            wall_stall_steps=self._wall_stall_steps[slot],
        )

    def _reward_state(self) -> dict[str, float | int | bool]:
        state = super()._reward_state()
        state["route_distance"] = route_distance(self.runtime, self.agent, self.opponent)
        return state

    @staticmethod
    def _empty_reward_components() -> dict[str, float]:
        components = TacticalPeachEnv._empty_reward_components()
        components.update({"timeout": 0.0, "activity": 0.0})
        return components

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
        components["ringout"] = 3.0 * (opponent_lost - agent_lost)
        if agent_lost == 0 and opponent_lost == 0:
            dealt = max(0.0, float(self.opponent.damage_amnt) - float(before["opponent_damage"]))
            taken = max(0.0, float(self.agent.damage_amnt) - float(before["agent_damage"]))
            components["damage"] = 0.0001 * (dealt - taken)
            edge_delta = (
                _edge_danger(self.runtime, self.opponent)
                - float(before["opponent_danger"])
                - _edge_danger(self.runtime, self.agent)
                + float(before["agent_danger"])
            )
            components["edge_progress"] = 0.05 * edge_delta
            if float(before["distance"]) > 110.0 and not bool(before["recovering"]):
                progress = float(before["route_distance"]) - route_distance(
                    self.runtime, self.agent, self.opponent
                )
                components["pursuit"] = 0.0015 * max(-20.0, min(20.0, progress))

        if bool(before["recovering"]) and agent_lost == 0:
            improvement = float(before["recovery_distance"]) - _platform_distance(
                self.runtime, self.agent
            )
            components["recovery"] = 0.20 * improvement
            if not _is_offstage(self.runtime, self.agent) and self.agent.on_ground:
                components["recovery"] += 0.10
        if terminated:
            if outcome == "win":
                components["result"] = 2.0
            elif outcome == "loss":
                components["result"] = -2.0
        elif outcome.startswith("timeout"):
            components["timeout"] = -1.0
        return sum(components.values()), components

    def _track_projectiles(self, before: Mapping[str, float | int]) -> float:
        reward = 0.0
        current = {
            id(projectile): projectile
            for projectile in [*self.runtime.bullets, *self.runtime.rockets]
            if projectile.sender is self.agent and projectile.alive
        }
        for projectile_id, projectile in current.items():
            if projectile_id not in self._projectile_trials:
                kind = _projectile_kind(projectile)
                self._projectile_trials[projectile_id] = ProjectileTrial(
                    projectile=projectile,
                    kind=kind,
                    spawned_tick=self._episode_ticks,
                )
                self._shot_outcomes[f"{kind}_fired"] += 1

        for projectile_id, trial in list(self._projectile_trials.items()):
            if projectile_id in current:
                continue
            projectile = trial.projectile
            contact = bool(projectile.hitbox().colliderect(self.opponent.visual_bounds()))
            damage = float(self.opponent.damage_amnt) > float(before["opponent_damage"])
            ringout = int(self.opponent.lives) < int(before["opponent_lives"])
            shield_drop = float(before["opponent_shield"]) - float(self.opponent.shield_size) > 0.5
            if contact and shield_drop and not damage:
                outcome = "blocked"
            elif contact and (damage or ringout) and self.opponent.last_sender is self.agent:
                outcome = "hit"
                reward += 0.05 if self.curriculum in {"aim_static", "aim_moving"} else 0.02
            else:
                outcome = "miss"
                if self.curriculum in {"aim_static", "aim_moving"}:
                    reward -= 0.03
                else:
                    # Rockets remain available as a real option, but their
                    # large commitment receives a slightly stronger miss cost
                    # than the pistol so the policy does not fire blindly.
                    reward -= 0.025 if trial.kind == "rocket" else 0.02
            self._shot_outcomes[f"{trial.kind}_{outcome}"] += 1
            del self._projectile_trials[projectile_id]
        return reward

    def _track_shield(self, before: Mapping[str, float | int], *, threat_before: float) -> float:
        shield_drop = float(before["agent_shield"]) - float(self.agent.shield_size)
        damage_unchanged = float(self.agent.damage_amnt) <= float(before["agent_damage"])
        if shield_drop > 0.5 and damage_unchanged and not self._shield_blocked_this_activation:
            self._shield_metrics["blocks"] += 1
            self._shield_blocked_this_activation = True
        if self.agent.shielded and threat_before < 0.20:
            self._shield_metrics["unthreatened_hold_steps"] += 1
            return -0.005
        return 0.0

    def step(
        self,
        action: np.ndarray | Sequence[int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        before_positions = [float(fighter.pos.x) for fighter in self.runtime.fighters]
        before_grounded = bool(self.agent.on_ground)
        before_ground_moving = bool(
            self.agent.ground_platform is not None and self.agent.ground_platform.moving
        )
        context = tactical_context(self.runtime, self.agent, self.opponent)
        before_mask = self._action_mask_for_slot(self.agent_slot)
        melee_opportunity = any(bool(before_mask[4 + index]) for index in (3, 4, 5))
        shield_starts_before = self.adapters[self.agent_slot].shield_starts

        observation, reward, terminated, truncated, info = super().step(action)
        accepted = tuple(int(value) for value in self.adapters[self.agent_slot].last_action)
        for slot, fighter in enumerate(self.runtime.fighters):
            moved = abs(float(fighter.pos.x) - before_positions[slot])
            other = self.runtime.fighters[1 - slot]
            direction = 1 if other.pos.x >= fighter.pos.x else -1
            blocked = bool(wall_probe(self.runtime, fighter, direction)["blocked"])
            if self.adapters[slot].movement_intent == 1 and blocked and moved < 1.0:
                self._wall_stall_steps[slot] += 1
            else:
                self._wall_stall_steps[slot] = max(0, self._wall_stall_steps[slot] - 1)

        if self.adapters[self.agent_slot].shield_starts > shield_starts_before:
            self._shield_blocked_this_activation = False
        if not self.agent.shielded:
            self._shield_blocked_this_activation = False
        if self.agent.shielded:
            self._shield_hold_decisions += 1
        if accepted[1] == 2 and before_grounded and not before_ground_moving:
            self._ground_crouch_decisions += 1
        if melee_opportunity:
            self._melee_opportunities += 1
            if accepted[1] in {3, 4, 5}:
                self._melee_opportunity_uses += 1

        activity_penalty = 0.0
        far_idle = bool(
            float(context["distance"]) > 130.0
            and float(context["threat_score"]) < 0.20
            and not bool(context["offstage"])
            and accepted[0] in {0, 2}
        )
        if far_idle:
            self._far_idle_decisions += 1
            activity_penalty -= 0.006
        if self._wall_stall_steps[self.agent_slot] >= 6:
            self._wall_stall_decisions += 1
            if accepted[1] != 1:
                activity_penalty -= 0.004
        if activity_penalty:
            reward += activity_penalty
            self._episode_reward += activity_penalty
            self._reward_totals["activity"] += activity_penalty

        for slot in range(2):
            if self._observation_buffers[slot]:
                self._observation_buffers[slot][-1] = self._current_observation(slot)
        return (
            observation,
            float(reward),
            terminated,
            truncated,
            self._info(str(info["outcome"])),
        )

    def _quality_metrics(self) -> dict[str, float]:
        quality = super()._quality_metrics()
        seconds = max(0.1, self._decision_steps / V4_POLICY_HZ)
        minutes = seconds / 60.0
        quality.update(
            {
                "ground_crouches_per_minute": self._ground_crouch_decisions / minutes,
                "shield_hold_fraction": self._shield_hold_decisions / max(1, self._decision_steps),
                "far_idle_fraction": self._far_idle_decisions / max(1, self._decision_steps),
                "wall_stall_fraction": self._wall_stall_decisions / max(1, self._decision_steps),
                "melee_opportunity_use_rate": self._melee_opportunity_uses
                / max(1, self._melee_opportunities),
            }
        )
        quality["shield_block_precision"] = min(1.0, quality["shield_block_precision"])
        return quality

    def _info(self, outcome: str) -> dict[str, Any]:
        info = super()._info(outcome)
        quality = self._quality_metrics()
        info.update(
            {
                "observation_version": V4_OBSERVATION_VERSION,
                "quality": quality,
                "behavior_counts": {
                    "accepted_action_changes": self._accepted_action_changes,
                    "ground_crouches": self._ground_crouch_decisions,
                    "shield_hold_decisions": self._shield_hold_decisions,
                    "far_idle_decisions": self._far_idle_decisions,
                    "wall_stall_decisions": self._wall_stall_decisions,
                    "melee_opportunities": self._melee_opportunities,
                    "melee_opportunity_uses": self._melee_opportunity_uses,
                },
                **quality,
            }
        )
        return info
