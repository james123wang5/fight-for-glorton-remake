from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from gymnasium import spaces

from .league_env import (
    BASE_OBSERVATION_SIZE,
    LEAGUE_OBSERVATION_SIZE,
    PeachLeagueEnv,
    _edge_danger,
    _is_offstage,
    _platform_distance,
    encode_league_observation,
)
from .tactical_input import (
    COMBAT_LABELS,
    MOVEMENT_LABELS,
    TACTICAL_ACTION_NVECS,
    TacticalInputAdapter,
)


TACTICAL_OBSERVATION_VERSION = "glorton-peach-tactical-v3"
TACTICAL_OBSERVATION_SIZE = 232
TACTICAL_POLICY_HZ = 10
TACTICAL_FRAME_SKIP = 4


@dataclass
class ProjectileTrial:
    projectile: Any
    kind: str
    spawned_tick: int


def _all_projectiles(runtime: Any) -> list[Any]:
    return [*runtime.bullets, *runtime.rockets, *runtime.special_projectiles]


def _projectile_kind(projectile: Any) -> str:
    name = type(projectile).__name__.lower()
    if "bullet" in name:
        return "bullet"
    if "rocket" in name:
        return "rocket"
    return "special"


def _projectile_velocity(projectile: Any) -> tuple[float, float]:
    return float(getattr(projectile, "xinc", 0.0)), float(getattr(projectile, "yinc", 0.0))


def projectile_threat(runtime: Any, fighter: Any, projectile: Any) -> dict[str, float | bool | str]:
    bounds = runtime.stage.bounds
    dx = float(projectile.pos.x - fighter.pos.x)
    dy = float(projectile.pos.y - fighter.pos.y)
    vx, vy = _projectile_velocity(projectile)
    rvx = vx - float(fighter.xinc)
    rvy = vy - float(fighter.yinc)
    speed_sq = rvx * rvx + rvy * rvy
    raw_time = -(dx * rvx + dy * rvy) / speed_sq if speed_sq > 1e-6 else 999.0
    time_ticks = max(0.0, min(120.0, raw_time))
    miss = math.hypot(dx + rvx * time_ticks, dy + rvy * time_ticks)
    kind = _projectile_kind(projectile)
    hit_radius = 62.0 if kind == "rocket" else 32.0
    approaching = raw_time > 0 and raw_time <= 24 and miss <= hit_radius
    time_factor = max(0.0, 1.0 - time_ticks / 24.0)
    miss_factor = max(0.0, 1.0 - miss / hit_radius)
    score = time_factor * miss_factor if approaching else 0.0
    return {
        "dx": dx,
        "dy": dy,
        "vx": vx,
        "vy": vy,
        "time_ticks": time_ticks,
        "miss": miss,
        "approaching": approaching,
        "score": score,
        "kind": kind,
        "bounds_w": float(bounds.w),
        "bounds_h": float(bounds.h),
    }


def enemy_projectile_threats(runtime: Any, fighter: Any, *, limit: int = 2) -> list[dict[str, Any]]:
    entries = [
        projectile_threat(runtime, fighter, projectile)
        for projectile in _all_projectiles(runtime)
        if bool(getattr(projectile, "alive", True))
        and getattr(projectile, "sender", None) is not fighter
    ]
    entries.sort(key=lambda item: (-float(item["score"]), float(item["time_ticks"]), float(item["miss"])))
    return entries[:limit]


def melee_threat(fighter: Any, opponent: Any) -> bool:
    if not str(opponent.current_attack) or str(opponent.current_attack).startswith("special"):
        return False
    dx = float(fighter.pos.x - opponent.pos.x)
    dy = float(fighter.pos.y - opponent.pos.y)
    facing = int(opponent.attack_facing if opponent.current_attack else opponent.facing)
    return bool(abs(dx) <= 75 and abs(dy) <= 55 and facing * dx >= -18)


def clear_shot(runtime: Any, fighter: Any, opponent: Any) -> bool:
    start = (round(fighter.pos.x), round(fighter.pos.y - 20))
    end = (round(opponent.pos.x), round(opponent.pos.y - 20))
    for platform in runtime.stage.platforms:
        if platform.rect.clipline(start, end):
            return False
    return True


def rocket_opportunity(fighter: Any, opponent: Any) -> bool:
    """Approximate Peach's upward 30-degree ballistic rocket intercept."""

    dx = float(opponent.pos.x - fighter.pos.x)
    if fighter.facing * dx <= 20 or abs(dx) > 260:
        return False
    ticks = abs(dx) / 7.5
    if not 3.0 <= ticks <= 35.0:
        return False
    rocket_dy = -12.99 * ticks + 0.25 * ticks * (ticks + 1.0)
    target_dy = float(opponent.pos.y - fighter.pos.y) + float(opponent.yinc) * ticks
    return abs(target_dy - rocket_dy) <= 75.0


def tactical_context(runtime: Any, fighter: Any, opponent: Any) -> dict[str, float | bool]:
    dx = float(opponent.pos.x - fighter.pos.x)
    dy = float(opponent.pos.y - fighter.pos.y)
    distance = math.hypot(dx, dy)
    threats = enemy_projectile_threats(runtime, fighter, limit=2)
    projectile_score = max((float(item["score"]) for item in threats), default=0.0)
    close_attack = melee_threat(fighter, opponent)
    intercept_ticks = min(60.0, abs(dx) / 20.0)
    predicted_y = float(opponent.pos.y) + float(opponent.yinc) * intercept_ticks
    lead_y = predicted_y - float(fighter.pos.y)
    target_behind = fighter.facing * (fighter.pos.x - opponent.pos.x) > 0
    target_in_front = fighter.facing * dx >= 0
    return {
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "clear_shot": clear_shot(runtime, fighter, opponent),
        "melee_threat": close_attack,
        "threat_score": max(projectile_score, 1.0 if close_attack else 0.0),
        "lead_y": lead_y,
        "intercept_ticks": intercept_ticks,
        "target_behind": target_behind,
        "target_in_front": target_in_front,
        "rocket_opportunity": rocket_opportunity(fighter, opponent),
        "offstage": _is_offstage(runtime, fighter),
        "opponent_above": dy < -18 and abs(dx) < 110,
    }


def environmental_combat_mask(runtime: Any, fighter: Any, opponent: Any) -> list[bool]:
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
    return [
        True,
        bool(fighter.jumpstate < 2),
        True,
        bool(distance <= 75 and abs(dy) <= 55 and bool(context["target_in_front"])),
        bool(fighter.on_ground and dy < -8 and abs(dx) <= 90),
        bool(
            fighter.on_ground
            and distance <= 30
            and abs(dy) <= 35
            and bool(context["target_behind"])
        ),
        bool(
            distance >= 45
            and bool(context["clear_shot"])
            and bool(context["target_in_front"])
            and abs(float(context["lead_y"])) <= 85
        ),
        bool(
            fighter.spec_up_ok
            and (bool(context["offstage"]) or bool(context["rocket_opportunity"]))
        ),
        True,
    ]


def tactical_action_mask(
    runtime: Any,
    fighter: Any,
    opponent: Any,
    adapter: TacticalInputAdapter,
    *,
    curriculum: str = "duel",
) -> np.ndarray:
    movement = adapter.action_mask_prefix()
    combat = environmental_combat_mask(runtime, fighter, opponent)
    if adapter.combat_cooldown > 0:
        combat = [True] + [False] * (len(COMBAT_LABELS) - 1)
    elif adapter.previous_combat in {1, 3, 4, 5, 6, 7} and not fighter.current_attack:
        # Force one explicit release/no-op decision before another attack.
        combat = [True] + [False] * (len(COMBAT_LABELS) - 1)
    if adapter.shield_rearm > 0:
        combat[8] = False

    allowed_by_lesson: dict[str, set[int]] = {
        "aim_static": {0, 6},
        "aim_moving": {0, 6},
        "rocket": {0, 1, 7},
        "defense": {0, 1, 8},
        "melee": {0, 1, 3, 4, 5},
        "recovery": {0, 1, 7},
    }
    allowed = allowed_by_lesson.get(curriculum)
    if allowed is not None:
        combat = [value and index in allowed for index, value in enumerate(combat)]
        combat[0] = True
    return np.asarray([*movement, *combat], dtype=bool)


def encode_tactical_observation(
    runtime: Any,
    agent: Any,
    opponent: Any,
    adapter: TacticalInputAdapter,
    *,
    episode_ticks: int,
    max_ticks: int,
    spawns_swapped: bool,
    curriculum: str = "duel",
) -> np.ndarray:
    base = encode_league_observation(
        runtime,
        agent,
        opponent,
        adapter,
        episode_ticks=episode_ticks,
        max_ticks=max_ticks,
        spawns_swapped=spawns_swapped,
    )
    bounds = runtime.stage.bounds
    diagonal = max(1.0, math.hypot(bounds.w, bounds.h))
    extras: list[float] = adapter.tactical_features()
    extras.extend(float(value) for value in environmental_combat_mask(runtime, agent, opponent))
    threats = enemy_projectile_threats(runtime, agent, limit=2)
    for threat in threats:
        kind = str(threat["kind"])
        extras.extend(
            [
                float(threat["dx"]) / max(1.0, bounds.w),
                float(threat["dy"]) / max(1.0, bounds.h),
                float(threat["vx"]) / 30.0,
                float(threat["vy"]) / 30.0,
                float(threat["time_ticks"]) / 40.0,
                float(threat["miss"]) / diagonal,
                float(bool(threat["approaching"])),
                float(kind == "bullet"),
                float(kind == "rocket"),
                float(kind == "special"),
            ]
        )
    extras.extend([0.0] * ((2 - len(threats)) * 10))
    context = tactical_context(runtime, agent, opponent)
    extras.extend(
        [
            float(bool(context["clear_shot"])),
            float(bool(context["melee_threat"])),
            float(context["threat_score"]),
            float(context["lead_y"]) / max(1.0, bounds.h),
            float(context["intercept_ticks"]) / 60.0,
            float(float(context["distance"]) > 150),
        ]
    )
    observation = np.concatenate((base, np.asarray(extras, dtype=np.float32)))
    if observation.shape != (TACTICAL_OBSERVATION_SIZE,):
        raise RuntimeError(f"v3 observation contract changed: {observation.shape}")
    return np.clip(observation, -5.0, 5.0).astype(np.float32, copy=False)


class TacticalPeachEnv(PeachLeagueEnv):
    """V3 curriculum/self-play task with persistent plans and measurable aim."""

    def __init__(
        self,
        *,
        seed: int = 0,
        max_episode_seconds: float = 180.0,
        items_probability: float = 0.20,
        curriculum_strength: float = 0.70,
        lesson_seconds: float = 20.0,
    ) -> None:
        self.curriculum_strength = float(np.clip(curriculum_strength, 0.0, 1.0))
        self.lesson_ticks = max(1, round(float(lesson_seconds) * 40))
        self.curriculum = "duel"
        self._episode_script = ""
        super().__init__(
            seed=seed,
            max_episode_seconds=max_episode_seconds,
            frame_skip=2,
            items_probability=items_probability,
            reaction_delay_decisions=1,
            recovery_start_probability=0.0,
        )
        self.frame_skip = TACTICAL_FRAME_SKIP
        self.adapters = [
            TacticalInputAdapter(frame_skip=self.frame_skip),
            TacticalInputAdapter(frame_skip=self.frame_skip),
        ]
        self.action_space = spaces.MultiDiscrete(TACTICAL_ACTION_NVECS.copy())
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(TACTICAL_OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._projectile_trials: dict[int, ProjectileTrial] = {}
        self._shot_outcomes: Counter[str] = Counter()
        self._shield_metrics: Counter[str] = Counter()
        self._intent_counts: Counter[str] = Counter()
        self._accepted_action_changes = 0
        self._previous_accepted_action: tuple[int, int] | None = None
        self._far_decisions = 0

    def set_curriculum_strength(self, value: float) -> None:
        self.curriculum_strength = float(np.clip(value, 0.0, 1.0))

    def set_tactical_opponent_pool(self, entries: Sequence[tuple[Any, float, str]]) -> None:
        self.opponent_pool = [
            (policy, float(weight), str(name))
            for policy, weight, name in entries
            if float(weight) > 0
        ]

    def _choose_curriculum(self, options: Mapping[str, Any]) -> str:
        explicit = options.get("curriculum")
        if explicit is not None:
            return str(explicit)
        if self.np_random.random() >= self.curriculum_strength:
            return "duel"
        lessons = ("aim_static", "aim_moving", "rocket", "defense", "melee", "recovery", "pursuit")
        probabilities = np.asarray([0.17, 0.15, 0.12, 0.17, 0.16, 0.12, 0.11])
        return str(self.np_random.choice(lessons, p=probabilities / probabilities.sum()))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        options = dict(options or {})
        _observation, _info = super().reset(seed=seed, options=options)
        self.curriculum = self._choose_curriculum(options)
        self._episode_script = "" if self.curriculum == "duel" else self.curriculum
        if self.curriculum != "duel":
            self.runtime.manifest["items"]["frequency"] = 0
            self.runtime.items.clear()
            self._items_enabled = False
            self.opponent_name = f"lesson:{self.curriculum}"
            self._setup_lesson()
        self._projectile_trials = {}
        self._shot_outcomes.clear()
        self._shield_metrics.clear()
        self._intent_counts.clear()
        self._accepted_action_changes = 0
        self._previous_accepted_action = None
        self._far_decisions = 0
        self._observation_buffers = [deque(), deque()]
        for slot in range(2):
            current = self._current_observation(slot)
            self._observation_buffers[slot] = deque(
                [current.copy() for _ in range(self.reaction_delay_decisions + 1)],
                maxlen=self.reaction_delay_decisions + 1,
            )
        return self._delayed_observation(self.agent_slot), self._info("ongoing")

    def _lesson_platform(self) -> Any:
        candidates = [
            platform
            for platform in self.runtime.stage.platforms
            if not platform.moving and 180 <= platform.rect.w <= 400
        ]
        return max(candidates or self.runtime.stage.platforms, key=lambda item: item.rect.w)

    @staticmethod
    def _place_fighter(fighter: Any, platform: Any, x: float) -> None:
        fighter.pos.x = max(platform.rect.left + 25, min(float(x), platform.rect.right - 25))
        fighter.pos.y = float(platform.rect.top)
        fighter.prev_pos.update(fighter.pos)
        fighter.xinc = 0.0
        fighter.yinc = 0.0
        fighter.state = "stop"
        fighter.current_attack = ""
        fighter.current_label = "still"
        fighter.has_control = True
        fighter.ctrl_loss = 0
        fighter.paralized = 0
        fighter.shielded = False
        fighter.invincible = False
        fighter.spawn_invincible_ms = 0
        fighter._land_on_platform(platform)

    def _setup_lesson(self) -> None:
        if self.curriculum == "rocket":
            by_name = {platform.name: platform for platform in self.runtime.stage.platforms}
            lower = by_name.get("Fixed6")
            upper = by_name.get("Fixed7")
            if lower is not None and upper is not None:
                self._place_fighter(self.agent, lower, lower.rect.right - 26)
                self._place_fighter(self.opponent, upper, upper.rect.left + 36)
                self.agent.facing = 1
                self.opponent.facing = -1
                return
        platform = self._lesson_platform()
        center = float(platform.rect.centerx)
        distance = 160.0
        if self.curriculum == "melee":
            distance = 42.0
        elif self.curriculum == "pursuit":
            distance = 190.0
        self._place_fighter(self.agent, platform, center - distance / 2)
        self._place_fighter(self.opponent, platform, center + distance / 2)
        self.agent.facing = 1
        self.opponent.facing = -1
        if self.curriculum == "recovery":
            side = -1 if self.np_random.random() < 0.5 else 1
            self.agent.pos.x = platform.rect.left - 70 if side < 0 else platform.rect.right + 70
            self.agent.pos.y = platform.rect.top - 65
            self.agent.prev_pos.update(self.agent.pos)
            self.agent.xinc = 2.0 * side
            self.agent.yinc = 2.0
            self.agent.on_ground = False
            self.agent.ground_platform = None
            self.agent.jumpstate = 1
            self.agent.spec_up_ok = True

    def action_masks(self) -> np.ndarray:
        return self._action_mask_for_slot(self.agent_slot)

    def _action_mask_for_slot(self, slot: int) -> np.ndarray:
        return tactical_action_mask(
            self.runtime,
            self.runtime.fighters[slot],
            self.runtime.fighters[1 - slot],
            self.adapters[slot],
            curriculum=self.curriculum if slot == self.agent_slot else "duel",
        )

    def _validate_action(self, action: np.ndarray | Sequence[int]) -> np.ndarray:
        value = np.asarray(action, dtype=np.int64).reshape(-1)
        if value.shape != (2,) or not self.action_space.contains(value):
            raise ValueError(f"invalid tactical action {action!r}; expected MultiDiscrete([4, 9])")
        return value

    def _legacy_action_to_tactical(self, action: Any, *, slot: int) -> np.ndarray:
        raw = np.asarray(action, dtype=np.int64).reshape(-1)
        if raw.shape != (3,):
            return np.zeros(2, dtype=np.int64)
        fighter = self.runtime.fighters[slot]
        opponent = self.runtime.fighters[1 - slot]
        horizontal, vertical, combat = (int(value) for value in raw)
        if horizontal == 0:
            movement = 0
        else:
            requested_right = horizontal == 2
            target_right = opponent.pos.x >= fighter.pos.x
            movement = 1 if requested_right == target_right else 2
        if combat == 1:
            intent = 4 if vertical == 1 else 3
            if environmental_combat_mask(self.runtime, fighter, opponent)[5]:
                intent = 5
        elif combat == 2:
            intent = 7 if vertical == 1 else 6
        elif combat == 3:
            intent = 8
        elif vertical == 1:
            intent = 1
        elif vertical == 2:
            intent = 2
        else:
            intent = 0
        return np.asarray([movement, intent], dtype=np.int64)

    def _script_action(self) -> np.ndarray:
        lesson = self._episode_script
        if lesson in {"aim_static", "rocket", "melee", "recovery"}:
            return np.asarray([0, 0], dtype=np.int64)
        if lesson == "defense":
            combat = 6 if self._decision_steps % 8 == 0 else 0
            return np.asarray([0, combat], dtype=np.int64)
        if lesson == "pursuit":
            return np.asarray([2, 0], dtype=np.int64)
        if lesson == "aim_moving":
            phase = (self._decision_steps // 20) % 2
            return np.asarray([2 if phase == 0 else 1, 0], dtype=np.int64)
        return np.asarray([0, 0], dtype=np.int64)

    def _opponent_action(self) -> np.ndarray:
        if self._episode_script:
            return self._script_action()
        policy = self.opponent_policy
        if policy is None or policy == "idle":
            return np.asarray([0, 0], dtype=np.int64)
        if policy == "retreat":
            return np.asarray([2, 0], dtype=np.int64)
        observation = self._delayed_observation(self.opponent_slot)
        expected = tuple(getattr(getattr(policy, "observation_space", None), "shape", ()) or ())
        if expected in {(BASE_OBSERVATION_SIZE,), (LEAGUE_OBSERVATION_SIZE,)}:
            legacy_observation = observation[: expected[0]]
            action, _state = policy.predict(
                legacy_observation,
                deterministic=self.opponent_deterministic,
            )
            return self._legacy_action_to_tactical(action, slot=self.opponent_slot)
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
        return encode_tactical_observation(
            self.runtime,
            self.runtime.fighters[slot],
            self.runtime.fighters[1 - slot],
            self.adapters[slot],
            episode_ticks=min(self._episode_ticks, self.max_ticks),
            max_ticks=self.max_ticks,
            spawns_swapped=self._spawns_swapped_by_slot[slot],
            curriculum=self.curriculum if slot == self.agent_slot else "duel",
        )

    def _event_state(self) -> dict[str, float | int]:
        state = super()._event_state()
        state.update(
            {
                "agent_damage": float(self.agent.damage_amnt),
                "agent_lives": int(self.agent.lives),
                "agent_shield": float(self.agent.shield_size),
                "opponent_shield": float(self.opponent.shield_size),
            }
        )
        return state

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
                reward += 0.005
            elif contact and (damage or ringout) and self.opponent.last_sender is self.agent:
                outcome = "hit"
                reward += 0.10 if self.curriculum in {"aim_static", "aim_moving", "rocket"} else 0.04
            else:
                outcome = "miss"
                reward -= 0.03 if self.curriculum in {"aim_static", "aim_moving", "rocket"} else 0.015
            self._shot_outcomes[f"{trial.kind}_{outcome}"] += 1
            del self._projectile_trials[projectile_id]
        return reward

    def _track_shield(self, before: Mapping[str, float | int], *, threat_before: float) -> float:
        reward = 0.0
        shield_drop = float(before["agent_shield"]) - float(self.agent.shield_size)
        damage_unchanged = float(self.agent.damage_amnt) <= float(before["agent_damage"])
        if shield_drop > 0.5 and damage_unchanged:
            self._shield_metrics["blocks"] += 1
            reward += 0.08 if self.curriculum == "defense" else 0.04
        if self.agent.shielded and threat_before < 0.05:
            reward -= 0.001
            self._shield_metrics["unthreatened_hold_steps"] += 1
        return reward

    def step(
        self,
        action: np.ndarray | Sequence[int],
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        learner_action = self._validate_action(action)
        opponent_action = self._opponent_action()
        decisions = [np.zeros(2, dtype=np.int64), np.zeros(2, dtype=np.int64)]
        decisions[self.agent_slot] = learner_action
        decisions[self.opponent_slot] = opponent_action
        masks = [self._action_mask_for_slot(slot) for slot in range(2)]
        shield_starts_before = self.adapters[self.agent_slot].shield_starts
        control_sequences = [
            self.adapters[slot].begin_decision(
                decisions[slot],
                fighter=self.runtime.fighters[slot],
                opponent=self.runtime.fighters[1 - slot],
                action_mask=masks[slot],
            )
            for slot in range(2)
        ]
        accepted = tuple(int(value) for value in self.adapters[self.agent_slot].last_action)
        self._intent_counts[f"movement:{MOVEMENT_LABELS[accepted[0]]}"] += 1
        self._intent_counts[f"combat:{COMBAT_LABELS[accepted[1]]}"] += 1
        if self._previous_accepted_action is not None and accepted != self._previous_accepted_action:
            self._accepted_action_changes += 1
        self._previous_accepted_action = accepted
        if self.agent.pos.distance_to(self.opponent.pos) > 150:
            self._far_decisions += 1

        before = self._reward_state()
        context_before = tactical_context(self.runtime, self.agent, self.opponent)
        shield_reward = 0.0
        new_shield_start = self.adapters[self.agent_slot].shield_starts > shield_starts_before
        if new_shield_start:
            self._shield_metrics["activations"] += 1
            if float(context_before["threat_score"]) < 0.10:
                self._shield_metrics["false_activations"] += 1
                shield_reward -= 0.008
            else:
                self._shield_metrics["threatened_activations"] += 1

        event_reward = 0.0
        projectile_reward = 0.0
        for internal_tick in range(self.frame_skip):
            tick_before = self._event_state()
            melee_hits_before = sum(self._successful_attacks.values())
            controls = [control_sequences[slot][internal_tick] for slot in range(2)]
            self.simulation.step_fast(controls)
            self._episode_ticks += 1
            whiffs_before = self._attack_counts.get("melee_whiff", 0)
            event_reward += super()._track_attack_events(tick_before)
            if self._attack_counts.get("melee_whiff", 0) > whiffs_before:
                event_reward -= 0.009
            if (
                self.curriculum == "melee"
                and sum(self._successful_attacks.values()) > melee_hits_before
            ):
                event_reward += 0.05
            projectile_reward += self._track_projectiles(tick_before)
            shield_reward += self._track_shield(
                tick_before,
                threat_before=float(context_before["threat_score"]),
            )
            if self.runtime.match_state == "game_set" or self._episode_ticks >= self.max_ticks:
                break

        self._decision_steps += 1
        for slot in range(2):
            self._observation_buffers[slot].append(self._current_observation(slot))
        terminated = self.runtime.match_state == "game_set"
        lesson_done = self.curriculum != "duel" and self._episode_ticks >= self.lesson_ticks
        truncated = (self._episode_ticks >= self.max_ticks or lesson_done) and not terminated
        outcome = self._outcome(terminated=terminated, truncated=truncated)
        reward, components = super()._reward(before, outcome=outcome, terminated=terminated)
        components["skill_timing"] += event_reward
        components["projectile_accuracy"] += projectile_reward
        components["shield_discipline"] += shield_reward
        if self.curriculum in {"aim_static", "aim_moving", "rocket", "defense", "melee"}:
            # A zero-action policy must not solve a timing lesson by merely
            # avoiding miss penalties. Correct attempts easily repay this.
            components["curriculum"] -= 0.0005
        if self.curriculum == "pursuit":
            closed = float(before["distance"]) - float(self.agent.pos.distance_to(self.opponent.pos))
            components["curriculum"] += 0.003 * max(-15.0, min(15.0, closed))
        reward += event_reward + projectile_reward + shield_reward + components["curriculum"]
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

    @staticmethod
    def _empty_reward_components() -> dict[str, float]:
        components = PeachLeagueEnv._empty_reward_components()
        components.update(
            {
                "projectile_accuracy": 0.0,
                "shield_discipline": 0.0,
                "curriculum": 0.0,
            }
        )
        return components

    def _quality_metrics(self) -> dict[str, float]:
        seconds = max(0.1, self._decision_steps / TACTICAL_POLICY_HZ)
        minutes = seconds / 60.0
        fired = self._shot_outcomes["bullet_fired"] + self._shot_outcomes["rocket_fired"]
        hits = self._shot_outcomes["bullet_hit"] + self._shot_outcomes["rocket_hit"]
        misses = self._shot_outcomes["bullet_miss"] + self._shot_outcomes["rocket_miss"]
        resolved = hits + misses + self._shot_outcomes["bullet_blocked"] + self._shot_outcomes["rocket_blocked"]
        melee_starts = sum(
            self._attack_counts[label]
            for label in ("punchGround", "punchRun", "punchUp", "punchAir", "specialBackThrow")
        )
        melee_hits = sum(
            self._successful_attacks[label]
            for label in ("punchGround", "punchRun", "punchUp", "punchAir", "specialBackThrow")
        )
        activations = self._shield_metrics["activations"]
        return {
            "action_change_rate": self._accepted_action_changes / max(1, self._decision_steps - 1),
            "projectiles_per_minute": fired / minutes,
            "projectile_accuracy": hits / max(1, resolved),
            "shield_activations_per_minute": activations / minutes,
            "false_shield_rate": self._shield_metrics["false_activations"] / max(1, activations),
            "shield_block_precision": self._shield_metrics["blocks"] / max(1, activations),
            "melee_hit_rate": melee_hits / max(1, melee_starts),
            "far_fraction": self._far_decisions / max(1, self._decision_steps),
        }

    def _info(self, outcome: str) -> dict[str, Any]:
        info = super()._info(outcome)
        quality = self._quality_metrics()
        info.update(
            {
                "curriculum": self.curriculum,
                "observation_version": TACTICAL_OBSERVATION_VERSION,
                "shot_outcomes": dict(self._shot_outcomes),
                "shield_metrics": dict(self._shield_metrics),
                "intent_counts": dict(self._intent_counts),
                "quality": quality,
                **quality,
            }
        )
        return info
