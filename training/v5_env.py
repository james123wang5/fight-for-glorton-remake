from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
from gymnasium import spaces

from .league_env import BASE_OBSERVATION_SIZE, LEAGUE_OBSERVATION_SIZE, _is_offstage
from .tactical_env import (
    TACTICAL_OBSERVATION_SIZE,
    tactical_action_mask,
    tactical_context,
)
from .v4_env import (
    V4_OBSERVATION_SIZE,
    V4PeachEnv,
    encode_v4_observation,
    v4_action_mask,
)
from .v5_options import (
    PURPOSE_COUNT,
    PURPOSE_LABELS,
    Purpose,
    PurposefulOptionController,
    purpose_action_mask,
)


V5_OBSERVATION_VERSION = "glorton-peach-purpose-v5"
V5_OBSERVATION_SIZE = 294
V5_POLICY_HZ = 10
V5_FRAME_SKIP = 4


def encode_v5_observation(
    runtime: Any,
    fighter: Any,
    opponent: Any,
    controller: PurposefulOptionController,
    *,
    episode_ticks: int,
    max_ticks: int,
    spawns_swapped: bool,
    curriculum: str = "duel",
    wall_stall_steps: int = 0,
) -> np.ndarray:
    base = encode_v4_observation(
        runtime,
        fighter,
        opponent,
        controller.adapter,
        episode_ticks=episode_ticks,
        max_ticks=max_ticks,
        spawns_swapped=spawns_swapped,
        curriculum=curriculum,
        wall_stall_steps=wall_stall_steps,
    )
    purpose = np.asarray(controller.features(fighter, opponent), dtype=np.float32)
    observation = np.concatenate((base, purpose))
    if observation.shape != (V5_OBSERVATION_SIZE,):
        raise RuntimeError(
            f"v5 observation contract changed: {observation.shape}; "
            f"purpose extras={purpose.shape}"
        )
    return np.clip(observation, -5.0, 5.0).astype(np.float32, copy=False)


class V5PeachEnv(V4PeachEnv):
    """Purpose-driven Peach: plans select goals; executors own raw key timing."""

    def __init__(
        self,
        *,
        seed: int = 0,
        max_episode_seconds: float = 120.0,
        items_probability: float = 0.0,
        curriculum_strength: float = 0.70,
        lesson_seconds: float = 16.0,
    ) -> None:
        self.fixed_curriculum: str | None = None
        self.intent_controllers: list[PurposefulOptionController] = []
        self._purpose_intent_counts: Counter[str] = Counter()
        self._purpose_switches = 0
        self._previous_purpose: Purpose | None = None
        self._air_chase_hits = 0
        self._air_chase_opportunities = 0
        self._air_chase_selections = 0
        self._escape_successes = 0
        self._escape_opportunities = 0
        self._lesson_success = False
        self._lesson_target_platform: Any | None = None
        super().__init__(
            seed=seed,
            max_episode_seconds=max_episode_seconds,
            items_probability=items_probability,
            curriculum_strength=curriculum_strength,
            lesson_seconds=lesson_seconds,
        )
        self.intent_controllers = [
            PurposefulOptionController(self.runtime),
            PurposefulOptionController(self.runtime),
        ]
        self.adapters = [controller.adapter for controller in self.intent_controllers]
        self.action_space = spaces.Discrete(PURPOSE_COUNT)
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(V5_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    def _choose_curriculum(self, options: Mapping[str, Any]) -> str:
        explicit = options.get("curriculum")
        if explicit is not None:
            return str(explicit)
        if self.fixed_curriculum is not None:
            return self.fixed_curriculum
        if self.np_random.random() >= self.curriculum_strength:
            return "duel"
        lessons = ("v5_navigation", "v5_air_chase", "v5_escape", "v5_combo")
        probabilities = np.asarray([0.25, 0.30, 0.20, 0.25], dtype=np.float64)
        return str(self.np_random.choice(lessons, p=probabilities / probabilities.sum()))

    def set_fixed_curriculum(self, curriculum: str | None) -> None:
        self.fixed_curriculum = curriculum

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        for controller in self.intent_controllers:
            controller.reset()
        self._purpose_intent_counts.clear()
        self._purpose_switches = 0
        self._previous_purpose = None
        self._air_chase_hits = 0
        self._air_chase_opportunities = 0
        self._air_chase_selections = 0
        self._escape_successes = 0
        self._escape_opportunities = 0
        self._lesson_success = False
        self._lesson_target_platform = None
        observation, info = super().reset(seed=seed, options=options)
        return self._delayed_observation(self.agent_slot), self._info(str(info["outcome"]))

    def _setup_lesson(self) -> None:
        by_name = {platform.name: platform for platform in self.runtime.stage.platforms}
        if self.curriculum == "v5_navigation":
            base = by_name.get("Fixed1")
            wall = by_name.get("Fixed12" if self.np_random.random() < 0.5 else "Fixed13")
            if base is not None and wall is not None:
                side = -1 if self.np_random.random() < 0.5 else 1
                start_x = wall.rect.left - 55 if side < 0 else wall.rect.right + 55
                target_x = wall.rect.right + 100 if side < 0 else wall.rect.left - 100
                self._place_fighter(self.agent, base, start_x)
                self._place_fighter(self.opponent, base, target_x)
                self.agent.facing = -side
                self.opponent.facing = side
                self._lesson_target_platform = wall
                return
        platform = self._lesson_platform()
        center = float(platform.rect.centerx)
        if self.curriculum == "v5_air_chase":
            self._place_fighter(self.agent, platform, center - 45)
            self._place_fighter(self.opponent, platform, center + 35)
            self.opponent.pos.y = float(platform.rect.top - 95)
            self.opponent.prev_pos.update(self.opponent.pos)
            self.opponent.on_ground = False
            self.opponent.ground_platform = None
            self.opponent.state = "thrown"
            self.opponent.current_label = "thrown"
            self.opponent.ctrl_loss = 650
            self.opponent.xinc = float(self.np_random.choice((-2.0, 2.0)))
            self.opponent.yinc = -6.0
            self.opponent.damage_amnt = 70
            return
        if self.curriculum == "v5_escape":
            self._place_fighter(self.agent, platform, center + 20)
            self._place_fighter(self.opponent, platform, center - 10)
            self.agent.pos.y = float(platform.rect.top - 95)
            self.agent.prev_pos.update(self.agent.pos)
            self.agent.on_ground = False
            self.agent.ground_platform = None
            self.agent.state = "thrown"
            self.agent.current_label = "thrown"
            self.agent.ctrl_loss = 125
            self.agent.jumpstate = 1
            self.agent.xinc = float(self.np_random.choice((-1.5, 1.5)))
            self.agent.yinc = -2.0
            return
        if self.curriculum == "v5_combo":
            self._place_fighter(self.agent, platform, center - 35)
            self._place_fighter(self.opponent, platform, center + 25)
            self.opponent.damage_amnt = 65
            self.agent.facing = 1
            self.opponent.facing = -1
            return
        super()._setup_lesson()

    def action_masks(self) -> np.ndarray:
        return self._action_mask_for_slot(self.agent_slot)

    def _action_mask_for_slot(self, slot: int) -> np.ndarray:
        curriculum = self.curriculum if slot == self.agent_slot else "duel"
        mask = purpose_action_mask(
            self.runtime,
            self.runtime.fighters[slot],
            self.runtime.fighters[1 - slot],
            self.intent_controllers[slot],
            curriculum=curriculum,
        )
        if slot == self.opponent_slot and (
            self._episode_script
            or (isinstance(self.opponent_policy, str) and self.opponent_policy == "idle")
        ):
            mask[Purpose.CONTINUE] = True
        return mask

    def _validate_action(self, action: int | np.integer[Any] | np.ndarray) -> int:
        values = np.asarray(action, dtype=np.int64).reshape(-1)
        if values.shape != (1,):
            raise ValueError(f"invalid v5 purpose {action!r}; expected Discrete({PURPOSE_COUNT})")
        value = int(values[0])
        if not self.action_space.contains(value):
            raise ValueError(f"invalid v5 purpose {action!r}; expected Discrete({PURPOSE_COUNT})")
        return value

    def _purpose_from_tactical(self, action: Sequence[int], *, slot: int) -> int:
        movement, combat = (int(value) for value in np.asarray(action).reshape(-1)[:2])
        fighter = self.runtime.fighters[slot]
        opponent = self.runtime.fighters[1 - slot]
        route = self.intent_controllers[slot].navigator.route(fighter, opponent)
        if combat == 1:
            value = Purpose.NAVIGATE if route.requires_jump else Purpose.AIR_CHASE
        elif combat == 2:
            value = Purpose.LAND
        elif combat == 3:
            value = Purpose.AIR_CHASE if not fighter.on_ground else Purpose.MELEE
        elif combat == 4:
            value = Purpose.ANTI_AIR
        elif combat == 5:
            value = Purpose.BACK_THROW
        elif combat == 6:
            value = Purpose.AIMED_SHOT
        elif combat == 7:
            value = Purpose.RECOVER if _is_offstage(self.runtime, fighter) else Purpose.ROCKET
        elif combat == 8:
            value = Purpose.SHIELD
        elif movement == 2:
            value = Purpose.EVADE
        else:
            value = Purpose.NAVIGATE if route.requires_jump else Purpose.CHASE
        mask = self._action_mask_for_slot(slot)
        if mask[int(value)]:
            return int(value)
        preferred = (
            Purpose.RECOVER,
            Purpose.HITSTUN_ESCAPE,
            Purpose.NAVIGATE,
            Purpose.AIR_CHASE,
            Purpose.MELEE,
            Purpose.CHASE,
            Purpose.CONTINUE,
        )
        return int(next(item for item in preferred if mask[int(item)]))

    def _scripted_opponent_purpose(self, style: str) -> int:
        slot = self.opponent_slot
        mask = self._action_mask_for_slot(slot)
        fighter = self.opponent
        target = self.agent
        if style == "idle":
            return int(Purpose.CONTINUE if mask[Purpose.CONTINUE] else np.flatnonzero(mask)[0])
        if style == "retreat":
            return int(Purpose.EVADE if mask[Purpose.EVADE] else np.flatnonzero(mask)[0])
        priorities = (
            Purpose.RECOVER,
            Purpose.HITSTUN_ESCAPE,
            Purpose.NAVIGATE,
            Purpose.AIR_CHASE,
            Purpose.BACK_THROW,
            Purpose.ANTI_AIR,
            Purpose.MELEE,
            Purpose.AIMED_SHOT if style == "active" else Purpose.CHASE,
            Purpose.CHASE,
            Purpose.CONTINUE,
        )
        if target.state == "thrown" and mask[Purpose.AIR_CHASE]:
            return int(Purpose.AIR_CHASE)
        return int(next(item for item in priorities if mask[int(item)]))

    def _script_action(self) -> int:
        if self.curriculum in {"v5_navigation", "v5_air_chase"}:
            return self._scripted_opponent_purpose("idle")
        if self.curriculum == "v5_escape":
            return self._scripted_opponent_purpose("melee")
        if self.curriculum == "v5_combo":
            return self._scripted_opponent_purpose("retreat")
        return self._scripted_opponent_purpose("active")

    def _opponent_action(self) -> int:
        if self._episode_script:
            return self._script_action()
        policy = self.opponent_policy
        if policy is None:
            return self._scripted_opponent_purpose("idle")
        if isinstance(policy, str):
            return self._scripted_opponent_purpose(policy)

        observation = self._delayed_observation(self.opponent_slot)
        expected = tuple(getattr(getattr(policy, "observation_space", None), "shape", ()) or ())
        if expected in {(BASE_OBSERVATION_SIZE,), (LEAGUE_OBSERVATION_SIZE,)}:
            action, _state = policy.predict(
                observation[: expected[0]], deterministic=self.opponent_deterministic
            )
            tactical = self._legacy_action_to_tactical(action, slot=self.opponent_slot)
            return self._purpose_from_tactical(tactical, slot=self.opponent_slot)
        if expected == (TACTICAL_OBSERVATION_SIZE,):
            mask = tactical_action_mask(
                self.runtime,
                self.opponent,
                self.agent,
                self.adapters[self.opponent_slot],
                curriculum="duel",
            )
            try:
                action, _state = policy.predict(
                    observation[:TACTICAL_OBSERVATION_SIZE],
                    action_masks=mask,
                    deterministic=self.opponent_deterministic,
                )
            except TypeError:
                action, _state = policy.predict(
                    observation[:TACTICAL_OBSERVATION_SIZE],
                    deterministic=self.opponent_deterministic,
                )
            return self._purpose_from_tactical(action, slot=self.opponent_slot)
        if expected == (V4_OBSERVATION_SIZE,):
            mask = v4_action_mask(
                self.runtime,
                self.opponent,
                self.agent,
                self.adapters[self.opponent_slot],
                curriculum="duel",
            )
            action, _state = policy.predict(
                observation[:V4_OBSERVATION_SIZE],
                action_masks=mask,
                deterministic=self.opponent_deterministic,
            )
            return self._purpose_from_tactical(action, slot=self.opponent_slot)
        if expected != (V5_OBSERVATION_SIZE,):
            return self._scripted_opponent_purpose("active")
        mask = self._action_mask_for_slot(self.opponent_slot)
        try:
            action, _state = policy.predict(
                observation,
                action_masks=mask,
                deterministic=self.opponent_deterministic,
            )
        except TypeError:
            action, _state = policy.predict(
                observation, deterministic=self.opponent_deterministic
            )
        return self._validate_action(action)

    def _current_observation(self, slot: int) -> np.ndarray:
        return encode_v5_observation(
            self.runtime,
            self.runtime.fighters[slot],
            self.runtime.fighters[1 - slot],
            self.intent_controllers[slot],
            episode_ticks=min(self._episode_ticks, self.max_ticks),
            max_ticks=self.max_ticks,
            spawns_swapped=self._spawns_swapped_by_slot[slot],
            curriculum=self.curriculum if slot == self.agent_slot else "duel",
            wall_stall_steps=self.intent_controllers[slot].no_progress_steps,
        )

    @staticmethod
    def _empty_reward_components() -> dict[str, float]:
        components = V4PeachEnv._empty_reward_components()
        components["purpose"] = 0.0
        return components

    def _purpose_lesson_succeeded(self) -> bool:
        if self.curriculum == "v5_navigation":
            return bool(
                self._lesson_target_platform is not None
                and self.agent.on_ground
                and self.agent.ground_platform is self._lesson_target_platform
            )
        if self.curriculum in {"v5_air_chase", "v5_combo"}:
            return self._air_chase_hits > 0
        if self.curriculum == "v5_escape":
            return self._escape_successes > 0
        return False

    def step(
        self,
        action: int | np.integer[Any] | np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        learner_intent = self._validate_action(action)
        opponent_intent = self._opponent_action()
        intents = [int(Purpose.CONTINUE), int(Purpose.CONTINUE)]
        intents[self.agent_slot] = learner_intent
        intents[self.opponent_slot] = opponent_intent
        masks = [self._action_mask_for_slot(slot) for slot in range(2)]
        before_events = [vars(controller.events).copy() for controller in self.intent_controllers]
        before_successes = sum(self._successful_attacks.values())
        before_air_hits = self._successful_attacks["punchAir"]
        before_agent_damage = float(self.agent.damage_amnt)
        before_opponent_yinc = float(self.opponent.yinc)
        before_controller_completions = self.intent_controllers[
            self.agent_slot
        ].events.plan_completions
        shield_starts_before = self.adapters[self.agent_slot].shield_starts
        context_before = tactical_context(self.runtime, self.agent, self.opponent)
        if masks[self.agent_slot][Purpose.AIR_CHASE]:
            self._air_chase_opportunities += 1
            if learner_intent == Purpose.AIR_CHASE:
                self._air_chase_selections += 1
        if self.agent.state == "thrown" or self.agent.ctrl_loss > 0:
            self._escape_opportunities += 1

        control_sequences = [
            self.intent_controllers[slot].begin_decision(
                intents[slot],
                fighter=self.runtime.fighters[slot],
                opponent=self.runtime.fighters[1 - slot],
                action_mask=masks[slot],
            )
            for slot in range(2)
        ]
        accepted_purpose = self.intent_controllers[self.agent_slot].intent
        self._purpose_intent_counts[PURPOSE_LABELS[int(accepted_purpose)]] += 1
        if self._previous_purpose is not None and accepted_purpose != self._previous_purpose:
            self._purpose_switches += 1
        self._previous_purpose = accepted_purpose

        before = self._reward_state()
        shield_reward = 0.0
        new_shield_start = self.adapters[self.agent_slot].shield_starts > shield_starts_before
        if new_shield_start:
            self._shield_metrics["activations"] += 1
            if float(context_before["threat_score"]) < 0.20:
                self._shield_metrics["false_activations"] += 1
                shield_reward -= 0.015
            else:
                self._shield_metrics["threatened_activations"] += 1

        event_reward = 0.0
        projectile_reward = 0.0
        for internal_tick in range(self.frame_skip):
            tick_before = self._event_state()
            whiffs_before = self._attack_counts.get("melee_whiff", 0)
            controls = [control_sequences[slot][internal_tick] for slot in range(2)]
            self.simulation.step_fast(controls)
            self._episode_ticks += 1
            event_reward += super()._track_attack_events(tick_before)
            if self._attack_counts.get("melee_whiff", 0) > whiffs_before:
                event_reward -= 0.015
            projectile_reward += self._track_projectiles(tick_before)
            shield_reward += self._track_shield(
                tick_before,
                threat_before=float(context_before["threat_score"]),
            )
            if self.runtime.match_state == "game_set" or self._episode_ticks >= self.max_ticks:
                break

        purpose_reward = 0.0
        for slot, option_controller in enumerate(self.intent_controllers):
            result = option_controller.observe_result(
                self.runtime.fighters[slot], self.runtime.fighters[1 - slot]
            )
            if slot == self.agent_slot:
                purpose_reward += result
        after_events = vars(self.intent_controllers[self.agent_slot].events)
        purpose_reward -= 0.02 * (
            int(after_events["purposeless_jumps"])
            - int(before_events[self.agent_slot]["purposeless_jumps"])
        )
        purpose_reward -= 0.05 * (
            int(after_events["jump_down_reversals"])
            - int(before_events[self.agent_slot]["jump_down_reversals"])
        )

        controller = self.intent_controllers[self.agent_slot]
        if self._successful_attacks["punchAir"] > before_air_hits:
            self._air_chase_hits += 1
            if accepted_purpose == Purpose.AIR_CHASE:
                controller.complete_current_plan()
                purpose_reward += 0.08
                if self.opponent.yinc < min(-1.0, before_opponent_yinc):
                    purpose_reward += 0.04
        if (
            accepted_purpose == Purpose.HITSTUN_ESCAPE
            and controller.events.plan_completions > before_controller_completions
            and self.agent.damage_amnt <= before_agent_damage
        ):
            self._escape_successes += 1
            purpose_reward += 0.04

        self._decision_steps += 1
        if self.agent.shielded:
            self._shield_hold_decisions += 1
        if self.agent.pos.distance_to(self.opponent.pos) > 130.0 and accepted_purpose not in {
            Purpose.CHASE,
            Purpose.NAVIGATE,
            Purpose.AIR_CHASE,
            Purpose.AIMED_SHOT,
        }:
            self._far_idle_decisions += 1
            purpose_reward -= 0.01
        wall_delta = (
            int(after_events["wall_stall_decisions"])
            - int(before_events[self.agent_slot]["wall_stall_decisions"])
        )
        self._wall_stall_decisions += max(0, wall_delta)
        self._wall_stall_steps = [
            option.no_progress_steps for option in self.intent_controllers
        ]
        melee_opportunity = bool(
            masks[self.agent_slot][Purpose.MELEE]
            or masks[self.agent_slot][Purpose.AIR_CHASE]
            or masks[self.agent_slot][Purpose.ANTI_AIR]
            or masks[self.agent_slot][Purpose.BACK_THROW]
        )
        if melee_opportunity:
            self._melee_opportunities += 1
            if accepted_purpose in {
                Purpose.MELEE,
                Purpose.AIR_CHASE,
                Purpose.ANTI_AIR,
                Purpose.BACK_THROW,
            }:
                self._melee_opportunity_uses += 1

        self._lesson_success = self._purpose_lesson_succeeded()
        for slot in range(2):
            self._observation_buffers[slot].append(self._current_observation(slot))
        terminated = self.runtime.match_state == "game_set"
        lesson_timed_out = bool(
            self.curriculum != "duel" and self._episode_ticks >= self.lesson_ticks
        )
        lesson_done = bool(self.curriculum != "duel" and self._lesson_success)
        truncated = bool(
            not terminated
            and (
                self._episode_ticks >= self.max_ticks
                or lesson_timed_out
                or lesson_done
            )
        )
        if self.curriculum != "duel" and truncated:
            outcome = "lesson_success" if self._lesson_success else "lesson_failure"
        else:
            outcome = self._outcome(terminated=terminated, truncated=truncated)
        reward, components = self._reward(before, outcome=outcome, terminated=terminated)
        if self.curriculum != "duel" and truncated:
            purpose_reward += 0.50 if self._lesson_success else -0.25
        if sum(self._successful_attacks.values()) > before_successes:
            if accepted_purpose in {
                Purpose.MELEE,
                Purpose.ANTI_AIR,
                Purpose.BACK_THROW,
            }:
                controller.complete_current_plan()
            event_reward += 0.01
        components["skill_timing"] += event_reward
        components["projectile_accuracy"] += projectile_reward
        components["shield_discipline"] += shield_reward
        components["purpose"] += purpose_reward
        reward += event_reward + projectile_reward + shield_reward + purpose_reward
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

    def _quality_metrics(self) -> dict[str, float]:
        quality = super()._quality_metrics()
        metrics = self.intent_controllers[self.agent_slot].metrics()
        quality.update(
            {
                "purpose_switch_rate": self._purpose_switches
                / max(1, self._decision_steps - 1),
                "plan_completion_rate": float(metrics["plan_completion_rate"]),
                "purposeful_jump_rate": float(metrics["purposeful_jump_rate"]),
                "jump_down_reversal_rate": int(metrics["jump_down_reversals"])
                / max(1, self._decision_steps),
                "air_chase_opportunity_use_rate": self._air_chase_selections
                / max(1, self._air_chase_opportunities),
                "air_chase_hit_rate": self._air_chase_hits
                / max(1, int(metrics["air_chase_attempts"])),
                "escape_success_rate": self._escape_successes
                / max(1, self._escape_opportunities),
            }
        )
        return quality

    def _info(self, outcome: str) -> dict[str, Any]:
        info = super()._info(outcome)
        quality = self._quality_metrics()
        purpose_metrics = dict(self.intent_controllers[self.agent_slot].metrics())
        info.update(
            {
                "observation_version": V5_OBSERVATION_VERSION,
                "purpose_counts": dict(self._purpose_intent_counts),
                "purpose_metrics": purpose_metrics,
                "lesson_success": self._lesson_success,
                "air_chase_hits": self._air_chase_hits,
                "air_chase_opportunities": self._air_chase_opportunities,
                "air_chase_selections": self._air_chase_selections,
                "escape_successes": self._escape_successes,
                "escape_opportunities": self._escape_opportunities,
                "quality": quality,
                **quality,
            }
        )
        return info
