from __future__ import annotations

import copy
from typing import Any, Mapping

import numpy as np
from gymnasium import spaces

from src.runtime import Stage

from .roster_contract import (
    FIGHTER_ORDER,
    STAGE_ORDER,
    make_training_match_config,
)
from .roster_observation import (
    ROSTER_OBSERVATION_SIZE,
    ROSTER_OBSERVATION_VERSION,
    encode_roster_observation,
)
from .roster_options import role_purpose_labels, roster_purpose_action_mask
from .v5_env import V5_OBSERVATION_SIZE, V5PeachEnv
from .v5_options import PURPOSE_COUNT, Purpose
from .v5_runtime_helpers import is_offstage


class RosterPurposeEnv(V5PeachEnv):
    """Role-aware Mogadishu environment built on the approved v5 behavior."""

    def __init__(
        self,
        *,
        fighter_name: str,
        opponent_fighter_name: str | None = None,
        stage_name: str = "Mogadishu",
        seed: int = 0,
        max_episode_seconds: float = 120.0,
        items_probability: float = 0.0,
        curriculum_strength: float = 0.70,
        lesson_seconds: float = 16.0,
    ) -> None:
        if fighter_name not in FIGHTER_ORDER:
            raise ValueError(f"unknown fighter: {fighter_name}")
        opponent_name = opponent_fighter_name or fighter_name
        if opponent_name not in FIGHTER_ORDER:
            raise ValueError(f"unknown opponent fighter: {opponent_name}")
        if stage_name not in STAGE_ORDER:
            raise ValueError(f"unknown stage: {stage_name}")
        self.roster_fighter_name = fighter_name
        self.roster_opponent_name = opponent_name
        self.roster_stage_name = stage_name
        self.roster_match_config = make_training_match_config(
            fighter_name, opponent_name, stage_name
        )
        super().__init__(
            seed=seed,
            max_episode_seconds=max_episode_seconds,
            items_probability=items_probability,
            curriculum_strength=curriculum_strength,
            lesson_seconds=lesson_seconds,
        )
        self.match_config = copy.deepcopy(self.roster_match_config)
        self.action_space = spaces.Discrete(PURPOSE_COUNT)
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(ROSTER_OBSERVATION_SIZE,),
            dtype=np.float32,
        )

    def _apply_training_overrides(self, *, items_enabled: bool) -> None:
        config = copy.deepcopy(self.roster_match_config)
        self.runtime.match_config = config
        self.runtime.manifest["match"]["limit_mode"] = "stock"
        self.runtime.manifest["match"]["starting_lives"] = 3
        self.runtime.manifest["items"]["frequency"] = 5 if items_enabled else 0
        if self.runtime.stage.name != self.roster_stage_name:
            self.runtime.stage = Stage(self.runtime.manifest, self.roster_stage_name)

    def _choose_curriculum(self, options: Mapping[str, Any]) -> str:
        explicit = options.get("curriculum")
        if explicit is not None:
            return str(explicit)
        if self.fixed_curriculum is not None:
            return self.fixed_curriculum
        if self.np_random.random() >= self.curriculum_strength:
            return "duel"
        lessons = (
            "v5_navigation",
            "v5_air_chase",
            "v5_escape",
            "v5_combo",
            "roster_special",
        )
        probabilities = np.asarray([0.20, 0.25, 0.16, 0.22, 0.17], dtype=np.float64)
        return str(self.np_random.choice(lessons, p=probabilities / probabilities.sum()))

    def _setup_lesson(self) -> None:
        if self.curriculum != "roster_special":
            super()._setup_lesson()
            return
        candidates = [
            platform
            for platform in self.runtime.stage.platforms
            if not platform.moving and 180.0 <= platform.rect.w <= 420.0
        ]
        if not candidates:
            raise RuntimeError(f"{self.roster_stage_name} has no special lesson platform")
        platform = max(candidates, key=lambda item: item.rect.w)
        side = -1 if self.np_random.random() < 0.5 else 1
        spacing = float(self.np_random.uniform(90.0, min(210.0, platform.rect.w * 0.45)))
        center = float(platform.rect.centerx)
        self._place_fighter(self.agent, platform, center - side * spacing / 2)
        self._place_fighter(self.opponent, platform, center + side * spacing / 2)
        self.agent.facing = side
        self.opponent.facing = -side
        if self.np_random.random() < 0.35:
            self.opponent.pos.y -= float(self.np_random.uniform(35.0, 85.0))
            self.opponent.prev_pos.update(self.opponent.pos)
            self.opponent.on_ground = False
            self.opponent.ground_platform = None
            self.opponent.yinc = 0.0

    def _purpose_lesson_succeeded(self) -> bool:
        if self.curriculum == "roster_special":
            return any(
                self._successful_attacks[label] > 0
                for label in ("specialGround", "specialAir", "specialUp")
            )
        return super()._purpose_lesson_succeeded()

    def _action_mask_for_slot(self, slot: int) -> np.ndarray:
        curriculum = self.curriculum if slot == self.agent_slot else "duel"
        mask = roster_purpose_action_mask(
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

    def _current_observation(self, slot: int) -> np.ndarray:
        return encode_roster_observation(
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

    def _opponent_action(self) -> int:
        policy = self.opponent_policy
        expected = tuple(
            getattr(getattr(policy, "observation_space", None), "shape", ()) or ()
        )
        if expected in {(V5_OBSERVATION_SIZE,), (ROSTER_OBSERVATION_SIZE,)}:
            observation = self._delayed_observation(self.opponent_slot)
            if expected == (V5_OBSERVATION_SIZE,):
                observation = observation[:V5_OBSERVATION_SIZE]
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
        return super()._opponent_action()

    @staticmethod
    def _empty_reward_components() -> dict[str, float]:
        components = V5PeachEnv._empty_reward_components()
        components["role_special"] = 0.0
        return components

    def step(
        self, action: int | np.integer[Any] | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        attempts_before = sum(
            self._attack_counts[label]
            for label in ("specialGround", "specialAir", "specialUp")
        )
        hits_before = sum(
            self._successful_attacks[label]
            for label in ("specialGround", "specialAir", "specialUp")
        )
        recovering_before = is_offstage(self.runtime, self.agent)
        observation, reward, terminated, truncated, info = super().step(action)
        attempts_after = sum(
            self._attack_counts[label]
            for label in ("specialGround", "specialAir", "specialUp")
        )
        hits_after = sum(
            self._successful_attacks[label]
            for label in ("specialGround", "specialAir", "specialUp")
        )
        new_attempts = max(0, attempts_after - attempts_before)
        new_hits = max(0, hits_after - hits_before)
        special_reward = 0.04 * new_hits
        if new_attempts and not recovering_before:
            special_reward -= 0.006 * new_attempts
        if special_reward:
            reward += special_reward
            self._episode_reward += special_reward
            self._reward_totals["role_special"] += special_reward
        return observation, float(reward), terminated, truncated, self._info(
            str(info["outcome"])
        )

    def _quality_metrics(self) -> dict[str, float]:
        quality = super()._quality_metrics()
        attempts = sum(
            self._attack_counts[label]
            for label in ("specialGround", "specialAir", "specialUp")
        )
        hits = sum(
            self._successful_attacks[label]
            for label in ("specialGround", "specialAir", "specialUp")
        )
        minutes = max(1.0 / 60.0, self._decision_steps / 10.0 / 60.0)
        quality.update(
            {
                "role_special_accuracy": hits / max(1, attempts),
                "role_specials_per_minute": attempts / minutes,
                "role_special_attempts": float(attempts),
                "role_special_hits": float(hits),
            }
        )
        return quality

    def _info(self, outcome: str) -> dict[str, Any]:
        info = super()._info(outcome)
        quality = self._quality_metrics()
        info.update(
            {
                "observation_version": ROSTER_OBSERVATION_VERSION,
                "fighter_name": self.roster_fighter_name,
                "opponent_fighter_name": self.roster_opponent_name,
                "stage_name": self.roster_stage_name,
                "role_purpose_labels": role_purpose_labels(self.agent),
                "quality": quality,
                **quality,
            }
        )
        return info
