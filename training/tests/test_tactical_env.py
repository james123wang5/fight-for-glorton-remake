from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import pygame
from gymnasium.utils.env_checker import check_env

from training.tactical_deployment import TacticalTrainedAIController
from training.tactical_env import (
    TACTICAL_OBSERVATION_SIZE,
    TacticalPeachEnv,
    enemy_projectile_threats,
    environmental_combat_mask,
    rocket_opportunity,
)


class FakeTacticalPolicy:
    observation_space = SimpleNamespace(shape=(TACTICAL_OBSERVATION_SIZE,))

    def __init__(self, action: tuple[int, int] = (0, 0)) -> None:
        self.action = np.asarray(action, dtype=np.int64)
        self.observation: np.ndarray | None = None
        self.mask: np.ndarray | None = None

    def predict(
        self,
        observation: np.ndarray,
        *,
        action_masks: np.ndarray,
        deterministic: bool = False,
    ):
        self.observation = observation.copy()
        self.mask = action_masks.copy()
        return self.action.copy(), None


class TacticalPeachEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = TacticalPeachEnv(
            seed=31,
            max_episode_seconds=2,
            items_probability=0,
            curriculum_strength=0,
        )

    def tearDown(self) -> None:
        self.env.close()

    def test_v3_contract_and_masked_step(self) -> None:
        observation, info = self.env.reset(
            seed=32,
            options={"curriculum": "duel", "items_enabled": False, "agent_slot": 0},
        )
        mask = self.env.action_masks()
        self.assertEqual(observation.shape, (TACTICAL_OBSERVATION_SIZE,))
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertEqual(mask.shape, (13,))
        self.assertTrue(mask[0])
        self.assertTrue(mask[4])
        observation, reward, *_ = self.env.step(np.asarray([0, 0]))
        self.assertEqual(observation.shape, (TACTICAL_OBSERVATION_SIZE,))
        self.assertIsInstance(reward, float)
        self.assertEqual(info["curriculum"], "duel")

    def test_gymnasium_contract(self) -> None:
        check_env(self.env, skip_render_check=True)

    def test_gun_and_rocket_require_facing_and_real_intercept(self) -> None:
        self.env.reset(seed=33, options={"curriculum": "aim_static", "agent_slot": 0})
        self.env.agent.facing = 1
        mask = environmental_combat_mask(self.env.runtime, self.env.agent, self.env.opponent)
        self.assertTrue(mask[6])
        self.assertFalse(mask[7])
        self.env.agent.facing = -1
        reverse_mask = environmental_combat_mask(
            self.env.runtime, self.env.agent, self.env.opponent
        )
        self.assertFalse(reverse_mask[6])
        self.assertFalse(reverse_mask[7])

        self.env.reset(seed=34, options={"curriculum": "rocket", "agent_slot": 0})
        self.assertTrue(rocket_opportunity(self.env.agent, self.env.opponent))
        self.assertTrue(environmental_combat_mask(self.env.runtime, self.env.agent, self.env.opponent)[7])

    def test_threat_features_ignore_own_projectile_and_predict_intercept(self) -> None:
        self.env.reset(seed=35, options={"curriculum": "duel", "agent_slot": 0})
        self.env.agent.pos.update(0, 0)
        own = SimpleNamespace(
            pos=pygame.Vector2(20, 0),
            xinc=-10,
            yinc=0,
            sender=self.env.agent,
            alive=True,
        )
        enemy = SimpleNamespace(
            pos=pygame.Vector2(100, 0),
            xinc=-10,
            yinc=0,
            sender=self.env.opponent,
            alive=True,
        )
        self.env.runtime.bullets = [own, enemy]
        threats = enemy_projectile_threats(self.env.runtime, self.env.agent)
        self.assertEqual(len(threats), 1)
        self.assertTrue(threats[0]["approaching"])
        self.assertAlmostEqual(float(threats[0]["time_ticks"]), 10.0)
        self.env.runtime.bullets = []

    def test_false_shield_request_is_measured_and_penalized(self) -> None:
        self.env.reset(seed=36, options={"curriculum": "duel", "agent_slot": 0})
        _obs, _reward, _terminated, _truncated, info = self.env.step(np.asarray([0, 8]))
        self.assertEqual(info["shield_metrics"]["activations"], 1)
        self.assertEqual(info["shield_metrics"]["false_activations"], 1)
        self.assertLess(info["reward_components"]["shield_discipline"], 0)

    def test_deployment_uses_same_observation_mask_and_four_tick_combo(self) -> None:
        self.env.reset(seed=37, options={"curriculum": "rocket", "agent_slot": 0})
        fake = FakeTacticalPolicy((0, 7))
        controller = TacticalTrainedAIController(
            self.env.runtime,
            self.env.agent,
            self.env.runtime.stage,
            "unused.zip",
            level=22,
            model=fake,
        )
        self.env.runtime.ai_controllers = {self.env.agent_slot: controller}
        for _ in range(4):
            self.env.simulation.step_fast([{}, {}])
        self.assertIsNotNone(fake.observation)
        self.assertEqual(fake.observation.shape, (TACTICAL_OBSERVATION_SIZE,))
        self.assertEqual(fake.mask.shape, (13,))
        self.assertEqual(self.env.agent.current_attack, "specialUp")


if __name__ == "__main__":
    unittest.main()
