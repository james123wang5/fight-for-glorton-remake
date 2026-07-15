from __future__ import annotations

import hashlib
import unittest
from types import SimpleNamespace

import numpy as np

from src.runtime import load_manifest
from training.deployment import TrainedAIController
from training.peach_env import PeachVsLevel20Env, encode_runtime_observation


class PeachTrainingEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = PeachVsLevel20Env(seed=123, max_episode_seconds=5)

    def tearDown(self) -> None:
        self.env.close()

    def test_scenario_is_fixed_and_normal_manifest_is_untouched(self) -> None:
        observation, info = self.env.reset(seed=123, options={"swap_spawns": False})
        self.assertEqual(observation.shape, self.env.observation_space.shape)
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertEqual(info["stage"], "Mogadishu")
        self.assertFalse(info["items_enabled"])
        self.assertEqual([fighter.fighter_name for fighter in self.env.runtime.fighters], ["PeachPlayer", "PeachPlayer"])
        self.assertEqual([fighter.lives for fighter in self.env.runtime.fighters], [3, 3])
        self.assertEqual(self.env.runtime.ai_controllers[1].level, 20)
        self.assertEqual(self.env.runtime.manifest["items"]["frequency"], 0)
        fresh_manifest = load_manifest()
        self.assertEqual(fresh_manifest["items"]["frequency"], 5)
        self.assertEqual(fresh_manifest["match"]["starting_lives"], 5)

    def test_action_contract_emits_source_style_combinations(self) -> None:
        uppercut = self.env._controls_for_action(0, 1, 1, press_tick=True)
        rocket = self.env._controls_for_action(0, 1, 2, press_tick=True)
        jump = self.env._controls_for_action(0, 1, 0, press_tick=True)
        self.assertTrue(uppercut["up_trace"] and uppercut["punch_pressed"])
        self.assertTrue(rocket["up_trace"] and rocket["special_pressed"])
        self.assertTrue(jump["jump_pressed"])
        self.assertFalse(jump["up_trace"])

    def test_ringout_reward_is_much_larger_than_one_hundred_damage(self) -> None:
        self.env.reset(seed=321, options={"swap_spawns": False})
        before = self.env._reward_state()
        self.env.opponent.damage_amnt += 100
        damage_reward, _ = self.env._reward(before, outcome="ongoing", terminated=False)
        self.env.opponent.damage_amnt = float(before["opponent_damage"])
        self.env.opponent.lives -= 1
        ringout_reward, _ = self.env._reward(before, outcome="ongoing", terminated=False)
        self.assertAlmostEqual(damage_reward, 0.05)
        self.assertAlmostEqual(ringout_reward, 1.0)
        self.assertGreater(ringout_reward, damage_reward * 10)

    def test_reset_and_step_are_seed_deterministic(self) -> None:
        actions = [
            np.asarray([2, 0, 0]),
            np.asarray([2, 1, 1]),
            np.asarray([0, 0, 2]),
            np.asarray([1, 0, 3]),
        ]
        first, _ = self.env.reset(seed=999, options={"swap_spawns": True})
        first_observations = [first]
        for action in actions:
            observation, *_ = self.env.step(action)
            first_observations.append(observation)
        second, _ = self.env.reset(seed=999, options={"swap_spawns": True})
        second_observations = [second]
        for action in actions:
            observation, *_ = self.env.step(action)
            second_observations.append(observation)
        for left, right in zip(first_observations, second_observations, strict=True):
            np.testing.assert_array_equal(left, right)

    def test_deployment_encoder_keeps_the_trained_model_contract(self) -> None:
        observation, _ = self.env.reset(seed=999, options={"swap_spawns": True})
        deployed = encode_runtime_observation(
            self.env.runtime,
            self.env.agent,
            self.env.opponent,
            episode_ticks=0,
            max_ticks=self.env.max_ticks,
            spawns_swapped=True,
        )
        np.testing.assert_array_equal(deployed, observation)
        self.assertEqual(
            hashlib.sha256(deployed.tobytes()).hexdigest(),
            "42ef18c8a245aa65897a55edd7e2fab4415e7c2e8e4725c069e8bd8135f86362",
        )

    def test_level21_controller_injects_model_actions_before_fighter_tick(self) -> None:
        class FakeModel:
            observation_space = SimpleNamespace(shape=(142,))

            def __init__(self) -> None:
                self.observation: np.ndarray | None = None

            def predict(self, observation: np.ndarray, deterministic: bool = True):
                self.observation = observation
                return np.asarray([0, 1, 1]), None

        self.env.reset(seed=111, options={"swap_spawns": False})
        fake = FakeModel()
        controller = TrainedAIController(
            self.env.runtime,
            self.env.agent,
            self.env.runtime.stage,
            "unused.zip",
            model=fake,
        )
        self.env.runtime.ai_controllers = {0: controller}
        self.env.simulation.step_fast([{}, {}])
        self.assertIsNotNone(fake.observation)
        self.assertEqual(fake.observation.shape, (142,))
        self.assertEqual(self.env.agent.current_attack, "punchUp")

    def test_short_episode_steps_without_rendering_or_items(self) -> None:
        self.env.reset(seed=555, options={"swap_spawns": False})
        for _ in range(10):
            observation, reward, terminated, truncated, info = self.env.step(
                np.asarray([2, 0, 1])
            )
            self.assertTrue(self.env.observation_space.contains(observation))
            self.assertIsInstance(reward, float)
            self.assertEqual(self.env.runtime.items, [])
            if terminated or truncated:
                break
        self.assertEqual(info["stage"], "Mogadishu")


if __name__ == "__main__":
    unittest.main()
