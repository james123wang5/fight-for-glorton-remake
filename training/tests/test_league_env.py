from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.runtime import load_manifest
from training.league_deployment import LeagueTrainedAIController
from training.league_env import (
    BASE_OBSERVATION_SIZE,
    LEAGUE_OBSERVATION_SIZE,
    AttackTrial,
    PeachLeagueEnv,
)


class ConstantPolicy:
    observation_space = SimpleNamespace(shape=(LEAGUE_OBSERVATION_SIZE,))

    def __init__(self, action: tuple[int, int, int] = (0, 0, 0)) -> None:
        self.action = np.asarray(action, dtype=np.int64)
        self.last_observation: np.ndarray | None = None

    def predict(self, observation: np.ndarray, deterministic: bool = False):
        self.last_observation = observation.copy()
        return self.action.copy(), None


class LegacyConstantPolicy(ConstantPolicy):
    observation_space = SimpleNamespace(shape=(BASE_OBSERVATION_SIZE,))


class PeachLeagueEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = PeachLeagueEnv(
            seed=22,
            max_episode_seconds=2,
            items_probability=0,
            recovery_start_probability=0,
        )

    def tearDown(self) -> None:
        self.env.close()

    def test_v2_contract_randomizes_slot_without_modifying_normal_manifest(self) -> None:
        observation, info = self.env.reset(
            seed=23,
            options={"agent_slot": 1, "items_enabled": False, "swap_spawns": True},
        )
        self.assertEqual(observation.shape, (LEAGUE_OBSERVATION_SIZE,))
        self.assertTrue(self.env.observation_space.contains(observation))
        self.assertEqual(info["agent_slot"], 2)
        self.assertEqual(self.env.runtime.ai_controllers, {})
        self.assertEqual(
            [fighter.fighter_name for fighter in self.env.runtime.fighters],
            ["PeachPlayer", "PeachPlayer"],
        )
        self.assertEqual(load_manifest()["items"]["frequency"], 5)

    def test_frozen_legacy_teacher_receives_only_its_original_142_values(self) -> None:
        teacher = LegacyConstantPolicy()
        self.env.set_opponent_pool(
            primary=teacher,
            primary_weight=1,
            teacher_weight=0,
            probe_weight=0,
        )
        self.env.reset(seed=24, options={"agent_slot": 0})
        self.env.step(np.asarray([0, 0, 0]))
        self.assertIsNotNone(teacher.last_observation)
        self.assertEqual(teacher.last_observation.shape, (BASE_OBSERVATION_SIZE,))

    def test_ringout_dominates_damage_and_skill_metrics_are_reported(self) -> None:
        self.env.reset(seed=25, options={"agent_slot": 0})
        before = self.env._reward_state()
        self.env.opponent.damage_amnt += 100
        damage_reward, _ = self.env._reward(before, outcome="ongoing", terminated=False)
        self.env.opponent.damage_amnt = float(before["opponent_damage"])
        self.env.opponent.lives -= 1
        ringout_reward, _ = self.env._reward(before, outcome="ongoing", terminated=False)
        self.assertAlmostEqual(damage_reward, 0.02)
        self.assertAlmostEqual(ringout_reward, 2.0)
        self.assertGreater(ringout_reward, damage_reward * 50)
        self.assertIn("successful_attacks", self.env._info("ongoing"))

    def test_v2_deployment_injects_combination_before_fighter_tick(self) -> None:
        self.env.reset(seed=26, options={"agent_slot": 0})
        self.env.agent.spec_up_ok = True
        fake = ConstantPolicy((0, 1, 2))
        controller = LeagueTrainedAIController(
            self.env.runtime,
            self.env.agent,
            self.env.runtime.stage,
            "unused.zip",
            level=22,
            model=fake,
            reaction_delay_decisions=2,
        )
        self.env.runtime.ai_controllers = {0: controller}
        self.env.simulation.step_fast([{}, {}])
        self.assertIsNotNone(fake.last_observation)
        self.assertEqual(fake.last_observation.shape, (LEAGUE_OBSERVATION_SIZE,))
        self.assertEqual(self.env.agent.current_attack, "specialUp")

    def test_actual_back_throw_replaces_the_opening_punch_trial(self) -> None:
        self.env.reset(seed=27, options={"agent_slot": 0})
        self.env.agent.pos.update(100, 100)
        self.env.opponent.pos.update(90, 100)
        self.env.agent.facing = 1
        self.env._previous_attack = "punchGround"
        self.env._active_melee = AttackTrial("punchGround", 0, self.env._attack_context())
        self.env.agent._start_back_throw(self.env.opponent)
        reward = self.env._track_attack_events(self.env._event_state())
        self.assertAlmostEqual(reward, 0.08)
        self.assertIsNone(self.env._active_melee)
        self.assertEqual(self.env._recent_specials[-1].label, "specialBackThrow")

        before = self.env._event_state()
        self.env.opponent.damage(15, self.env.agent)
        self.env.agent.current_attack = ""
        reward = self.env._track_attack_events(before)
        self.assertAlmostEqual(reward, 0.03)
        self.assertEqual(self.env._successful_attacks["specialBackThrow"], 1)


if __name__ == "__main__":
    unittest.main()
