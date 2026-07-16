from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from training.tactical_input import TacticalInputAdapter
from training.v4_deployment import V4TrainedAIController
from training.v4_env import (
    V4_OBSERVATION_SIZE,
    V4PeachEnv,
    route_distance,
    v4_action_mask,
    v4_combat_mask,
    wall_probe,
)


class FakeV4Policy:
    observation_space = SimpleNamespace(shape=(V4_OBSERVATION_SIZE,))

    def __init__(self) -> None:
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
        return np.asarray([1, 0], dtype=np.int64), None


class V4EnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = V4PeachEnv(seed=41, max_episode_seconds=3, curriculum_strength=0.0)
        self.observation, _info = self.env.reset(
            seed=41,
            options={"curriculum": "duel", "items_enabled": False, "agent_slot": 0},
        )

    def tearDown(self) -> None:
        self.env.close()

    def test_observation_and_action_contracts_are_new_and_maskable(self) -> None:
        self.assertEqual(self.observation.shape, (V4_OBSERVATION_SIZE,))
        self.assertEqual(self.env.action_masks().shape, (13,))

    def test_far_safe_fighter_must_approach_instead_of_stop_or_retreat(self) -> None:
        agent = self.env.agent
        opponent = self.env.opponent
        opponent.pos.x = agent.pos.x + 400
        opponent.pos.y = agent.pos.y
        agent.on_ground = True
        mask = v4_action_mask(
            self.env.runtime,
            agent,
            opponent,
            self.env.adapters[self.env.agent_slot],
        )
        self.assertFalse(mask[0])
        self.assertTrue(mask[1])
        self.assertFalse(mask[2])

    def test_ground_crouch_is_illegal_on_fixed_floor_but_moving_drop_is_preserved(self) -> None:
        agent = self.env.agent
        opponent = self.env.opponent
        fixed = next(item for item in self.env.runtime.stage.platforms if item.name == "Fixed1")
        moving = next(item for item in self.env.runtime.stage.platforms if item.name == "Moving1")
        agent.on_ground = True
        agent.ground_platform = fixed
        opponent.pos.y = agent.pos.y + 80
        fixed_mask = v4_combat_mask(
            self.env.runtime, agent, opponent, self.env.adapters[self.env.agent_slot]
        )
        self.assertFalse(fixed_mask[2])

        agent.ground_platform = moving
        moving_mask = v4_combat_mask(
            self.env.runtime, agent, opponent, self.env.adapters[self.env.agent_slot]
        )
        self.assertTrue(moving_mask[2])

    def test_shield_requires_a_real_incoming_threat(self) -> None:
        agent = self.env.agent
        opponent = self.env.opponent
        quiet = v4_combat_mask(
            self.env.runtime, agent, opponent, self.env.adapters[self.env.agent_slot]
        )
        self.assertFalse(quiet[8])

        projectile = SimpleNamespace(
            pos=agent.pos + (100, 0),
            xinc=-10.0,
            yinc=0.0,
            sender=opponent,
            alive=True,
        )
        self.env.runtime.bullets.append(projectile)
        threatened = v4_combat_mask(
            self.env.runtime, agent, opponent, self.env.adapters[self.env.agent_slot]
        )
        self.assertTrue(threatened[8])

    def test_close_range_uses_melee_and_masks_the_gun(self) -> None:
        agent = self.env.agent
        opponent = self.env.opponent
        opponent.pos.update(agent.pos.x + 45, agent.pos.y)
        agent.facing = 1
        mask = v4_combat_mask(
            self.env.runtime, agent, opponent, self.env.adapters[self.env.agent_slot]
        )
        self.assertTrue(mask[3])
        self.assertFalse(mask[6])

    def test_narrow_wall_is_visible_to_navigation_and_increases_route_distance(self) -> None:
        by_name = {item.name: item for item in self.env.runtime.stage.platforms}
        base = by_name["Fixed1"]
        wall = by_name["Fixed12"]
        self.env._place_fighter(self.env.agent, base, wall.rect.left - 35)
        self.env._place_fighter(self.env.opponent, base, wall.rect.right + 80)
        probe = wall_probe(self.env.runtime, self.env.agent, 1)
        direct = self.env.agent.pos.distance_to(self.env.opponent.pos)
        self.assertTrue(probe["blocked"])
        self.assertEqual(probe["platform"].name, "Fixed12")
        self.assertGreater(route_distance(self.env.runtime, self.env.agent, self.env.opponent), direct)

    def test_one_shield_activation_cannot_count_the_same_block_twice(self) -> None:
        before = self.env._event_state()
        self.env.agent.shielded = True
        self.env.agent.shield_size = float(before["agent_shield"]) - 5
        self.env._track_shield(before, threat_before=1.0)
        self.env.agent.shield_size -= 5
        self.env._track_shield(before, threat_before=1.0)
        self.assertEqual(self.env._shield_metrics["blocks"], 1)

    def test_timeout_is_a_loss_of_training_value_not_a_damage_based_win(self) -> None:
        before = self.env._reward_state()
        reward, components = self.env._reward(
            before,
            outcome="timeout_win",
            terminated=False,
        )
        self.assertEqual(components["timeout"], -1.0)
        self.assertLess(reward, 0.0)

    def test_masked_idle_request_cannot_turn_into_ground_crouch(self) -> None:
        for _ in range(8):
            action = np.asarray([1, 2], dtype=np.int64)
            self.env.step(action)
        self.assertEqual(self.env._ground_crouch_decisions, 0)

    def test_adapter_releases_shield_after_five_decisions_and_rearms_gun(self) -> None:
        adapter = TacticalInputAdapter(
            frame_skip=4,
            shield_min_hold_decisions=1,
            shield_max_hold_decisions=5,
            shoot_rearm_decisions=8,
        )
        fighter = SimpleNamespace(shielded=False, pos=SimpleNamespace(x=0.0))
        opponent = SimpleNamespace(pos=SimpleNamespace(x=100.0))
        mask = np.ones(13, dtype=bool)
        start = adapter.begin_decision(
            np.asarray([0, 8]), fighter=fighter, opponent=opponent, action_mask=mask
        )
        self.assertTrue(start[0]["shield_pressed"])
        fighter.shielded = True
        held = []
        for _ in range(5):
            held.append(
                adapter.begin_decision(
                    np.asarray([0, 8]), fighter=fighter, opponent=opponent, action_mask=mask
                )
            )
        self.assertTrue(held[-1][0]["shield_released"])

        fighter.shielded = False
        adapter.begin_decision(
            np.asarray([0, 6]), fighter=fighter, opponent=opponent, action_mask=mask
        )
        self.assertEqual(adapter.shoot_rearm, 8)

    def test_deployment_uses_v4_observation_and_mask(self) -> None:
        fake = FakeV4Policy()
        controller = V4TrainedAIController(
            self.env.runtime,
            self.env.agent,
            self.env.runtime.stage,
            "unused.zip",
            level=22,
            model=fake,
        )
        controller.controls_for_tick(self.env.runtime.fighters)
        self.assertIsNotNone(fake.observation)
        self.assertEqual(fake.observation.shape, (V4_OBSERVATION_SIZE,))
        self.assertEqual(fake.mask.shape, (13,))


if __name__ == "__main__":
    unittest.main()
