from __future__ import annotations

import hashlib
import math
import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.runtime import PeachFighter, RuntimeApp, Stage, StageItem, TICK_MS, load_manifest


class PeachBattleParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))
        cls.manifest = load_manifest()
        cls.stage = Stage(cls.manifest)

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def fighter(self, name: str = "P1", color: int = 0) -> PeachFighter:
        return PeachFighter(
            self.manifest,
            self.stage.spawn_point("SpawnP1"),
            name,
            color,
        )

    def combat_runtime(self, *fighters: PeachFighter) -> RuntimeApp:
        runtime = RuntimeApp.__new__(RuntimeApp)
        runtime.fighters = list(fighters)
        runtime.items = []
        runtime.audio = None
        runtime.punch_damage_frames = []
        runtime.punch_damage_source_scale = 1.0
        runtime.hit_effects = []
        return runtime

    @staticmethod
    def surface_digest(surface: pygame.Surface) -> str:
        return hashlib.sha256(pygame.image.tobytes(surface, "RGBA")).hexdigest()

    def test_source_physics_constants(self) -> None:
        fighter = self.fighter()
        self.assertEqual(fighter.weight, 0.4)
        self.assertEqual(fighter.speed, 0.5)
        self.assertEqual(fighter.move_xinc, 4.0)
        self.assertEqual(fighter.jump_yinc, -9.0)
        self.assertEqual(fighter.gravity, 0.5)
        self.assertEqual(fighter.max_fall, 6.0)

    def test_special_up_unlocks_on_source_landing_or_damage_events(self) -> None:
        attacker = self.fighter("P1")
        target = self.fighter("P2", 1)
        self.assertFalse(attacker.spec_up_ok)

        attacker._land_on_platform(self.stage.platforms[0])
        self.assertTrue(attacker.spec_up_ok)
        attacker.spec_up_ok = False
        attacker.damage(1, target)
        self.assertTrue(attacker.spec_up_ok)

    def test_thrown_item_ignores_ground_spawn_lifetime_like_item_do_common(self) -> None:
        item = StageItem(
            kind="Grenade",
            pos=pygame.Vector2(300, 0),
            frames=[],
            life_ms=20000,
            age_ms=20000,
            state=2,
            xinc=1,
        )

        item.fixed_tick(self.stage)

        self.assertTrue(item.alive)
        self.assertEqual(item.age_ms, 20000 + TICK_MS)
        self.assertEqual(item.pos.x, 301)

    def test_unclaimed_items_use_source_airbone_artwork(self) -> None:
        runtime = RuntimeApp()
        for kind in ("Grenade", "Mine"):
            frames = runtime.item_frames[kind]
            labels = runtime.item_frame_labels[kind]
            item = StageItem(
                kind,
                pygame.Vector2(),
                frames,
                frame_labels=labels,
            )
            self.assertIs(item.display_frame(), frames[labels["airbone"] - 1])
            self.assertIsNot(item.display_frame(), frames[labels["spawn"] - 1])

    def test_run_and_second_jump_use_complete_source_timelines(self) -> None:
        fighter = self.fighter()
        self.assertEqual(fighter.animations["run"]["frame_count"], 18)
        self.assertEqual(
            fighter.animations["run"]["playback"],
            {"loop_from": 4, "loop_at": 18},
        )
        self.assertEqual(fighter.animations["jump2"]["frame_count"], 25)
        self.assertEqual(fighter.animations["jump2"]["playback"], {"stop_at": 25})
        jump_hashes = {
            self.surface_digest(frame)
            for frame in fighter.animations["jump2"]["frames"].values()
        }
        self.assertGreaterEqual(len(jump_hashes), 24)

    def test_held_item_art_is_explicit_and_returns_to_empty_after_use(self) -> None:
        fighter = self.fighter()
        fighter.current_label = "run"
        base = self.surface_digest(fighter.current_image())

        mine = StageItem("Mine", pygame.Vector2(fighter.pos), [])
        fighter.current_item = "mine"
        fighter.current_item_obj = mine
        held_mine = self.surface_digest(fighter.current_image())
        self.assertNotEqual(base, held_mine)

        fighter.on_ground = True
        fighter.xinc = fighter.move_xinc
        fighter.attack("punch", "none")
        self.assertEqual(fighter.current_attack, "punchRun")
        self.assertEqual(fighter.current_item, "")
        self.assertIsNone(fighter.current_item_obj)
        self.assertEqual(mine.state, 2)

    def test_death_respawn_preserves_source_held_item_state(self) -> None:
        fighter = self.fighter()
        item = StageItem("Grenade", pygame.Vector2(fighter.pos), [])
        item.state = 3
        item.sender = fighter
        fighter.current_item = "grenade"
        fighter.current_item_obj = item
        fighter.lives = 2

        fighter.die("bot", self.stage)

        self.assertEqual(fighter.current_item, "grenade")
        self.assertIs(fighter.current_item_obj, item)
        self.assertTrue(item.alive)

    def test_death_preserves_combo_and_last_sender_until_landing(self) -> None:
        fighter = self.fighter()
        killer = self.fighter("P2", 1)
        fighter.lives = 2
        fighter.combo = 6
        fighter.last_sender = killer

        fighter.die("bot", self.stage)

        self.assertEqual(fighter.combo, 6)
        self.assertIs(fighter.last_sender, killer)
        platform = self.stage.platforms[0]
        fighter._land_on_platform(platform)
        self.assertEqual(fighter.combo, 2)
        self.assertIsNone(fighter.last_sender)

    def test_last_sender_self_counts_as_source_ko_not_suicide(self) -> None:
        fighter = self.fighter()
        fighter.lives = 2
        fighter.last_sender = fighter

        fighter.die("bot", self.stage)

        self.assertEqual(fighter.kos, 1)
        self.assertEqual(fighter.sds, 0)
        self.assertEqual(fighter.osd_score_event, "plus")

    def test_damage_swaps_depth_only_when_attacker_started_below_target(self) -> None:
        p1 = self.fighter("P1")
        p2 = self.fighter("P2", 1)
        p1.draw_depth = 0
        p2.draw_depth = 1

        p2.damage(5, p1)
        self.assertEqual((p1.draw_depth, p2.draw_depth), (1, 0))

        p1.paralized = 0
        p2.paralized = 0
        p1.damage(5, p2)
        self.assertEqual((p1.draw_depth, p2.draw_depth), (0, 1))

    def test_time_death_overrides_self_plus_popup_with_minus_popup(self) -> None:
        fighter = self.fighter()
        fighter.limit_mode = "time"
        fighter.last_sender = fighter

        fighter.die("bot", self.stage)

        self.assertEqual(fighter.kos, 1)
        self.assertEqual(fighter.osd_score_event, "minus")

    def test_force_throw_still_respects_source_dodge_guard(self) -> None:
        fighter = self.fighter()
        fighter.shielded = True
        fighter.current_label = "dodge"
        old_velocity = pygame.Vector2(fighter.xinc, fighter.yinc)

        applied = fighter.throw_impulse(8, 45, None, force=True)

        self.assertFalse(applied)
        self.assertEqual(pygame.Vector2(fighter.xinc, fighter.yinc), old_velocity)

    def test_final_attack_frame_remains_collidable_until_resolution(self) -> None:
        fighter = self.fighter()
        fighter._animate_attack("punchGround")
        total = fighter.animations["punchGround"]["frame_count"]
        fighter.animation_time_ms = (total - 1) * 1000 / 30

        fighter._advance_current_attack(True, defer_finish=True)

        self.assertEqual(fighter.current_attack, "punchGround")
        self.assertEqual(fighter.attack_pending_finish, "punchGround")
        self.assertIsNotNone(fighter.attack_hitbox())

    def test_running_punch_uses_source_damage_and_throw_formula(self) -> None:
        attacker = self.fighter()
        target = self.fighter("P2", 1)
        attacker.pos.update(100, 100)
        target.pos.update(110, 100)
        for fighter in (attacker, target):
            fighter.prev_pos.update(fighter.pos)
            fighter.on_ground = True
        attacker.facing = 1
        attacker.xinc = attacker.move_xinc
        attacker.attack("punch", "none")
        runtime = self.combat_runtime(attacker, target)

        runtime._resolve_melee_hits(attacker)

        expected_power = 3 + 3 * attacker.weight - 3 * target.weight
        expected_power += target.damage_amnt / 100 * math.log(3) * 3
        expected_power += target.combo / 4
        self.assertEqual(target.damage_amnt, 5)
        self.assertEqual(target.combo, 3)
        self.assertEqual(target.state, "thrown")
        self.assertAlmostEqual(target.xinc, expected_power / 2**0.5, places=5)
        self.assertAlmostEqual(target.yinc, -expected_power / 2**0.5, places=5)

    def test_new_attack_can_chain_a_farther_throw_on_airborne_target(self) -> None:
        attacker = self.fighter()
        target = self.fighter("P2", 1)
        attacker.pos.update(100, 100)
        target.pos.update(110, 100)
        for fighter in (attacker, target):
            fighter.prev_pos.update(fighter.pos)
            fighter.on_ground = True
        runtime = self.combat_runtime(attacker, target)

        attacker.facing = 1
        attacker.xinc = attacker.move_xinc
        attacker.attack("punch", "none")
        runtime._resolve_melee_hits(attacker)
        first_speed = pygame.Vector2(target.xinc, target.yinc).length()

        attacker.paralized = 0
        attacker.current_attack = ""
        attacker.attack_done = True
        attacker.attack("punch", "none")
        runtime._resolve_melee_hits(attacker)
        second_speed = pygame.Vector2(target.xinc, target.yinc).length()

        # Damage uses floor(5 * Combo / 2): 5 on the first hit, then 7.
        self.assertEqual(target.damage_amnt, 12)
        self.assertGreater(second_speed, first_speed)

    def test_thrown_fighter_overlap_keeps_source_unreachable_bounce_quirk(self) -> None:
        thrown = self.fighter()
        target = self.fighter("P2", 1)
        thrown.pos.update(100, 100)
        target.pos.update(100, 100)
        thrown.state = "thrown"
        thrown.current_label = "thrown"
        thrown.xinc = 6
        target.xinc = -4
        runtime = self.combat_runtime(thrown, target)

        runtime._resolve_melee_hits(thrown)

        # Fighter P-code reads the result Array's undefined `type` member,
        # not the collision element's `objType`, so the reversal never runs.
        self.assertEqual(thrown.xinc, 6)
        self.assertEqual(target.xinc, -4)

    def test_back_throw_requires_target_behind_and_within_twenty_units(self) -> None:
        attacker = self.fighter()
        target = self.fighter("P2", 1)
        attacker.pos.update(100, 100)
        target.pos.update(85, 100)
        for fighter in (attacker, target):
            fighter.prev_pos.update(fighter.pos)
            fighter.on_ground = True
        attacker.facing = 1
        attacker.attack("punch", "none")
        runtime = self.combat_runtime(attacker, target)

        runtime._resolve_melee_hits(attacker)

        self.assertEqual(attacker.current_attack, "specialBackThrow")
        self.assertIs(attacker.throw_victim, target)
        self.assertEqual(target.damage_amnt, 0)


if __name__ == "__main__":
    unittest.main()
