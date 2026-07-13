from __future__ import annotations

import hashlib
import math
import os
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame

from src.runtime import Bullet, PeachFighter, RuntimeApp, SpecialProjectile, Stage, load_manifest


ROOT = Path(__file__).resolve().parents[1]


class FighterRosterParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))
        cls.manifest = load_manifest()

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def fighter(self, fighter_name: str, color: int = 0) -> PeachFighter:
        fighter = PeachFighter(
            self.manifest,
            pygame.Vector2(300, 300),
            "P1",
            color,
            fighter_name,
        )
        fighter.intro_visible = True
        fighter.has_control = True
        return fighter

    def advance_attack(
        self,
        fighter: PeachFighter,
        label: str,
        frame: int,
    ) -> list[SpecialProjectile]:
        projectiles: list[SpecialProjectile] = []
        fighter._animate_attack(label)
        fighter.animation_time_ms = (frame - 1) * 1000 / 30 + 0.1
        fighter._advance_current_attack(
            True,
            special_projectiles=projectiles,
            defer_finish=True,
        )
        return projectiles

    def test_source_constructor_parameters_for_all_six_fighters(self) -> None:
        expected = {
            "PeachPlayer": (0.4, 0.5, 0.6, "PeachLock"),
            "SBLPlayer": (0.5, 0.5, 0.5, "StrawberryLock"),
            "TrashPlayer": (0.6, 0.4, 0.5, "TrashLock"),
            "CoffeePlayer": (0.4, 0.7, 0.4, "CoffeeLock"),
            "DefaultPlayer": (0.5, 0.5, 0.5, "BallLock"),
            "AuberginePlayer": (0.5, 0.6, 0.4, "AubergineLock"),
        }
        self.assertEqual(set(self.manifest["fighters"]), set(expected))
        for name, (weight, speed, power, character_name) in expected.items():
            data = self.manifest["fighters"][name]
            self.assertEqual((data["weight"], data["speed"], data["power"]), (weight, speed, power))
            self.assertEqual(data["character_name"], character_name)
            self.assertEqual(data["base_move_xinc"], 8 * speed)

    def test_every_roster_action_uses_a_complete_exported_timeline(self) -> None:
        required = {
            "still",
            "run",
            "jump1",
            "jump2",
            "punchGround",
            "punchRun",
            "punchAir",
            "punchUp",
            "specialGround",
            "specialAir",
            "specialUp",
            "takingHit",
            "thrown",
            "ko",
            "spawn",
        }
        for fighter in self.manifest["fighters"].values():
            for color in range(1, 5):
                animations = fighter["color_state_animations"][str(color)]
                self.assertTrue(required.issubset(animations))
                for animation in animations.values():
                    self.assertEqual(animation["frame_count"], len(animation["frames"]))
                    self.assertTrue(all((ROOT / item["image"]).exists() for item in animation["frames"]))
            self.assertEqual(
                fighter["color_state_animations"]["1"]["run"]["frame_count"],
                18,
            )

    def test_character_projectiles_spawn_on_the_actionscript_frames(self) -> None:
        cases = (
            ("AuberginePlayer", "Pencil", 9, 7.0, 0.8),
            ("CoffeePlayer", "Poop", 9, 7.0, 0.4),
            ("TrashPlayer", "Garbage", 6, 8.0, 1.0),
            ("DefaultPlayer", "EnergyBall", 9, 5.0, 0.5),
        )
        for fighter_name, kind, frame, xinc, scale in cases:
            with self.subTest(fighter=fighter_name):
                fighter = self.fighter(fighter_name)
                spawned = self.advance_attack(fighter, "specialGround", frame)
                self.assertEqual(len(spawned), 1)
                self.assertEqual(spawned[0].kind, kind)
                self.assertEqual(spawned[0].xinc, xinc)
                self.assertEqual(spawned[0].display_scale, scale)

    def test_poop_plays_once_then_loops_source_frames_nineteen_to_twenty_eight(self) -> None:
        fighter = self.fighter("CoffeePlayer")
        projectile = self.advance_attack(fighter, "specialGround", 9)[0]
        self.assertEqual(projectile.config["playback"], {"loop_from": 19, "loop_at": 28})

        projectile.age = 27 * 1000 / 30
        self.assertEqual(projectile.frame, projectile.frames[27])
        projectile.age = 28 * 1000 / 30
        self.assertEqual(projectile.frame, projectile.frames[18])
        projectile.age = 38 * 1000 / 30
        self.assertEqual(projectile.frame, projectile.frames[18])

    def test_trash_special_up_emits_twenty_source_radial_projectiles(self) -> None:
        fighter = self.fighter("TrashPlayer")
        fighter.facing = -1
        fighter.attack_facing = -1
        spawned = self.advance_attack(fighter, "specialUp", 11)
        self.assertEqual(len(spawned), 20)
        self.assertTrue(all(projectile.kind == "Garbage" for projectile in spawned))
        self.assertTrue(all(projectile.facing == -1 for projectile in spawned))
        for projectile in spawned:
            self.assertAlmostEqual(math.hypot(projectile.xinc, projectile.yinc), 15.0)

    def test_trash_normal_garbage_uses_the_embedded_source_random_frame(self) -> None:
        observed = []
        held_visuals = []
        for source_random in range(7):
            fighter = self.fighter("TrashPlayer")
            projectiles: list[SpecialProjectile] = []
            with patch("src.runtime.random.randrange", return_value=source_random):
                fighter._animate_attack("specialGround")
            fighter.animation_time_ms = 4 * 1000 / 30 + 0.1
            held_visuals.append(
                hashlib.sha256(pygame.image.tobytes(fighter.current_image(), "RGBA")).hexdigest()
            )
            fighter.animation_time_ms = 5 * 1000 / 30 + 0.1
            fighter._advance_current_attack(
                True,
                special_projectiles=projectiles,
                defer_finish=True,
            )
            observed.append(projectiles[0].variant)

        self.assertEqual(observed, [1, 1, 2, 3, 4, 5, 6])
        self.assertEqual(held_visuals[0], held_visuals[1])
        self.assertEqual(len(set(held_visuals[1:])), 6)

    def test_special_projectile_hit_uses_its_own_source_damage_and_throw(self) -> None:
        attacker = self.fighter("AuberginePlayer")
        target = self.fighter("PeachPlayer", 1)
        projectile = self.advance_attack(attacker, "specialGround", 9)[0]
        target.pos.update(projectile.hitbox().center)
        target.prev_pos.update(target.pos)
        runtime = RuntimeApp.__new__(RuntimeApp)
        runtime.audio = None
        runtime.fighters = [attacker, target]
        runtime.special_projectiles = [projectile]
        runtime._resolve_special_projectile_hits()
        self.assertEqual(target.damage_amnt, 20)
        self.assertEqual(target.state, "thrown")
        self.assertFalse(projectile.alive)

    def test_peach_bullet_is_destroyed_by_source_platform_collision(self) -> None:
        stage = Stage(self.manifest)
        sender = self.fighter("PeachPlayer")
        platform = max(stage.platforms, key=lambda item: item.rect.w)
        projectile_data = self.manifest["projectiles"]["Bullet"]
        bullet = Bullet(
            pos=pygame.Vector2(platform.rect.centerx - 20, platform.rect.centery),
            xinc=20,
            image=pygame.image.load(str(ROOT / projectile_data["image"])).convert_alpha(),
            sender=sender,
            offset=pygame.Vector2(projectile_data["offset"]["x"], projectile_data["offset"]["y"]),
            source_scale=float(projectile_data.get("render_scale", 1)),
        )

        bullet.fixed_tick(stage)

        self.assertFalse(bullet.alive)

    def test_peach_bullet_registration_and_hitbox_use_source_fifty_percent_scale(self) -> None:
        sender = self.fighter("PeachPlayer")
        projectile_data = self.manifest["projectiles"]["Bullet"]
        image = pygame.image.load(str(ROOT / projectile_data["image"])).convert_alpha()
        bullet = Bullet(
            pos=pygame.Vector2(300, 200),
            xinc=20,
            image=image,
            sender=sender,
            offset=pygame.Vector2(projectile_data["offset"]["x"], projectile_data["offset"]["y"]),
            source_scale=float(projectile_data.get("render_scale", 1)),
        )

        hitbox = bullet.hitbox()

        self.assertAlmostEqual(hitbox.w, image.get_width() / bullet.source_scale * 0.5, delta=2)
        self.assertAlmostEqual(hitbox.h, image.get_height() / bullet.source_scale * 0.5, delta=2)

    def test_energy_ball_uses_itself_as_throw_sender_like_actionscript(self) -> None:
        attacker = self.fighter("DefaultPlayer")
        target = self.fighter("PeachPlayer", 1)
        projectile = self.advance_attack(attacker, "specialGround", 999)[0]
        target.pos.update(projectile.hitbox().center)
        target.prev_pos.update(target.pos)
        runtime = RuntimeApp.__new__(RuntimeApp)
        runtime.audio = None
        runtime.fighters = [attacker, target]
        runtime.special_projectiles = [projectile]

        runtime._resolve_special_projectile_hits()

        self.assertIs(target.last_sender, projectile)
        stage = Stage(self.manifest)
        target.lives = 1
        target.die("rig", stage)
        self.assertEqual(attacker.kos, 0)
        self.assertEqual(projectile.kos, 1)

    def test_energy_ball_calls_the_source_wall_bounce_at_world_bounds(self) -> None:
        attacker = self.fighter("DefaultPlayer")
        projectile = self.advance_attack(attacker, "specialGround", 999)[0]
        stage = Stage(self.manifest)
        projectile.pos.update(stage.bounds.right + 5, stage.bounds.centery)
        projectile.xinc = 5

        projectile.fixed_tick(stage)

        self.assertTrue(projectile.alive)
        self.assertEqual(projectile.xinc, -5)


if __name__ == "__main__":
    unittest.main()
