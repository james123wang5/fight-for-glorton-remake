from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import Mock

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from src.audio import AudioManager
from src.runtime import RuntimeApp


ROOT = Path(__file__).resolve().parents[1]


class AudioParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        pygame.display.set_mode((1, 1))

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def test_frame_42_sound_registry_and_variants_are_complete(self) -> None:
        expected = {
            "boom": ("boom_1.mp3", "boom_2.mp3"),
            "electric": ("electric.wav",),
            "fart": ("fart_1.wav", "fart_2.wav"),
            "helicopter": ("helicopter.mp3",),
            "hit_ground": ("hit_ground.mp3", "headshot.mp3"),
            "jet_engine": ("jet_engine.wav",),
            "gun": ("gun.mp3",),
            "rocket": ("rocket.wav",),
            "punch": ("punch_1.wav", "punch_2.wav", "punch_3.wav"),
            "thrown": ("thrown.mp3",),
            "thunder": ("thunder.mp3",),
            "water_splash": ("water_splash.wav",),
            "headshot": ("headshot.mp3",),
            "woosh": ("woosh.mp3",),
            "mine_activate": ("mine_activate.mp3",),
            "kamehameha": ("kamehameha.wav",),
            "b52": ("b52.mp3",),
            "rooftop": ("rooftop.mp3",),
            "mogadishu": ("mogadishu.mp3",),
            "menu_music": ("menu_music.mp3",),
            "space": ("space.mp3",),
        }
        self.assertEqual(AudioManager.FILES, expected)

    def test_every_registered_original_sound_exists_and_decodes(self) -> None:
        audio_root = ROOT / "assets/audio/original"
        files = {filename for names in AudioManager.FILES.values() for filename in names}
        self.assertEqual(len(files), 25)
        for filename in files:
            path = audio_root / filename
            self.assertTrue(path.is_file(), path)
            self.assertGreater(pygame.mixer.Sound(str(path)).get_length(), 0.0, path)

        manager = AudioManager(ROOT)
        self.assertTrue(manager.available)
        self.assertEqual(set(manager.sounds), set(AudioManager.FILES))

    def test_muting_stops_new_playback_and_unmuting_does_not_restart_a_loop(self) -> None:
        manager = AudioManager(ROOT)
        manager.play_loop("menu_music", "music")
        self.assertTrue(manager.loop_channels["music"].get_busy())
        manager.set_muted(True)
        manager.stop_all()
        self.assertFalse(manager.loop_channels["music"].get_busy())
        manager.play_loop("menu_music", "music")
        self.assertFalse(manager.loop_channels["music"].get_busy())
        manager.set_muted(False)
        self.assertFalse(manager.loop_channels["music"].get_busy())

    def test_menu_music_starts_only_when_root_reaches_frame_42(self) -> None:
        runtime = RuntimeApp.__new__(RuntimeApp)
        runtime.audio = Mock()
        runtime.app_state = "menu"
        runtime.menu_music_started = False
        runtime.menu = Mock(sound_on=True, scene="preloader")

        for scene in ("preloader", "sponsor_intro", "opening"):
            runtime.menu.scene = scene
            runtime._sync_menu_music()
        runtime.audio.play_loop.assert_not_called()

        runtime.menu.scene = "intro"
        runtime._sync_menu_music()
        runtime.audio.play_loop.assert_called_once_with("menu_music", "music")
        runtime._sync_menu_music()
        runtime.audio.play_loop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
