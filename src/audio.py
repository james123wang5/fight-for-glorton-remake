from __future__ import annotations

import random
from pathlib import Path

import pygame


class AudioManager:
    """Original SWF sounds grouped the same way as Snd.AddSound."""

    FILES = {
        "menu_music": ("menu_music.mp3",),
        "rooftop": ("rooftop.mp3",),
        "mogadishu": ("mogadishu.mp3",),
        "b52": ("b52.mp3",),
        "space": ("space.mp3",),
        "helicopter": ("helicopter.mp3",),
        "jet_engine": ("jet_engine.wav",),
        "punch": ("punch_1.wav", "punch_2.wav", "punch_3.wav"),
        "gun": ("gun.mp3",),
        "rocket": ("rocket.wav",),
        "headshot": ("headshot.mp3",),
        "thrown": ("thrown.mp3",),
        "hit_ground": ("hit_ground.mp3", "headshot.mp3"),
        "mine_activate": ("mine_activate.mp3",),
        "thunder": ("thunder.mp3",),
        "boom": ("boom_1.mp3", "boom_2.mp3"),
        "woosh": ("woosh.mp3",),
        "fart": ("fart_1.wav", "fart_2.wav"),
        "electric": ("electric.wav",),
        "kamehameha": ("kamehameha.wav",),
        "water_splash": ("water_splash.wav",),
    }

    def __init__(self, root: Path) -> None:
        self.available = False
        self.muted = False
        self.sounds: dict[str, list[pygame.mixer.Sound]] = {}
        self.loop_channels: dict[str, pygame.mixer.Channel] = {}
        try:
            if pygame.mixer.get_init() is None:
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            pygame.mixer.set_num_channels(max(16, pygame.mixer.get_num_channels()))
            pygame.mixer.set_reserved(2)
            audio_root = root / "assets/audio/original"
            for name, filenames in self.FILES.items():
                self.sounds[name] = [pygame.mixer.Sound(str(audio_root / filename)) for filename in filenames]
            self.loop_channels = {
                "music": pygame.mixer.Channel(0),
                "ambience": pygame.mixer.Channel(1),
            }
            self.available = True
        except (pygame.error, OSError):
            self.sounds.clear()
            self.loop_channels.clear()

    def set_muted(self, muted: bool) -> None:
        self.muted = bool(muted)
        volume = 0.0 if self.muted else 1.0
        for sounds in self.sounds.values():
            for sound in sounds:
                sound.set_volume(volume)
        for channel in self.loop_channels.values():
            channel.set_volume(volume)

    def play(self, name: str) -> None:
        if not self.available or self.muted:
            return
        choices = self.sounds.get(name, [])
        if choices:
            random.choice(choices).play()

    def play_loop(self, name: str, slot: str) -> None:
        if not self.available or self.muted:
            return
        choices = self.sounds.get(name, [])
        channel = self.loop_channels.get(slot)
        if not choices or channel is None:
            return
        channel.stop()
        channel.set_volume(0.0 if self.muted else 1.0)
        channel.play(random.choice(choices), loops=-1)

    def stop(self, name: str) -> None:
        if not self.available:
            return
        for sound in self.sounds.get(name, []):
            sound.stop()

    def stop_all(self) -> None:
        if self.available:
            pygame.mixer.stop()
