from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path

import pygame


class SurfaceCache:
    """Byte-bounded cache for source PNGs decoded by Pygame."""

    def __init__(self, max_bytes: int = 160 * 1024 * 1024) -> None:
        self.max_bytes = max(1, int(max_bytes))
        self.current_bytes = 0
        self._surfaces: OrderedDict[Path, tuple[pygame.Surface, int]] = OrderedDict()

    @staticmethod
    def surface_bytes(surface: pygame.Surface) -> int:
        return surface.get_pitch() * surface.get_height()

    def get(self, path: str | Path) -> pygame.Surface:
        key = Path(path)
        if not key.is_absolute():
            key = key.resolve()
        cached = self._surfaces.pop(key, None)
        if cached is not None:
            self._surfaces[key] = cached
            return cached[0]

        surface = pygame.image.load(str(key))
        size = self.surface_bytes(surface)
        while self._surfaces and self.current_bytes + size > self.max_bytes:
            _, (_, evicted_size) = self._surfaces.popitem(last=False)
            self.current_bytes -= evicted_size
        self._surfaces[key] = (surface, size)
        self.current_bytes += size
        return surface

    def clear(self) -> None:
        self._surfaces.clear()
        self.current_bytes = 0

    def __len__(self) -> int:
        return len(self._surfaces)


SURFACE_CACHE = SurfaceCache()


class LazySurfaceSequence(Sequence[pygame.Surface]):
    def __init__(self, paths: Iterable[str | Path]) -> None:
        self.paths = tuple(Path(path) for path in paths)

    def __getitem__(self, index: int | slice) -> pygame.Surface | list[pygame.Surface]:
        if isinstance(index, slice):
            return [SURFACE_CACHE.get(path) for path in self.paths[index]]
        return SURFACE_CACHE.get(self.paths[index])

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[pygame.Surface]:
        for path in self.paths:
            yield SURFACE_CACHE.get(path)


class LazySurfaceMap(Mapping[int, pygame.Surface]):
    def __init__(self, paths: Mapping[int, str | Path]) -> None:
        self.paths = {int(key): Path(path) for key, path in paths.items()}

    def __getitem__(self, key: int) -> pygame.Surface:
        return SURFACE_CACHE.get(self.paths[key])

    def __iter__(self) -> Iterator[int]:
        return iter(self.paths)

    def __len__(self) -> int:
        return len(self.paths)
