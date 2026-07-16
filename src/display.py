from __future__ import annotations

import ctypes
import os
import sys
from dataclasses import dataclass

import pygame


@dataclass(frozen=True)
class DisplayMetrics:
    logical_size: tuple[int, int]
    window_size: tuple[int, int]
    pixel_ratio: float
    native_pixels: bool
    source: str


def _mac_backing_scale() -> float | None:
    if sys.platform != "darwin" or not pygame.display.get_init():
        return None
    try:
        capsule = pygame.display.get_wm_info().get("window")
        if capsule is None:
            return None
        get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
        get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        get_pointer.restype = ctypes.c_void_p
        window = get_pointer(capsule, b"window")
        if not window:
            return None
        objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        send = objc.objc_msgSend
        send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        send.restype = ctypes.c_double
        value = float(send(window, objc.sel_registerName(b"backingScaleFactor")))
        return value if 0.5 <= value <= 4.0 else None
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def detect_display_metrics(surface: pygame.Surface) -> DisplayMetrics:
    logical_size = surface.get_size()
    try:
        window_size = pygame.display.get_window_size()
    except pygame.error:
        window_size = logical_size
    override = os.environ.get("GLORTON_PIXEL_RATIO")
    if override:
        try:
            ratio = max(0.5, min(4.0, float(override)))
            return DisplayMetrics(logical_size, window_size, ratio, ratio > 1.0, "environment")
        except ValueError:
            pass
    backing = _mac_backing_scale()
    if backing is not None:
        return DisplayMetrics(logical_size, window_size, backing, backing > 1.0, "cocoa")
    ratios = [
        window_size[index] / logical_size[index]
        for index in (0, 1)
        if logical_size[index] > 0
    ]
    ratio = min(ratios, default=1.0)
    if abs(ratio - 1.0) < 0.05:
        ratio = 1.0
    return DisplayMetrics(logical_size, window_size, ratio, ratio > 1.0, "sdl")


def snap_to_device_pixel(value: float, pixel_ratio: float) -> float:
    ratio = max(1.0, float(pixel_ratio))
    return round(float(value) * ratio) / ratio


def recommended_window_size(default: tuple[int, int]) -> tuple[int, int]:
    """Return the browser canvas size requested by the isolated mobile entry.

    Desktop launches never set these variables and therefore retain the exact
    historical window size.  The web entry supplies physical canvas pixels so
    a Retina screen is crisp without asking Safari to decode the 4x originals.
    """

    try:
        width = int(os.environ.get("GLORTON_WEB_WIDTH", "0"))
        height = int(os.environ.get("GLORTON_WEB_HEIGHT", "0"))
    except ValueError:
        return default
    if width <= 0 or height <= 0:
        return default
    width, height = max(width, height), min(width, height)
    scale = min(1.0, 1920 / width, 1080 / height)
    return max(960, round(width * scale)), max(540, round(height * scale))


def aligned_aspect_rect(
    size: tuple[int, int],
    reference_size: tuple[int, int] = (600, 400),
    pixel_ratio: float = 1.0,
) -> pygame.Rect:
    width, height = size
    reference_w, reference_h = reference_size
    scale = min(width / reference_w, height / reference_h)
    draw_w = max(1, round(reference_w * scale * pixel_ratio) / pixel_ratio)
    draw_h = max(1, round(reference_h * scale * pixel_ratio) / pixel_ratio)
    left = snap_to_device_pixel((width - draw_w) / 2, pixel_ratio)
    top = snap_to_device_pixel((height - draw_h) / 2, pixel_ratio)
    return pygame.Rect(round(left), round(top), round(draw_w), round(draw_h))
