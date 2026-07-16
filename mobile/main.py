from __future__ import annotations

import asyncio
import os
import sys

# Pygbag detects browser wheels from top-level imports.  NumPy runs only the
# exported actor network; Torch, Gymnasium and SB3 remain desktop-only.
import numpy as np  # noqa: F401


os.environ["GLORTON_MOBILE"] = "1"
os.environ["GLORTON_ASSET_SCALE"] = "2"
os.environ["GLORTON_AI_V5_WEB"] = "1"
os.environ["GLORTON_AI21_MODEL"] = "assets/ai/v5_purpose_policy.npz"
os.environ["GLORTON_AI22_MODEL"] = "assets/ai/v5_purpose_policy.npz"

if sys.platform == "emscripten":
    # Pygbag exposes the browser window through its platform bridge.  Render at
    # up to 2x CSS pixels: sharp on Retina, but bounded for Safari memory use.
    import platform

    css_width = max(1, int(platform.window.innerWidth))
    css_height = max(1, int(platform.window.innerHeight))
    pixel_ratio = max(1.0, min(2.0, float(platform.window.devicePixelRatio or 1.0)))
    os.environ["GLORTON_WEB_WIDTH"] = str(round(css_width * pixel_ratio))
    os.environ["GLORTON_WEB_HEIGHT"] = str(round(css_height * pixel_ratio))
    os.environ["GLORTON_PIXEL_RATIO"] = str(pixel_ratio)

from src.assets import SURFACE_CACHE
from src.runtime import RuntimeApp


# Safari has a much tighter practical memory ceiling than desktop Python.
# Frames remain lazy; this only lowers the decoded-surface LRU for the web app.
SURFACE_CACHE.max_bytes = 96 * 1024 * 1024


async def main() -> None:
    await RuntimeApp().run_async()


asyncio.run(main())
