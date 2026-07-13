from __future__ import annotations

import json
from pathlib import Path

import pygame


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "assets/manifests/glorton_manifest.json"


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit("Manifest missing. Run: python tools/build_manifest.py")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def world_to_screen(rect: dict, offset: pygame.Vector2, zoom: float) -> pygame.Rect:
    return pygame.Rect(
        int((rect["x"] + offset.x) * zoom),
        int((rect["y"] + offset.y) * zoom),
        max(1, int(rect["w"] * zoom)),
        max(1, int(rect["h"] * zoom)),
    )


def draw_text(surface: pygame.Surface, font: pygame.font.Font, text: str, pos: tuple[int, int], color=(230, 230, 230)) -> None:
    surface.blit(font.render(text, True, color), pos)


def main() -> None:
    manifest = load_manifest()
    pygame.init()
    screen = pygame.display.set_mode((1280, 760), pygame.RESIZABLE)
    pygame.display.set_caption("Glorton Remake - FFDec Visual Check")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo", 14)
    title_font = pygame.font.SysFont("menlo", 18, bold=True)

    bg = pygame.image.load(str(ROOT / manifest["stage"]["background"])).convert_alpha()
    peach_frames = [pygame.image.load(str(ROOT / f["image"])).convert_alpha() for f in manifest["fighter"]["frames"]]
    bullet = pygame.image.load(str(ROOT / manifest["projectiles"]["Bullet"]["image"])).convert_alpha()

    frame_index = 0
    show_objects = True
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_RIGHT:
                    frame_index = (frame_index + 1) % len(peach_frames)
                elif event.key == pygame.K_LEFT:
                    frame_index = (frame_index - 1) % len(peach_frames)
                elif event.key == pygame.K_o:
                    show_objects = not show_objects

        w, h = screen.get_size()
        screen.fill((18, 20, 24))

        stage = manifest["stage"]
        bounds = stage["bounds"]
        zoom = min((w - 360) / bounds["w"], (h - 70) / bounds["h"])
        zoom = max(0.35, min(1.15, zoom))
        offset = pygame.Vector2(40 - bounds["x"], 40 - bounds["y"])

        bg_pos = (int(offset.x * zoom), int(offset.y * zoom))
        bg_scaled = pygame.transform.smoothscale(bg, (int(bg.get_width() * zoom), int(bg.get_height() * zoom)))
        screen.blit(bg_scaled, bg_pos)

        pygame.draw.rect(screen, (220, 80, 80), world_to_screen(stage["bounds"], offset, zoom), 2)
        pygame.draw.rect(screen, (80, 170, 255), world_to_screen(stage["bounds_cam"], offset, zoom), 2)

        if show_objects:
            for obj in stage["objects"]:
                name = obj["name"]
                rect = obj["estimated_rect"]
                if name.startswith("Fixed"):
                    color = (255, 220, 80)
                elif name.startswith("Moving"):
                    color = (120, 255, 140)
                elif name.startswith("Spawn"):
                    color = (255, 120, 220)
                elif name.startswith("AI"):
                    color = (130, 130, 130)
                else:
                    color = (200, 200, 200)
                sr = world_to_screen(rect, offset, zoom)
                pygame.draw.rect(screen, color, sr, 2)
                draw_text(screen, font, name, (sr.x, sr.y - 14), color)

        panel_x = w - 300
        pygame.draw.rect(screen, (32, 35, 42), (panel_x, 0, 300, h))
        draw_text(screen, title_font, "FFDec Manifest Check", (panel_x + 18, 18))
        draw_text(screen, font, "Left/Right: Peach frame", (panel_x + 18, 52))
        draw_text(screen, font, "O: toggle stage objects", (panel_x + 18, 72))
        draw_text(screen, font, f"Peach frame: {frame_index + 1:02d}/{len(peach_frames)}", (panel_x + 18, 112), (255, 220, 160))

        labels = manifest["fighter"]["timeline"]["labels"]
        label = next((item["name"] for item in labels if item["frame"] == frame_index + 1), "")
        draw_text(screen, title_font, label or "(no label)", (panel_x + 18, 138), (255, 220, 160))

        frame = peach_frames[frame_index]
        scale = min(5.0, 180 / max(frame.get_width(), frame.get_height()))
        preview = pygame.transform.scale(frame, (max(1, int(frame.get_width() * scale)), max(1, int(frame.get_height() * scale))))
        screen.blit(preview, (panel_x + 80, 180))
        screen.blit(pygame.transform.scale(bullet, (44, 44)), (panel_x + 128, 380))

        fighter = manifest["fighter"]
        draw_text(screen, font, f"Weight {fighter['weight']}  Speed {fighter['speed']}", (panel_x + 18, 450))
        draw_text(screen, font, f"Move xInc {fighter['base_move_xinc']}", (panel_x + 18, 470))
        draw_text(screen, font, f"Jump yInc {fighter['jump_yinc']}", (panel_x + 18, 490))
        draw_text(screen, font, f"Gravity/tick {fighter['gravity_per_tick']}", (panel_x + 18, 510))
        draw_text(screen, font, f"Objects: {len(stage['objects'])}", (panel_x + 18, 550))
        draw_text(screen, font, "Yellow Fixed  Green Moving", (panel_x + 18, 580), (230, 230, 180))
        draw_text(screen, font, "Blue BoundsCam  Red Bounds", (panel_x + 18, 600), (180, 220, 255))

        pygame.display.flip()

    pygame.quit()
