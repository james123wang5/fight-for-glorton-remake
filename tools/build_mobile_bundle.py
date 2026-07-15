from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE_MANIFEST = ROOT / "assets" / "manifests" / "glorton_manifest.json"
BUILD_ROOT = ROOT / "build" / "mobile"
APP_ROOT = BUILD_ROOT / "app"
WEB_ROOT = APP_ROOT / "build" / "web"
PRESERVED_TREES = ("menu", "audio", "fonts")


def _scaled_png(path: Path, *, asset_scale: int) -> bytes:
    if asset_scale == 4:
        return path.read_bytes()
    with Image.open(path) as image:
        target = (
            max(1, round(image.width * asset_scale / 4)),
            max(1, round(image.height * asset_scale / 4)),
        )
        if image.size != target:
            image = image.resize(target, Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True, compress_level=9)
        return output.getvalue()


def _copy_preserved_tree(name: str, *, asset_scale: int) -> tuple[int, int]:
    source = ROOT / "assets" / name
    target = APP_ROOT / "assets" / name
    count = 0
    total = 0
    for path in source.rglob("*"):
        if not path.is_file() or path.name == ".DS_Store":
            continue
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = (
            _scaled_png(path, asset_scale=asset_scale)
            if path.suffix.lower() == ".png"
            else path.read_bytes()
        )
        destination.write_bytes(data)
        count += 1
        total += len(data)
    return count, total


def _prepare_manifest_assets(manifest: dict[str, Any], *, asset_scale: int) -> tuple[int, int]:
    by_source: dict[str, str] = {}
    by_digest: dict[tuple[str, str], str] = {}
    total_bytes = 0

    def rewrite(value: Any) -> Any:
        nonlocal total_bytes
        if isinstance(value, dict):
            rewritten = {key: rewrite(item) for key, item in value.items()}
            if "render_scale" in rewritten:
                try:
                    source_scale = float(rewritten["render_scale"])
                except (TypeError, ValueError):
                    pass
                else:
                    if source_scale == 4.0:
                        rewritten["render_scale"] = asset_scale
            return rewritten
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if not isinstance(value, str) or not value.startswith("assets/"):
            return value
        source = ROOT / value
        if not source.is_file():
            # The sponsor intro stores a directory and is copied verbatim.
            return value
        if any(value.startswith(f"assets/{name}/") for name in PRESERVED_TREES):
            return value
        cached = by_source.get(value)
        if cached is not None:
            return cached
        data = (
            _scaled_png(source, asset_scale=asset_scale)
            if source.suffix.lower() == ".png"
            else source.read_bytes()
        )
        suffix = source.suffix.lower() or ".bin"
        digest = hashlib.sha256(data).hexdigest()
        key = (digest, suffix)
        relative = by_digest.get(key)
        if relative is None:
            relative = f"assets/web/{digest[:24]}{suffix}"
            destination = APP_ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            by_digest[key] = relative
            total_bytes += len(data)
        by_source[value] = relative
        return relative

    rewritten = rewrite(manifest)
    manifest_path = APP_ROOT / "assets" / "manifests" / "glorton_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(rewritten, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return len(by_digest), total_bytes


def prepare(*, asset_scale: int) -> dict[str, int]:
    if APP_ROOT.exists():
        shutil.rmtree(APP_ROOT)
    APP_ROOT.mkdir(parents=True)
    shutil.copy2(ROOT / "mobile" / "main.py", APP_ROOT / "main.py")
    main_path = APP_ROOT / "main.py"
    main_source = main_path.read_text(encoding="utf-8").replace(
        'os.environ["GLORTON_ASSET_SCALE"] = "2"',
        f'os.environ["GLORTON_ASSET_SCALE"] = "{asset_scale}"',
        1,
    )
    main_path.write_text(main_source, encoding="utf-8")
    shutil.copytree(
        ROOT / "src",
        APP_ROOT / "src",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"),
    )

    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    hashed_files, hashed_bytes = _prepare_manifest_assets(manifest, asset_scale=asset_scale)
    preserved_files = 0
    preserved_bytes = 0
    for tree in PRESERVED_TREES:
        count, size = _copy_preserved_tree(tree, asset_scale=asset_scale)
        preserved_files += count
        preserved_bytes += size
    report = {
        "asset_scale": asset_scale,
        "hashed_files": hashed_files,
        "hashed_bytes": hashed_bytes,
        "preserved_files": preserved_files,
        "preserved_bytes": preserved_bytes,
    }
    (BUILD_ROOT / "bundle_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def postprocess_web_output() -> str:
    index_path = WEB_ROOT / "index.html"
    apk_path = WEB_ROOT / "app.apk"
    if not index_path.is_file() or not apk_path.is_file():
        raise SystemExit("Packaged web output missing. Run the mobile builder without --prepare-only first.")

    digest = hashlib.sha256()
    with apk_path.open("rb") as apk_file:
        for chunk in iter(lambda: apk_file.read(1024 * 1024), b""):
            digest.update(chunk)
    build_id = digest.hexdigest()[:12]
    shutil.copy2(ROOT / "mobile" / "manifest.webmanifest", WEB_ROOT / "manifest.webmanifest")
    service_worker = (ROOT / "mobile" / "sw.js").read_text(encoding="utf-8").replace(
        "__BUILD_ID__", build_id
    )
    (WEB_ROOT / "sw.js").write_text(service_worker, encoding="utf-8")
    shutil.copy2(ROOT / "assets" / "ui" / "osd_bigicon" / "002.png", WEB_ROOT / "apple-touch-icon.png")

    html = index_path.read_text(encoding="utf-8")
    if 'name="glorton-mobile-build"' in html:
        return build_id
    html = html.replace('<html lang="en-us">', '<html lang="zh-CN">', 1)
    html = html.replace(
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        '    <meta name="viewport" content="height=device-height, initial-scale=1.0">',
        '    <meta name="viewport" content="width=device-width,height=device-height,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">',
        1,
    )
    head = f"""    <meta name="glorton-mobile-build" content="{build_id}">
    <meta name="theme-color" content="#000000">
    <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
    <meta name="apple-mobile-web-app-title" content="Glorton">
    <link rel="manifest" href="manifest.webmanifest">
    <link rel="apple-touch-icon" href="apple-touch-icon.png">
"""
    html = html.replace("    <title>The Fight for Glorton</title>\n", "    <title>The Fight for Glorton</title>\n" + head, 1)
    mobile_css = """
        html, body {
            position: fixed;
            inset: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            overscroll-behavior: none;
            touch-action: none;
            user-select: none;
            -webkit-user-select: none;
            -webkit-touch-callout: none;
            background: #000 !important;
        }
        canvas.emscripten {
            position: fixed !important;
            top: env(safe-area-inset-top) !important;
            right: env(safe-area-inset-right) !important;
            bottom: env(safe-area-inset-bottom) !important;
            left: env(safe-area-inset-left) !important;
            width: calc(100vw - env(safe-area-inset-left) - env(safe-area-inset-right)) !important;
            height: calc(100vh - env(safe-area-inset-top) - env(safe-area-inset-bottom)) !important;
            margin: 0 !important;
            touch-action: none;
            background: #000 !important;
        }
        #rotate-device {
            display: none;
            position: fixed;
            inset: 0;
            z-index: 100;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            color: white;
            background: #050608;
            font: 700 20px/1.5 -apple-system, BlinkMacSystemFont, sans-serif;
            text-align: center;
        }
        @media (orientation: portrait) {
            #rotate-device { display: flex; }
            canvas.emscripten { visibility: hidden !important; }
        }
"""
    html = html.replace("    </style>", mobile_css + "    </style>", 1)
    html = html.replace("<body>\n", '<body>\n    <div id="rotate-device">请将 iPhone 横过来游玩<br>建议从 Safari “共享”中添加到主屏幕</div>\n', 1)
    mobile_script = """
    <script>
      if ("serviceWorker" in navigator) {
        window.addEventListener("load", () => navigator.serviceWorker.register("./sw.js"));
      }
      document.addEventListener("touchmove", (event) => event.preventDefault(), {passive: false});
      window.addEventListener("orientationchange", () => {
        window.setTimeout(() => {
          if (window.innerWidth > window.innerHeight) window.location.reload();
        }, 350);
      });
      window.addEventListener("pointerdown", () => {
        const root = document.documentElement;
        if (root.requestFullscreen && !document.fullscreenElement) root.requestFullscreen().catch(() => {});
        if (screen.orientation && screen.orientation.lock) screen.orientation.lock("landscape").catch(() => {});
      }, {once: true});
    </script>
"""
    html = html.replace("</body>", mobile_script + "\n</body>", 1)
    index_path.write_text(html, encoding="utf-8")
    return build_id


def package() -> str:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pygbag",
            "--build",
            "--ume_block",
            "0",
            "--title",
            "The Fight for Glorton",
            "--app_name",
            "glorton-mobile",
            str(APP_ROOT),
        ],
        cwd=ROOT,
        check=True,
    )
    return postprocess_web_output()


def cleanup_staging() -> None:
    """Keep the publishable web directory, remove its duplicate source stage."""

    for path in (APP_ROOT / "assets", APP_ROOT / "src", APP_ROOT / "main.py"):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    cache = APP_ROOT / "build" / "web-cache"
    if cache.is_dir():
        shutil.rmtree(cache)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the isolated iPhone/WebAssembly bundle without touching desktop assets"
    )
    parser.add_argument(
        "--asset-scale",
        type=int,
        choices=(1, 2, 4),
        default=2,
        help="1 is compact phone quality, 2 keeps Retina-density assets, and 4 is desktop source quality",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--postprocess-only",
        action="store_true",
        help="Add PWA metadata to an existing pygbag output without rebuilding assets",
    )
    parser.add_argument(
        "--cleanup-only",
        action="store_true",
        help="Delete duplicate staging files but retain the publishable web output",
    )
    parser.add_argument(
        "--keep-staging",
        action="store_true",
        help="Retain the unpacked asset staging tree after a successful package",
    )
    args = parser.parse_args()
    if args.cleanup_only:
        cleanup_staging()
        print("Removed duplicate mobile staging; publishable web output retained")
        return
    if args.postprocess_only:
        print(f"Postprocessed PWA build {postprocess_web_output()}")
        return
    report = prepare(asset_scale=args.asset_scale)
    mib = (report["hashed_bytes"] + report["preserved_bytes"]) / 1024 / 1024
    print(
        f"Prepared {report['hashed_files'] + report['preserved_files']} files "
        f"at {args.asset_scale}x ({mib:.1f} MiB before web packaging)"
    )
    if not args.prepare_only:
        build_id = package()
        if not args.keep_staging:
            cleanup_staging()
        print(f"Packaged PWA build {build_id}")


if __name__ == "__main__":
    main()
