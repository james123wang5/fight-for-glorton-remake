#!/usr/bin/env python3
"""Build the compact desktop runtime-assets archive from an existing Pygbag APK.

The Pygbag archive already contains the dependency-pruned 1x assets used by the
web build.  Reusing that set gives desktop clones a small, complete fallback
without committing generated SWF exports to Git history.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "build/mobile/app/build/web/app.apk"
DEFAULT_OUTPUT = ROOT / "artifacts/release/glorton-runtime-assets-1x.zip"
SOURCE_PREFIX = "assets/assets/"
SKIPPED_PREFIXES = ("assets/assets/ai/",)
FIXED_ZIP_TIME = (2026, 7, 16, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package(source: Path, output: Path) -> tuple[int, int, str]:
    if not source.is_file():
        raise SystemExit(f"Pygbag archive not found: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    raw_bytes = 0
    with zipfile.ZipFile(source) as incoming, zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as outgoing:
        names = sorted(
            info.filename
            for info in incoming.infolist()
            if not info.is_dir()
            and info.filename.startswith(SOURCE_PREFIX)
            and not info.filename.startswith(SKIPPED_PREFIXES)
        )
        if not names:
            raise SystemExit(f"No runtime assets found inside: {source}")

        for source_name in names:
            relative = PurePosixPath(source_name.removeprefix(SOURCE_PREFIX))
            destination = PurePosixPath("assets") / relative
            data = incoming.read(source_name)
            info = zipfile.ZipInfo(str(destination), FIXED_ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            outgoing.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
            count += 1
            raw_bytes += len(data)

    digest = sha256(output)
    sidecar = output.with_name(f"{output.name}.sha256")
    sidecar.write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return count, raw_bytes, digest


def main() -> None:
    parser = argparse.ArgumentParser(description="打包可迁移的 1× 桌面运行素材")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    count, raw_bytes, digest = package(args.source.resolve(), args.output.resolve())
    print(f"Packed {count:,} files ({raw_bytes / 1024 / 1024:.1f} MiB raw)")
    print(f"Archive: {args.output.resolve()}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
