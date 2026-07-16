#!/usr/bin/env python3
"""Install the compact runtime assets published with a GitHub release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ASSET_NAME = "glorton-runtime-assets-1x.zip"
DEFAULT_REPOSITORY = "james123wang5/fight-for-glorton-remake"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_from_origin() -> str | None:
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip().removesuffix(".git")
    for prefix in ("git@github.com:", "https://github.com/", "http://github.com/"):
        if value.startswith(prefix):
            candidate = value.removeprefix(prefix)
            if candidate.count("/") == 1:
                return candidate
    return None


def request_bytes(url: str, *, accept: str = "application/octet-stream") -> bytes:
    request = Request(
        url,
        headers={
            "Accept": accept,
            "User-Agent": "glorton-runtime-assets-installer/1.0",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            return response.read()
    except (HTTPError, URLError) as exc:
        raise SystemExit(f"Download failed: {url}\n{exc}") from exc


def latest_release_assets(repository: str) -> dict[str, str]:
    payload = request_bytes(
        f"https://api.github.com/repos/{repository}/releases/latest",
        accept="application/vnd.github+json",
    )
    release = json.loads(payload.decode("utf-8"))
    return {
        str(item["name"]): str(item["browser_download_url"])
        for item in release.get("assets", [])
        if item.get("name") and item.get("browser_download_url")
    }


def expected_digest(sidecar: str) -> str:
    digest = sidecar.strip().split(maxsplit=1)[0].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise SystemExit("Release SHA-256 sidecar is invalid")
    return digest


def validate_member(info: zipfile.ZipInfo) -> PurePosixPath:
    path = PurePosixPath(info.filename)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "assets":
        raise SystemExit(f"Unsafe archive member: {info.filename}")
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise SystemExit(f"Archive symlink is not allowed: {info.filename}")
    return path


def install(archive: Path) -> int:
    assets_root = ROOT / "assets"
    installed = 0
    with tempfile.TemporaryDirectory(prefix="glorton-assets-") as temp_name:
        temporary = Path(temp_name)
        with zipfile.ZipFile(archive) as package:
            for info in package.infolist():
                path = validate_member(info)
                if info.is_dir():
                    continue
                target = temporary.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(info) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                installed += 1

        staged_assets = temporary / "assets"
        if not (staged_assets / "manifests/glorton_manifest.json").is_file():
            raise SystemExit("Archive is missing assets/manifests/glorton_manifest.json")
        assets_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(staged_assets, assets_root, dirs_exist_ok=True)
    return installed


def main() -> None:
    parser = argparse.ArgumentParser(description="下载并安装 GitHub Release 的精简运行素材")
    parser.add_argument("--archive", type=Path, help="使用本地 ZIP，不访问 GitHub")
    parser.add_argument("--sha256", help="本地 ZIP 的预期 SHA-256")
    parser.add_argument("--repo", help="GitHub owner/repository")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="glorton-download-") as temp_name:
        if args.archive:
            archive = args.archive.expanduser().resolve()
            if not archive.is_file():
                raise SystemExit(f"Archive not found: {archive}")
            expected = args.sha256.lower() if args.sha256 else None
        else:
            repository = (
                args.repo
                or os.environ.get("GLORTON_GITHUB_REPO")
                or repository_from_origin()
                or DEFAULT_REPOSITORY
            )
            assets = latest_release_assets(repository)
            sidecar_name = f"{ASSET_NAME}.sha256"
            missing = [name for name in (ASSET_NAME, sidecar_name) if name not in assets]
            if missing:
                raise SystemExit(f"Latest release of {repository} is missing: {', '.join(missing)}")
            expected = expected_digest(request_bytes(assets[sidecar_name]).decode("utf-8"))
            archive = Path(temp_name) / ASSET_NAME
            archive.write_bytes(request_bytes(assets[ASSET_NAME]))

        actual = sha256(archive)
        if expected and actual != expected:
            raise SystemExit(f"SHA-256 mismatch: expected {expected}, got {actual}")
        count = install(archive)

    print(f"Installed {count:,} runtime asset files into: {ROOT / 'assets'}")
    print("Ready: python play.py")


if __name__ == "__main__":
    main()
