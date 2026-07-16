from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import zipfile

from tools import install_runtime_assets, package_runtime_assets


class RuntimeAssetToolTests(unittest.TestCase):
    def test_packager_rewrites_pygbag_prefix_and_excludes_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            source = root / "app.apk"
            output = root / "runtime.zip"
            with zipfile.ZipFile(source, "w") as package:
                package.writestr(
                    "assets/assets/manifests/glorton_manifest.json", "{}"
                )
                package.writestr("assets/assets/web/frame.png", b"png")
                package.writestr("assets/assets/ai/v5_purpose_policy.npz", b"old")
                package.writestr("assets/src/runtime.py", b"ignored")

            count, raw_bytes, digest = package_runtime_assets.package(source, output)

            self.assertEqual(count, 2)
            self.assertEqual(raw_bytes, 5)
            self.assertEqual(len(digest), 64)
            with zipfile.ZipFile(output) as package:
                self.assertEqual(
                    set(package.namelist()),
                    {
                        "assets/manifests/glorton_manifest.json",
                        "assets/web/frame.png",
                    },
                )

    def test_installer_rejects_path_traversal(self) -> None:
        info = zipfile.ZipInfo("../outside.txt")
        with self.assertRaises(SystemExit):
            install_runtime_assets.validate_member(info)

    def test_installer_merges_only_valid_asset_members(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            archive = root / "runtime.zip"
            with zipfile.ZipFile(archive, "w") as package:
                package.writestr(
                    "assets/manifests/glorton_manifest.json", '{"stages": {}}'
                )
                package.writestr("assets/web/frame.png", b"png")

            with patch.object(install_runtime_assets, "ROOT", root):
                count = install_runtime_assets.install(archive)

            self.assertEqual(count, 2)
            self.assertEqual((root / "assets/web/frame.png").read_bytes(), b"png")


if __name__ == "__main__":
    unittest.main()
