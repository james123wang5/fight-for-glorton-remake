from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
SWF_PATH = PROJECTS / "fight-for-glorton.swf"
FFDEC_JAR = PROJECTS / "FFDec.app/Contents/Resources/ffdec-cli.jar"
OUTPUT = ROOT / "assets/audio/original"

SOUNDS = {
    1: "rooftop.mp3",
    932: "woosh.mp3",
    933: "space.mp3",
    934: "menu_music.mp3",
    935: "kamehameha.wav",
    936: "mine_activate.mp3",
    937: "mogadishu.mp3",
    938: "b52.mp3",
    939: "headshot.mp3",
    940: "water_splash.wav",
    941: "thunder.mp3",
    942: "thrown.mp3",
    943: "punch_3.wav",
    944: "punch_2.wav",
    945: "punch_1.wav",
    946: "rocket.wav",
    947: "gun.mp3",
    948: "jet_engine.wav",
    949: "hit_ground.mp3",
    950: "helicopter.mp3",
    951: "fart_2.wav",
    952: "fart_1.wav",
    953: "electric.wav",
    954: "boom_2.mp3",
    955: "boom_1.mp3",
}


def find_java() -> Path:
    candidates = [
        Path("/Library/Internet Plug-Ins/JavaAppletPlugin.plugin/Contents/Home/bin/java"),
        Path(shutil.which("java") or ""),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit("Java was not found; FFDec CLI requires a Java runtime.")


def main() -> None:
    if not SWF_PATH.is_file() or not FFDEC_JAR.is_file():
        raise SystemExit("fight-for-glorton.swf or FFDec.app is missing beside the project folder.")
    with tempfile.TemporaryDirectory(prefix="glorton-sounds-") as temporary:
        raw = Path(temporary)
        subprocess.run(
            [
                str(find_java()),
                "-jar",
                str(FFDEC_JAR),
                "-format",
                "sound:mp3_wav",
                "-export",
                "sound",
                str(raw),
                str(SWF_PATH),
            ],
            check=True,
        )
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for sound_id, target_name in SOUNDS.items():
            matches = sorted(raw.glob(f"{sound_id}_*"))
            if len(matches) != 1:
                raise SystemExit(f"Expected one FFDec sound for character id {sound_id}, found {len(matches)}.")
            shutil.copy2(matches[0], OUTPUT / target_name)
    print(f"Exported {len(SOUNDS)} original SWF sounds to {OUTPUT}")


if __name__ == "__main__":
    main()
