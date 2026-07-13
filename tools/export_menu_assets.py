from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from PIL import Image, ImageChops, ImageFilter


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
SWF_PATH = PROJECTS / "fight-for-glorton.swf"
FFDEC_JAR = PROJECTS / "FFDec.app/Contents/Resources/ffdec-cli.jar"
OUTPUT = ROOT / "assets/menu"
CANVAS_SIZE = (2400, 1600)
BUTTON_IDS = {
    965,
    969,
    974,
    979,
    983,
    988,
    994,
    999,
    1004,
    1010,
    1011,
    1019,
    1022,
    1023,
    1025,
    1026,
    1052,
    1065,
    1077,
}
SPRITE_IDS = {
    10,
    110,
    230,
    326,
    422,
    560,
    713,
    793,
    802,
    809,
    816,
    819,
    821,
    830,
    850,
    852,
    855,
    874,
    900,
    912,
    913,
    1028,
    1029,
    1041,
    1048,
    1051,
    1056,
    1066,
    1069,
    1071,
    1076,
    1085,
}
SHAPE_IDS = {
    1020,
    1037,
    794,
    795,
    804,
    805,
    806,
    807,
    810,
    811,
    813,
    814,
    815,
    958,
}

KEY_PRESSER_FIELDS = {
    "name": (724, 866, 904, 101),
    "key": (720, 982, 912, 164),
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


def ffdec(*arguments: str, animate_subsprites: bool = False) -> None:
    command = [
        str(find_java()),
        "-jar",
        str(FFDEC_JAR),
        "-config",
        f"animateSubsprites={str(animate_subsprites).lower()}",
        "-zoom",
        "4",
        *arguments,
        str(SWF_PATH),
    ]
    subprocess.run(command, check=True)


def build_key_presser_textless() -> None:
    source = OUTPUT / "sprites/DefineSprite_1041"
    backdrop_path = OUTPUT / "shapes/1037.png"
    target = OUTPUT / "sprites/DefineSprite_1041_textless"
    if not source.is_dir() or not backdrop_path.is_file():
        raise SystemExit("KeyPresser sprite 1041 or backdrop shape 1037 was not exported.")
    target.mkdir(parents=True, exist_ok=True)
    backdrop = Image.open(backdrop_path).convert("RGBA").crop((0, 0, *CANVAS_SIZE))
    for frame_no in range(1, 21):
        frame = Image.open(source / f"{frame_no}.png").convert("RGBA")
        textless = frame.copy()
        for field_name, (x, y, width, height) in KEY_PRESSER_FIELDS.items():
            box = (x, y, x + width, y + height)
            region = frame.crop(box)
            alpha = region.getchannel("A").point(lambda value: 255 if value >= 250 else 0)
            brightness = region.convert("RGB").convert("L").point(
                lambda value: 255 if value > 24 else 0
            )
            visible = ImageChops.multiply(alpha, brightness)
            red, green, blue, _ = region.split()
            if field_name == "key":
                colored = ImageChops.multiply(
                    red.point(lambda value: 255 if value > 180 else 0),
                    ImageChops.multiply(
                        green.point(lambda value: 255 if value > 180 else 0),
                        blue.point(lambda value: 255 if value < 180 else 0),
                    ),
                )
            else:
                colored = ImageChops.multiply(
                    red.point(lambda value: 255 if value > 180 else 0),
                    ImageChops.multiply(
                        green.point(lambda value: 255 if value > 180 else 0),
                        blue.point(lambda value: 255 if value > 180 else 0),
                    ),
                )
            text_and_shadow = ImageChops.multiply(
                alpha,
                colored.filter(ImageFilter.MaxFilter(49)),
            )
            replacement = ImageChops.lighter(visible, text_and_shadow)
            textless.paste(backdrop.crop(box), box, replacement)
            mask = Image.new("RGBA", (width, height), (255, 255, 255, 0))
            mask.putalpha(visible)
            mask_dir = target / f"{field_name}_masks"
            mask_dir.mkdir(parents=True, exist_ok=True)
            mask.save(mask_dir / f"{frame_no}.png")
        textless.save(target / f"{frame_no}.png")


def main() -> None:
    if not SWF_PATH.is_file() or not FFDEC_JAR.is_file():
        raise SystemExit("fight-for-glorton.swf or FFDec.app is missing beside the project folder.")
    sponsor_output = OUTPUT / "sponsor_intro"
    shutil.rmtree(sponsor_output, ignore_errors=True)
    ffdec(
        "-selectid",
        "2",
        "-format",
        "font:ttf",
        "-export",
        "font",
        str(ROOT / "assets/fonts"),
    )
    ffdec(
        "-select",
        "0:2",
        "-sublength",
        "81",
        "-format",
        "frame:png",
        "-export",
        "frame",
        str(sponsor_output),
        animate_subsprites=True,
    )
    ffdec(
        "-select",
        "0:1-50",
        "-format",
        "frame:png",
        "-export",
        "frame",
        str(OUTPUT / "main_frames"),
    )
    ffdec(
        "-select",
        "0:100-170",
        "-format",
        "frame:png",
        "-export",
        "frame",
        str(OUTPUT / "end_frames"),
    )
    ffdec(
        "-selectid",
        ",".join(str(item) for item in sorted(BUTTON_IDS)),
        "-format",
        "button:png",
        "-export",
        "button",
        str(OUTPUT / "buttons"),
    )
    ffdec(
        "-selectid",
        ",".join(str(item) for item in sorted(SPRITE_IDS)),
        "-format",
        "sprite:png",
        "-export",
        "sprite",
        str(OUTPUT / "sprites"),
    )
    ffdec(
        "-selectid",
        ",".join(str(item) for item in sorted(SHAPE_IDS)),
        "-format",
        "shape:png",
        "-export",
        "shape",
        str(OUTPUT / "shapes"),
    )
    build_key_presser_textless()
    print(f"Exported original menu frames and dynamic symbols to {OUTPUT}")


if __name__ == "__main__":
    main()
