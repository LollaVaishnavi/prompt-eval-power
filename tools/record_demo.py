"""Capture real CLI stdout and wall time, then render paced MP4 and GIF replays."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "docs" / "video"
MANIFEST_PATH = VIDEO_DIR / "manifest.json"

COMMANDS: list[tuple[str, list[str]]] = [
    (
        "1. Small suite (n=30) refuses fake green -> exit 3 UNDERPOWERED",
        [
            sys.executable, "-m", "prompt_eval_power.cli", "gate",
            "--baseline", "v1_baseline_grounded", "--candidate", "v2_concise_rewrite",
            "--sample-size", "30",
        ],
    ),
    (
        "2. Full suite (n=300) catches regression -> exit 2 REJECT",
        [
            sys.executable, "-m", "prompt_eval_power.cli", "gate",
            "--baseline", "v1_baseline_grounded", "--candidate", "v2_concise_rewrite",
            "--sample-size", "300",
        ],
    ),
    (
        "3. Powered safe prompt (v3) -> exit 0 PASS",
        [
            sys.executable, "-m", "prompt_eval_power.cli", "gate",
            "--baseline", "v1_baseline_grounded", "--candidate", "v3_strict_citation",
            "--sample-size", "300",
        ],
    ),
    (
        "4. Zero-failure resolution floor (0/30 = 9.5% upper bound)",
        [
            sys.executable, "-m", "prompt_eval_power.cli", "zero-bound",
            "--failures", "0", "--sample-size", "30",
        ],
    ),
    (
        "5. Machine-checked receipts verification",
        [sys.executable, "tools/check_numbers.py"],
    ),
]


def _render_frame(title: str, cmd_display: str, lines: list[str], exit_code: int) -> Image.Image:
    width, height = 960, 520
    img = Image.new("RGB", (width, height), color=(15, 23, 42))
    draw = ImageDraw.Draw(img)
    draw.rectangle([(0, 0), (width, 38)], fill=(30, 41, 59))
    draw.ellipse([(14, 13), (26, 25)], fill=(239, 68, 68))
    draw.ellipse([(34, 13), (46, 25)], fill=(234, 179, 8))
    draw.ellipse([(54, 13), (66, 25)], fill=(34, 197, 94))

    try:
        font_mono = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        bold_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
        font_bold = ImageFont.truetype(bold_path, 14)
    except OSError:
        font_mono = ImageFont.load_default()
        font_bold = font_mono

    draw.text((84, 11), f"prompt-eval-power  |  {title}", fill=(226, 232, 240), font=font_bold)
    y = 56
    draw.text((22, y), f"$ {cmd_display[:98]}", fill=(56, 189, 248), font=font_bold)
    y += 28

    for line in lines[:20]:
        color = (241, 245, 249)
        if "REJECT" in line:
            color = (248, 113, 113)
        elif "UNDERPOWERED" in line:
            color = (251, 191, 36)
        elif "PASS" in line:
            color = (134, 239, 172)
        draw.text((22, y), line[:106], fill=color, font=font_mono)
        y += 22

    status_color = (134, 239, 172) if exit_code == 0 else (251, 191, 36)
    draw.rectangle([(0, height - 34), (width, height)], fill=(30, 41, 59))
    draw.text(
        (22, height - 25),
        f"exit code: {exit_code}   (replay of captured stdout paced by measured wall time)",
        fill=status_color,
        font=font_mono,
    )
    return img


def main() -> int:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    subprocess.run(
        [sys.executable, "-c", "import prompt_eval_power"],
        cwd=str(ROOT),
        env=env,
        check=True,
    )

    records = []
    for title, cmd in COMMANDS:
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        duration = round(time.perf_counter() - t0, 3)
        cmd_display = " ".join(["python3" if i == 0 else part for i, part in enumerate(cmd)])
        records.append(
            {
                "title": title,
                "command": cmd_display,
                "exit_code": proc.returncode,
                "duration_sec": duration,
                "stdout_lines": proc.stdout.strip().splitlines(),
            }
        )

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(
            {
                "note": "Real captured stdout and wall time for each demo segment.",
                "segments": [
                    {
                        "title": r["title"],
                        "command": r["command"],
                        "exit_code": r["exit_code"],
                        "duration_sec": r["duration_sec"],
                    }
                    for r in records
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        frame_idx, fps = 0, 4
        for rec in records:
            lines = rec["stdout_lines"]
            n_frames = int(round(min(4.0, max(2.2, 1.8 + rec["duration_sec"] * 0.4)) * fps))
            for f in range(n_frames):
                if f < 4 and len(lines) > 4:
                    shown = lines[: max(1, int(len(lines) * (f + 1) / 4))]
                else:
                    shown = lines
                _render_frame(rec["title"], rec["command"], shown, rec["exit_code"]).save(
                    tmp_path / f"frame_{frame_idx:04d}.png"
                )
                frame_idx += 1

        mp4_path = VIDEO_DIR / "demo.mp4"
        gif_path = VIDEO_DIR / "demo.gif"
        palette_path = tmp_path / "palette.png"

        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", str(tmp_path / "frame_%04d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(mp4_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", str(tmp_path / "frame_%04d.png"),
                "-vf", "palettegen=stats_mode=diff", str(palette_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", str(tmp_path / "frame_%04d.png"),
                "-i", str(palette_path),
                "-lavfi", "paletteuse=dither=bayer:bayer_scale=3", str(gif_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    gif_kb = round((VIDEO_DIR / "demo.gif").stat().st_size / 1024.0, 1)
    print(f"record_demo: wrote demo.mp4 and demo.gif ({gif_kb} KB) with {len(records)} segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
