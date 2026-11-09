"""Extract a pool of candidate frames (one every N seconds) from each match video.

Output: data/court_frames/<video_id>/cand/<index:06d>.jpg   (16:9, max 960 px wide)
The sample time of cand/<n>.jpg is approximately (n-1) * interval seconds; the exact value is
stored in data/court_frames/<video_id>/cand_meta.json.  Existing candidates are never overwritten.
"""
import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "court_frames"
INTERVAL = float(sys.argv[1]) if len(sys.argv) > 1 else 2.0
ONLY = set(sys.argv[2:])

# video_id -> filename fragment (ids follow keypoint_images/manifest.csv numbering)
VIDEOS = {
    "v1": "France-Argentine",
    "v2": "Ukraine-France",
    "v3": "Palma Futsal",
    "v4": "Sporting - Benfica",
    "v5": "videoplayback",
    "v6": "WILD ENDING",
}

ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
for vid, fragment in VIDEOS.items():
    if ONLY and vid not in ONLY:
        continue
    matches = [p for p in sorted(ROOT.glob("*.mp4")) if fragment in p.name]
    if len(matches) != 1:
        print(f"{vid}: expected 1 video for '{fragment}', found {len(matches)}", flush=True)
        continue
    source = matches[0]
    target = OUT / vid / "cand"
    if target.exists() and any(target.iterdir()):
        print(f"{vid}: candidates already extracted, skipping", flush=True)
        continue
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vf", f"fps=1/{INTERVAL},scale='min(960,iw)':-2", "-q:v", "3",
        str(target / "%06d.jpg"),
    ]
    print(f"{vid}: extracting from {source.name}", flush=True)
    subprocess.run(cmd, check=True)
    count = len(list(target.glob("*.jpg")))
    (OUT / vid / "cand_meta.json").write_text(
        json.dumps({"source": source.name, "interval_s": INTERVAL, "count": count}, indent=2), "utf-8")
    print(f"{vid}: {count} candidates", flush=True)
print("done", flush=True)
