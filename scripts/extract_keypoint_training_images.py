"""Extract 100 full-resolution, temporally distributed frames per source video."""

import csv
import hashlib
import json
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "keypoint_images_100"
COUNT = 100
EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def main():
    cv2.setNumThreads(2)
    videos = sorted(
        (p for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: p.name.casefold(),
    )
    if not videos:
        raise RuntimeError("No source videos found")
    OUTPUT.mkdir(exist_ok=False)
    previews = OUTPUT / "previews"
    previews.mkdir()
    rows = []
    summaries = []
    for video_number, source in enumerate(videos, 1):
        slug = re.sub(r"[^A-Za-z0-9_-]+", "_", source.stem).strip("_")
        folder = OUTPUT / f"{video_number:02d}_{slug}"
        folder.mkdir()
        capture = cv2.VideoCapture(str(source))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"Cannot open {source.name!a}")
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            if total < COUNT or fps <= 0:
                raise RuntimeError(f"Insufficient frames or invalid FPS: {source.name!a}")
            print(f"Video {video_number}/{len(videos)}: {source.name!a}", flush=True)
            hashes = set()
            selected_indices = set()
            video_rows = []
            for index in range(COUNT):
                # Sample the center of each of 100 equal temporal bins. If a
                # frozen/black frame repeats exactly, try nearby points in its bin.
                start = index * total // COUNT
                end = (index + 1) * total // COUNT
                center = (start + end) // 2
                offsets = [0, .15, -.15, .30, -.30, .45, -.45]
                for offset in offsets:
                    requested = min(end - 1, max(start, center + round(offset * (end - start))))
                    if requested in selected_indices:
                        continue
                    if not capture.set(cv2.CAP_PROP_POS_FRAMES, requested):
                        continue
                    ok, frame = capture.read()
                    if not ok:
                        continue
                    actual = int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1
                    digest = hashlib.sha256(frame.tobytes()).hexdigest()
                    if digest not in hashes and actual not in selected_indices:
                        break
                else:
                    raise RuntimeError(f"No distinct decodable frame in bin {index + 1}: {source.name!a}")
                hashes.add(digest)
                selected_indices.add(actual)
                name = f"v{video_number:02d}_{index + 1:03d}_frame_{actual:08d}.jpg"
                destination = folder / name
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ok:
                    raise RuntimeError(f"JPEG encoding failed: {name}")
                destination.write_bytes(encoded.tobytes())
                row = {
                    "image": destination.relative_to(OUTPUT).as_posix(),
                    "source_video": source.name,
                    "frame_index": actual,
                    "timestamp_seconds": round(actual / fps, 6),
                    "width": frame.shape[1],
                    "height": frame.shape[0],
                    "sha256": hashlib.sha256(encoded.tobytes()).hexdigest(),
                }
                video_rows.append(row)
                rows.append(row)
                if (index + 1) % 25 == 0:
                    print(f"  {index + 1}/{COUNT} images saved", flush=True)
            # Verify every saved JPEG is readable and retains its source size.
            for row in video_rows:
                with Image.open(OUTPUT / row["image"]) as saved:
                    saved.load()
                    if saved.size != (row["width"], row["height"]):
                        raise RuntimeError(f"Size mismatch: {row['image']}")
            sheet = Image.new("RGB", (5 * 320, 4 * 205), "#202020")
            draw = ImageDraw.Draw(sheet)
            for position, selected in enumerate(np.linspace(0, COUNT - 1, 20, dtype=int)):
                row = video_rows[int(selected)]
                with Image.open(OUTPUT / row["image"]) as saved:
                    thumb = ImageOps.contain(saved, (320, 180))
                    x, y = position % 5 * 320, position // 5 * 205
                    sheet.paste(thumb, (x + (320 - thumb.width) // 2, y))
                    draw.text((x + 5, y + 183), f"{selected + 1:03d} | {row['timestamp_seconds']:.2f}s", fill="white")
            sheet.save(previews / f"video_{video_number:02d}.jpg", quality=90)
            summaries.append({
                "source_video": source.name,
                "folder": folder.name,
                "image_count": len(video_rows),
                "fps": fps,
                "source_frames": total,
                "duration_seconds": round(total / fps, 6),
                "width": video_rows[0]["width"],
                "height": video_rows[0]["height"],
            })
        finally:
            capture.release()
    with (OUTPUT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT / "summary.json").write_text(
        json.dumps({"total_images": len(rows), "videos": summaries}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (OUTPUT / "README.txt").write_text(
        "600 extracted training images: 100 per original source video.\n"
        "Images are sampled across 100 equal time bins at original resolution, JPEG quality 95.\n"
        "Exact repeated decoded frames within a video are avoided.\n"
        "Images are unannotated: add your keypoint labels before supervised training.\n"
        "manifest.csv records each image's source, zero-based frame index, timestamp, size, and SHA-256.\n"
        "previews/ contains 20-image contact sheets for review; these are not training images.\n"
        "Original videos and the existing keypoint_images directory are unchanged.\n",
        encoding="utf-8",
    )
    print(f"DONE: {len(rows)} verified training images from {len(videos)} videos.", flush=True)


if __name__ == "__main__":
    main()
