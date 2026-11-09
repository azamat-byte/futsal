"""Keep uploaded assets identifiable while storing their bytes under content-addressed paths."""

import hashlib
from pathlib import Path

from .storage import read_json, utcnow, write_json


def save_upload(upload, directory):
    directory = Path(directory)
    original_name = upload.name.replace("\\", "/").rsplit("/", 1)[-1]
    suffix = Path(original_name).suffix.lower()
    if suffix not in {".mp4", ".pt"}:
        raise ValueError("Choose an MP4 video or PT model weights")
    content = upload.getbuffer()
    if not len(content):
        raise ValueError("The uploaded file is empty")
    digest = hashlib.sha256(content).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / (digest[:16] + suffix)
    if not target.exists():
        temporary = target.with_suffix(suffix + ".tmp")
        with temporary.open("wb") as stream:
            stream.write(content)
        temporary.replace(target)
    index = read_json(directory / "index.json", {})
    index[target.name] = {"original_name": original_name, "sha256": digest,
                          "size_bytes": len(content), "saved_at": utcnow()}
    write_json(directory / "index.json", index)
    return target


def asset_label(path):
    path = Path(path)
    if path.parent.name != "uploads":
        return path.name
    entry = read_json(path.parent / "index.json", {}).get(path.name, {})
    if entry.get("original_name"):
        return entry["original_name"] + " · uploaded"
    # Older uploads have no original-name record. Keep them usable without uploading again.
    return f"Uploaded {'video' if path.suffix.lower() == '.mp4' else 'weights'} · {path.stat().st_size / 1024**2:.1f} MB"


def preferred_video(videos):
    uploads = [p for p in videos if p.parent.name == "uploads"]
    return max(uploads, key=lambda p: p.stat().st_mtime_ns) if uploads else videos[0]
