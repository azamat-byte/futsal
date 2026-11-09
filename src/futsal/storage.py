import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import distributions
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def sha256(path, block=4 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(block), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), "utf-8")
    os.replace(temp, path)


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text("utf-8")) if path.exists() else default


def environment():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {d.metadata["Name"]: d.version for d in distributions() if d.metadata["Name"]},
    }


TRACK_SCHEMA = pa.schema(
    [
        ("frame_index", pa.int64()),
        ("time_s", pa.float64()),
        ("dt_s", pa.float64()),
        ("shot_id", pa.int64()),
        ("tracklet_id", pa.string()),
        ("detection_id", pa.string()),
        ("class_name", pa.string()),
        ("role", pa.string()),
        ("score", pa.float64()),
        ("display_box", pa.list_(pa.float64())),
        ("x1", pa.float64()),
        ("y1", pa.float64()),
        ("x2", pa.float64()),
        ("y2", pa.float64()),
        ("team", pa.string()),
        ("team_score", pa.float64()),
        ("observed", pa.bool_()),
        ("court_x", pa.float64()),
        ("court_y", pa.float64()),
        ("calibration_valid", pa.bool_()),
        ("scale_verified", pa.bool_()),
        ("calibration_error_px", pa.float64()),
    ]
)


class TrackWriter:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.rows = []
        self.part = 0

    def add(self, rows):
        self.rows.extend(rows)
        if len(self.rows) >= 3000:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        path = self.directory / f"part-{self.part:06d}.parquet"
        temp = path.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(self.rows, schema=TRACK_SCHEMA), temp, compression="zstd")
        os.replace(temp, path)
        self.rows.clear()
        self.part += 1


def iter_tracks(run, columns=None):
    for path in sorted((Path(run) / "tracks").glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=5000, columns=columns):
            yield batch.to_pandas()


def load_tracks(run, columns=None, start=None, end=None, max_rows=None):
    """Read a source-time window, applying filters before column projection and row limits."""
    import pandas as pd

    if start is not None and end is not None and start > end:
        raise ValueError("Window start must not exceed end")
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be nonnegative")
    paths = sorted((Path(run) / "tracks").glob("*.parquet"))
    if not paths or max_rows == 0:
        return pd.DataFrame(columns=columns or TRACK_SCHEMA.names)
    filters = []
    if start is not None:
        filters.append(("time_s", ">=", start))
    if end is not None:
        filters.append(("time_s", "<=", end))
    if max_rows is not None:
        import pyarrow.dataset as ds

        predicate = None
        if start is not None:
            predicate = ds.field("time_s") >= start
        if end is not None:
            upper = ds.field("time_s") <= end
            predicate = upper if predicate is None else predicate & upper
        scanner = ds.dataset(paths, format="parquet").scanner(
            columns=columns, filter=predicate, batch_size=5000
        )
        frames = []
        remaining = max_rows
        for batch in scanner.to_batches():
            if not batch.num_rows:
                continue
            frames.append(batch.slice(0, min(batch.num_rows, remaining)).to_pandas())
            remaining -= len(frames[-1])
            if remaining <= 0:
                break
        return (
            pd.concat(frames, ignore_index=True)
            if frames
            else pd.DataFrame(columns=columns or TRACK_SCHEMA.names)
        )
    return pq.ParquetDataset(paths, filters=filters or None).read(columns=columns).to_pandas()
