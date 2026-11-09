import json
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

from .storage import utcnow


def connect(root="runs"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(root / "jobs.sqlite", timeout=30)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE IF NOT EXISTS jobs (
        id TEXT PRIMARY KEY, source TEXT NOT NULL, config TEXT NOT NULL,
        state TEXT NOT NULL, created TEXT, updated TEXT, progress REAL DEFAULT 0,
        frames INTEGER DEFAULT 0, error TEXT, pid INTEGER, attempt INTEGER DEFAULT 0,
        run_path TEXT, cancel INTEGER DEFAULT 0)""")
    db.commit()
    return db


def list_jobs(root="runs"):
    with connect(root) as db:
        return [dict(r) for r in db.execute("SELECT * FROM jobs ORDER BY created DESC")]


def update_job(root, job_id, **values):
    allowed = {"state", "updated", "progress", "frames", "error", "pid", "attempt", "run_path", "cancel"}
    if not values.keys() <= allowed:
        raise ValueError("Invalid job field")
    values["updated"] = utcnow()
    with connect(root) as db:
        db.execute(
            "UPDATE jobs SET " + ",".join(f"{k}=?" for k in values) + " WHERE id=?",
            [*values.values(), job_id],
        )


def get_job(root, job_id):
    with connect(root) as db:
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        raise ValueError("Unknown job")
    return dict(row)


def create_job(source, config, root="runs", launch=True):
    job_id = uuid.uuid4().hex[:12]
    with connect(root) as db:
        db.execute(
            "INSERT INTO jobs (id,source,config,state,created,updated) VALUES (?,?,?,?,?,?)",
            (job_id, str(Path(source).resolve()), config.model_dump_json(), "queued", utcnow(), utcnow()),
        )
    if launch:
        launch_job(root, job_id)
    return job_id


def launch_job(root, job_id):
    job = get_job(root, job_id)
    if job["state"] not in {"queued", "failed", "cancelled", "interrupted"}:
        raise ValueError("Only a queued, interrupted, failed, or cancelled job can be started")
    root = Path(root).resolve()
    directory = root / job_id
    directory.mkdir(parents=True, exist_ok=True)
    update_job(root, job_id, state="queued", cancel=0, error=None)
    with (directory / "worker.log").open("a", encoding="utf-8") as log:
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        process = subprocess.Popen(
            [sys.executable, "-m", "futsal.worker", "--root", str(root), "--job", job_id],
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=flags,
            cwd=str(Path.cwd()),
        )
    update_job(root, job_id, pid=process.pid)


def claim_job(root, job_id):
    with connect(root) as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or row["state"] != "queued":
            raise ValueError("Job already claimed or not queued")
        attempt = row["attempt"] + 1
        path = Path(root).resolve() / job_id / f"attempt-{attempt:04d}"
        db.execute(
            "UPDATE jobs SET state='running',attempt=?,run_path=?,progress=0,frames=0 WHERE id=?",
            (attempt, str(path), job_id),
        )
    return get_job(root, job_id)


def corrections(run):
    from .storage import read_json

    value = read_json(Path(run) / "review.json", {})
    return {
        k: value.get(k, {}) for k in ("segments", "identities", "teams", "roles", "events", "ball_ground")
    }


def save_correction(run, category, key, value, reviewer="local_user"):
    """Append audit evidence before publishing a new materialized review snapshot."""
    from .storage import write_json

    if category not in {"segments", "identities", "teams", "roles", "events", "ball_ground"}:
        raise ValueError("Invalid correction category")
    run = Path(run)
    with (run / "review_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "time": utcnow(),
                    "reviewer": reviewer,
                    "category": category,
                    "key": str(key),
                    "value": value,
                }
            )
            + "\n"
        )
    current = corrections(run)
    current[category][str(key)] = value
    write_json(run / "review.json", current)
