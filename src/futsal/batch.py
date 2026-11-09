"""Durable sequential video batch. Each GPU worker exits before the next starts."""

import argparse
import subprocess
import sys
import uuid
from pathlib import Path

from .jobs import create_job, get_job
from .storage import read_json, write_json, utcnow


def start_batch(sources, config, root="runs", background=True):
    root = Path(root).resolve()
    batch_id = "batch-" + uuid.uuid4().hex[:10]
    path = root / (batch_id + ".json")
    jobs = [create_job(source, config, root, launch=False) for source in sources]
    write_json(
        path,
        {
            "id": batch_id,
            "created": utcnow(),
            "state": "queued",
            "jobs": jobs,
            "scope": "full source processing; analytical validity remains independently gated",
        },
    )
    if background:
        with (root / (batch_id + ".log")).open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-m", "futsal.batch", "--manifest", str(path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )
        manifest = read_json(path)
        manifest["supervisor_pid"] = process.pid
        write_json(path, manifest)
    else:
        supervise(path)
    return path


def supervise(path):
    path = Path(path).resolve()
    root = path.parent
    batch = read_json(path)
    batch["state"] = "running"
    write_json(path, batch)
    for job_id in batch["jobs"]:
        job = get_job(root, job_id)
        if job["state"] != "queued":
            continue
        subprocess.run(
            [sys.executable, "-m", "futsal.worker", "--root", str(root), "--job", job_id], check=False
        )
    results = [{"id": j, "state": get_job(root, j)["state"]} for j in batch["jobs"]]
    batch = read_json(path)
    batch.update(
        state="completed" if all(r["state"] == "completed" for r in results) else "finished_with_errors",
        results=results,
        finished=utcnow(),
    )
    write_json(path, batch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    supervise(parser.parse_args().manifest)
