import argparse
import json
import traceback
import os

from .config import RunConfig
from .jobs import claim_job, get_job, update_job
from .pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    job = claim_job(args.root, args.job)
    update_job(args.root, args.job, pid=os.getpid())
    try:
        result = run_pipeline(
            job["source"],
            job["run_path"],
            RunConfig.model_validate(json.loads(job["config"])),
            progress=lambda n, p: update_job(args.root, args.job, frames=n, progress=p),
            cancelled=lambda: bool(get_job(args.root, args.job)["cancel"]),
        )
        if result["state"] == "completed":
            from .analytics import derive_run

            derive_run(job["run_path"])
        update_job(
            args.root,
            args.job,
            state=result["state"],
            frames=result["frames"],
            progress=1 if result["state"] == "completed" else get_job(args.root, args.job)["progress"],
        )
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        update_job(args.root, args.job, state="failed", error=error)


if __name__ == "__main__":
    main()
