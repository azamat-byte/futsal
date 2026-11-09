"""Real smoke validation on every supplied video, not sporting accuracy evaluation."""

import sys
from pathlib import Path

from futsal.analytics import derive_run
from futsal.config import RunConfig
from futsal.jobs import create_job, claim_job, update_job
from futsal.pipeline import run_pipeline
from futsal.storage import write_json, utcnow
from futsal.video import probe


def main():
    root = Path(__file__).resolve().parents[1]
    report = []
    for source in sorted(root.glob("*.mp4")):
        config = RunConfig(
            model=str(root / "models/football-player-detection.pt"),
            device="0",
            start=1200,
            duration=8,
            sample_fps=5,
            imgsz=640,
            render=True,
        )
        job_id = create_job(source, config, root / "runs", launch=False)
        job = claim_job(root / "runs", job_id)
        try:
            result = run_pipeline(
                source,
                job["run_path"],
                config,
                progress=lambda n, p: update_job(root / "runs", job_id, frames=n, progress=p),
            )
            update_job(root / "runs", job_id, state=result["state"], frames=result["frames"], progress=1)
            derive_run(job["run_path"])
            video = probe(Path(job["run_path"]) / "annotated.mp4")
            if result["frames"] < 35 or video["duration_s"] < 7:
                raise ValueError("Output does not cover the configured sample interval")
            report.append(
                {
                    "video": source.name,
                    "job": job_id,
                    "state": "passed_execution_smoke",
                    "frames": result["frames"],
                    "run_path": job["run_path"],
                    "wall_time_s": result["wall_time_s"],
                    "accuracy": "not_evaluated",
                }
            )
        except Exception as exc:
            update_job(root / "runs", job_id, state="failed", error=str(exc))
            report.append({"video": source.name, "job": job_id, "state": "failed", "error": str(exc)})
        write_json(
            root / "reports/generated/video_validation.json",
            {
                "created": utcnow(),
                "scope": "8-second real execution clips at source 1200s; not full-match release validation",
                "results": report,
            },
        )
        print(report[-1], flush=True)
    if any(r["state"] == "failed" for r in report):
        sys.exit(1)


if __name__ == "__main__":
    main()
