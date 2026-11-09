import argparse
import json
from pathlib import Path

from .config import RunConfig
from .storage import write_json


def main():
    parser = argparse.ArgumentParser(
        prog="futsal", description="Local futsal analysis; unknown values stay unknown"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check-model", help="Inspect your weights and export a short detection video")
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--model", default="best.pt")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float, default=10)
    p.add_argument("--fps", type=float, default=15)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="cpu")
    p.add_argument("--confidence", type=float, default=0.15)
    p = sub.add_parser("quality", help="Report execution coverage, without inventing accuracy")
    p.add_argument("run")
    p = sub.add_parser("compare-teams", help="Compare LAB, HSV and optional SigLIP appearance clusters")
    p.add_argument("source")
    p.add_argument("detections")
    p.add_argument("output")
    p.add_argument("--siglip", action="store_true")
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("inventory")
    p.add_argument("directory", nargs="?", default=".")
    p.add_argument("--output", default="reports/asset_inventory.json")
    p = sub.add_parser("doctor")
    p = sub.add_parser("download-model")
    p.add_argument("name", choices=["yolo11n", "football-player", "football-ball"])
    p = sub.add_parser("audit-dataset")
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--copy", action="store_true")
    p = sub.add_parser("remove-class")
    p.add_argument("dataset")
    p.add_argument("output")
    p.add_argument("name")
    p = sub.add_parser("export-dataset")
    p.add_argument("dataset")
    p.add_argument("output")
    p = sub.add_parser("evaluate")
    p.add_argument("dataset")
    p.add_argument("model")
    p.add_argument("output")
    p.add_argument("--device", default="cpu")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--limit", type=int)
    p.add_argument("--exploratory", action="store_true")
    p = sub.add_parser("train")
    p.add_argument("dataset")
    p.add_argument("model")
    p.add_argument("output")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=50)
    p = sub.add_parser("propose")
    p.add_argument("source")
    p.add_argument("output")
    p.add_argument("--model", default="best.pt")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float, default=60)
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--device", default="cpu")
    p.add_argument("--group", required=True)
    p = sub.add_parser("process")
    p.add_argument("source")
    p.add_argument("--config")
    p.add_argument("--start", type=float)
    p.add_argument("--duration", type=float)
    p.add_argument("--full", action="store_true")
    p.add_argument("--device")
    p.add_argument("--model")
    p.add_argument("--root", default="runs")
    p.add_argument("--background", action="store_true")
    p = sub.add_parser("jobs")
    p.add_argument("--root", default="runs")
    p = sub.add_parser("derive")
    p.add_argument("run")
    p = sub.add_parser("identity-candidates")
    p.add_argument("run")
    p = sub.add_parser("export")
    p.add_argument("run")
    p.add_argument("output")
    p = sub.add_parser("shots")
    p.add_argument("source")
    p.add_argument("output")
    p = sub.add_parser("dashboard")
    p.add_argument("--port", type=int, default=8501)
    p = sub.add_parser("download-reid")
    p = sub.add_parser("reid")
    p.add_argument("run")
    p.add_argument("--checkpoint", default="models/osnet-msmt17.pth")
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("keypoint-export")
    p.add_argument("annotations")
    p.add_argument("output")
    p = sub.add_parser("train-court")
    p.add_argument("dataset")
    p.add_argument("checkpoint")
    p.add_argument("output")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=50)
    p = sub.add_parser("evaluate-tracking")
    p.add_argument("ground_truth")
    p.add_argument("prediction")
    p.add_argument("output")
    p = sub.add_parser("working-dataset")
    p.add_argument("dataset")
    p.add_argument("output")
    p = sub.add_parser("train-formations")
    p.add_argument("annotations")
    p.add_argument("output")
    args = parser.parse_args()
    try:
        result = dispatch(args)
        if result is not None:
            print(json.dumps(result, indent=2, ensure_ascii=True, default=str))
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")


def dispatch(a):
    if a.command == "quality":
        from .quality import summarize_run

        return summarize_run(a.run)
    if a.command == "compare-teams":
        from .team_comparison import compare_teams

        result = compare_teams(a.source, a.detections, a.output, a.siglip, a.device)
        return {"sample_count": result["sample_count"], "methods": result["methods"], "team_accuracy": None}
    if a.command == "check-model":
        from .diagnostics import detection_check

        report = detection_check(a.source, a.model, a.output, a.start, a.duration, a.fps,
                                 a.imgsz, a.device, a.confidence)
        return {k: report[k] for k in ("frames", "class_counts", "inference_median_ms", "processing_fps")}
    if a.command == "train-formations":
        from .formations import train_formations

        return str(train_formations(a.annotations, a.output))
    if a.command == "download-reid":
        from .reid import download_osnet

        return str(download_osnet())
    if a.command == "reid":
        from .reid import osnet_candidates

        return osnet_candidates(a.run, a.checkpoint, a.device)
    if a.command == "keypoint-export":
        from .court import keypoint_export

        return str(keypoint_export(a.annotations, a.output))
    if a.command == "train-court":
        from .court import train_court

        train_court(a.dataset, a.checkpoint, a.output, a.device, a.epochs)
        return {"status": "training_completed_requires_evaluation"}
    if a.command == "evaluate-tracking":
        from .tracking_eval import evaluate_mot

        return evaluate_mot(a.ground_truth, a.prediction, a.output)
    if a.command == "working-dataset":
        from .datasets import working_export

        return str(working_export(a.dataset, a.output))
    if a.command == "inventory":
        from .video import probe

        result = [probe(p) for p in sorted(Path(a.directory).glob("*.mp4"))]
        write_json(a.output, result)
        return result
    if a.command == "doctor":
        import torch
        import psutil
        from .video import ffmpeg_path

        result = {
            "cuda_available": torch.cuda.is_available(),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
            "ffmpeg": ffmpeg_path(),
        }
        write_json("reports/hardware.json", result)
        return result
    if a.command == "download-model":
        from .models import download_model

        return str(download_model(a.name))
    if a.command == "audit-dataset":
        from .datasets import audit_dataset

        return {k: v for k, v in audit_dataset(a.source, a.output, copy=a.copy).items() if k != "records"}
    if a.command == "remove-class":
        from .datasets import remove_class_version

        return {k: v for k, v in remove_class_version(a.dataset, a.output, a.name).items() if k != "records"}
    if a.command == "export-dataset":
        from .datasets import export_yolo

        return str(export_yolo(a.dataset, a.output))
    if a.command == "evaluate":
        from .evaluation import evaluate_detector

        return evaluate_detector(a.dataset, a.model, a.output, a.device, a.imgsz, a.limit, a.exploratory)
    if a.command == "train":
        from .evaluation import train_model

        train_model(a.dataset, a.model, a.output, a.device, a.epochs)
        return {"status": "training_finished_evaluate_before_promotion"}
    if a.command == "propose":
        from .datasets import propose_video
        from .models import Detector

        records = propose_video(
            a.source, Detector(a.model, a.device), a.output, a.start, a.duration, a.count, a.group
        )
        return {"proposed_images": len(records), "review_required": True}
    if a.command == "process":
        from .jobs import create_job, get_job
        from .pipeline import run_pipeline

        config = RunConfig.load(a.config)
        changes = {
            k: getattr(a, k) for k in ("start", "duration", "device", "model") if getattr(a, k) is not None
        }
        if a.full:
            changes["duration"] = None
        config = RunConfig.model_validate({**config.model_dump(), **changes})
        job_id = create_job(a.source, config, a.root, launch=a.background)
        if not a.background:
            from .jobs import claim_job, update_job

            job = claim_job(a.root, job_id)
            try:
                result = run_pipeline(
                    a.source,
                    job["run_path"],
                    config,
                    progress=lambda n, p: update_job(a.root, job_id, frames=n, progress=p),
                )
                update_job(a.root, job_id, state=result["state"], frames=result["frames"], progress=1)
                from .analytics import derive_run

                derive_run(job["run_path"])
            except Exception as exc:
                update_job(a.root, job_id, state="failed", error=str(exc))
                raise
        return get_job(a.root, job_id)
    if a.command == "jobs":
        from .jobs import list_jobs

        return list_jobs(a.root)
    if a.command == "derive":
        from .analytics import derive_run

        return derive_run(a.run)
    if a.command == "identity-candidates":
        from .reid import propose_links

        return propose_links(a.run)
    if a.command == "export":
        from .exports import export_run

        return str(export_run(a.run, a.output))
    if a.command == "shots":
        from scenedetect import detect, AdaptiveDetector

        values = detect(a.source, AdaptiveDetector())
        result = [
            {"shot_id": i, "start_s": s.get_seconds(), "end_s": e.get_seconds(), "label": "unknown"}
            for i, (s, e) in enumerate(values)
        ]
        write_json(a.output, result)
        return {"shots": len(result)}
    if a.command == "dashboard":
        import subprocess
        import sys

        app = Path(__file__).resolve().parents[2] / "dashboard" / "app.py"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app),
                "--server.address",
                "127.0.0.1",
                "--server.port",
                str(a.port),
                "--browser.gatherUsageStats",
                "false",
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
