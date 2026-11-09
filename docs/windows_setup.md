# Windows setup

Use 64-bit Python 3.11 and PowerShell. The existing `.venv` is usable in this workspace. C: has limited space; keep temporary downloads and caches on D:.

For a clean installation, run `scripts/setup_windows.ps1`. CUDA installation is optional via `-Cuda`. The CUDA wheel does not require installing a full CUDA toolkit. GPU availability is checked by `futsal doctor`; CPU is always explicitly selectable.

Core packages are pinned in `pyproject.toml`, with the tested environment snapshot in `reports/environment.json`. Optional RF-DETR/TrackEval dependencies are isolated from the normal startup path.

FFmpeg is provided by the imageio-ffmpeg wheel, so a global ffmpeg executable is not mandatory. PyAV handles decoded timestamps and non-ASCII video paths. Annotation image reads/writes use byte buffers because some OpenCV path APIs do not support Windows Unicode paths reliably.

Set `PYTHONUTF8=1` when launching from legacy terminals. Set `YOLO_AUTOINSTALL=false` to prevent runtime dependency changes. Use the project launcher to set local model/cache directories.

If an interrupted process remains marked running, confirm its PID is no longer active before marking it interrupted and restarting. Do not launch multiple large GPU jobs on the 4 GB card. Inference-size changes require a new run and a new configuration record.
