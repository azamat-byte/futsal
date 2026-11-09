# Futsal AI Tactical Analysis — Futsal Lab

New modular RF-DETR starter: [Phase 1 setup and code guide](docs/phase1.md),
[five-phase architecture](docs/phase1_architecture.md). Run it separately with
`python -m futsal.phase1`; it uses your Roboflow RF-DETR model and Supervision ByteTrack.

[Phase 2 calibration and radar](docs/phase2.md) reuses a completed Phase 1 run:
`python -m futsal.phase2 calibrate` then `python -m futsal.phase2 render`.

A working research pipeline built around **your supplied best.pt**. No detector was trained or replaced, and no dataset was rebuilt. Accuracy is not yet established for production use.

For improved motion handling, the dashboard now defaults to 20 samples/second and BoT-SORT.
Keep **Use motion-aware tracking settings** enabled. Optional **Extra ball search** uses your
same detector on overlapping crops. [Optimization settings and evidence](docs/optimization.md)
explain the tradeoffs and the uploaded clip's remaining detection limitation.

## Open the deliverables

- `notebooks/futsal_ai_visual_walkthrough.ipynb`: start here for pictures and explanations. Includes executed real demo outputs, player/ball detections, crop and cluster panels, goalkeeper role errors, tracking videos, court line candidates, reviewed landmarks, homography and radar. Switch from demo to upload/Drive mode for a new match; optional labeled validation reports accuracy without training.
- `notebooks/futsal_ai_colab.ipynb`: upload to Google Colab and select a T4 GPU. The notebook embeds the full source, supports Drive or direct upload, inspects the checkpoint, and runs each stage behind explicit review gates.
- `notebooks/futsal_ai_source.zip`: portable source package without weights, videos or the Windows environment.
- `outputs/01_detection_blue_court/detections.mp4`: your model's detections.
- `outputs/05_tracking_refined/annotated.mp4`: ByteTrack player IDs.
- `outputs/06_teams_refined/annotated.mp4`: experimental jersey team colors.
- `outputs/07_tactical_refined/tactical.mp4`: broadcast with synchronized radar.
- `outputs/07_tactical_refined/radar.mp4`: standalone tactical radar.
- `outputs/07_tactical_refined/tracks.csv` and `tracks.json`: source-timestamped observations.
- [Measured evidence and limitations](docs/milestone_report.md).

Detection/tracking use France–Argentina seconds 600–608. Radar uses Spain–Portugal seconds 120–128 and manually checked floor landmarks. Court dimensions are assumed to be 40×20 m; metric speed/distance and event statistics remain gated.

## Run on Windows

From this project directory:

    .\launch.ps1

Open **http://127.0.0.1:8501**. Alternatively:

    $env:PYTHONUTF8='1'
    $env:YOLO_AUTOINSTALL='false'
    .\.venv\Scripts\python.exe -m futsal dashboard

The interface selects root `best.pt` first. It supports video/weights upload, module and color controls, tracking/radar playback, manual calibration, team/role corrections, heatmap inspection and exports. The inference pipeline is independent of Streamlit. Another entry point is `streamlit run app/streamlit_app.py`.

## Run each stage

Inspect the model and render a detection clip:

    .\.venv\Scripts\python.exe -m futsal check-model 'Coupe du monde Futsal 2024 - Le replay intégral de France-Argentine.mp4' outputs/my-detection --model best.pt --start 600 --duration 10 --imgsz 1280 --device 0

Run ByteTrack and LAB team clustering; radar positions remain unavailable without calibration:

    .\.venv\Scripts\python.exe -m futsal process 'Coupe du monde Futsal 2024 - Le replay intégral de France-Argentine.mp4' --config configs/best_t4.yaml --device 0

Reproduce the manually anchored radar for the supplied Spain–Portugal source only:

    .\.venv\Scripts\python.exe -m futsal process 'WILD ENDING! Extra-time of Spain v Portugal _ 2021 #FutsalWC.mp4' --config configs/spain_demo.yaml

Use the actual run path printed by `process`:

    .\.venv\Scripts\python.exe -m futsal quality PATH_TO_RUN
    .\.venv\Scripts\python.exe -m futsal export PATH_TO_RUN outputs/my-export
    .\.venv\Scripts\python.exe -m futsal compare-teams 'YOUR_MATCH.mp4' PATH_TO_RUN/detections.jsonl outputs/team-comparison
    .\.venv\Scripts\python.exe -m pytest -q

Output directories must be new. Switch `tracker` to `botsort` to compare camera-motion compensation. Set `reuse_detections_from` to a compatible completed run to compare downstream settings without repeating inference; source/checkpoint checksums and detector configuration are checked. **Cached-run throughput is not inference speed.** Original full-inference measurements remain in `outputs/02_tracking`, `03_teams` and `04_radar`.

## Model and dependencies

Actual checkpoint: **YOLO12x, 59,123,004 parameters, detect task**. Names are `0: ball`, `1: goalkeeper`, `2: player`, `3: player- ball`. Mapping is derived from the checkpoint. The mixed class is excluded from tactical tracking and is never relabelled as referee. Raw diagnostic outputs retain it. Near-identical people boxes across role labels are suppressed before tracking; ball association stays separate.

The checkpoint reports Ultralytics 8.4.182 training. Local inference worked with **Ultralytics 8.3.240, torch 2.5.1+cu121, Python 3.11, GTX 1650 Ti**. The notebook pins that verified Ultralytics version and keeps Colab's supplied PyTorch/CUDA. Its core functions ran locally; T4/Colab execution was not available to test here. Start at 1280 inference size: 640 missed many players, and sampled black-court broadcasts remain a serious model limitation.

For a new environment, install the appropriate PyTorch wheel first, then `pip install -e .` or `pip install -r requirements.txt`. Notebook dependencies: `pip install -e ".[colab]"`. Optional `pip install -e ".[embeddings]"` enables the implemented SigLIP→UMAP→KMeans comparison (`compare-teams --siglip --device cuda`). This appearance encoder was not downloaded or evaluated locally and does not replace your detector. Supervision is included in the Colab extras; native Ultralytics tracker adapters directly consume your detections.

## Court calibration and honest analytics

OpenCV produces **image-space court-line proposals**, not automatically named futsal landmarks. Use the notebook's point picker or the Calibration page to choose actual visible floor markings. Supply at least four non-degenerate fitting correspondences plus independent validation points. RANSAC thresholds and held-out reprojection errors use pixels. Optical flow propagates the mapping briefly; camera cuts, unsupported motion and expiry invalidate it. Elevated goalposts are not floor landmarks.

The radar uses the same source PTS as the sampled broadcast. A/B are display labels: blue/red, unknown gray, referee yellow when a supported/reviewed role exists, ball orange. Goalkeepers are not assigned by the nearest team centroid. A visible ball may be airborne, so it is not automatically projected to the floor.

Trajectories, heatmaps, movement estimates, visible-team geometry and conservative possession/pass hypotheses exist behind quality gates. They require reviewed live-play segments, reliable identities/calibration and verified scale. Formation training utilities remain untrained. No match events, sporting accuracy or production reliability are asserted. IDF1/HOTA require identity ground truth; other independent references are also missing.

`frame_index` is a **source presentation-timestamp tick**, not an ordinal original frame number. `frames.jsonl` maps sampled `proxy_index` to source PTS and seconds. Output videos are sampled silent H264 proxies. Unknown values remain null in CSV/JSON/Parquet.

## Project organization

The existing package is preserved to retain its review, job and export workflows:

| Responsibility | Modules |
|---|---|
| Detection and first milestone | `src/futsal/models.py`, `diagnostics.py` |
| People and ball tracking | `src/futsal/tracking.py` |
| Teams and appearance comparisons | `src/futsal/teams.py`, `team_comparison.py` |
| Court and homography | `src/futsal/court.py`, `geometry.py`, `notebook_ui.py` |
| Radar and overlays | `src/futsal/radar.py`, `rendering.py` |
| Analytics/evaluation/export | `src/futsal/analytics.py`, `quality.py`, `evaluation.py`, `tracking_eval.py`, `exports.py` |
| Processing and jobs | `src/futsal/pipeline.py`, `cli.py`, `jobs.py` |
| Interface | `dashboard/app.py`, `app/streamlit_app.py` |
| Colab/configuration/tests | `notebooks/`, `configs/`, `tests/` |

The version-5 dataset ZIP has 192 training, 29 validation and 19 test images. Its labels match the model. These splits alone do not establish match independence. Earlier dataset versions and review history remain intact.

## References and licenses

Adapted after inspecting the [Roboflow Sports source](https://github.com/roboflow/sports), [original notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/football-ai.ipynb) and sampled key sections of the provided tutorial. See [reference audit](docs/reference_audit.md). Tracking follows [Ultralytics interfaces](https://docs.ultralytics.com/modes/track/). The futsal penalty template uses the 3.16 m join in the [laws published by the Japan Football Association](https://www.jfa.jp/laws/futsal/2025_26/).

Ultralytics is AGPL-3.0; Roboflow Sports source is MIT. Model/dataset terms are separate. The supplied dataset declares CC BY 4.0. See `THIRD_PARTY_NOTICES.md` before redistribution.
