"""Build a self-contained Colab notebook embedding this project's actual Python source."""

import base64
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    files = [ROOT / name for name in ("pyproject.toml", "THIRD_PARTY_NOTICES.md", "README.md",
                                     "requirements.txt", "launch.ps1", ".streamlit/config.toml")]
    files += sorted((ROOT / "src").rglob("*.py"))
    files += sorted((ROOT / "src").rglob("*.txt"))
    files += sorted((ROOT / "app").glob("*.py"))
    files += sorted((ROOT / "tests").glob("*.py"))
    files += sorted((ROOT / "scripts").glob("*.py"))
    files += sorted((ROOT / "scripts").glob("*.ps1"))
    files += sorted((ROOT / "docs").glob("*.md"))
    files += sorted((ROOT / "configs").glob("*.yaml"))
    files += sorted((ROOT / "configs").glob("*.json"))
    files += sorted(p for p in (ROOT / "dashboard").rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    bundle = io.BytesIO()
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.writestr(path.relative_to(ROOT).as_posix(), path.read_bytes())
    encoded = base64.b64encode(bundle.getvalue()).decode()
    cells = []

    def md(s):
        cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip().splitlines(True)})

    def code(s):
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                      "outputs": [], "source": s.strip().splitlines(True)})

    md('''# Futsal AI — your existing YOLO weights to tactical radar
Use **Runtime → Change runtime type → T4 GPU**. This notebook embeds the project's source,
uses your `best.pt`, and contains no detector training or replacement-model downloads.
Core inference needs no Roboflow or Hugging Face API key. Run one milestone, inspect its
evidence, then enable the next. Download outputs before the Colab session expires.

Local validation used Python 3.11, torch 2.5.1+cu121 and Ultralytics 8.3.240. Your checkpoint
reports training with 8.4.182, but loaded and ran locally with 8.3.240. The notebook retains
Colab's PyTorch/CUDA installation. T4 speed and Colab compatibility need runtime validation.
''')
    code('''import os, sys, subprocess, base64, io, zipfile
from pathlib import Path
os.environ['YOLO_AUTOINSTALL'] = 'false'
PROJECT = Path('/content/futsal-ai')
PROJECT.mkdir(exist_ok=True)
'''+f"SOURCE_BUNDLE = {encoded!r}\n"+'''
with zipfile.ZipFile(io.BytesIO(base64.b64decode(SOURCE_BUNDLE))) as archive:
    archive.extractall(PROJECT)
# Pin the version actually exercised locally, rather than automatically upgrading.
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT)+'[colab]'])
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / 'src'))
print('Source installed:', PROJECT)
''')
    md('''## 1. Select your weights and video
Drive is recommended for large matches. Set `INPUT_MODE='drive'` and edit both paths, or
use direct uploads. `data.yaml` is optional and is only inspected; no dataset is rebuilt.
''')
    code('''INPUT_MODE = 'drive'  # 'drive' or 'upload'
WEIGHTS_PATH = '/content/drive/MyDrive/FutsalAI/best.pt'
VIDEO_PATH = '/content/drive/MyDrive/FutsalAI/match.mp4'
DATA_YAML_PATH = ''  # optional local/Drive path

if INPUT_MODE == 'drive':
    from google.colab import drive
    drive.mount('/content/drive')
elif INPUT_MODE == 'upload':
    from google.colab import files
    print('Upload best.pt')
    uploaded = files.upload()
    weights = [PROJECT / name for name in uploaded if name.lower().endswith('.pt')]
    if len(weights) != 1:
        raise ValueError('Select exactly one .pt weights file')
    WEIGHTS_PATH = str(weights[0])
    del uploaded
    print('Upload one MP4 match')
    uploaded = files.upload()
    videos = [PROJECT / name for name in uploaded if name.lower().endswith('.mp4')]
    if len(videos) != 1:
        raise ValueError('Select exactly one MP4')
    VIDEO_PATH = str(videos[0])
    del uploaded
else:
    raise ValueError('INPUT_MODE must be drive or upload')
assert Path(WEIGHTS_PATH).is_file(), 'Weights not found; edit WEIGHTS_PATH'
assert Path(VIDEO_PATH).is_file(), 'Video not found; edit VIDEO_PATH'
''')
    code('''import json, torch, ultralytics, yaml
from ultralytics import YOLO
from futsal.diagnostics import inspect_model, detection_check
from futsal.video import probe, frame_at

DEVICE = '0' if torch.cuda.is_available() else 'cpu'
print('torch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'device:', DEVICE)
if DEVICE != 'cpu':
    print('GPU:', torch.cuda.get_device_name(0))
    print('CUDA tensor check:', torch.ones(1, device='cuda').sum().item())
else:
    print('CPU mode is supported but slow. Select a T4 runtime for match processing.')
model = YOLO(WEIGHTS_PATH)
MODEL_INFO = inspect_model(model, WEIGHTS_PATH)
print(json.dumps(MODEL_INFO, indent=2, default=str))
if model.task != 'detect':
    raise ValueError('This pipeline requires an object detection checkpoint')
if DATA_YAML_PATH:
    dataset = yaml.safe_load(Path(DATA_YAML_PATH).read_text())
    print('Dataset names:', dataset.get('names'))
    print('Checkpoint names are authoritative:', model.names)
print(probe(VIDEO_PATH))
del model
if DEVICE != 'cpu': torch.cuda.empty_cache()
''')
    md('''## 2. Detection evidence — first acceptance gate
Set the source interval to visible live play. Start with 1280 because the supplied YOLO12x
was trained at that size. A blue-court clip worked locally; black-court samples performed poorly.
The raw video preserves **all actual class names**, including `player- ball`. Tactical analysis
excludes that unsupported mixed class; it is never relabelled as referee.
''')
    code('''import uuid
from IPython.display import display, Image, Video
OUTPUTS = PROJECT / 'outputs'
OUTPUTS.mkdir(exist_ok=True)
START_S = 600.0  # change to an actual wide-view interval in YOUR selected match
DURATION_S = 10.0
SAMPLE_FPS = 15
IMGSZ = 1280
ATTEMPT = uuid.uuid4().hex[:8]
DETECTION_DIR = OUTPUTS / f'01_detection_{ATTEMPT}'
report = detection_check(VIDEO_PATH, WEIGHTS_PATH, DETECTION_DIR,
    start=START_S, duration=DURATION_S, fps=SAMPLE_FPS, imgsz=IMGSZ,
    device=DEVICE, confidence=.15)
display(Image(filename=str(DETECTION_DIR / 'representative_frames.jpg')))
display(Video(str(DETECTION_DIR / 'detections.mp4'), embed=True))
print({k: report[k] for k in ['frames', 'class_counts', 'inference_median_ms', 'processing_fps']})
print('No independent accuracy claim; inspect misses, false positives and role confusions.')
''')
    md('''## 3. ByteTrack — separate people and ball
Enable only after inspecting the detection video. IDs are local to each camera shot and run;
they are not jersey numbers or guaranteed identities. Missing observations are not interpolated.
Switch `TRACKER='botsort'` to compare camera-motion compensation using the same detector.
''')
    code('''from futsal.config import RunConfig
from futsal.pipeline import run_pipeline
from futsal.quality import summarize_run
DETECTIONS_REVIEWED = False  # set True after examining milestone 2
TRACKER = 'bytetrack'  # alternative: 'botsort'
TRACKING_DIR = None
if DETECTIONS_REVIEWED:
    TRACKING_DIR = OUTPUTS / f'02_tracking_{ATTEMPT}'
    config = RunConfig(model=WEIGHTS_PATH, device=DEVICE, imgsz=IMGSZ,
        start=START_S, duration=DURATION_S, sample_fps=SAMPLE_FPS,
        tracker=TRACKER, team_mode='off', radar=False)
    run_pipeline(VIDEO_PATH, TRACKING_DIR, config)
    print(summarize_run(TRACKING_DIR))
    display(Video(str(TRACKING_DIR / 'annotated.mp4'), embed=True))
else:
    print('Inspect detections, then set DETECTIONS_REVIEWED=True to proceed.')
''')
    md('''### Optional motion-aware comparison with the same detector
After reviewing the ByteTrack baseline, enable this cell to compare BoT-SORT camera-motion
compensation, 20 fps sampling, confirmed track starts and display-only adaptive smoothing.
Missing observations remain missing. Extra ball search uses the same detector on overlapping
crops and is slower. Optional appearance matching needs separately supplied local OSNet weights;
it remains experimental for athletes wearing the same kit. These controls do not retrain YOLO.
''')
    code('''RUN_MOTION_COMPARISON = False
EXTRA_BALL_SEARCH = False
USE_APPEARANCE_REID = False
REID_WEIGHTS_PATH = ''  # optional local/Drive OSNet checkpoint, required only when enabled
if RUN_MOTION_COMPARISON and TRACKING_DIR:
    if USE_APPEARANCE_REID and not Path(REID_WEIGHTS_PATH).is_file():
        raise ValueError('Provide your local OSNet checkpoint or leave appearance matching disabled')
    motion_config = config.model_copy(update={
        'tracker': 'botsort', 'sample_fps': 20, 'confidence': .05,
        'person_confidence': .05, 'ball_confidence': .05,
        'track_low_threshold': .05, 'track_high_threshold': .25,
        'track_new_threshold': .35, 'track_buffer_seconds': 1.5,
        'track_min_observations': 2, 'smooth_boxes': True,
        'ball_confirmation_frames': 2, 'ball_start_confidence': .2, 'ball_max_gap': .25,
        'tile_ball': EXTRA_BALL_SEARCH, 'ball_tile_imgsz': 960,
        'appearance_reid': USE_APPEARANCE_REID, 'reid_checkpoint': REID_WEIGHTS_PATH})
    MOTION_DIR = OUTPUTS / f'02_motion_{ATTEMPT}'
    run_pipeline(VIDEO_PATH, MOTION_DIR, motion_config)
    print('Baseline:', summarize_run(TRACKING_DIR))
    print('Motion-aware:', summarize_run(MOTION_DIR))
    display(Video(str(MOTION_DIR / 'annotated.mp4'), embed=True))
    # The next stages use this configuration. Review this video before continuing.
    config, TRACKING_DIR = motion_config, MOTION_DIR
''')
    md('''## 4. Team classification and appearance comparison
The active baseline uses jersey LAB features, fixed match-specific clusters and temporal track
votes. A/B are display labels, not known club names. Goalkeepers remain unknown until reviewed;
the detector has no referee class. The optional SigLIP encoder downloads appearance weights
only; it never replaces `best.pt`. Silhouette is cluster separation, not team accuracy.
''')
    code('''TRACKING_REVIEWED = False
TEAM_DIR = None
if DETECTIONS_REVIEWED and TRACKING_REVIEWED:
    TEAM_DIR = OUTPUTS / f'03_teams_{ATTEMPT}'
    config = config.model_copy(update={'team_mode': 'color'})
    run_pipeline(VIDEO_PATH, TEAM_DIR, config)
    print(summarize_run(TEAM_DIR))
    display(Video(str(TEAM_DIR / 'annotated.mp4'), embed=True))
else:
    print('Review the tracking video, then set TRACKING_REVIEWED=True.')
''')
    code('''RUN_TEAM_COMPARISON = False
INCLUDE_SIGLIP = False  # optional download; install embedding dependencies below when enabled
if RUN_TEAM_COMPARISON and TEAM_DIR:
    if INCLUDE_SIGLIP:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT)+'[embeddings]'])
    from futsal.team_comparison import compare_teams
    comparison = compare_teams(VIDEO_PATH, TEAM_DIR / 'detections.jsonl',
        OUTPUTS / f'team_comparison_{ATTEMPT}', include_siglip=INCLUDE_SIGLIP,
        device='cuda' if DEVICE != 'cpu' else 'cpu')
    print({k: v['silhouette'] for k, v in comparison['methods'].items()})
    import matplotlib.pyplot as plt
    for method in comparison['methods']:
        import numpy as np
        features = np.load(OUTPUTS / f'team_comparison_{ATTEMPT}' / f'{method}_features.npy')
        plt.figure(figsize=(6, 4))
        plt.scatter(features[:,0], features[:,1], c=comparison['methods'][method]['labels'], cmap='coolwarm')
        plt.title(method + ' — unverified appearance clusters'); plt.show()
''')
    md('''## 5. Court lines, named landmarks, RANSAC homography
OpenCV produces **image-space line candidates**, including false positives from logos and boards.
They do not establish semantic correspondences. The first radar uses explicitly selected floor
landmarks and at least one independent validation point. No outdoor-football keypoint model is reused.
''')
    code('''from futsal.court import line_candidates
import cv2
image = frame_at(VIDEO_PATH, START_S)
line_image = image.copy()
for x1,y1,x2,y2 in line_candidates(image):
    cv2.line(line_image, (x1,y1), (x2,y2), (0,220,255), 2)
display(Image(data=cv2.imencode('.jpg', line_image)[1].tobytes()))
''')
    code('''# Enable this cell when you have inspected the team video and can identify floor landmarks.
PICK_CALIBRATION = False
CALIBRATION_FILE = None
if PICK_CALIBRATION:
    from futsal.notebook_ui import pick_calibration
    anchor = pick_calibration(image, START_S, length=40, width=20)
    print(anchor.model_dump_json(indent=2))
''')
    code('''CORRESPONDENCES_REVIEWED = False
COURT_SCALE_INDEPENDENTLY_VERIFIED = False
if PICK_CALIBRATION and CORRESPONDENCES_REVIEWED:
    from futsal.geometry import fit_calibration
    from futsal.storage import write_json, sha256
    anchor = anchor.model_copy(update={'reviewed': True,
        'scale_verified': COURT_SCALE_INDEPENDENTLY_VERIFIED,
        'source_sha256': sha256(VIDEO_PATH)})
    matrix, calibration_quality = fit_calibration(anchor)
    print(calibration_quality)
    if calibration_quality['valid']:
        CALIBRATION_FILE = str(OUTPUTS / f'calibration_{ATTEMPT}.json')
        write_json(CALIBRATION_FILE, [anchor.model_dump()])
    else:
        print('Calibration rejected. Revisit correspondences; metric results remain unavailable.')
''')
    md('''## 6. Tactical radar and synchronized export
Valid calibration is propagated for a short time using background optical flow. Camera cuts,
unsupported motion and expired anchors invalidate positions. Add more reviewed anchors for longer
clips. Assumed 40×20 m dimensions allow an explicitly labelled geometric preview; distances,
speeds and event statistics still require verified scale and reviewed live-play observations.
The ball appears on the radar only when its ground location has been reliably reviewed.
''')
    code('''TEAMS_REVIEWED = False
TACTICAL_DIR = None
if TEAM_DIR and TEAMS_REVIEWED:
    TACTICAL_DIR = OUTPUTS / f'04_tactical_{ATTEMPT}'
    tactical_config = config.model_copy(update={
        'radar': True, 'radar_overlay': True, 'show_trails': True,
        'calibration_file': CALIBRATION_FILE,
        'scale_verified': COURT_SCALE_INDEPENDENTLY_VERIFIED,
        'team_colors': {'team_a':'#3498ff', 'team_b':'#f05252'},
        'calibration_max_age': 2})
    run_pipeline(VIDEO_PATH, TACTICAL_DIR, tactical_config)
    print(summarize_run(TACTICAL_DIR))
    display(Video(str(TACTICAL_DIR / 'tactical.mp4'), embed=True))
    display(Video(str(TACTICAL_DIR / 'radar.mp4'), embed=True))
    if CALIBRATION_FILE is None:
        print('Radar correctly reports positions unavailable until calibration is supplied.')
else:
    print('Review team assignments before enabling tactical export.')
''')
    md('''## 7. CSV / JSON / Parquet and downloads
Source PTS and seconds are authoritative; `proxy_index` is the sampled output frame number.
Videos are sampled silent H264 analysis proxies, not complete source-frame reproductions.
''')
    code('''from futsal.storage import load_tracks
from google.colab import files
import shutil
RUN_DIR = TACTICAL_DIR or TEAM_DIR or TRACKING_DIR
if RUN_DIR:
    observations = load_tracks(RUN_DIR)
    observations.to_csv(RUN_DIR / 'tracks.csv', index=False)
    observations.to_json(RUN_DIR / 'tracks.json', orient='records', indent=2)
    observations.to_parquet(RUN_DIR / 'tracks.parquet', index=False)
    print('Observations:', len(observations))
archive = shutil.make_archive(str(OUTPUTS / f'futsal_analysis_{ATTEMPT}'), 'zip',
    root_dir=OUTPUTS, base_dir=(RUN_DIR or DETECTION_DIR).name)
files.download(archive)
''')
    md('''## Running on Windows and next evaluation gates
Unzip the source bundle into a dedicated project folder, place your `best.pt` there, install the
appropriate PyTorch wheel, then `pip install -e .` and `python -m futsal dashboard`.
The interface includes calibration and team/role correction tools. The existing analytics module
requires reviewed segments and scale; no possession, pass or speed accuracy is invented here.
Next acceptance work: independent match-level detection labels, IDF1/HOTA tracking labels,
team truth, futsal landmark truth and independently measured position references.

References: [Roboflow Sports](https://github.com/roboflow/sports),
[original notebook](https://colab.research.google.com/github/roboflow-ai/notebooks/blob/main/notebooks/football-ai.ipynb),
[Ultralytics tracking](https://docs.ultralytics.com/modes/track/).
''')
    notebook = {"cells": cells, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                "language_info": {"name": "python"}, "accelerator": "GPU", "colab": {"provenance": []}},
                "nbformat": 4, "nbformat_minor": 5}
    for i, cell in enumerate(cells):
        cell["id"] = f"futsal-{i:03d}"
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell_{i}", "exec")
    folder = ROOT / "notebooks"
    folder.mkdir(exist_ok=True)
    (folder / "futsal_ai_colab.ipynb").write_text(json.dumps(notebook, indent=1), "utf-8")
    (folder / "futsal_ai_source.zip").write_bytes(bundle.getvalue())
    print(f"Built {len(cells)} cells, {len(bundle.getvalue())} source ZIP bytes")


if __name__ == "__main__":
    build()
