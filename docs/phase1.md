# Phase 1: tracking and shirt-color assignment

The complete modular implementation is in `src/futsal/phase1/`. It uses your
RF-DETR detection model, Supervision ByteTrack/annotations, OpenCV, NumPy,
scikit-learn K-Means, and pandas CSV export. The five-phase architecture is in
[phase1_architecture.md](phase1_architecture.md). Phase 2 is now available in
[phase2.md](phase2.md); Phases 3–5 await your confirmation.

## Start in this workspace (PowerShell)

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-phase1.txt
$env:ROBOFLOW_API_KEY = 'YOUR_ROBOFLOW_API_KEY'
.\.venv\Scripts\python.exe -m futsal.phase1 --video 'D:\path\match.mp4' --output runs/phase1-preview --config configs/phase1.json --max-frames 600
```

Use a real video path and a new output directory. Start with a short segment showing
both teams. For a broadcast with an intro, specify `--start-frame 18000` (for example,
ten minutes at 30 FPS). Frame indices are **absolute, zero-based source frame numbers**.
For a separate color-calibration segment use `--calibration-start 18000`. Do not
put keys in config, source code, notebooks or command arguments; the adapter reads
`ROBOFLOW_API_KEY`. The Roboflow connector's authorization is not automatically
available to a Python subprocess.

The existing workspace environment already has the base scientific packages; the
additional Supervision dependency was installed for local validation. Alternatively,
`pip install -e ".[phase1]"` installs this repository's full dependency set and a
`futsal-phase1` console command. The old `futsal` CLI remains unchanged.

For a **new minimal environment**, use Python 3.11 or 3.12:

```powershell
py -3.11 -m venv .venv-phase1
.\.venv-phase1\Scripts\python.exe -m pip install -r requirements-phase1.txt
.\.venv-phase1\Scripts\python.exe -m pip install -e . --no-deps
$env:ROBOFLOW_API_KEY = 'YOUR_ROBOFLOW_API_KEY'
.\.venv-phase1\Scripts\python.exe -m futsal.phase1 --video 'D:\path\match.mp4' --output runs/phase1-preview --config configs/phase1.json --max-frames 600
```

`--no-deps` here is intentional: the minimal requirements have already been installed,
and the repository's training/dashboard dependencies are not required by Phase 1.

## What the code does

1. Opens and validates a local constant-frame-rate video.
2. Samples up to 40 calibration frames, every 15 source frames by default. Uses
   only high-confidence detections with the **class name** `player`.
3. Crops the central torso within the upper half: horizontal 20–80%, vertical
   15–50% of the box. Extracts the dominant quantized HSV color, representing hue
   on a circle and retaining low-saturation white/gray/black shirts.
4. Fits K-Means once for the selected match. Rejects insufficient, near-identical,
   or very imbalanced color clusters. Writes a representative crop montage.
5. Reopens the requested segment and detects **every frame**. Passes player and
   goalkeeper boxes through the same ByteTrack instance. Empty frames advance
   tracker state. Low-confidence detections are retained for ByteTrack association.
6. Predicts a player's color group and applies a bounded per-track voting history.
   Ambiguous/outlier colors remain unknown. Known goalkeeper/referee roles bypass
   the team classifier. Role and team are stored separately.
7. Draws boxes/IDs with Supervision using a separate color lookup, preserving the
   model's original class IDs. Streams MP4 and CSV output rather than holding a
   match's frames or observations in RAM.
8. Resets tracker/team state at detected hard cuts or supplied `--cut-frame` values.
   IDs stay unique within the run after resets. Writes a completion/failure manifest
   and reopens the encoded video to verify it is readable.

The calibration pass incurs additional inference calls: with default settings a
600-frame preview uses up to 40 calibration calls plus 600 processing calls (and
bounded retries only after transient errors). Hosted processing uploads JPEG frames
to your configured Roboflow endpoint. This synchronous implementation prioritizes
simple ordered tracking; its throughput depends on inference/network latency.

## Outputs

| File | Purpose |
| --- | --- |
| `annotated.mp4` | Same FPS as input; player/GK boxes and track IDs; silent MP4V video |
| `tracks.csv` | Source frame/time, segment, ID, detector role, team, vote agreement, confidence, boxes and bottom-center anchor |
| `calibration.jpg` | Real torso crops closest to each color center, labeled with cluster indices |
| `teams.json` | Learned color centers, sample counts, display labels and review reminder |
| `run.json` | Source/configuration, package versions, completion status and observation counts |

Team A is blue, Team B pink, goalkeepers purple, referees yellow (only if a detector
actually supplies that class), Other orange, Unknown gray. These are overlay colors;
they are not an assertion of the actual jersey color or club name. A ball is excluded
from person MOT and from Phase 1's CSV/video annotations; its pipeline belongs in Phase 4.
`foot_x_px/foot_y_px` are box bottom centers, **not measured anatomical foot keypoints**.
`team_vote_fraction` is temporal agreement, not calibrated probability.

## Reviewing team groups

Open `calibration.jpg` and check that both teams' crops are consistent. K-Means does
not know club names. To swap A and B, set:

```json
"teams": {"cluster_labels": ["team_b", "team_a"]}
```

The model's detector classes are `ball`, `goalkeeper`, `player`; **there is no
referee detector**. Default clustering therefore uses two outfield-shirt groups,
with detected goalkeepers kept separate. The optional three-color mode is:

```json
"teams": {"cluster_labels": ["team_a", "team_b", "other"]}
```

Run a preview, inspect the montage, then reorder those three labels to match the
visible groups. `other` is an unverified color group, not a referee prediction.
Three-way K-Means can split one team's shirts by illumination, and two goalkeeper
kits need not share a color. This mode cannot recover referees the detector missed.
Goalkeepers' club affiliation remains unknown; automatic nearest-team-centroid
assignment would be unreliable when players move or camera views change.

The upper-body color method remains vulnerable to similar uniforms, stripes,
occlusion, floor contamination and lighting. Adjust the calibration segment before
loosening quality thresholds. A/B label ordering is only stable for the fitted
calibration; do not assume consistent club mapping across separate runs.

## Python API

```python
import os
from pathlib import Path
from futsal.phase1.config import Phase1Config
from futsal.phase1.detection import RoboflowDetector
from futsal.phase1.pipeline import Phase1Pipeline

config = Phase1Config.load(Path("configs/phase1.json"))
detector = RoboflowDetector(config.detector, os.environ["ROBOFLOW_API_KEY"])
try:
    Phase1Pipeline(detector, config).run(
        video=Path("match.mp4"),
        output=Path("runs/phase1-preview"),
        start_frame=0,
        max_frames=600,
    )
finally:
    detector.close()
```

`Detector` is a protocol: another backend only needs
`detect(frame: np.ndarray) -> supervision.Detections`, containing confidences,
class IDs, and `data['role']` / `data['class_name']`. Current local Roboflow Inference
servers can use `detector.api_url = "http://localhost:9001"`; bearer-header
authentication requires server release 1.5.0 or newer. The adapter uses the official
`/<workspace>/<model-slug>` HTTP model endpoint. The class mapping is configurable
and does not assume numeric detector IDs.

## Validation and boundaries

```powershell
.\.venv\Scripts\python.exe -m pytest tests/phase1 -q
.\.venv\Scripts\python.exe -m ruff check src/futsal/phase1 tests/phase1
```

Tests cover actual Supervision ByteTrack short-gap recovery, low-score association,
track expiry, unique IDs after cuts, hue wraparound, achromatic shirts, rejected
crops/colors, optional three-way mapping, voting, HTTP request shape/retries, output
preservation and an end-to-end synthetic MP4-to-MP4/CSV pipeline. Synthetic tests
establish implementation behavior, not model accuracy on a real match. Live HTTP
inference is not part of the test suite. No local API key was available during this
delivery, so the exact detector's real-match pipeline remains to be run with your key.

This is a production-oriented Phase 1 foundation, not a validated broadcast product:

- ByteTrack can switch identities at crossings, severe occlusions, camera motion
  or re-entry. Track IDs are not player names or jersey numbers. Long-term identity
  requires appearance/jersey evidence and labeled tracking validation.
- The hard-cut detector is a histogram heuristic; dissolves and similar-court cuts
  may be missed. `--cut-frame 1234 --cut-frame 5678` supplies reviewed boundaries.
  This does not classify replays/live play. Camera-motion compensation is not implemented.
- Use constant-frame-rate clips. Phase 1 timestamps are `frame_index / fps`; variable
  frame rate requires normalization or a PTS-aware reader before metric analysis.
- MP4V output is silent; browser-compatible H.264 and audio-preserving muxing belong
  to Phase 5. Odd source dimensions are padded on the bottom/right for codec safety.
- Existing run directories are refused. On failure, the manifest is `failed`; partial
  files remain for diagnosis and must not be treated as complete output.

The second supplied model was verified as **pitch keypoints**, suitable for Phase 2.
Phase 4's ball-to-feet method will require a separate human-pose model.
