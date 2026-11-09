# Supplied-model tactical pipeline: evidence and limits

20 October 2026. No detector training, replacement detector, dataset reconstruction or cloud
upload was performed. Inference used root `best.pt`, SHA256
`e3d8d18b0461a93a80a7b9d627bc7bf7452e1e5f5d073395f92b646fc1c8415f`.

## Model and references

YOLO12x, DetectionModel, detect task, 59,123,004 parameters. Actual labels: ball, goalkeeper,
player, player- ball. There is no referee class. Configured training: 50 epochs, size 1280,
batch 1. Checkpoint metrics are historical reported values, not independent accuracy evidence.
Runtime: Python 3.11, torch 2.5.1+cu121, Ultralytics 8.3.240, GTX 1650 Ti. Checkpoint training
metadata reports Ultralytics 8.4.182; loading and inference worked with the pinned local version.

Inspected sports-main ZIP and the upstream 62-cell notebook. Reviewed six sampled visual
sections of the supplied tutorial, including appearance embeddings and radar. Audio was not
transcribed; this is not a claim to have watched every minute. The source notebook is saved
in `reports/reference_sources/football-ai.ipynb`. The version-5 dataset ZIP contains 192 train,
29 validation and 19 test images with the checkpoint's four labels. No labels were changed.
Those splits alone do not establish independent source matches.

## Progressive executions

| Stage | Output folder | Evidence |
|---|---|---|
| Retained failure | `outputs/01_detection` | Palma 900–908 s at 640 and confidence .15: no detections |
| Detection | `outputs/01_detection_blue_court` | France–Argentina 600–608 s; 120 frames; 749 player, 70 goalkeeper, 5 ball observations at .15 |
| Tracking | `outputs/02_tracking` | Full inference at confidence .08; ByteTrack; 22 tracklets |
| Team baseline | `outputs/03_teams` | Full inference; LAB clusters, temporal votes, unknown goalkeeper teams |
| First radar | `outputs/04_radar` | Spain–Portugal 120–128 s; full inference and manual calibration |
| Refined tracking | `outputs/05_tracking_refined` | Compatible detection replay with duplicate role boxes suppressed |
| Refined teams | `outputs/06_teams_refined` | Same replay, compact readable labels and temporal team estimates |
| Refined tactical output | `outputs/07_tactical_refined` | Radar video, broadcast inset, CSV/JSON observations |

Counts are per-frame box observations, not unique players or accuracy metrics. Diagnostic and
tracking confidence thresholds differ, so their counts are not directly comparable.

The France diagnostic measured median inference at about 335 ms/frame and total processing
at 2.83 fps. The original Spain radar execution processed about 1.77 fps. The exported sample
cadence is 15 fps, independent of wall-clock processing speed. T4 speed has not been measured.
Refined runs reuse saved detections; their higher throughput is not inference performance.

Spain initially contained 264 near-duplicate person boxes among 1,354 person observations
(IoU > .8). Role-agnostic suppression reduced input to 1,090 people observations and tracklets
from 34 to 19 on the same clip. This does not prove an IDF1/HOTA improvement. Detector misses,
role confusion and tracking fragmentation remain visible. Unobserved tracker predictions are
not exported as observed player positions.

## Team comparison

Refined LAB silhouette: approximately .750; HSV: .619. These measure cluster separation,
not team accuracy. About 77% of the France tracked person observations receive an A/B label;
the denominator includes startup, unknowns and unclassified goalkeepers. SigLIP→UMAP→KMeans
is implemented as an optional comparison path and notebook cell, but was not run locally.
Goalkeeper/player confusion is visible in Spain; referees cannot be automatically identified
by a checkpoint without that class. Manual team and role corrections remain available.

## Court and homography

The first radar is **manually anchored**, not automatic futsal landmark detection. OpenCV
supplies image-space line proposals only. Four fitting landmarks were visually identified:
both centre-circle/halfway intersections, the far right floor corner and the right second
penalty mark. The two withheld landmarks were halfway intersections with the far and near
touchlines. Maximum withheld reprojection error was **3.64 px** on the 1920×1080 anchor frame.
The agent visually checked the projected court overlay. Provenance is in
`reports/calibration_demo_provenance.json`; calibration is source-checksummed in
`configs/spain_demo_calibration.json`. This is not independent physical position accuracy.

The demo assumes 40×20 m; `scale_verified` stays false. It propagates this anchor for eight
seconds; default interactive configurations expire after two seconds. During propagation,
fresh absolute reprojection error is unknown and exported as null. Anchor error and motion
inlier ratio are separately recorded. All 120 demo frames retained the calibration gate:
that is coverage, not independently measured positional correctness. Cuts, unsupported motion
and expired anchors invalidate positions. The ball is not projected unless its grounded
location has been reliably reviewed. No position is invented when the ball is airborne.

## Acceptance limits

No reviewed match-level detection benchmark, identity truth, team truth, futsal landmark
truth or independent physical reference is available. Precision, recall, IDF1, HOTA, team
accuracy and metre position error remain null. Black-court samples perform poorly even at
1280; ball coverage is low in the France example. No passes, turnovers, formations, real
identities, measured distances or possession percentages are asserted. Existing advanced
analytics remain gated pending reliable reviewed inputs.

The notebook embeds executable project source. Its Python cells were syntax checked and
the core pipeline exercised locally. Colab/T4 execution and optional SigLIP downloads still
need validation in that external runtime.

Software validation: **58 tests passed** in the full local pytest run; Ruff checks passed.
All six delivered demonstration MP4 files decoded to 120 frames at 15 fps with strictly
increasing timestamps. The source ZIP passed CRC and Python compilation checks, and the
CSV/JSON/Parquet/video export ZIP passed its integrity check.
