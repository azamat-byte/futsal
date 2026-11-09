# Detection and tracking optimization without retraining

All object detections still come from the supplied `best.pt`. Its weights were not altered.
No replacement player/ball detector was enabled and no training was performed.

## Recommended operation

Launch `launch.ps1`, then use Overview:

- Detector: `best.pt`; inference size: 1280; device: 0.
- Analysis samples/second: 20, or the source frame rate if lower.
- Tracker: BoT-SORT; **Use motion-aware tracking settings** enabled.
- Start with 5–10 seconds of visible live play.
- Enable **Extra ball search with the selected detector** only when the extra processing cost is acceptable.
- Keep appearance matching disabled initially. It is an optional experimental comparison, not a proven upgrade.
- Radar still requires a valid calibration and is independent of whether detections can be tracked.

CLI equivalents:

    .\.venv\Scripts\python.exe -m futsal process 'YOUR_MATCH.mp4' --config configs/motion_aware.yaml --start 0 --duration 10
    .\.venv\Scripts\python.exe -m futsal process 'YOUR_MATCH.mp4' --config configs/small_objects.yaml --start 0 --duration 10

For a reproducible tracker comparison using the same detector outputs and source timestamps:

    .\.venv\Scripts\python.exe scripts/compare_tracking.py 'YOUR_MATCH.mp4' outputs/my-comparison --start 120 --duration 4 --fps 20

Add `--appearance` to compare the existing local OSNet encoder. New output directories are required.
The Colab notebook also contains an optional motion-aware comparison cell.

## What changed

Detection remains conservative. Person roles share duplicate suppression, while ball boxes are
merged separately. Overlapping ball crops can recover small objects that full-frame resizing
misses; nested duplicate ball boxes are removed before temporal association. Crop inference
defaults to 960 to limit local GPU memory use. Full-frame inference remains 1280.
An optional full-frame-plus-tiles mode and mild luminance CLAHE are available for controlled
tests, but neither improved the uploaded light-court sample enough to recommend as a default.
Separate person and ball score floors avoid discarding recovery candidates prematurely.

Tracking uses BoT-SORT background-motion compensation and a 1.5-second lost-track buffer.
High/low/new score thresholds are .25/.05/.35 in the motion profile. Two observed updates
are required before a new track is shown, including on the first frame. Low-score detections
can recover existing tracks; they do not independently create identities. Short timestamp
gaps advance the tracker prediction clock instead of being labelled as broadcast cuts.
Actual visual cuts and long unsupported gaps still reset tracking and calibration.

Optional OSNet appearance embeddings are supplied directly to BoT-SORT with a proximity gate.
They use the already available local person encoder and never replace the YOLO detector.
Generic pedestrian embeddings are not a verified identity solution for same-kit futsal players.

Adaptive smoothing affects **display boxes only**. Raw detector measurements remain in x1/y1/x2/y2
and supply positional analytics. The separately exported `display_box` records the smoothed
overlay. Filters reset after gaps and cuts. No boxes or floor positions are fabricated during
occlusion; missing detections stay missing.

Ball observations are associated separately using a short-horizon motion prediction. The motion
profile requires two consistent observations with adequate startup confidence. Isolated flashes,
ambiguous candidates and missing frames are withheld. Confirmed reacquisition after a motion
discontinuity does not interpolate unseen ball locations. This can reduce coverage while trading
against false positives; coverage is not accuracy.

A manual fixed-camera ROI can exclude spectators and staff. `court_roi_file` must contain
`source_sha256`, `image_size: [width, height]` and a 3+ point `polygon` in original pixel coordinates.
It is checked against this source and image size. Do not reuse a fixed pixel ROI for a panning
or zooming camera. No automatic mask is trusted to delete players.

Long crop runs publish per-frame progress and flush timestamp logs. Track batches become
inspectable during processing. Interrupted artifacts are excluded from completed comparisons.

## Validation and limits

Software checks cover tile-edge coverage, person/ball threshold separation, duplicate suppression,
manual ROI rejection, low-score recovery, appearance-feature integration, first-frame false-track
confirmation, temporal ball confirmation, display jitter reduction and reset after gaps.
They do not establish sporting accuracy. The final local test suite passed all 73 tests in
33.47 seconds; Ruff also passed. The weights' SHA-256 remains
`e3d8d18b0461a93a80a7b9d627bc7bf7452e1e5f5d073395f92b646fc1c8415f`.

### Completed local comparisons

| Check | Baseline | Updated | Interpretation |
| --- | --- | --- | --- |
| Spain–Portugal 120–123s, same 30 frames: accepted ball observations | 5/30 | 17/30 | Greater observed coverage with extra 960-pixel ball crops; not measured accuracy |
| Same ball comparison, full inference processing rate | 1.24 fps | 0.275 fps | Optional extra search costs substantially more processing time |
| Spain–Portugal 120–124s, same 80 frames and cached detections: player tracklets | 11 | 9 | Less fragmentation by this diagnostic; identity correctness is unmeasured |
| Uploaded `download.mp4`, first 14s at 20 samples/s | Original sparse run: no tracks | 280 frames, 23 person candidates, no accepted tracks | The supplied detector still does not recognize this match reliably |

The tracker comparison uses identical detections. Its updated run replays cached detections,
so its reported throughput must not be interpreted as full inference speed. Ball and tracker
comparisons use different frame cadences; compare each matched pair only. Measurements were
made on the local GTX 1650 Ti, not a Colab T4. Short clips do not establish full-match reliability.

Nine accepted ball observations sampled across the updated run were visually inspected and
followed the visible ball. This is qualitative review, not an independently labeled accuracy test.
The stricter confirmation filter also withholds more full-frame ball candidates when extra
search is disabled. Review the paired videos before choosing the cost/coverage tradeoff.

Videos: `outputs/optimization/ball_before_after.mp4` and
`outputs/optimization/tracking_before_after.mp4`. Per-run provenance, coverage and unavailable
accuracy fields are preserved in the completed run manifests and `quality_report.json` files.

Actual inference sweeps, images and JSON records are in `reports/optimization/`:

- Uploaded light-court frames at 2, 5 and 10 seconds: ordinary inference, larger 1920 inference,
  mild CLAHE and two overlapping-crop layouts did not produce useful player detections.
- The exact training version, Ultralytics 8.4.182, was tested in an isolated directory. It did
  not fix that failure; the working environment remains pinned to 8.3.240.
- Three Spain–Portugal frames demonstrate useful ball crop detections. At source 121 seconds,
  full-frame inference missed the ball, while the 960 crop pass detected it. Weak false crop
  candidates and overlapping boxes motivated confirmation and duplicate suppression.
- Same-input comparisons separate tracker behavior from detector changes. Fewer tracklets
  alone do not demonstrate fewer ID switches: wrong identity merges can also reduce the count.
- Appearance matching did not improve the preliminary tracklet count, so it is not enabled by default.
- One 1280-crop experiment stopped before finalization and is explicitly marked interrupted.
  It is not used as evidence of improvement. The cause was not confirmed.

No independent identity/ball ground truth was supplied, so IDF1, HOTA, detector precision/recall
and real ball-tracking accuracy remain unmeasured. More confident boxes or denser coverage alone
are not claimed as improved accuracy. The uploaded light-court match still needs better detector
generalization; pipeline tuning cannot supply players that the model does not recognize.

Technical references: [Ultralytics tracker controls](https://docs.ultralytics.com/modes/track/)
and [prediction controls](https://docs.ultralytics.com/modes/predict/). Local implementation uses
the inspected 8.3.240 tracker APIs, rather than assuming the latest documentation matches every API.
