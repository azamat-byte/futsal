# Phase 2: pitch homography and radar

Phase 2 is implemented in `src/futsal/phase2/`. It reads a completed Phase 1 run,
projects the bounding-box bottom centers onto a 40 x 20 m floor plane, and renders
both a tactical overlay and a separate radar video. It does not rerun player inference
and does not require an API key. The original Phase 1 run is preserved.

Your actual input is `1111.mp4` (1920 x 1080, 30 FPS). The completed run
`runs/1111-phase1-test-02` contains frames 0–499 and 3,718 observations.

## First test: calibrate a frame

In PowerShell, from this project directory:

```powershell
.\.venv\Scripts\python.exe -m futsal.phase2 calibrate --video "1111.mp4" --frame 150 --output "configs/1111-calibration.json"
```

This opens a window with the source frame and a list of **named floor landmarks**.
Frame 150 (five seconds) has a clearer center-circle view than the opening frame,
where a player covers a circle intersection.
Choose a landmark on the right, then click the corresponding visible point in the
video. The first four clicks fit the homography; further clicks are independent
checks. At least six distinct points are required. The list's default first six
landmarks are intended for a view of the center of the court:

1. Halfway line meeting the far touchline: `(20, 0)`.
2. Near substitution-zone marking 5 m left of halfway: `(15, 20)`.
3. Near substitution-zone marking 5 m right of halfway: `(25, 20)`.
4. Far intersection of the center circle and halfway line: `(20, 7)`.
5. Halfway line meeting the near touchline: `(20, 20)` — check point.
6. Near intersection of the center circle and halfway line: `(20, 13)` — check point.

For substitution marks, click where the mark meets the touchline, not its outer tip.
Only select these points if the markings are visible and their meaning is correct
for the source. If one is occluded, choose another visible named landmark from the
list, or another frame. Do not estimate hidden corners, click a goal crossbar, or use
the apparent left/right extremes of the projected circle as named pitch points.
For a nonstandard court, supply a JSON calibration with its measured coordinates
instead of using this 40 x 20 m picker preset.

- **Right-click / U:** undo the last point.
- **Enter:** fit the transform and check the independent points. Default maximum
  validation error is 8 source pixels. If it fails, correct the points.
- Inspect the yellow court/circle overlay. It should follow the actual floor markings.
- **Enter again:** save the visually reviewed anchor.
- **Q / Esc:** close. Saved anchors remain in the JSON file.
- **N / P:** move forward/backward two seconds for another anchor. Unsaved clicks
  are cleared. Previously saved anchors remain available.

## Render a two-second preview

After saving an anchor at frame 150:

```powershell
.\.venv\Scripts\python.exe -m futsal.phase2 render --phase1-run "runs/1111-phase1-test-02" --calibration "configs/1111-calibration.json" --output "runs/1111-phase2-preview" --start-frame 150 --max-frames 60
Invoke-Item ".\runs\1111-phase2-preview"
```

No new dependencies are needed beyond Phase 1. Choose a new output directory on
each run. If you selected a different calibration frame, change `--start-frame`
to match. It must lie within the completed Phase 1 range (0–499 here).

## Handle the moving broadcast camera

A homography fits one camera view. Your clip pans and zooms. The camera module
updates the transform using optical flow on floor texture, excluding tracked people.
It checks forward/backward flow consistency, inlier support, spatial spread, and
plausible frame-to-frame camera movement. With motion `M` mapping previous pixels
to current pixels, the update is `H_current = H_previous @ inverse(M)`.

Propagation is limited to **two seconds after an anchor by default**. Low-texture
floors, players, blur, camera cuts and dissolves can invalidate calibration earlier.
In those cases the radar explicitly displays a calibration gap, and positions are
blank in the CSV. The code never freezes stale positions, clips off-court people to
the border, or interpolates across cuts.

For the full 500-frame clip, save additional anchors roughly every 60 frames wherever
enough floor landmarks are visible (N advances by 60 frames for this source). You can
also reopen it with `--frame NUMBER` to add a targeted anchor after a failure. Frames
before the first anchor remain uncalibrated; do not guess occluded opening landmarks.
Then render without `--max-frames`:

```powershell
.\.venv\Scripts\python.exe -m futsal.phase2 render --phase1-run "runs/1111-phase1-test-02" --calibration "configs/1111-calibration.json" --output "runs/1111-phase2-full"
```

`--max-anchor-age` changes the propagation limit in seconds. A longer limit can
increase coverage but also drift; it is not a replacement for reviewing geometry.
For match-length processing, automatic re-anchoring from your pitch model is the
next calibration refinement. This delivery uses clicked correspondences: the local
26-point schema says it was inferred from dataset version 2 and is unvalidated;
it has not been assumed to match the supplied version-11 RF-DETR keypoint model.
Its output-ID-to-landmark mapping must be verified before automatic use.

## Outputs and units

| Output | Contents |
| --- | --- |
| `tactical.mp4` | Source video with Phase 1 boxes/IDs and a synchronized radar |
| `radar.mp4` | Separate top-down pitch video, same frame count and FPS |
| `positions.csv` | Original observations plus pitch x/y, projection validity, calibration status and units |
| `calibration_frames.jsonl` | One record per frame, including empty-detection frames; status and image-to-pitch H |
| `calibration.json` | Exact calibration snapshot used by this run |
| `run.json` | Source SHA-256, settings, coverage, observation counts and completed/failed status |

Coordinates use `x=0..40` along the length and `y=0..20` from the far to near
touchline in the reference main-camera view. The origin and orientation must remain
consistent across all anchors, even when a camera view changes. Source frame indices
remain absolute; video outputs start at output frame zero and match Phase 1's range.

`coordinate_units=assumed_m` means the court dimensions were assumed. Only set
`scale_verified=true` after confirming the physical dimensions. Geometric validation
does not independently verify court scale or roster identity. Bottom-center anchors
are approximate floor contact positions, not anatomical feet. Goalkeepers keep
their separate purple color; their club affiliation remains unknown.

## Module responsibilities

- `geometry.py`: OpenCV `findHomography`, inversion, `perspectiveTransform`, robust
  fitting and held-out landmark validation. Fits court-to-image first so the
  RANSAC residual threshold is in pixels.
- `calibration.py`: source-bound calibration, cut/expiry handling and short camera-motion propagation.
- `picker.py`: native interactive calibration and visual overlay review.
- `radar.py`: metric pitch, player dots/IDs and video composition.
- `pipeline.py`: streaming CSV replay, synchronized video export and per-frame audit.
- `cli.py`: `calibrate` and `render` commands.

The picker is a calibration tool, not the Phase 5 dashboard. Speed, distance,
Voronoi/convex-hull metrics, possession, ball events and audio muxing are not added
in this phase. These MP4V outputs are silent, as in Phase 1.

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest tests/phase1 tests/phase2 -q
.\.venv\Scripts\python.exe -m ruff check src/futsal/phase2 tests/phase2
```

Tests include known projective transforms, RANSAC outlier rejection, independent
validation failure, degenerate/horizon inputs, translated-camera composition,
cut/gap/age rejection, floor-texture failure, source binding, point-picker coordinate
conversion, and synthetic video-to-radar export with absolute frame offsets.
These validate the software, not the accuracy of a real-match calibration.

API reference: [OpenCV homography and calibration](https://docs.opencv.org/4.8.0/d9/d0c/group__calib3d.html).
