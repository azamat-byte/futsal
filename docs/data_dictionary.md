# Data dictionary

## Run files

- `manifest.json`: source path/checksum, model/checksum, class mappings, config, environment, effective cadence, state, elapsed processing time and limitations.
- `frames.jsonl`: one line per sampled observation; original presentation timestamp ticks, source seconds, preceding elapsed time, shot, calibration validity and optional image-to-court matrix. `proxy_index` maps constant-cadence output frames to source time.
- `detections.jsonl`: all detector observations, including low-score/untracked candidates.
- `tracks/part-*.parquet`: bounded batches of accepted person tracks and ball observations.
- `shots.json`: automatic boundary proposals and default unknown segment classes.
- `review.json` plus `review_history.jsonl`: materialized corrections and audit trail.
- `movement.parquet`, `trajectory.parquet`, `tactics.parquet`, `possession.parquet`: derived estimates, never fabricated values.
- `analytics_summary.json`: coverage, gates, and explicit unavailability reasons.

## Observation columns

`frame_index` is the **source PTS tick**, not an ordinal frame count. `time_s` is presentation time relative to video-stream start. `tracklet_id` is namespaced by run/shot and is not a roster identity. `class_name` is `person` or `ball`; `role` is separate. Team assignments are experimental cluster labels unless reviewed.

`court_x/y` are null unless a supported ground-contact projection exists. `scale_verified` must be true before meter-based physical analytics. `calibration_error_px` describes anchor check error, not a calibrated world-position uncertainty. Propagated transforms retain that anchor diagnostic while losing validity at expiry.

Missing values remain null in Parquet and empty in CSV. Zero is a real numerical value, never a substitute for unknown. Ball image positions remain available even when grounded status is unknown; no 3D reconstruction is claimed.

## Statistics

Observed distance integrates only valid contiguous filtered positions. The three-sample median filter is offline and noncausal. Movement duration is the sum of accepted adjacent time differences. Gaps/cuts do not add distance. Uncertain identity links do not join trajectories.

Heatmaps weight positions by bounded elapsed time. Team shape is a visible-subset measurement unless four outfield players are identified. Possession/pass outputs are reviewable hypotheses; unknown and contested intervals break transfer continuity. No pass-completion denominator is inferred from completed-pass candidates alone.
