# Evaluation protocol

## Release gates

Separate software execution, model accuracy, physical accuracy and operational stability. A test pass is not a model-accuracy result.

Freeze reviewed validation/test groups before model selection. Keep all frames from a match in one split. For the current dataset, all 240 files are in train and no source-match manifest is supplied. No independent supervised accuracy or acceptance claim is therefore available yet.

## Current detection diagnostics

`futsal evaluate ... --exploratory` explicitly permits unverified label comparisons and records that warning in the report. Such results are not model promotion evidence. The implemented evaluator uses 101-point interpolated AP and one-to-one greedy IoU matching at thresholds 0.50:0.05:0.95. Person evaluation is role-agnostic and must not be called goalkeeper AP. Precision/recall at the exported score floor are reported separately from AP.

Small-ball center error, role confusion, PR sweeps, occlusion strata and calibration reference errors require their own reviewed benchmark. Do not substitute diagnostic aggregate AP for those missing measurements.

## Tracking

`futsal evaluate-tracking gt.txt prediction.txt output.json` calls TrackEval HOTA, Identity and CLEAR. Supply one reviewed continuous sequence, with frame/id/x/y/width/height and optional active flag in column 7. Ignore-region handling must be resolved before this generic adapter; do not feed unprocessed official MOT challenge files with distractor semantics and claim challenge-equivalent scores.

TrackEval is installed from official commit `12c8791b303e0a0b50f753af204249e622d0281a` with the already tested NumPy/SciPy environment. A perfect synthetic sequence checks the adapter; this is not real tracking accuracy evidence. To reproduce: `python -m pip install --no-deps git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a`. Use a separate evaluation environment if future dependency requirements conflict. The adapter refuses to invent metrics when TrackEval is absent.

No reviewed tracking sequences have been supplied. Cross-cut identity quality requires reviewed same/different pairs, accepted-link precision, coverage and uncertainty. OSNet feature extraction alone does not satisfy that gate.

## Geometry and events

Fit court-to-image with a pixel RANSAC threshold, then invert. Check landmarks not used for fitting. Four points alone provide no independent error estimate. A calibration without review and an independent check remains invalid. Actual court dimensions must be confirmed before physical measurements.

Position and speed references must be independently sourced. Ground truth reconstructed using the same uncertain homography cannot independently validate it. Grounded-ball verification and reviewed possession/pass events are needed before event quality can be measured.

Initial engineering targets follow the master prompt: player precision/recall 0.90, ball precision 0.90 and recall 0.75 on resolvable cases, IDF1 0.80/HOTA 0.60, accepted identity-link precision 0.98, and pass precision 0.90 on reviewed eligible events. They are targets, not achieved results. Report sample sizes, per-match variation, coverage and confidence intervals. Do not tune on the final test match.
