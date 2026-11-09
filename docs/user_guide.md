# User workflow

## 1. Inspect a real run

Launch the app. Six short verified execution runs are available: one ten-second initial clip and eight seconds from each supplied video. Select Match viewer and a run, then use the source time slider. The court view stays empty when geometry is unavailable. That empty state is intentional.

Use Observation window to select a bounded part of a long run. Playback of the annotated proxy is independent of the source-time inspector; frame-to-source mapping is preserved in the exported frame table.

## 2. Correct the training data

Open Dataset studio and choose `futsal-v2`. Use Show only flagged images to inspect the removed-class image, duplicate annotations, and small person components. Then review all images before training. The count of reviewed images starts at zero; importing labels did not certify them.

Draw boxes on the image, or edit/delete existing boxes in the table. Select the proper class for new boxes. Enter the actual source match/group and choose its split. Check completeness only after inspecting every in-scope object, including the ball. Save the annotation explicitly.

`configs/source_groups.yaml` proposes groups from filenames and visible scoreboards. The BRA-KAZ source may overlap `videoplayback (1).mp4`, so do not casually use that video as an independent test match. Confirm provenance. ARG-POR halves remain the same match even when filename prefixes differ.

Export a reviewed YOLO release only after independent train/validation groups exist. Structural cleanup and polygon conversion are already available in the three-class unreviewed working export.

## 3. Improve detection

Compare pretrained candidates on reviewed validation labels. The saved initial 60-frame comparisons are diagnostic only. All three tested checkpoints performed poorly on tiny balls in the stretched 512px export. Add high-resolution frames and carefully label balls rather than assuming a larger network alone fixes the problem.

Generate new annotation proposals from training sources with the CLI. Audit that proposal folder into a new dataset version, then review it in Dataset studio. Never promote empty detector predictions to verified negative images automatically.

Train with `futsal train` only after creating a reviewed release. Keep the original model for rollback. Training is not automatically launched from unreviewed data.

## 4. Calibrate a court

Choose a frame with sufficiently distributed recognizable floor markings. Click at least four fit landmarks plus held-out validation landmarks. Enter matching world coordinates, designate validation rows, confirm correspondence review and provide actual court dimensions. Do not use the crossbar or elevated goal points as floor landmarks.

Save the calibration JSON. Reprocess that interval using its path. A manual anchor only supports a short propagation interval; add anchors across changing views. A separately trained futsal keypoint model can later provide repeated anchors through configuration. Outdoor soccer landmarks are not relabeled as futsal landmarks.

## 5. Review identities and broadcast context

Label shots as live wide, close-up, replay, non-play, transition or unknown. Only reviewed live-wide segments enter physical analytics. Boundaries are proposals and need checking.

In Identities & events, correct team and role assignments. You can link reviewed tracklets to an anonymous player or roster label. Simultaneous tracklets cannot be merged. OSNet candidates are appearance suggestions, not confirmed identities.

## 6. Read analytics and export

Heatmaps and movement estimates require valid calibration, verified scale, in-scope player roles and eligible live footage. They describe observed intervals. No full-match distance is extrapolated.

Ball positions stay in image coordinates unless floor contact is explicitly supported. Reviewed grounded observations can support conservative possession/pass hypotheses. Accepted event reviews are stored separately and included in exports.

Export center builds a ZIP of CSV, Parquet, MOT observations, timing and provenance. Annotated videos remain available in their run folders. Dataset export and analysis export are separate operations.
