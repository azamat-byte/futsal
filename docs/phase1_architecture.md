# Five-phase futsal architecture

This roadmap and the new `futsal.phase1` package are independent of the existing
research pipeline. Phases 1 and 2 are now implemented; see [Phase 2](phase2.md).
Phases 3–5 await confirmation. The supplied
notebook was reviewed as reference material, not executed as instructions.

## Model inventory checked through Roboflow

- Detection: `s-workspace-bimfr/new_futsal-2-rfdetr-medium-t1` is a completed
  RF-DETR Medium object detector. Its project classes are `ball`, `goalkeeper`,
  and `player`. There is no referee class.
- Keypoints: `s-workspace-bimfr/keypoints-ovj9c-11-rfdetr-keypoint-two-stage-t1`
  is a completed keypoint model whose project class is `pitch`. The notebook
  also uses this model for court landmarks. It is useful for Phase 2, but it
  does not supply player ankle/foot keypoints for Phase 4.

## Modules and data flow

| Phase | Components | Output / acceptance criteria |
| --- | --- | --- |
| 1: MOT and teams | Detector adapter, person-only ByteTrack, torso HSV features, K-Means, temporal team voting, annotations | Tracked video, observation CSV, reviewed color clusters. Test identities through short occlusions and review team assignments. |
| 2: Geometry and radar | Named court landmarks, calibration validation, camera motion/cut handling, projection, pitch renderer | Player bottom-center coordinates in meters, calibration quality, synchronized radar. Validate on held-out floor landmarks. |
| 3: Spatial analytics | Timestamp-aware trajectory smoothing, speed/distance, pitch-clipped Voronoi cells, team convex hulls | Per-track metrics and team geometry. Do not connect gaps/cuts or interpret missing off-camera players as absent players. |
| 4: Possession and events | Separate human-pose adapter, ball association/filter, foot proximity, possession state machine, pass/shot candidates | Events with time, actor, confidence, and evidence. Validate against annotated events. Pitch keypoints cannot substitute for human pose. |
| 5: Delivery | Overlay compositor, audio-preserving video export, statistics aggregation, optional Streamlit UI | Tactical video and dashboard driven by the same stored observations/events. Validate synchronization and review examples. |

Keep the pipeline independent of Streamlit. Each module consumes typed observations
and returns data; rendering never modifies measurement boxes. Phase 1 stores
`frame_index`, `timestamp_s`, `segment_id`, `track_id`, detector `role`, separate
`team`, confidence, source-pixel `xyxy`, and `foot_x_px` / `foot_y_px`. A track ID
identifies a tracklet, not a verified roster identity. The canonical identity is
`(run, segment_id, track_id)`; re-identification and roster matching are later work.

Phase 1 assumes constant-frame-rate footage and uses `frame_index / fps`. Normalize
variable-frame-rate footage before use, or introduce a presentation-timestamp video
reader before physical metrics. Use the measured court dimensions; 40 x 20 m is the
initial configuration, not a substitute for validating scale. A single static
homography is insufficient for a panning/zooming broadcast. Cuts invalidate the
mapping. An airborne ball cannot be reliably mapped to the floor using its center.

## Folder layout

```text
futsal/
  configs/phase1.json
  requirements-phase1.txt
  src/futsal/
    phase1/                 # implemented now
      config.py             # validated, credential-free configuration
      detection.py          # Roboflow HTTP adapter and response normalization
      tracking.py           # Supervision ByteTrack, cut/reset identity scope
      teams.py              # dominant HSV, K-Means, per-track voting
      annotations.py        # Supervision boxes and labels
      video.py              # bounded video I/O with cleanup
      pipeline.py           # calibration pass + streaming processing pass
      cli.py
      __main__.py
    phase2/                 # implemented: geometry, calibration, picker, radar, pipeline, CLI
    spatial/                # proposed Phase 3 module boundary
    events/                 # proposed Phase 4 module boundary
    presentation/           # proposed Phase 5 module boundary
  tests/phase1/
  docs/phase1.md
  docs/phase1_architecture.md
  runs/<run-name>/           # created by CLI
    annotated.mp4
    tracks.csv
    teams.json
    calibration.jpg
    run.json
```

The future folders above are a proposed organization. Existing modules with similar
responsibilities remain available for later integration; they are not automatically
promoted into this implementation.

## Team and role semantics

Two player-shirt clusters are the default. Goalkeepers remain in the distinct
`goalkeeper` role and display group. The optional three-cluster mode exposes a
reviewable `other` color group. Cluster indices are ordered deterministically by
their color centers, then mapped to Team A/B/Other through configuration. Clustering
does not establish referee identity or the goalkeeper's club. Multiple goalkeeper
kits also need not form one color cluster. Review `calibration.jpg` and edit the
cluster label mapping. Low-quality or ambiguous observations remain `unknown`.

## References

- [Supervision ByteTrack API](https://supervision.roboflow.com/0.25.0/trackers/)
- [Roboflow Serverless API](https://docs.roboflow.com/deployment/roboflow-cloud/serverless-api)
- [Roboflow inference authentication](https://docs.roboflow.com/reference/inference/inference-sdk)

Supervision is deliberately pinned to 0.25.1, matching the existing project's
notebook extra and the tested built-in ByteTrack API. This is not a claim that it
is the latest release. The small HTTP adapter avoids requiring a new inference SDK
NumPy stack in the existing environment.
