# Architecture

`src/futsal` is an importable package. The CLI and Streamlit dashboard call the same services. SQLite stores jobs; separate Python worker processes perform expensive inference. Raw source video and annotations remain unchanged.

## Processing

1. PyAV probes media and decodes PTS-based samples.
2. Histogram/appearance discontinuities propose shot boundaries; all segments start unknown.
3. An explicit YOLO adapter maps checkpoint class names to object types and roles.
4. ByteTrack or BoT-SORT associates person-family detections within each shot. Referee role is retained. IDs are namespaced by run and shot.
5. A separate short-horizon ball candidate filter rejects ambiguity and missing observations.
6. A match-specific Lab jersey-color model produces experimental team assignments; no team name is inferred.
7. Reviewed court correspondences anchor a homography. Short-term optical-flow propagation expires after a bounded duration or failed estimation.
8. Raw detections, observations, calibration state, sampled source timestamps and manifests are stored independently.
9. Human corrections overlay immutable observations. Derivation applies live-play, calibration, scale, and visibility gates.
10. Exports preserve missing values, timestamps, evidence, and correction history.

## Model decisions

YOLO11n is the lightweight starting profile on the 4 GB GPU. Football-specific player and ball checkpoints from the pinned Roboflow example are available as challengers. Generic YOLO is not a goalkeeper classifier. Football keypoint weights are deliberately not relabeled as futsal geometry.

RF-DETR is an optional adapter rather than a mandatory heavyweight dependency. It is not benchmarked on this installation. SigLIP, SAM2, hosted NAS and multiple simultaneous detectors are not required by the default path. OSNet supplies actual pretrained appearance features in an optional review workflow; it does not automatically merge players.

## State and reproducibility

Each processing attempt has a unique directory. Restart creates a new attempt from the configured interval beginning; it never splices an empty tracker into an old sequence. The database uses transactional claim semantics and WAL. Artifact publication uses temporary files and atomic replacement. Environment versions and model/source checksums are recorded per run.

The dashboard uses a time inspector, not advertised frame-accurate playback synchronization. Long-term multiuser serving and remote access are outside the current local deployment profile.
