# Capability ledger

This ledger distinguishes implemented software from verified sporting accuracy.

20 October update: the primary detector is now the supplied `best.pt`. Colab authoring,
custom-model clips, team comparison and synchronized manual-calibration radar exports are
documented in [the current milestone report](milestone_report.md). Historical entries below
describe earlier work and do not establish accuracy for this checkpoint.

| Capability | Implementation | Validation | Conditions / remaining evidence |
|---|---|---|---|
| Video probing and PTS decoding | Implemented | Real videos and synthetic round-trip tested | Source PTS ticks are not ordinal frames |
| Dataset audit and polygon conversion | Implemented | Supplied 240-image export processed | Full visual annotation review still required |
| Remove player- ball class | Implemented | Three-class snapshot exported | One affected image needs relabeling |
| YOLO player/ball inference | Implemented | Real footage and exploratory dataset diagnostics | No independent reviewed accuracy benchmark |
| ByteTrack / BoT-SORT | Implemented | Real within-shot execution | HOTA/IDF1 require identity ground truth |
| TrackEval HOTA / IDF1 | Official pinned implementation integrated | Perfect synthetic sequence tested | No real identity ground truth supplied |
| Team classification | Implemented, experimental color model | Real execution | Team mappings and precision need review |
| Cross-cut ReID | Pretrained OSNet retrieval plus manual linking | Encoder executed | Automatic merging disabled; accuracy unmeasured |
| Shot/live/replay management | Boundary proposals plus manual labels | Software tests | Automatic replay classifier not trained |
| Manual court calibration | Implemented | Geometric tests | User's court dimensions and reviewed points required |
| Dynamic calibration | Short-term optical-flow propagation | Geometry tests | Real metric error not validated; bounded expiry |
| Automatic futsal court landmarks | Annotation export and training workflow | Not trained | No futsal keypoint labels provided |
| Heatmaps / trajectories | Implemented with gates | Synthetic tests | Requires eligible calibrated observations |
| Speed / distance | Implemented observed-interval estimates | Synthetic tests | Independent physical reference unavailable |
| Tactical geometry | Implemented visible-subset metrics | Software implementation | Full team visibility and roles needed |
| Formation classifier | Supervised training/inference workflow and abstention implemented | Not trained | Reviewed four-player windows and attack direction required |
| Possession / passes | Conservative hypotheses and review | State-machine tests | Ball ground contact/events need review; no automatic quality claim |
| Interactive dashboard | Implemented | AppTest and browser inspection | Video playback separate from time inspector |
| CSV / Parquet / MOT / ZIP / H264 exports | Implemented | Round-trip tests and real sample | Analysis proxy, not lossless source reproduction |
| Fine-tuning | Reviewed-data training command | Not trained | Current export has no reviewed independent val split |
| RF-DETR | Optional adapter | Not installed/evaluated | Deferred for current hardware |
| Five-video execution | Real sample clips passed | Eight seconds per video, plus first ten-second clip | Full match accuracy and stability not established |

The complete-file stability batch is running sequentially in `runs/batch-e6d23e9184.json` using the explicitly low-cadence profile in `configs/full_video_smoke.yaml`. Its job state is visible in Overview. Completion of a file establishes execution, not sporting accuracy. The user's supplied court-size ranges do not establish exact venue dimensions, so metric scale remains unverified.

Unsupported or unevaluated features are not considered complete. No random outputs or fabricated benchmark values are used.
