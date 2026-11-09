# Additional supervised inputs

The application does not manufacture missing labels. These are the JSON interfaces for future reviewed inputs.

## Court landmarks

`futsal keypoint-export annotations.json output-directory` expects `landmark_names` and `frames`. Every frame supplies an existing local `image`, `group`, `split`, `reviewed: true`, and a `points` array in exactly the same semantic order. Points are `[image_x, image_y, visibility]`; visibility is 0 missing, 1 occluded, or 2 visible. Do not guess coordinates for missing landmarks. Every point count must equal the schema length.

Example schema structure (illustrative coordinates, **not annotations for supplied footage**):

```json
{
  "landmark_names": ["corner_00", "corner_L0", "corner_LW", "corner_0W"],
  "frames": [
    {"image": "C:/approved/example.jpg", "group": "match-a", "split": "train", "reviewed": true,
     "points": [[0,0,0],[0,0,0],[0,0,0],[0,0,0]]}
  ]
}
```

The example is intentionally missing all landmarks and is not a usable training set. Actual training needs reviewed visible correspondences and separate source matches. Start from an officially sourced human-pose checkpoint only for backbone transfer; training must replace the keypoint head for court semantics.

`futsal train-court dataset-directory local-pose-checkpoint.pt output-directory --device 0` uses that export. For inference, the court schema additionally needs `names`, matching `court_points` in court coordinates, `length`, `width`, `scale_verified`, and an independently justified `acceptance_status`. Configure `court_model` and `court_schema`. The runtime checks held-out landmark consistency; a model without accepted validation remains proposal-only.

## Formations

`futsal train-formations annotations.json output-directory` expects `windows`, each with `group`, `split`, `reviewed`, `label`, and `positions`. Positions have shape T × 4 × 2, containing four visible outfield players in metric coordinates for each sampled frame. Normalize attacking direction using reviewed period metadata before training. Use consistent labels and enough windows from independent matches for each class.

The classifier learns geometric window features and includes an unknown output below its score threshold. Scores are uncalibrated votes. No default trained formation model is included.

## Tracking evaluation

Use reviewed continuous sequences with stable IDs in a MOT table. Source and predictions must use the same sampled-frame mapping. CVAT can be used to create reviewed tracking annotations, then converted to this interface. Do not compare PTS ticks to ordinal frame indices.

## Re-identification evaluation

Collect reviewed tracklet pairs and same/different identities across cuts. Include difficult same-team negatives and substitutions. The OSNet retrieval results supply candidate scores; no automatic merging is enabled by default. Real match identity evaluation remains separate from within-shot HOTA/IDF1.
