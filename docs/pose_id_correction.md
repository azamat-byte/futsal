# Court pose ID correction

Run from the repository with its Python environment:

```powershell
$env:PYTHONPATH = 'src'
python -m futsal.pose_id_correction INPUT.zip outputs/pose-review
python -m futsal.pose_id_correction INPUT.zip outputs/pose-corrected --evidence evidence.json --apply
```

Default mode is dry-run. Each run requires a new output directory, backs up its input,
and creates a dataset copy, SHA-256 backup manifest, report.json, review.json and
before/after visualization for each available image. Originals are never written.
The dry-run dataset is unchanged; after panels show proposed changes only.

Supports standard YOLO Pose (data.yaml kpt_shape, 2 or 3 dimensions) and COCO
Keypoints (JSON annotations with flattened x,y,v arrays). All triplet values move
together when IDs swap. YOLO bbox/class fields and numeric strings are retained;
COCO unrelated annotation fields and categories are retained. COCO JSON formatting
can change when a correction is applied. Separate per-keypoint arrays in custom
COCO extensions are not supported; do not apply to such extensions.

Detection requires independently reviewed image landmarks in pixel coordinates.
Symmetric court geometry or an unconstrained homography cannot establish semantic
left/right orientation. This tool therefore abstains without evidence instead of
guessing. It does not include a trained visual landmark detector.

Evidence JSON is keyed by dataset-relative image path, then zero-based pose instance
index (YOLO row order or COCO annotation order). Example structure:

```json
{
  "train/images/example.jpg": {
    "0": {
      "reviewed": true,
      "orientation": "Describe the confirmed camera/court orientation",
      "tolerance_px": 8,
      "landmarks": [
        {"id": 0, "x": 100, "y": 200, "visible": true},
        {"id": 1, "x": 120, "y": 250, "visible": true},
        {"id": 2, "x": 130, "y": 280, "visible": true}
      ]
    }
  }
}
```

Supply at least three independently identifiable landmarks from distinct symmetry
pairs, using the intended correct IDs. All must support the same hypothesis within
tolerance and reject the alternate hypothesis by at least 3 times tolerance. Only
visible annotations are used. Every pose in an image must agree; missing evidence,
mixed decisions, or incompatible schemas send the whole image to review unchanged.

Roboflow inspection confirmed 26 active native skeleton IDs:
0–16, 20, 21, 23, 25, 26, 27, 28, 29, 31. The YOLO export renumbers these
as contiguous positions 0–25. The script translates the requested native symmetry
pairs to these positions and verifies data.yaml flip_idx agrees before proceeding.
Evidence IDs and visualization labels use exported positions. Without reviewed
landmark evidence, images still go to manual review unchanged.
This automatic translation applies only to the verified 26-point YOLO layout;
COCO arrays currently use the native-index path and require matching array indices.
