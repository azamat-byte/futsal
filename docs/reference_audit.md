# Roboflow reference audit

Reference: https://github.com/roboflow/sports/tree/main/examples/soccer

Inspected commit: `42c80c06b6b65a7f89455b89fe31cdf4c38ba227`.

| Reference | Decision |
|---|---|
| Football player/ball checkpoints | Download as separate empirical baselines; record hashes and original class metadata |
| Supervision/ByteTrack example | Use the tested Ultralytics tracker adapters with explicit configuration and per-shot IDs |
| SigLIP + UMAP + KMeans | Defer heavyweight dependency on 4 GB hardware; compare a simpler color baseline first |
| Nearest-centroid goalkeeper team assignment | Replace with unknown/manual review; unsafe for futsal role changes |
| Outdoor 12000 × 7000 cm court / 32 landmarks | Replace completely with a meter-based futsal template and separate landmark schema |
| Bare findHomography helper | Replace with robust fitting, independent validation, explicit direction and validity |
| Buffered-centroid ball heuristic | Replace with short velocity-aware gating, ambiguity and missingness |
| Bash downloader | Replace with cross-platform downloads and PowerShell launcher |
| Demo radar | Reimplement with Plotly and explicit missing data |

The example is evidence of functional demonstration components. It is not evidence of futsal event accuracy or complete broadcast reconstruction.

OSNet source is separately vendored from `KaiyangZhou/deep-person-reid` commit `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50`, with its original MIT notice. Weights come from the author's Hugging Face repository and are trained for general person ReID, not same-kit futsal athletes.
