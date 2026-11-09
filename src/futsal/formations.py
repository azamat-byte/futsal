"""Supervised tactical-window classification with source-group validation and abstention."""

from pathlib import Path
import numpy as np

from .storage import read_json, write_json


def formation_features(positions):
    points = np.asarray(positions, float)
    if points.ndim != 3 or points.shape[1:] != (4, 2) or not np.isfinite(points).all():
        raise ValueError("Each frame must have four observed outfield-player positions; shape T,4,2")
    features = []
    for frame in points:
        frame = frame - frame.mean(0)
        # Attack direction must already be normalized by reviewed match metadata.
        order = np.lexsort((frame[:, 1], frame[:, 0]))
        frame = frame[order]
        scale = max(np.sqrt((frame * frame).sum() / 4), 1)
        ordered = (frame / scale).ravel()
        distances = sorted(np.linalg.norm(frame[i] - frame[j]) for i in range(4) for j in range(i + 1, 4))
        features.append(np.r_[ordered, np.ptp(frame, axis=0), distances])
    return np.r_[np.median(features, axis=0), np.std(features, axis=0)]


def train_formations(annotations, output):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    import joblib

    records = read_json(annotations)["windows"]
    if not records or any(not r.get("reviewed") or not r.get("group") for r in records):
        raise ValueError("Reviewed formation windows and source groups are required")
    train = [r for r in records if r["split"] == "train"]
    val = [r for r in records if r["split"] == "val"]
    if not train or not val or {r["group"] for r in train} & {r["group"] for r in val}:
        raise ValueError("Independent train/validation matches are required")
    model = RandomForestClassifier(n_estimators=150, max_depth=5, class_weight="balanced", random_state=42)
    x = np.stack([formation_features(r["positions"]) for r in train])
    y = [r["label"] for r in train]
    if len(set(y)) < 2:
        raise ValueError("At least two formation labels are required")
    model.fit(x, y)
    vx = np.stack([formation_features(r["positions"]) for r in val])
    vy = [r["label"] for r in val]
    prediction = model.predict(vx)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, output / "formation.joblib")
    write_json(
        output / "evaluation.json",
        {
            "report": classification_report(vy, prediction, output_dict=True, zero_division=0),
            "training_windows": len(train),
            "validation_windows": len(val),
            "status": "validation_only_not_final_test",
            "warning": "Probabilities are uncalibrated model scores",
        },
    )
    return output


def predict_formation(model_path, positions, minimum_score=0.8):
    import joblib

    model = joblib.load(model_path)
    scores = model.predict_proba(formation_features(positions)[None])[0]
    best = int(np.argmax(scores))
    return {
        "formation": str(model.classes_[best]) if scores[best] >= minimum_score else "unknown",
        "score": float(scores[best]),
        "score_type": "uncalibrated_random_forest_vote",
        "review_required": True,
    }
