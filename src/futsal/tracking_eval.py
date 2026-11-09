"""Run official TrackEval metrics against a reviewed MOT sequence export."""

from pathlib import Path
import importlib
import numpy as np

from .evaluation import iou
from .storage import write_json


class _NumpyCompatibility:
    """Legacy aliases local to pinned TrackEval modules; never patch global NumPy."""

    def __getattr__(self, name):
        aliases = {"float": float, "int": int, "bool": bool}
        return aliases[name] if name in aliases else getattr(np, name)


def evaluate_mot(ground_truth, prediction, output):
    try:
        import trackeval
    except ImportError as exc:
        raise RuntimeError(
            "Install optional TrackEval from the pinned source in docs/evaluation_protocol.md"
        ) from exc

    def read(path):
        value = np.loadtxt(path, delimiter=",", ndmin=2)
        if value.shape[1] < 6 or not np.isfinite(value[:, :6]).all():
            raise ValueError("MOT needs finite frame,id,x,y,w,h fields")
        return value

    gt, pred = read(ground_truth), read(prediction)
    if gt.shape[1] >= 7:
        gt = gt[gt[:, 6] > 0]
    gt_ids = {v: i for i, v in enumerate(sorted(set(gt[:, 1])))}
    pred_ids = {v: i for i, v in enumerate(sorted(set(pred[:, 1])))}
    data = {
        "num_timesteps": int(max(gt[:, 0].max(initial=0), pred[:, 0].max(initial=0))),
        "num_gt_ids": len(gt_ids),
        "num_tracker_ids": len(pred_ids),
        "num_gt_dets": len(gt),
        "num_tracker_dets": len(pred),
        "gt_ids": [],
        "tracker_ids": [],
        "similarity_scores": [],
    }
    for frame in range(1, data["num_timesteps"] + 1):
        a, b = gt[gt[:, 0] == frame], pred[pred[:, 0] == frame]
        axy = a[:, 2:6].copy()
        bxy = b[:, 2:6].copy()
        axy[:, 2:] += axy[:, :2]
        bxy[:, 2:] += bxy[:, :2]
        data["gt_ids"].append(np.array([gt_ids[v] for v in a[:, 1]], dtype=int))
        data["tracker_ids"].append(np.array([pred_ids[v] for v in b[:, 1]], dtype=int))
        data["similarity_scores"].append(np.array([iou(box, bxy) for box in axy]).reshape(len(a), len(b)))
    results = {}
    modules = [importlib.import_module("trackeval.metrics." + name) for name in ("hota", "identity", "clear")]
    original = [module.np for module in modules]
    try:
        for module in modules:
            module.np = _NumpyCompatibility()
        for metric in [trackeval.metrics.HOTA(), trackeval.metrics.Identity(), trackeval.metrics.CLEAR()]:
            result = metric.eval_sequence(data)
            results[type(metric).__name__] = {
                k: v.tolist() if isinstance(v, np.ndarray) else float(v) for k, v in result.items()
            }
    finally:
        for module, value in zip(modules, original):
            module.np = value
    write_json(
        output,
        {
            "status": "evaluated_against_user_supplied_reference",
            "metrics": results,
            "ground_truth": str(Path(ground_truth).resolve()),
            "prediction": str(Path(prediction).resolve()),
            "scope": "one reviewed continuous sequence; do not concatenate unrelated shot IDs",
            "compatibility": "Legacy np.float/int/bool aliases mapped only inside pinned metric modules",
        },
    )
    return results
