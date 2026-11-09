import pytest

from futsal.tracking_eval import evaluate_mot


def test_official_metrics_perfect_synthetic_tracks(tmp_path):
    pytest.importorskip("trackeval")
    gt = tmp_path / "gt.txt"
    pred = tmp_path / "pred.txt"
    content = "\n".join(f"{i},1,10,10,20,40,1,-1,-1,-1" for i in range(1, 6))
    gt.write_text(content)
    pred.write_text(content)
    result = evaluate_mot(gt, pred, tmp_path / "metrics.json")
    assert result["Identity"]["IDF1"] == pytest.approx(1)
    assert all(v == pytest.approx(1) for v in result["HOTA"]["HOTA"])
