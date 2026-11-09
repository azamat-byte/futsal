import pytest
from futsal.evaluation import detection_metrics, iou


def test_duplicate_detection_is_false_positive():
    truth = {"f": [{"class": "person", "box": [0, 0, 10, 10]}]}
    predictions = {
        "f": [
            {"class": "person", "box": [0, 0, 10, 10], "score": 0.9},
            {"class": "person", "box": [0, 0, 10, 10], "score": 0.8},
        ]
    }
    result = detection_metrics(truth, predictions, "person")
    assert result["tp"] == 1 and result["fp"] == 1
    assert result["ap"] == pytest.approx(1)


def test_missing_class_ap_is_unknown():
    assert detection_metrics({}, {}, "ball")["ap"] is None


def test_iou():
    assert iou([0, 0, 10, 10], [[0, 0, 10, 10], [20, 20, 30, 30]]).tolist() == [1, 0]
