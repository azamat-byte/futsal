import numpy as np
import pytest

from futsal.datasets import yolo_boxes, audit_dataset, review_record, export_yolo, remove_class_version
from futsal.video import save_image


def dataset(tmp_path):
    source = tmp_path / "input"
    save_image(source / "train/images/a.jpg", np.zeros((100, 200, 3), np.uint8))
    labels = source / "train/labels"
    labels.mkdir(parents=True)
    (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.4")
    (source / "data.yaml").write_text("names: [player, remove_me]")
    return source


def test_polygon_conversion(tmp_path):
    path = tmp_path / "label.txt"
    path.write_text("0 0.1 0.2 0.5 0.2 0.5 0.8 0.1 0.8")
    boxes, issues = yolo_boxes(path, 100, 100, 1)
    assert not issues
    assert boxes[0] == pytest.approx(dict(class_id=0, x1=10, y1=20, x2=50, y2=80))


def test_invalid_coordinates(tmp_path):
    path = tmp_path / "label.txt"
    path.write_text("0 0.5 0.5 -0.2 0.4")
    assert yolo_boxes(path, 100, 100, 1)[1]


def test_unreviewed_labels_not_exportable(tmp_path):
    source = dataset(tmp_path)
    audit_dataset(source, tmp_path / "audit")
    with pytest.raises(ValueError, match="No fully reviewed"):
        export_yolo(tmp_path / "audit", tmp_path / "export")


def test_review_export_roundtrip(tmp_path):
    source = dataset(tmp_path)
    manifest = audit_dataset(source, tmp_path / "audit")
    record = manifest["records"][0]
    review_record(tmp_path / "audit", record["id"], record["boxes"], "match1", "train", True)
    path = export_yolo(tmp_path / "audit", tmp_path / "export")
    assert (path / "data.yaml").exists()
    assert len(list((path / "labels/train").glob("*.txt"))) == 1


def test_remove_class_preserves_raw(tmp_path):
    source = dataset(tmp_path)
    audit_dataset(source, tmp_path / "audit")
    result = remove_class_version(tmp_path / "audit", tmp_path / "clean", "remove_me")
    assert result["names"] == ["player"]
    assert "remove_me" in (source / "data.yaml").read_text()
