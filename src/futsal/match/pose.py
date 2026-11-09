"""COCO human-pose adapter: one-to-one box matching, confidence-gated ankles."""

from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_boxes(people, poses, min_iou=.3):
    a, b = np.asarray(people).reshape(-1, 4), np.asarray(poses).reshape(-1, 4)
    if not len(a) or not len(b):
        return []
    intersection = np.maximum(0, np.minimum(a[:, None, 2:], b[None, :, 2:])-
                              np.maximum(a[:, None, :2], b[None, :, :2])).prod(axis=2)
    area_a = np.maximum(0, a[:, 2:]-a[:, :2]).prod(axis=1)
    area_b = np.maximum(0, b[:, 2:]-b[:, :2]).prod(axis=1)
    iou = intersection/np.maximum(area_a[:, None]+area_b[None]-intersection, 1e-9)
    rows, cols = linear_sum_assignment(1-iou)
    return [(int(i), int(j)) for i, j in zip(rows, cols) if iou[i, j] >= min_iou]


class AnklePoseEstimator:
    def __init__(self, checkpoint, device='cpu', confidence=.4):
        from ultralytics import YOLO
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f'Human-pose checkpoint not found: {path}')
        self.model = YOLO(str(path))
        if self.model.task != 'pose':
            raise ValueError('Human-pose adapter requires a COCO 17-keypoint pose checkpoint')
        self.device, self.confidence = device, confidence

    def estimate(self, frame, tracks):
        if not len(tracks):
            return {}
        result = self.model.predict(frame, imgsz=960, conf=.25, device=self.device, verbose=False)[0]
        if result.keypoints is None or result.boxes is None:
            return {}
        xy = result.keypoints.xy.cpu().numpy()
        conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
        if xy.shape[1:] != (17, 2) or conf is None:
            raise ValueError('Pose output must use the COCO 17-keypoint schema with confidences')
        poses = result.boxes.xyxy.cpu().numpy()
        feet = {}
        h, w = frame.shape[:2]
        for person, pose in match_boxes(tracks.xyxy, poses):
            points = [xy[pose, k].tolist() for k in (15, 16)
                      if conf[pose, k] >= self.confidence and
                      0 < xy[pose, k, 0] < w and 0 < xy[pose, k, 1] < h]
            if points:
                feet[int(tracks.tracker_id[person])] = points
        return feet


class NoPoseEstimator:
    def estimate(self, frame, tracks):
        return {}
