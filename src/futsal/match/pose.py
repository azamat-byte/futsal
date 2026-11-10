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
    def __init__(self, checkpoint, device='cpu', confidence=.3):
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
        feet = {}
        h, w = frame.shape[:2]
        matched_people = set()
        if result.keypoints is not None and result.boxes is not None and len(result.boxes):
            xy = result.keypoints.xy.cpu().numpy()
            conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            if xy.shape[1:] != (17, 2) or conf is None:
                raise ValueError('Pose output must use the COCO 17-keypoint schema with confidences')
            poses = result.boxes.xyxy.cpu().numpy()
            for person, pose in match_boxes(tracks.xyxy, poses):
                points = [xy[pose, k].tolist() for k in (15, 16)
                          if conf[pose, k] >= self.confidence and
                          0 < xy[pose, k, 0] < w and 0 < xy[pose, k, 1] < h]
                if points:
                    feet[int(tracks.tracker_id[person])] = points
                    matched_people.add(person)

        # Fallback per-player crop inference for tracked players missed in wide-angle full frame
        unmatched = [i for i in range(len(tracks)) if i not in matched_people]
        if unmatched:
            crops = []
            offsets = []
            valid_indices = []
            for i in unmatched:
                x1, y1, x2, y2 = tracks.xyxy[i]
                bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
                cx1 = max(0, int(round(x1 - 0.15 * bw)))
                cy1 = max(0, int(round(y1 - 0.10 * bh)))
                cx2 = min(w, int(round(x2 + 0.15 * bw)))
                cy2 = min(h, int(round(y2 + 0.15 * bh)))
                if cx2 - cx1 >= 16 and cy2 - cy1 >= 24:
                    crops.append(frame[cy1:cy2, cx1:cx2])
                    offsets.append((cx1, cy1))
                    valid_indices.append(i)
            if crops:
                crop_results = self.model.predict(crops, imgsz=256, conf=.15, device=self.device, verbose=False)
                for person_idx, (ox, oy), res in zip(valid_indices, offsets, crop_results):
                    if res.keypoints is None or res.boxes is None or not len(res.boxes):
                        continue
                    c_xy = res.keypoints.xy.cpu().numpy()
                    c_conf = res.keypoints.keypoints if hasattr(res.keypoints, 'keypoints') else None
                    c_conf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
                    if c_xy.shape[1:] != (17, 2) or c_conf is None:
                        continue
                    points = [
                        [float(c_xy[0, k, 0] + ox), float(c_xy[0, k, 1] + oy)]
                        for k in (15, 16)
                        if c_conf[0, k] >= self.confidence
                        and 0 < c_xy[0, k, 0] + ox < w
                        and 0 < c_xy[0, k, 1] + oy < h
                    ]
                    if points:
                        feet[int(tracks.tracker_id[person_idx])] = points
        return feet


class NoPoseEstimator:
    def estimate(self, frame, tracks):
        return {}
