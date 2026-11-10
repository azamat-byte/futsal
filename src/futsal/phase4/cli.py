"""Phase 4 CLI: Ball Possession & Event Engine using Pose Keypoints."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from futsal.match.pose import AnklePoseEstimator, NoPoseEstimator
from futsal.phase1.config import Phase1Config
from futsal.phase1.detection import RoboflowDetector
from futsal.roboflow_keypoints import DEFAULT_KEYPOINT_MODEL_ID, RoboflowKeypointModel
class LocalYoloBallDetector:
    """Local YOLO ball detector using tiled inference + court-ball continuity for wide-angle futsal."""

    def __init__(self, weights_path: Path, device: str = "cpu", confidence: float = 0.08) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights_path))
        self.device = device
        self.confidence = confidence
        self.last_xy: tuple[float, float] | None = None
        self.missed: int = 0

    def detect(self, frame, rows=None, feet=None):
        import cv2
        import numpy as np
        import supervision as sv
        from futsal.inference import tile_windows

        h, w = frame.shape[:2]
        rows = rows or []
        feet = feet or {}

        def _is_player_sock_or_shoe(cx: float, cy: float, is_yolo: bool) -> bool:
            # Reject any candidate that sits directly on a player's ankle/shoe keypoint (< 18px)
            for pts in feet.values():
                for ax, ay in pts:
                    if float(np.hypot(cx - ax, cy - ay)) < (12.0 if is_yolo else 20.0):
                        return True
            # Reject candidates inside a player's torso/leg/sock column above the floor
            for r in rows:
                x1, y1, x2, y2 = float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])
                bw, bh = x2 - x1, y2 - y1
                if (x1 - 4.0) <= cx <= (x2 + 4.0) and y1 <= cy <= (y2 - 0.06 * bh):
                    return True
                # For non-YOLO white blobs, also reject the entire lower foot/shoe box of white-kit players
                if not is_yolo and (x1 - 6.0) <= cx <= (x2 + 6.0) and (y1 + 0.50 * bh) <= cy <= (y2 + 8.0):
                    if str(r.get("team", "")) == "team_b":
                        return True
            return False

        wins = [
            (x1, y1, x2, y2)
            for x1, y1, x2, y2 in tile_windows(w, h, size=480, overlap=0.2)
            if y2 > int(h * 0.36) and y1 < int(h * 0.86)
        ]
        crops = [frame[y1:y2, x1:x2] for x1, y1, x2, y2 in wins]
        results = self.model.predict(crops, imgsz=640, conf=self.confidence, device=self.device, verbose=False)

        candidates: list[tuple[float, float, float, float, float]] = []
        for (ox, oy, _, _), res in zip(wins, results):
            if res.boxes is None or len(res.boxes) == 0:
                continue
            xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            conf = res.boxes.conf.cpu().numpy().astype(np.float32)
            cls_ids = res.boxes.cls.cpu().numpy().astype(int)
            for box, cf, cid in zip(xyxy, conf, cls_ids):
                name = str(res.names.get(int(cid), "unknown")).lower()
                if "ball" not in name:
                    continue
                bx1, by1, bx2, by2 = float(box[0] + ox), float(box[1] + oy), float(box[2] + ox), float(box[3] + oy)
                bw, bh = bx2 - bx1, by2 - by1
                cx, cy = (bx1 + bx2) * 0.5, (by1 + by2) * 0.5
                if not (8.0 <= bw <= 38.0 and 8.0 <= bh <= 38.0 and 0.65 <= bw / max(1.0, bh) <= 1.55):
                    continue
                if not (h * 0.39 <= cy <= h * 0.85):
                    continue
                if _is_player_sock_or_shoe(cx, cy, is_yolo=True):
                    continue
                score = float(cf)
                if self.last_xy is not None:
                    dist = float(np.hypot(cx - self.last_xy[0], cy - self.last_xy[1]))
                    if dist <= 110.0:
                        score = min(0.98, score + 0.30 * (1.0 - dist / 110.0))
                    elif self.missed <= 3 and dist > 240.0:
                        score *= 0.45
                candidates.append((score, bx1, by1, bx2, by2))

        # Court-ball pattern detector (high-contrast white + dark pentagons on blue court, outside player shoes/socks)
        if not candidates or max(c[0] for c in candidates) < 0.25:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            blue = cv2.inRange(hsv, (95, 55, 95), (128, 255, 255))
            white = cv2.inRange(hsv, (0, 0, 190), (180, 70, 255))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
            court_area = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
            ball_mask = cv2.bitwise_and(white, court_area)
            num, _, stats, centroids = cv2.connectedComponentsWithStats(ball_mask)
            for i in range(1, num):
                x, y, bw, bh, area = stats[i]
                cx, cy = float(centroids[i][0]), float(centroids[i][1])
                if not (38 <= area <= 320 and 8 <= bw <= 24 and 8 <= bh <= 24 and 0.68 <= bw / max(1, bh) <= 1.48):
                    continue
                if not (h * 0.40 <= cy <= h * 0.84 and w * 0.08 <= cx <= w * 0.94):
                    continue
                if _is_player_sock_or_shoe(cx, cy, is_yolo=False):
                    continue
                px1, py1 = max(0, int(cx - 16)), max(0, int(cy - 16))
                px2, py2 = min(w, int(cx + 16)), min(h, int(cy + 16))
                blue_frac = float(np.mean(blue[py1:py2, px1:px2] > 0))
                patch_gray = cv2.cvtColor(
                    frame[max(0, y - 2) : min(h, y + bh + 2), max(0, x - 2) : min(w, x + bw + 2)],
                    cv2.COLOR_BGR2GRAY,
                )
                if (
                    blue_frac < 0.45
                    or patch_gray.size == 0
                    or float(np.std(patch_gray)) < 35.0
                    or int(patch_gray.min()) > 90
                    or int(patch_gray.max()) < 205
                ):
                    continue
                score = 0.30 + 0.15 * min(1.0, float(np.std(patch_gray)) / 60.0)
                if self.last_xy is not None:
                    dist = float(np.hypot(cx - self.last_xy[0], cy - self.last_xy[1]))
                    if dist <= 95.0:
                        score += 0.25 * (1.0 - dist / 95.0)
                    elif self.missed <= 3:
                        continue
                candidates.append((min(0.95, score), float(x), float(y), float(x + bw), float(y + bh)))

        if not candidates:
            self.missed += 1
            return sv.Detections.empty()

        # Sort and deduplicate so BallSelector receives a single clear winner
        candidates.sort(key=lambda item: -item[0])
        best_score, bx1, by1, bx2, by2 = candidates[0]
        final_score = max(0.35, min(0.98, best_score * 1.35))
        self.last_xy = ((bx1 + bx2) * 0.5, (by1 + by2) * 0.5)
        self.missed = 0
        return sv.Detections(
            xyxy=np.array([[bx1, by1, bx2, by2]], dtype=np.float32),
            confidence=np.array([final_score], dtype=np.float32),
            class_id=np.array([0], dtype=int),
            data={"class_name": np.array(["ball"]), "role": np.array(["ball"])},
        )

    def close(self) -> None:
        pass


from .pipeline import Phase4Pipeline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 4: Ball Possession & Pass/Shot Event Detection using Pose"
    )
    parser.add_argument("--phase2-run", type=Path, required=True, help="Completed Phase 2 run directory")
    parser.add_argument("--phase3-run", type=Path, help="Optional completed Phase 3 run directory for speeds")
    parser.add_argument("--output", type=Path, required=True, help="New output directory for Phase 4")
    parser.add_argument("--config", type=Path, help="Phase 1 JSON config for ball detector")
    parser.add_argument("--detector-weights", type=Path, help="Optional local YOLO detector weights (e.g. best.pt)")
    parser.add_argument(
        "--pose-model-id",
        default=DEFAULT_KEYPOINT_MODEL_ID,
        help="Roboflow keypoint/pose model ID (default: s-workspace-bimfr/keypoints-ovj9c-11-rfdetr-keypoint-two-stage-t1)",
    )
    parser.add_argument("--pose-weights", type=Path, help="Optional local YOLO pose checkpoint instead of Roboflow")
    parser.add_argument("--device", default="cpu", help="Device for local inference (e.g. cpu or 0)")
    parser.add_argument("--confirm-seconds", type=float, default=0.03, help="Temporal smoothing dwell time (s)")
    parser.add_argument("--no-pose", action="store_true", help="Disable pose inference")
    parser.add_argument("--video", type=Path, help="Optional relocated source video")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    detector = None
    pose = None
    try:
        api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
        cfg = Phase1Config.load(args.config)
        if args.detector_weights is not None:
            detector = LocalYoloBallDetector(args.detector_weights, device=args.device)
        elif not api_key and Path("best.pt").is_file():
            detector = LocalYoloBallDetector(Path("best.pt"), device=args.device)
        else:
            detector = RoboflowDetector(cfg.detector, api_key)

        if args.no_pose:
            pose = NoPoseEstimator()
        elif args.pose_weights is not None:
            pose = AnklePoseEstimator(args.pose_weights, device=args.device)
        elif not api_key and Path("yolo11n-pose.pt").is_file():
            pose = AnklePoseEstimator(Path("yolo11n-pose.pt"), device=args.device)
        else:
            pose = RoboflowKeypointModel(
                api_key=api_key,
                model_id=args.pose_model_id,
                api_url=cfg.detector.api_url,
            )
        output = Phase4Pipeline(detector, pose, confirm_s=args.confirm_seconds).run(
            args.phase2_run, args.output, args.phase3_run, args.video
        )
        logging.info("Completed Phase 4: %s", output)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        logging.error("%s", error)
        return 1
    except KeyboardInterrupt:
        return 130
    finally:
        if detector is not None:
            detector.close()
        if hasattr(pose, "close"):
            pose.close()


if __name__ == "__main__":
    raise SystemExit(main())
