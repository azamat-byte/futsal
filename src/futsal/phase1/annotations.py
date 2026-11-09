"""Display colors are separate from original detection class IDs."""

import cv2
import numpy as np
import supervision as sv

GROUPS = ("team_a", "team_b", "goalkeeper", "referee", "other", "unknown")
COLORS = ("#00BFFF", "#FF4081", "#B388FF", "#FFD740", "#FF9100", "#B0BEC5")


class TrackingAnnotator:
    def __init__(self) -> None:
        palette = sv.ColorPalette.from_hex(list(COLORS))
        self.boxes = sv.BoxAnnotator(color=palette, thickness=2)
        self.labels = sv.LabelAnnotator(color=palette, text_color=sv.Color.BLACK, text_scale=.5,
                                        text_padding=4)

    def annotate(self, frame: np.ndarray, tracks: sv.Detections, teams: list[str],
                 frame_index: int, segment_id: int) -> np.ndarray:
        scene = frame.copy()
        if len(tracks):
            roles = tracks.data["role"]
            groups = [role if role in {"goalkeeper", "referee"} else team for role, team in zip(roles, teams)]
            color_indices = np.array([GROUPS.index(group) for group in groups], dtype=int)
            names = {"team_a": "Team A", "team_b": "Team B", "goalkeeper": "GK",
                     "referee": "Ref", "other": "Other", "unknown": "Unknown"}
            labels = [f"#{int(track_id)} {names[group]}"
                      for track_id, group in zip(tracks.tracker_id, groups)]
            scene = self.boxes.annotate(scene, tracks, custom_color_lookup=color_indices)
            # Move only display label anchors to keep text inside the image; raw
            # measurement boxes and box annotations retain their source geometry.
            label_tracks = tracks[np.arange(len(tracks))]
            for i, label in enumerate(labels):
                (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)
                label_tracks.xyxy[i, 0] = np.clip(label_tracks.xyxy[i, 0], 0, max(0, scene.shape[1] - width - 8))
                label_tracks.xyxy[i, 1] = min(scene.shape[0] - 1, max(label_tracks.xyxy[i, 1], 30 + height + 8))
            scene = self.labels.annotate(scene, label_tracks, labels=labels, custom_color_lookup=color_indices)
        cv2.rectangle(scene, (0, 0), (min(scene.shape[1] - 1, 520), 30), (24, 24, 24), -1)
        cv2.putText(scene, f"Phase 1 | frame {frame_index} | segment {segment_id}", (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (240, 240, 240), 1, cv2.LINE_AA)
        return scene
