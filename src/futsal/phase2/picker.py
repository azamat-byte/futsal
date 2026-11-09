"""Interactive OpenCV point picker. No inferred model-index mapping is assumed."""

import json
from pathlib import Path

import cv2
import numpy as np

from futsal.phase1.pipeline import write_json
from futsal.phase1.video import VideoReader
from .calibration import fingerprint
from .geometry import Homography, project

# Origin: far left floor corner in the main broadcast view. All locations lie on the floor.
LANDMARKS = [
    ("Halfway / far touchline", (20, 0)),
    ("Near substitution mark, x=15", (15, 20)),
    ("Near substitution mark, x=25", (25, 20)),
    ("Circle / halfway, far", (20, 7)),
    ("Halfway / near touchline", (20, 20)),
    ("Circle / halfway, near", (20, 13)),
    ("Far left corner", (0, 0)), ("Far right corner", (40, 0)),
    ("Near left corner", (0, 20)), ("Near right corner", (40, 20)),
    ("Left 6 m penalty mark", (6, 10)), ("Right 6 m penalty mark", (34, 10)),
    ("Left 10 m penalty mark", (10, 10)), ("Right 10 m penalty mark", (30, 10)),
    ("Near substitution mark, x=10", (10, 20)), ("Near substitution mark, x=30", (30, 20)),
]


def pick(video: Path, output: Path, frame_index=0):
    """Select >=4 fit points and >=2 distinct check points, review the overlay, save."""
    with VideoReader(video) as reader:
        info = reader.info
        if not 0 <= frame_index < info.frame_count:
            raise ValueError("Calibration frame is outside the source video")
        source_hash = fingerprint(video)
        raw = {"schema_version": 1, "source_sha256": source_hash,
               "image_size": [info.width, info.height], "pitch_size_m": [40, 20],
               "scale_verified": False, "max_error_px": 8.0, "anchors": []}
        if output.exists():
            raw = json.loads(output.read_text(encoding="utf-8"))
            if (raw.get("source_sha256") != source_hash or raw.get("image_size") != [info.width, info.height]
                    or raw.get("pitch_size_m") != [40, 20]):
                raise ValueError("Picker requires a matching source and 40 x 20 m calibration file")
        scale = min(1000/info.width, 700/info.height, 1)
        dw, dh = round(info.width*scale), round(info.height*scale)
        height = max(dh+70, 690)
        title = "Futsal Phase 2 - floor calibration"
        state = {"selected": 0, "points": [], "review": None}
        message = "Select a landmark on the right, then click its FLOOR location."
        _, frame = next(reader.frames(frame_index, limit=1))

        def clicked(event, x, y, flags, parameter):
            if event == cv2.EVENT_RBUTTONDOWN and state["points"]:
                state["points"].pop()
                state["review"] = None
            elif event == cv2.EVENT_LBUTTONDOWN:
                if x >= dw and 70 <= y < 70 + len(LANDMARKS)*25:
                    state["selected"] = (y - 70)//25
                elif 0 <= x < dw and 70 <= y < dh+70:
                    index = state["selected"]
                    if index not in [p[0] for p in state["points"]]:
                        state["points"].append((index, [x/scale, (y-70)/scale]))
                        state["review"] = None
                        remaining = [i for i in range(len(LANDMARKS)) if i not in [p[0] for p in state["points"]]]
                        if remaining:
                            state["selected"] = remaining[0]

        cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(title, clicked)
        try:
            while True:
                canvas = np.full((height, dw+390, 3), 28, np.uint8)
                canvas[70:70+dh, :dw] = cv2.resize(frame, (dw, dh))
                for number, (index, xy) in enumerate(state["points"]):
                    p = (round(xy[0]*scale), round(xy[1]*scale)+70)
                    color = (80, 255, 80) if number < 4 else (0, 210, 255)
                    cv2.circle(canvas, p, 5, color, -1)
                    cv2.putText(canvas, str(number+1), (p[0]+7, p[1]-7), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
                if state["review"] is not None:
                    model = state["review"]
                    curves = [[[0, 0], [40, 0], [40, 20], [0, 20], [0, 0]], [[20, 0], [20, 20]]]
                    curves.append([[20+3*np.cos(t), 10+3*np.sin(t)] for t in np.linspace(0, 2*np.pi, 100)])
                    for curve in curves:
                        pixels = project(model.pitch_to_image, curve)*scale
                        cv2.polylines(canvas[70:70+dh, :dw], [np.clip(pixels, -10000, 10000).astype(np.int32)], False,
                                      (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(canvas, f"Frame {frame_index} | first 4 clicks: FIT | further clicks: CHECK", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                cv2.putText(canvas, message[:125], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 220, 255), 1)
                for i, (name, point) in enumerate(LANDMARKS):
                    color = (0, 220, 255) if state["selected"] == i else (225, 225, 225)
                    cv2.putText(canvas, f"{i+1}. {name}", (dw+10, 88+i*25), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                help_lines = ["Click name, then click the image", "Right click / U: undo", "Enter: validate, then Enter: save",
                              "N / P: next / previous 2 seconds", "Q / Esc: close", "Only use visible FLOOR landmarks.",
                              "Do not click crossbars or guess hidden points.", "Yellow overlay must follow court markings.",
                              "40 x 20 m scale is assumed, not measured."]
                for i, text in enumerate(help_lines):
                    cv2.putText(canvas, text, (dw+10, 500+i*20), cv2.FONT_HERSHEY_SIMPLEX, .42, (180, 210, 190), 1)
                cv2.imshow(title, canvas)
                key = cv2.waitKey(30) & 0xff
                if key in (27, ord('q')) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord('u') and state["points"]:
                    state["points"].pop()
                    state["review"] = None
                elif key in (ord('n'), ord('p')):
                    frame_index = int(np.clip(frame_index + (1 if key == ord('n') else -1)*round(2*info.fps),
                                              0, info.frame_count-1))
                    _, frame = next(reader.frames(frame_index, limit=1))
                    state.update(points=[], selected=0, review=None)
                    message = "Select visible landmarks. Unsaved clicks were cleared."
                elif key in (10, 13):
                    if len(state["points"]) < 6:
                        message = "Need 4 fitting and at least 2 independent check landmarks."
                        continue
                    pixels = [p for _, p in state["points"]]
                    meters = [LANDMARKS[i][1] for i, _ in state["points"]]
                    anchor = {"frame_index": frame_index, "image_points": pixels[:4], "pitch_points": meters[:4],
                              "check_image_points": pixels[4:], "check_pitch_points": meters[4:],
                              "landmark_names": [LANDMARKS[i][0] for i, _ in state["points"]], "reviewed": True}
                    if state["review"] is None:
                        try:
                            state["review"] = Homography.fit(pixels[:4], meters[:4], pixels[4:], meters[4:],
                                                             raw.get("max_error_px", 8))
                            message = f"Check error {state['review'].validation_error_px:.1f}px. Inspect overlay; Enter saves."
                        except ValueError as error:
                            message = str(error)
                    else:
                        raw["anchors"] = sorted([a for a in raw["anchors"] if a["frame_index"] != frame_index] + [anchor],
                                                key=lambda a: a["frame_index"])
                        output.parent.mkdir(parents=True, exist_ok=True)
                        write_json(output, raw)
                        message = f"Saved frame {frame_index}. N: next anchor, or Q: finish."
        finally:
            cv2.destroyWindow(title)
    return output if output.exists() else None
