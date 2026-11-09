# Futsal live overlay

`futsal_live_overlay.py` uses the same preloaded models as the original script:
two Roboflow-compatible models with `infer(frame, confidence=...)` and a trained
team classifier with `predict(list_of_BGR_crops)` returning labels `0` or `1`.
Model loading/training remains your responsibility; importing this module does
not start inference or open a video.

Install dependencies in your notebook or Python environment:

```sh
pip install numpy opencv-python supervision==0.25.1 tqdm imageio-ffmpeg
```

In Colab, upload the script, then run this after loading your models:

```python
from futsal_live_overlay import OverlayConfig, run_overlay

stats = run_overlay(
    PLAYER_DETECTION_MODEL,
    FIELD_DETECTION_MODEL,
    team_classifier,
    OverlayConfig(
        source_path="/content/1103.mp4",
        output_path="/content/futsal_live_overlay.mp4",
        radar_position="bottom-center",  # also bottom-left or bottom-right
        radar_width_ratio=0.47,
        radar_opacity=0.62,
        keypoint_interval=5,
        max_homography_age=10,
        team_interval=5,
        overwrite=True,  # explicitly allow replacing a previous result
    ),
)
print(stats)
```

Pasting the complete script into a notebook cell also uses those existing
model globals. The explicit function call is recommended for changing settings.

For a short preview, set `max_frames=300`. The video is decoded until EOF even
when its frame count is unknown or inaccurate. The report counts frames actually
processed. Processing remains sequential; "live" refers to the rendered overlay,
not a low-latency streaming server.

For command-line use, define a zero-argument factory in an importable module:

```python
# my_models.py
def load_models():
    # Load your existing models and trained crop classifier here.
    return player_model, field_model, team_classifier
```

Then run:

```sh
python futsal_live_overlay.py --source match.mp4 --output overlay.mp4 --model-factory my_models:load_models
python futsal_live_overlay.py --help
```

The radar keeps the original team colors (cyan and magenta), neutral gray
goalkeepers, and yellow ball. Player IDs appear beside radar dots; disable them
with `show_radar_ids=False` or `--hide-radar-ids`. The status indicates calibration
landmark count, age in frames when reused, or unavailable calibration. Dots are
hidden once calibration expires.

Calibration age is measured from the last successful fit. Multiple court
instances are evaluated, and RANSAC fits require enough inliers, noncollinear
points, and acceptable error in court meters. These checks reject some bad
fits; they do not establish the correctness of the inherited landmark mapping.
Supply `active_ids`, `full_keypoint_count`, and `landmarks` through `OverlayConfig`
if your trained skeleton differs. The court is 40 by 20 meters. Rapid camera
motion can invalidate a reused fit before its age expires: reduce the keypoint
interval and maximum age for moving cameras. Ball projection assumes the ball
is on the floor and is inaccurate while airborne.

Team classification uses clipped, nonempty crops and a rolling vote. Known
tracks are classified every five frames by default; new or unknown tracks are
classified immediately. Old track histories expire, preventing unbounded
growth. Goalkeeper teams are left unknown.

OpenCV capture/writer resources close on failures. Each run has its own temporary
directory, and the output is published only after rendering and encoding finish.
An existing output requires `overwrite=True` or `--overwrite`; source and output
cannot be the same file. Temporary files are removed after success or failure.
The default publication uses a hard link within the destination filesystem;
use a filesystem supporting hard links, such as NTFS or ext4.

FFmpeg (from PATH or `imageio-ffmpeg`) produces H.264 MP4 with the first source
audio stream when present. A short audio stream does not truncate the video.
If encoding is unavailable or fails, the script logs a warning and saves the
OpenCV MPEG-4 video without audio. Odd image dimensions are padded to even
dimensions for video encoding.

Regression checks use synthetic videos and deterministic fake model outputs,
with actual Supervision conversion/tracking and OpenCV encoding:

```sh
python -m pytest tests/test_live_overlay.py -q
```

The tests skip when Supervision is not installed. No trained weights or
Roboflow credentials are required for these checks.
