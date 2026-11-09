import base64
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from futsal.analytics import apply_review, derive_run, eligible_positions, movement, occupancy
from futsal.config import RunConfig
from futsal.datasets import reviewed_records, review_record, export_yolo
from futsal.geometry import CalibrationSpec, fit_calibration, project
from futsal.jobs import list_jobs, create_job, update_job, launch_job, corrections, save_correction
from futsal.rendering import court_figure
from futsal.storage import load_tracks, read_json, write_json, sha256
from futsal.video import frame_at, load_image, probe
from futsal.uploads import save_upload, asset_label, preferred_video
from futsal.quality import empty_observation_message

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Futsal Lab", page_icon="⚽", layout="wide", initial_sidebar_state="auto")
st.markdown(
    """<style>
.stApp {background:#0b1421;color:#e4edf7} [data-testid="stSidebar"]{background:#101e30}
h1,h2,h3{letter-spacing:-.025em} [data-testid="stMetric"]{background:#13243a;padding:18px;border-radius:12px;border:1px solid #253952}
.eyebrow{color:#49d8ba;font-size:12px;font-weight:700;letter-spacing:2px}.subtle{color:#99acc4}
div.stButton>button[kind="primary"]{background:#2fc2a2;color:#08231f;border:0}
</style>""",
    unsafe_allow_html=True,
)
canvas_component = components.declare_component(
    "futsal_canvas", path=str(ROOT / "dashboard/components/canvas")
)


def canvas(image, mode="points", boxes=None, class_id=0, key="canvas"):
    _, encoded = cv2.imencode(".jpg", image)
    return canvas_component(
        image="data:image/jpeg;base64," + base64.b64encode(encoded).decode(),
        mode=mode,
        boxes=boxes or [],
        class_id=class_id,
        token=key,
        key=key,
        default=[],
    )


@st.cache_data(show_spinner=False)
def cached_tracks(path, stamp, start, end):
    return load_tracks(path, start=start, end=end, max_rows=250000)


def tracks(run):
    path = Path(run)
    parts = list((path / "tracks").glob("*.parquet"))
    window = st.session_state.get("observation_window", {})
    start = window.get("start") if window.get("run") == str(path) else None
    end = window.get("end") if window.get("run") == str(path) else None
    data = cached_tracks(str(path), max((p.stat().st_mtime_ns for p in parts), default=0), start, end)
    if len(data) >= 250000:
        st.warning("Display capped at 250,000 observations. Narrow the source time window.")
    return data


def pick_run(jobs):
    available = [j for j in jobs if j["run_path"] and (Path(j["run_path"]) / "manifest.json").exists()]
    if not available:
        st.info("Process a clip from the Overview page to create a real analysis run.")
        return None
    job = st.selectbox(
        "Analysis run",
        available,
        format_func=lambda j: f"{Path(j['source']).stem[:65]} · {j['id']} · {j['state']}",
    )
    path = Path(job["run_path"])
    manifest = read_json(path / "manifest.json")
    with st.expander("Observation window", expanded=False):
        start = st.number_input(
            "Window start (source seconds)",
            min_value=0.0,
            value=float(manifest["config"]["start"]),
            step=30.0,
            key=f"window-start-{path}",
        )
        seconds = st.selectbox("Window duration", [30, 120, 300, 600], index=1, key=f"window-duration-{path}")
        st.caption("Inspection loads this bounded interval. Run exports preserve all observations.")
    st.session_state["observation_window"] = {"run": str(path), "start": start, "end": start + seconds}
    return path


def overview(jobs):
    st.markdown('<div class="eyebrow">WORKSPACE</div>', unsafe_allow_html=True)
    st.title("Your match analysis workspace")
    st.caption("Local video intelligence · observations, review, and measurable coverage")
    with st.expander("Import a match or trained weights", expanded=False):
        video_upload = st.file_uploader("Match MP4", type=["mp4"])
        weights_upload = st.file_uploader("Your YOLO weights", type=["pt"])
        if st.button("Save uploaded files"):
            destination = ROOT / "data/uploads"
            for upload in (video_upload, weights_upload):
                if upload is None:
                    continue
                target = save_upload(upload, destination)
                if target.suffix == ".mp4":
                    st.session_state["analysis_source"] = target
                st.success(f"Saved {upload.name}")
    videos = sorted(ROOT.glob("*.mp4")) + sorted((ROOT / "data/uploads").glob("*.mp4"))
    datasets = list((ROOT / "data/datasets").glob("*/manifest.json"))
    cols = st.columns(4)
    cols[0].metric("Match videos", len(videos))
    cols[1].metric("Completed runs", sum(j["state"] == "completed" for j in jobs))
    cols[2].metric("Active jobs", sum(j["state"] in {"running", "queued"} for j in jobs))
    cols[3].metric("Dataset versions", len(datasets))
    with st.container(border=True):
        st.subheader("Analyze a video")
        if not videos:
            st.info("Place an MP4 file in the project folder.")
            return
        if st.session_state.get("analysis_source") not in videos:
            st.session_state["analysis_source"] = preferred_video(videos)
        source = st.selectbox("Source video", videos, format_func=asset_label, key="analysis_source")
        if st.button("Refresh video list"):
            st.rerun()
        source_meta = probe(source)
        source_duration = source_meta.get("duration_s")
        if source_duration:
            st.caption(f"Video length: {source_duration:.1f} seconds")
        c1, c2, c3 = st.columns(3)
        start = c1.number_input("Start at source time (seconds)", min_value=0.0, value=0.0,
                                max_value=max(0.0, source_duration - .1) if source_duration else None,
                                step=1.0, key=f"analysis_start_{source}")
        duration = c2.number_input("Duration (seconds)", min_value=.1,
                                   value=min(10.0, source_duration) if source_duration else 10.0,
                                   step=1.0, key=f"analysis_duration_{source}")
        full = c3.checkbox("Process until end", value=False)
        model_files = ([ROOT / "best.pt"] if (ROOT / "best.pt").is_file() else []) + sorted(
            p for p in (ROOT / "models").glob("*.pt") if "ball" not in p.name and "pose" not in p.name
        )
        model_files += sorted((ROOT / "data/uploads").glob("*.pt"))
        if not model_files:
            st.warning("Place your trained best.pt in the project folder, or select local weights in models/.")
            return
        model = st.selectbox("Detector checkpoint", model_files, format_func=asset_label)
        ball_options = [None] + sorted((ROOT / "models").glob("*ball*.pt"))
        ball_model = st.selectbox(
            "Optional dedicated ball checkpoint",
            ball_options,
            format_func=lambda p: p.name if p else "Use primary model outputs",
        )
        tile_ball = st.checkbox(
            "Extra ball search with the selected detector (slower)", value=False,
            help="Searches overlapping crops with best.pt unless you explicitly select a separate ball checkpoint."
        )
        c1, c2, c3, c4 = st.columns(4)
        device = c1.selectbox("Device", ["0", "cpu"])
        fps = c2.selectbox("Analysis samples / second", [2, 5, 10, 15, 20, 30], index=4)
        imgsz = c3.selectbox("Inference size", [640, 960, 1280], index=2)
        tracker = c4.selectbox("Tracker", ["bytetrack", "botsort"], index=1)
        motion_profile = st.checkbox("Use motion-aware tracking settings", value=True,
                                     help="Recover weak detections into existing tracks, confirm new tracks, and smooth display boxes.")
        appearance_reid = st.checkbox("Use appearance matching for nearby players (BoT-SORT)", value=False,
                                     disabled=tracker != "botsort",
                                     help="Uses the existing local OSNet encoder. Same-kit identity accuracy is unverified.")
        with st.expander("Detection controls", expanded=False):
            enhance = st.checkbox("Mild contrast enhancement", value=False)
            multi_view = st.checkbox("Full-frame and tiled player detection (much slower)", value=False)
            roi_file = st.text_input("Reviewed fixed-camera court ROI file (optional)", value="")
            st.caption("A fixed image ROI must match this source video and is unsuitable for a panning camera.")
        calibration = st.text_input("Calibration file (optional)", value="")
        known_scale = st.checkbox("Calibration uses independently verified court dimensions", value=False)
        render = st.checkbox("Render annotated video", value=True)
        radar = st.checkbox("Export synchronized tactical radar (requires valid calibration)", value=True)
        overlay = st.checkbox("Overlay radar on broadcast", value=True)
        enable_teams = st.checkbox("Classify teams using jersey appearance", value=True)
        show_boxes = st.checkbox("Show detection boxes", value=True)
        show_ids = st.checkbox("Show tracking IDs", value=True)
        show_lines = st.checkbox("Show experimental court line candidates", value=False)
        show_trails = st.checkbox("Show short observed radar trails", value=False)
        c1, c2 = st.columns(2)
        team_a_color = c1.color_picker("Team A visualization color", "#3498ff")
        team_b_color = c2.color_picker("Team B visualization color", "#f05252")
        if st.button("Start analysis", type="primary"):
            active = [j for j in jobs if j["state"] in {"queued", "running"}]
            if active:
                st.warning("A job is active. Finish or cancel it before starting another GPU analysis.")
            else:
                config = RunConfig(
                    model=str(model),
                    device=device,
                    sample_fps=fps,
                    imgsz=imgsz,
                    tracker=tracker,
                    start=start,
                    duration=None if full else duration,
                    render=render,
                    calibration_file=calibration or None,
                    scale_verified=known_scale,
                    ball_model=str(ball_model) if ball_model else None,
                    tile_ball=tile_ball,
                    radar=radar,
                    radar_overlay=overlay,
                    team_mode="color" if enable_teams else "off",
                    show_boxes=show_boxes,
                    show_ids=show_ids,
                    show_court_lines=show_lines,
                    show_trails=show_trails,
                    team_colors={"team_a": team_a_color, "team_b": team_b_color},
                    confidence=.05 if motion_profile else .08,
                    person_confidence=.05 if motion_profile else None,
                    ball_confidence=.05 if motion_profile else None,
                    track_low_threshold=.05 if motion_profile else .07,
                    track_new_threshold=.35 if motion_profile else .30,
                    track_buffer_seconds=1.5 if motion_profile else 1.0,
                    track_min_observations=2 if motion_profile else 1,
                    smooth_boxes=motion_profile,
                    appearance_reid=appearance_reid and tracker == "botsort",
                    reid_checkpoint=str(ROOT / "models/osnet-msmt17.pth"),
                    ball_confirmation_frames=2 if motion_profile else 1,
                    ball_start_confidence=.20 if motion_profile else 0,
                    ball_max_gap=.25 if motion_profile else .4,
                    inference_mode="full+tiled" if multi_view else "full",
                    enhance_contrast=enhance,
                    court_roi_file=roi_file or None,
                )
                job_id = create_job(source, config, ROOT / "runs")
                st.success(f"Started job {job_id}. Refresh progress below.")
    st.subheader("Processing jobs")
    if st.button("Refresh progress"):
        st.rerun()
    for job in jobs[:15]:
        with st.expander(f"{job['state'].upper()} · {Path(job['source']).name} · {job['id']}"):
            st.progress(float(job["progress"] or 0))
            st.caption(f"{job['frames']} sampled frames · attempt {job['attempt']}")
            if job["error"]:
                st.code(job["error"][-4000:])
            if job["state"] in {"running", "queued"}:
                if st.button("Request cancellation", key="cancel" + job["id"]):
                    update_job(ROOT / "runs", job["id"], cancel=1)
                    st.rerun()
            elif job["state"] in {"failed", "cancelled", "interrupted"}:
                st.caption("Restart creates a new complete attempt; it does not splice tracker states.")
                if st.button("Restart safely", key="retry" + job["id"]):
                    launch_job(ROOT / "runs", job["id"])
                    st.rerun()


def run_videos(run, manifest):
    if (run / "annotated.mp4").exists() and manifest["state"] != "running":
        st.video(str(run / "annotated.mp4"))
        for name in ["tactical.mp4", "radar.mp4"]:
            if (run / name).exists():
                with st.expander(name, expanded=False):
                    st.video(str(run / name))
                    if (run / name).stat().st_size < 200_000_000:
                        st.download_button("Download " + name, (run / name).read_bytes(),
                                           file_name=name, mime="video/mp4")
    elif manifest["state"] == "running":
        st.caption("The annotated video becomes playable when encoding finishes.")


def match_viewer(jobs):
    st.title("Match viewer")
    run = pick_run(jobs)
    if run is None:
        return
    manifest = read_json(run / "manifest.json")
    shots = read_json(run / "shots.json", [])
    review = corrections(run)
    data = tracks(run)
    if data.empty:
        message = empty_observation_message(manifest, any((run / "tracks").glob("*.parquet")))
        if manifest["state"] == "completed":
            st.warning(message)
        else:
            st.info(message)
        run_videos(run, manifest)
        return
    lo, hi = float(data.time_s.min()), float(data.time_s.max())
    timestamp = st.slider("Inspect source timestamp", lo, max(hi, lo + 0.1), lo, step=0.1)
    selected = data.iloc[np.argmin(abs(data.time_s - timestamp))].time_s
    current = data[data.time_s == selected]
    image = frame_at(manifest["asset"]["path"], float(selected))
    from futsal.rendering import annotate

    rows = current.to_dict("records")
    cols = st.columns([1.6, 1])
    cols[0].image(
        cv2.cvtColor(
            annotate(
                image, rows, selected, int(current.shot_id.iloc[0]), {"reason": "see calibration records"}
            ),
            cv2.COLOR_BGR2RGB,
        ),
        caption=f"Source PTS time {selected:.3f}s",
    )
    valid = current[current.court_x.notna()]
    if not manifest["config"]["scale_verified"]:
        st.caption("Court scale is unverified. Any court projection uses assumed dimensions.")
    fig = court_figure(manifest["config"]["court_length"], manifest["config"]["court_width"])
    if len(valid):
        fig.add_trace(
            go.Scatter(
                x=valid.court_x,
                y=valid.court_y,
                mode="markers+text",
                text=valid.tracklet_id,
                marker=dict(size=12, color="#44d9b4"),
                textposition="top center",
            )
        )
    cols[1].plotly_chart(fig, use_container_width=True)
    cols[1].caption("Timestamp inspector uses selected observations. Video playback below is separate.")
    run_videos(run, manifest)
    st.subheader("Review broadcast segments")
    st.caption("Only reviewed live wide shots may contribute to court analytics. Replays remain excluded.")
    if shots:
        shot = st.selectbox(
            "Shot", shots, format_func=lambda s: f"{s['shot_id']} · {s['start_s']:.1f}–{s['end_s']:.1f}s"
        )
        label = st.selectbox(
            "Segment label",
            ["unknown", "live_play_wide", "live_play_closeup", "replay", "non_play", "transition"],
            index=[
                "unknown",
                "live_play_wide",
                "live_play_closeup",
                "replay",
                "non_play",
                "transition",
            ].index(review["segments"].get(str(shot["shot_id"]), "unknown")),
        )
        if st.button("Save segment review"):
            save_correction(run, "segments", shot["shot_id"], label)
            derive_run(run)
            st.success("Review saved and analytics refreshed.")
    st.dataframe(current, use_container_width=True, hide_index=True)
    ball_rows = current[current.class_name == "ball"]
    if len(ball_rows):
        with st.expander("Verify ball contact with the floor"):
            st.caption("Only confirm if the ball is on the floor and this frame has a valid calibration.")
            frame_records = [
                json.loads(line) for line in (run / "frames.jsonl").read_text("utf-8").splitlines()
            ]
            mapping = next((r for r in frame_records if r["time_s"] == selected), None)
            grounded = st.checkbox("The visible ball is on the floor in this frame")
            if st.button("Save grounded-ball observation"):
                if (
                    not grounded
                    or not mapping
                    or mapping["image_to_court"] is None
                    or not manifest["config"]["scale_verified"]
                ):
                    st.error(
                        "Ground-contact confirmation, valid calibration, and verified scale are required."
                    )
                else:
                    b = ball_rows.iloc[0]
                    xy = project(mapping["image_to_court"], [[(b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2]])[0]
                    save_correction(
                        run,
                        "ball_ground",
                        int(b.frame_index),
                        {"x": float(xy[0]), "y": float(xy[1]), "reviewed": True, "scale_verified": True},
                    )
                    st.success("Saved. Recompute event hypotheses from the review page.")


def analytics(jobs):
    st.title("Player & team analytics")
    run = pick_run(jobs)
    if run is None:
        return
    data = tracks(run)
    if data.empty:
        return
    review = corrections(run)
    data = apply_review(data, review)
    valid = eligible_positions(data)
    c1, c2, c3 = st.columns(3)
    c1.metric("Anonymous tracklets", data.tracklet_id.nunique())
    c2.metric("Metric position observations", len(valid))
    c3.metric("Times with metric coverage", f"{valid.time_s.nunique() / max(data.time_s.nunique(), 1):.1%}")
    st.caption(
        "Tracklet count is not player count. Coverage reflects sampled times with at least one valid position."
    )
    if valid.empty:
        st.info(
            "Metric analytics need reviewed live-wide segments, calibrated frames, and verified court scale. "
            "Detection and image-space tracking remain available in Match viewer."
        )
        count = data.groupby("time_s").size().rename("observations")
        st.line_chart(count)
        return
    entities = sorted(valid.entity_id.unique())
    selected = st.multiselect("Entities", entities, default=entities[:1])
    visible = valid[valid.entity_id.isin(selected)]
    lo, hi = float(valid.time_s.min()), float(valid.time_s.max())
    interval = st.slider("Source time range", lo, max(lo + 0.1, hi), (lo, max(lo + 0.1, hi)), step=0.1)
    visible = visible[visible.time_s.between(*interval)]
    config = read_json(run / "manifest.json")["config"]
    fig = court_figure(config["court_length"], config["court_width"])
    if len(visible):
        heat, x, y = occupancy(visible, config["court_length"], config["court_width"])
        fig.add_trace(
            go.Heatmap(
                x=(x[:-1] + x[1:]) / 2,
                y=(y[:-1] + y[1:]) / 2,
                z=heat,
                colorscale="Teal",
                opacity=0.8,
                colorbar=dict(title="Observed s"),
            )
        )
    st.plotly_chart(fig, use_container_width=True)
    stats, trajectory = movement(visible)
    st.dataframe(stats, use_container_width=True, hide_index=True)
    if len(trajectory):
        for entity, part in trajectory.groupby("entity_id"):
            for piece, seg in part.groupby("piece"):
                fig.add_trace(
                    go.Scatter(
                        x=seg.filtered_court_x,
                        y=seg.filtered_court_y,
                        mode="lines",
                        name=str(entity),
                        line=dict(width=2),
                    )
                )
        st.plotly_chart(fig, use_container_width=True, key="trajectory")
    st.warning(
        "Physical accuracy requires an independent positional reference. Distances cover valid observed intervals only."
    )


def dataset_studio():
    st.title("Dataset studio")
    paths = sorted(
        (
            p
            for p in (ROOT / "data/datasets").glob("*/manifest.json")
            if not read_json(p).get("superseded_audit")
        ),
        reverse=True,
    )
    if not paths:
        st.info("Import a dataset using futsal audit-dataset.")
        return
    dataset_paths = [p.parent for p in paths]
    preferred = next((i for i, p in enumerate(dataset_paths) if p.name == "futsal-v2"), 0)
    dataset = st.selectbox("Dataset version", dataset_paths, index=preferred, format_func=lambda p: p.name)
    manifest, records = reviewed_records(dataset)
    cols = st.columns(4)
    cols[0].metric("Images", len(records))
    cols[1].metric("Reviewed complete", sum(r["reviewed"] and r["complete"] for r in records))
    cols[2].metric("Images flagged", sum(bool(r["issues"] or r.get("warnings")) for r in records))
    cols[3].metric("Classes", len(manifest["names"]))
    st.caption("Original labels are immutable. Polygons have been converted to enclosing detection boxes.")
    chart = pd.DataFrame(list(manifest["class_counts"].items()), columns=["class", "boxes"])
    st.bar_chart(chart, x="class", y="boxes")
    issues_only = st.checkbox("Show only flagged images")
    filtered = [r for r in records if not issues_only or r["issues"] or r.get("warnings")]
    if not filtered:
        return
    index = st.number_input("Image number", min_value=1, max_value=len(filtered), value=1, step=1)
    record = filtered[index - 1]
    st.caption(record["id"])
    if record["issues"]:
        st.warning("; ".join(record["issues"]))
    if record.get("warnings"):
        st.warning("; ".join(record["warnings"]))
    image = load_image(record["image"])
    if image is None:
        st.error("Unreadable image")
        return
    selected_class = st.selectbox(
        "Class for new boxes", range(len(manifest["names"])), format_func=lambda i: manifest["names"][i]
    )
    marks = canvas(
        image,
        mode="boxes",
        boxes=record["boxes"],
        class_id=selected_class,
        key=f"boxes-{dataset.name}-{record['id']}",
    )
    table = pd.DataFrame(record["boxes"] + marks, columns=["class_id", "x1", "y1", "x2", "y2"])
    edited = st.data_editor(
        table,
        num_rows="dynamic",
        use_container_width=True,
        key=f"table-{dataset.name}-{record['id']}-{len(marks)}",
    )
    c1, c2 = st.columns(2)
    group = c1.text_input("Source match/group (required)", value=record.get("group") or "")
    splits = ["train", "val", "test"]
    split = c2.selectbox(
        "Split", splits, index=splits.index(record["split"]) if record["split"] in splits else 0
    )
    complete = st.checkbox("I checked the whole image and all in-scope objects are labeled", value=False)
    if st.button("Save reviewed annotation", type="primary"):
        try:
            boxes = edited.dropna().to_dict("records")
            for b in boxes:
                b["class_id"] = int(b["class_id"])
            review_record(dataset, record["id"], boxes, group, split, complete)
            st.success("Review saved. Original labels preserved.")
        except ValueError as exc:
            st.error(str(exc))
    with st.expander("Export reviewed YOLO dataset"):
        name = st.text_input("New export folder name", value="reviewed-v1")
        if st.button("Export complete reviewed labels"):
            try:
                if Path(name).name != name:
                    raise ValueError("Enter a folder name only")
                target = export_yolo(dataset, ROOT / "data/exports" / name)
                st.success(str(target))
            except (ValueError, FileExistsError) as exc:
                st.error(str(exc))


def calibration_page(jobs):
    st.title("Court calibration")
    run = pick_run(jobs)
    if run is None:
        return
    manifest = read_json(run / "manifest.json")
    config = manifest["config"]
    timestamp = st.number_input(
        "Exact source time for anchor", min_value=0.0,
        value=min(float(config["start"]), max(0.0, (manifest["asset"].get("duration_s") or float('inf')) - .1)),
        max_value=max(0.0, manifest["asset"]["duration_s"] - .1) if manifest["asset"].get("duration_s") else None,
        step=0.1
    )
    image = frame_at(manifest["asset"]["path"], timestamp)
    points = canvas(image, key=f"cal-{run}-{timestamp}")
    st.caption(
        "Click 4+ fitting landmarks and at least one independent validation landmark. "
        "Use actual visible floor markings; do not use elevated goal posts."
    )
    values = [
        {"image_x": p[0], "image_y": p[1], "court_x": None, "court_y": None, "validation": False}
        for p in points
    ]
    table = st.data_editor(
        pd.DataFrame(values, columns=["image_x", "image_y", "court_x", "court_y", "validation"]),
        use_container_width=True,
        num_rows="dynamic",
        key=f"cal-table-{run}-{timestamp}-{len(points)}",
    )
    c1, c2 = st.columns(2)
    length = c1.number_input("Court length (m)", min_value=1.0, value=float(config["court_length"]))
    width = c2.number_input("Court width (m)", min_value=1.0, value=float(config["court_width"]))
    scale = st.checkbox("Court dimensions independently verified")
    approved = st.checkbox("Correspondences manually checked")
    if st.button("Validate and save calibration", type="primary"):
        try:
            if table.isna().any().any():
                raise ValueError("Complete all point coordinates")
            fit = table[~table.validation.astype(bool)]
            check = table[table.validation.astype(bool)]
            spec = CalibrationSpec(
                time_s=timestamp,
                image_points=fit[["image_x", "image_y"]].values.tolist(),
                court_points=fit[["court_x", "court_y"]].values.tolist(),
                validation_image_points=check[["image_x", "image_y"]].values.tolist(),
                validation_court_points=check[["court_x", "court_y"]].values.tolist(),
                length=length,
                width=width,
                scale_verified=scale,
                reviewed=approved,
                source_sha256=sha256(manifest["asset"]["path"]),
            )
            matrix, info = fit_calibration(spec)
            target = ROOT / "data/calibrations" / f"{run.parent.name}-{run.name}.json"
            anchors = read_json(target, [])
            anchors = [a for a in anchors if abs(a["time_s"] - timestamp) > 0.01] + [spec.model_dump()]
            write_json(target, sorted(anchors, key=lambda a: a["time_s"]))
            st.json(info)
            st.success(
                f"Saved {target}. Use this file in a new analysis run; existing observations stay unchanged."
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            st.error(str(exc))
    st.plotly_chart(court_figure(length, width), use_container_width=True)


def identities_events(jobs):
    st.title("Identity & event review")
    run = pick_run(jobs)
    if run is None:
        return
    data = tracks(run)
    ids = sorted(data.tracklet_id.dropna().unique())
    if not ids:
        return
    selected = st.multiselect("Tracklets belonging to the same player", ids)
    entity = st.text_input("Reviewed anonymous or roster identity")
    if st.button("Save identity mapping"):
        try:
            from futsal.reid import validate_merge

            if not selected or not entity.strip():
                raise ValueError("Select tracklets and enter an identity")
            existing = corrections(run)["identities"]
            combined = set(selected) | {k for k, v in existing.items() if v == entity}
            validate_merge(load_tracks(run, columns=["time_s", "tracklet_id"]), list(combined))
            for tracklet in selected:
                save_correction(run, "identities", tracklet, entity)
            st.success("Mapping saved.")
        except ValueError as exc:
            st.error(str(exc))
    tracklet = st.selectbox("Correct team for tracklet", ids)
    team = st.selectbox("Team", ["unknown", "team_a", "team_b"])
    if st.button("Save team assignment"):
        save_correction(run, "teams", tracklet, team)
        st.success("Team review saved.")
    role = st.selectbox("Reviewed role", ["unknown", "outfield_player", "goalkeeper", "referee", "staff"])
    if st.button("Save role assignment"):
        save_correction(run, "roles", tracklet, role)
        st.success("Role review saved.")
    with st.expander("Formation classification (requires a trained, reviewed model)"):
        model_path = st.text_input("Local formation model path", value="")
        st.caption("Uses a full window of four outfield players, with reviewed attacking direction.")
        normalized = st.checkbox("The selected team attacks in increasing court x over this window")
        if st.button("Classify observed team window"):
            try:
                from futsal.formations import predict_formation

                reviewed = apply_review(data, corrections(run))
                valid = eligible_positions(reviewed)
                positions = []
                for timestamp, part in valid[
                    (valid.team == team) & (valid.role == "outfield_player")
                ].groupby("time_s"):
                    if len(part) == 4:
                        positions.append(part[["court_x", "court_y"]].to_numpy())
                if not normalized or len(positions) < 10 or not Path(model_path).is_file():
                    raise ValueError(
                        "Need a local trained model, 10+ complete frames and direction confirmation"
                    )
                st.json(predict_formation(model_path, positions))
            except (ValueError, FileNotFoundError) as exc:
                st.error(str(exc))
    st.subheader("Event hypotheses")
    if st.button("Generate OSNet identity candidates"):
        from futsal.reid import osnet_candidates

        checkpoint = ROOT / "models/osnet-msmt17.pth"
        if not checkpoint.exists():
            st.info("Run futsal download-reid first.")
        else:
            with st.spinner("Extracting appearance features; candidates always need review..."):
                osnet_candidates(run, str(checkpoint), max_tracks=50)
            st.success("Identity candidates saved.")
    candidates = read_json(run / "identity_candidates.json", [])
    if candidates:
        st.dataframe(pd.DataFrame(candidates), use_container_width=True, hide_index=True)
    if st.button("Refresh quality-gated analytics"):
        try:
            derive_run(run)
        except ValueError as exc:
            st.error(str(exc))
    events = read_json(run / "pass_candidates.json", [])
    if not events:
        st.info("No supported pass hypotheses. Ball ground-contact evidence is required.")
    else:
        event = st.selectbox(
            "Candidate pass",
            events,
            format_func=lambda e: f"{e['start_s']:.2f}s · {e['source']} → {e['receiver']}",
        )
        status = st.selectbox("Review decision", ["uncertain", "accepted", "rejected"])
        if st.button("Save event review"):
            save_correction(run, "events", event["event_id"], status)
            st.success("Event review saved.")


def exports_page(jobs):
    st.title("Export center")
    run = pick_run(jobs)
    if run is None:
        return
    st.caption("Exports include source timing, quality, missingness, model provenance, and human reviews.")
    if st.button("Build verified export bundle", type="primary"):
        from futsal.exports import export_run
        import uuid

        target = ROOT / "data/exports" / f"run-{run.parent.name}-{uuid.uuid4().hex[:6]}"
        archive = export_run(run, target)
        st.session_state["export_archive"] = str(archive)
    archive = st.session_state.get("export_archive")
    if archive and Path(archive).exists():
        st.download_button(
            "Download analysis ZIP",
            Path(archive).read_bytes(),
            file_name=Path(archive).name,
            mime="application/zip",
        )
    if (run / "annotated.mp4").exists():
        st.caption(f"Annotated H264 analysis proxy: {run / 'annotated.mp4'}")
    summary = read_json(run / "analytics_summary.json", {})
    st.json(summary)


def evaluation_page():
    st.title("Evaluation & evidence")
    results = sorted((ROOT / "reports").glob("**/evaluation.json"))
    if not results:
        st.info("No evaluated benchmark yet. Review labels before an acceptance evaluation.")
    for path in results:
        report = read_json(path)
        with st.expander(f"{path.parent.name} · {report['status']}", expanded=True):
            if report.get("warning"):
                st.warning(report["warning"])
            st.json(report)
    status = ROOT / "docs/status.md"
    if status.exists():
        st.markdown(status.read_text("utf-8"))


with st.sidebar:
    st.markdown("## ⚽ Futsal Lab")
    st.caption("BROADCAST ANALYSIS STUDIO")
    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Match viewer",
            "Analytics",
            "Dataset studio",
            "Calibration",
            "Identities & events",
            "Exports",
            "Evaluation",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("Local processing · private workspace")
    st.caption("Automatic outputs remain estimates until reviewed and evaluated.")

jobs = list_jobs(ROOT / "runs")
try:
    {
        "Overview": lambda: overview(jobs),
        "Match viewer": lambda: match_viewer(jobs),
        "Analytics": lambda: analytics(jobs),
        "Dataset studio": dataset_studio,
        "Calibration": lambda: calibration_page(jobs),
        "Identities & events": lambda: identities_events(jobs),
        "Exports": lambda: exports_page(jobs),
        "Evaluation": evaluation_page,
    }[page]()
except Exception as exc:
    st.error(f"{type(exc).__name__}: {exc}")
    with st.expander("Technical details"):
        st.exception(exc)
