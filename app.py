"""Streamlit Analytics Dashboard for the 5-Phase AI-Powered Futsal Match Analysis System."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from futsal.phase5.pipeline import build_match_report, compute_heatmaps


def discover_phase_runs(runs_root: Path) -> list[Path]:
    """Find completed Phase 3, 4, or 5 run directories."""
    if not runs_root.is_dir():
        return []
    candidates = []
    for run_json in sorted(runs_root.glob("*/run.json"), reverse=True):
        try:
            meta = json.loads(run_json.read_text(encoding="utf-8"))
            if meta.get("status") == "completed" and (
                (run_json.parent / "dashboard_tracks.csv").is_file()
                or (run_json.parent / "possession_tracks.csv").is_file()
                or (run_json.parent / "metrics.csv").is_file()
                or (run_json.parent / "positions.csv").is_file()
            ):
                candidates.append(run_json.parent)
        except (ValueError, OSError):
            continue
    return candidates


def render_pitch_heatmap(grid: np.ndarray, title: str, length: float = 40.0, width: float = 20.0) -> go.Figure:
    """Create a 40x20m futsal pitch Plotly figure with a 2D occupancy heatmap."""
    bins_y, bins_x = grid.shape
    xs = np.linspace(0.5 * length / bins_x, length - 0.5 * length / bins_x, bins_x)
    ys = np.linspace(0.5 * width / bins_y, width - 0.5 * width / bins_y, bins_y)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=xs,
            y=ys,
            z=grid,
            colorscale="YlOrRd",
            opacity=0.82,
            showscale=True,
            colorbar=dict(title="Frames"),
        )
    )
    # Pitch boundary and halfway line
    fig.add_shape(type="rect", x0=0, y0=0, x1=length, y1=width, line=dict(color="white", width=2))
    fig.add_shape(type="line", x0=length / 2, y0=0, x1=length / 2, y1=width, line=dict(color="white", width=2))
    fig.add_shape(
        type="circle",
        x0=length / 2 - 3,
        y0=width / 2 - 3,
        x1=length / 2 + 3,
        y1=width / 2 + 3,
        line=dict(color="white", width=2),
    )
    fig.update_layout(
        title=title,
        xaxis=dict(range=[-1, length + 1], constrain="domain", showgrid=False, title="Pitch Length (m)"),
        yaxis=dict(
            range=[width + 1, -1],
            scaleanchor="x",
            scaleratio=1,
            showgrid=False,
            title="Pitch Width (m)",
        ),
        plot_bgcolor="#1f3b28",
        paper_bgcolor="#0e1621",
        font=dict(color="#e4edf7"),
        margin=dict(l=20, r=20, t=45, b=20),
        height=390,
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="AI Futsal Match Analysis — Phase 5 Dashboard",
        page_icon="⚽",
        layout="wide",
    )
    st.title("⚽ AI-Powered Futsal Match Analysis Dashboard")
    st.caption("5-Phase Tactical Pipeline · RF-DETR + ByteTrack + Homography Radar + Spatial Voronoi + Pose Events")

    runs_root = Path("runs")
    available_runs = discover_phase_runs(runs_root)

    with st.sidebar:
        st.header("Match Run Selection")
        custom_dir = st.text_input("Run directory path", value=str(available_runs[0]) if available_runs else "")
        if available_runs:
            selected = st.selectbox(
                "Discovered runs",
                options=available_runs,
                format_func=lambda p: p.name,
            )
            run_dir = Path(custom_dir) if custom_dir else selected
        else:
            run_dir = Path(custom_dir) if custom_dir else None

        st.divider()
        st.subheader("Upload New Video")
        uploaded_video = st.file_uploader("Upload Futsal Match (.mp4)", type=["mp4"])
        if uploaded_video is not None:
            uploads_dir = Path("outputs/uploads")
            uploads_dir.mkdir(parents=True, exist_ok=True)
            saved_path = uploads_dir / uploaded_video.name
            saved_path.write_bytes(uploaded_video.getbuffer())
            st.success(f"Saved to `{saved_path}`")
            st.code(
                f"python -m futsal.phase1 --video \"{saved_path}\" --output runs/{saved_path.stem}-p1",
                language="powershell",
            )

    if run_dir is None or not run_dir.is_dir():
        st.info("Select or enter a completed run directory in the sidebar to view match analytics.")
        return

    # Load tracks, summary, events, and report
    csv_candidates = ["dashboard_tracks.csv", "possession_tracks.csv", "metrics.csv", "positions.csv"]
    tracks_file = next((run_dir / name for name in csv_candidates if (run_dir / name).is_file()), None)
    if tracks_file is None:
        st.error(f"No tracking/spatial CSV found in `{run_dir}`.")
        return

    tracks_df = pd.read_csv(tracks_file)
    summary_file = run_dir / "player_summary.csv"
    player_summary_df = pd.read_csv(summary_file) if summary_file.is_file() else pd.DataFrame()
    events_file = run_dir / "events.csv"
    events_df = pd.read_csv(events_file) if events_file.is_file() else pd.DataFrame()

    report_file = run_dir / "match_report.json"
    if report_file.is_file():
        report = json.loads(report_file.read_text(encoding="utf-8"))
    else:
        report = build_match_report(tracks_df, player_summary_df, events_df)

    # Top KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    poss_a = report.get("team_a_possession_pct")
    ctrl_a = report.get("team_a_pitch_control_pct")
    col1.metric(
        "Ball Possession (Team A / B)",
        f"{poss_a:.1f}% / {100.0 - poss_a:.1f}%" if poss_a is not None else "N/A",
    )
    col2.metric(
        "Pitch Control Voronoi (Team A / B)",
        f"{ctrl_a:.1f}% / {100.0 - ctrl_a:.1f}%" if ctrl_a is not None else "N/A",
    )
    col3.metric(
        "Team Distance Covered (A / B)",
        f"{report.get('team_a_distance_m', 0.0):.1f} m / {report.get('team_b_distance_m', 0.0):.1f} m",
    )
    col4.metric(
        "Passes / Shots Detected",
        f"{report.get('passes_detected', 0)} Passes · {report.get('shots_detected', 0)} Shots",
    )

    # Broadcast Tactical Video Player
    video_candidates = [
        run_dir / "broadcast_tactical.mp4",
        run_dir / "tactical_events.mp4",
        run_dir / "tactical_spatial.mp4",
        run_dir / "tactical.mp4",
    ]
    video_file = next((v for v in video_candidates if v.is_file()), None)
    if video_file is not None:
        st.subheader("🎥 Broadcast Tactical Video & 2D Radar")
        st.video(str(video_file))

    # Tabs: Physical Metrics, Heatmaps, Events
    tab_phys, tab_heat, tab_events = st.tabs(
        ["🏃 Distance & Speed per Player", "🔥 2D Pitch Heatmaps", "⚡ Possession & Pass/Shot Events"]
    )

    with tab_phys:
        if not player_summary_df.empty:
            c1, c2 = st.columns([3, 2])
            with c1:
                st.dataframe(player_summary_df, width="stretch")
            with c2:
                valid_p = player_summary_df.dropna(subset=["observed_distance_m"]).copy()
                if not valid_p.empty:
                    valid_p["player_label"] = valid_p["track_id"].apply(lambda x: f"#{int(x)}")
                    fig_bar = px.bar(
                        valid_p,
                        x="player_label",
                        y="observed_distance_m",
                        color="team",
                        title="Total Distance Covered per Player (m)",
                        labels={"player_label": "Player ID", "observed_distance_m": "Distance (m)"},
                        color_discrete_map={"team_a": "#00BFFF", "team_b": "#FF4081", "unknown": "#90A4AE"},
                    )
                    st.plotly_chart(fig_bar, width="stretch")
        else:
            st.info("Run Phase 3 or Phase 5 to populate `player_summary.csv`.")

    with tab_heat:
        heatmaps = compute_heatmaps(tracks_df)
        options = ["team_a", "team_b"] + sorted(k for k in heatmaps if k.startswith("player_"))
        choice = st.selectbox("Select Team or Player ID Heatmap", options=options)
        st.plotly_chart(
            render_pitch_heatmap(heatmaps[choice], title=f"2D Court Occupancy Heatmap — {choice}"),
            width="stretch",
        )

    with tab_events:
        if not events_df.empty:
            st.dataframe(events_df, width="stretch")
        else:
            st.info("No pass/shot candidates recorded in `events.csv` for this segment.")


if __name__ == "__main__":
    main()
