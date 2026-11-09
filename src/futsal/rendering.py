import cv2

COLORS = {"team_a": (240, 160, 60), "team_b": (90, 100, 245), "unknown": (160, 160, 160)}


def annotate(image, rows, time, shot, calibration, show_boxes=True, show_ids=True, colors=None):
    from .radar import palette

    colors = palette(colors)
    output = image.copy()
    for row in rows:
        box = tuple(int(row[k]) for k in ("x1", "y1", "x2", "y2"))
        display_box = row.get("display_box")
        if isinstance(display_box, (list, tuple)) and len(display_box) == 4:
            box = tuple(int(v) for v in display_box)
        elif hasattr(display_box, "shape") and display_box.shape == (4,):
            box = tuple(int(v) for v in display_box)
        color = colors["ball"] if row["class_name"] == "ball" else colors[
            "referee" if row.get("role") == "referee" else row["team"]]
        if show_boxes:
            cv2.rectangle(output, box[:2], box[2:], color, 2)
        short_id = row["tracklet_id"].split(":")[-1] if row["tracklet_id"] else ""
        role = {"outfield_player": "P", "goalkeeper": "GK", "referee": "REF"}.get(
            row.get("role"), "BALL" if row["class_name"] == "ball" else "P?")
        label = f"{role} {row['score']:.2f}"
        if show_ids:
            label = (short_id + " " + label).strip()
        if show_boxes or show_ids:
            scale = .42 if image.shape[1] < 1000 else .55
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
            x = max(0, min(box[0], output.shape[1] - tw - 4))
            y = max(th + 4, box[1] - 5)
            cv2.rectangle(output, (x, y - th - 3), (x + tw + 4, y + baseline), (18, 20, 22), -1)
            cv2.putText(output, label, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        if row["class_name"] == "ball":
            cv2.circle(output, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2), 5, color, 2)
    cv2.rectangle(output, (0, 0), (min(output.shape[1], 1000), 55), (20, 20, 20), -1)
    cv2.putText(
        output,
        f"Source {time:.2f}s | shot {shot} | team/identity estimates unverified",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (240, 240, 240),
        1,
    )
    cv2.putText(
        output,
        f"Calibration: {calibration['reason']}",
        (10, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (170, 210, 240),
        1,
    )
    return output


def court_figure(length=40, width=20):
    import plotly.graph_objects as go
    from .radar import court_paths

    fig = go.Figure()
    line = dict(color="#9BAEC2", width=1.5)

    def path(x, y):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=line, hoverinfo="skip", showlegend=False))

    paths, marks = court_paths(length, width)
    for points in paths:
        path(points[:, 0], points[:, 1])
    fig.add_trace(go.Scatter(x=marks[:, 0], y=marks[:, 1], mode="markers",
                            marker=dict(size=4, color="#9BAEC2"), showlegend=False, hoverinfo="skip"))
    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="#101b2a",
        plot_bgcolor="#162b35",
        xaxis=dict(range=[-1, length + 1], title="Length (m)"),
        yaxis=dict(range=[-1, width + 1], scaleanchor="x", scaleratio=1, title="Width (m)"),
        showlegend=False,
    )
    return fig
