"""Appearance retrieval produces review candidates, never automatic roster identities."""

from pathlib import Path
import numpy as np

from .storage import load_tracks, write_json
from .teams import jersey_feature
from .video import frame_at

OSNET_FILENAME = "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth"


def download_osnet(directory="models"):
    import urllib.request
    from .storage import sha256

    target = Path(directory) / "osnet-msmt17.pth"
    target.parent.mkdir(parents=True, exist_ok=True)
    source = "https://huggingface.co/kaiyangzhou/osnet/resolve/main/" + OSNET_FILENAME
    if not target.exists():
        temporary = target.with_suffix(".download")
        urllib.request.urlretrieve(source, temporary)
        temporary.replace(target)
    write_json(
        target.with_suffix(".manifest.json"),
        {
            "source": source,
            "sha256": sha256(target),
            "architecture": "OSNet x0.25",
            "training_domain": "MSMT17, not futsal",
            "source_commit": "f8cd150fdf77e8d9e1ed143b7f308c2c609ded50",
            "futsal_identity_accuracy": "not_evaluated",
        },
    )
    return target


class OSNetEncoder:
    def __init__(self, checkpoint, device="cpu"):
        import torch
        from .vendor.osnet import osnet_x0_25

        self.torch = torch
        self.device = device
        self.model = osnet_x0_25(num_classes=1, pretrained=False)
        weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = weights.get("state_dict", weights)
        state = {
            k.removeprefix("module."): v
            for k, v in state.items()
            if not k.removeprefix("module.").startswith("classifier.")
        }
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected or any(not k.startswith("classifier.") for k in missing):
            raise ValueError(f"Incompatible OSNet checkpoint: {missing}, {unexpected}")
        self.model.eval().to(device)

    def encode(self, crops):
        if not crops:
            return np.empty((0, 512))
        import cv2

        arrays = []
        for crop in crops:
            rgb = cv2.cvtColor(cv2.resize(crop, (128, 256)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            rgb = (rgb - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
                [0.229, 0.224, 0.225], np.float32
            )
            arrays.append(rgb.transpose(2, 0, 1))
        tensor = self.torch.from_numpy(np.stack(arrays)).to(self.device)
        with self.torch.inference_mode():
            output = self.model(tensor).cpu().numpy()
        return output / np.maximum(np.linalg.norm(output, axis=1, keepdims=True), 1e-12)


def osnet_candidates(run, checkpoint="models/osnet-msmt17.pth", device="cpu", max_tracks=150):
    from .storage import read_json, sha256

    run = Path(run)
    data = load_tracks(run)
    manifest = read_json(run / "manifest.json")
    source = manifest["asset"]["path"]
    encoder = OSNetEncoder(checkpoint, device)
    descriptors = []
    groups = list(data[(data.class_name == "person") & data.role.ne("referee")].groupby("tracklet_id"))[
        :max_tracks
    ]
    for tracklet, part in groups:
        part = part.sort_values("score", ascending=False)
        crops = []
        for row in part.iloc[:3].itertuples():
            image = frame_at(source, float(row.time_s))
            x1, y1, x2, y2 = map(int, [row.x1, row.y1, row.x2, row.y2])
            crop = image[max(0, y1) : min(image.shape[0], y2), max(0, x1) : min(image.shape[1], x2)]
            if crop.size and crop.shape[0] >= 40 and crop.shape[1] >= 12:
                crops.append(crop)
        if not crops:
            continue
        vector = encoder.encode(crops).mean(0)
        vector /= max(np.linalg.norm(vector), 1e-12)
        descriptors.append(
            {
                "tracklet": tracklet,
                "start": float(part.time_s.min()),
                "end": float(part.time_s.max()),
                "shot": int(part.shot_id.iloc[0]),
                "team": part.team.mode().iloc[0],
                "embedding": vector,
            }
        )
    candidates = []
    for a in descriptors:
        for b in descriptors:
            if a["shot"] == b["shot"] or not 0 <= b["start"] - a["end"] <= 20:
                continue
            if a["team"] != "unknown" and b["team"] != "unknown" and a["team"] != b["team"]:
                continue
            candidates.append(
                {
                    "source": a["tracklet"],
                    "target": b["tracklet"],
                    "cosine_similarity": float(a["embedding"] @ b["embedding"]),
                    "status": "review_required",
                    "model_sha256": sha256(checkpoint),
                }
            )
    candidates = sorted(candidates, key=lambda r: -r["cosine_similarity"])[:200]
    write_json(run / "identity_candidates.json", candidates)
    return candidates


def propose_links(run):
    from .storage import read_json

    run = Path(run)
    data = load_tracks(run)
    manifest = read_json(run / "manifest.json")
    source = manifest["asset"]["path"]
    descriptors = []
    for tracklet, part in data[data.class_name == "person"].groupby("tracklet_id"):
        part = part.sort_values("time_s")
        sample = part.iloc[len(part) // 2]
        image = frame_at(source, float(sample.time_s))
        feature = jersey_feature(image, [sample.x1, sample.y1, sample.x2, sample.y2])
        if feature is not None:
            descriptors.append(
                {
                    "tracklet": tracklet,
                    "start": float(part.time_s.min()),
                    "end": float(part.time_s.max()),
                    "shot": int(sample.shot_id),
                    "feature": feature,
                    "team": part.team.mode().iloc[0],
                }
            )
    links = []
    for i, a in enumerate(descriptors):
        for b in descriptors[i + 1 :]:
            if a["shot"] == b["shot"] or a["team"] != b["team"] or a["team"] == "unknown":
                continue
            earlier, later = (a, b) if a["end"] <= b["start"] else (b, a)
            if not 0 <= later["start"] - earlier["end"] <= 15:
                continue
            links.append(
                {
                    "source": earlier["tracklet"],
                    "target": later["tracklet"],
                    "appearance_distance": float(np.linalg.norm(a["feature"] - b["feature"])),
                    "status": "review_only_color_similarity_not_identity_evidence",
                }
            )
    links = sorted(links, key=lambda r: r["appearance_distance"])[:200]
    write_json(run / "identity_candidates.json", links)
    return links


def validate_merge(data, tracklets):
    selected = data[data.tracklet_id.isin(tracklets)]
    if selected.duplicated("time_s").any():
        raise ValueError("Cannot merge simultaneous tracklets into one player")
    return True
