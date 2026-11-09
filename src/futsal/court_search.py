"""Global court-pose search for fixed-position broadcast cameras (no neural network).

Given the camera position C (per venue), a frame is explained by three numbers: where the optical axis hits
the court plane (u, v) and the focal length f.  We score a grid of (u, v, f) hypotheses by how many
projected wireframe samples land on detected white lines, then hand the best few to the free 8-DoF
refinement in `Refiner`.
"""
import numpy as np

from .court_camera import look_at


def camera_homographies(C, targets, fs, size, rolls=(0.0,)):
    """Batch of court->image homographies; shape (len(rolls), len(fs), len(targets), 3, 3)."""
    w, h = size
    C = np.asarray(C, float)
    out = np.zeros((len(rolls), len(fs), len(targets), 3, 3))
    base = []
    for roll in rolls:
        Ms = []
        for t in targets:
            R = look_at(C, np.array([t[0], t[1], 0.0]), roll)
            Ms.append(np.column_stack([R[:, 0], R[:, 1], -R @ C]))
        base.append(np.array(Ms))
    for ri in range(len(rolls)):
        for fi, f in enumerate(fs):
            K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
            out[ri, fi] = np.einsum("ij,mjk->mik", K, base[ri])
    return out


def score_homographies(Hs, samples, dt, thr, chunk=400):
    """Vectorised alignment score: matched samples minus half the unmatched visible ones."""
    Hs = Hs.reshape(-1, 3, 3)
    h, w = dt.shape
    P = np.c_[samples, np.ones(len(samples))]  # (N,3)
    scores = np.zeros(len(Hs))
    for s in range(0, len(Hs), chunk):
        H = Hs[s:s + chunk]
        q = np.einsum("bij,nj->bni", H, P)
        z = q[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = q[..., 0] / z
            y = q[..., 1] / z
        ok = (z > 1e-6) & (x >= 3) & (x <= w - 4) & (y >= 3) & (y <= h - 4)
        xi = np.clip(np.nan_to_num(x), 0, w - 1).astype(np.int32)
        yi = np.clip(np.nan_to_num(y), 0, h - 1).astype(np.int32)
        match = dt[yi, xi] <= thr
        scores[s:s + chunk] = (ok & match).sum(1) - 0.5 * (ok & ~match).sum(1)
    return scores


def best_candidates(refiner, C, size, top=4, u_range=(-6, 46, 2.0), v_range=(-4, 24, 2.0),
                    f_range=(450, 3600, 14)):
    w, h = size
    us = np.arange(u_range[0], u_range[1] + 1e-6, u_range[2])
    vs = np.arange(v_range[0], v_range[1] + 1e-6, v_range[2])
    targets = np.array([(u, v) for u in us for v in vs])
    fs = np.geomspace(f_range[0] * w / 960, f_range[1] * w / 960, f_range[2])
    Hs = camera_homographies(C, targets, fs, size)  # (1, F, M, 3, 3)
    sc = score_homographies(Hs, refiner.samples, refiner.dt, 2.5 * refiner.scale).reshape(len(fs), len(targets))
    picked = []
    sc = sc.copy()
    for _ in range(top):
        fi, mi = np.unravel_index(int(np.argmax(sc)), sc.shape)
        if sc[fi, mi] <= 0:
            break
        picked.append((float(sc[fi, mi]), Hs[0, fi, mi].copy(), (targets[mi], fs[fi])))
        # suppress neighbours in (u,v,f)
        tu, tv = targets[mi]
        near = (np.abs(targets[:, 0] - tu) <= 4) & (np.abs(targets[:, 1] - tv) <= 4)
        for dfi in (-2, -1, 0, 1, 2):
            if 0 <= fi + dfi < len(fs):
                sc[fi + dfi, near] = -1e9
    return picked
