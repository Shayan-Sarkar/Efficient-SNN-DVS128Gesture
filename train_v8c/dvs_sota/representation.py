from __future__ import annotations
import numpy as np


def events_to_voxel_grid(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    t: np.ndarray,
    T: int = 20,
    H: int = 64,
    W: int = 64,
    src_hw: int = 128,
    normalize: bool = True,
) -> np.ndarray:
    voxel = np.zeros((2, T, H, W), dtype=np.float32)

    if len(t) == 0:
        return voxel

    scale = 1.0 if src_hw == H else H / src_hw
    xf = np.clip(x.astype(np.float32) * scale, 0, W - 1)
    yf = np.clip(y.astype(np.float32) * scale, 0, H - 1)
    xi = xf.astype(np.int32)
    yi = yf.astype(np.int32)

    t0, t1 = t.min(), t.max()
    if t0 == t1:
        for pol in (0, 1):
            m = p == pol
            np.add.at(voxel[pol, 0], (yi[m], xi[m]), 1.0)
        return voxel

    t_norm = (t - t0).astype(np.float64) / (t1 - t0) * (T - 1)

    t_floor = np.floor(t_norm).astype(np.int32)
    t_ceil  = t_floor + 1
    w_ceil  = (t_norm - t_floor).astype(np.float32)
    w_floor = 1.0 - w_ceil

    for pol in (0, 1):
        m = (p == pol)
        xp, yp = xi[m], yi[m]
        tf, tc  = t_floor[m], t_ceil[m]
        wf, wc  = w_floor[m], w_ceil[m]

        valid_f = tf < T
        np.add.at(voxel[pol], (tf[valid_f], yp[valid_f], xp[valid_f]), wf[valid_f])

        valid_c = tc < T
        np.add.at(voxel[pol], (tc[valid_c], yp[valid_c], xp[valid_c]), wc[valid_c])

    if normalize:
        voxel = _normalize_voxel(voxel)

    return voxel


def _normalize_voxel(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    for c in range(v.shape[0]):
        plane = v[c]
        nz = plane[plane != 0]
        if len(nz) > 1:
            mu, sigma = nz.mean(), nz.std()
            if sigma > 1e-6:
                out[c] = (plane - mu) / sigma
    return out
