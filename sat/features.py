"""Hand-crafted per-patch features for the tabular baselines (XGBoost, logistic regression).

`patch_features` works on any raw-DN array (N, C, H, W) whose channels are named by
`source_bands`, so the same function can later be run on Cluj patches. It only uses the
12 bands that Sentinel-2 L2A also has (no B10).

Feature groups (names are returned alongside the matrix):
  * band:   per band mean, std, p10, p50, p90 of reflectance         (12 x 5 = 60)
  * index:  per spectral index mean, std, p10, p90                    (5 x 4 = 20)
  * texture: mean gradient magnitude of B08 and of the RGB brightness (2)
All features are invariant to flips and 90 degree rotations.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from sat.bands import BANDS, L2A_BANDS, REFLECTANCE_SCALE

# Normalised-difference indices: name -> (a, b), index = (a - b) / (a + b).
INDICES: dict[str, tuple[str, str]] = {
    "NDVI": ("B08", "B04"),   # vegetation
    "NDWI": ("B03", "B08"),   # open water (McFeeters)
    "MNDWI": ("B03", "B11"),  # water, also separates built-up
    "NDBI": ("B11", "B08"),   # built-up
    "NDRE": ("B8A", "B05"),   # red edge: crop vs pasture vs forest
}
FEATURE_BANDS: tuple[str, ...] = L2A_BANDS
BAND_STATS = ("mean", "std", "p10", "p50", "p90")
INDEX_STATS = ("mean", "std", "p10", "p90")


def feature_names(bands: Sequence[str] = FEATURE_BANDS) -> list[str]:
    names = [f"{b}_{s}" for b in bands for s in BAND_STATS]
    names += [f"{i}_{s}" for i in INDICES for s in INDEX_STATS]
    names += ["grad_B08", "grad_RGB"]
    return names


def _stats(x: np.ndarray, kinds: Sequence[str]) -> np.ndarray:
    """x: (N, K, P) -> (N, K, len(kinds))."""
    out = []
    pct = {"p10": 10, "p50": 50, "p90": 90}
    q = [pct[k] for k in kinds if k in pct]
    qv = dict(zip([k for k in kinds if k in pct], np.percentile(x, q, axis=-1))) if q else {}
    for k in kinds:
        if k == "mean":
            out.append(x.mean(-1))
        elif k == "std":
            out.append(x.std(-1))
        else:
            out.append(qv[k])
    return np.stack(out, -1)


def _grad_mag(img: np.ndarray) -> np.ndarray:
    """Mean gradient magnitude of (N, H, W) images (central differences, interior pixels)."""
    gy = img[:, 2:, 1:-1] - img[:, :-2, 1:-1]
    gx = img[:, 1:-1, 2:] - img[:, 1:-1, :-2]
    return np.sqrt(gx ** 2 + gy ** 2).mean((1, 2)) / 2


def patch_features(
    images: np.ndarray,
    source_bands: Sequence[str] = BANDS,
    bands: Sequence[str] = FEATURE_BANDS,
) -> np.ndarray:
    """(N, C, H, W) raw DNs -> (N, F) float32 features, in the order of `feature_names(bands)`."""
    src = list(source_bands)
    refl = images[:, [src.index(b) for b in bands]].astype(np.float32) / REFLECTANCE_SCALE
    n = refl.shape[0]
    flat = refl.reshape(n, len(bands), -1)
    band_f = _stats(flat, BAND_STATS).reshape(n, -1)

    bi = {b: i for i, b in enumerate(bands)}
    idx = np.stack([(flat[:, bi[a]] - flat[:, bi[b]]) / np.maximum(flat[:, bi[a]] + flat[:, bi[b]], 1e-6)
                    for a, b in INDICES.values()], 1)
    index_f = _stats(idx, INDEX_STATS).reshape(n, -1)

    rgb = refl[:, [bi["B04"], bi["B03"], bi["B02"]]].mean(1)
    tex_f = np.stack([_grad_mag(refl[:, bi["B08"]]), _grad_mag(rgb)], 1)
    return np.concatenate([band_f, index_f, tex_f], 1).astype(np.float32)


def compute_features(images: np.ndarray, chunk: int = 1000, **kw) -> np.ndarray:
    """`patch_features` in chunks, so a memory-mapped (27000, 13, 64, 64) array fits in RAM."""
    return np.concatenate([patch_features(np.asarray(images[i:i + chunk]), **kw)
                           for i in range(0, len(images), chunk)])
