"""Sentinel-2 band definitions and display helpers shared by EuroSAT and the Cluj imagery.

Keep every band-order assumption and every RGB conversion here, so EuroSAT patches and
Cluj patches go through exactly the same code.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

# Band order of the EuroSAT MS GeoTIFFs (13 bands, all resampled to 10 m).
# NOTE: B8A is stored LAST, not between B08 and B09. Verified from the data: channel 12
# correlates 1.00 with B08 (so it is narrow NIR), channel 9 is ~10 DN everywhere (cirrus, B10),
# and over forest channel 10 > channel 11 (SWIR1 > SWIR2).
BANDS: tuple[str, ...] = (
    "B01", "B02", "B03", "B04", "B05", "B06", "B07",
    "B08", "B09", "B10", "B11", "B12", "B8A",
)
# Central wavelengths (nm), Sentinel-2A, in the same order as BANDS.
WAVELENGTHS_NM: tuple[int, ...] = (443, 490, 560, 665, 705, 740, 783, 842, 945, 1375, 1610, 2190, 865)
# Bands sorted by wavelength, for plotting spectra.
BANDS_BY_WAVELENGTH: tuple[str, ...] = tuple(b for _, b in sorted(zip(WAVELENGTHS_NM, BANDS)))
BAND_DESCRIPTIONS: dict[str, str] = {
    "B01": "coastal aerosol", "B02": "blue", "B03": "green", "B04": "red",
    "B05": "red edge 1", "B06": "red edge 2", "B07": "red edge 3", "B08": "NIR",
    "B8A": "narrow NIR", "B09": "water vapour", "B10": "cirrus", "B11": "SWIR 1", "B12": "SWIR 2",
}

# Sentinel-2 L2A (what Earth Search serves for Cluj) has no B10, so a model meant to
# run on Cluj should be trained on these 12 bands.
L2A_BANDS: tuple[str, ...] = tuple(b for b in BANDS if b != "B10")
RGB_BANDS: tuple[str, ...] = ("B04", "B03", "B02")
FALSE_COLOR_BANDS: tuple[str, ...] = ("B08", "B04", "B03")  # vegetation shows as red

# Digital numbers are reflectance x 10000.
REFLECTANCE_SCALE = 10_000.0
# Reflectance mapped to full brightness in RGB composites (keeps vegetation and fields visible).
RGB_MAX_REFLECTANCE = 0.3


def band_indices(bands: Sequence[str]) -> list[int]:
    """Positions of `bands` in the 13-band EuroSAT order."""
    unknown = [b for b in bands if b not in BANDS]
    if unknown:
        raise ValueError(f"Unknown band(s) {unknown}; expected names from {BANDS}")
    return [BANDS.index(b) for b in bands]


def composite(
    image: np.ndarray,
    bands: Sequence[str] = RGB_BANDS,
    max_reflectance: float = RGB_MAX_REFLECTANCE,
    source_bands: Sequence[str] = BANDS,
) -> np.ndarray:
    """3-band composite in [0, 1], shape (H, W, 3).

    `image` is a raw-DN array (C, H, W) whose channels are `source_bands`.
    A fixed reflectance ceiling (not a per-image stretch) is used, so brightness is
    comparable across patches, datasets and years.
    """
    idx = [list(source_bands).index(b) for b in bands]
    rgb = image[idx].astype(np.float32) / REFLECTANCE_SCALE / max_reflectance
    return np.clip(rgb, 0.0, 1.0).transpose(1, 2, 0)


def normalized_difference(image: np.ndarray, a: str, b: str, source_bands: Sequence[str] = BANDS) -> np.ndarray:
    """(a - b) / (a + b) per pixel, e.g. NDVI = nd(B08, B04)."""
    src = list(source_bands)
    x = image[src.index(a)].astype(np.float32)
    y = image[src.index(b)].astype(np.float32)
    return (x - y) / np.maximum(x + y, 1e-6)


def ndvi(image: np.ndarray, source_bands: Sequence[str] = BANDS) -> np.ndarray:
    return normalized_difference(image, "B08", "B04", source_bands)
