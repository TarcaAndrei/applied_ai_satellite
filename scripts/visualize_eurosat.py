"""EDA figures for EuroSAT MS, written to figures/eurosat/.

    python -m scripts.visualize_eurosat            # from the repo root
    python -m scripts.visualize_eurosat --n-per-class 6 --seed 1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from sat.bands import (BAND_DESCRIPTIONS, BANDS, BANDS_BY_WAVELENGTH, FALSE_COLOR_BANDS, RGB_BANDS,
                       WAVELENGTHS_NM, composite, ndvi)
from sat.eurosat import CLASSES, REPO_ROOT, load_cache, load_split

OUT_DIR = REPO_ROOT / "figures" / "eurosat"

# Colours from the dataviz reference palette (light mode).
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b9b8b2"
GRID = "#e6e5e0"
ACCENT = "#2a78d6"        # single-series colour
ACCENT_FILL = "#b7d3f6"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", ["#1c5cab", "#86b6ef", "#f0efec", "#ec9a99", "#b8312f"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.edgecolor": GRID, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO_ROOT)}")


def sample_rows(labels: np.ndarray, n: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [np.sort(rng.choice(np.flatnonzero(labels == c), n, replace=False)) for c in range(len(CLASSES))]


# --------------------------------------------------------------------------- figures
def class_counts(labels: np.ndarray) -> None:
    counts = np.bincount(labels, minlength=len(CLASSES))
    order = np.argsort(counts)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(np.arange(len(CLASSES)), counts[order], height=0.6, color=ACCENT)
    ax.set_yticks(np.arange(len(CLASSES)), [CLASSES[i] for i in order])
    for y, v in enumerate(counts[order]):
        ax.text(v + 30, y, f"{v:,}", va="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, counts.max() * 1.12)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("patches")
    ax.set_title(f"EuroSAT MS: {len(labels):,} patches in {len(CLASSES)} classes")
    save(fig, "01_class_counts.png")


def sample_grid(arr: np.ndarray, picks: list[np.ndarray], bands: tuple[str, ...], title: str, name: str) -> None:
    n = len(picks[0])
    fig, axes = plt.subplots(len(CLASSES), n, figsize=(1.15 * n + 1.6, 1.15 * len(CLASSES) + 0.4), layout="compressed")
    for r, rows in enumerate(picks):
        for c, i in enumerate(rows):
            ax = axes[r, c]
            ax.imshow(composite(arr[i], bands), interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[r, 0].set_ylabel(CLASSES[r], rotation=0, ha="right", va="center", color=INK, fontsize=9)
    fig.suptitle(title, x=0.02, ha="left", fontweight="bold")
    save(fig, name)


def bands_per_class(arr: np.ndarray, picks: list[np.ndarray]) -> None:
    """One patch per class, every band, in wavelength order. Each band is stretched by its own
    2-98th percentile over the shown patches, so brightness compares classes within a column."""
    order = [BANDS.index(b) for b in BANDS_BY_WAVELENGTH]
    imgs = np.stack([arr[rows[0]] for rows in picks]).astype(np.float32)[:, order]  # (10, 13, 64, 64)
    lo = np.percentile(imgs, 2, axis=(0, 2, 3), keepdims=True)
    hi = np.percentile(imgs, 98, axis=(0, 2, 3), keepdims=True)
    imgs = np.clip((imgs - lo) / np.maximum(hi - lo, 1), 0, 1)
    fig, axes = plt.subplots(len(CLASSES), len(order) + 1, figsize=(15, 11), layout="compressed")
    for r in range(len(CLASSES)):
        axes[r, 0].imshow(composite(arr[picks[r][0]], RGB_BANDS), interpolation="nearest")
        axes[r, 0].set_ylabel(CLASSES[r], rotation=0, ha="right", va="center", color=INK, fontsize=9)
        for c in range(len(order)):
            axes[r, c + 1].imshow(imgs[r, c], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0, 0].set_title("RGB", fontsize=9, loc="center", fontweight="normal")
    for c, b in enumerate(BANDS_BY_WAVELENGTH):
        axes[0, c + 1].set_title(f"{b}\n{BAND_DESCRIPTIONS[b]}", fontsize=8, loc="center", fontweight="normal")
    fig.suptitle("All 13 bands of one patch per class (wavelength order; per-band contrast stretch)",
                 x=0.02, ha="left", fontweight="bold")
    save(fig, "04_bands_per_class.png")


def spectral_signatures(patch_means: np.ndarray, labels: np.ndarray) -> None:
    """Small multiples: each class's median spectrum (IQR shaded), other classes in gray behind it."""
    order = [BANDS.index(b) for b in BANDS_BY_WAVELENGTH]
    wl = np.array(WAVELENGTHS_NM)[order]
    refl = patch_means[:, order] / 10_000
    med = np.stack([np.median(refl[labels == c], axis=0) for c in range(len(CLASSES))])
    q25 = np.stack([np.percentile(refl[labels == c], 25, axis=0) for c in range(len(CLASSES))])
    q75 = np.stack([np.percentile(refl[labels == c], 75, axis=0) for c in range(len(CLASSES))])
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharex=True, sharey=True)
    for c, ax in enumerate(axes.flat):
        for other in range(len(CLASSES)):
            if other != c:
                ax.plot(wl, med[other], color=MUTED, lw=1, alpha=0.8)
        ax.fill_between(wl, q25[c], q75[c], color=ACCENT_FILL, lw=0)
        ax.plot(wl, med[c], color=ACCENT, lw=2, marker="o", ms=3.5)
        ax.set_title(CLASSES[c], fontsize=10)
        ax.yaxis.grid(True); ax.set_axisbelow(True)
        ax.set_xscale("log")
        ax.set_xticks([500, 700, 1000, 1500, 2000], ["500", "700", "1000", "1500", "2000"])
        ax.minorticks_off()
    for ax in axes[1]:
        ax.set_xlabel("wavelength (nm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("TOA reflectance")
    fig.suptitle("Spectral signature per class: median patch spectrum, IQR shaded, other classes in gray\n"
                 "(the dip at 1375 nm is B10 cirrus, which is absorbed by the atmosphere and absent from L2A)",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "05_spectral_signatures.png")


def ndvi_by_class(arr: np.ndarray, labels: np.ndarray, rng: np.random.Generator, per_class: int = 800) -> None:
    picks = sample_rows(labels, per_class, rng)
    values = [np.array([ndvi(arr[i]).mean() for i in rows]) for rows in picks]
    order = np.argsort([np.median(v) for v in values])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot([values[i] for i in order], orientation="horizontal", widths=0.55, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK_2), capprops=dict(color=INK_2))
    for box in bp["boxes"]:
        box.set(facecolor=ACCENT_FILL, edgecolor=ACCENT, lw=1.2)
    ax.set_yticks(np.arange(1, len(CLASSES) + 1), [CLASSES[i] for i in order])
    ax.axvline(0, color=MUTED, lw=1)
    ax.xaxis.grid(True); ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("mean NDVI of patch = (B08 − B04) / (B08 + B04)")
    ax.set_title(f"NDVI by class ({per_class} random patches per class; whiskers 1.5×IQR)")
    save(fig, "06_ndvi_by_class.png")


def band_correlation(patch_means: np.ndarray) -> None:
    order = [BANDS.index(b) for b in BANDS_BY_WAVELENGTH]
    corr = np.corrcoef(patch_means[:, order].T)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(order)), BANDS_BY_WAVELENGTH, rotation=90)
    ax.set_yticks(range(len(order)), BANDS_BY_WAVELENGTH)
    for i in range(len(order)):
        for j in range(len(order)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(corr[i, j]) > 0.75 else INK)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="Pearson r")
    ax.set_title("Correlation of per-patch band means (all 27,000 patches)")
    save(fig, "07_band_correlation.png")


# --------------------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-per-class", type=int, default=8, help="patches per class in the image grids")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    labels = load_split()["label"].to_numpy()
    arr = load_cache()

    print("computing per-patch band means ...")
    patch_means = np.concatenate([arr[s:s + 2000].mean(axis=(2, 3)) for s in range(0, len(arr), 2000)])

    picks = sample_rows(labels, args.n_per_class, rng)
    class_counts(labels)
    sample_grid(arr, picks, RGB_BANDS, "True colour (B04, B03, B02), reflectance 0–0.3", "02_samples_rgb.png")
    sample_grid(arr, picks, FALSE_COLOR_BANDS, "False colour (B08 NIR, B04, B03): vegetation appears red",
                "03_samples_false_color.png")
    bands_per_class(arr, picks)
    spectral_signatures(patch_means, labels)
    ndvi_by_class(arr, labels, rng)
    band_correlation(patch_means)


if __name__ == "__main__":
    main()
