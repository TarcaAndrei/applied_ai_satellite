"""EuroSAT MS: 27,000 Sentinel-2 L1C patches, 64x64 px, 13 bands, 10 classes.

Typical use::

    from sat.eurosat import get_dataloaders
    loaders = get_dataloaders(batch_size=128, bands=L2A_BANDS)
    x, y = next(iter(loaders["train"]))   # x: (B, 12, 64, 64) float32, y: (B,) int64

The first call does three things once and saves the results:
  * splits/eurosat_ms_seed42.csv: a stratified 70/15/15 train/val/test split (committed to git,
    so everyone uses the same split);
  * dataset/cache/eurosat_ms_uint16.npy: every patch in one (27000, 13, 64, 64) uint16 array
    (2.9 GB). Reading it as a memory map is much faster than opening 27k small tifs over NFS;
  * splits/eurosat_ms_band_stats_seed42.json: per-band mean/std computed on the train split only.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import DataLoader, Dataset

from sat.bands import BANDS, band_indices

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "dataset" / "EuroSAT_MS"
CACHE_DIR = REPO_ROOT / "dataset" / "cache"
SPLITS_DIR = REPO_ROOT / "splits"

CLASSES: tuple[str, ...] = (
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
PATCH_SIZE = 64
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- index & splits
def build_index(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """One row per patch: path (relative to root), class_name, label. Sorted, so the order is fixed."""
    root = Path(root)
    rows = []
    for cls in CLASSES:
        files = sorted((root / cls).glob("*.tif"), key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if not files:
            raise FileNotFoundError(f"No .tif files in {root / cls}. Is the dataset extracted to {root}?")
        rows += [(f"{cls}/{f.name}", cls, CLASS_TO_IDX[cls]) for f in files]
    return pd.DataFrame(rows, columns=["path", "class_name", "label"])


def split_path(seed: int = 42) -> Path:
    return SPLITS_DIR / f"eurosat_ms_seed{seed}.csv"


def make_split(root: Path = DEFAULT_ROOT, seed: int = 42, val_frac: float = 0.15, test_frac: float = 0.15) -> pd.DataFrame:
    """Stratified random split per class, written to splits/. Returns the index with a `split` column.

    Note: EuroSAT patches come from contiguous scenes, so a random split puts neighbouring
    patches in train and test. Test scores are therefore somewhat optimistic.
    """
    df = build_index(root)
    rng = np.random.default_rng(seed)
    df["split"] = "train"
    for _, group in df.groupby("label"):
        idx = rng.permutation(group.index.to_numpy())
        n_test = round(len(idx) * test_frac)
        n_val = round(len(idx) * val_frac)
        df.loc[idx[:n_test], "split"] = "test"
        df.loc[idx[n_test:n_test + n_val], "split"] = "val"
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(split_path(seed), index=False)
    return df


def load_split(root: Path = DEFAULT_ROOT, seed: int = 42) -> pd.DataFrame:
    """The split table (creating it on first use). Row order = cache order."""
    path = split_path(seed)
    if not path.exists():
        print(f"[eurosat] creating split {path}")
        return make_split(root, seed)
    df = pd.read_csv(path)
    expected = build_index(root)
    if not df["path"].equals(expected["path"]):
        raise RuntimeError(f"{path} does not match the files in {root}; delete it to regenerate.")
    return df


# --------------------------------------------------------------------------- uint16 cache
def cache_path() -> Path:
    return CACHE_DIR / "eurosat_ms_uint16.npy"


def read_tif(path: Path) -> np.ndarray:
    """A EuroSAT MS tif as a (13, 64, 64) uint16 array."""
    arr = tifffile.imread(path)  # stored as (H, W, C)
    if arr.shape != (PATCH_SIZE, PATCH_SIZE, len(BANDS)):
        raise ValueError(f"{path}: unexpected shape {arr.shape}")
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


def build_cache(root: Path = DEFAULT_ROOT, workers: int = 16) -> Path:
    """Write every patch, in build_index() order, into one .npy file (written to a .tmp file, then renamed)."""
    root = Path(root)
    paths = build_index(root)["path"].tolist()
    out = cache_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint16,
                                    shape=(len(paths), len(BANDS), PATCH_SIZE, PATCH_SIZE))

    def load(i: int) -> None:
        arr[i] = read_tif(root / paths[i])

    print(f"[eurosat] building cache {out} from {len(paths)} tifs ...")
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(load, range(len(paths))))
    arr.flush()
    del arr
    tmp.rename(out)
    (out.parent / "eurosat_ms_paths.txt").write_text("\n".join(paths) + "\n")
    return out


def load_cache(root: Path = DEFAULT_ROOT) -> np.ndarray:
    """Read-only memory map of shape (27000, 13, 64, 64), uint16, in build_index() order."""
    if not cache_path().exists():
        build_cache(root)
    return np.load(cache_path(), mmap_mode="r")


# --------------------------------------------------------------------------- normalisation stats
def stats_path(seed: int = 42) -> Path:
    return SPLITS_DIR / f"eurosat_ms_band_stats_seed{seed}.json"


def band_stats(root: Path = DEFAULT_ROOT, seed: int = 42) -> dict[str, dict[str, float]]:
    """Per-band mean and std of raw DNs over the train split, cached as JSON: {band: {mean, std}}."""
    path = stats_path(seed)
    if path.exists():
        return json.loads(path.read_text())
    split = load_split(root, seed)
    arr = load_cache(root)
    train_idx = np.flatnonzero(split["split"].to_numpy() == "train")
    total = np.zeros(len(BANDS))
    total_sq = np.zeros(len(BANDS))
    n = 0
    for start in range(0, len(train_idx), 1024):
        chunk = arr[train_idx[start:start + 1024]].astype(np.float64)
        total += chunk.sum(axis=(0, 2, 3))
        total_sq += (chunk ** 2).sum(axis=(0, 2, 3))
        n += chunk.shape[0] * PATCH_SIZE * PATCH_SIZE
    mean = total / n
    std = np.sqrt(total_sq / n - mean ** 2)
    stats = {b: {"mean": float(m), "std": float(s)} for b, m, s in zip(BANDS, mean, std)}
    path.write_text(json.dumps(stats, indent=2) + "\n")
    return stats


# --------------------------------------------------------------------------- dataset
class EuroSATMS(Dataset):
    """EuroSAT MS split as a torch Dataset.

    Args:
        split: "train", "val", "test" or "all".
        bands: band names to keep, in this order (e.g. sat.bands.L2A_BANDS for the 12 L2A bands).
        normalize: standardise each band with train-split mean/std. If False, x is raw DN as float32.
        transform: applied to the (C, 64, 64) float tensor after normalisation (e.g. RandomD4()).
        use_cache: read from the .npy memory map (built on first use) instead of the tifs.

    Returns (x, y): x float32 (C, 64, 64), y int.
    """

    def __init__(
        self,
        split: str = "train",
        root: Path = DEFAULT_ROOT,
        bands: Sequence[str] = BANDS,
        normalize: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        use_cache: bool = True,
        seed: int = 42,
    ) -> None:
        if split not in (*SPLITS, "all"):
            raise ValueError(f"split must be one of {SPLITS} or 'all', got {split!r}")
        self.root = Path(root)
        self.bands = tuple(bands)
        self.band_idx = band_indices(self.bands)
        self.transform = transform
        self.use_cache = use_cache

        table = load_split(self.root, seed)
        mask = np.ones(len(table), bool) if split == "all" else (table["split"] == split).to_numpy()
        self.rows = np.flatnonzero(mask)          # positions in the cache
        self.table = table.iloc[self.rows].reset_index(drop=True)
        self.labels = self.table["label"].to_numpy()

        if use_cache and not cache_path().exists():
            build_cache(self.root)
        self._arr: np.ndarray | None = None       # opened lazily, once per DataLoader worker

        if normalize:
            stats = band_stats(self.root, seed)
            self.mean = torch.tensor([stats[b]["mean"] for b in self.bands]).view(-1, 1, 1)
            self.std = torch.tensor([stats[b]["std"] for b in self.bands]).view(-1, 1, 1)
        else:
            self.mean = self.std = None

    def __len__(self) -> int:
        return len(self.rows)

    def raw(self, i: int) -> np.ndarray:
        """Patch i as raw uint16 DN, selected bands only: (C, 64, 64)."""
        if self.use_cache:
            if self._arr is None:
                self._arr = np.load(cache_path(), mmap_mode="r")
            img = self._arr[self.rows[i]]
        else:
            img = read_tif(self.root / self.table.at[i, "path"])
        return np.asarray(img[self.band_idx])

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        x = torch.from_numpy(self.raw(i).astype(np.float32))
        if self.mean is not None:
            x = (x - self.mean) / self.std
        if self.transform is not None:
            x = self.transform(x)
        return x, int(self.labels[i])

    def class_counts(self) -> dict[str, int]:
        counts = np.bincount(self.labels, minlength=len(CLASSES))
        return dict(zip(CLASSES, counts.tolist()))


class RandomD4:
    """Random flip + 90-degree rotation (the 8 symmetries of a square).
    Satellite patches have no canonical "up", so all eight are valid augmentations."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        k = int(torch.randint(4, ()).item())
        if k:
            x = torch.rot90(x, k, dims=(-2, -1))
        if torch.rand(()) < 0.5:
            x = torch.flip(x, dims=(-1,))
        return x


def get_dataloaders(
    batch_size: int = 128,
    bands: Sequence[str] = BANDS,
    augment: bool = True,
    num_workers: int = 8,
    root: Path = DEFAULT_ROOT,
    seed: int = 42,
    **dataset_kwargs,
) -> dict[str, DataLoader]:
    """Train/val/test loaders. Only train is shuffled and augmented."""
    loaders = {}
    for split in SPLITS:
        ds = EuroSATMS(split, root=root, bands=bands, seed=seed,
                       transform=RandomD4() if (augment and split == "train") else None, **dataset_kwargs)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=split == "train", num_workers=num_workers,
            pin_memory=torch.cuda.is_available(), persistent_workers=num_workers > 0,
            drop_last=False,
        )
    return loaders


if __name__ == "__main__":
    # python -m sat.eurosat  -> builds split, cache and stats, then prints a summary.
    loaders = get_dataloaders(num_workers=4)
    for name, loader in loaders.items():
        print(f"{name:5s} {len(loader.dataset):6d} patches  {loader.dataset.class_counts()}")
    x, y = next(iter(loaders["train"]))
    print("batch", tuple(x.shape), x.dtype, "labels", y[:8].tolist())
    print("per-band batch mean (should be ~0):", x.mean(dim=(0, 2, 3)).round(decimals=2).tolist())
