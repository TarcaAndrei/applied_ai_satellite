"""Checks for the EuroSAT MS dataset code. Run from the repo root: python -m pytest tests -q"""
import numpy as np
import pytest
import torch

from sat.bands import BANDS, L2A_BANDS, band_indices
from sat.eurosat import CLASSES, DEFAULT_ROOT, EuroSATMS, RandomD4, load_cache, load_split, read_tif

pytestmark = pytest.mark.skipif(not DEFAULT_ROOT.exists(), reason="EuroSAT MS not extracted to dataset/")


def test_split_is_complete_disjoint_and_stratified():
    df = load_split()
    assert len(df) == 27_000 and df["path"].is_unique
    assert set(df["split"]) == {"train", "val", "test"}
    frac = df.groupby("class_name")["split"].value_counts(normalize=True).unstack()
    assert np.allclose(frac["train"], 0.70, atol=0.01)
    assert np.allclose(frac["test"], 0.15, atol=0.01)


def test_cache_matches_tifs():
    df, arr = load_split(), load_cache()
    for i in (0, 12_345, 26_999):
        np.testing.assert_array_equal(arr[i], read_tif(DEFAULT_ROOT / df.at[i, "path"]))


def test_band_order_b8a_is_last():
    """B8A is stored last; its patch means track B08 almost perfectly, and B10 (cirrus) is near zero."""
    arr = load_cache()
    means = arr[::50].astype(np.float64).mean(axis=(2, 3))
    assert np.corrcoef(means[:, BANDS.index("B08")], means[:, BANDS.index("B8A")])[0, 1] > 0.99
    assert means[:, BANDS.index("B10")].mean() < 50


def test_dataset_item_and_band_selection():
    ds = EuroSATMS("val", bands=L2A_BANDS)
    x, y = ds[0]
    assert x.shape == (12, 64, 64) and x.dtype == torch.float32
    assert 0 <= y < len(CLASSES)
    raw = EuroSATMS("val", normalize=False)[0][0]
    torch.testing.assert_close(raw[band_indices(L2A_BANDS)], EuroSATMS("val", bands=L2A_BANDS, normalize=False)[0][0])


def test_no_cache_path_gives_same_values():
    a = EuroSATMS("test", use_cache=True)[7][0]
    b = EuroSATMS("test", use_cache=False)[7][0]
    torch.testing.assert_close(a, b)


def test_random_d4_preserves_pixels():
    x = torch.arange(2 * 4 * 4, dtype=torch.float32).view(2, 4, 4)
    for _ in range(20):
        y = RandomD4()(x)
        assert y.shape == x.shape
        torch.testing.assert_close(y.flatten().sort().values, x.flatten().sort().values)
