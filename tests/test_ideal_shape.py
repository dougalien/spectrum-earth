"""Checks that the ideal-shape fit recovers the perfect shapes it is built on.

Run:  py -m pytest tests
"""
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ideal_shape as I  # noqa: E402


def two_sided(g_fn, half_width=1000.0, height=200.0, left_share=0.5, n=1201, kind=I.HILL):
    """A profile built from a half-profile shape, with flat ground beyond both feet."""
    x = np.linspace(-1.5 * half_width, 1.5 * half_width, n)
    wl = 2.0 * half_width * left_share
    wr = 2.0 * half_width - wl
    xa = -half_width + wl
    u = np.where(x <= xa, (xa - x) / wl, (x - xa) / wr)
    g = np.where(u <= 1.0, g_fn(np.clip(u, 0.0, 1.0)), 0.0)
    z = height * g
    return x, (height - z if kind == I.VALLEY else z)


@pytest.mark.parametrize("key", [k for k in I.PERFECT_SHAPES if k != "circle"])
@pytest.mark.parametrize("kind", [I.HILL, I.VALLEY])
def test_each_perfect_shape_finds_itself(key, kind):
    s = I.PERFECT_SHAPES[key]
    x, z = two_sided(lambda u: I.ideal_curve(u, s["p"], s["q"]), kind=kind)
    item = I.measure(x, z, kind)
    assert item["ok"], item["problem"]
    assert item["nearest"] == key
    assert item["departure"] < 0.02


def test_true_circle_is_recovered_on_a_steep_feature():
    # Half-width 1000, height 600: the circle in true proportions is far from the parabola here.
    a = 0.6
    x, z = two_sided(lambda u: I.true_circle(u, a), height=600.0)
    item = I.measure(x, z, I.HILL)
    assert item["ok"]
    assert item["comparisons"]["circle"]["departure"] < 0.01
    assert item["comparisons"]["parabola"]["departure"] > 0.02


def test_true_circle_passes_through_apex_and_foot():
    for a in (0.05, 0.2, 0.5, 0.9):
        g = I.true_circle(I.U, a)
        assert g[0] == pytest.approx(1.0)
        assert g[-1] == pytest.approx(0.0, abs=1e-9)


def test_asymmetry_is_kept_not_averaged():
    x, z = two_sided(lambda u: 1.0 - u, left_share=0.25)
    item = I.measure(x, z, I.HILL)
    assert item["ok"]
    assert item["nearest"] == "straight"
    left = item["m"]["x_apex"] - item["m"]["x_left"]
    right = item["m"]["x_right"] - item["m"]["x_apex"]
    # The apex of a sharp, lopsided cone is located to within a few percent of
    # the width (smoothing rounds it toward the gentler side); the point is that
    # each half keeps its own width instead of being averaged to 1:1.
    assert left / right == pytest.approx(1.0 / 3.0, rel=0.10)


def test_rock_sign_is_the_same_for_hills_and_valleys():
    # A cone and a V, each with extra rock added near the foot.
    def bumped(u):
        return np.clip(1.0 - u + 0.15 * np.exp(-((u - 0.8) / 0.08) ** 2), 0.0, None)

    hill = I.measure(*two_sided(bumped), I.HILL)
    # For a valley, extra rock near the rim is LESS depth there.
    valley = I.measure(*two_sided(lambda u: np.clip(1.0 - u - 0.15 * np.exp(-((u - 0.8) / 0.08) ** 2),
                                                    0.0, None), kind=I.VALLEY), I.VALLEY)
    assert hill["comparisons"]["straight"]["net"] > 0
    assert valley["comparisons"]["straight"]["net"] > 0
    assert hill["comparisons"]["straight"]["zones"]["foot"] > 0
    assert valley["comparisons"]["straight"]["zones"]["rim"] > 0


def test_flat_floor_is_reported_and_read_as_extra_excavation():
    x, z = two_sided(lambda u: np.minimum(1.0 - u, 0.7) / 0.7,
                     kind=I.VALLEY)
    item = I.measure(x, z, I.VALLEY)
    assert item["ok"]
    assert "flat floor" in item["name"]
    # Pinned at the same floor and rim as a plain V, a flat floor holds LESS
    # rock than the V does there: the trough is excavated wider at the bottom.
    assert item["comparisons"]["straight"]["zones"]["floor"] < -0.05


def test_long_profile_is_one_half_and_direction_does_not_matter():
    x = np.linspace(0.0, 20000.0, 2001)
    z = 1500.0 * I.ideal_curve(x / 20000.0, 1.0, 0.5)
    down = I.measure(x, z, I.SLOPE)
    up = I.measure(x, z[::-1], I.SLOPE)
    assert down["ok"] and up["ok"]
    assert down["nearest"] == up["nearest"] == "concave"
    assert down["departure"] == pytest.approx(up["departure"], abs=1e-6)


def test_summit_at_the_end_is_refused():
    x = np.linspace(0.0, 1000.0, 500)
    item = I.measure(x, 1000.0 - x, I.HILL)
    assert not item["ok"]
    assert item["problem"]


@pytest.mark.parametrize("path", sorted((ROOT / "samples").glob("sample_*.csv")))
def test_every_sample_profile_measures(path):
    if "landform_reference" in path.name:
        pytest.skip("reference table, not a profile")
    kind = I.SLOPE if "longprofile" in path.name else I.VALLEY if "transect" in path.name else I.HILL
    d = pd.read_csv(path)
    item = I.measure(d.iloc[:, 0], d.iloc[:, 1], kind)
    assert item["ok"], item["problem"]
    assert np.isfinite(item["departure"])
    assert I.summary_row(item, path.stem)
