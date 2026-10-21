"""Checks that the fixed fitting procedure recovers the ideal curves it is built on.

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


def two_sided(g_fn, half_width=1000.0, height=200.0, left_share=0.5, n=1201, kind=I.HILL,
              right_height=None):
    """A profile built from a half-profile shape, with flat ground beyond both feet.

    right_height gives the right half its own relief (a valley with one rim lower).
    """
    x = np.linspace(-1.5 * half_width, 1.5 * half_width, n)
    wl = 2.0 * half_width * left_share
    wr = 2.0 * half_width - wl
    xa = -half_width + wl
    u = np.where(x <= xa, (xa - x) / wl, (x - xa) / wr)
    g = np.where(u <= 1.0, g_fn(np.clip(u, 0.0, 1.0)), 0.0)
    hr = height if right_height is None else right_height
    drop = np.where(x <= xa, height, hr)
    z = height - drop * (1.0 - g)
    return x, (height - z if kind == I.VALLEY else z)


def params(item, fam=None):
    return item["comparisons"][fam or item["family"]]["params"]


@pytest.mark.parametrize("k", [-1.0, -0.5, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("kind", [I.HILL, I.VALLEY])
def test_a_parabola_is_fitted_with_a_parabola(k, kind):
    x, z = two_sided(lambda u: I.parabola_curve(u, k), kind=kind)
    item = I.measure(x, z, kind)
    assert item["ok"], item["problem"]
    assert item["family"] == "parabola"
    for v in params(item).values():
        assert v == pytest.approx(k, abs=0.03)
    assert item["departure"] < 0.005


@pytest.mark.parametrize("k", [-1.0, -0.5, 0.5, 1.0])
def test_a_curved_profile_is_not_left_with_a_bent_triangle(k):
    # The complaint this procedure answers: a real parabola must beat the
    # straight comparison by a wide margin and not be named straight.
    item = I.measure(*two_sided(lambda u: I.parabola_curve(u, k)), I.HILL)
    assert item["comparisons"][I.STRAIGHT]["departure"] > 10 * item["departure"]
    assert item["landmark"] != "straight"


@pytest.mark.parametrize("e", [0.5, 3.0, 5.0])
@pytest.mark.parametrize("kind", [I.HILL, I.VALLEY])
def test_shapes_beyond_any_parabola_go_to_the_superellipse(e, kind):
    x, z = two_sided(lambda u: I.ideal_curve(u, e), kind=kind)
    item = I.measure(x, z, kind)
    assert item["ok"], item["problem"]
    assert item["family"] == "superellipse"
    for v in params(item).values():
        assert v == pytest.approx(e, rel=0.05)
    assert item["departure"] < 0.01


@pytest.mark.parametrize("height,bulge", [(600.0, 0.10), (600.0, -0.10), (1000.0, 0.15)])
def test_a_true_circular_arc_is_fitted_with_the_circle(height, bulge):
    a = height / 1000.0
    x, z = two_sided(lambda u: I.circle_curve(u, bulge, a), height=height)
    item = I.measure(x, z, I.HILL)
    assert item["ok"], item["problem"]
    assert item["family"] == "circle"
    for v in params(item).values():
        assert v == pytest.approx(bulge, abs=0.01)
    assert item["departure"] < 0.005


def test_circle_passes_through_apex_and_foot_and_refuses_overhangs():
    for a in (0.05, 0.2, 0.6, 1.0):
        for b in (-0.01, 0.01):
            g = I.circle_curve(I.U, b, a)
            assert g[0] == pytest.approx(1.0)
            assert g[-1] == pytest.approx(0.0, abs=1e-9)
    # On a low feature a strongly bulging arc would rise above the apex.
    assert I.circle_curve(I.U, 0.3, 0.1) is None


def test_a_straight_cone_stays_straight():
    item = I.measure(*two_sided(lambda u: 1.0 - u), I.HILL)
    assert item["family"] == "parabola"           # the tie goes to the parabola
    for v in params(item).values():
        assert abs(v) < 0.03
    assert item["landmark"] == "straight"
    assert item["name"] == "straight-sided cone"


def test_asymmetry_is_kept_not_averaged():
    x, z = two_sided(lambda u: 1.0 - u, left_share=0.25)
    item = I.measure(x, z, I.HILL)
    assert item["ok"]
    left = item["m"]["x_apex"] - item["m"]["x_left"]
    right = item["m"]["x_right"] - item["m"]["x_apex"]
    # Each half keeps its own width instead of being averaged to 1:1.
    assert left / right == pytest.approx(1.0 / 3.0, rel=0.10)


def test_a_volcano_on_a_regional_slope_is_read_to_its_own_foot():
    # A straight cone on ground that keeps falling to the right. Each side
    # running to its own base sent the right foot out to the end of the line.
    x = np.linspace(-3000.0, 6000.0, 1801)
    z = 200.0 * np.clip(1.0 - np.abs(x) / 1000.0, 0.0, None) - 0.02 * x
    item = I.measure(x, z, I.HILL)
    assert item["ok"], item["problem"]
    assert -1100.0 < item["m"]["x_left"] < -800.0
    assert 700.0 < item["m"]["x_right"] < 1100.0
    assert item["departure"] < 0.02


def test_a_valley_with_unequal_rims_is_read_to_the_lower_rim():
    # A U-shaped valley whose right rim is 40% lower than its left.
    x, z = two_sided(lambda u: I.parabola_curve(u, -1.0), kind=I.VALLEY, right_height=120.0)
    item = I.measure(x, z, I.VALLEY)
    assert item["ok"], item["problem"]
    for h in item["halves"]:
        assert h["height"] == pytest.approx(120.0, rel=0.03)
    # Cut at the lower rim, each wall is still a U-shaped parabola.
    for v in params(item, "parabola").values():
        assert v == pytest.approx(-1.0, abs=0.05)
    assert item["comparisons"]["parabola"]["departure"] < 0.01
    assert item["departure"] < 0.01


@pytest.mark.parametrize("a", [3.0, 5.0, 8.0])
@pytest.mark.parametrize("kind", [I.HILL, I.SLOPE])
def test_a_graded_profile_is_fitted_with_the_exponential(a, kind):
    if kind == I.SLOPE:
        x = np.linspace(0.0, 20000.0, 2001)
        z = 1500.0 * I.exponential_curve(x / 20000.0, a)
    else:
        x, z = two_sided(lambda u: I.exponential_curve(u, a))
    item = I.measure(x, z, kind)
    assert item["ok"], item["problem"]
    assert item["family"] == "exponential"
    for v in params(item).values():
        assert v == pytest.approx(a, rel=0.05)
    assert item["departure"] < 0.005


def test_flat_floor_is_measured_and_the_walls_fitted_beyond_it():
    # Straight walls on a flat floor three tenths of the width across.
    x, z = two_sided(lambda u: np.minimum(1.0 - u, 0.7) / 0.7, kind=I.VALLEY)
    item = I.measure(x, z, I.VALLEY)
    assert item["ok"]
    assert item["flat"]
    assert item["name"] == "V-shaped with a flat floor"
    assert item["family"] == "parabola"
    for v in params(item).values():
        assert abs(v) < 0.1                      # the walls are straight
    assert item["departure"] < 0.02


def test_rock_sign_is_the_same_for_hills_and_valleys():
    # A cone and a V, each with extra rock added near the foot.
    def bump(u):
        return 0.15 * np.exp(-((u - 0.8) / 0.08) ** 2)

    hill = I.measure(*two_sided(lambda u: np.clip(1.0 - u + bump(u), 0.0, None)), I.HILL)
    # For a valley, extra rock near the rim is LESS depth there.
    valley = I.measure(*two_sided(lambda u: np.clip(1.0 - u - bump(u), 0.0, None), kind=I.VALLEY),
                       I.VALLEY)
    assert hill["comparisons"][I.STRAIGHT]["net"] > 0
    assert valley["comparisons"][I.STRAIGHT]["net"] > 0
    assert hill["comparisons"][I.STRAIGHT]["zones"]["foot"] > 0
    assert valley["comparisons"][I.STRAIGHT]["zones"]["rim"] > 0


def test_long_profile_is_one_half_and_direction_does_not_matter():
    x = np.linspace(0.0, 20000.0, 2001)
    z = 1500.0 * I.parabola_curve(x / 20000.0, 1.0)
    down = I.measure(x, z, I.SLOPE)
    up = I.measure(x, z[::-1], I.SLOPE)
    assert down["ok"] and up["ok"]
    assert down["family"] == up["family"] == "parabola"
    assert down["landmark"] == up["landmark"] == "concave"
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
    # The chosen ideal is never worse than a straight cone or V.
    assert item["departure"] <= item["comparisons"][I.STRAIGHT]["departure"] + 1e-9
    assert I.summary_row(item, path.stem)
