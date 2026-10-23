"""Checks that the one fixed procedure recovers the ideal curves it is built on,
and that it treats every landform alike.

Run:  py -m pytest tests
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ideal_shape as I  # noqa: E402


def families(item):
    """The families the halves took, each half choosing its own."""
    return {f for f, _, _ in item["fit"]["sides"].values()}


def bends(item):
    return [v for _, _, v in item["fit"]["sides"].values()]


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


def assert_recovers(item, g_true, tol=0.01):
    """The fitted ideal lies on the true curve: less than tol of the box between them on each half."""
    for s, g in item["ideal"].items():
        assert np.trapezoid(np.abs(g - g_true(I.U)), I.U) < tol, s


def straight_departure(item):
    """Departure from a fixed straight cone or V on the same frame, for scale."""
    straight = {h["side"]: I.parabola_curve(I.U, 0.0) for h in item["halves"]}
    return I.compare(item["halves"], straight, item["kind"])["departure"]


LIBRARY = ROOT / "samples" / "library"
FEATURES = pd.read_csv(LIBRARY / "features.csv")
KIND = {"volcano": I.HILL, "valley": I.VALLEY}
FITS = {(r["feature_id"], r["column"]): r for r in json.loads((LIBRARY / "fits.json").read_text())}


def transect(fid, column=None):
    """(x, z) of one library transect; the first one when no column is named."""
    d = pd.read_csv(LIBRARY / f"{fid}.csv")
    col = column or d.columns[1]
    return d.iloc[:, 0].to_numpy(dtype=float), d[col].to_numpy(dtype=float)


def library_transects():
    out = []
    for f in FEATURES.itertuples():
        for col in pd.read_csv(LIBRARY / f"{f.id}.csv", nrows=0).columns[1:]:
            out.append(pytest.param(f.id, f.feature, col, id=f"{f.id}-{col}"))
    return out


def measure_transect(fid, feature, col):
    x, z = transect(fid, col)
    if feature == "ridge":
        return I.read_ridge(x, z)["whole"]
    return I.measure(x, z, KIND[feature])


@pytest.mark.parametrize("k", [-1.0, -0.5, 0.0, 0.5, 1.0])
@pytest.mark.parametrize("kind", [I.HILL, I.VALLEY])
def test_a_parabola_is_fitted_with_a_parabola(k, kind):
    x, z = two_sided(lambda u: I.parabola_curve(u, k), kind=kind)
    item = I.measure(x, z, kind)
    assert item["ok"], item["problem"]
    if k != 0.0:
        # A straight side is a parabola, an exponential and a superellipse at
        # once; any of them is right.
        assert families(item) == {"parabola"}
    assert_recovers(item, lambda u: I.parabola_curve(u, k))
    assert item["departure"] < 0.005


@pytest.mark.parametrize("k", [-1.0, -0.5, 0.5, 1.0])
def test_a_curved_profile_is_not_left_with_a_bent_triangle(k):
    # A real parabola must beat a straight cone by a wide margin and not be named straight.
    item = I.measure(*two_sided(lambda u: I.parabola_curve(u, k)), I.HILL)
    assert straight_departure(item) > 10 * item["departure"]
    assert item["landmark"] != "straight"


@pytest.mark.parametrize("e", [0.5, 2.0, 3.0, 5.0])
@pytest.mark.parametrize("kind", [I.HILL, I.VALLEY])
def test_shapes_beyond_any_parabola_go_to_the_superellipse(e, kind):
    x, z = two_sided(lambda u: I.superellipse_curve(u, e), kind=kind)
    item = I.measure(x, z, kind)
    assert item["ok"], item["problem"]
    assert families(item) == {"superellipse"}
    # e = 0.5 comes to a cusp at the apex that the smoothing rounds off.
    assert_recovers(item, lambda u: I.superellipse_curve(u, e), tol=0.025 if e < 1 else 0.01)
    assert item["departure"] < 0.01


def test_a_straight_cone_stays_straight():
    item = I.measure(*two_sided(lambda u: 1.0 - u), I.HILL)
    assert_recovers(item, lambda u: 1.0 - u)
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
    assert families(item) == {"parabola"}
    assert_recovers(item, lambda u: I.parabola_curve(u, -1.0))
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
    assert families(item) == {"exponential"}
    for v in bends(item):
        assert v == pytest.approx(a, rel=0.05)
    assert item["departure"] < 0.005


def test_a_flat_floor_is_fitted_as_a_level_stretch_with_straight_walls():
    # Straight walls on a flat floor three tenths of the width across.
    x, z = two_sided(lambda u: np.minimum(1.0 - u, 0.7) / 0.7, kind=I.VALLEY)
    item = I.measure(x, z, I.VALLEY)
    assert item["ok"]
    assert item["flat"]
    assert item["name"] == "V-shaped with a flat floor"
    for fam, level, v in item["fit"]["sides"].values():
        assert level == pytest.approx(0.3, abs=0.03)
        assert fam == "parabola" and abs(v) < 0.1      # the walls are straight
    assert item["departure"] < 0.01


def test_a_rounded_floor_is_not_given_a_flat_one():
    item = I.measure(*two_sided(lambda u: I.parabola_curve(u, -1.0), kind=I.VALLEY), I.VALLEY)
    assert not item["flat"]
    assert item["name"] == "parabolic U"


def test_rock_sign_is_the_same_for_hills_and_valleys():
    # A cone and a V, each with extra rock added near the foot.
    def bump(u):
        return 0.15 * np.exp(-((u - 0.8) / 0.08) ** 2)

    hill = I.measure(*two_sided(lambda u: np.clip(1.0 - u + bump(u), 0.0, None)), I.HILL)
    # For a valley, extra rock near the rim is LESS depth there.
    valley = I.measure(*two_sided(lambda u: np.clip(1.0 - u - bump(u), 0.0, None), kind=I.VALLEY),
                       I.VALLEY)
    for item in (hill, valley):
        straight = {h["side"]: I.parabola_curve(I.U, 0.0) for h in item["halves"]}
        c = I.compare(item["halves"], straight, item["kind"])
        assert c["net"] > 0
        near_foot = I.U > 0.7
        assert all(c["residual"][s][near_foot].mean() > 0 for s in c["residual"])


def test_long_profile_is_one_half_and_direction_does_not_matter():
    x = np.linspace(0.0, 20000.0, 2001)
    z = 1500.0 * I.parabola_curve(x / 20000.0, 1.0)
    down = I.measure(x, z, I.SLOPE)
    up = I.measure(x, z[::-1], I.SLOPE)
    assert down["ok"] and up["ok"]
    assert families(down) == families(up) == {"parabola"}
    assert down["landmark"] == up["landmark"] == "concave"
    assert down["departure"] == pytest.approx(up["departure"], abs=1e-6)


def test_summit_at_the_end_is_refused():
    x = np.linspace(0.0, 1000.0, 500)
    item = I.measure(x, 1000.0 - x, I.HILL)
    assert not item["ok"]
    assert item["problem"]


# ---------------------------------------------------------------------
# One procedure, the same for every landform
# ---------------------------------------------------------------------
def test_the_departure_does_not_depend_on_size():
    """The same landform drawn bigger, wider or taller gets the same number."""
    x, z = transect("taranaki")
    base = I.measure(x, z, I.HILL)["departure"]
    for sx, sz in ((10.0, 1.0), (1.0, 0.1), (0.003, 7.0)):
        assert I.measure(x * sx, z * sz + 500.0, I.HILL)["departure"] == pytest.approx(base, abs=1e-6)


def test_the_departure_is_the_least_rock_out_of_place():
    """Nudging the fitted top, level stretch or bend never leaves less rock out of place."""
    item = I.measure(*transect("fournaise"), I.HILL)
    fit, halves = item["fit"], item["halves"]
    assert I._out_of_place(halves, fit["top"], fit["sides"]) == pytest.approx(item["departure"], abs=1e-9)
    rng = np.random.default_rng(0)
    for _ in range(40):
        top = fit["top"] + rng.normal(0, 0.01)
        sides = {s: (f, c + rng.normal(0, 0.01), v * (1 + rng.normal(0, 0.01)))
                 for s, (f, c, v) in fit["sides"].items()}
        assert I._out_of_place(halves, top, sides) >= item["departure"] - 1e-6


def test_a_caldera_gets_one_level_top_not_a_staircase():
    """Crater Lake: one ideal, level across the caldera at one height on both sides."""
    item = I.measure(*transect("craterlake"), I.HILL)
    left, right = item["halves"]
    assert left["height"] == right["height"]
    assert item["ideal"]["L"][0] == pytest.approx(item["ideal"]["R"][0])
    assert item["flat"]


def test_a_spike_on_a_broad_rise_is_extra_rock_not_the_top_of_the_ideal():
    """A narrow volcano on a broad dome: the ideal follows the dome and the spike is extra rock."""
    x = np.linspace(-1500.0, 1500.0, 3001)
    z = 200.0 * np.clip(I.parabola_curve(np.abs(x) / 1000.0, -1.0), 0.0, None) \
        + 150.0 * np.exp(-(x / 15.0) ** 2)
    item = I.measure(x, z, I.HILL)
    assert item["fit"]["top"] < 0.75
    assert item["excess"] > item["deficit"]


@pytest.mark.parametrize("fid, feature, col", library_transects())
def test_every_library_transect_measures(fid, feature, col):
    item = measure_transect(fid, feature, col)
    assert item["ok"], item["problem"]
    assert np.isfinite(item["departure"])
    # The ideal never stands above the real top, and is never worse than a straight cone or V.
    assert 0.0 < item["fit"]["top"] <= 1.0
    assert item["departure"] <= straight_departure(item) + 1e-9
    assert I.summary_row(item, fid)
    # The library view draws from fits.json, so it must still be what the engine gives.
    stored = FITS[(fid, col)]
    assert stored["ok"] and stored["departure"] == pytest.approx(item["departure"], abs=1e-9)
    # The fixed triangle on the same frame: never closer than the best fit, and stored alike.
    tri = I.as_triangle(item)
    assert item["departure"] <= tri["departure"] + 1e-9
    assert stored["tri_departure"] == pytest.approx(tri["departure"], abs=1e-9)
    assert I.summary_row(tri, fid)["Ideal"] == "fixed triangle"


def test_the_library_has_ten_of_each_and_every_transect_is_placed():
    assert FEATURES.feature.value_counts().to_dict() == {"volcano": 10, "valley": 10, "ridge": 10}
    placed = pd.read_csv(LIBRARY / "transects.csv")
    for f in FEATURES.itertuples():
        cols = list(pd.read_csv(LIBRARY / f"{f.id}.csv", nrows=0).columns[1:])
        assert len(cols) >= 3
        assert sorted(placed[placed.feature_id == f.id].column) == sorted(cols)


def test_transects_are_averaged():
    x, z = two_sided(lambda u: 1.0 - u)
    items = [I.measure(x, z + 30.0 * np.exp(-((x - c) / 60.0) ** 2), I.HILL) for c in (-500.0, 0.0, 400.0)]
    items.append({"ok": False})
    avg = I.average(items)
    d = [it["departure"] for it in items[:3]]
    assert avg["n"] == 3 and avg["failed"] == 1
    assert avg["mean"] == pytest.approx(np.mean(d))
    assert avg["sd"] == pytest.approx(np.std(d, ddof=1))
    assert (avg["lo"], avg["hi"]) == (min(d), max(d))
    one = I.average(items[:1])
    assert one["sd"] == 0.0 and one["mean"] == d[0]


# ---------------------------------------------------------------------
# Spreading ridges
# ---------------------------------------------------------------------

RIDGES = [p for p in library_transects() if p.values[1] == "ridge"]


@pytest.mark.parametrize("fid, feature, col", RIDGES)
def test_each_real_ridge_reads_as_its_spreading_rate(fid, feature, col):
    """Every transect of every library ridge, against its MORVEL full rate."""
    rate = float(FEATURES.set_index("id").loc[fid, "full_rate_mm_yr"])
    expected = "slow" if rate < 40 else "intermediate" if rate < 80 else "fast"
    x, z = transect(fid, col)
    r = I.read_ridge(x, z)
    assert r["rate_class"] == expected
    assert r["axis_type"] == ("axial valley" if r["rate_class"] == "slow" else "axial high")
    assert r["whole"]["ok"]
    # The whole ridge is framed from the axis, down both flanks.
    m = r["whole"]["m"]
    assert m["x_apex"] == r["axis"]
    assert min(r["axis"] - m["x_left"], m["x_right"] - r["axis"]) > 50_000.0


def _cooling_ridge(axial):
    """Sea floor deepening with the square root of distance, plus an axial feature."""
    x = np.arange(-200_000.0, 200_001.0, 500.0)
    return x, -2500.0 - 6.0 * np.sqrt(np.abs(x)) + axial(np.abs(x))


def test_an_axial_rift_valley_reads_slow_and_is_missing_rock():
    x, z = _cooling_ridge(lambda r: -1500.0 * np.clip(1 - r / 15_000, 0, 1) + 800.0 * np.exp(-((r - 20_000) / 6000) ** 2))
    r = I.read_ridge(x, z)
    assert r["axis_type"] == "axial valley" and r["rate_class"] == "slow"
    w = r["whole"]
    # The rift is cut below the ideal at the axis on both sides.
    assert all(w["residual"][s][0] < 0 for s in ("L", "R"))


def test_a_peaked_axial_high_reads_fast():
    x, z = _cooling_ridge(lambda r: 300.0 * np.exp(-r / 4000))
    r = I.read_ridge(x, z)
    assert r["axis_type"] == "axial high" and r["rate_class"] == "fast"


def test_each_half_takes_its_own_family():
    """A dome on one side and a graded concave flank on the other are not forced into one shape."""
    x = np.linspace(-1500.0, 1500.0, 1801)
    u = np.clip(np.abs(x) / 1000.0, 0.0, 1.0)
    g = np.where(x < 0, I.parabola_curve(u, -1.0), I.exponential_curve(u, 5.0))
    item = I.measure(x, 200.0 * g, I.HILL)
    assert {s: f for s, (f, _, _) in item["fit"]["sides"].items()} == {"L": "parabola", "R": "exponential"}
    assert item["departure"] < 0.005


def test_real_transects_of_one_volcano_are_averaged():
    """Four transects through the summit of Mount Fuji, each fitted the same way."""
    df = pd.read_csv(LIBRARY / "fuji.csv")
    avg = I.average([I.measure(df.iloc[:, 0], df[c], I.HILL) for c in df.columns[1:]])
    assert avg["n"] == 4 and avg["failed"] == 0
    assert avg["lo"] < avg["mean"] < avg["hi"] < 0.025


# ---------------------------------------------------------------------
# The fixed triangle: one rigid ideal, nothing fitted
# ---------------------------------------------------------------------
@pytest.mark.parametrize("kind", [I.HILL, I.VALLEY])
def test_a_perfect_triangle_is_the_fixed_triangle(kind):
    item = I.as_triangle(I.measure(*two_sided(lambda u: 1.0 - u, kind=kind), kind))
    assert item["departure"] < 0.005
    assert item["against"] == I.TRIANGLE and item["grade"] == ""


@pytest.mark.parametrize("kind", [I.HILL, I.VALLEY])
@pytest.mark.parametrize("k", [1.0, -1.0])
def test_a_centred_parabola_is_one_sixth_from_the_triangle(k, kind):
    """The other end member: a perfect parabola, bowed in or domed, leaves 1/6 of its box out of place."""
    item = I.measure(*two_sided(lambda u: I.parabola_curve(u, k), kind=kind), kind)
    tri = I.as_triangle(item)
    assert tri["departure"] == pytest.approx(I.TRIANGLE_PARABOLA, abs=0.005)
    # A cone with bowed-in flanks is missing rock against the triangle, a dome extra rock; a
    # U-shaped valley (the dome, upside down) is missing rock against the V.
    assert (tri["deficit"] > tri["excess"]) == ((k > 0) == (kind == I.HILL))
    # The best fit still finds the parabola.
    assert item["departure"] < 0.01


def test_a_summit_off_the_middle_is_rock_out_of_place():
    """Straight sides meeting off the middle fit perfectly, but are not the isosceles triangle."""
    item = I.measure(*two_sided(lambda u: 1.0 - u, left_share=0.3), I.HILL)
    tri = I.as_triangle(item)
    assert item["departure"] < 0.005
    assert item["triangle"]["offset"] == pytest.approx(-0.2, abs=0.01)
    assert tri["departure"] > 0.05
    # The triangle's apex is midway between the feet, at the full height.
    m = item["m"]
    x, _, zt = I.drawn_halves(tri)
    assert x[np.argmax(zt)] == pytest.approx(0.5 * (m["x_left"] + m["x_right"]), abs=0.01 * m["width"])
    assert zt.max() == pytest.approx(m["apex_elev"], abs=0.01 * m["height"])


def test_the_triangle_depends_only_on_the_frame():
    """Two different profiles on the same feet and top get the same triangle."""
    a = I.measure(*two_sided(lambda u: I.parabola_curve(u, 0.6)), I.HILL)
    b = I.measure(*two_sided(lambda u: I.superellipse_curve(u, 1.8)), I.HILL)
    for s in ("L", "R"):
        assert np.allclose(a["triangle"]["ideal"][s], b["triangle"]["ideal"][s], atol=0.01)


def test_the_triangle_departure_does_not_depend_on_size():
    x, z = transect("taranaki")
    base = I.measure(x, z, I.HILL)["triangle"]["departure"]
    for sx, sz in ((10.0, 1.0), (1.0, 0.1), (0.003, 7.0)):
        assert I.measure(x * sx, z * sz + 500.0, I.HILL)["triangle"]["departure"] == pytest.approx(base, abs=1e-6)


def test_a_long_profile_is_compared_with_one_straight_side():
    x = np.linspace(0.0, 20000.0, 2001)
    straight = I.as_triangle(I.measure(x, 1500.0 * (1.0 - x / 20000.0), I.SLOPE))
    graded = I.as_triangle(I.measure(x, 1500.0 * I.parabola_curve(x / 20000.0, 1.0), I.SLOPE))
    assert straight["departure"] < 0.005
    assert graded["departure"] == pytest.approx(I.TRIANGLE_PARABOLA, abs=0.005)
