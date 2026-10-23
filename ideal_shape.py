"""
Ideal-shape fitting for topographic profiles.

The premise is Wadell's grain construction. Wadell (1932, 1935) fitted a sand
grain with a perfect sphere and measured the material left over: how far a
grain departs from the sphere records how it has been worn. Here the grain is a
landform and the sphere is the best-fitting ideal curve.

Every landform, of every type and size, goes through ONE fixed procedure:

  1. THE FRAME, measured. Find the apex (the summit of a hill, the floor of a
     valley, the head of a long profile, the axis of a spreading ridge) and the
     two feet, both at one base level. Each half, from the apex to a foot, sits
     in a box as wide as that half and as tall as the landform. Real and ideal
     are compared in that box, so a 200 m valley and a 5 km volcano are
     compared at the same scale.

  2. THE IDEAL, fitted. Down each half runs one smooth curve, a parabola, an
     exponential or a superellipse, each with one bend control, held level
     across the top for as far as fits best (a level top, or a flat floor).
     Both halves share the ideal's top height, anywhere up to the real top.
     The top height, and each half's level stretch, curve and bend, are chosen
     together to leave the least rock out of place.

  3. THE NUMBER. Departure is the rock out of place - extra rock above the
     ideal plus missing rock below it - as a share of the box. 0 means the
     landform IS its ideal; the further it departs, the bigger the number.

  4. SEVERAL TRANSECTS of one landform are each put through steps 1 to 3 and
     their departures averaged.

There are no other rules: no thresholds decide whether a top is flat, no
summit is read from one rim, no ideal is pinned to a spike. The departure is
exactly the quantity the fit makes as small as it can.

A valley is a hill with earth and sky exchanged. It is measured upside down,
so one engine serves both, and a river long profile is a single half on its
own. The measuring engine (PROCESSING ... analyse_profile) is ported from
Spectrum Jax, where it was validated on real volcano transects. Nothing in this
module touches Streamlit, so it can be tested on its own.
"""

from __future__ import annotations

import numpy as np
from itertools import product

from scipy.optimize import minimize
from scipy.signal import find_peaks, savgol_filter

HILL = "hill"
VALLEY = "valley"
SLOPE = "slope"

PROCESSING = {
    # Profile is resampled to this many evenly spaced points before measurement.
    "resample_points": 2001,
    # Savitzky-Golay smoothing, as a fraction of profile length. Smoothing
    # stabilises the fit; too much erases a sharp apex.
    "smooth_fraction": 0.01,
    "smooth_polyorder": 3,
    # Each half's foot is this quantile of that half's (oriented) elevations,
    # not its single lowest sample, and the HIGHER of the two is used. A single
    # stray reading on the ground beyond the feature would otherwise capture the
    # measurement entirely.
    "base_quantile": 0.05,
    # The flat top (or floor) is the ground within this fraction of the height
    # of the apex, as a fraction of the width.
    "plateau_level": 0.05,
    # The apex is the middle of the ground within this finer fraction of the
    # relief of the top (inside the coarser summit run).
    "apex_level": 0.01,
    # Two-pass measurement: the second pass crops to the feature plus this
    # fraction of its width on each side and smooths again at that scale, so
    # the answer does not depend on how much surrounding ground the file holds.
    "window_margin": 0.30,
    # High ground closer than this fraction of the width is one apex (the two
    # rims of a crater, the two banks of an inner channel).
    "summit_merge": 0.25,
    # High ground is also one apex when the dip between the pieces is shallower
    # than this fraction of the relief.
    "saddle_depth": 0.15,
    # The shorter half must be at least this share of the longer, or the line
    # does not cross the feature.
    "min_side_ratio": 0.10,
    # Nor when one side has less than this share of the other side's relief.
    "min_side_relief": 0.05,
}

VOCAB = {
    HILL: {"feature": "hill", "apex": "summit", "flat": "flat top",
           "side": "flank",
           "sides": "flanks", "foot": "foot"},
    VALLEY: {"feature": "valley", "apex": "valley floor", "flat": "flat floor",
             "side": "wall",
             "sides": "walls", "foot": "rim"},
    SLOPE: {"feature": "profile", "apex": "head", "flat": "flat head",
            "side": "slope",
            "sides": "slope", "foot": "mouth"},
}

U = np.linspace(0.0, 1.0, 401)


# =====================================================================
# Measuring the feature
# =====================================================================
def _smooth(zr: np.ndarray) -> np.ndarray:
    n = len(zr)
    w = int(max(5, round(n * PROCESSING["smooth_fraction"])))
    if w % 2 == 0:
        w += 1
    if w <= PROCESSING["smooth_polyorder"]:
        w = PROCESSING["smooth_polyorder"] + 2 + (PROCESSING["smooth_polyorder"] % 2)
    return savgol_filter(zr, w, PROCESSING["smooth_polyorder"]) if w < n else zr.copy()


def _resample(x: np.ndarray, z: np.ndarray, a: float, b: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample the stretch from a to b, from the ORIGINAL samples, and smooth it."""
    xs = np.linspace(a, b, int(PROCESSING["resample_points"]))
    return xs, _smooth(np.interp(xs, x, z))


def _summit_run(mask: np.ndarray, k0: int, x: np.ndarray, merge_gap: float,
                z: np.ndarray | None = None, shallow: float | None = None) -> tuple[int, int]:
    """Index span of the run of True samples containing k0.

    Only CONTIGUOUS high ground around the highest point is the apex. A
    neighbouring run joins it when it lies within merge_gap (crater rims), or
    when the ground between never drops below `shallow` (the ridges a smoothing
    filter leaves at the edges of a steep-walled flat top).
    """
    n = len(mask)
    lo = hi = k0
    while True:
        while lo > 0 and mask[lo - 1]:
            lo -= 1
        while hi < n - 1 and mask[hi + 1]:
            hi += 1
        grew = False
        for step in (-1, 1):
            edge = lo if step < 0 else hi
            j = edge + step
            shallow_ok = z is not None and shallow is not None
            found = False
            while 0 <= j < n:
                if mask[j]:
                    found = True
                    break
                if shallow_ok and z[j] < shallow:
                    shallow_ok = False
                if not shallow_ok and abs(x[j] - x[edge]) > merge_gap:
                    break
                j += step
            if found and (abs(x[j] - x[edge]) <= merge_gap or shallow_ok):
                if step < 0:
                    lo = j
                else:
                    hi = j
                grew = True
        if not grew:
            return lo, hi


def _measure_apex(x: np.ndarray, z: np.ndarray, kind: str, merge_gap: float = 0.0) -> tuple[dict, list]:
    """Locate apex and feet on an ORIENTED profile (apex up). Returns (metrics, notes)."""
    w = VOCAB[kind]
    notes: list = []
    if len(x) < 32:
        return {}, ["The profile is too short to measure."]

    # Centre the apex on the flat top, not on the single highest sample: on a
    # flat-topped feature argmax lands wherever noise happens to peak.
    z_top = float(np.max(z))
    z_rng = z_top - float(np.min(z))
    band = z_top - PROCESSING["plateau_level"] * z_rng
    lo_s, hi_s = _summit_run(z >= band, int(np.argmax(z)), x, merge_gap,
                             z=z, shallow=z_top - PROCESSING["saddle_depth"] * z_rng)
    # Within that run, the apex is the middle of the ground standing within a
    # finer band of the top. On a flat top, or between two crater rims of equal
    # height, that is still the middle of the run; on a lopsided point it stays
    # at the point instead of sliding out along the gentler side, which is where
    # the middle of the coarse run falls.
    fine = np.flatnonzero(z[lo_s:hi_s + 1] >= z_top - PROCESSING["apex_level"] * z_rng) + lo_s
    t = int(round(0.5 * (fine[0] + fine[-1]))) if len(fine) else int(np.argmax(z))
    if t < 4 or t > len(z) - 5:
        return {}, [
            f"The {w['apex']} sits at the end of the profile, so it does not cross the whole "
            f"{w['feature']}. Extend the profile past the {w['apex']} on both sides, or choose "
            "'One slope' if this is a long profile."
        ]

    left, right = z[: t + 1], z[t:]
    q = PROCESSING["base_quantile"]
    base_l = float(np.quantile(left, q))
    base_r = float(np.quantile(right, q))
    # A line that starts or ends at the apex has almost no relief on one side.
    drops = (z_top - base_l, z_top - base_r)
    if min(drops) < PROCESSING["min_side_relief"] * max(drops):
        return {}, [
            f"The {w['apex']} has almost no relief on one side: that side has only "
            f"{min(drops) / max(drops):.0%} of the other side's. The line probably starts or ends at "
            f"the {w['apex']}. Use a line that runs past it on both sides, or choose 'One slope'."
        ]

    # Both halves stop at the HIGHER of the two bases, at the first crossing
    # going out from the apex (SpectrumJax's rule, checked on 60 real
    # transects). Letting each half run down to its own base sent the lower
    # side of a volcano on a regional slope out across the slope, far past the
    # foot of the edifice, and the fit followed it.
    z_base = max(base_l, base_r)
    height = z_top - z_base
    if height <= 0:
        return {}, ["There is no relief between the apex and the foot."]

    if abs(base_l - base_r) > 0.25 * height:
        notes.append(
            f"The two {w['sides']} reach their {w['foot']} {abs(base_l - base_r) / height:.0%} of the "
            f"height apart. Both are measured to the higher one, so the lower {w['side']} stops "
            f"there: the {w['feature']} is read from its {w['apex']} to the level where both "
            f"{w['sides']} still stand."
        )

    k_left = k_right = t
    def crossing(direction: int, zb: float):
        i = k_left if direction < 0 else k_right
        while 0 <= i < len(z):
            if z[i] <= zb:
                break
            i += direction
        if i < 0 or i >= len(z):
            return None
        j = i - direction
        if not (0 <= j < len(z)) or z[j] == z[i]:
            return float(x[i])
        f = (zb - z[i]) / (z[j] - z[i])
        return float(x[i] + f * (x[j] - x[i]))

    xl, xr = crossing(-1, z_base), crossing(+1, z_base)
    if xl is None or xr is None or xr <= xl:
        return {}, [f"Could not find the {w['foot']} on both sides of the {w['apex']}."]

    width = xr - xl
    xs = np.linspace(xl, xr, 2001)
    zs = np.interp(xs, x, z)
    x_apex = float(x[t])

    lvl = z_top - PROCESSING["plateau_level"] * height
    above = zs >= lvl
    k0 = int(np.argmin(np.abs(xs - x_apex)))
    if not above[k0]:
        k0 = int(np.argmax(zs))
    kl, kr = _summit_run(above, k0, xs, merge_gap, z=zs,
                         shallow=z_top - PROCESSING["saddle_depth"] * height)
    plateau = (float(xs[kr]) - float(xs[kl])) / width if kr > kl else 0.0
    x_flat_l, x_flat_r = (float(xs[kl]), float(xs[kr])) if kr > kl else (x_apex, x_apex)

    pk, _ = find_peaks(zs, prominence=0.05 * height)
    far = [k for k in pk if (xs[k] < xs[kl] or xs[k] > xs[kr]) and zs[k] >= z_base + 0.85 * height]
    if far:
        k = max(far, key=lambda i: zs[i])
        notes.append(
            f"A second high point stands {abs(xs[k] - x_apex) / width:.0%} of the width from the "
            f"{w['apex']}. It is treated as part of a {w['side']}, so the fit describes one "
            f"{w['feature']} only."
            if kind == HILL else
            f"A second low point stands {abs(xs[k] - x_apex) / width:.0%} of the width from the "
            f"{w['apex']}. It is treated as part of a {w['side']}, so the fit describes one "
            f"{w['feature']} only."
        )

    depth = np.clip(zs - z_base, 0.0, None)
    lm, rm = xs <= x_apex, xs >= x_apex
    a_l = float(np.trapezoid(depth[lm], xs[lm])) if lm.sum() > 1 else 0.0
    a_r = float(np.trapezoid(depth[rm], xs[rm])) if rm.sum() > 1 else 0.0

    return {
        "height": height,
        "width": width,
        "width_height": width / height,
        "plateau": plateau,
        "asymmetry": abs(a_r - a_l) / (a_r + a_l) if (a_r + a_l) > 0 else float("nan"),
        "apex_elev": z_top,
        "base_elev": z_base,
        "base_left": z_base,
        "base_right": z_base,
        "x_left": xl,
        "x_right": xr,
        "x_apex": x_apex,
        "x_flat_left": min(x_flat_l, x_apex),
        "x_flat_right": max(x_flat_r, x_apex),
    }, notes


def analyse_profile(x, z, kind: str = HILL) -> tuple[dict, list]:
    """The one route every two-sided profile takes, real or ideal: locate, crop, re-measure.

    A valley is inverted first, so `base_elev` and `apex_elev` in the result
    are in the inverted frame; `real_elevation` turns them back.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    zo = -z if kind == VALLEY else z
    xs, zs = _resample(x, zo, float(x.min()), float(x.max()))
    m, notes = _measure_apex(xs, zs, kind)
    if not m:
        return m, notes
    for _ in range(2):
        w = m["width"]
        pad = PROCESSING["window_margin"] * w
        a = max(float(x.min()), m["x_left"] - pad)
        b = min(float(x.max()), m["x_right"] + pad)
        if b - a <= 0:
            break
        xc, zc = _resample(x, zo, a, b)
        m2, n2 = _measure_apex(xc, zc, kind, merge_gap=PROCESSING["summit_merge"] * w)
        if not m2:
            break
        settled = abs(m2["width"] - w) < 0.05 * w
        m, notes = m2, n2
        if settled:
            break
    return m, notes


def real_elevation(value: float, kind: str) -> float:
    return -float(value) if kind == VALLEY else float(value)


def half_profiles(x, z, m: dict, kind: str = HILL) -> list:
    """The two halves of a measured profile, u from apex to foot, each in its box.

    g is the height above the base level as a share of the landform's height,
    for both halves alike. It is not clipped: ground standing above the top of
    the box or dropping below the base (the floor of a rift valley) is real
    rock, or a real hole, and counts as such.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    zo = -z if kind == VALLEY else z
    w = m["width"]
    a = max(float(np.min(x)), m["x_left"] - PROCESSING["window_margin"] * w)
    b = min(float(np.max(x)), m["x_right"] + PROCESSING["window_margin"] * w)
    xc, zc = _resample(x, zo, a, b)
    xa = m["x_apex"]
    height = m["apex_elev"] - m["base_elev"]
    out = []
    for side, radius, sign, base in (("L", xa - m["x_left"], -1.0, m["base_left"]),
                                     ("R", m["x_right"] - xa, 1.0, m["base_right"])):
        zz = np.interp(xa + sign * U * radius, xc, zc)
        out.append({"side": side, "radius": float(radius), "sign": sign, "base": float(base),
                    "height": float(height), "g": (zz - base) / height})
    return out


def analyse_slope(x, z) -> tuple[dict, list, list]:
    """A one-sided profile (a river long profile, a single hillslope) is one half on its own.

    The head is the high end and the mouth the low end, whichever way the file
    runs. Returns (metrics, notes, halves).
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    notes: list = []
    xs, zs = _resample(x, z, float(x.min()), float(x.max()))
    k_hi, k_lo = int(np.argmax(zs)), int(np.argmin(zs))
    if k_hi == k_lo:
        return {}, ["There is no relief along the profile."], []
    lo, hi = sorted((k_hi, k_lo))
    if (hi - lo) < 0.5 * len(xs):
        notes.append(
            "The highest and lowest points are less than half the profile apart, so only the "
            "stretch between them is fitted. Trim the profile to one slope for a cleaner reading."
        )
    x_head, x_mouth = float(xs[k_hi]), float(xs[k_lo])
    z_head, z_mouth = float(zs[k_hi]), float(zs[k_lo])
    length = abs(x_mouth - x_head)
    height = z_head - z_mouth
    sign = 1.0 if x_mouth > x_head else -1.0
    zz = np.interp(x_head + sign * U * length, xs, zs)
    g = (zz - z_mouth) / height
    m = {
        "height": height,
        "width": length,
        "width_height": length / height,
        "plateau": 0.0,
        "asymmetry": float("nan"),
        "apex_elev": z_head,
        "base_elev": z_mouth,
        "base_left": z_mouth,
        "base_right": z_mouth,
        "x_left": min(x_head, x_mouth),
        "x_right": max(x_head, x_mouth),
        "x_apex": x_head,
    }
    return m, notes, [{"side": "D", "radius": float(length), "sign": sign, "base": z_mouth,
                       "height": float(height), "g": g}]


# =====================================================================
# The ideal curves
#
# Every curve is a height fraction g(u) of a half profile: u = 0 at the apex
# and 1 at the foot, g = 1 at the top and 0 at the foot. Each family has ONE
# bend control and contains the straight line, so the families compete on
# equal terms and a straight-sided profile is never forced into a curve.
#
#   parabola      g = (1 - u)(1 - k u)                     -1 <= k <= 1
#   exponential   g = (e^(-a u) - e^(-a)) / (1 - e^(-a))
#   superellipse  g = (1 - u^e)^(1/e)
#
# Any of them can be held level across the top for a share of the half's
# width (the level stretch) and then run down to the foot over the rest.
#
# A circle drawn in true proportions is not offered: its shape in the box
# would change with the map units and the vertical scale, so it could not be
# compared from one landform to the next. In the box the superellipse with
# e = 2 is the quarter circle.
# =====================================================================
def parabola_curve(u, k: float) -> np.ndarray:
    """A true parabola (vertical axis) through apex and foot.

    k = 0 is the straight line, k = 1 the concave parabola with its vertex at
    the foot, k = -1 the dome (or U) with its vertex at the apex. Outside that
    range the vertex would fall inside the half and the curve would turn back.
    """
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    k = float(np.clip(k, -1.0, 1.0))
    return (1.0 - u) * (1.0 - k * u)


def exponential_curve(u, a: float) -> np.ndarray:
    """An exponential decay through apex and foot.

    a = 0 is the straight line, a > 0 concave (steep at the apex, still sloping
    at the foot, like a graded river), a < 0 convex.
    """
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    a = float(a)
    if abs(a) < 1e-6:
        return 1.0 - u
    return (np.exp(-a * u) - np.exp(-a)) / (1.0 - np.exp(-a))


def superellipse_curve(u, e: float) -> np.ndarray:
    """g = (1 - u^e)^(1/e): e = 1 straight, e = 2 the quarter circle, e < 1 concave, e > 2 toward a box."""
    e = max(float(e), 0.05)
    return np.clip(1.0 - np.clip(np.asarray(u, float), 0.0, 1.0) ** e, 0.0, None) ** (1.0 / e)


FAMILIES = {
    "parabola": {
        "name": "Parabola", "param": "k", "curve": parabola_curve,
        "grid": np.linspace(-1.0, 1.0, 41), "bounds": (-1.0, 1.0),
    },
    "exponential": {
        "name": "Exponential", "param": "a", "curve": exponential_curve,
        "grid": np.linspace(-10.0, 20.0, 61), "bounds": (-15.0, 30.0),
    },
    "superellipse": {
        "name": "Superellipse", "param": "e", "curve": superellipse_curve,
        "grid": np.geomspace(0.3, 8.0, 49), "bounds": (0.2, 12.0),
    },
}

FIT = {
    # The search that gives each pairing of families its starting point: the
    # ideal's top height as a share of the landform's height, and each half's
    # level stretch as a share of its width.
    "top_grid": np.linspace(0.2, 1.0, 33),
    "level_grid": np.linspace(0.0, 0.8, 17),
    # Limits of the refined fit.
    "top_min": 0.05,
    "level_max": 0.9,
    # Pairings within this share of the box of the best are the same fit (a
    # straight side is a parabola, an exponential and a superellipse at once),
    # and the one with the earlier families in FAMILIES is kept.
    "tie": 0.0001,
}

# Trapezoid weights on U, so that w @ f is the integral of f over the half.
_W = np.full_like(U, U[1] - U[0])
_W[0] = _W[-1] = 0.5 * (U[1] - U[0])
_TABLES: dict = {}


def family_curve(fam: str, value: float) -> np.ndarray:
    return FAMILIES[fam]["curve"](U, value)


def level_top(g, level: float):
    """Curve(s) g on U held level at their top out to u = level, then run down over the rest.

    g may be one curve or an array of them along its last axis.
    """
    g = np.asarray(g, dtype=float)
    if level <= 0:
        return g
    pos = np.clip((U - level) / (1.0 - level), 0.0, 1.0) * (len(U) - 1)
    i = np.clip(np.floor(pos).astype(int), 0, len(U) - 2)
    f = pos - i
    return g[..., i] * (1.0 - f) + g[..., i + 1] * f


def ideal_half(fam: str, value: float, level: float = 0.0) -> np.ndarray:
    """One half of an ideal, at full height: the family's curve with its level stretch."""
    return level_top(family_curve(fam, value), float(np.clip(level, 0.0, FIT["level_max"])))


def _tables() -> dict:
    """Every family's curves over its search grid, with every level stretch: [level, bend, u]."""
    if not _TABLES:
        for fam, spec in FAMILIES.items():
            curves = np.array([family_curve(fam, v) for v in spec["grid"]])
            _TABLES[fam] = np.stack([level_top(curves, c) for c in FIT["level_grid"]])
    return _TABLES


def _out_of_place(halves: list, top: float, sides: dict) -> float:
    """Rock out of place against an ideal, as a share of the box (sum of width x height)."""
    if not (FIT["top_min"] <= top <= 1.0):
        return float("inf")
    num = den = 0.0
    for h in halves:
        fam, level, value = sides[h["side"]]
        lo, hi = FAMILIES[fam]["bounds"]
        if not (0.0 <= level <= FIT["level_max"] and lo <= value <= hi):
            return float("inf")
        num += h["radius"] * float(np.abs(h["g"] - top * ideal_half(fam, value, level)) @ _W)
        den += h["radius"]
    return num / den


def fit_ideal(halves: list) -> dict:
    """The ideal that leaves the least rock out of place: step 2 of the procedure.

    Unknowns: the top height (shared by both halves) and, on each half, the
    family, its bend and the level stretch. A search over a grid of all of
    them gives each pairing of families a starting point; every pairing is
    then refined (Nelder-Mead inside the same limits), and the pairing that
    leaves the least rock out of place is the ideal.

    Returns {"top": share of the height, "sides": {side: (family, level, bend)},
    "departure": rock out of place as a share of the box}.
    """
    tables = _tables()
    fams = list(FAMILIES)
    sides = [h["side"] for h in halves]
    r = np.array([h["radius"] for h in halves])
    # For each top height on the grid, each family's best level and bend on each half.
    search = []
    for top in FIT["top_grid"]:
        row = {}
        for h in halves:
            row[h["side"]] = {}
            for fam in fams:
                d = np.abs(h["g"] - top * tables[fam]) @ _W
                i, j = np.unravel_index(int(np.argmin(d)), d.shape)
                row[h["side"]][fam] = (float(d[i, j]), float(FIT["level_grid"][i]),
                                       float(FAMILIES[fam]["grid"][j]))
        search.append(row)
    found = []
    for pairing in product(fams, repeat=len(halves)):
        scores = [float(r @ [row[s][f][0] for s, f in zip(sides, pairing)]) for row in search]
        i = int(np.argmin(scores))
        row = search[i]
        x0 = np.array([FIT["top_grid"][i]] + [v for s, f in zip(sides, pairing) for v in row[s][f][1:]])
        bounds = [(FIT["top_min"], 1.0)] + [b for f in pairing
                                            for b in ((0.0, FIT["level_max"]), FAMILIES[f]["bounds"])]

        def cost(v, pairing=pairing):
            return _out_of_place(halves, v[0], {s: (f, v[1 + 2 * j], v[2 + 2 * j])
                                                for j, (s, f) in enumerate(zip(sides, pairing))})

        lo, hi = np.array([b[0] for b in bounds]), np.array([b[1] for b in bounds])
        steps = np.r_[0.05, [0.05] * (len(x0) - 1)]
        simplex = np.array([x0] + [np.clip(x0 + np.eye(len(x0))[j] * steps[j], lo, hi)
                                   for j in range(len(x0))])
        res = minimize(cost, x0, method="Nelder-Mead", bounds=bounds,
                       options={"initial_simplex": simplex, "xatol": 1e-5, "fatol": 1e-7,
                                "maxiter": 3000})
        v, d = (res.x, float(res.fun)) if res.fun <= cost(x0) else (x0, cost(x0))
        found.append((d, tuple(fams.index(f) for f in pairing), pairing, v))
    least = min(t[0] for t in found)
    d, _, pairing, v = min((t for t in found if t[0] <= least + FIT["tie"]),
                           key=lambda t: (sum(t[1]), t[1], t[0]))
    return {
        "top": float(v[0]),
        "sides": {s: (f, float(v[1 + 2 * j]), float(v[2 + 2 * j])) for j, (s, f) in enumerate(zip(sides, pairing))},
        "departure": d,
    }


# =====================================================================
# Reading the fit
# =====================================================================
# Landmark shapes, from the most sagging to the fullest, used only to put the
# fitted ideal into words: the ideal takes the name of the landmark whose share
# of its box under the curve is closest to its own.
PERFECT_SHAPES = {
    "strongly_concave": {
        "name": "Strongly concave", "g": lambda u: superellipse_curve(u, 0.5),
        "names": {HILL: "strongly concave cone", VALLEY: "slot gorge (convex walls)",
                  SLOPE: "strongly concave"},
    },
    "concave": {
        "name": "Concave parabola", "g": lambda u: parabola_curve(u, 1.0),
        "names": {HILL: "concave-sided cone", VALLEY: "V with flaring walls",
                  SLOPE: "concave (graded)"},
    },
    "gently_concave": {
        "name": "Gently concave", "g": lambda u: parabola_curve(u, 0.5),
        "names": {HILL: "gently concave cone", VALLEY: "V with gently flaring walls",
                  SLOPE: "gently concave"},
    },
    "straight": {
        "name": "Straight", "g": lambda u: parabola_curve(u, 0.0),
        "names": {HILL: "straight-sided cone", VALLEY: "V-shaped", SLOPE: "straight"},
    },
    "gently_convex": {
        "name": "Gently convex", "g": lambda u: parabola_curve(u, -0.5),
        "names": {HILL: "gently convex cone", VALLEY: "rounded V", SLOPE: "gently convex"},
    },
    "parabola": {
        "name": "Parabola", "g": lambda u: parabola_curve(u, -1.0),
        "names": {HILL: "parabolic dome", VALLEY: "parabolic U", SLOPE: "convex (parabolic)"},
    },
    "semicircle": {
        "name": "Semicircle", "g": lambda u: superellipse_curve(u, 2.0),
        "names": {HILL: "rounded dome", VALLEY: "rounded U", SLOPE: "strongly convex"},
    },
    "box": {
        "name": "Box", "g": lambda u: superellipse_curve(u, 4.0),
        "names": {HILL: "box / mesa", VALLEY: "box trough", SLOPE: "plateau edge"},
    },
}

# Departure grades, the analogue of Powers' roundness classes. PROVISIONAL:
# set from the profiles that ship with the app, not a calibrated population.
DEPARTURE_CLASSES = (
    (0.01, "near-ideal"),
    (0.025, "slightly modified"),
    (0.05, "modified"),
    (float("inf"), "strongly modified"),
)

# A level stretch across at least this share of the width is named in words
# ("with a flat top"). It changes the name only, never the fit or the number.
LEVEL_NAMED = 0.10


def box_share(g) -> float:
    """Share of the unit box under a half profile: 1/2 for a straight side, 2/3 for a dome parabola."""
    return float(np.clip(np.asarray(g, float), 0.0, 1.0) @ _W)


def landmark_for(share: float) -> str:
    """The landmark shape whose share of its box is closest to `share`."""
    return min(PERFECT_SHAPES, key=lambda k: abs(box_share(PERFECT_SHAPES[k]["g"](U)) - share))


def compare(halves: list, ideal: dict, kind: str) -> dict:
    """Real minus ideal, as rock: positive is material the ideal does not have.

    Each half is weighted by its own box (width times height), so the areas are
    shares of the landform's box: size-independent and comparable between shapes.
    """
    rock = -1.0 if kind == VALLEY else 1.0
    excess = deficit = box = 0.0
    res = {}
    for h in halves:
        r = rock * (h["g"] - ideal[h["side"]])
        res[h["side"]] = r
        w = h["radius"] * h["height"]
        excess += w * float(np.clip(r, 0.0, None) @ _W)
        deficit += w * float(np.clip(-r, 0.0, None) @ _W)
        box += w
    if box <= 0:
        return {}
    return {
        "excess": excess / box,
        "deficit": deficit / box,
        "departure": (excess + deficit) / box,
        "net": (excess - deficit) / box,
        "residual": res,
    }


# =====================================================================
# The fixed triangle
#
# The best fit can bend its sides, hold them level and lower its top, so with
# enough freedom it might in the end match any profile. The fixed triangle is
# the answer to that: one rigid ideal with nothing fitted. Its base is the line
# between the two feet at the base level, its apex stands at the landform's
# full height midway between the feet, and its sides are straight, so it is
# isosceles in real distance. A summit off the middle counts as rock out of
# place, just as a bent side does. A long profile has one foot, so its ideal is
# one side of the triangle: a straight line from the head to the mouth. The
# rock out of place is measured exactly as for the best fit (compare), as a
# share of the same box.
#
# Landforms lie between two end members, the triangle and the parabola: a
# perfect parabola centred between its feet, bowed in or domed, leaves 1/6 of
# its box out of place against the triangle.
# =====================================================================
TRIANGLE = "triangle"
TRIANGLE_PARABOLA = 1.0 / 6.0
TRIANGLE_NAMES = {HILL: "the fixed triangle", VALLEY: "the fixed triangle (a V)", SLOPE: "a straight slope"}


def triangle_departure_of(g) -> float:
    """How far a perfect shape, centred between its feet, sits from the fixed triangle: the share
    of the box between one of its halves, g(u), and the triangle's straight side."""
    return float(np.abs(np.asarray(g, dtype=float) - (1.0 - U)) @ _W)


# Perfect shapes as the fixed triangle sees them. The best fit matches each one
# exactly, so on a plot of the two departures they all sit at 0 on the best-fit
# axis, spread along the triangle axis.
TRIANGLE_LANDMARKS = {
    "perfect triangle": 0.0,
    "perfect parabola": triangle_departure_of(parabola_curve(U, 1.0)),
    "quarter circle": triangle_departure_of(superellipse_curve(U, 2.0)),
}


def triangle_ideal(m: dict, halves: list) -> dict:
    """The fixed triangle on each half's own grid: 1 midway between the feet, 0 at each foot."""
    if len(halves) == 1:
        return {halves[0]["side"]: 1.0 - U}
    mid = 0.5 * (m["x_left"] + m["x_right"])
    half = 0.5 * (m["x_right"] - m["x_left"])
    return {h["side"]: 1.0 - np.abs(m["x_apex"] + h["sign"] * U * h["radius"] - mid) / half
            for h in halves}


def as_triangle(item: dict) -> dict:
    """The same profile read against the fixed triangle instead of its best fit.

    The frame and the halves are unchanged; the ideal and the numbers are the
    triangle's. There is no grade: the provisional grades were set on the best fit.
    """
    if not item.get("ok"):
        return item
    t = item["triangle"]
    return {**item, **{k: t[k] for k in ("ideal", "excess", "deficit", "departure", "net", "residual")},
            "against": TRIANGLE, "name": TRIANGLE_NAMES[item["kind"]], "grade": ""}


def departure_class(d: float) -> str:
    if not np.isfinite(d):
        return "-"
    for limit, label in DEPARTURE_CLASSES:
        if d < limit:
            return label
    return DEPARTURE_CLASSES[-1][1]


def half_words(fam: str, value: float) -> str:
    """One half's curve, e.g. 'parabola k = +0.61'."""
    spec = FAMILIES[fam]
    fmt = "{:+.2f}" if fam != "superellipse" else "{:.2f}"
    return f"{spec['name'].lower()} {spec['param']} = {fmt.format(value)}"


def fit_words(item: dict) -> str:
    """The fitted curve on each half: 'parabola, k = +0.61 / +0.58' when both halves
    take the same family, 'exponential a = +1.89 / superellipse e = 0.63' when not."""
    sides = item["fit"]["sides"]
    fams = [f for f, _, _ in sides.values()]
    if len(set(fams)) == 1:
        spec = FAMILIES[fams[0]]
        fmt = "{:+.2f}" if fams[0] != "superellipse" else "{:.2f}"
        parts = " / ".join(fmt.format(v) for _, _, v in sides.values())
        return f"{spec['name'].lower()}, {spec['param']} = {parts}"
    return " / ".join(half_words(f, v) for f, _, v in sides.values())


def frame(x, z, kind: str = HILL, ridge_axis: float | None = None) -> tuple[dict, list, list, str]:
    """Step 1 of the procedure on a clean profile: (metrics, notes, halves, problem).

    `problem` is empty when the profile was framed.
    """
    if kind == SLOPE:
        m, notes, halves = analyse_slope(x, z)
    elif ridge_axis is not None:
        m, notes = locate_ridge(x, z, ridge_axis)
        halves = half_profiles(x, z, m, kind) if m else []
    else:
        m, notes = analyse_profile(x, z, kind)
        halves = half_profiles(x, z, m, kind) if m else []
    if not m:
        return {}, [], [], notes[0] if notes else "This profile could not be measured."
    if kind != SLOPE:
        lw, rw = m["x_apex"] - m["x_left"], m["x_right"] - m["x_apex"]
        if max(lw, rw) <= 0 or min(lw, rw) < PROCESSING["min_side_ratio"] * max(lw, rw):
            return {}, [], [], (
                f"The {VOCAB[kind]['apex']} is at, or very near, one end of the profile, so the line "
                f"does not cross the {VOCAB[kind]['feature']}. Use a line that runs past it on both "
                "sides, or choose 'One slope'."
            )
    return m, notes, halves, ""


def measure(x, z, kind: str = HILL, ridge_axis: float | None = None) -> dict:
    """One profile through the fixed procedure. Never raises.

    With ridge_axis, a spreading ridge is framed from that axis (locate_ridge)
    instead of from its highest point; the fit and the number are the same.
    """
    item = {"kind": kind, "ok": False, "notes": [], "problem": ""}
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    ok = np.isfinite(x) & np.isfinite(z)
    x, z = x[ok], z[ok]
    if len(x) < 32:
        item["problem"] = "At least 32 points are needed to fit a profile."
        return item
    order = np.argsort(x, kind="stable")
    x, z = x[order], z[order]
    keep = np.concatenate([[True], np.diff(x) > 0])
    x, z = x[keep], z[keep]
    if np.ptp(z) <= 0:
        item["problem"] = "Every elevation is the same, so there is no relief to fit."
        return item

    # Step 1: the frame.
    m, notes, halves, problem = frame(x, z, kind, ridge_axis)
    item["notes"] = list(notes)
    if problem:
        item["problem"] = problem
        item["notes"] = []
        return item

    # Step 2: the ideal.
    fit = fit_ideal(halves)
    ideal = {s: fit["top"] * ideal_half(f, v, c) for s, (f, c, v) in fit["sides"].items()}

    # Step 3: the number.
    comp = compare(halves, ideal, kind)

    # The fixed triangle on the same frame, measured the same way.
    tri = triangle_ideal(m, halves)
    offset = (m["x_apex"] - 0.5 * (m["x_left"] + m["x_right"])) / (m["x_right"] - m["x_left"]) \
        if kind != SLOPE else 0.0
    triangle = {"ideal": tri, "offset": float(offset), **compare(halves, tri, kind)}

    # Words for the ideal: the landmark nearest its curves (below any level
    # stretch), and the level stretch when it is a real part of the shape.
    r = {h["side"]: h["radius"] for h in halves}
    rs = sum(r.values())
    share = sum(r[s] * box_share(family_curve(f, v)) for s, (f, _, v) in fit["sides"].items()) / rs
    level = sum(r[s] * c for s, (_, c, _) in fit["sides"].items()) / rs
    landmark = landmark_for(share)
    flat = kind != SLOPE and level >= LEVEL_NAMED
    name = PERFECT_SHAPES[landmark]["names"][kind] + (f" with a {VOCAB[kind]['flat']}" if flat else "")
    item.update(
        ok=True, x=x, z=z, m=m, halves=halves, fit=fit, ideal=ideal, **comp, triangle=triangle,
        landmark=landmark, flat=flat, level=level, name=name,
        grade=departure_class(comp["departure"]),
    )
    if m["width_height"] < 1.0 or m["width_height"] > 200.0:
        item["notes"].append(
            f"The {VOCAB[kind]['feature']} is {m['width_height']:.1f} times as wide as it is high, "
            "which is unusual. Check that distance and elevation are in the same units."
        )
    return item


def average(items: list) -> dict:
    """Step 4: the departures of several transects of one landform, averaged.

    Returns n (transects fitted), mean, sd (sample standard deviation; 0 for
    one transect), lo and hi (the range). Transects that could not be fitted
    are left out and counted in `failed`.
    """
    d = np.array([it["departure"] for it in items if it.get("ok")], dtype=float)
    out = {"n": int(len(d)), "failed": int(len(items) - len(d))}
    if len(d) == 0:
        return {**out, "mean": float("nan"), "sd": float("nan"), "lo": float("nan"), "hi": float("nan")}
    return {**out, "mean": float(d.mean()), "sd": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "lo": float(d.min()), "hi": float(d.max())}


def drawn_halves(item: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(distance, real elevation, ideal elevation) across both halves, left to right."""
    m = item["m"]
    xs, zr, zi = [], [], []
    for h in item["halves"]:
        x = m["x_apex"] + h["sign"] * U * h["radius"]
        z_real = h["base"] + h["g"] * h["height"]
        z_ideal = h["base"] + item["ideal"][h["side"]] * h["height"]
        if item["kind"] == VALLEY:
            z_real, z_ideal = -z_real, -z_ideal
        if h["sign"] < 0:
            x, z_real, z_ideal = x[::-1], z_real[::-1], z_ideal[::-1]
        xs.append(x)
        zr.append(z_real)
        zi.append(z_ideal)
    order = np.argsort([a[0] for a in xs])
    return (np.concatenate([xs[i] for i in order]), np.concatenate([zr[i] for i in order]),
            np.concatenate([zi[i] for i in order]))


def summary_row(item: dict, label: str = "", group: str = "") -> dict:
    """One flat row for the shape chart and the table."""
    if not item.get("ok"):
        return {}
    side_name = {"L": "left", "R": "right", "D": "downslope"}
    row = {
        "Label": label,
        "Group": group,
        "Feature": VOCAB[item["kind"]]["feature"],
        "Shape": item["name"],
        "Departure": item["departure"],
        "Grade": item["grade"],
        "Extra rock": item["excess"],
        "Missing rock": item["deficit"],
        "Ideal": "fixed triangle" if item.get("against") == TRIANGLE else "best-fit curve",
    }
    if item.get("against") == TRIANGLE:
        row["Summit off the middle (share of width)"] = item["triangle"]["offset"]
    else:
        row["Ideal top (share of height)"] = item["fit"]["top"]
        for s, (f, c, v) in item["fit"]["sides"].items():
            row[f"Ideal curve ({side_name[s]})"] = half_words(f, v)
            row[f"Level stretch ({side_name[s]})"] = c
    row.update({
        "Width / height": item["m"]["width_height"],
        "Asymmetry": item["m"]["asymmetry"],
    })
    return row


# =====================================================================
# Mid-ocean ridges
#
# A spreading ridge is read twice by the same fixed procedure, at two scales:
#
#   the WHOLE RIDGE, the full profile as a hill, framed from the axis
#   (locate_ridge), not from its highest point: on a slow ridge the highest
#   point is a rift mountain to one side, and a frame set there reads only
#   that mountain. Away from the axis the sea floor deepens with the square
#   root of its age (half-space cooling), so the ridge is a broad concave rise.
#   That subsidence is symmetric and is the ridge itself, not a tilt under it,
#   so it is fitted, not removed. The ideal is fitted exactly as on any other
#   landform; a rift valley cut into the crest is missing rock.
#
#   the AXIS, the ground within axis_window of the spreading axis, read both
#   as a hill and as a valley. Whichever reading puts its apex on the axis
#   says what the axis is: an axial high (its summit is on the axis) or an
#   axial rift valley (its floor is). The other reading lands off the axis, on
#   a rift mountain or in a trough on the flank.
#
# On 27 swath-median ETOPO1 profiles across 9 ridges from 14 to 145 mm/yr
# (three parallel profiles each, 20 km apart along the axis), every profile
# across a slow or ultraslow ridge (under 40 mm/yr) had an axial valley, its
# crest split 30 to 72 km wide, and every profile across an intermediate or
# fast ridge had an axial high. The axial highs split by the share of their
# box the ground fills: intermediate ridges (56 and 70 mm/yr) 0.45 to 0.62, a
# low straight-sided to dome-shaped rise; fast ridges (90 to 145 mm/yr) 0.27
# to 0.34, a peaked concave rise. Removing a fitted square-root-of-age ramp
# before reading the axis blurred that split, so the profile is read as it is.
# The limit was set on these same profiles; the high-or-valley reading was not
# tuned.
#
# PROVISIONAL: the class limits come from those 9 ridges at about 2 km
# resolution, not from a calibrated population, and hotspot-influenced
# ridges (Reykjanes, Galapagos) were not tested.
# =====================================================================
RIDGE = {
    # The axis zone read as a hill and as a valley, either side of the axis (metres).
    "axis_window": 40_000.0,
    # An apex this close to the axis is on it (metres).
    "on_axis": 10_000.0,
    # Axial highs whose ground fills more than this share of their box read as
    # intermediate (a rounded rise), less as fast (a peaked rise).
    "fast_share": 0.40,
}

RIDGE_CLASSES = {
    "slow": "slow or ultraslow (under about 40 mm/yr)",
    "intermediate": "intermediate (about 50 to 80 mm/yr)",
    "fast": "fast (over about 80 mm/yr)",
    "unclear": "unclear",
}


def locate_ridge(x, z, axis: float) -> tuple[dict, list]:
    """Apex, crest and feet of a spreading ridge, measured out from its axis.

    The highest ground within axis_window on each side is that side's crest
    (on a slow ridge, the rift mountain; on a fast one, the axial high itself).
    The apex is the axis, at the level of the higher crest, so the two flanks
    are read as one ridge even where a rift valley splits the crest. The feet
    follow the fixed rule: both at the higher of the two sides' bases, at the
    first crossing going out from each crest.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    xs, zs = _resample(x, z, float(x.min()), float(x.max()))
    w = RIDGE["axis_window"]
    left = np.flatnonzero((xs >= axis - w) & (xs <= axis))
    right = np.flatnonzero((xs >= axis) & (xs <= axis + w))
    if len(left) < 2 or len(right) < 2 or left[0] < 4 or right[-1] > len(xs) - 5:
        return {}, ["The axis is too close to an end of the profile. Centre the profile on the "
                    "spreading axis, at distance 0, with at least 50 km on each side."]
    kl = int(left[np.argmax(zs[left])])
    kr = int(right[np.argmax(zs[right])])
    z_top = float(max(zs[kl], zs[kr]))
    q = PROCESSING["base_quantile"]
    base_l = float(np.quantile(zs[: kl + 1], q))
    base_r = float(np.quantile(zs[kr:], q))
    z_base = max(base_l, base_r)
    height = z_top - z_base
    if height <= 0:
        return {}, ["There is no relief between the crest of the ridge and its foot."]

    def crossing(k: int, step: int):
        i = k
        while 0 <= i < len(zs) and zs[i] > z_base:
            i += step
        return float(xs[min(max(i, 0), len(xs) - 1)])

    xl, xr = crossing(kl, -1), crossing(kr, 1)
    width = xr - xl
    notes = []
    if abs(zs[kl] - zs[kr]) > 0.25 * height:
        notes.append(
            f"The two crests stand {abs(zs[kl] - zs[kr]) / height:.0%} of the height apart; the ridge "
            "is read from the higher one, so the lower crest shows as missing rock."
        )
    depth = np.clip(zs - z_base, 0.0, None)
    lm, rm = (xs >= xl) & (xs <= axis), (xs >= axis) & (xs <= xr)
    a_l = float(np.trapezoid(depth[lm], xs[lm])) if lm.sum() > 1 else 0.0
    a_r = float(np.trapezoid(depth[rm], xs[rm])) if rm.sum() > 1 else 0.0
    return {
        "height": height,
        "width": width,
        "width_height": width / height,
        "plateau": float(xs[kr] - xs[kl]) / width,
        "asymmetry": abs(a_r - a_l) / (a_r + a_l) if (a_r + a_l) > 0 else float("nan"),
        "apex_elev": z_top,
        "base_elev": z_base,
        "base_left": z_base,
        "base_right": z_base,
        "x_left": xl,
        "x_right": xr,
        "x_apex": float(axis),
        "x_flat_left": float(xs[kl]),
        "x_flat_right": float(xs[kr]),
    }, notes


def ridge_axis_x(x) -> float:
    """The spreading axis: distance 0 when it lies in the middle half of the profile, else the middle."""
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    q1, q3 = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
    return 0.0 if q1 <= 0.0 <= q3 else 0.5 * (lo + hi)


def ground_share(fr: dict) -> float:
    """Share of its box the measured ground fills, both halves weighted by width."""
    hs = fr["halves"]
    return sum(h["radius"] * box_share(h["g"]) for h in hs) / sum(h["radius"] for h in hs)


def read_ridge(x, z, axis: float | None = None) -> dict:
    """A spreading ridge by the fixed procedure, read whole and at its axis. Distances in metres.

    Returns the whole-ridge measurement, the axis position, what the axis is
    (axial high, axial valley) and the provisional spreading-rate class.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    ax = ridge_axis_x(x) if axis is None else float(axis)
    near = np.abs(x - ax) <= RIDGE["axis_window"]
    order = np.argsort(x[near], kind="stable")
    xn, zn = x[near][order], z[near][order]
    hill = dict(zip(("m", "notes", "halves", "problem"), frame(xn, zn, HILL)))
    valley = dict(zip(("m", "notes", "halves", "problem"), frame(xn, zn, VALLEY)))

    def on_axis(fr):
        return not fr["problem"] and abs(fr["m"]["x_apex"] - ax) <= RIDGE["on_axis"]

    hi, lo = on_axis(hill), on_axis(valley)
    if hi and lo:
        # Both apexes on the axis: the bigger feature is the axis.
        hi = hill["m"]["height"] >= valley["m"]["height"]
        lo = not hi
    if lo:
        axis_type, rate = "axial valley", "slow"
    elif hi:
        axis_type = "axial high"
        rate = "intermediate" if ground_share(hill) >= RIDGE["fast_share"] else "fast"
    else:
        axis_type, rate = "no clear axial feature", "unclear"
    return {
        "axis": ax,
        "whole": measure(x, z, HILL, ridge_axis=ax),
        "axis_type": axis_type,
        "rate_class": rate,
        "rate_words": RIDGE_CLASSES[rate],
    }
