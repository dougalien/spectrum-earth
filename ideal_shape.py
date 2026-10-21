"""
Ideal-shape fitting for topographic profiles.

The premise is Wadell's grain construction. Wadell (1932, 1935) fitted a sand
grain with a perfect sphere and measured the material left over: a fresh grain
departs from the sphere in one way, a weathered one in another, and how far it
departs is the reading. Here the grain is a landform and the sphere is a family
of perfect curves.

THE HALF PROFILE. A profile is split at its apex (the summit of a hill, the
floor of a valley) into two halves. Each half is stretched onto its OWN width
and the height of the feature, so it runs from u = 0 at the apex to u = 1 at
the foot (the base of a hill, the rim of a valley), and from g = 1 at the apex
to g = 0 at the foot. An off-centre summit is therefore not straightened out:
the two halves keep their own widths and the asymmetry stays visible.

A valley is a hill with earth and sky exchanged. It is measured upside down, so
one engine serves both, and a river long profile is a single half on its own.

THE IDEAL CURVES. One continuous family, the superellipse

    g = (1 - u^e) ^ (1/e)

with a second control, the cut, that slices the apex flat (a flat top or a flat
floor). e alone sets the curvature: e = 1 is straight-sided (a cone or a V),
e = 2 is a semicircle in the normalised frame (a semi-ellipse in real units),
e < 1 sags inward and e > 2 approaches a box. The named perfect shapes are
members of the family, which is why the family is used rather than a set of
unrelated outlines. The circle in TRUE proportions, the literal Wadell sphere,
is offered as a separate reference because it depends on the aspect of the
landform.

THE DEPARTURE. Each half is compared with the ideal and the difference split
into rock ABOVE the ideal (excess) and rock MISSING relative to it (deficit),
as fractions of the ideal's own area, so every number is size-independent:

    departure = (excess + deficit) / A_ideal     how far from the ideal
    net       = (excess - deficit) / A_ideal     which way, on balance

The sign convention is about rock for every feature type: positive means more
material than the perfect shape, whether that material stands on a hill or
fills a valley.

The measuring engine (PROCESSING ... analyse_profile, half_profiles) is ported
from Spectrum Jax, where it was validated on real volcano transects. Nothing in
this module touches Streamlit, so it can be tested on its own.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
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
}

VOCAB = {
    HILL: {"feature": "hill", "apex": "summit", "flat": "flat top", "side": "flank",
           "sides": "flanks", "foot": "foot", "zones": ("summit", "mid-flank", "foot")},
    VALLEY: {"feature": "valley", "apex": "valley floor", "flat": "flat floor", "side": "wall",
             "sides": "walls", "foot": "rim", "zones": ("floor", "mid-wall", "rim")},
    SLOPE: {"feature": "profile", "apex": "head", "flat": "flat head", "side": "slope",
            "sides": "slope", "foot": "mouth", "zones": ("head", "middle", "mouth")},
}

U = np.linspace(0.0, 1.0, 401)
ZONE_EDGES = (0.15, 0.60)


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
    t = int(round(0.5 * (lo_s + hi_s)))
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
    z_base = max(base_l, base_r)
    height = z_top - z_base
    if height <= 0:
        return {}, ["There is no relief between the apex and the foot."]

    if abs(base_l - base_r) > 0.25 * height:
        notes.append(
            f"The two {w['sides']} reach their {w['foot']} {abs(base_l - base_r) / height:.0%} of the "
            f"height apart. The nearer {w['foot']} level is used, so the other {w['side']} is measured "
            "only as far as that level."
        )

    def crossing(direction: int):
        i = t
        while 0 <= i < len(z):
            if z[i] <= z_base:
                break
            i += direction
        if i < 0 or i >= len(z):
            return None
        j = i - direction
        if not (0 <= j < len(z)) or z[j] == z[i]:
            return float(x[i])
        f = (z_base - z[i]) / (z[j] - z[i])
        return float(x[i] + f * (x[j] - x[i]))

    xl, xr = crossing(-1), crossing(+1)
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
        "x_left": xl,
        "x_right": xr,
        "x_apex": x_apex,
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
    """The two halves of a measured profile, each on its own width: g(u), u from apex to foot."""
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    zo = -z if kind == VALLEY else z
    w = m["width"]
    a = max(float(np.min(x)), m["x_left"] - PROCESSING["window_margin"] * w)
    b = min(float(np.max(x)), m["x_right"] + PROCESSING["window_margin"] * w)
    xc, zc = _resample(x, zo, a, b)
    xa = m["x_apex"]
    out = []
    for side, radius, sign in (("L", xa - m["x_left"], -1.0), ("R", m["x_right"] - xa, 1.0)):
        zz = np.interp(xa + sign * U * radius, xc, zc)
        out.append({"side": side, "radius": float(radius), "sign": sign,
                    "g": np.clip((zz - m["base_elev"]) / m["height"], 0.0, None)})
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
    g = np.clip((zz - z_mouth) / height, 0.0, None)
    m = {
        "height": height,
        "width": length,
        "width_height": length / height,
        "plateau": 0.0,
        "asymmetry": float("nan"),
        "apex_elev": z_head,
        "base_elev": z_mouth,
        "x_left": min(x_head, x_mouth),
        "x_right": max(x_head, x_mouth),
        "x_apex": x_head,
    }
    return m, notes, [{"side": "D", "radius": float(length), "sign": sign, "g": g}]


# =====================================================================
# The perfect shapes
#
# Each is a height fraction g(u) of a half profile: g = 1 at the apex, 0 at the
# foot, u = 0 at the apex, 1 at the foot. Most are members of one family,
#
#     g = (1 - u^p) ^ (1/q),
#
# where p shapes the apex (p = 1 a sharp point, p > 1 rounded) and q shapes the
# foot (q = 1 meets the ground at an angle, q > 1 steepens into it, q < 1 flares
# out along it). p = q is the superellipse - the cone, the semicircle and the box
# - and q = 1 is the power law, which holds the parabola.
# =====================================================================
def ideal_curve(u, p: float, q: float | None = None, cut: float = 0.0) -> np.ndarray:
    """A member of the family, with the share `cut` of the height sliced off the apex."""
    p = max(float(p), 0.05)
    q = p if q is None else max(float(q), 0.05)
    g = np.clip(1.0 - np.clip(np.asarray(u, float), 0.0, 1.0) ** p, 0.0, None) ** (1.0 / q)
    c = float(np.clip(cut, 0.0, 0.95))
    return np.minimum(g, 1.0 - c) / (1.0 - c)


def true_circle(u, aspect: float) -> np.ndarray:
    """The circle in true proportions: level at the apex and passing through the foot.

    aspect is height / half-width. Landforms are far wider than they are high,
    so in the normalised frame the true circle is a shallow arc, close to a
    parabola. Steeper than 45 degrees no such arc exists without overhang, and
    the semicircle of the normalised frame stands in.
    """
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    a = float(aspect)
    if not np.isfinite(a) or a <= 0 or a >= 1.0:
        return ideal_curve(u, 2.0)
    rho = (1.0 + a * a) / (2.0 * a)
    return np.clip((a - rho + np.sqrt(np.maximum(rho * rho - u * u, 0.0))) / a, 0.0, 1.0)


# The perfect shapes, from the most sagging to the fullest. "names" gives what
# each is called on a hill, a valley and a one-sided slope. Fullness is the
# share of the box each shape fills, the form axis of the shape chart.
PERFECT_SHAPES = {
    "strongly_concave": {
        "name": "Strongly concave", "p": 0.5, "q": 0.5,
        "names": {HILL: "strongly concave cone", VALLEY: "slot gorge (convex walls)",
                  SLOPE: "strongly concave"},
        "note": "A parabola tangent to both the axis and the ground: a needle-like apex and a long "
                "apron flaring out along the ground.",
    },
    "concave": {
        "name": "Concave parabola", "p": 1.0, "q": 0.5,
        "names": {HILL: "concave-sided cone", VALLEY: "V with flaring walls",
                  SLOPE: "concave (graded)"},
        "note": "A parabola with its vertex at the foot: straight into the apex, flaring out "
                "along the ground. The stratovolcano flank, and the graded river.",
    },
    "straight": {
        "name": "Straight", "p": 1.0, "q": 1.0,
        "names": {HILL: "straight-sided cone", VALLEY: "V-shaped", SLOPE: "straight"},
        "note": "A perfect cone or a perfect V.",
    },
    "parabola": {
        "name": "Parabola", "p": 2.0, "q": 1.0,
        "names": {HILL: "parabolic dome", VALLEY: "parabolic U", SLOPE: "convex (parabolic)"},
        "note": "Rounded apex, meeting the ground at an angle. The classic glacial trough.",
    },
    "circle": {
        "name": "Circle (true proportions)",
        "names": {HILL: "circular dome", VALLEY: "circular bowl", SLOPE: "convex (circular)"},
        "note": "Wadell's sphere: a real circle, level at the apex and passing through the foot, "
                "drawn without vertical exaggeration. On a landform much wider than it is high it is "
                "a shallow arc, close to the parabola.",
    },
    "semicircle": {
        "name": "Semicircle", "p": 2.0, "q": 2.0,
        "names": {HILL: "semi-elliptical dome", VALLEY: "semi-elliptical U",
                  SLOPE: "strongly convex"},
        "note": "A semicircle once height and width are scaled to the same box, which is a "
                "semi-ellipse in real units. Its sides meet the ground vertically.",
    },
    "box": {
        "name": "Box", "p": 4.0, "q": 4.0,
        "names": {HILL: "box / mesa", VALLEY: "box trough", SLOPE: "plateau edge"},
        "note": "Near-vertical sides with a rounded shoulder.",
    },
}
BEST_FIT = "best"
BEST_FIT_NAME = "Best-fitting smooth curve"

# Departure classes, the analogue of Powers' roundness classes: mean distance
# from the nearest perfect shape, as a share of the feature's height.
# PROVISIONAL - set from the synthetic and real profiles that ship with the
# app, not from a calibrated population.
DEPARTURE_CLASSES = (
    (0.02, "near-ideal"),
    (0.04, "slightly modified"),
    (0.08, "modified"),
    (float("inf"), "strongly modified"),
)


def perfect_curve(key: str, aspect: float = float("nan")) -> np.ndarray:
    s = PERFECT_SHAPES[key]
    if key == "circle":
        return true_circle(U, aspect)
    return ideal_curve(U, s["p"], s["q"])


def fullness(g) -> float:
    """Share of the unit box under a half profile: 0.5 for a cone, 2/3 for a parabola."""
    return float(np.trapezoid(np.clip(np.asarray(g, float), 0.0, 1.0), U))


def fit_smooth(halves: list, fit_cut: bool = True) -> dict:
    """The smooth member of the family (apex p, foot q, flat top cut) closest to the halves.

    This is NOT a perfect shape: it has three free controls, so it follows the
    gross form wherever it goes. What it leaves behind is the small-scale
    irregularity - the roundness of grain work, where departure from the
    nearest perfect shape is the sphericity.
    """
    # Weighted by width, as the departure is, so a narrow half does not count
    # for as much as a wide one.
    wts = [np.sqrt(max(h["radius"], 0.0)) for h in halves]

    def resid(v):
        c = v[2] if fit_cut else 0.0
        g = ideal_curve(U, v[0], v[1], c)
        return np.concatenate([w * (g - h["g"]) for w, h in zip(wts, halves)])

    best = None
    for p0, q0 in ((0.5, 0.5), (1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (4.0, 4.0)):
        r = least_squares(resid, [p0, q0, 0.02 if fit_cut else 0.0],
                          bounds=([0.2, 0.2, 0.0], [8.0, 8.0, 0.7 if fit_cut else 1e-9]))
        if best is None or r.cost < best.cost:
            best = r
    return {"p": float(best.x[0]), "q": float(best.x[1]), "cut": float(best.x[2]) if fit_cut else 0.0}


def compare(halves: list, ideal: dict, kind: str) -> dict:
    """Real minus ideal, as rock: positive is material the perfect shape does not have.

    Areas are shares of the box the feature fits in (its width times its
    height), so they are size-independent and comparable between shapes.
    """
    rock = -1.0 if kind == VALLEY else 1.0
    excess = deficit = box = 0.0
    res = {}
    for h in halves:
        r = rock * (h["g"] - ideal[h["side"]])
        res[h["side"]] = r
        w = h["radius"]
        excess += w * float(np.trapezoid(np.clip(r, 0.0, None), U))
        deficit += w * float(np.trapezoid(np.clip(-r, 0.0, None), U))
        box += w
    if box <= 0:
        return {}
    mean_r = np.mean([res[h["side"]] for h in halves], axis=0)
    names = VOCAB[kind]["zones"]
    masks = (U < ZONE_EDGES[0], (U >= ZONE_EDGES[0]) & (U < ZONE_EDGES[1]), U >= ZONE_EDGES[1])
    # Signed mean per zone, and the extra and missing parts separately: within
    # one zone the two can cancel, and both are part of the reading.
    zones = {n: float(mean_r[k].mean()) for n, k in zip(names, masks)}
    zones_extra = {n: float(np.clip(mean_r[k], 0.0, None).mean()) for n, k in zip(names, masks)}
    zones_missing = {n: float(np.clip(-mean_r[k], 0.0, None).mean()) for n, k in zip(names, masks)}
    return {
        "excess": excess / box,
        "deficit": deficit / box,
        "departure": (excess + deficit) / box,
        "net": (excess - deficit) / box,
        "zones": zones,
        "zones_extra": zones_extra,
        "zones_missing": zones_missing,
        "residual": res,
    }


def departure_class(d: float) -> str:
    if not np.isfinite(d):
        return "-"
    for limit, label in DEPARTURE_CLASSES:
        if d < limit:
            return label
    return DEPARTURE_CLASSES[-1][1]


def zone_words(comp: dict) -> str:
    """Where the real profile carries more or less rock than the ideal.

    Each zone's figure is the mean over that zone of the extra (or missing)
    rock, as a share of the height.
    """
    found = [(k, v) for k, v in comp["zones_extra"].items() if v >= 0.02]
    found += [(k, -v) for k, v in comp["zones_missing"].items() if v >= 0.02]
    if not found:
        return "It stays within 2% of its height of this shape everywhere."
    found.sort(key=lambda kv: -abs(kv[1]))
    parts = [f"{'more' if v > 0 else 'less'} rock at the {k} ({v * 100:+.0f}% of the height)"
             for k, v in found]
    return "Compared with this shape it has " + ", ".join(parts) + "."


def shape_name(item: dict) -> str:
    """The nearest perfect shape in the feature's own words, with a flat top or floor if measured."""
    kind = item["kind"]
    name = PERFECT_SHAPES[item["nearest"]]["names"][kind]
    if kind != SLOPE and has_flat_apex(item):
        name += f" with a {VOCAB[kind]['flat']}"
    return name


def has_flat_apex(item: dict) -> bool:
    """Flat ground at the apex that the smooth fit also needs a cut for.

    The measured flat-top width alone is not enough: a rounded apex (the floor
    of a parabolic U) stays within 5% of the height of its lowest point for a
    fifth of the width without being flat at all.
    """
    cuts = [s["cut"] for s in item["smooth_by_side"].values()]
    return item["m"]["plateau"] >= 0.10 and bool(cuts) and float(np.median(cuts)) >= 0.04


def measure(x, z, kind: str = HILL) -> dict:
    """Everything for one profile. Never raises: a failure is reported in the result."""
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

    if kind == SLOPE:
        m, notes, halves = analyse_slope(x, z)
    else:
        m, notes = analyse_profile(x, z, kind)
        halves = half_profiles(x, z, m, kind) if m else []
    item["notes"] = list(notes)
    if not m:
        item["problem"] = notes[0] if notes else "This profile could not be measured."
        item["notes"] = []
        return item
    if kind != SLOPE:
        lw, rw = m["x_apex"] - m["x_left"], m["x_right"] - m["x_apex"]
        if max(lw, rw) <= 0 or min(lw, rw) < PROCESSING["min_side_ratio"] * max(lw, rw):
            item["problem"] = (
                f"The {VOCAB[kind]['apex']} is at, or very near, one end of the profile, so the line "
                f"does not cross the {VOCAB[kind]['feature']}. Use a line that runs past it on both "
                "sides, or choose 'One slope'."
            )
            item["notes"] = []
            return item

    aspect = {h["side"]: m["height"] / h["radius"] if h["radius"] > 0 else float("nan")
              for h in halves}
    comparisons = {}
    for key in PERFECT_SHAPES:
        ideal = {h["side"]: perfect_curve(key, aspect[h["side"]]) for h in halves}
        comparisons[key] = {"ideal": ideal, **compare(halves, ideal, kind)}
    # The smooth curve is fitted to each half on its own, so that what it
    # leaves behind is small-scale irregularity only, not asymmetry.
    smooth_by_side = {h["side"]: fit_smooth([h], fit_cut=kind != SLOPE) for h in halves}
    ideal = {h["side"]: ideal_curve(U, smooth_by_side[h["side"]]["p"], smooth_by_side[h["side"]]["q"],
                                    smooth_by_side[h["side"]]["cut"]) for h in halves}
    comparisons[BEST_FIT] = {"ideal": ideal, **compare(halves, ideal, kind)}

    named = [k for k in PERFECT_SHAPES if np.isfinite(comparisons[k].get("departure", np.nan))]
    nearest = min(named, key=lambda k: comparisons[k]["departure"])
    by_side = {}
    for h in halves:
        d = {k: compare([h], {h["side"]: comparisons[k]["ideal"][h["side"]]}, kind)["departure"]
             for k in PERFECT_SHAPES}
        k_side = min(d, key=d.get)
        by_side[h["side"]] = {"nearest": k_side, "departure": d[k_side], "fullness": fullness(h["g"])}
    total_w = sum(h["radius"] for h in halves)

    item.update(
        ok=True, x=x, z=z, m=m, halves=halves, aspect=aspect, comparisons=comparisons,
        smooth_by_side=smooth_by_side, nearest=nearest, by_side=by_side,
        departure=comparisons[nearest]["departure"],
        irregularity=comparisons[BEST_FIT].get("departure", float("nan")),
        fullness=sum(h["radius"] * fullness(h["g"]) for h in halves) / total_w,
    )
    item["grade"] = departure_class(item["departure"])
    item["name"] = shape_name(item)
    if m["width_height"] < 1.0 or m["width_height"] > 200.0:
        item["notes"].append(
            f"The {VOCAB[kind]['feature']} is {m['width_height']:.1f} times as wide as it is high, "
            "which is unusual. Check that distance and elevation are in the same units."
        )
    return item


def summary_row(item: dict, label: str = "", group: str = "") -> dict:
    """One flat row for the shape chart and the table."""
    if not item.get("ok"):
        return {}
    c = item["comparisons"]
    row = {
        "Label": label,
        "Group": group,
        "Feature": VOCAB[item["kind"]]["feature"],
        "Shape": item["name"],
        "Nearest perfect shape": PERFECT_SHAPES[item["nearest"]]["name"],
        "Departure": item["departure"],
        "Grade": item["grade"],
        "Net rock": c[item["nearest"]]["net"],
        "Fullness": item["fullness"],
        "Irregularity": item["irregularity"],
    }
    for key, s in PERFECT_SHAPES.items():
        row[f"Departure from {s['name']}"] = c[key]["departure"]
    for side, d in item["by_side"].items():
        row[f"Nearest ({side})"] = PERFECT_SHAPES[d["nearest"]]["name"]
        row[f"Fullness ({side})"] = d["fullness"]
    row.update({
        "Width / height": item["m"]["width_height"],
        "Flat top/floor share": item["m"]["plateau"],
        "Asymmetry": item["m"]["asymmetry"],
    })
    for side, f in item["smooth_by_side"].items():
        row[f"Smooth fit p ({side})"] = f["p"]
        row[f"Smooth fit q ({side})"] = f["q"]
        row[f"Smooth fit cut ({side})"] = f["cut"]
    return row
