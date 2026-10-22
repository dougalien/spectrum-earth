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
from scipy.optimize import least_squares, minimize_scalar
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

    def crossing(direction: int, zb: float):
        i = t
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
    for side, radius, sign, base in (("L", xa - m["x_left"], -1.0, m["base_left"]),
                                     ("R", m["x_right"] - xa, 1.0, m["base_right"])):
        zz = np.interp(xa + sign * U * radius, xc, zc)
        height = m["apex_elev"] - base
        out.append({"side": side, "radius": float(radius), "sign": sign, "base": float(base),
                    "height": float(height), "g": np.clip((zz - base) / height, 0.0, None)})
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
        "base_left": z_mouth,
        "base_right": z_mouth,
        "x_left": min(x_head, x_mouth),
        "x_right": max(x_head, x_mouth),
        "x_apex": x_head,
    }
    return m, notes, [{"side": "D", "radius": float(length), "sign": sign, "base": z_mouth,
                       "height": float(height), "g": g}]


# =====================================================================
# The ideal curves and the fitting procedure
#
# Every curve is a height fraction g(u) of a half profile: g = 1 at the apex,
# 0 at the foot, u = 0 at the apex, 1 at the foot. Every curve passes through
# both of those points, so it is pinned to the landform the way Wadell's sphere
# is pinned to the grain.
#
# THE FIXED PROCEDURE
#
#   1. Locate the apex and each foot (analyse_profile / analyse_slope). Both
#      feet sit at one level, the higher of the two sides' bases.
#   2. Normalise each half onto its own width and its own height.
#   3. Fit each of the four ideal families to each half. Every family has ONE
#      free shape control, so they compete on equal terms:
#        parabola      g = (1 - u)(1 - k u),  -1 <= k <= 1
#        exponential   g = (e^(-a u) - e^(-a)) / (1 - e^(-a))
#        circle        a real circular arc through apex and foot, drawn in true
#                      proportions; its control is the bulge (sagitta) of the arc
#        superellipse  g = (1 - u^e)^(1/e)
#      The best member is the one leaving the least rock out of place
#      (minimum departure), found by a fine grid then a bounded refinement.
#   4. The family with the smallest total departure over both halves is the
#      ideal. On a tie (within 0.1% of the height) the order above wins.
#   5. Departure, extra and missing rock are measured against that ideal.
#
# All four families contain the straight line (k = 0, a = 0, zero bulge, e = 1), so a
# straight-sided profile is never forced into a curve, and a curved profile is
# never left with a bent triangle: the parabola family holds every true parabola
# from the concave one (vertex at the foot, k = 1) to the dome or U (vertex at
# the apex, k = -1). The exponential is the concave flank that is steep at the
# apex and still sloping at the foot: the graded river long profile, and the
# flank of many stratovolcanoes, which no parabola or superellipse follows.
# =====================================================================
def ideal_curve(u, p: float, q: float | None = None, cut: float = 0.0) -> np.ndarray:
    """g = (1 - u^p)^(1/q) with the share `cut` of the height sliced off the apex.

    p = q is the superellipse. The two-exponent form is used only by the smooth
    fit that detects a flat top or floor.
    """
    p = max(float(p), 0.05)
    q = p if q is None else max(float(q), 0.05)
    g = np.clip(1.0 - np.clip(np.asarray(u, float), 0.0, 1.0) ** p, 0.0, None) ** (1.0 / q)
    c = float(np.clip(cut, 0.0, 0.95))
    return np.minimum(g, 1.0 - c) / (1.0 - c)


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
    at the foot), a < 0 convex.
    """
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    a = float(a)
    if abs(a) < 1e-6:
        return 1.0 - u
    return (np.exp(-a * u) - np.exp(-a)) / (1.0 - np.exp(-a))


def circle_curve(u, bulge: float, aspect: float) -> np.ndarray | None:
    """A circular arc through apex and foot, in true proportions.

    aspect is the half's height / width. bulge is the arc's sagitta as a share
    of the chord: positive bulges outward (more rock on a hill), negative sags
    inward, 0 is straight. Returns None where the arc would overhang or rise
    above the apex, so no such arc is ever offered as an ideal.
    """
    u = np.asarray(u, float)
    a = float(aspect)
    if not np.isfinite(a) or a <= 0:
        return None
    if abs(bulge) < 1e-6:
        return 1.0 - np.clip(u, 0.0, 1.0)
    p0 = np.array([0.0, a])
    p1 = np.array([1.0, 0.0])
    chord = float(np.hypot(1.0, a))
    mid = 0.5 * (p0 + p1)
    nrm = np.array([a, 1.0]) / chord          # outward, away from the box corner
    s = float(bulge) * chord
    radius = (0.25 * chord * chord + s * s) / (2.0 * abs(s))
    centre = mid + (s - np.sign(s) * radius) * nrm
    q = mid + s * nrm
    th0 = np.arctan2(*(p0 - centre)[::-1])
    th1 = np.arctan2(*(p1 - centre)[::-1])
    thq = np.arctan2(*(q - centre)[::-1])
    ccw = (th1 - th0) % (2 * np.pi)
    if (thq - th0) % (2 * np.pi) > ccw:     # Q is not on the ccw sweep, so go clockwise
        ccw -= 2 * np.pi
    th = th0 + np.linspace(0.0, ccw, 1601)
    xs = centre[0] + radius * np.cos(th)
    zs = centre[1] + radius * np.sin(th)
    if np.any(np.diff(xs) < -1e-9) or xs.min() < -1e-6 or xs.max() > 1.0 + 1e-6:
        return None
    if zs.max() > a * (1.0 + 1e-6) or zs.min() < -1e-6 * a:
        return None
    return np.clip(np.interp(np.clip(u, 0.0, 1.0), xs, zs) / a, 0.0, 1.0)


FAMILIES = {
    "parabola": {
        "name": "Parabola", "param": "k", "grid": np.linspace(-1.0, 1.0, 201), "log": False,
        "note": "A true parabola through the apex and the foot. k = 0 is straight, k = -1 the dome "
                "or U with its vertex at the apex, k = 1 the concave parabola with its vertex at the foot.",
    },
    "exponential": {
        "name": "Exponential", "param": "a", "grid": np.linspace(-10.0, 20.0, 301), "log": False,
        "note": "g = (e^(-a u) - e^(-a)) / (1 - e^(-a)). a = 0 is straight, a > 0 concave: steep at "
                "the apex and still sloping at the foot, like a graded river or the flank of a "
                "stratovolcano. a < 0 is convex.",
    },
    "circle": {
        "name": "Circle (true proportions)", "param": "bulge", "grid": None, "log": False,
        "note": "Wadell's sphere: a real circular arc through the apex and the foot, drawn without "
                "vertical exaggeration. Its bulge is the sagitta as a share of the chord.",
    },
    "superellipse": {
        "name": "Superellipse", "param": "e", "grid": np.geomspace(0.3, 8.0, 241), "log": True,
        "note": "g = (1 - u^e)^(1/e): e = 1 straight, e = 2 a semicircle scaled to the box, e < 1 "
                "strongly concave with a flaring apron, e > 2 toward a box.",
    },
}
STRAIGHT = "straight"
TIE = 0.001


def family_curve(fam: str, value: float, aspect: float = float("nan")):
    if fam == "parabola":
        return parabola_curve(U, value)
    if fam == "exponential":
        return exponential_curve(U, value)
    if fam == "circle":
        return circle_curve(U, value, aspect)
    return ideal_curve(U, value)


def _half_departure(h: dict, g) -> float:
    """Mean |real - ideal| over the half, as a share of its height."""
    if g is None:
        return float("inf")
    return float(np.trapezoid(np.abs(h["g"] - g), U))


def with_flat(g_wall, u0: float):
    """A wall curve that starts at the edge of a flat top or floor, u0 along the half."""
    if g_wall is None or u0 <= 0:
        return g_wall
    return np.interp(np.clip((U - u0) / (1.0 - u0), 0.0, 1.0), U, g_wall)


def fit_family(fam: str, h: dict) -> dict:
    """The member of one family that leaves the least rock out of place on one half.

    Where the half has a measured flat top or floor (h["flat_u"] > 0), the
    ideal is flat out to the edge of it and the curve is fitted to the wall
    beyond: a flat-floored trough is straight walls on a flat floor, not a
    parabola stretched across the floor.
    """
    spec = FAMILIES[fam]
    u0 = float(h.get("flat_u", 0.0))
    wall = h["radius"] * (1.0 - u0)
    aspect = h["height"] / wall if wall > 0 else float("nan")
    grid = spec["grid"]
    if fam == "circle":
        # The range of arcs that neither overhang nor rise above the apex
        # depends on the aspect, so the grid is laid over that range.
        lo, hi = 0.0, 0.5
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if circle_curve(U, mid, aspect) is not None else (lo, mid)
        grid = np.linspace(-lo, lo, 201)
    d = np.array([_half_departure(h, with_flat(family_curve(fam, v, aspect), u0)) for v in grid])
    if not np.isfinite(d).any():
        return {"value": float("nan"), "departure": float("inf"), "g": None}
    i = int(np.nanargmin(d))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    best_v, best_d = float(grid[i]), float(d[i])
    if hi > lo:
        f = (lambda t: _half_departure(h, with_flat(family_curve(fam, np.exp(t), aspect), u0))) \
            if spec["log"] else (lambda t: _half_departure(h, with_flat(family_curve(fam, t, aspect), u0)))
        a, b = (np.log(lo), np.log(hi)) if spec["log"] else (lo, hi)
        r = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-4})
        if r.success and r.fun < best_d:
            best_v, best_d = (float(np.exp(r.x)) if spec["log"] else float(r.x)), float(r.fun)
    g_wall = family_curve(fam, best_v, aspect)
    return {"value": best_v, "departure": best_d, "g": with_flat(g_wall, u0), "g_wall": g_wall}


# Landmarks along the form axis, from the most sagging to the fullest. They name
# a fitted ideal by its fullness and mark the floor of the shape chart.
PERFECT_SHAPES = {
    "strongly_concave": {
        "name": "Strongly concave", "g": lambda u: ideal_curve(u, 0.5),
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
        "name": "Semicircle", "g": lambda u: ideal_curve(u, 2.0),
        "names": {HILL: "semi-elliptical dome", VALLEY: "semi-elliptical U",
                  SLOPE: "strongly convex"},
    },
    "box": {
        "name": "Box", "g": lambda u: ideal_curve(u, 4.0),
        "names": {HILL: "box / mesa", VALLEY: "box trough", SLOPE: "plateau edge"},
    },
}

# Departure classes, the analogue of Powers' roundness classes: rock out of
# place against the best-fitting ideal, as a share of width times height.
# PROVISIONAL - set from the synthetic and real profiles that ship with the
# app, not from a calibrated population.
DEPARTURE_CLASSES = (
    (0.01, "near-ideal"),
    (0.025, "slightly modified"),
    (0.05, "modified"),
    (float("inf"), "strongly modified"),
)


def fullness(g) -> float:
    """Share of the unit box under a half profile: 1/2 for a cone, 2/3 for a parabola."""
    return float(np.trapezoid(np.clip(np.asarray(g, float), 0.0, 1.0), U))


def landmark_for(f: float) -> str:
    """The landmark shape whose fullness is closest to f."""
    return min(PERFECT_SHAPES, key=lambda k: abs(fullness(PERFECT_SHAPES[k]["g"](U)) - f))


def fit_smooth(h: dict) -> dict:
    """A free three-control curve (apex p, foot q, flat top cut) on one half.

    Used only to tell a genuinely flat top or floor from a merely rounded one.
    """
    def resid(v):
        return ideal_curve(U, v[0], v[1], v[2]) - h["g"]

    best = None
    for p0, q0 in ((0.5, 0.5), (1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (4.0, 4.0)):
        r = least_squares(resid, [p0, q0, 0.02], bounds=([0.2, 0.2, 0.0], [8.0, 8.0, 0.7]))
        if best is None or r.cost < best.cost:
            best = r
    return {"p": float(best.x[0]), "q": float(best.x[1]), "cut": float(best.x[2])}


def compare(halves: list, ideal: dict, kind: str) -> dict:
    """Real minus ideal, as rock: positive is material the ideal does not have.

    Each half is weighted by its own box (width times height), so the areas are
    shares of the feature's box: size-independent and comparable between shapes.
    """
    rock = -1.0 if kind == VALLEY else 1.0
    excess = deficit = box = 0.0
    res = {}
    for h in halves:
        r = rock * (h["g"] - ideal[h["side"]])
        res[h["side"]] = r
        w = h["radius"] * h["height"]
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
        return "It stays within 2% of its height of this ideal everywhere."
    found.sort(key=lambda kv: -abs(kv[1]))
    parts = [f"{'more' if v > 0 else 'less'} rock at the {k} ({v * 100:+.0f}% of the height)"
             for k, v in found]
    return "Compared with this ideal it has " + ", ".join(parts) + "."


def has_flat_apex(item: dict) -> bool:
    """Flat ground at the apex that a free smooth curve also needs a cut for.

    The measured flat-top width alone is not enough: a rounded apex (the floor
    of a parabolic U) stays within 5% of the height of its lowest point for a
    fifth of the width without being flat at all.
    """
    cuts = [s["cut"] for s in item["smooth_by_side"].values()]
    return item["m"]["plateau"] >= 0.10 and bool(cuts) and float(np.median(cuts)) >= 0.04


def fit_words(item: dict) -> str:
    """The fitted ideal and its control on each half, e.g. 'parabola, k = +0.61 / +0.58'."""
    fam = item["family"]
    spec = FAMILIES[fam]
    vals = item["comparisons"][fam]["params"]
    fmt = "{:+.2f}" if fam != "superellipse" else "{:.2f}"
    parts = " / ".join(fmt.format(vals[h["side"]]) for h in item["halves"])
    return f"{spec['name'].lower()}, {spec['param']} = {parts}"


def measure(x, z, kind: str = HILL) -> dict:
    """Everything for one profile, by the fixed procedure above. Never raises."""
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

    # Steps 1 and 2: locate and normalise.
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

    # A flat top or floor is measured, not fitted: the ideal is flat out to its
    # edge and the curves are fitted to the walls beyond.
    smooth_by_side = {h["side"]: fit_smooth(h) for h in halves} if kind != SLOPE else {}
    flat = kind != SLOPE and has_flat_apex({"m": m, "smooth_by_side": smooth_by_side})
    for h in halves:
        edge = {"L": m["x_apex"] - m.get("x_flat_left", m["x_apex"]),
                "R": m.get("x_flat_right", m["x_apex"]) - m["x_apex"]}.get(h["side"], 0.0)
        if not flat or h["radius"] <= 0:
            h["flat_u"] = 0.0
            continue
        # The flat is measured where the ground has dropped plateau_level of the
        # height, a little way down the wall; follow the wall back up to where it
        # leaves the flat, so a straight wall is fitted as straight.
        u_edge = edge / h["radius"]
        drop = PROCESSING["plateau_level"] * (m["apex_elev"] - m["base_elev"]) / h["height"]
        h["flat_u"] = float(np.clip(1.0 - (1.0 - u_edge) / (1.0 - drop), 0.0, 0.9))

    # Step 3: fit every family to every half.
    comparisons = {}
    for fam in FAMILIES:
        fits = {h["side"]: fit_family(fam, h) for h in halves}
        if any(f["g"] is None for f in fits.values()):
            continue
        ideal = {s: f["g"] for s, f in fits.items()}
        comparisons[fam] = {"ideal": ideal, "params": {s: f["value"] for s, f in fits.items()},
                            "wall": {s: f["g_wall"] for s, f in fits.items()},
                            **compare(halves, ideal, kind)}
    straight = {h["side"]: with_flat(parabola_curve(U, 0.0), h["flat_u"]) for h in halves}
    comparisons[STRAIGHT] = {"ideal": straight, "params": {}, **compare(halves, straight, kind)}

    # Step 4: the family that leaves the least rock out of place, ties to the
    # earlier family in the fixed order.
    fams = [f for f in FAMILIES if f in comparisons]
    best_d = min(comparisons[f]["departure"] for f in fams)
    family = next(f for f in fams if comparisons[f]["departure"] <= best_d + TIE)

    # The name follows the wall's own form, so a flat floor does not make a V read as a U.
    ideal_full = {h["side"]: fullness(comparisons[family]["wall"][h["side"]]) for h in halves}
    wts = {h["side"]: h["radius"] * h["height"] for h in halves}
    tw = sum(wts.values())
    by_side = {h["side"]: {"fullness": fullness(h["g"]), "ideal_fullness": ideal_full[h["side"]],
                           "departure": compare([h], {h["side"]: comparisons[family]["ideal"][h["side"]]},
                                                kind)["departure"],
                           "landmark": landmark_for(ideal_full[h["side"]])}
               for h in halves}

    item.update(
        ok=True, x=x, z=z, m=m, halves=halves, comparisons=comparisons, family=family,
        smooth_by_side=smooth_by_side, by_side=by_side,
        departure=comparisons[family]["departure"],
        fullness=sum(wts[s] * by_side[s]["fullness"] for s in wts) / tw,
        ideal_fullness=sum(wts[s] * ideal_full[s] for s in wts) / tw,
    )
    item["landmark"] = landmark_for(item["ideal_fullness"])
    item["grade"] = departure_class(item["departure"])
    name = PERFECT_SHAPES[item["landmark"]]["names"][kind]
    if flat:
        name += f" with a {VOCAB[kind]['flat']}"
    item["name"] = name
    item["flat"] = flat
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
        "Best-fitting ideal": fit_words(item),
        "Departure": item["departure"],
        "Grade": item["grade"],
        "Net rock": c[item["family"]]["net"],
        "Fullness": item["fullness"],
        "Departure from straight": c[STRAIGHT]["departure"],
    }
    for fam, spec in FAMILIES.items():
        if fam in c:
            row[f"Departure, best {spec['name'].lower()}"] = c[fam]["departure"]
            for side, v in c[fam]["params"].items():
                row[f"{spec['name']} {spec['param']} ({side})"] = v
    for side, d in item["by_side"].items():
        row[f"Fullness ({side})"] = d["fullness"]
        row[f"Departure ({side})"] = d["departure"]
    row.update({
        "Width / height": item["m"]["width_height"],
        "Flat top/floor share": item["m"]["plateau"],
        "Asymmetry": item["m"]["asymmetry"],
    })
    return row


# =====================================================================
# Mid-ocean ridges
#
# A spreading ridge is read twice by the same fixed procedure, at two scales:
#
#   the WHOLE RIDGE, the full profile as a hill. Away from the axis the sea
#   floor deepens with the square root of its age (half-space cooling), so the
#   ridge is a broad concave rise. That subsidence is symmetric and is the
#   ridge itself, not a tilt under it, so it is fitted, not removed.
#
#   the AXIS, the ground within axis_window of the spreading axis, read both
#   as a hill and as a valley. Whichever reading puts its apex on the axis
#   says what the axis is: an axial high (its summit is on the axis) or an
#   axial rift valley (its floor is). The other reading lands off the axis, on
#   a rift mountain or in a trough on the flank.
#
# On 27 swath-median ETOPO1 profiles across 9 ridges from 14 to 145 mm/yr
# (three parallel profiles each, 20 km apart along the axis), every profile
# across a slow or ultraslow ridge (under 40 mm/yr) had an axial valley except
# two of the three on the southern Mid-Atlantic Ridge, and every profile
# across an intermediate or fast ridge had an axial high. The axial highs split by their fullness:
# intermediate ridges (56 and 70 mm/yr) 0.45 to 0.62, a low straight-sided to
# dome-shaped rise; fast ridges (90 to 145 mm/yr) 0.27 to 0.34, a peaked
# concave rise. Removing a fitted square-root-of-age ramp before reading the
# axis blurred that split, so the profile is read as it is. The fullness limit
# was set on these same profiles; the high-or-valley reading was not tuned.
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
    # Axial highs fuller than this read as intermediate, emptier as fast.
    "fast_fullness": 0.40,
}

RIDGE_CLASSES = {
    "slow": "slow or ultraslow (under about 40 mm/yr)",
    "intermediate": "intermediate (about 50 to 80 mm/yr)",
    "fast": "fast (over about 80 mm/yr)",
    "unclear": "unclear",
}


def cooling_curve(u) -> np.ndarray:
    """Depth below the axis growing with the square root of age, as a half profile: g = 1 - sqrt(u)."""
    return 1.0 - np.sqrt(np.clip(np.asarray(u, float), 0.0, 1.0))


def ridge_axis_x(x) -> float:
    """The spreading axis: distance 0 when it lies in the middle half of the profile, else the middle."""
    x = np.asarray(x, dtype=float)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    q1, q3 = lo + 0.25 * (hi - lo), lo + 0.75 * (hi - lo)
    return 0.0 if q1 <= 0.0 <= q3 else 0.5 * (lo + hi)


def read_ridge(x, z, axis: float | None = None) -> dict:
    """A spreading ridge by the fixed procedure, read whole and at its axis. Distances in metres.

    Returns the three measurements (whole, axis_hill, axis_valley), the axis
    position, which reading is on the axis, and the provisional rate class.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    ax = ridge_axis_x(x) if axis is None else float(axis)
    whole = measure(x, z, HILL)
    near = np.abs(x - ax) <= RIDGE["axis_window"]
    hill = measure(x[near], z[near], HILL)
    valley = measure(x[near], z[near], VALLEY)

    def on_axis(it):
        return bool(it.get("ok")) and abs(it["m"]["x_apex"] - ax) <= RIDGE["on_axis"]

    hi, lo = on_axis(hill), on_axis(valley)
    if hi and lo:
        # Both apexes on the axis: the bigger feature is the axis.
        hi = hill["m"]["height"] >= valley["m"]["height"]
        lo = not hi
    if lo:
        axis_type, rate, axis_item = "axial valley", "slow", valley
    elif hi:
        axis_type, axis_item = "axial high", hill
        rate = "intermediate" if hill["fullness"] >= RIDGE["fast_fullness"] else "fast"
    else:
        axis_type, rate, axis_item = "no clear axial feature", "unclear", None
    cool = float("nan")
    if whole.get("ok"):
        g = cooling_curve(U)
        cool = compare(whole["halves"], {h["side"]: g for h in whole["halves"]}, HILL)["departure"]
    return {
        "axis": ax,
        "whole": whole,
        "axis_hill": hill,
        "axis_valley": valley,
        "axis_type": axis_type,
        "axis_item": axis_item,
        "rate_class": rate,
        "rate_words": RIDGE_CLASSES[rate],
        "whole_on_axis": on_axis(whole),
        "cooling_departure": cool,
    }
