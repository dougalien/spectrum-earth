
import io
import os
from pathlib import Path
from dataclasses import dataclass

import requests

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import dougalien_brand as brand
import ideal_shape
import ideal_shape_view as ideal_view

import ai_client
from scipy import sparse
from scipy.fft import rfft, rfftfreq
from scipy.signal import find_peaks, savgol_filter
from scipy.sparse.linalg import spsolve


st.set_page_config(
    page_title="Spectrum Earth",
    page_icon=brand.page_icon("⛰️"),
    layout="wide",
)
brand.apply_branding()


DATASET_PRESETS = {}



# ============================================================
# Spectrum Earth - landform (topographic profile) extension
# ============================================================

TOPO_TRANSECT = "Topography (Transect)"
TOPO_LONG_PROFILE = "Topography (Long Profile)"
TOPO_DATASET_TYPES = (TOPO_TRANSECT, TOPO_LONG_PROFILE)

DATASET_PRESETS.update({
    TOPO_TRANSECT: {
        "x_label": "Distance along transect",
        "y_label": "Elevation",
        # Normalising destroys absolute relief, which is diagnostic. Keep it off.
        "normalize": False,
        # The power-law exponent b is markedly noise-sensitive: on a synthetic
        # parabola with 3 m of noise on 200 m of relief it reads 0.67 unsmoothed,
        # 0.89 at window 7 and 1.31 at window 31 against a true value of 2.00.
        # Hence a wide default window. The V-index is far more robust.
        "smoothing_on": True,
        "smoothing_method": "Savitzky-Golay",
        "smoothing_window": 21,
        # AsLS is a spectroscopic background model. It has no physical meaning
        # for an elevation series, so baseline correction stays off.
        "baseline_on": False,
        "baseline_method": "Linear",
        "asls_lambda": 1e5,
        "asls_p": 0.01,
        "asls_niter": 10,
        # A transect is referenced to its own thalweg inside the metric code,
        # so a global linear detrend would corrupt the asymmetry metric.
        "detrend_on": False,
        # Even x-spacing is required before the FFT can be trusted.
        "interpolate_on": True,
        "peak_prominence": 0.0,
        "peak_distance": 5,
        "peak_width": 0.0,
    },
    TOPO_LONG_PROFILE: {
        "x_label": "Distance downstream",
        "y_label": "Elevation",
        "normalize": False,
        "smoothing_on": True,
        "smoothing_method": "Savitzky-Golay",
        "smoothing_window": 11,
        "baseline_on": False,
        "baseline_method": "Linear",
        # A long profile must lose its regional gradient before the FFT, or all
        # the power piles into the lowest frequencies.
        "detrend_on": True,
        "interpolate_on": True,
        "peak_prominence": 0.0,
        "peak_distance": 10,
        "peak_width": 0.0,
    },
})

TOPO_STATE_DEFAULTS = {
    "topo_height_mode": "Auto (fraction of valley relief)",
    "topo_vertical_precision": 0.5,
    "topo_match_quantile": 0.95,
    "topo_height_fraction": 0.5,
    "topo_spec_height": 100.0,
    "topo_harmonic_count": 8,
    "topo_beta_low_frac": 0.02,
    "topo_beta_high_frac": 0.50,
    "topo_overdeepening_min_rise": 5.0,
    "topo_knick_prominence_frac": 0.50,
    "topo_slope_baseline_frac": 0.02,
    "topo_flow_left_to_right": True,
    "topo_show_uncalibrated": True,
    "topo_body": "Earth",
    "register_pinned": [],
    "register_pin_group": "",
    "register_backdrop": "Shaded regions",
    "overlay_ideal": "",
}


# ------------------------------------------------------------
# Landform reference library.
#
# Every row carries its own source string. Rows with weight 0 are COMPUTED AND
# REPORTED but never scored - they are placeholders awaiting calibration on
# labelled profiles. Only v_index and power_law_b carry published centres.
# ------------------------------------------------------------
LANDFORM_REFERENCE_ROWS = [
    {
        "landform": "Fluvial (river-carved)",
        "profile_type": TOPO_TRANSECT,
        "metric": "v_index",
        "center": 0.06,
        "sd": 0.14,
        "weight": 1.0,
        "source": "Zimmer (2016) MS thesis, San Jose State Univ. (see also Zimmer et al., 2018, Geomorphology 318): unglaciated valleys, mean V-index 0.06, s.d. 0.14.",
    },
    {
        "landform": "Glacial (ice-carved)",
        "profile_type": TOPO_TRANSECT,
        "metric": "v_index",
        "center": 0.19,
        "sd": 0.16,
        "weight": 1.0,
        "source": "Zimmer (2016) MS thesis, San Jose State Univ. (see also Zimmer et al., 2018, Geomorphology 318): glaciated valleys, mean V-index 0.19, s.d. 0.16.",
    },
    {
        "landform": "Fluvial (river-carved)",
        "profile_type": TOPO_TRANSECT,
        "metric": "power_law_b",
        "center": 1.00,
        "sd": 0.25,
        "weight": 1.0,
        "source": "Centre is the theoretical V-form exponent b = 1 (Svensson, 1959, J. Glaciol. 3(25), 362-363). THE SPREAD IS NOT PUBLISHED - set by the app author. Recalibrate.",
    },
    {
        "landform": "Glacial (ice-carved)",
        "profile_type": TOPO_TRANSECT,
        "metric": "power_law_b",
        "center": 1.90,
        "sd": 0.60,
        "weight": 1.0,
        "source": "Li, Liu & Cui (2001) Geomorphology 38, 153-166: Tian Shan glacial valleys, b = 1.03-3.50 with 71.4% between 1.3 and 2.5. Centre and spread summarise that distribution.",
    },
    {
        "landform": "Tectonic (fault-bounded)",
        "profile_type": TOPO_TRANSECT,
        "metric": "asymmetry_index",
        "center": 0.35,
        "sd": 0.20,
        "weight": 0.0,
        "source": "UNCALIBRATED. No published range located. Weight 0: reported, never scored. Calibrate on labelled fault-bounded profiles before raising the weight.",
    },
    {
        "landform": "Fluvial (river-carved)",
        "profile_type": TOPO_TRANSECT,
        "metric": "asymmetry_index",
        "center": 0.10,
        "sd": 0.12,
        "weight": 0.0,
        "source": "UNCALIBRATED placeholder. Weight 0.",
    },
    {
        "landform": "Glacial (ice-carved)",
        "profile_type": TOPO_TRANSECT,
        "metric": "asymmetry_index",
        "center": 0.12,
        "sd": 0.12,
        "weight": 0.0,
        "source": "UNCALIBRATED placeholder. Weight 0.",
    },
    {
        "landform": "Fluvial (river-carved)",
        "profile_type": TOPO_TRANSECT,
        "metric": "even_odd_harmonic_ratio",
        "center": 0.05,
        "sd": 0.15,
        "weight": 0.0,
        "source": "SYNTHETIC ANCHOR ONLY, not field-calibrated. A symmetric V-section has null even harmonics: measured 0.000 on a synthetic V. CONFOUNDED BY ASYMMETRY - an asymmetric V measured 0.32, indistinguishable from a symmetric parabola. Read only alongside asymmetry_index. Weight 0.",
    },
    {
        "landform": "Glacial (ice-carved)",
        "profile_type": TOPO_TRANSECT,
        "metric": "even_odd_harmonic_ratio",
        "center": 0.35,
        "sd": 0.15,
        "weight": 0.0,
        "source": "SYNTHETIC ANCHOR ONLY, not field-calibrated. Measured 0.31 on a synthetic parabola and 0.45 on a flat-floored trough. CONFOUNDED BY ASYMMETRY - see the fluvial row. Weight 0.",
    },
    {
        "landform": "Glacial (ice-carved)",
        "profile_type": TOPO_LONG_PROFILE,
        "metric": "overdeepening_count",
        "center": 2.0,
        "sd": 1.5,
        "weight": 0.0,
        "source": "UNCALIBRATED. Reversed downstream gradient (overdeepening) is a recognised glacial indicator, but no class-specific count distribution is encoded here. Weight 0.",
    },
    {
        "landform": "Fluvial (river-carved)",
        "profile_type": TOPO_LONG_PROFILE,
        "metric": "overdeepening_count",
        "center": 0.0,
        "sd": 1.0,
        "weight": 0.0,
        "source": "UNCALIBRATED placeholder. Weight 0.",
    },
    {
        "landform": "Tectonic (fault-bounded)",
        "profile_type": TOPO_LONG_PROFILE,
        "metric": "knickpoint_count",
        "center": 3.0,
        "sd": 2.0,
        "weight": 0.0,
        "source": "UNCALIBRATED. Knickpoints are a standard tectonic / base-level signal (Wobus et al., 2006, GSA Spec. Pap. 398; Kirby & Whipple, 2012, J. Struct. Geol. 44, 54-75), but no count threshold is published for this use. Weight 0.",
    },
]

LANDFORM_METRIC_LABELS = {
    "v_index": "V-index  (Ax / Av - 1)",
    "shape_conformity": "Conformity to nearest ideal (1 = ideal)",
    "shape_departure": "Departure from ideal (0 = ideal)",
    "form_net": "Net area vs ideal (signed, global form)",
    "form_irregularity": "Irregularity: total area vs ideal (unsigned)",
    "form_excess": "Area above the ideal",
    "form_deficit": "Area below the ideal",
    "power_law_b": "Power-law exponent b  (both walls)",
    "power_law_b_left": "Power-law exponent b  (left wall)",
    "power_law_b_right": "Power-law exponent b  (right wall)",
    "power_law_r2": "Power-law fit r-squared",
    "asymmetry_index": "Asymmetry index  |A_R - A_L| / (A_R + A_L)",
    "asymmetry_signed": "Asymmetry, signed  (+ = bulk to the right)",
    "width_depth_ratio": "Width / depth at benchmark height",
    "cross_section_area": "Cross-sectional area below benchmark",
    "benchmark_height": "Benchmark height above thalweg",
    "valley_relief": "Usable valley relief",
    "harmonic_decay_exponent": "FFT odd-harmonic decay exponent",
    "even_odd_harmonic_ratio": "FFT even / odd harmonic amplitude ratio",
    "spectral_slope_beta": "Spectral slope beta  (P ~ k^-beta)",
    "overdeepening_count": "Overdeepening count",
    "overdeepening_total_rise": "Total reversed-gradient rise",
    "knickpoint_count": "Knickpoint count",
    "slope_baseline": "Slope measurement baseline",
    "max_knickpoint_slope": "Steepest knickpoint gradient",
}

LANDFORM_METRIC_UNITS = {
    "v_index": "dimensionless",
    "shape_conformity": "dimensionless",
    "shape_departure": "dimensionless",
    "form_net": "fraction of ideal area",
    "form_irregularity": "fraction of ideal area",
    "form_excess": "fraction of ideal area",
    "form_deficit": "fraction of ideal area",
    "power_law_b": "dimensionless",
    "power_law_b_left": "dimensionless",
    "power_law_b_right": "dimensionless",
    "power_law_r2": "dimensionless",
    "asymmetry_index": "dimensionless",
    "asymmetry_signed": "dimensionless",
    "width_depth_ratio": "dimensionless",
    "cross_section_area": "y-units x x-units",
    "benchmark_height": "y-units",
    "valley_relief": "y-units",
    "harmonic_decay_exponent": "dimensionless",
    "even_odd_harmonic_ratio": "dimensionless",
    "spectral_slope_beta": "dimensionless",
    "overdeepening_count": "count",
    "overdeepening_total_rise": "y-units",
    "knickpoint_count": "count",
    "slope_baseline": "x-units",
    "max_knickpoint_slope": "y-units per x-unit",
}


def builtin_landform_reference_df() -> pd.DataFrame:
    return pd.DataFrame(LANDFORM_REFERENCE_ROWS + VOLCANO_REFERENCE_ROWS)


def standardize_landform_reference(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a landform reference table to the columns the scorer expects."""
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]

    required = {"landform", "metric", "center", "sd"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            "Landform reference needs columns: landform, metric, center, sd "
            "(optional: profile_type, weight, source). Missing: "
            + ", ".join(sorted(missing))
        )

    if "profile_type" not in out.columns:
        out["profile_type"] = TOPO_TRANSECT
    if "weight" not in out.columns:
        out["weight"] = 1.0
    if "source" not in out.columns:
        out["source"] = "User-supplied; no citation given."
    if "rule" not in out.columns:
        out["rule"] = "gaussian"
    out["rule"] = out["rule"].fillna("gaussian").astype(str).str.strip().str.lower()

    out["landform"] = out["landform"].astype(str).str.strip()
    out["metric"] = (
        out["metric"].astype(str).str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    out["profile_type"] = out["profile_type"].astype(str).str.strip()
    for col in ("center", "sd", "weight"):
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["center", "sd"]).reset_index(drop=True)
    out["weight"] = out["weight"].fillna(0.0).clip(lower=0.0)
    out = out[out["sd"] > 0].reset_index(drop=True)
    if out.empty:
        raise ValueError("Landform reference table has no usable rows after cleaning.")
    return out


# ------------------------------------------------------------
# Cross-valley (transect) metrics
# ------------------------------------------------------------
def _topo_crossing_left(x: np.ndarray, y: np.ndarray, t: int, zb: float):
    idx = None
    for i in range(t, -1, -1):
        if y[i] >= zb:
            idx = i
            break
    if idx is None:
        return None
    if idx == t:
        return float(x[t])
    y0, y1 = float(y[idx]), float(y[idx + 1])
    if np.isclose(y0, y1):
        return float(x[idx])
    frac = (zb - y1) / (y0 - y1)
    return float(x[idx + 1] + frac * (x[idx] - x[idx + 1]))


def _topo_crossing_right(x: np.ndarray, y: np.ndarray, t: int, zb: float):
    idx = None
    for i in range(t, len(y)):
        if y[i] >= zb:
            idx = i
            break
    if idx is None:
        return None
    if idx == t:
        return float(x[t])
    y0, y1 = float(y[idx]), float(y[idx - 1])
    if np.isclose(y0, y1):
        return float(x[idx])
    frac = (zb - y1) / (y0 - y1)
    return float(x[idx - 1] + frac * (x[idx] - x[idx - 1]))


def topo_valley_window(
    x: np.ndarray,
    y: np.ndarray,
    height_fraction: float = 0.5,
    spec_height: float | None = None,
):
    """Locate the thalweg and the benchmark-height window that brackets it.

    Returns (thalweg_index, x_left, x_right, benchmark_height, usable_relief),
    or None when no closed cross-section can be defined.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 7:
        return None

    # Centre the thalweg on the minimum plateau. Plain argmin puts the origin at
    # the LEFT EDGE of a flat valley floor, which manufactures a large false
    # asymmetry in exactly the flat-floored troughs this app most needs to read.
    z_min = float(np.min(y))
    tol = max((float(np.max(y)) - z_min) * 1e-3, 1e-9)
    a_idx = int(np.argmin(y))
    lo_i = a_idx
    while lo_i > 0 and y[lo_i - 1] <= z_min + tol:
        lo_i -= 1
    hi_i = a_idx
    while hi_i < len(y) - 1 and y[hi_i + 1] <= z_min + tol:
        hi_i += 1
    t = (lo_i + hi_i) // 2
    if t == 0 or t == len(y) - 1:
        return None

    z0 = float(y[t])
    left_max = float(np.max(y[: t + 1]))
    right_max = float(np.max(y[t:]))
    usable_relief = min(left_max, right_max) - z0
    if usable_relief <= 0:
        return None

    h = usable_relief * float(height_fraction) if spec_height is None else float(spec_height)
    h = min(h, usable_relief)
    if h <= 0:
        return None

    # Nudge the benchmark a hair below the target. At full relief the benchmark
    # otherwise lands exactly on the matching elevation, and floating-point
    # rounding can place it a fraction ABOVE the highest qualifying sample - no
    # point then satisfies the crossing test, the window returns None and the
    # whole measurement is silently skipped. The nudge is far below any real
    # instrument precision and cannot affect a reported value.
    zb = z0 + h - 1e-9 * max(abs(usable_relief), 1.0)
    xl = _topo_crossing_left(x, y, t, zb)
    xr = _topo_crossing_right(x, y, t, zb)
    if xl is None or xr is None or xr <= xl:
        return None
    return t, xl, xr, h, usable_relief


def _fit_power_law(
    xs: np.ndarray,
    ys: np.ndarray,
    x_origin: float,
    z_origin: float,
    min_points: int = 8,
    min_dz: float = 0.0,
):
    """Least-squares fit of |dz| = a * |dx|^b in log-log space. Returns (b, r2).

    Points within min_dz of the thalweg are excluded. In log-log space the
    near-origin points carry enormous leverage, so on a noisy real profile a few
    centimetres of DEM noise beside the channel will drag b well below its true
    value. Li, Liu & Cui (2001, Geomorphology 38, 153-166) flagged this origin
    sensitivity; the floor is how this app contains it.
    """
    dx = np.abs(xs - x_origin)
    dz = ys - z_origin
    mask = (dx > 0) & (dz > float(min_dz)) & np.isfinite(dx) & np.isfinite(dz)
    if int(mask.sum()) < min_points:
        return np.nan, np.nan

    lx = np.log(dx[mask])
    lz = np.log(dz[mask])
    if not (np.all(np.isfinite(lx)) and np.all(np.isfinite(lz))):
        return np.nan, np.nan

    slope, intercept = np.polyfit(lx, lz, 1)
    pred = slope * lx + intercept
    ss_res = float(np.sum((lz - pred) ** 2))
    ss_tot = float(np.sum((lz - np.mean(lz)) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return float(slope), float(r2)


TOPO_MATCH_MODE = "Matched elevation (highest level on both sides)"


def matched_elevation_height(
    x: np.ndarray,
    y: np.ndarray,
    vertical_precision: float = 0.5,
    quantile: float = 0.95,
) -> tuple[float, list]:
    """Highest elevation at which BOTH sides of the feature carry terrain.

    The rule: walk up from the thalweg; the baseline is the highest level at
    which a matching point still exists on both sides. It is the 1-D form of the
    spill (pour) elevation used in DEM depression filling, and it is
    parameter-free in principle - no operator picks a height.

    Two corrections make it work on real data.

    TOLERANCE. Elevations are matched within the instrument's vertical
    precision rather than exactly. Matching to the last floating-point bit is
    not physically meaningful, and at exactly full relief the comparison sits on
    a knife edge: the benchmark lands a rounding error above the matching
    elevation, no sample qualifies, and the whole measurement is skipped.
    Quantizing to `vertical_precision` removes that failure and makes the
    operation mean what a real measurement can support.

    ROBUSTNESS. Using each side's single highest sample lets one stray reading
    define the baseline. Where the profile runs out onto flat terrain beyond the
    feature - a valley cut into a plateau, a volcano standing on a plain - the
    deepest noise excursion on that flat surface captures the rule entirely, and
    the measurement stops describing the landform. Each side's maximum is
    therefore a high quantile of its elevations, not its extremum.

    Measured on synthetic profiles carrying 1 m and 5 m noise, true index 0.000:

        estimator        valley-in-plateau     volcano-on-plain
        raw maximum          -0.493 / -0.333      -0.493 / -0.333
        quantile 0.95        -0.010 / -0.001      -0.010 / -0.001
        quantile 0.98        -0.090 / -0.016      -0.090 / -0.016
        stall detection      -0.026 / -0.049      -0.026 / -0.049

    Quantile 0.95 was the most accurate and the most stable across noise levels,
    so it is the default. Note the valley and volcano columns are identical:
    the rule is exactly reflection-symmetric, and the failure it guards against
    is driven by flat terrain beyond the feature, not by which way up it is.

    Returns (height above the thalweg, notes).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    notes: list = []

    if len(y) < 8:
        return float("nan"), ["Too few points to establish a matched-elevation baseline."]

    t = int(np.argmin(y))
    if t == 0 or t == len(y) - 1:
        return float("nan"), [
            "Matched-elevation baseline needs a low point inside the profile, not at an end."
        ]

    z0 = float(y[t])
    left = y[: t + 1][::-1]
    right = y[t:]

    d = max(float(vertical_precision), 0.0)
    q = float(np.clip(quantile, 0.50, 1.00))

    def side_level(side: np.ndarray) -> float:
        if len(side) < 4:
            return float(np.max(side))
        v = float(np.quantile(side, q))
        return float(np.round(v / d) * d) if d > 0 else v

    lm = side_level(left)
    rm = side_level(right)
    h = min(lm, rm) - z0

    if not np.isfinite(h) or h <= 0:
        return float("nan"), [
            "No matching elevation was found on both sides of the feature."
        ]

    if abs(lm - rm) > 0.25 * max(h, 1e-9):
        notes.append(
            f"The two sides top out {abs(lm - rm):.4g} y-units apart, so the matched baseline is "
            "set by the lower side and the taller side is measured only up to that level. That is "
            "the intended behaviour, but it under-measures width on strongly asymmetric features."
        )

    return float(h), notes


def compute_cross_section_metrics(
    x: np.ndarray,
    y: np.ndarray,
    height_fraction: float = 0.5,
    spec_height: float | None = None,
    harmonic_count: int = 8,
    baseline_mode: str = "fraction",
    vertical_precision: float = 0.5,
    match_quantile: float = 0.95,
) -> tuple[dict, list]:
    """Cross-valley form metrics computed about the thalweg.

    V-index follows Zimmer (2016, MS thesis, San Jose State Univ.; see also
    Zimmer et al., 2018, Geomorphology 318): V-index = Ax/Av - 1, where Ax is
    the cross-sectional area below a benchmark height above the valley floor and
    Av is the area of an ideal V of the same width and height. A perfect V gives
    0, a parabola gives 1/3, a rectangular trough approaches 1.

    The power-law exponent b follows Svensson (1959, J. Glaciol. 3(25), 362-363).
    Li, Liu & Cui (2001, Geomorphology 38, 153-166) showed that a and b are
    sensitive to the choice of origin, so the thalweg is used explicitly and the
    two walls are also fitted separately.
    """
    metrics: dict = {}
    notes: list = []

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if str(baseline_mode).lower().startswith("match"):
        matched_h, matched_notes = matched_elevation_height(x, y, vertical_precision, match_quantile)
        notes.extend(matched_notes)
        if np.isfinite(matched_h) and matched_h > 0:
            spec_height = matched_h
        else:
            notes.append("Falling back to the fractional baseline for this profile.")

    win = topo_valley_window(x, y, height_fraction, spec_height)
    if win is None:
        notes.append(
            "No closed valley cross-section could be defined: the profile minimum sits at an end "
            "of the transect, or one wall never reaches the benchmark height. Cross-section "
            "metrics were skipped."
        )
        return metrics, notes

    t, xl, xr, h, relief = win
    z0 = float(y[t])
    zb = z0 + h
    x_t = float(x[t])

    n_res = int(np.clip(len(x) * 4, 201, 20001))
    xs = np.linspace(xl, xr, n_res)
    ys = np.interp(xs, x, y)
    depth = np.clip(zb - ys, 0.0, None)

    ax_area = float(np.trapezoid(depth, xs))
    av_area = 0.5 * (xr - xl) * h

    metrics["benchmark_height"] = float(h)
    metrics["valley_relief"] = float(relief)
    metrics["cross_section_area"] = ax_area
    if av_area > 0:
        metrics["v_index"] = ax_area / av_area - 1.0
    if h > 0:
        metrics["width_depth_ratio"] = float((xr - xl) / h)

    left_mask = xs <= x_t
    right_mask = xs >= x_t
    a_left = float(np.trapezoid(depth[left_mask], xs[left_mask])) if int(left_mask.sum()) > 1 else 0.0
    a_right = float(np.trapezoid(depth[right_mask], xs[right_mask])) if int(right_mask.sum()) > 1 else 0.0
    total_area = a_left + a_right
    if total_area > 0:
        metrics["asymmetry_signed"] = (a_right - a_left) / total_area
        metrics["asymmetry_index"] = abs(a_right - a_left) / total_area

    dz_floor = 0.05 * h
    if metrics.get("v_index") is not None and np.isfinite(metrics.get("v_index", np.nan)):
        if metrics["v_index"] < -0.05:
            notes.append(
                f"The shape index came out at {metrics['v_index']:+.3f}. A negative value means the "
                "section holds LESS area than an ideal V, which a simple valley or edifice cannot "
                "do. That almost always means the profile extends past the feature onto terrain "
                "beyond it, so the baseline was set out there rather than on the landform. Trim the "
                "profile to the rim (or the foot), or use the matched-elevation baseline."
            )

    b_all, r2_all = _fit_power_law(xs, ys, x_t, z0, min_dz=dz_floor)
    if np.isfinite(b_all):
        metrics["power_law_b"] = b_all
        metrics["power_law_r2"] = r2_all

    if int(left_mask.sum()) > 8:
        b_l, _ = _fit_power_law(xs[left_mask], ys[left_mask], x_t, z0, min_dz=dz_floor)
        if np.isfinite(b_l):
            metrics["power_law_b_left"] = b_l
    if int(right_mask.sum()) > 8:
        b_r, _ = _fit_power_law(xs[right_mask], ys[right_mask], x_t, z0, min_dz=dz_floor)
        if np.isfinite(b_r):
            metrics["power_law_b_right"] = b_r

    # Harmonic roll-off of the windowed cross-section. No taper is applied: the
    # window already starts and ends at the benchmark elevation, so it is close
    # to naturally periodic, and a taper would suppress the harmonic content
    # this metric is trying to measure.
    try:
        fft_win = compute_fft(xs, ys, window_name="None")
        f = np.asarray(fft_win.frequencies, dtype=float)
        a = np.asarray(fft_win.amplitudes, dtype=float)
        pos = (f > 0) & np.isfinite(f) & np.isfinite(a)
        f = f[pos]
        a = a[pos]
        if len(f) > 6:
            k0 = int(np.argmax(a))
            n_harm = int(np.clip(harmonic_count, 3, 40))
            # Sample the amplitude AT each integer multiple of the fundamental
            # rather than every bin in between. A symmetric V-section is a
            # triangular wave with NULL even harmonics; averaging those nulls
            # into a log-log fit produces a spuriously steep slope (measured
            # -3.84 instead of the true -2.00 on a synthetic V).
            harmonics = []
            for n in range(1, n_harm + 1):
                idx = (k0 + 1) * n - 1
                harmonics.append(a[idx] if idx < len(a) else np.nan)
            harmonics = np.asarray(harmonics, dtype=float)

            if np.isfinite(harmonics[0]) and harmonics[0] > 0:
                rel = harmonics / harmonics[0]
                odd_sum = float(np.nansum(rel[0::2]))
                even_sum = float(np.nansum(rel[1::2]))
                if odd_sum > 0:
                    metrics["even_odd_harmonic_ratio"] = even_sum / odd_sum

                ns = np.arange(1, len(rel) + 1)
                odd_mask = np.zeros(len(rel), dtype=bool)
                odd_mask[0::2] = True
                fit_mask = odd_mask & np.isfinite(rel) & (rel > 0)
                if int(fit_mask.sum()) >= 3:
                    slope, _ = np.polyfit(np.log(ns[fit_mask]), np.log(rel[fit_mask]), 1)
                    metrics["harmonic_decay_exponent"] = float(slope)
    except Exception:
        notes.append("Harmonic metrics could not be computed for this transect.")

    if metrics.get("power_law_r2") is not None and np.isfinite(metrics.get("power_law_r2", np.nan)):
        if metrics["power_law_r2"] < 0.7:
            notes.append(
                "Power-law fit r-squared is below 0.70, so the exponent b is weakly constrained "
                "for this transect. Treat the b-based evidence with caution."
            )

    return metrics, notes


# ------------------------------------------------------------
# Down-valley (long profile) metrics
# ------------------------------------------------------------
def compute_long_profile_metrics(
    x: np.ndarray,
    y_geometry: np.ndarray,
    fft_results=None,
    flow_left_to_right: bool = True,
    beta_low_frac: float = 0.02,
    beta_high_frac: float = 0.50,
    min_rise: float = 5.0,
    knick_prominence_frac: float = 0.50,
    slope_baseline_frac: float = 0.02,
) -> tuple[dict, list]:
    """Long-profile descriptors.

    Overdeepening and knickpoint counts are computed on the UN-detrended
    elevation, because detrending a concave profile manufactures artificial
    reversed-gradient reaches. The spectral slope beta is computed from the
    app-level FFT, which does operate on the detrended signal, following the
    topographic power-spectrum approach of Perron, Kirchner & Dietrich (2008,
    J. Geophys. Res. Earth Surf. 113, F04003) and Pike & Rozema (1975, Ann.
    Assoc. Am. Geogr. 65, 499-516).
    """
    metrics: dict = {}
    notes: list = []

    x = np.asarray(x, dtype=float)
    yg = np.asarray(y_geometry, dtype=float)
    if len(x) < 16 or len(x) != len(yg):
        notes.append("Too few points for long-profile metrics (16 minimum).")
        return metrics, notes

    if not flow_left_to_right:
        yg = yg[::-1].copy()
        x = (float(np.max(x)) - x)[::-1].copy()

    # OVERDEEPENING DETECTION - net rise above a running minimum.
    #
    # An earlier version counted strictly monotonic rising runs. On real data
    # that fails: a few tenths of a metre of noise fragments one genuine 14 m
    # basin into sub-threshold pieces, and the count then depends on the
    # smoothing window rather than on the topography. On the bundled glacial
    # sample (two real basins, 13.8 m and 77.7 m) the old method returned 4
    # unsmoothed, 1 at window 11 and 2 at window 51. This accumulator returns 2
    # at every window from 11 upward, and 0 on the fluvial sample.
    count = 0
    total_rise = 0.0
    run_min = float(yg[0])
    peak = float(yg[0])
    counted = False
    for v in yg[1:]:
        v = float(v)
        if v <= run_min:
            # The profile has dropped below the floor this rise started from, so
            # the rise is closed. Its FULL height is banked here, not at the
            # moment the threshold was first crossed - banking it early records
            # only min_rise and under-reports the total badly (10.3 m against a
            # true 91.5 m on the bundled glacial sample).
            if counted:
                total_rise += peak - run_min
                counted = False
            run_min = v
            peak = v
        else:
            if v > peak:
                peak = v
            if (not counted) and (peak - run_min) >= float(min_rise):
                count += 1
                counted = True
    if counted:
        total_rise += peak - run_min
    metrics["overdeepening_count"] = float(count)
    metrics["overdeepening_total_rise"] = float(total_rise)

    # KNICKPOINT DETECTION - slope measured over a stated BASELINE.
    #
    # A point-to-point gradient on a 10 m-spaced profile carrying sub-metre
    # noise is almost pure noise, and the detector fired 104 times on a sample
    # holding two real steps. Slope is measured over a baseline here (a stated
    # fraction of profile length), the prominence threshold keys off the
    # interquartile spread rather than the noise-inflated full range, and a
    # minimum separation stops one step being counted repeatedly.
    try:
        k = max(int(len(x) * float(slope_baseline_frac)) // 2 * 2 + 1, 3)
        half = k // 2
        y_pad = np.pad(yg, half, mode="edge")
        x_pad = np.pad(x, half, mode="reflect", reflect_type="odd")
        steepness = -(y_pad[k - 1:] - y_pad[:len(yg)]) / (x_pad[k - 1:] - x_pad[:len(x)])
        q1, q3 = np.percentile(steepness[np.isfinite(steepness)], [25, 75])
        prom = max(float(q3 - q1) * float(knick_prominence_frac), 1e-12)
        sep = max(int(len(x) * 0.02), 1)
        idx, _props = find_peaks(steepness, prominence=prom, distance=sep)
        metrics["knickpoint_count"] = float(len(idx))
        metrics["slope_baseline"] = float(k) * float(np.mean(np.diff(x)))
        if len(idx):
            metrics["max_knickpoint_slope"] = float(np.nanmax(steepness[idx]))
    except Exception:
        notes.append("Knickpoint detection could not be completed for this profile.")

    if fft_results is not None and len(getattr(fft_results, "frequencies", [])) > 8:
        f = np.asarray(fft_results.frequencies, dtype=float)
        a = np.asarray(fft_results.amplitudes, dtype=float)
        mask = (f > 0) & (a > 0) & np.isfinite(f) & np.isfinite(a)
        f = f[mask]
        a = a[mask]
        if len(f) > 8:
            fmax = float(np.max(f))
            lo = fmax * float(beta_low_frac)
            hi = fmax * float(beta_high_frac)
            sel = (f >= lo) & (f <= hi)
            if int(sel.sum()) >= 6:
                power = a[sel] ** 2
                slope_ll, _ = np.polyfit(np.log10(f[sel]), np.log10(power), 1)
                metrics["spectral_slope_beta"] = float(-slope_ll)
            else:
                notes.append("Too few FFT points inside the beta fitting band; widen the band.")

    return metrics, notes


# ------------------------------------------------------------
# Volcanic edifice metrics.
#
# A volcano is an inverted valley: negate elevation and the summit becomes the
# thalweg, so the same window/area/power-law core serves both. Nothing below
# re-implements that geometry - it inverts the profile and calls it.
# ------------------------------------------------------------
TOPO_VOLCANO = "Volcano (Radial Profile)"

# Surface gravity, m/s2. Used for CONTEXT ONLY - it is displayed beside the
# result and never applied to the thresholds, because no published gravity
# scaling for edifice slope was located to justify one.
PLANETARY_BODIES = {
    "Earth": 9.81,
    "Mars": 3.72,
    "Moon": 1.62,
    "Venus": 8.87,
    "Mercury": 3.70,
    "Io": 1.80,
    "Titan": 1.35,
    "Ceres": 0.28,
    "Pluto": 0.62,
    "Enceladus": 0.11,
    "Triton": 0.78,
    "Other / unknown": float("nan"),
}

VOLCANO_REFERENCE_ROWS = [
    {
        "landform": "Shield volcano",
        "profile_type": TOPO_VOLCANO,
        "metric": "mean_flank_slope_deg",
        "center": 12.0,
        "sd": 3.0,
        "weight": 1.0,
        "rule": "threshold_below",
        "source": "Grosse & Kervyn (2018) Geomorphology 304, 1-14: 158 terrestrial shields, average slopes mostly 1-15 deg; an approximate shield/stratovolcano threshold is drawn at 12 deg average slope. Softness (sd) is set by the app author, not published.",
    },
    {
        "landform": "Stratovolcano (composite)",
        "profile_type": TOPO_VOLCANO,
        "metric": "mean_flank_slope_deg",
        "center": 17.0,
        "sd": 6.0,
        "weight": 1.0,
        "rule": "gaussian",
        "source": "Grosse et al. (2013/2014) Bull. Volcanol. 76:784, MORVOLC global database of ~750 composite volcanoes: average slope 3-30 deg, MEDIAN 17 deg. Centre is the published median; the spread (sd) is the app author's summary of the published range, not a published dispersion.",
    },
    {
        "landform": "Shield volcano",
        "profile_type": TOPO_VOLCANO,
        "metric": "height_basal_width_ratio",
        "center": 0.10,
        "sd": 0.03,
        "weight": 1.0,
        "rule": "threshold_below",
        "source": "Grosse & Kervyn (2018) Geomorphology 304, 1-14: shield H/W_B mostly 0.01-0.10, with the approximate shield/stratovolcano threshold at 0.10. Softness (sd) is set by the app author, not published.",
    },
    {
        "landform": "Stratovolcano (composite)",
        "profile_type": TOPO_VOLCANO,
        "metric": "height_basal_width_ratio",
        "center": 0.12,
        "sd": 0.06,
        "weight": 1.0,
        "rule": "gaussian",
        "source": "Grosse et al. (2013/2014) Bull. Volcanol. 76:784, MORVOLC global database of ~750 composite volcanoes: H/W_B 0.01-0.28, MEDIAN 0.12 (H 100-2500 m median 1020 m; W_B 2-36 km median 8.8 km). Centre is the published median; the spread (sd) is the app author's summary of the published range, not a published dispersion.",
    },
    {
        "landform": "Shield volcano",
        "profile_type": TOPO_VOLCANO,
        "metric": "cone_index",
        "center": 0.15,
        "sd": 0.20,
        "weight": 0.0,
        "source": "UNCALIBRATED. Inverted V-index: a perfect cone reads 0, a convex (shield-like) flank reads positive, a concave (stratocone) flank reads negative. No published distribution located. Weight 0.",
    },
    {
        "landform": "Stratovolcano (composite)",
        "profile_type": TOPO_VOLCANO,
        "metric": "cone_index",
        "center": -0.15,
        "sd": 0.20,
        "weight": 0.0,
        "source": "UNCALIBRATED. See the shield cone_index row. Weight 0.",
    },
]

VOLCANO_METRIC_LABELS = {
    "edifice_height": "Edifice height above basal boundary",
    "basal_width": "Basal width at the boundary",
    "summit_width": "Summit width (near-peak)",
    "summit_basal_width_ratio": "Summit / basal width (truncation)",
    "height_basal_width_ratio": "Height / basal width  (H/W_B)",
    "mean_flank_slope_deg": "Mean flank slope, area-weighted (degrees)",
    "mean_flank_slope_unweighted_deg": "Mean flank slope, unweighted (degrees)",
    "bulk_flank_slope_deg": "Bulk summit-to-base slope (degrees)",
    "max_flank_slope_deg": "Maximum flank slope (degrees)",
    "cone_index": "Cone index  (inverted V-index)",
    "karatson_upper_slope_coef": "Karatson upper-flank slope coefficient (deg/m)",
    "flank_roughness_beta": "Flank roughness spectral slope beta",
    "flank_hurst_exponent": "Flank Hurst exponent H = (beta - 1) / 2",
}

VOLCANO_METRIC_UNITS = {
    "edifice_height": "y-units",
    "basal_width": "x-units",
    "summit_width": "x-units",
    "summit_basal_width_ratio": "dimensionless",
    "height_basal_width_ratio": "dimensionless",
    "mean_flank_slope_deg": "degrees",
    "mean_flank_slope_unweighted_deg": "degrees",
    "bulk_flank_slope_deg": "degrees",
    "max_flank_slope_deg": "degrees",
    "cone_index": "dimensionless",
    "karatson_upper_slope_coef": "degrees per x-unit",
    "flank_roughness_beta": "dimensionless",
    "flank_hurst_exponent": "dimensionless",
}

# Keys the inverted valley core returns that carry over unchanged, and the
# volcano-facing name for each.
_VALLEY_TO_EDIFICE_KEYS = {
    "v_index": "cone_index",
    "power_law_b": "power_law_b",
    "power_law_b_left": "power_law_b_left",
    "power_law_b_right": "power_law_b_right",
    "power_law_r2": "power_law_r2",
    "asymmetry_index": "asymmetry_index",
    "asymmetry_signed": "asymmetry_signed",
    "harmonic_decay_exponent": "harmonic_decay_exponent",
    "even_odd_harmonic_ratio": "even_odd_harmonic_ratio",
}


def _karatson_upper_slope_coefficient(r: np.ndarray, z: np.ndarray, summit_r: float, height: float):
    """Angular coefficient of local slope against distance over the UPPER HALF.

    Karatson et al. (2010, J. Volcanol. Geotherm. Res. 193, 171-181) split
    regular stratovolcanoes into C-type (conical, near-linear upper flanks) and
    P-type (parabolic, markedly concave upper flanks) at an upper-flank slope
    angular coefficient of -50e-4 deg/m. Returned here in degrees per x-unit, so
    a profile in metres is directly comparable to that threshold.
    """
    if height <= 0 or len(r) < 12:
        return np.nan
    z_top = float(np.max(z))
    upper = z >= (z_top - 0.5 * height)
    if int(upper.sum()) < 8:
        return np.nan
    ru = r[upper]
    zu = z[upper]
    order = np.argsort(ru)
    ru = ru[order]
    zu = zu[order]
    if len(np.unique(ru)) < 8:
        return np.nan
    local = np.degrees(np.arctan(np.abs(np.gradient(zu, ru))))
    dist = np.abs(ru - summit_r)
    good = np.isfinite(local) & np.isfinite(dist)
    if int(good.sum()) < 8:
        return np.nan
    coef, _ = np.polyfit(dist[good], local[good], 1)
    return float(coef)


def _flank_roughness(
    r: np.ndarray,
    z: np.ndarray,
    low_frac: float = 0.10,
    high_frac: float = 0.45,
    min_relief_fraction: float = 1e-6,
):
    """Flank surface roughness: spectral slope beta and Hurst exponent H.

    Each flank is detrended SEPARATELY with a quadratic and the summit region is
    excluded. A single polynomial fitted across the whole edifice cannot follow
    the apex, and its misfit there swamps the surface texture: on a smooth
    synthetic cone that error alone produced a residual of 7% of relief and a
    meaningless beta of 3.81.

    Roughness parameters for planetary surfaces follow Shepard et al. (2001,
    J. Geophys. Res. Planets 106, 32777-32795); Morris et al. (2008, JGR Planets
    113, E12007) applied them to Hawaiian volcanic terrains. For a 1-D profile,
    beta = 2H + 1.

    NOTE 1: the numeric Hurst values in those papers were NOT verified against
    the published text (the hosts blocked retrieval) - only the method
    attribution was checked.

    NOTE 2: calibrated end to end through the app against synthetic fractional
    Brownian flanks of known Hurst exponent. Recovered H = 0.29, 0.51, 0.69 and
    0.90 against true values of 0.3, 0.5, 0.7 and 0.9 - essentially unbiased -
    with a per-profile scatter of about +/-0.14. Average over several profiles
    before quoting a value.

    NOTE 3: this measures SELF-AFFINE terrain texture. Uncorrelated (white)
    measurement noise drives beta toward 0 and is not roughness in this sense.

    Returns (beta, H), or (nan, nan) when the flank carries no measurable texture.
    """
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    if len(r) < 160:
        return np.nan, np.nan

    relief = float(np.max(z) - np.min(z))
    if relief <= 0:
        return np.nan, np.nan

    t = int(np.argmax(z))
    span = len(z)
    margin = int(0.10 * span)
    segments = [slice(0, max(t - margin, 0)), slice(min(t + margin, span), span)]

    betas = []
    for seg in segments:
        rr = r[seg]
        zz = z[seg]
        if len(rr) < 64:
            continue
        try:
            resid = zz - np.polyval(np.polyfit(rr, zz, 2), rr)
        except Exception:
            continue
        # Suppress the no-texture case rather than fitting rounding error.
        if float(np.std(resid)) / relief < float(min_relief_fraction):
            continue
        try:
            fftr = compute_fft(rr, resid, window_name="Hann")
        except Exception:
            continue
        f = np.asarray(fftr.frequencies, dtype=float)
        a = np.asarray(fftr.amplitudes, dtype=float)
        mask = (f > 0) & (a > 0) & np.isfinite(f) & np.isfinite(a)
        f = f[mask]
        a = a[mask]
        if len(f) < 12:
            continue
        fmax = float(np.max(f))
        sel = (f >= fmax * float(low_frac)) & (f <= fmax * float(high_frac))
        if int(sel.sum()) < 8:
            continue
        try:
            slope, _ = np.polyfit(np.log10(f[sel]), np.log10(a[sel] ** 2), 1)
        except Exception:
            continue
        betas.append(-float(slope))

    if not betas:
        return np.nan, np.nan
    beta = float(np.mean(betas))
    return beta, float((beta - 1.0) / 2.0)


def compute_edifice_metrics(
    x: np.ndarray,
    y: np.ndarray,
    height_fraction: float = 0.9,
    spec_height: float | None = None,
    harmonic_count: int = 8,
    y_rough: np.ndarray | None = None,
    baseline_mode: str = "fraction",
    vertical_precision: float = 0.5,
    match_quantile: float = 0.95,
) -> tuple[dict, list]:
    """Volcanic edifice metrics from a radial elevation transect through the summit.

    The profile is inverted and handed to the valley core, so the summit is
    located, the basal boundary is set and the areas and power-law fits are done
    by exactly the code the valley path uses.

    Pass y_rough as the UNSMOOTHED elevation. Smoothing stabilises the geometric
    metrics but is a low-pass filter, so it destroys exactly the high-frequency
    content the roughness metric measures: with the default Savitzky-Golay window
    of 21, measured Hurst exponents came back near 1.8, which is not physically
    possible. Geometry uses y; roughness uses y_rough.
    """
    metrics: dict = {}
    notes: list = []

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 16:
        notes.append("Too few points for edifice metrics (16 minimum).")
        return metrics, notes

    inv = -y
    base_metrics, base_notes = compute_cross_section_metrics(
        x, inv,
        height_fraction=height_fraction,
        spec_height=spec_height,
        harmonic_count=harmonic_count,
        baseline_mode=baseline_mode,
        vertical_precision=vertical_precision,
        match_quantile=match_quantile,
    )
    # The inverted profile is what the matched-elevation rule must see, which is
    # why it is resolved inside the valley routine rather than out here: a
    # volcano is a valley with earth and sky exchanged, and the rule is exactly
    # symmetric under that exchange.
    if str(baseline_mode).lower().startswith("match"):
        _mh, _mn = matched_elevation_height(x, inv, vertical_precision, match_quantile)
        if np.isfinite(_mh) and _mh > 0:
            spec_height = _mh
            height_fraction = 1.0
    for n in base_notes:
        notes.append(
            n.replace("valley cross-section", "volcanic edifice")
             .replace("transect", "radial profile")
        )
    if not base_metrics:
        return metrics, notes

    for src_key, dst_key in _VALLEY_TO_EDIFICE_KEYS.items():
        if src_key in base_metrics:
            metrics[dst_key] = base_metrics[src_key]

    height = float(base_metrics.get("benchmark_height", np.nan))
    width = float(base_metrics.get("width_depth_ratio", np.nan)) * height if np.isfinite(height) else np.nan
    if np.isfinite(height):
        metrics["edifice_height"] = height
    if np.isfinite(width) and width > 0:
        metrics["basal_width"] = width
        metrics["height_basal_width_ratio"] = height / width
        metrics["bulk_flank_slope_deg"] = float(np.degrees(np.arctan(2.0 * height / width)))

    win = topo_valley_window(x, inv, height_fraction, spec_height)
    if win is None:
        return metrics, notes
    t, xl, xr, h, _relief = win
    summit_r = float(x[t])

    seg = (x >= xl) & (x <= xr)
    rs = x[seg]
    zs = y[seg]
    if len(rs) >= 8:
        local = np.degrees(np.arctan(np.abs(np.gradient(zs, rs))))
        finite = np.isfinite(local)
        if int(finite.sum()) >= 8:
            # RADIAL (AREA) WEIGHTING. A DEM-based mean slope such as MORVOLC's
            # averages over every cell of the edifice surface, so the outer
            # annuli - which carry area proportional to r*dr - dominate. A plain
            # mean along a profile weights every sample equally and therefore
            # does NOT reproduce that number.
            #
            # Measured against a 2-D reference implementation on synthetic
            # edifices: plain length weighting gave 21.4% RMS error against the
            # DEM definition, over-reading concave stratocones by up to +52% and
            # under-reading convex shields by up to -25%. Radial weighting cuts
            # that to 4.4% RMS, and to under 2.5% for axisymmetric forms.
            radius = np.abs(rs - summit_r)
            w = radius[finite]
            loc = local[finite]
            if float(np.sum(w)) > 0:
                metrics["mean_flank_slope_deg"] = float(np.sum(loc * w) / np.sum(w))
            else:
                metrics["mean_flank_slope_deg"] = float(np.mean(loc))
            # Kept for comparison: this is what a naive profile mean returns.
            metrics["mean_flank_slope_unweighted_deg"] = float(np.mean(loc))
            metrics["max_flank_slope_deg"] = float(np.max(loc))

        z_top = float(np.max(zs))
        near_summit = zs >= (z_top - 0.10 * h)
        if int(near_summit.sum()) >= 2:
            sw = float(np.max(rs[near_summit]) - np.min(rs[near_summit]))
            metrics["summit_width"] = sw
            if np.isfinite(width) and width > 0:
                metrics["summit_basal_width_ratio"] = sw / width

        metrics["karatson_upper_slope_coef"] = _karatson_upper_slope_coefficient(rs, zs, summit_r, h)

        z_rough = zs if y_rough is None else np.asarray(y_rough, dtype=float)[seg]
        beta, hurst = _flank_roughness(rs, z_rough)
        if np.isfinite(beta):
            metrics["flank_roughness_beta"] = beta
            metrics["flank_hurst_exponent"] = hurst
            if not (0.0 <= hurst <= 1.0):
                notes.append(
                    f"Flank Hurst exponent came out at {hurst:.2f}, outside the physical range 0 to 1. "
                    "That normally means the elevation reaching the roughness fit was smoothed or "
                    "under-sampled relative to the texture being measured. Reduce the smoothing window "
                    "or use a denser profile before quoting this number."
                )

    metrics = {k: v for k, v in metrics.items() if v is None or np.isfinite(v)}
    return metrics, notes

LANDFORM_METRIC_LABELS.update(VOLCANO_METRIC_LABELS)
LANDFORM_METRIC_UNITS.update(VOLCANO_METRIC_UNITS)
TOPO_DATASET_TYPES = (TOPO_TRANSECT, TOPO_LONG_PROFILE, TOPO_VOLCANO)

DATASET_PRESETS[TOPO_VOLCANO] = {
    "x_label": "Distance along radial profile",
    "y_label": "Elevation",
    "normalize": False,
    "smoothing_on": True,
    "smoothing_method": "Savitzky-Golay",
    "smoothing_window": 21,
    "baseline_on": False,
    "baseline_method": "Linear",
    "asls_lambda": 1e5,
    "asls_p": 0.01,
    "asls_niter": 10,
    "detrend_on": False,
    "interpolate_on": True,
    "peak_prominence": 0.0,
    "peak_distance": 5,
    "peak_width": 0.0,
    "topo_height_fraction": 0.90,
}
DATASET_PRESETS[TOPO_TRANSECT]["topo_height_fraction"] = 0.50
DATASET_PRESETS[TOPO_LONG_PROFILE]["topo_height_fraction"] = 0.50


# ------------------------------------------------------------
# Scoring
# ------------------------------------------------------------
@dataclass
class TopographyInterpretationResult:
    metrics: pd.DataFrame
    candidates: pd.DataFrame
    evidence: pd.DataFrame
    top_candidate: str | None
    explanation: str
    caveats: list


def summarize_topo_confidence(top_score: float, margin: float, scored_metrics: int) -> str:
    """Deliberately capped at 'moderate'.

    The published fluvial and glacial V-index distributions overlap heavily
    (0.06 +/- 0.14 against 0.19 +/- 0.16), so a single profile cannot support a
    high-confidence call no matter how the numbers fall.
    """
    if scored_metrics < 2:
        return "very low"
    if top_score >= 0.70 and margin >= 0.25:
        return "moderate"
    if top_score >= 0.50 and margin >= 0.12:
        return "low"
    return "very low"


def topo_metrics_dataframe(metrics: dict, reference_df: pd.DataFrame, profile_type: str) -> pd.DataFrame:
    scored_metrics = set()
    if reference_df is not None and not reference_df.empty:
        active = reference_df[
            (reference_df["profile_type"].astype(str) == profile_type)
            & (reference_df["weight"] > 0)
        ]
        scored_metrics = set(active["metric"].tolist())

    rows = []
    for key, value in metrics.items():
        rows.append({
            "Metric": LANDFORM_METRIC_LABELS.get(key, key),
            "Key": key,
            "Value": value,
            "Units": LANDFORM_METRIC_UNITS.get(key, ""),
            "Used in score": "yes" if key in scored_metrics else "no (reported only)",
        })
    if not rows:
        return pd.DataFrame(columns=["Metric", "Key", "Value", "Units", "Used in score"])
    return pd.DataFrame(rows).sort_values(["Used in score", "Metric"]).reset_index(drop=True)


def interpret_topography(
    metrics: dict,
    profile_type: str,
    reference_df: pd.DataFrame,
) -> TopographyInterpretationResult:
    """Score continuous landform metrics against class centres.

    Landforms have no diagnostic spectral positions to match against a line
    list. Each metric is instead
    scored as a Gaussian likelihood against a published class centre and spread,
    and the weighted mean likelihood is the class score. Overlapping classes
    therefore produce genuinely close scores rather than a false winner.
    """
    empty_metrics = pd.DataFrame(columns=["Metric", "Key", "Value", "Units", "Used in score"])
    empty_candidates = pd.DataFrame(
        columns=["Landform", "Score", "Relative support (%)", "Confidence", "Metrics used", "Evidence"]
    )
    empty_evidence = pd.DataFrame(
        columns=["Landform", "Metric", "Observed", "Class centre", "Class s.d.", "z", "Likelihood", "Weight", "Rule", "Source"]
    )

    caveats = [
        "Published fluvial and glacial V-index distributions overlap substantially "
        "(0.06 +/- 0.14 against 0.19 +/- 0.16, Zimmer 2016). A single profile can narrow the "
        "possibilities; it cannot settle them.",
        "The power-law exponent b is sensitive to the choice of origin (Li, Liu & Cui, 2001). "
        "This app fixes the origin at the profile minimum (thalweg) and also reports each wall "
        "separately so origin sensitivity is visible.",
        "The tectonic (fault-bounded) class carries NO calibrated metric in the built-in library. "
        "It cannot currently win a scoring run. Calibrate asymmetry on labelled profiles first.",
        "The power-law exponent b is markedly more noise-sensitive than the V-index. On a "
        "synthetic parabola carrying 3 m of noise on 200 m of relief, b read 0.67 unsmoothed "
        "against a true 2.00, while the V-index held at 0.26 against a true 0.333. If the two "
        "metrics disagree, trust the V-index and widen the smoothing window.",
        "FFT harmonic decay does NOT separate V from U sections: both a triangular (V) and a "
        "parabolic (U) cross-section decay as 1/n^2, verified numerically. The even/odd harmonic "
        "ratio does order V to U correctly, but asymmetry raises it too, so it is only meaningful "
        "read together with asymmetry_index. Both are shipped unweighted.",
        "Amerson et al. (2007, Geomorphology) found glaciated and fluvial valley scaling to differ "
        "modestly and, for width, not significantly. Expect weak separation, not clean classes.",
    ]

    if profile_type == TOPO_VOLCANO:
        caveats = [
            "The 12 degree and 0.10 H/W_B thresholds are EARTH-CALIBRATED (Grosse & Kervyn, 2018, "
            "from terrestrial SRTM data) and are not rescaled for gravity here, because no published "
            "gravity scaling for edifice slope was located to justify one. Martian shields sit at the "
            "low end of the terrestrial envelope (Plescia, 2004: Olympus Mons H/W 0.025 at 5.2 deg), "
            "so the threshold direction holds for Mars. It is NOT validated for higher-gravity bodies, "
            "and it is meaningless for ice edifices on Ceres, Pluto, Titan or Enceladus, whose rheology "
            "is not that of silicate lava.",
            "Grosse et al. (2013, Bull. Volcanol. 76:784) measured 759 composite volcanoes and report "
            "that most morphometric parameters show large variation without clear separations. Overlap "
            "between shield and stratovolcano is expected, not a failure of the measurement.",
            "Defining the basal boundary - where the edifice ends and the surrounding plain begins - is "
            "the dominant uncertainty in volcano morphometry. Here it is set by the basal boundary "
            "fraction. Vary that control and watch how far the result moves before trusting any number.",
            "One radial profile is not the edifice. Volcanoes are not radially symmetric: sector "
            "collapses, rift zones and flank vents bias any single azimuth. Run several azimuths per "
            "edifice and compare rather than relying on one transect.",
            "Erosion and degradation flatten stratocones toward shield-like slopes, and on airless "
            "bodies impact gardening and mantling do the same. A morphometric class is a statement "
            "about present shape, not about eruptive style or composition.",
            "Mean flank slope here is the mean of local |dz/dr| along ONE profile. The 12 degree "
            "threshold is an edifice-wide average from a full DEM. Related quantities, not identical "
            "ones - do not read exact agreement into them. Note also that pixel-scale slope "
            "distributions (Bleacher & Greeley, 2003: Kilauea 8 deg, inactive Hawaiian shields 19-20 "
            "deg median) are a third, different quantity again.",
            "VALIDATION AGAINST THE DEM METHOD. Mean flank slope here is area-weighted so that it "
            "reproduces a DEM-based mean rather than a naive profile mean. Measured against a 2-D "
            "reference on synthetic edifices: H/W_B agrees to within 1-2 percent for circular to "
            "moderately elliptical forms, converging by 2-4 azimuths, degrading to about 11 percent "
            "at 2.5:1 ellipticity; area-weighted mean slope agrees to 4.4 percent RMS overall and "
            "under 2.5 percent for axisymmetric forms. Slope agreement holds to about 2 percent while "
            "surface roughness stays under roughly 1.5 percent of relief, then falls away to about "
            "-22 percent at 10 percent of relief, because a single profile samples only the "
            "along-track gradient and misses the cross-track component that a 2-D slope includes.",
            "Irregular PLAN shape - lobes, sector collapses, strong ellipticity - is where profile "
            "sampling has irreducible error. Adding azimuths converges the estimate quickly but does "
            "not remove that bias. Treat a strongly non-circular edifice as outside this tool's "
            "reliable range and say so rather than quoting a number.",
            "MORVOLC (Grosse et al., 2012, Geomorphology 136:114-131) already performs systematic "
            "terrestrial edifice morphometry from DEMs. Nothing in this tab improves on it for Earth. "
            "Any original contribution lies in the consistent cross-body profile workflow, not in the "
            "terrestrial metrics.",
            "The Karatson upper-flank slope coefficient is reported but NOT scored. Karatson et al. "
            "(2010, J. Volcanol. Geotherm. Res. 193:171-181) split regular stratovolcanoes into C-type "
            "(conical) and P-type (concave) at -50e-4 deg/m, correlating with SiO2 and H2O content. "
            "That threshold applies to symmetrical terrestrial stratovolcanoes and was not tested here "
            "on shields or on other bodies.",
        ]

    metrics_df = topo_metrics_dataframe(metrics, reference_df, profile_type)

    if not metrics:
        return TopographyInterpretationResult(
            empty_metrics, empty_candidates, empty_evidence, None,
            "No landform metrics could be computed for this profile.", caveats,
        )

    if reference_df is None or reference_df.empty:
        return TopographyInterpretationResult(
            metrics_df, empty_candidates, empty_evidence, None,
            "Metrics were computed but no landform reference table is loaded, so nothing was scored.",
            caveats,
        )

    ref = reference_df[reference_df["profile_type"].astype(str) == profile_type]
    ref = ref[(ref["weight"] > 0) & (ref["metric"].isin(metrics.keys()))]

    if ref.empty:
        return TopographyInterpretationResult(
            metrics_df, empty_candidates, empty_evidence, None,
            "Metrics were computed, but no reference row for this profile type has a non-zero "
            "weight and a matching metric, so no landform was scored. This is the expected state "
            "until the library is calibrated.",
            caveats,
        )

    candidates = []
    evidence_rows = []

    for landform, group in ref.groupby("landform"):
        numerator = 0.0
        denominator = 0.0
        used = 0
        for _, row in group.iterrows():
            key = row["metric"]
            value = float(metrics[key])
            if not np.isfinite(value):
                continue
            center = float(row["center"])
            sd = float(row["sd"])
            weight = float(row["weight"])
            z = (value - center) / sd
            rule = str(row.get("rule", "gaussian") or "gaussian")
            if rule == "threshold_below":
                # Logistic: full support well below the threshold, falling through it.
                likelihood = float(1.0 / (1.0 + np.exp(np.clip(z, -60.0, 60.0))))
            elif rule == "threshold_above":
                likelihood = float(1.0 / (1.0 + np.exp(np.clip(-z, -60.0, 60.0))))
            else:
                rule = "gaussian"
                likelihood = float(np.exp(-0.5 * z * z))
            numerator += weight * likelihood
            denominator += weight
            used += 1
            evidence_rows.append({
                "Landform": landform,
                "Metric": LANDFORM_METRIC_LABELS.get(key, key),
                "Observed": round(value, 4),
                "Class centre": center,
                "Class s.d.": sd,
                "z": round(z, 3),
                "Likelihood": round(likelihood, 4),
                "Weight": weight,
                "Rule": rule,
                "Source": row.get("source", ""),
            })

        if denominator <= 0 or used == 0:
            continue

        candidates.append({
            "Landform": landform,
            "Score": round(numerator / denominator, 4),
            "Metrics used": used,
        })

    if not candidates:
        return TopographyInterpretationResult(
            metrics_df, empty_candidates, empty_evidence, None,
            "No landform class could be scored from the available metrics.", caveats,
        )

    candidates_df = pd.DataFrame(candidates).sort_values("Score", ascending=False).reset_index(drop=True)
    total_score = float(candidates_df["Score"].sum())
    if total_score > 0:
        candidates_df["Relative support (%)"] = (candidates_df["Score"] / total_score * 100.0).round(1)
    else:
        candidates_df["Relative support (%)"] = 0.0

    top_score = float(candidates_df.iloc[0]["Score"])
    second_score = float(candidates_df.iloc[1]["Score"]) if len(candidates_df) > 1 else 0.0
    margin = top_score - second_score
    scored_metric_count = int(candidates_df.iloc[0]["Metrics used"])
    confidence = summarize_topo_confidence(top_score, margin, scored_metric_count)
    candidates_df["Confidence"] = [
        confidence if i == 0 else "" for i in range(len(candidates_df))
    ]
    candidates_df = candidates_df[
        ["Landform", "Score", "Relative support (%)", "Confidence", "Metrics used"]
    ]

    evidence_df = pd.DataFrame(evidence_rows)
    top_candidate = str(candidates_df.iloc[0]["Landform"])

    explanation = (
        "Best-supported class: "
        + top_candidate
        + " (score "
        + str(round(top_score, 3))
        + " of a possible 1.0, "
        + str(candidates_df.iloc[0]["Relative support (%)"])
        + "% relative support, "
        + str(scored_metric_count)
        + " calibrated metric(s), confidence: "
        + confidence
        + "). Margin over the next class is "
        + str(round(margin, 3))
        + ". This is a first-pass morphometric screen against published class statistics, "
        + "not a determination of valley genesis."
    )

    return TopographyInterpretationResult(
        metrics=metrics_df,
        candidates=candidates_df,
        evidence=evidence_df,
        top_candidate=top_candidate,
        explanation=explanation,
        caveats=caveats,
    )


@dataclass
class PeakResults:

    indices: np.ndarray
    x: np.ndarray
    y: np.ndarray
    properties: dict
    table: pd.DataFrame


@dataclass
class FFTResults:
    frequencies: np.ndarray
    amplitudes: np.ndarray
    table: pd.DataFrame


LOCAL_OLLAMA_URL = os.getenv("OLLAMA_LOCAL_URL", "http://127.0.0.1:11434")
AI_SYSTEM_PROMPT = ("You are a careful scientific assistant. Use only the supplied analysis context. "
                    "Be concise, clear about uncertainty, and do not invent results.")


def ollama_available(base_url: str = LOCAL_OLLAMA_URL) -> bool:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/tags", timeout=1.5)
        return resp.ok
    except Exception:
        return False


def call_ollama(prompt: str, model: str, base_url: str = LOCAL_OLLAMA_URL) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    resp = requests.post(f"{base_url.rstrip('/')}/api/generate", json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return str(data.get("response", "")).strip()


def ai_warranted(dataset_type: str, peak_results) -> bool:
    if dataset_type in TOPO_DATASET_TYPES:
        return True
    return peak_results is not None and len(getattr(peak_results, "indices", [])) >= 3


def build_ai_context(dataset_type: str, x_label: str, y_label: str, peak_results, fft_results, notes: list[str], topo_interpretation=None) -> str:
    lines = [
        f"Dataset type: {dataset_type}",
        f"Axes: {x_label} vs {y_label}",
    ]
    if peak_results is not None:
        lines.append(f"Detected peaks: {len(peak_results.indices)}")
        if not peak_results.table.empty:
            top_peaks = peak_results.table.sort_values("Intensity", ascending=False).head(8)
            peak_text = ", ".join([f"{row['X Position']:.4g} ({row['Intensity']:.4g})" for _, row in top_peaks.iterrows()])
            lines.append(f"Top peaks: {peak_text}")
    if fft_results is not None and not fft_results.table.empty:
        dom = fft_results.table.iloc[0]
        lines.append(f"Dominant FFT frequency: {dom['Frequency']:.4g}")
        lines.append(f"Dominant FFT period: {dom['Period']:.4g}")
    if topo_interpretation is not None:
        lines.append(f"Landform explanation: {topo_interpretation.explanation}")
        if not topo_interpretation.metrics.empty:
            lines.append("Landform metrics: " + "; ".join([
                f"{row['Metric']} = {row['Value']}" for _, row in topo_interpretation.metrics.iterrows()
            ]))
        if topo_interpretation.caveats:
            lines.append("Landform caveats: " + " ".join(topo_interpretation.caveats))
    if notes:
        lines.append("Processing notes: " + "; ".join(notes))
    return "\n".join(lines)


def builtin_summary(context: str) -> str:
    """Plain summary used when no AI model can answer, so the assistant panel always gives something useful."""
    lines = [f"- {line}" for line in context.splitlines() if line.strip()]
    return ("**Built-in summary** (the AI models are busy or unavailable right now, so this lists what the analysis found):\n\n"
            + "\n".join(lines)
            + "\n\nCheck the measured shape against the reference landform classes, and treat low-confidence matches as hypotheses to test.")


def generate_ai_summary(dataset_type: str, x_label: str, y_label: str, peak_results, fft_results, notes: list[str], question: str, topo_interpretation=None, prefer_local: bool = True, ollama_model: str = "llama3.1:8b") -> tuple[str, str]:
    """Answer with local Ollama (if preferred and running), then free Ollama Cloud models, then a built-in summary."""
    context = build_ai_context(dataset_type, x_label, y_label, peak_results, fft_results, notes, topo_interpretation)
    prompt = (
        "Use the following Spectrum Earth analysis context to answer the user. "
        "Keep it concise, mention uncertainty, and stay within the evidence provided.\n\n"
        + context
        + "\n\nUser request: " + question
    )

    if prefer_local and ollama_available():
        try:
            return call_ollama(prompt, ollama_model), f"Local Ollama ({ollama_model})"
        except Exception:
            pass

    reply = ai_client.chat([{"role": "system", "content": AI_SYSTEM_PROMPT}, {"role": "user", "content": prompt}], temperature=0.2)
    if reply:
        return reply, f"Ollama Cloud ({ai_client.last_model})"

    return builtin_summary(context), "Built-in summary (no AI model available)"


def init_state():
    defaults = {
        "raw_df": None,
        "processed_df": None,
        "baseline_df": None,
        "x_col": None,
        "y_col": None,
        "dataset_type": TOPO_TRANSECT,
        "x_label": DATASET_PRESETS[TOPO_TRANSECT]["x_label"],
        "y_label": DATASET_PRESETS[TOPO_TRANSECT]["y_label"],
        "plot_title": "Spectrum Earth Profile",
        "peak_results": None,
        "fft_results": None,
        "status_notes": [],
        "last_preset_dataset": None,
        "normalize": False,
        "log_y": False,
        "crop_enabled": False,
        "show_peak_labels": True,
        "label_top_n": 8,
        "fft_window": "Hann",
        "fft_remove_zero": True,
        "fft_view_mode": "Frequency",
        "fft_log_amplitude": False,
        "fft_log_x": False,
        "fft_smooth_display": True,
        "fft_display_smooth_window": 7,
        "fft_detect_peaks": True,
        "fft_peak_prominence": 0.0,
        "fft_peak_distance": 3,
        "fft_max_fraction": 0.35,
        "ai_enable": False,
        "ai_prefer_local": True,
        "ollama_model": "llama3.1:8b",
        "ai_last_provider": "",
        "ai_last_response": "",
        "ai_question": "Give me a concise interpretation of this run.",
        "ideal_chart_rows": [],
        "ideal_distance_unit": "metres",
        "ideal_elevation_unit": "metres",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    for key, value in TOPO_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "topo_interpretation" not in st.session_state:
        st.session_state["topo_interpretation"] = None
    if "topo_metrics" not in st.session_state:
        st.session_state["topo_metrics"] = None

    preset = DATASET_PRESETS[TOPO_TRANSECT]
    for key, value in preset.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_dataset_preset(dataset_type: str):
    preset = DATASET_PRESETS[dataset_type]
    for key, value in preset.items():
        st.session_state[key] = value
    st.session_state["x_label"] = preset["x_label"]
    st.session_state["y_label"] = preset["y_label"]
    st.session_state["plot_title"] = f"Spectrum Earth - {dataset_type}"
    st.session_state["last_preset_dataset"] = dataset_type


def note(msg: str):
    st.session_state.status_notes.append(msg)


SAMPLE_FILES = {   # synthetic profiles in samples/; the Dataset Type each one suits is in its label
    "Volcano: perfect cone (reference)": "sample_volcano_perfect_cone_reference.csv",
    "Volcano: stratocone, Fuji-like": "sample_volcano_stratocone_fuji_like.csv",
    "Volcano: stratocone, Mayon-like": "sample_volcano_stratocone_mayon_like.csv",
    "Volcano: shield, Mauna Loa-like": "sample_volcano_shield_maunaloa_like.csv",
    "Volcano: shield, Olympus Mons-like (Mars)": "sample_volcano_shield_olympus_like_mars.csv",
    "Transect: V-shaped river valley": "sample_transect_V_fluvial.csv",
    "Transect: U-shaped glacial valley": "sample_transect_U_glacial.csv",
    "Transect: flat-floored trough": "sample_transect_flatfloor_trough.csv",
    "Transect: asymmetric fault scarp": "sample_transect_asymmetric_fault.csv",
    "Long profile: concave river": "sample_longprofile_fluvial_concave.csv",
    "Long profile: overdeepened glacial": "sample_longprofile_glacial_overdeepened.csv",
    # Real transects, from Spectrum Jax
    "Real volcano: Shishaldin, clean cone": "sample_real_volcano_shishaldin_clean_cone.csv",
    "Real volcano: Taranaki, stratovolcano": "sample_real_volcano_taranaki_stratovolcano.csv",
    "Real volcano: Piton de la Fournaise, shield": "sample_real_volcano_pitondelafournaise_shield.csv",
    "Real volcano: Newberry, broad with caldera": "sample_real_volcano_newberry_broad_with_caldera.csv",
    "Real volcano: Crater Lake, caldera": "sample_real_volcano_craterlake_caldera.csv",
    # Real mid-ocean ridge cross-sections, ETOPO1 swath medians centred on the axis (README)
    "Real ridge: SW Indian Ridge 10°E, ultraslow (14 mm/yr)": "sample_real_ridge_swir_10e_ultraslow.csv",
    "Real ridge: Mohns Ridge 72°N, ultraslow (16 mm/yr)": "sample_real_ridge_mohns_72n_ultraslow.csv",
    "Real ridge: Mid-Atlantic Ridge 26°N, slow (25 mm/yr)": "sample_real_ridge_mar_26n_slow.csv",
    "Real ridge: Mid-Atlantic Ridge 29°N, slow (24 mm/yr)": "sample_real_ridge_mar_29n_slow.csv",
    "Real ridge: Juan de Fuca 45°N, intermediate (56 mm/yr)": "sample_real_ridge_juandefuca_45n_intermediate.csv",
    "Real ridge: SE Indian Ridge 98°E, intermediate (70 mm/yr)": "sample_real_ridge_seir_98e_intermediate.csv",
    "Real ridge: East Pacific Rise 12.5°N, fast (90 mm/yr)": "sample_real_ridge_epr_12n_fast.csv",
    "Real ridge: East Pacific Rise 9.5°N, fast (110 mm/yr)": "sample_real_ridge_epr_9n_fast.csv",
    "Real ridge: East Pacific Rise 17.5°S, superfast (145 mm/yr)": "sample_real_ridge_epr_17s_superfast.csv",
}


def sample_dataset_type(label: str) -> str:
    """The Dataset Type a sample profile suits, read from its menu label."""
    if label.startswith("Volcano") or label.startswith("Real volcano"):
        return TOPO_VOLCANO
    if label.startswith("Long profile"):
        return TOPO_LONG_PROFILE
    return TOPO_TRANSECT


def sample_feature(label: str) -> str:
    """The 'What is the feature?' choice a sample profile suits."""
    if label.startswith("Real ridge"):
        return ideal_view.RIDGE_FEATURE
    return ideal_view.default_feature(sample_dataset_type(label), TOPO_VOLCANO, TOPO_LONG_PROFILE)


def on_sample_change():
    choice = st.session_state.sample_choice
    if choice != "None":
        st.session_state.dataset_type = sample_dataset_type(choice)
        apply_dataset_preset(st.session_state.dataset_type)
        # The feature choice follows the sample (a ridge and a valley are both transects).
        st.session_state[f"ideal_feature_{st.session_state.dataset_type}"] = sample_feature(choice)


def read_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)

    if name.endswith(".txt"):
        raw = uploaded_file.read()
        uploaded_file.seek(0)
        text = raw.decode("utf-8", errors="ignore")
        for sep in [None, "\t", ",", ";", r"\s+"]:
            try:
                if sep is None:
                    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
                else:
                    df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
                if df.shape[1] >= 2:
                    return df
            except Exception:
                continue
        raise ValueError("Could not parse TXT file.")

    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)

    raise ValueError("Unsupported file type. Use CSV, TXT, or XLSX.")


def clean_numeric_xy(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    out = df[[x_col, y_col]].copy()
    out.columns = ["x", "y"]
    out["x"] = pd.to_numeric(out["x"], errors="coerce")
    out["y"] = pd.to_numeric(out["y"], errors="coerce")
    out = out.dropna().sort_values("x").reset_index(drop=True)
    return out


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    numeric_cols = []
    for col in df.columns:
        series = pd.to_numeric(df[col], errors="coerce")
        if series.notna().sum() > 0:
            numeric_cols.append(col)
    return numeric_cols


def normalize_signal(y: np.ndarray) -> np.ndarray:
    y_min = np.min(y)
    y_max = np.max(y)
    if np.isclose(y_max, y_min):
        return y.copy()
    return (y - y_min) / (y_max - y_min)


def moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def apply_savgol(y: np.ndarray, window: int, polyorder: int = 3) -> np.ndarray:
    n = len(y)
    if n < 7:
        return y.copy()

    window = max(5, int(window))
    if window % 2 == 0:
        window += 1
    if window >= n:
        window = n - 1 if n % 2 == 0 else n
    if window < 5:
        window = 5
    if window % 2 == 0:
        window -= 1
    if window <= polyorder:
        window = polyorder + 2
        if window % 2 == 0:
            window += 1
    if window >= n:
        return y.copy()

    return savgol_filter(y, window_length=window, polyorder=polyorder)


def detrend_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coeffs = np.polyfit(x, y, 1)
    trend = coeffs[0] * x + coeffs[1]
    return y - trend


def baseline_linear(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coeffs = np.polyfit(x, y, 1)
    return coeffs[0] * x + coeffs[1]


def baseline_asls(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    L = len(y)
    if L < 3:
        return np.zeros_like(y)

    D = sparse.diags([1, -2, 1], [0, 1, 2], shape=(L - 2, L))
    w = np.ones(L)

    for _ in range(int(niter)):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.T @ D
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)

    return np.asarray(z)


def is_evenly_spaced(x: np.ndarray, rtol: float = 1e-3) -> bool:
    if len(x) < 3:
        return True
    dx = np.diff(x)
    return np.allclose(dx, dx[0], rtol=rtol, atol=0)


def interpolate_even_spacing(x: np.ndarray, y: np.ndarray, num_points: int | None = None):
    if num_points is None:
        num_points = len(x)
    new_x = np.linspace(np.min(x), np.max(x), num_points)
    new_y = np.interp(new_x, x, y)
    return new_x, new_y


def trapezoid_peak_areas(x: np.ndarray, y: np.ndarray, peak_indices: np.ndarray) -> list[float]:
    if len(peak_indices) == 0:
        return []

    areas = []
    boundaries = [0] + [int((peak_indices[i] + peak_indices[i + 1]) / 2) for i in range(len(peak_indices) - 1)] + [len(x) - 1]

    for i, _peak_idx in enumerate(peak_indices):
        left = boundaries[i]
        right = boundaries[i + 1]
        if right <= left:
            areas.append(0.0)
        else:
            areas.append(float(np.trapezoid(y[left:right + 1], x[left:right + 1])))
    return areas


def detect_peaks(x: np.ndarray, y: np.ndarray, prominence: float, distance: int, width: float) -> PeakResults:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise ValueError("X and Y must have the same length for peak detection.")
    if len(y) < 3:
        empty = pd.DataFrame(columns=["Peak #", "X Position", "Intensity", "Area"])
        return PeakResults(indices=np.array([], dtype=int), x=np.array([]), y=np.array([]), properties={}, table=empty)

    finite_mask = np.isfinite(x) & np.isfinite(y)
    x = x[finite_mask]
    y = y[finite_mask]

    if len(y) < 3 or np.allclose(y, y[0]):
        empty = pd.DataFrame(columns=["Peak #", "X Position", "Intensity", "Area"])
        return PeakResults(indices=np.array([], dtype=int), x=np.array([]), y=np.array([]), properties={}, table=empty)

    prominence = max(float(prominence), 0.0)
    distance = max(int(distance), 1)
    width = max(float(width), 0.0)

    idx, props = find_peaks(y, prominence=prominence, distance=distance, width=width)

    if len(idx) == 0:
        table = pd.DataFrame(columns=["Peak #", "X Position", "Intensity", "Area"])
        return PeakResults(indices=idx, x=np.array([]), y=np.array([]), properties=props, table=table)

    peak_x = x[idx]
    peak_y = y[idx]
    areas = trapezoid_peak_areas(x, y, idx)

    table = pd.DataFrame({
        "Peak #": np.arange(1, len(idx) + 1),
        "X Position": peak_x,
        "Intensity": peak_y,
        "Area": areas,
    })

    if "prominences" in props:
        table["Prominence"] = props["prominences"]
    if "widths" in props:
        table["Width"] = props["widths"]

    table = table.sort_values("Intensity", ascending=False).reset_index(drop=True)
    return PeakResults(indices=idx, x=peak_x, y=peak_y, properties=props, table=table)



def compute_fft(x: np.ndarray, y: np.ndarray, window_name: str = "None") -> FFTResults:
    if len(x) < 4:
        return FFTResults(
            frequencies=np.array([]),
            amplitudes=np.array([]),
            table=pd.DataFrame(columns=["Frequency", "Period", "Amplitude"])
        )

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = np.mean(np.diff(x))
    y_centered = y - np.mean(y)

    n = len(y_centered)
    if window_name == "Hann":
        window = np.hanning(n)
    elif window_name == "Hamming":
        window = np.hamming(n)
    elif window_name == "Blackman":
        window = np.blackman(n)
    else:
        window = np.ones(n)

    yf = rfft(y_centered * window)
    xf = rfftfreq(n, d=dx)
    amp = np.abs(yf)
    period = np.where(xf > 0, 1.0 / xf, np.inf)

    table = pd.DataFrame({"Frequency": xf, "Period": period, "Amplitude": amp})
    table = table.sort_values("Amplitude", ascending=False).reset_index(drop=True)
    return FFTResults(frequencies=xf, amplitudes=amp, table=table)


def smooth_fft_display(amplitudes: np.ndarray, window: int = 7) -> np.ndarray:
    amps = np.asarray(amplitudes, dtype=float)
    if len(amps) < 3:
        return amps.copy()
    return moving_average(amps, max(3, int(window)))


def detect_fft_peaks(freq: np.ndarray, amp: np.ndarray, prominence: float = 0.0, distance: int = 3) -> pd.DataFrame:
    freq = np.asarray(freq, dtype=float)
    amp = np.asarray(amp, dtype=float)
    mask = np.isfinite(freq) & np.isfinite(amp)
    freq = freq[mask]
    amp = amp[mask]

    if len(freq) < 3 or np.allclose(amp, amp[0]):
        return pd.DataFrame(columns=["Frequency", "Period", "Amplitude", "Prominence"])

    distance = max(1, int(distance))
    if prominence <= 0:
        prominence = max(float(np.nanmax(amp)) * 0.05, 1e-12)

    idx, props = find_peaks(amp, prominence=prominence, distance=distance)
    if len(idx) == 0:
        return pd.DataFrame(columns=["Frequency", "Period", "Amplitude", "Prominence"])

    out = pd.DataFrame({
        "Frequency": freq[idx],
        "Period": np.where(freq[idx] > 0, 1.0 / freq[idx], np.inf),
        "Amplitude": amp[idx],
        "Prominence": props.get("prominences", np.full(len(idx), np.nan)),
    })
    return out.sort_values("Amplitude", ascending=False).reset_index(drop=True)


def make_fft_plot(
    x_vals: np.ndarray,
    amp_vals: np.ndarray,
    view_mode: str = "Frequency",
    log_amplitude: bool = False,
    log_x: bool = False,
    labeled_peaks: pd.DataFrame | None = None,
):
    x_title = "Frequency" if view_mode == "Frequency" else "Period"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=amp_vals,
        mode="lines",
        name="FFT Spectrum",
        hovertemplate=f"{x_title}: %{{x:.4g}}<br>Amplitude: %{{y:.4g}}<extra></extra>",
    ))

    if labeled_peaks is not None and not labeled_peaks.empty:
        label_x = labeled_peaks["Frequency"].to_numpy() if view_mode == "Frequency" else labeled_peaks["Period"].to_numpy()
        label_text = [f"{v:.3g}" for v in label_x]
        fig.add_trace(go.Scatter(
            x=label_x,
            y=labeled_peaks["Amplitude"].to_numpy(),
            mode="markers+text",
            name="FFT Peaks",
            text=label_text,
            textposition="top center",
            marker=dict(size=9, symbol="diamond"),
            hovertemplate=f"{x_title}: %{{x:.4g}}<br>Amplitude: %{{y:.4g}}<extra></extra>",
        ))

    fig.update_layout(
        title=f"FFT / {'Frequency' if view_mode == 'Frequency' else 'Period'} View",
        xaxis_title=x_title,
        yaxis_title="Amplitude",
        template="plotly_white",
        height=560,
        margin=dict(l=20, r=20, t=60, b=20),
    )

    if log_amplitude:
        fig.update_yaxes(type="log")
    if log_x:
        fig.update_xaxes(type="log")

    return fig


def load_reference_table(uploaded_file) -> pd.DataFrame:
    df = read_uploaded_file(uploaded_file)
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "mineral" not in df.columns:
        raise ValueError("Reference table needs a 'mineral' column.")

    if "two_theta" not in df.columns and "d_spacing" not in df.columns:
        raise ValueError("Reference table needs 'two_theta' or 'd_spacing'.")

    if "relative_intensity" not in df.columns:
        df["relative_intensity"] = 100.0

    return df


def format_peak_label(x_value: float, dataset_type: str) -> str:
    return f"{x_value:.2f}"


# ------------------------------------------------------------
# Shape register: idealized reference forms in metric space.
#
# Every reference form is measured with compute_cross_section_metrics - the same
# function real profiles go through - so the register and the measurement can
# never drift apart. Change the metric and the register moves with it.
#
# The organising identity: for any pure power-law wall z = h*(|x|/W)**b,
#
#       V-index = (b - 1) / (b + 1)
#
# verified to four decimals for b from 0.5 to 4.0. V-index and b are therefore
# the SAME quantity for a self-similar section, not two independent measurements.
# That makes the power-law relation a reference manifold rather than a grid, and
# the useful quantity is departure from it:
#
#       delta = V_measured - (b_fit - 1) / (b_fit + 1)
#
# Measured on the idealized families: delta is 0 for pure power-law walls,
# NEGATIVE for a flat floor (-0.05 to -0.09) and POSITIVE for asymmetry (up to
# +0.26) or a tightly curved arc (+0.16). Its sign says which way a section
# departs from self-similarity.
# ------------------------------------------------------------

SHAPE_REGISTER_FAMILIES = {
    "Power-law walls": {
        "param": "b",
        "values": [0.4, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0],
        "colour": "#2a78d6",
        "note": "z = h(|x|/W)^b. The reference manifold. b=1 straight V, b=2 parabola.",
    },
    "Triangle aspect": {
        "param": "width_depth",
        "values": [4.0, 6.0, 8.0, 10.0, 13.0, 16.0, 20.0],
        "colour": "#eb6834",
        "note": "Isoceles triangles. Every one sits at b=1 regardless of proportion, so this family IS the aspect axis.",
    },
    "Circular arc": {
        "param": "radius_over_width",
        "values": [1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0],
        "colour": "#1baf7a",
        "note": "Converges on the parabola point as R/W grows. A shield-like form.",
    },
    "Flat floor": {
        "param": "floor_fraction",
        "values": [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7],
        "colour": "#eda100",
        "note": "V walls with a flat floor. Drives delta NEGATIVE.",
    },
    "Asymmetric walls": {
        "param": "steepness_ratio",
        "values": [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 5.0],
        "colour": "#e87ba4",
        "note": "One wall steeper. Drives delta POSITIVE and pulls width/depth down.",
    },
    "Composite V-in-U": {
        "param": "gorge_fraction",
        "values": [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8],
        "colour": "#008300",
        "note": "A V inner gorge inside a parabolic trough - the polygenetic case. Its track shows how a re-incised valley migrates through the register.",
    },
}


def generate_ideal_profile(
    family: str,
    value: float,
    n_points: int = 2001,
    half_width: float = 1000.0,
    height: float = 200.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One idealized cross-section, normalised to a standard height and width.

    Everything here is scale-free by construction: the register must not depend
    on how big the landform is, only on its shape.
    """
    x = np.linspace(-half_width, half_width, int(n_points))
    ax = np.abs(x)

    if family == "Power-law walls":
        z = height * (ax / half_width) ** float(value)

    elif family == "Triangle aspect":
        # width/depth ratio w fixes half_width/height = w/2
        w = max(float(value), 0.5)
        h = 2.0 * half_width / w
        z = h * (ax / half_width)
        height = h

    elif family == "Circular arc":
        R = max(float(value), 1.0) * half_width
        z = R - np.sqrt(np.maximum(R ** 2 - x ** 2, 0.0))
        peak = float(np.max(z))
        z = z / peak * height if peak > 0 else z

    elif family == "Flat floor":
        f = float(np.clip(value, 0.0, 0.9)) * half_width
        denom = max(half_width - f, 1e-9)
        z = height * np.clip((ax - f) / denom, 0.0, None)

    elif family == "Asymmetric walls":
        r = max(float(value), 1.0)
        z_right = height * (ax / half_width)
        z_left = height * np.clip(ax / (half_width / r), 0.0, None)
        z = np.where(x < 0, z_left, z_right)

    elif family == "Composite V-in-U":
        g = float(np.clip(value, 0.0, 0.95))
        trough = height * (ax / half_width) ** 2
        if g <= 0:
            z = trough
        else:
            gw = g * half_width
            gorge_depth = height * (g ** 2)
            inner = gorge_depth * (ax / gw) if gw > 0 else trough
            z = np.where(ax <= gw, inner, trough)
    else:
        z = height * (ax / half_width)

    return x, np.minimum(z, height * 6.0)


def shape_register_delta(v_index: float, b_fit: float) -> float:
    """Departure from the power-law manifold. Zero means self-similar walls."""
    if not (np.isfinite(v_index) and np.isfinite(b_fit)) or (b_fit + 1.0) == 0:
        return float("nan")
    return float(v_index - (b_fit - 1.0) / (b_fit + 1.0))


@st.cache_data(show_spinner=False)
def build_shape_register(n_points: int = 2001, height_fraction: float = 0.99) -> pd.DataFrame:
    """Measure every idealized form with the production metric code."""
    rows = []
    for family, spec in SHAPE_REGISTER_FAMILIES.items():
        for value in spec["values"]:
            x, z = generate_ideal_profile(family, value, n_points=n_points)
            try:
                m, _notes = compute_cross_section_metrics(x, z, height_fraction=height_fraction)
            except Exception:
                continue
            if not m:
                continue
            v = m.get("v_index", float("nan"))
            b = m.get("power_law_b", float("nan"))
            _cs, _cb, _cv = profile_conformity(x, z, height_fraction=height_fraction)
            rows.append({
                "Conformity": _cv,
                "Nearest ideal": _cb or "",
                "Family": family,
                "Parameter": spec["param"],
                "Value": float(value),
                "b (form)": b,
                "V-index": v,
                "Width / depth": m.get("width_depth_ratio", float("nan")),
                "Asymmetry": m.get("asymmetry_index", float("nan")),
                "delta (off-manifold)": shape_register_delta(v, b),
                "Fit r2": m.get("power_law_r2", float("nan")),
            })
    if not rows:
        return pd.DataFrame(columns=["Conformity", "Nearest ideal", "Family", "Parameter",
                                     "Value", "b (form)", "V-index", "Width / depth",
                                     "Asymmetry", "delta (off-manifold)", "Fit r2"])
    return pd.DataFrame(rows)


def _register_axis_pair(view: str) -> tuple[str, str, str, str]:
    if view == "Conformity vs form":
        return ("b (form)", "Conformity",
                "Power-law exponent b",
                "Conformity to nearest ideal  (1.000 = ideal)")
    if view == "Form vs aspect":
        return "b (form)", "Width / depth", "Power-law exponent b", "Width / depth ratio"
    if view == "Off-manifold vs asymmetry":
        return "Asymmetry", "delta (off-manifold)", "Asymmetry index", "delta  (V - (b-1)/(b+1))"
    return "b (form)", "delta (off-manifold)", "Power-law exponent b", "delta  (V - (b-1)/(b+1))"


# ------------------------------------------------------------
# Conformity to ideal sections - Wadell's logic applied to profiles.
#
# Wadell (1932, 1935) scored a grain against the sphere of equal volume, so a
# perfect sphere reads 1 and nature's departure from it carries the story. The
# same construction works on a cross-section: normalise it to the unit square,
# fit each ideal form, and score the fit. An ideal scores exactly 1.000 against
# itself. Everything real sits below, and HOW FAR below is the measurement.
#
# Verified on synthetics: a perfect V scores 1.000 on Cone/V, a parabola 1.000
# on Parabola, a flat-floored trough 1.000 on Flat-floored. An asymmetric 3:1
# section scores only 0.363 against the best symmetric ideal - low conformity
# to every ideal is itself the signal that the form is irregular.
#
# A single free scale factor is allowed on each ideal so that a mismatch in
# amplitude is not scored as a mismatch in SHAPE. Everything here is therefore
# size-independent, like the rest of the register.
# ------------------------------------------------------------

IDEAL_SECTIONS = {
    "Cone / V": lambda u: np.abs(u),
    "Convex": lambda u: np.abs(u) ** 0.5,
    "Parabola": lambda u: u ** 2,
    "Semicircular arc": lambda u: 1.0 - np.sqrt(np.maximum(1.0 - u ** 2, 0.0)),
    "Flat-floored": lambda u: np.clip((np.abs(u) - 0.5) / 0.5, 0.0, 1.0),
    "Box trough": lambda u: np.abs(u) ** 6,
}

# The b each ideal sits at. Conformity is a magnitude and throws the SIGN of a
# departure away: a convex shield and a concave composite can both score ~0.99
# against Cone / V while lying on opposite sides of it. Measured through the
# app: b = 1.184, 1.092, 1.014, 0.938, 0.891 across a convex-to-concave series
# all returned "Cone / V" with conformity 0.958 to 1.000, while the cone index
# ran +0.236, +0.106, +0.004, -0.094, -0.150 and separated them cleanly. So the
# shape name carries a side, taken from b relative to the ideal's own b.
IDEAL_SECTION_B = {
    "Cone / V": 1.0,
    "Convex": 0.5,
    "Parabola": 2.0,
    "Semicircular arc": 2.37,
    "Flat-floored": 3.0,
    "Box trough": 6.0,
}

IDEAL_SECTION_NOTES = {
    "Cone / V": "Straight walls. b = 1. A fluvial V, or a straight-sided cone.",
    "Convex": "Walls that steepen outward. b < 1. Shield-like when inverted.",
    "Parabola": "The classic glacial trough form. b = 2.",
    "Semicircular arc": "Curved more tightly than a parabola.",
    "Flat-floored": "Straight walls standing on a flat floor - or a flat summit, inverted.",
    "Box trough": "Near-vertical walls, almost rectangular.",
}


def profile_conformity(
    x: np.ndarray,
    y: np.ndarray,
    height_fraction: float = 0.5,
    spec_height: float | None = None,
) -> tuple[dict, str | None, float]:
    """Score a section against each ideal form. 1.000 means it IS that ideal.

    Returns (scores by ideal, best-fitting ideal, best score). The score is a
    coefficient of determination against the normalised section, so it can go
    negative when a form is wildly unlike the ideal being tested.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    win = topo_valley_window(x, y, height_fraction, spec_height)
    if win is None:
        return {}, None, float("nan")

    t, xl, xr, h, _relief = win
    if not np.isfinite(h) or h <= 0 or xr <= xl:
        return {}, None, float("nan")

    xs = np.linspace(xl, xr, 1201)
    zs = np.interp(xs, x, y)
    z0 = float(np.min(zs))
    u = (xs - 0.5 * (xl + xr)) / (0.5 * (xr - xl))
    zn = (zs - z0) / max(h, 1e-9)

    sst = float(np.sum((zn - np.mean(zn)) ** 2))
    if sst <= 0:
        return {}, None, float("nan")

    scores = {}
    for name, fn in IDEAL_SECTIONS.items():
        try:
            pred = np.asarray(fn(u), dtype=float)
        except Exception:
            continue
        denom = float(np.dot(pred, pred))
        k = float(np.dot(pred, zn) / denom) if denom > 0 else 1.0
        ssr = float(np.sum((zn - k * pred) ** 2))
        scores[name] = float(1.0 - ssr / sst)

    if not scores:
        return {}, None, float("nan")
    best = max(scores, key=lambda n: scores[n])
    return scores, best, float(scores[best])


def shape_name_with_side(best: str | None, b_fit: float, tol: float = 0.04) -> tuple[str, str]:
    """Name the nearest ideal and say which side of it the profile falls on.

    For a valley, a b above the ideal's b means the walls are more trough-like;
    below means more convex. On an inverted volcano profile the same wording
    reads correctly, because the whole section was inverted before measurement.
    Returns (full name, side word).
    """
    if not best:
        return "-", ""
    b0 = IDEAL_SECTION_B.get(best)
    if b0 is None or not np.isfinite(b_fit) or b0 <= 0:
        return best, ""
    rel = (b_fit - b0) / b0
    if abs(rel) <= tol:
        return best, "on it"
    side = "trough side" if rel > 0 else "convex side"
    return f"{best}, {side}", side


# ------------------------------------------------------------
# Ideal overlay - draw the perfect shape on the real one and subtract.
#
# This is Wadell's grain construction done on a section: pin the ideal to the
# same apex and the same base, then measure the area between them. Two numbers
# come out of it, and they are NOT the same measurement:
#
#   net   = (excess - deficit) / A_ideal   - signed, global form.
#           This reproduces the V-index exactly, because the V-index already IS
#           this subtraction against a pinned triangle (Zimmer's construction).
#   total = (excess + deficit) / A_ideal   - unsigned, local irregularity.
#
# Measured on synthetics: a cone carrying waves that preserve its area reads
# V-index +0.0065 - indistinguishable from a perfect cone - while total reads
# 0.0988. A terraced cone reads V-index -0.056 and total 0.142. Net area
# cancels an excess against a deficit; the total does not.
#
# That is the sphericity / roundness split of grain-shape work: one number for
# gross form, a second for how ragged the outline is. Neither substitutes for
# the other.
# ------------------------------------------------------------

def ideal_overlay(
    x: np.ndarray,
    y: np.ndarray,
    ideal_name: str = "Cone / V",
    height_fraction: float = 0.5,
    spec_height: float | None = None,
) -> tuple[dict, dict]:
    """Fit the named ideal over the section and split the difference in two.

    The ideal is pinned to the measured section: same thalweg, same base, same
    height. A single scale factor is allowed so an amplitude mismatch is not
    charged as a shape mismatch.

    Returns (metrics, curves). curves carries the arrays needed to draw the
    overlay: distance, the real section, the ideal, and the two signed parts.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fn = IDEAL_SECTIONS.get(ideal_name)
    if fn is None:
        return {}, {}

    win = topo_valley_window(x, y, height_fraction, spec_height)
    if win is None:
        return {}, {}
    t, xl, xr, h, _relief = win
    if not np.isfinite(h) or h <= 0 or xr <= xl:
        return {}, {}

    xs = np.linspace(xl, xr, 2001)
    zs = np.interp(xs, x, y)
    z0 = float(np.min(zs))
    u = (xs - 0.5 * (xl + xr)) / (0.5 * (xr - xl))
    zn = (zs - z0) / max(h, 1e-9)

    try:
        ideal = np.asarray(fn(u), dtype=float)
    except Exception:
        return {}, {}
    denom = float(np.dot(ideal, ideal))
    k = float(np.dot(ideal, zn) / denom) if denom > 0 else 1.0
    ideal = k * ideal

    diff = zn - ideal
    excess = float(np.trapezoid(np.clip(diff, 0.0, None), u))
    deficit = float(np.trapezoid(np.clip(-diff, 0.0, None), u))
    a_ideal = float(np.trapezoid(ideal, u))
    if not np.isfinite(a_ideal) or abs(a_ideal) < 1e-12:
        return {}, {}

    metrics = {
        "form_excess": excess / a_ideal,
        "form_deficit": deficit / a_ideal,
        "form_net": (excess - deficit) / a_ideal,
        "form_irregularity": (excess + deficit) / a_ideal,
    }
    curves = {
        "u": u,
        "section": zn,
        "ideal": ideal,
        "x_real": xs,
        "z_real": zs,
        "z_ideal": z0 + ideal * max(h, 1e-9),
        "ideal_name": ideal_name,
    }
    return metrics, curves


def make_overlay_plot(curves: dict, metrics: dict, inverted: bool = False):
    """The real section with the ideal drawn over it, difference shaded.

    Excess and deficit are a polarity, so they take the diverging pair - blue
    against red with a neutral midpoint - rather than two categorical hues.
    """
    fig = go.Figure()
    if not curves:
        return fig

    xr_ = curves["x_real"]
    zr = curves["z_real"]
    zi = curves["z_ideal"]
    if inverted:
        zr = -zr
        zi = -zi

    upper = np.maximum(zr, zi)
    lower = np.minimum(zr, zi)

    fig.add_trace(go.Scatter(
        x=xr_, y=zi, mode="lines", name=f"ideal: {curves['ideal_name']}",
        line=dict(width=2, color="#52514e", dash="dash"),
        hovertemplate="ideal %{y:.4g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xr_, y=np.where(zr >= zi, zr, zi), mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xr_, y=zi, mode="lines", fill="tonexty",
        fillcolor="rgba(42,120,214,0.30)",
        line=dict(width=0), name="excess (above ideal)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xr_, y=np.where(zr <= zi, zr, zi), mode="lines",
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xr_, y=zi, mode="lines", fill="tonexty",
        fillcolor="rgba(208,59,59,0.30)",
        line=dict(width=0), name="deficit (below ideal)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=xr_, y=zr, mode="lines", name="measured section",
        line=dict(width=3, color="#0b0b0b"),
        hovertemplate="measured %{y:.4g}<extra></extra>",
    ))

    net = metrics.get("form_net", float("nan"))
    tot = metrics.get("form_irregularity", float("nan"))
    fig.update_layout(
        title=(f"Ideal overlaid on the section - net {net:+.3f}, irregularity {tot:.3f} "
               "(both as a fraction of the ideal's area)"),
        xaxis_title="Distance", yaxis_title="Elevation",
        template="plotly_white", height=470,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#eceae4", zeroline=False)
    fig.update_yaxes(gridcolor="#eceae4", zeroline=False)
    return fig


def make_conformity_bars(scores: dict, best: str | None = None):
    """How close this section comes to each ideal. The 1.000 line is the ideal."""
    fig = go.Figure()
    if not scores:
        return fig
    names = list(scores.keys())
    vals = [scores[n] for n in names]
    order = np.argsort(vals)
    names = [names[i] for i in order]
    vals = [vals[i] for i in order]
    colours = ["#2a78d6" if n == best else "#b9c6d8" for n in names]
    fig.add_trace(go.Bar(
        x=vals, y=names, orientation="h",
        marker=dict(color=colours, line=dict(width=0)),
        text=[f"{v:.3f}" for v in vals], textposition="outside",
        textfont=dict(size=12, color="#0b0b0b"),
        hovertemplate="<b>%{y}</b><br>conformity %{x:.3f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_vline(x=1.0, line=dict(color="#52514e", width=2))
    fig.add_annotation(x=1.0, y=1.04, yref="paper", text="ideal = 1.000",
                       showarrow=False, font=dict(size=11, color="#52514e"))
    lo = min(min(vals), 0.0)
    fig.update_layout(
        title="Conformity to each ideal section",
        xaxis_title="Conformity  (1.000 = this profile IS that ideal)",
        template="plotly_white", height=360,
        margin=dict(l=20, r=70, t=60, b=20),
        xaxis=dict(range=[lo - 0.12, 1.22], gridcolor="#eceae4", zeroline=False),
        yaxis=dict(gridcolor="#eceae4"),
        bargap=0.35,
    )
    return fig


# ------------------------------------------------------------
# The shape lattice - "graph paper" for cross-section form.
#
# Measured result, not an assumption: for power-law walls the two coordinates
# are EXACTLY independent. b_fit recovers the input b at every aspect from
# W/D = 2 to W/D = 40, and W/D recovers its input at every b; V-index depends
# on b alone (-0.333, 0.000, +0.200, +0.333, +0.500 for b = 0.5, 1, 1.5, 2, 3)
# and does not move with aspect at all. The non-power-law forms behave the same
# way: a flat floor of 0.3 reads b = 2.275 and delta = -0.087 whether its
# aspect is 2, 10 or 40.
#
# So the reference is a true rectangular grid - iso-b verticals crossed with
# iso-aspect horizontals - and every shape family occupies a whole vertical
# LINE rather than a point. An earlier version of this register drew each
# family at a single arbitrary aspect (W/D = 10), which made the grid look like
# two crossing lines and hid that structure.
# ------------------------------------------------------------

SHAPE_LATTICE_B = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0]
SHAPE_LATTICE_WD = [2.0, 4.0, 6.0, 10.0, 15.0, 22.0, 32.0]

SHAPE_ANCHORS = {
    0.5: "convex walls",
    1.0: "straight V / cone",
    1.5: "intermediate",
    2.0: "parabola",
    3.0: "trough",
    4.0: "box trough",
}


def generate_lattice_profile(b: float, width_depth: float,
                             half_width: float = 1000.0, n_points: int = 2001):
    """One power-law cross-section with BOTH its shape and its aspect specified."""
    h = 2.0 * float(half_width) / max(float(width_depth), 1e-6)
    x = np.linspace(-half_width, half_width, int(n_points))
    return x, h * (np.abs(x) / half_width) ** float(b)


@st.cache_data(show_spinner=False)
def build_shape_lattice(height_fraction: float = 0.99) -> pd.DataFrame:
    """Measure every node of the b x aspect grid with the production metric code."""
    rows = []
    for b in SHAPE_LATTICE_B:
        for wd in SHAPE_LATTICE_WD:
            x, z = generate_lattice_profile(b, wd)
            try:
                m, _n = compute_cross_section_metrics(x, z, height_fraction=height_fraction)
            except Exception:
                continue
            if not m:
                continue
            rows.append({
                "b_in": b,
                "wd_in": wd,
                "b (form)": m.get("power_law_b", float("nan")),
                "Width / depth": m.get("width_depth_ratio", float("nan")),
                "V-index": m.get("v_index", float("nan")),
            })
    return pd.DataFrame(rows)


def add_lattice_grid(fig, lattice: pd.DataFrame):
    """Draw the lattice as recessive graph paper beneath the data."""
    if lattice is None or lattice.empty:
        return fig
    grid = dict(color="#d8d6ce", width=1)
    for b in sorted(lattice["b_in"].unique()):
        sub = lattice[lattice["b_in"] == b].sort_values("wd_in")
        fig.add_trace(go.Scatter(
            x=sub["b (form)"], y=sub["Width / depth"], mode="lines",
            line=grid, hoverinfo="skip", showlegend=False,
        ))
    for wd in sorted(lattice["wd_in"].unique()):
        sub = lattice[lattice["wd_in"] == wd].sort_values("b_in")
        fig.add_trace(go.Scatter(
            x=sub["b (form)"], y=sub["Width / depth"], mode="lines",
            line=grid, hoverinfo="skip", showlegend=False,
        ))
    top = float(lattice["Width / depth"].max())
    left = float(lattice["b (form)"].min())
    for b, name in SHAPE_ANCHORS.items():
        if b < left:
            continue
        fig.add_annotation(
            x=b, y=top, text=f"b={b:g}<br>{name}", showarrow=False,
            yshift=26, font=dict(size=10, color="#52514e"), align="center",
        )
    for wd in sorted(lattice["wd_in"].unique()):
        fig.add_annotation(
            x=left, y=wd, text=f"W/D {wd:g}", showarrow=False,
            xanchor="right", xshift=-6, font=dict(size=10, color="#8a8981"),
        )
    return fig


def make_shape_gallery(b_values=None, wd_values=None, half_width: float = 1000.0):
    """A thumbnail of the ideal section at every lattice node.

    Rows are aspect, columns are form. Reading across a row shows b changing at
    fixed proportions; reading down a column shows the same shape getting wider
    and shallower.
    """
    bs = list(b_values or [0.5, 1.0, 1.5, 2.0, 3.0])
    wds = list(wd_values or [2.0, 6.0, 15.0, 32.0])
    fig = make_subplots(
        rows=len(wds), cols=len(bs),
        shared_xaxes=True, shared_yaxes=False,
        horizontal_spacing=0.012, vertical_spacing=0.05,
        column_titles=[f"b = {b:g}" for b in bs],
        row_titles=[f"W/D {wd:g}" for wd in wds],
    )
    for r, wd in enumerate(wds, start=1):
        for c, b in enumerate(bs, start=1):
            x, z = generate_lattice_profile(b, wd, half_width=half_width, n_points=401)
            fig.add_trace(go.Scatter(
                x=x / half_width, y=z, mode="lines",
                line=dict(width=2, color="#2a78d6"),
                hovertemplate=f"b = {b:g}, W/D = {wd:g}<extra></extra>",
                showlegend=False,
            ), row=r, col=c)
            fig.update_yaxes(range=[0, 2.0 * half_width / wd * 1.08],
                             showticklabels=False, showgrid=False,
                             zeroline=False, row=r, col=c)
            fig.update_xaxes(showticklabels=False, showgrid=False,
                             zeroline=False, row=r, col=c)
    fig.update_layout(
        title="Ideal sections at every lattice node - true aspect, drawn to scale within each row",
        template="plotly_white", height=170 * len(wds) + 90,
        margin=dict(l=20, r=70, t=90, b=20),
    )
    for ann in fig.layout.annotations:
        ann.font.size = 11
        ann.font.color = "#52514e"
    return fig


REGISTER_PIN_SYMBOLS = [
    "circle", "square", "triangle-up", "star", "cross",
    "hexagon", "triangle-down", "x-thin", "diamond-tall", "pentagon",
]


def register_group_symbol(group: str, groups_in_order: list) -> str:
    """Groups are separated by SYMBOL, not colour.

    The reference families already hold the categorical palette, so colouring
    pinned samples would collide with them. Shape carries the grouping instead,
    which also keeps the plot readable in greyscale and for colour-vision
    deficiency, and does not cap the number of groups at the all-pairs limit.
    """
    if not group:
        return "circle"
    try:
        i = groups_in_order.index(group)
    except ValueError:
        return "circle"
    return REGISTER_PIN_SYMBOLS[i % len(REGISTER_PIN_SYMBOLS)]


# ------------------------------------------------------------
# Shape regions - the register drawn as a field diagram.
#
# Each region is the measured envelope of one or more ideal families, not a
# hand-drawn box: the polygon is the convex hull of where those families
# actually land in the current view, so it moves if the metrics change.
#
# Colours are all-pairs validated for colour-vision deficiency (worst pair
# deltaE 13.0, OKLab x100) because filled areas can border each other anywhere.
# Every region also carries a text label inside it, so identity never rests on
# colour alone - which is also what discharges the contrast relief rule for the
# lighter fills.
# ------------------------------------------------------------

SHAPE_REGIONS = {
    "Power-law manifold": {
        "families": ["Power-law walls", "Triangle aspect"],
        "colour": "#2a78d6",
        "note": "Self-similar walls. A single power law describes the whole section.",
    },
    "Flattened floor / summit": {
        "families": ["Flat floor"],
        "colour": "#eda100",
        "note": "A flat floor (or, inverted, a flat summit). Drives delta negative.",
    },
    "Asymmetric": {
        "families": ["Asymmetric walls"],
        "colour": "#e34948",
        "note": "One wall steeper than the other. Drives delta positive and pulls width/depth down.",
    },
    "Tightly curved": {
        "families": ["Circular arc"],
        "colour": "#4a3aa7",
        "note": "Strongly curved section, tighter than a parabola. Drives delta positive at low asymmetry.",
    },
}

# Drawn as a trajectory rather than a region: a re-incised valley MOVES between
# regions as its inner gorge grows, so a filled area would misrepresent it.
SHAPE_TRAJECTORY_FAMILIES = ["Composite V-in-U"]


def _hull_polygon(px: np.ndarray, py: np.ndarray, pad_x: float, pad_y: float):
    """Convex hull of a point set, buffered outward. Falls back to a padded box."""
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    good = np.isfinite(px) & np.isfinite(py)
    px, py = px[good], py[good]
    if len(px) == 0:
        return None, None

    cx, cy = float(np.mean(px)), float(np.mean(py))
    hx = hy = None
    if len(px) >= 3:
        try:
            from scipy.spatial import ConvexHull
            h = ConvexHull(np.column_stack([px, py]))
            hx = px[h.vertices]
            hy = py[h.vertices]
        except Exception:
            hx = hy = None
    if hx is None:
        x0, x1 = float(np.min(px)), float(np.max(px))
        y0, y1 = float(np.min(py)), float(np.max(py))
        if x1 - x0 < pad_x:
            x0, x1 = cx - pad_x, cx + pad_x
        if y1 - y0 < pad_y:
            y0, y1 = cy - pad_y, cy + pad_y
        hx = np.array([x0, x1, x1, x0])
        hy = np.array([y0, y0, y1, y1])

    ox = np.sign(hx - cx) * pad_x
    oy = np.sign(hy - cy) * pad_y
    hx = hx + np.where(np.abs(hx - cx) < 1e-12, pad_x, ox)
    hy = hy + np.where(np.abs(hy - cy) < 1e-12, pad_y, oy)
    return np.append(hx, hx[0]), np.append(hy, hy[0])


def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def add_shape_regions(fig, register: pd.DataFrame, view: str,
                      lattice: pd.DataFrame | None = None, alpha: float = 0.16):
    """Fill each region and label it in place. Replaces the contour backdrop."""
    if register is None or register.empty:
        return fig
    xk, yk, _xlab, _ylab = _register_axis_pair(view)

    xs = pd.to_numeric(register[xk], errors="coerce").to_numpy()
    ys = pd.to_numeric(register[yk], errors="coerce").to_numpy()
    if lattice is not None and not lattice.empty and xk in lattice and yk in lattice:
        xs = np.concatenate([xs, pd.to_numeric(lattice[xk], errors="coerce").to_numpy()])
        ys = np.concatenate([ys, pd.to_numeric(lattice[yk], errors="coerce").to_numpy()])
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    if len(xs) == 0 or len(ys) == 0:
        return fig
    pad_x = max((float(np.max(xs)) - float(np.min(xs))) * 0.035, 1e-6)
    pad_y = max((float(np.max(ys)) - float(np.min(ys))) * 0.045, 1e-6)

    aspect_span = None
    if view == "Form vs aspect" and lattice is not None and not lattice.empty:
        aspect_span = (float(lattice["Width / depth"].min()),
                       float(lattice["Width / depth"].max()))

    for name, spec in SHAPE_REGIONS.items():
        sub = register[register["Family"].isin(spec["families"])]
        if sub.empty:
            continue
        px = pd.to_numeric(sub[xk], errors="coerce").to_numpy()
        py = pd.to_numeric(sub[yk], errors="coerce").to_numpy()
        if aspect_span is not None:
            # Shape is aspect-independent, so a family fills a vertical BAND
            # across every aspect, not a blob at one arbitrary row.
            fin = np.isfinite(px)
            if not fin.any():
                continue
            px = np.concatenate([px[fin], px[fin]])
            py = np.concatenate([np.full(int(fin.sum()), aspect_span[0]),
                                 np.full(int(fin.sum()), aspect_span[1])])
        gx, gy = _hull_polygon(px, py, pad_x, pad_y)
        if gx is None:
            continue
        fig.add_trace(go.Scatter(
            x=gx, y=gy, mode="lines", fill="toself",
            fillcolor=_hex_to_rgba(spec["colour"], alpha),
            line=dict(color=_hex_to_rgba(spec["colour"], 0.55), width=1.5),
            name=name, legendgroup=name,
            hovertemplate=f"<b>{name}</b><br>{spec['note']}<extra></extra>",
        ))
        fig.add_annotation(
            x=float(np.mean(gx[:-1])), y=float(np.mean(gy[:-1])),
            text=name, showarrow=False, align="center",
            font=dict(size=12, color="#52514e"),
            bgcolor="rgba(252,252,251,0.72)", borderpad=3,
        )

    for fam in SHAPE_TRAJECTORY_FAMILIES:
        sub = register[register["Family"] == fam].sort_values("Value")
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[xk], y=sub[yk], mode="lines+markers",
            name=f"{fam} (trajectory)",
            line=dict(width=2, color="#52514e", dash="dot"),
            marker=dict(size=7, color="#52514e", line=dict(width=1.5, color="#fcfcfb")),
            customdata=sub["Value"],
            hovertemplate=(f"<b>{fam}</b><br>gorge fraction %{{customdata:.2f}}"
                           "<br>%{x:.3f}, %{y:.3f}<extra></extra>"),
        ))
    return fig


def make_shape_register_plot(
    register: pd.DataFrame,
    view: str = "Form vs aspect",
    measured: dict | None = None,
    measured_label: str = "This profile",
    pinned: list | None = None,
    lattice: pd.DataFrame | None = None,
    backdrop: str = "Shaded regions",
):
    """Reference families as polylines; the loaded profile as a single marker.

    Families are drawn as connected tracks rather than a scatter cloud, so the
    palette is read on adjacent pairs. Each track carries a direct end label as
    well as a legend entry, so identity never rests on colour alone - which also
    discharges the contrast relief rule for the lighter hues.
    """
    xk, yk, xlab, ylab = _register_axis_pair(view)
    fig = go.Figure()

    # Backdrop first, so every data mark sits on top of it.
    _regions = backdrop == "Shaded regions"
    if _regions:
        add_shape_regions(fig, register, view, lattice)
    elif backdrop == "Lattice grid" and view == "Form vs aspect" \
            and lattice is not None and not lattice.empty:
        add_lattice_grid(fig, lattice)

    for family, spec in ({} if _regions else SHAPE_REGISTER_FAMILIES).items():
        sub = register[register["Family"] == family]
        sub = sub[np.isfinite(sub[xk]) & np.isfinite(sub[yk])]
        if sub.empty:
            continue
        sub = sub.sort_values("Value")
        if view == "Form vs aspect" and lattice is not None and not lattice.empty:
            # Shape is independent of aspect, so each family member is a whole
            # vertical line across the grid, not a point on one arbitrary row.
            lo = float(lattice["Width / depth"].min())
            hi = float(lattice["Width / depth"].max())
            for _, r_ in sub.iterrows():
                if not np.isfinite(r_[xk]):
                    continue
                fig.add_trace(go.Scatter(
                    x=[r_[xk], r_[xk]], y=[lo, hi], mode="lines",
                    line=dict(width=2, color=spec["colour"]), opacity=0.55,
                    showlegend=False, hoverinfo="skip",
                ))
        fig.add_trace(go.Scatter(
            x=sub[xk], y=sub[yk],
            mode="lines+markers",
            name=family,
            line=dict(width=2, color=spec["colour"]),
            marker=dict(size=8, color=spec["colour"], line=dict(width=2, color="#fcfcfb")),
            customdata=np.stack([sub["Parameter"], sub["Value"]], axis=-1),
            hovertemplate=(f"<b>{family}</b><br>%{{customdata[0]}} = %{{customdata[1]:.3g}}"
                           f"<br>{xlab}: %{{x:.3f}}<br>{ylab}: %{{y:.3f}}<extra></extra>"),
        ))
        tail = sub.iloc[-1]
        fig.add_annotation(
            x=float(tail[xk]), y=float(tail[yk]), text=family,
            showarrow=False, xanchor="left", xshift=8,
            font=dict(size=11, color="#52514e"),
        )

    if (not _regions) and view in ("Form vs aspect", "Off-manifold vs form"):
        fig.add_vline(x=1.0, line=dict(color="#c9c8c2", width=1, dash="dot"))
        fig.add_annotation(x=1.0, y=1.02, yref="paper", text="straight V",
                           showarrow=False, font=dict(size=10, color="#8a8981"))
        fig.add_vline(x=2.0, line=dict(color="#c9c8c2", width=1, dash="dot"))
        fig.add_annotation(x=2.0, y=1.02, yref="paper", text="parabola",
                           showarrow=False, font=dict(size=10, color="#8a8981"))
    if (not _regions) and "delta" in yk:
        fig.add_hline(y=0.0, line=dict(color="#c9c8c2", width=1))
        fig.add_annotation(x=0.02, y=0.0, xref="paper", text="power-law manifold",
                           showarrow=False, yshift=10, xanchor="left",
                           font=dict(size=10, color="#8a8981"))

    if pinned:
        groups_in_order = []
        for p in pinned:
            g = str(p.get("Group", "") or "")
            if g and g not in groups_in_order:
                groups_in_order.append(g)
        by_group = {}
        for i, p in enumerate(pinned, start=1):
            px, py = p.get(xk), p.get(yk)
            if px is None or py is None or not (np.isfinite(px) and np.isfinite(py)):
                continue
            g = str(p.get("Group", "") or "ungrouped")
            by_group.setdefault(g, {"x": [], "y": [], "t": [], "h": []})
            by_group[g]["x"].append(px)
            by_group[g]["y"].append(py)
            by_group[g]["t"].append(str(i))
            by_group[g]["h"].append(p.get("Label", f"sample {i}"))
        for g, d in by_group.items():
            sym = register_group_symbol("" if g == "ungrouped" else g, groups_in_order)
            fig.add_trace(go.Scatter(
                x=d["x"], y=d["y"], mode="markers+text",
                name=f"{g} (pinned)" if g != "ungrouped" else "Pinned samples",
                text=d["t"], textposition="top center",
                textfont=dict(size=11, color="#0b0b0b"),
                customdata=d["h"],
                marker=dict(size=13, symbol=sym, color="#0b0b0b",
                            line=dict(width=2, color="#fcfcfb")),
                hovertemplate=("<b>%{customdata}</b><br>" + xlab + ": %{x:.3f}<br>"
                               + ylab + ": %{y:.3f}<extra></extra>"),
            ))

    if measured:
        mx, my = measured.get(xk), measured.get(yk)
        if mx is not None and my is not None and np.isfinite(mx) and np.isfinite(my):
            fig.add_trace(go.Scatter(
                x=[mx], y=[my], mode="markers+text",
                name=measured_label,
                text=[measured_label], textposition="top center",
                textfont=dict(size=12, color="#0b0b0b"),
                marker=dict(size=18, symbol="diamond-open", color="#0b0b0b",
                            line=dict(width=3)),
                hovertemplate=(f"<b>{measured_label}</b><br>{xlab}: %{{x:.3f}}"
                               f"<br>{ylab}: %{{y:.3f}}<extra></extra>"),
            ))

    fig.update_layout(
        title=f"Shape register - {view}",
        xaxis_title=xlab, yaxis_title=ylab,
        template="plotly_white", height=620,
        hovermode="closest",
        margin=dict(l=20, r=120, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eceae4", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eceae4", zeroline=False)
    return fig


def make_ideal_shapes_plot(family: str, measured_xy: tuple | None = None):
    """The reference forms themselves, normalised so only shape is visible."""
    spec = SHAPE_REGISTER_FAMILIES.get(family)
    fig = go.Figure()
    if spec is None:
        return fig
    n = len(spec["values"])
    for i, value in enumerate(spec["values"]):
        x, z = generate_ideal_profile(family, value)
        h = float(np.max(z)) or 1.0
        shade = 0.25 + 0.75 * (i / max(n - 1, 1))
        fig.add_trace(go.Scatter(
            x=x / float(np.max(np.abs(x))), y=z / h,
            mode="lines", name=f"{spec['param']} = {value:g}",
            line=dict(width=2, color=spec["colour"]),
            opacity=shade,
            hovertemplate=f"{spec['param']} = {value:g}<extra></extra>",
        ))
    if measured_xy is not None:
        mx, mz = measured_xy
        mx = np.asarray(mx, dtype=float); mz = np.asarray(mz, dtype=float)
        if len(mx) > 4:
            mz = mz - float(np.min(mz))
            h = float(np.max(mz)) or 1.0
            c = float(mx[int(np.argmin(mz))])
            span = float(np.max(np.abs(mx - c))) or 1.0
            fig.add_trace(go.Scatter(
                x=(mx - c) / span, y=mz / h, mode="lines",
                name="This profile",
                line=dict(width=3, color="#0b0b0b", dash="dash"),
                hovertemplate="This profile<extra></extra>",
            ))
    fig.update_layout(
        title=f"Reference forms - {family}",
        xaxis_title="Normalised distance from thalweg",
        yaxis_title="Normalised height",
        template="plotly_white", height=460,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eceae4")
    fig.update_yaxes(showgrid=True, gridcolor="#eceae4")
    return fig


def make_signal_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    dataset_type: str,
    log_y: bool = False,
    peaks: PeakResults | None = None,
    overlay_series: list[tuple[np.ndarray, np.ndarray, str]] | None = None,
    show_peak_labels: bool = False,
    label_top_n: int = 8,
    reference_lines: pd.DataFrame | None = None,
):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="lines",
        name="Primary",
        hovertemplate=f"{x_label}: %{{x:.4g}}<br>{y_label}: %{{y:.4g}}<extra></extra>",
    ))

    if overlay_series:
        for ox, oy, name in overlay_series:
            fig.add_trace(go.Scatter(
                x=ox,
                y=oy,
                mode="lines",
                name=name,
            ))

    if reference_lines is not None and not reference_lines.empty:
        x_col = "Reference X"
        label_col = "Reference Label" if "Reference Label" in reference_lines.columns else None
        y_top = float(np.nanmax(y)) if len(y) else 1.0

        for _, row in reference_lines.iterrows():
            label_text = None
            if label_col is not None:
                label_text = str(row[label_col])
            else:
                label_text = f"{row[x_col]:.2f}"

            fig.add_vline(
                x=float(row[x_col]),
                line_dash="dot",
                line_color="rgba(220, 80, 80, 0.45)",
                annotation_text=label_text,
                annotation_position="top",
            )

        hover_label = "Reference position"

        fig.add_trace(go.Scatter(
            x=reference_lines[x_col],
            y=np.full(len(reference_lines), y_top * 1.02),
            mode="markers",
            marker=dict(symbol="line-ns-open", size=12),
            name="Reference peaks",
            hovertemplate=f"{hover_label}: %{{x:.3f}}<extra></extra>",
        ))

    if peaks is not None and len(peaks.indices) > 0:
        fig.add_trace(go.Scatter(
            x=peaks.x,
            y=peaks.y,
            mode="markers",
            name="Peaks",
            marker=dict(size=9, symbol="circle"),
            hovertemplate=f"{x_label}: %{{x:.4g}}<br>{y_label}: %{{y:.4g}}<extra>Peak</extra>",
        ))

        if show_peak_labels:
            label_df = peaks.table.sort_values("Intensity", ascending=False).head(max(int(label_top_n), 1)).copy()
            fig.add_trace(go.Scatter(
                x=label_df["X Position"],
                y=label_df["Intensity"],
                mode="text",
                text=[format_peak_label(v, dataset_type) for v in label_df["X Position"]],
                textposition="top center",
                textfont=dict(size=11),
                name="Peak labels",
                showlegend=False,
                hoverinfo="skip",
            ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=560,
        margin=dict(l=20, r=20, t=70, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if log_y:
        fig.update_yaxes(type="log")

    return fig


def dataframe_download_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


init_state()

st.markdown(
    """
    <div style="padding:0.8rem 1rem; border-radius:6px; background:#FFFFFF; border:1px solid #D6DBDA; border-left:4px solid #B4532A; margin-bottom:0.8rem;">
        <h1 style="margin:0; font-size:2rem;">Spectrum Earth</h1>
        <p style="margin:0.3rem 0 0 0;">
            Fit any topographic profile - a valley, a hill or mountain, a river long profile - with perfect shapes, and measure how far it departs from them: Wadell's grain-shape construction applied to landforms.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Top control band
# -----------------------------
control_1, control_ideal = st.columns([1.25, 2.2], gap="medium")

with control_1:
    st.markdown("**Load Data**")
    uploaded_file = st.file_uploader("Signal file", type=["csv", "txt", "xlsx"])
    sample_choice = st.selectbox("Or try a sample profile", ["None"] + list(SAMPLE_FILES), key="sample_choice",
                                 on_change=on_sample_change,
                                 help="Synthetic example profiles that ship with Spectrum Earth. Choosing one sets the matching Dataset Type.")
    dataset_type = st.selectbox(
        "Dataset Type",
        [TOPO_TRANSECT, TOPO_LONG_PROFILE, TOPO_VOLCANO],
        key="dataset_type",
    )
    if st.button("Apply Preset", width="stretch"):
        apply_dataset_preset(dataset_type)

with control_ideal:
    st.markdown("**Ideal shape fit**")
    _feature_names = list(ideal_view.FEATURE_CHOICES)
    _feature_default = ideal_view.default_feature(st.session_state.dataset_type, TOPO_VOLCANO, TOPO_LONG_PROFILE)
    _feature_key = f"ideal_feature_{st.session_state.dataset_type}"
    if _feature_key not in st.session_state:
        st.session_state[_feature_key] = _feature_default
    ideal_feature = st.radio(
        "What is the feature?", _feature_names,
        horizontal=True, key=_feature_key,
        help="A valley is fitted from its floor up to each rim, a hill from its summit down to each "
             "foot, and a long profile as one slope from its head to its mouth. A spreading ridge is "
             "fitted whole as a hill and read again at its axis (distance 0, or the middle of the "
             "profile) to tell an axial valley from an axial high.",
    )
    _u1, _u2, _u3 = st.columns([1.0, 1.0, 1.0], gap="small")
    with _u1:
        st.selectbox("Distance units", list(ideal_view.DISTANCE_UNITS), key="ideal_distance_unit",
                     help="Only the circle in true proportions and the width/height ratio need "
                          "these: they compare distance with elevation directly.")
    with _u2:
        st.selectbox("Elevation units", list(ideal_view.ELEVATION_UNITS), key="ideal_elevation_unit")
    with _u3:
        st.checkbox("Crop X Range", key="crop_enabled",
                    help="Trim the profile to one feature before fitting.")
    st.caption(
        "Each half of the profile is pinned at the apex and the foot and fitted with the best "
        "parabola, exponential, true circle and superellipse. The curve that fits best is the "
        "ideal, and the extra and missing rock against it is the departure."
    )

with st.expander("Settings for the other analyses (axes, smoothing, peaks, FFT, landform screening, AI)"):
    control_2, control_3, control_4 = st.columns(3, gap="medium")

with control_2:
    landform_reference_upload = None
    if st.session_state.dataset_type in TOPO_DATASET_TYPES:
        landform_reference_upload = st.file_uploader(
            "Custom landform reference",
            type=["csv", "txt", "xlsx"],
            help=(
                "Columns: landform, metric, center, sd, plus optional profile_type, weight, source. "
                "Set weight to 0 to report a metric without scoring it."
            ),
            key="landform_reference_upload_widget",
        )
    st.markdown("**Axes & Display**")
    if st.session_state.raw_df is None:
        st.caption("Load a file to activate x/y settings.")
    st.text_input("X-axis Label", key="x_label")
    st.text_input("Y-axis Label", key="y_label")
    st.text_input("Plot Title", key="plot_title")
    st.checkbox("Normalize Signal", key="normalize")
    st.checkbox("Log Y Axis", key="log_y")

with control_2:
    st.markdown("**Prepare Signal**")
    st.checkbox("Apply Smoothing", key="smoothing_on")
    st.selectbox("Smoothing Method", ["Savitzky-Golay", "Moving Average"], key="smoothing_method")
    st.slider("Smoothing Window", 3, 101, key="smoothing_window", step=2)
    st.checkbox("Baseline Correction", key="baseline_on")
    st.selectbox("Baseline Method", ["AsLS", "Linear"], key="baseline_method")
    st.checkbox("Detrend", key="detrend_on")
    st.checkbox("Interpolate Even Spacing", key="interpolate_on")

with control_3:
    st.markdown("**Peaks & Interpretation**")
    st.number_input("Peak Prominence", min_value=0.0, step=0.01, key="peak_prominence")
    st.number_input("Peak Distance", min_value=1, step=1, key="peak_distance")
    st.number_input("Peak Width", min_value=0.0, step=0.5, key="peak_width")
    st.checkbox("Show Peak Labels", key="show_peak_labels")
    st.number_input("Label Top N Peaks", min_value=1, max_value=30, step=1, key="label_top_n")
    if st.session_state.dataset_type in TOPO_DATASET_TYPES:
        st.markdown("**Landform Analysis**")
        if st.session_state.dataset_type == TOPO_TRANSECT:
            st.selectbox(
                "Benchmark Height Mode",
                ["Auto (fraction of valley relief)", TOPO_MATCH_MODE, "Manual (absolute height)"],
                key="topo_height_mode",
                help="The height above the thalweg at which valley width and area are measured.",
            )
            if st.session_state.topo_height_mode == TOPO_MATCH_MODE:
                st.number_input(
                    "Vertical Precision (y-units)",
                    min_value=0.0, step=0.1, key="topo_vertical_precision", format="%.3f",
                    help="Instrument vertical precision. Elevations are matched within this band "
                         "rather than exactly. Set it from the dataset, not by eye.",
                )
                st.slider(
                    "Match Quantile", 0.50, 1.00, key="topo_match_quantile", step=0.01,
                    help="Each side's top level is this quantile of its elevations, not its single "
                         "highest sample. 0.95 tested best; 1.00 reverts to the raw extremum.",
                )
            else:
                st.slider("Benchmark Height Fraction", 0.1, 0.95, key="topo_height_fraction", step=0.05)
                st.number_input("Manual Benchmark Height (y-units)", min_value=0.1, step=1.0, key="topo_spec_height")
            st.number_input("Harmonics for Decay Fit", min_value=3, max_value=40, step=1, key="topo_harmonic_count")
        elif st.session_state.dataset_type == TOPO_LONG_PROFILE:
            st.checkbox(
                "Downstream is to the right",
                key="topo_flow_left_to_right",
                help="Overdeepening detection depends on flow direction. Uncheck if your profile runs upstream.",
            )
            st.number_input("Min Overdeepening Rise (y-units)", min_value=0.0, step=1.0, key="topo_overdeepening_min_rise")
            st.number_input(
                "Knickpoint Prominence Fraction",
                min_value=0.05, max_value=3.0, step=0.05,
                key="topo_knick_prominence_frac",
                help="Fraction of the interquartile spread of slope a step must exceed.",
            )
            st.slider(
                "Slope Baseline Fraction",
                0.002, 0.10, key="topo_slope_baseline_frac", step=0.002,
                help="Slope is measured over this fraction of profile length. Below about 0.01 the "
                     "gradient is noise-dominated and knickpoint counts become meaningless.",
            )
            st.slider("Beta Fit Low Frequency Fraction", 0.005, 0.30, key="topo_beta_low_frac", step=0.005)
            st.slider("Beta Fit High Frequency Fraction", 0.10, 1.0, key="topo_beta_high_frac", step=0.05)
        elif st.session_state.dataset_type == TOPO_VOLCANO:
            st.selectbox(
                "Planetary Body",
                list(PLANETARY_BODIES.keys()),
                key="topo_body",
                help="Context only. Thresholds are Earth-calibrated and are NOT rescaled for gravity.",
            )
            st.selectbox(
                "Basal Boundary Mode",
                ["Auto (fraction of valley relief)", TOPO_MATCH_MODE, "Manual (absolute height)"],
                key="topo_height_mode",
                help="Where the edifice is taken to end. This is the dominant source of morphometric uncertainty.",
            )
            if st.session_state.topo_height_mode == TOPO_MATCH_MODE:
                st.number_input(
                    "Vertical Precision (y-units)",
                    min_value=0.0, step=0.1, key="topo_vertical_precision", format="%.3f",
                    help="Instrument vertical precision. The edifice foot is matched within this "
                         "band rather than exactly.",
                )
                st.slider(
                    "Match Quantile", 0.50, 1.00, key="topo_match_quantile", step=0.01,
                    help="Each side's basal level is this quantile of its elevations, not its "
                         "single lowest sample. 0.95 tested best on a volcano standing on a plain.",
                )
            else:
                st.slider("Basal Boundary Fraction", 0.1, 0.95, key="topo_height_fraction", step=0.05)
                st.number_input("Manual Edifice Height (y-units)", min_value=0.1, step=1.0, key="topo_spec_height")
            st.number_input("Harmonics for Decay Fit", min_value=3, max_value=40, step=1, key="topo_harmonic_count")

        st.checkbox("Show uncalibrated reference rows", key="topo_show_uncalibrated")

with control_4:
    st.markdown("**FFT View**")
    st.selectbox("Window Function", ["None", "Hann", "Hamming", "Blackman"], key="fft_window")
    st.checkbox("Remove Zero Frequency", key="fft_remove_zero")
    st.selectbox("FFT X View", ["Frequency", "Period"], key="fft_view_mode")
    st.checkbox("Log FFT Amplitude", key="fft_log_amplitude")
    st.checkbox("Log FFT X-Axis", key="fft_log_x")
    st.checkbox("Smooth FFT Display", key="fft_smooth_display")
    st.slider("FFT Display Smooth Window", 3, 31, key="fft_display_smooth_window", step=2)
    st.checkbox("Detect FFT Peaks", key="fft_detect_peaks")
    st.number_input("FFT Peak Prominence", min_value=0.0, step=0.01, key="fft_peak_prominence")
    st.number_input("FFT Peak Distance", min_value=1, step=1, key="fft_peak_distance")
    st.slider("FFT X-Range Fraction", min_value=0.05, max_value=1.0, key="fft_max_fraction", step=0.05)
    st.markdown("**Optional AI**")
    st.checkbox("Enable AI help", key="ai_enable", help="The app still works fully without AI. When enabled, it uses free Ollama Cloud models, with a built-in summary if they are busy.")
    st.checkbox("Prefer local Ollama (on your own computer)", key="ai_prefer_local")
    st.text_input("Local Ollama model", key="ollama_model")

if uploaded_file is not None:
    try:
        st.session_state.raw_df = read_uploaded_file(uploaded_file)
        st.session_state["source_name"] = getattr(uploaded_file, "name", "")
    except Exception as e:
        st.error(f"File read error: {e}")
        st.stop()
elif sample_choice != "None":
    st.session_state.raw_df = pd.read_csv(Path(__file__).resolve().parent / "samples" / SAMPLE_FILES[sample_choice])
    st.session_state["source_name"] = SAMPLE_FILES[sample_choice]

raw_df = st.session_state.raw_df

# -----------------------------
# Empty state
# -----------------------------
if raw_df is None:
    st.markdown(
        """
        <div style="padding:2.3rem 1rem; border:1px dashed #D6B27C; border-radius:6px; text-align:center; background:#FFFFFF;">
            <h3 style="margin-top:0;">Load a signal file to begin</h3>
            <p style="max-width:850px; margin:0 auto;">
                Spectrum Earth works with simple x-y tables: distance and elevation across a valley, over a hill or mountain,
                or down a river long profile. Or pick one of the sample profiles. Once a profile is loaded, it is fitted with
                perfect shapes and its departure from the ideal is measured; the Shape Chart organises many profiles by that departure.
                For XRD, GC and SEM/EDS data, use TraceCode.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

df = raw_df
cols = list(df.columns)
numeric_cols = get_numeric_columns(df)

if st.session_state.x_col in cols:
    default_x_name = st.session_state.x_col
elif numeric_cols:
    default_x_name = numeric_cols[0]
else:
    default_x_name = cols[0]

if st.session_state.y_col in cols and st.session_state.y_col != default_x_name:
    default_y_name = st.session_state.y_col
elif len(numeric_cols) >= 2:
    default_y_name = numeric_cols[1]
elif len(numeric_cols) == 1:
    default_y_name = numeric_cols[0]
else:
    default_y_name = cols[1] if len(cols) > 1 else cols[0]

default_x_idx = cols.index(default_x_name)
default_y_idx = cols.index(default_y_name)

meta_1, meta_2, meta_3, meta_4 = st.columns([1.2, 1.0, 0.9, 1.1], gap="medium")
with meta_1:
    x_col = st.selectbox("X Column", cols, index=default_x_idx)
with meta_2:
    y_col = st.selectbox("Y Column", cols, index=default_y_idx)
with meta_3:
    st.metric("Rows", f"{len(df):,}")
with meta_4:
    st.metric("Dataset", st.session_state.dataset_type)

st.session_state.x_col = x_col
st.session_state.y_col = y_col

temp_xy = clean_numeric_xy(df, x_col, y_col)
if temp_xy.empty:
    st.error(
        f"The selected columns '{x_col}' and '{y_col}' do not produce numeric x-y data. "
        f"Choose numeric columns instead."
    )
    if numeric_cols:
        st.info(f"Numeric-looking columns found: {', '.join(map(str, numeric_cols))}")
    st.caption("If you uploaded a landform reference table, that belongs in 'Custom landform reference', not the main signal file uploader.")
    st.stop()

xmin = float(temp_xy["x"].min())
xmax = float(temp_xy["x"].max())
x_min_crop, x_max_crop = xmin, xmax
if st.session_state.crop_enabled:
    x_min_crop, x_max_crop = st.slider("Crop X Range", min_value=xmin, max_value=xmax, value=(xmin, xmax))

# -----------------------------
# Processing
# -----------------------------
xy = clean_numeric_xy(df, x_col, y_col)
if st.session_state.crop_enabled:
    xy = xy[(xy["x"] >= x_min_crop) & (xy["x"] <= x_max_crop)].reset_index(drop=True)

if xy.empty:
    st.error("No valid numeric data remain after cleaning/cropping. Try different x/y columns or widen the crop range.")
    st.stop()

x = xy["x"].to_numpy()
y = xy["y"].to_numpy()
raw_x = x.copy()
raw_y = y.copy()

# The ideal-shape fit runs on the raw (cleaned, cropped) profile in common
# units. It resamples and smooths at the scale of the feature itself, so the
# display settings for the other analyses cannot change it.
_kind = ideal_view.FEATURE_CHOICES[ideal_feature]
_sx = ideal_view.DISTANCE_UNITS[st.session_state.ideal_distance_unit]
_sz = ideal_view.ELEVATION_UNITS[st.session_state.ideal_elevation_unit]
ridge_reading = None
if ideal_feature == ideal_view.RIDGE_FEATURE:
    ridge_reading = ideal_shape.read_ridge(raw_x * _sx, raw_y * _sz)
    ideal_item = ridge_reading["whole"]
else:
    ideal_item = ideal_shape.measure(raw_x * _sx, raw_y * _sz, _kind)
if ideal_item.get("ok"):
    # Draw in the file's own units.
    ideal_item = ideal_view.to_display_units(ideal_item, _sx, _sz)
if ridge_reading is not None:
    ridge_reading = ideal_view.ridge_to_display_units(ridge_reading, _sx, _sz)

st.session_state.status_notes = []
if st.session_state.normalize:
    y = normalize_signal(y)
    note("Signal normalized.")

if st.session_state.smoothing_on:
    if st.session_state.smoothing_method == "Savitzky-Golay":
        y = apply_savgol(y, st.session_state.smoothing_window)
    else:
        y = moving_average(y, st.session_state.smoothing_window)
    note(f"Smoothing applied: {st.session_state.smoothing_method}.")

baseline_values = np.zeros_like(y)
if st.session_state.baseline_on:
    if st.session_state.baseline_method == "AsLS":
        baseline_values = baseline_asls(
            y,
            lam=float(st.session_state.asls_lambda),
            p=float(st.session_state.asls_p),
            niter=int(st.session_state.asls_niter),
        )
    else:
        baseline_values = baseline_linear(x, y)
    y = y - baseline_values
    note(f"Baseline correction applied: {st.session_state.baseline_method}.")

if st.session_state.detrend_on:
    y = detrend_linear(x, y)
    note("Linear detrending applied.")

even_spacing_before = is_evenly_spaced(x)
if st.session_state.interpolate_on and not even_spacing_before:
    x, y = interpolate_even_spacing(x, y, len(x))
    note("Interpolated to even spacing for FFT compatibility.")
elif not even_spacing_before:
    note("Warning: x spacing is uneven.")

processed_df = pd.DataFrame({"x": x, "y": y})
baseline_df = pd.DataFrame({"x": x, "baseline": baseline_values}) if st.session_state.baseline_on else None
st.session_state.processed_df = processed_df
st.session_state.baseline_df = baseline_df
st.session_state.status_notes = list(dict.fromkeys(st.session_state.status_notes))[-8:]

peak_results = None
fft_results = None
st.session_state.peak_results = None
st.session_state.fft_results = None
topo_interpretation = None
topo_metrics = None
landform_ref_table = None
st.session_state.topo_interpretation = None
st.session_state.topo_metrics = None

try:
    peak_results = detect_peaks(
        x,
        y,
        float(st.session_state.peak_prominence),
        int(st.session_state.peak_distance),
        float(st.session_state.peak_width),
    )
    st.session_state.peak_results = peak_results
except Exception as e:
    st.error(f"Peak detection error: {e}")

try:
    fft_results = compute_fft(x, y, window_name=st.session_state.fft_window)
    st.session_state.fft_results = fft_results
except Exception as e:
    st.warning(f"FFT could not be computed: {e}")

reference_source_label = "Built-in educational library"
reference_preview = None
if st.session_state.dataset_type in TOPO_DATASET_TYPES and peak_results is not None:
    try:
        if landform_reference_upload is not None:
            landform_ref_table = standardize_landform_reference(load_reference_table(landform_reference_upload))
            reference_source_label = "Custom uploaded landform library"
        else:
            landform_ref_table = standardize_landform_reference(builtin_landform_reference_df())
        reference_preview = landform_ref_table.copy()

        # Landform metrics are computed on RAW elevation re-gridded onto the
        # processed x axis, so display-oriented steps (normalise, AsLS baseline,
        # linear detrend) cannot contaminate the geometry. Smoothing is
        # re-applied because it genuinely stabilises derivative-based metrics.
        y_metric = np.interp(x, raw_x, raw_y)
        if st.session_state.smoothing_on:
            if st.session_state.smoothing_method == "Savitzky-Golay":
                y_metric = apply_savgol(y_metric, st.session_state.smoothing_window)
            else:
                y_metric = moving_average(y_metric, st.session_state.smoothing_window)

        spec_h = None
        if st.session_state.topo_height_mode == "Manual (absolute height)":
            spec_h = float(st.session_state.topo_spec_height)
        _baseline_mode = (
            "matched" if st.session_state.topo_height_mode == TOPO_MATCH_MODE else "fraction"
        )
        _vprec = float(st.session_state.topo_vertical_precision)
        _mq = float(st.session_state.topo_match_quantile)

        if st.session_state.dataset_type == TOPO_TRANSECT:
            topo_metrics, topo_notes = compute_cross_section_metrics(
                x,
                y_metric,
                height_fraction=float(st.session_state.topo_height_fraction),
                spec_height=spec_h,
                harmonic_count=int(st.session_state.topo_harmonic_count),
                baseline_mode=_baseline_mode,
                vertical_precision=_vprec,
                match_quantile=_mq,
            )
        elif st.session_state.dataset_type == TOPO_LONG_PROFILE:
            topo_metrics, topo_notes = compute_long_profile_metrics(
                x,
                y_metric,
                fft_results=fft_results,
                flow_left_to_right=bool(st.session_state.topo_flow_left_to_right),
                beta_low_frac=float(st.session_state.topo_beta_low_frac),
                beta_high_frac=float(st.session_state.topo_beta_high_frac),
                min_rise=float(st.session_state.topo_overdeepening_min_rise),
                knick_prominence_frac=float(st.session_state.topo_knick_prominence_frac),
                slope_baseline_frac=float(st.session_state.topo_slope_baseline_frac),
            )
        elif st.session_state.dataset_type == TOPO_VOLCANO:
            topo_metrics, topo_notes = compute_edifice_metrics(
                x,
                y_metric,
                height_fraction=float(st.session_state.topo_height_fraction),
                spec_height=spec_h,
                harmonic_count=int(st.session_state.topo_harmonic_count),
                # Unsmoothed, so the roughness fit still has texture to measure.
                y_rough=np.interp(x, raw_x, raw_y),
                baseline_mode=_baseline_mode,
                vertical_precision=_vprec,
                match_quantile=_mq,
            )


        # Conformity runs on the same orientation the metrics used: a volcano is
        # measured inverted, so its section must be scored inverted too.
        _conf_scores, _conf_best, _conf_val = {}, None, float("nan")
        if st.session_state.dataset_type in (TOPO_TRANSECT, TOPO_VOLCANO):
            _cy = -y_metric if st.session_state.dataset_type == TOPO_VOLCANO else y_metric
            try:
                _conf_scores, _conf_best, _conf_val = profile_conformity(
                    x, _cy,
                    height_fraction=float(st.session_state.topo_height_fraction),
                    spec_height=spec_h,
                )
            except Exception:
                _conf_scores, _conf_best, _conf_val = {}, None, float("nan")
            if np.isfinite(_conf_val):
                topo_metrics["shape_conformity"] = float(_conf_val)
                topo_metrics["shape_departure"] = float(1.0 - _conf_val)
        _ov_metrics, _ov_curves = {}, {}
        if _conf_best and st.session_state.dataset_type in (TOPO_TRANSECT, TOPO_VOLCANO):
            try:
                _ov_metrics, _ov_curves = ideal_overlay(
                    x, _cy, _conf_best,
                    height_fraction=float(st.session_state.topo_height_fraction),
                    spec_height=spec_h,
                )
                topo_metrics.update(_ov_metrics)
            except Exception:
                _ov_metrics, _ov_curves = {}, {}
        st.session_state["overlay_metrics"] = _ov_metrics
        st.session_state["overlay_curves"] = _ov_curves

        st.session_state["conformity_scores"] = _conf_scores
        st.session_state["conformity_best"] = _conf_best
        st.session_state["conformity_value"] = _conf_val

        for _msg in topo_notes:
            note(_msg)

        topo_interpretation = interpret_topography(
            metrics=topo_metrics,
            profile_type=st.session_state.dataset_type,
            reference_df=landform_ref_table,
        )
        st.session_state.topo_interpretation = topo_interpretation
        st.session_state.topo_metrics = topo_metrics
    except Exception as e:
        st.warning(f"Landform interpretation could not run: {e}")


# -----------------------------
# Main workspace
# -----------------------------
main_tabs = st.tabs(["Ideal Shape", "Shape Chart", "Other analyses"])

with main_tabs[0]:
    ideal_view.render_ideal_tab(
        ideal_item, st.session_state.x_label, st.session_state.y_label,
        st.session_state.get("source_name", ""), ridge=ridge_reading,
    )

with main_tabs[1]:
    ideal_view.render_shape_chart(
        ideal_item, st.session_state.get("source_name", ""), SAMPLE_FILES,
        lambda label: ideal_view.FEATURE_CHOICES[sample_feature(label)],
        Path(__file__).resolve().parent / "samples",
    )

with main_tabs[2]:
    st.caption(
        "The earlier analyses, kept for reference: signal preparation, peaks, FFT, landform "
        "screening against published classes, and the power-law shape register. Their settings "
        "are in the 'Settings for the other analyses' panel above."
    )
    tabs = st.tabs(["Raw Trace", "Prepared Signal", "Peaks", "FFT / Frequency View", "Compare", "Interpret", "Report", "Shape Register"])

with tabs[0]:
    fig_raw = make_signal_plot(
        raw_x,
        raw_y,
        title=f"{st.session_state.plot_title} - Raw",
        x_label=st.session_state.x_label,
        y_label=st.session_state.y_label,
        dataset_type=st.session_state.dataset_type,
        log_y=st.session_state.log_y,
    )
    st.plotly_chart(fig_raw, width="stretch")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Points", len(raw_x))
    c2.metric("X Min", f"{raw_x.min():.4g}")
    c3.metric("X Max", f"{raw_x.max():.4g}")
    c4.metric("Even Spacing", "Yes" if is_evenly_spaced(raw_x) else "No")

with tabs[1]:
    overlays = [(raw_x, raw_y, "Raw")]
    if baseline_df is not None:
        overlays.append((baseline_df["x"].to_numpy(), baseline_df["baseline"].to_numpy(), "Baseline"))
    fig_clean = make_signal_plot(
        x,
        y,
        title=f"{st.session_state.plot_title} - Prepared",
        x_label=st.session_state.x_label,
        y_label=st.session_state.y_label,
        dataset_type=st.session_state.dataset_type,
        log_y=st.session_state.log_y,
        overlay_series=overlays,
    )
    st.plotly_chart(fig_clean, width="stretch")
    st.dataframe(processed_df.head(20), width="stretch")

with tabs[2]:
    if peak_results is None:
        st.info("Peak detection is not available for the current data.")
    elif peak_results.table.empty:
        st.info("No peaks were found with the current settings. Try lowering prominence or distance.")
    else:
        fig_peaks = make_signal_plot(
            x,
            y,
            title=f"{st.session_state.plot_title} - Peaks",
            x_label=st.session_state.x_label,
            y_label=st.session_state.y_label,
            dataset_type=st.session_state.dataset_type,
            log_y=st.session_state.log_y,
            peaks=peak_results,
            show_peak_labels=st.session_state.show_peak_labels,
            label_top_n=int(st.session_state.label_top_n),
        )
        st.plotly_chart(fig_peaks, width="stretch")
        st.dataframe(peak_results.table, width="stretch")


with tabs[3]:
    st.info(
        "FFT turns your signal into a frequency view. Low frequency means broad, slow changes like baseline drift. "
        "High frequency means fast oscillations or fine noise. Tall FFT peaks mark dominant repeating scales in the signal."
    )
    help_a, help_b, help_c, help_d = st.columns(4)
    help_a.markdown("**Left side**<br>Broad trend / baseline", unsafe_allow_html=True)
    help_b.markdown("**Right side**<br>Fine wiggles / noise", unsafe_allow_html=True)
    help_c.markdown("**Tall peaks**<br>Dominant repeating scale", unsafe_allow_html=True)
    help_d.markdown("**Best practice**<br>Even x-spacing helps", unsafe_allow_html=True)

    if not is_evenly_spaced(x):
        st.warning("X spacing is uneven. FFT is more trustworthy after interpolation to even spacing.")

    if fft_results is not None and len(fft_results.frequencies) > 0:
        fft_freq = np.asarray(fft_results.frequencies, dtype=float)
        fft_amp = np.asarray(fft_results.amplitudes, dtype=float)

        mask = np.isfinite(fft_freq) & np.isfinite(fft_amp)
        if st.session_state.fft_remove_zero:
            mask &= fft_freq > 0

        fft_freq = fft_freq[mask]
        fft_amp = fft_amp[mask]

        if len(fft_freq) > 0:
            max_fraction = float(st.session_state.fft_max_fraction)
            n_keep = max(5, int(np.ceil(len(fft_freq) * max_fraction)))
            fft_freq = fft_freq[:n_keep]
            fft_amp = fft_amp[:n_keep]

            display_amp = fft_amp.copy()
            if st.session_state.fft_smooth_display:
                display_amp = smooth_fft_display(display_amp, int(st.session_state.fft_display_smooth_window))

            fft_peak_table = pd.DataFrame(columns=["Frequency", "Period", "Amplitude", "Prominence"])
            if st.session_state.fft_detect_peaks:
                fft_peak_table = detect_fft_peaks(
                    fft_freq,
                    display_amp,
                    prominence=float(st.session_state.fft_peak_prominence),
                    distance=int(st.session_state.fft_peak_distance),
                ).head(12)

            x_vals = fft_freq if st.session_state.fft_view_mode == "Frequency" else np.where(fft_freq > 0, 1.0 / fft_freq, np.inf)

            valid_mask = np.isfinite(x_vals) & np.isfinite(display_amp)
            x_vals = x_vals[valid_mask]
            display_amp = display_amp[valid_mask]

            peak_labels = fft_peak_table.head(8) if not fft_peak_table.empty else None

            fig_fft = make_fft_plot(
                x_vals,
                display_amp,
                view_mode=st.session_state.fft_view_mode,
                log_amplitude=st.session_state.fft_log_amplitude,
                log_x=st.session_state.fft_log_x,
                labeled_peaks=peak_labels,
            )
            st.plotly_chart(fig_fft, width="stretch")

            dominant_table = pd.DataFrame({
                "Frequency": fft_freq,
                "Period": np.where(fft_freq > 0, 1.0 / fft_freq, np.inf),
                "Amplitude": fft_amp,
                "Displayed Amplitude": display_amp if len(display_amp) == len(fft_amp) else np.nan,
            }).sort_values("Amplitude", ascending=False).reset_index(drop=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top FFT Components**")
                st.dataframe(dominant_table.head(15), width="stretch")
            with col2:
                st.markdown("**Detected FFT Peaks**")
                if not fft_peak_table.empty:
                    st.dataframe(fft_peak_table, width="stretch")
                else:
                    st.info("No distinct FFT peaks were found with the current settings.")

            if not dominant_table.empty:
                dominant = dominant_table.iloc[0]
                st.markdown(
                    f"**Dominant frequency:** {dominant['Frequency']:.4g} &nbsp;&nbsp; "
                    f"**Dominant period:** {dominant['Period']:.4g} &nbsp;&nbsp; "
                    f"**Amplitude:** {dominant['Amplitude']:.4g}"
                )
        else:
            st.info("No FFT points remain after the current FFT display filters.")
    else:
        st.info("FFT is not available for the current data.")

with tabs[4]:
    compare_df = pd.DataFrame({
        st.session_state.x_label: x,
        "Raw/Interpolated": np.interp(x, raw_x, raw_y),
        "Prepared": y,
    })
    compare_df["Difference"] = compare_df["Prepared"] - compare_df["Raw/Interpolated"]

    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(x=compare_df[st.session_state.x_label], y=compare_df["Raw/Interpolated"], mode="lines", name="Raw/Interpolated"))
    fig_compare.add_trace(go.Scatter(x=compare_df[st.session_state.x_label], y=compare_df["Prepared"], mode="lines", name="Prepared"))
    fig_compare.add_trace(go.Scatter(x=compare_df[st.session_state.x_label], y=compare_df["Difference"], mode="lines", name="Difference"))
    fig_compare.update_layout(
        title="Raw vs Prepared Comparison",
        xaxis_title=st.session_state.x_label,
        yaxis_title=st.session_state.y_label,
        template="plotly_white",
        height=560,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    st.plotly_chart(fig_compare, width="stretch")
    st.dataframe(compare_df.head(20), width="stretch")

with tabs[5]:
    if st.session_state.dataset_type in TOPO_DATASET_TYPES:
        if topo_interpretation is None:
            st.info("Landform interpretation is not available yet for this run.")
        else:
            st.markdown(f"**Reference source:** {reference_source_label}")
            if st.session_state.dataset_type == TOPO_VOLCANO:
                _g = PLANETARY_BODIES.get(st.session_state.topo_body, float("nan"))
                if np.isfinite(_g):
                    st.caption(
                        f"Body: {st.session_state.topo_body} (surface gravity {_g:.2f} m/s2, "
                        f"{_g / 9.81:.2f} g). Thresholds below are Earth-calibrated and are NOT "
                        "rescaled for gravity."
                    )
                else:
                    st.caption("Body gravity unknown. Thresholds below are Earth-calibrated.")
            st.markdown(
                "This is a morphometric screen. Each metric is scored as a Gaussian likelihood "
                "against a published class centre and spread, and the weighted mean likelihood is "
                "the class score. Because the published classes overlap, close scores are the "
                "honest result rather than a failure of the method."
            )

            if topo_interpretation.candidates.empty:
                st.warning(topo_interpretation.explanation)
            else:
                st.success(topo_interpretation.explanation)
                cand_left, cand_right = st.columns([1.2, 1.0], gap="large")
                with cand_left:
                    st.markdown("**Ranked landform classes**")
                    st.dataframe(topo_interpretation.candidates, width="stretch")
                with cand_right:
                    st.markdown("**Computed metrics**")
                    st.dataframe(topo_interpretation.metrics, width="stretch")

                if not topo_interpretation.evidence.empty:
                    st.markdown("**Per-metric evidence (with sources)**")
                    st.dataframe(topo_interpretation.evidence, width="stretch")

            if topo_interpretation.candidates.empty and not topo_interpretation.metrics.empty:
                st.markdown("**Computed metrics**")
                st.dataframe(topo_interpretation.metrics, width="stretch")

            st.markdown("**Interpretation limits**")
            for _caveat in topo_interpretation.caveats:
                st.markdown(f"- {_caveat}")

            if reference_preview is not None:
                with st.expander("Landform reference library"):
                    _preview = reference_preview
                    if not st.session_state.topo_show_uncalibrated and "weight" in _preview.columns:
                        _preview = _preview[_preview["weight"] > 0]
                    st.dataframe(_preview, width="stretch")
                    st.caption(
                        "Rows with weight 0 are computed and reported but never scored. Raise a "
                        "weight only after calibrating that metric on labelled profiles."
                    )

    else:
        st.info("Choose a landform Dataset Type to see the interpretation.")


    st.markdown("---")
    st.markdown("### Optional AI assistant")
    if not st.session_state.ai_enable:
        st.caption("AI is off. Core analysis remains fully available without it.")
    else:
        provider_bits = []
        if st.session_state.ai_prefer_local and ollama_available():
            provider_bits.append("Local Ollama ready")
        if ai_client.available():
            provider_bits.append("Ollama Cloud ready")
        st.caption(" | ".join(provider_bits) if provider_bits else "No AI model is set up right now, so you will get a built-in summary instead.")
        ai_question = st.text_area("AI request", key="ai_question", placeholder="Example: Summarize the most likely interpretation and what I should check next.")
        if st.button("Generate AI summary", width="stretch", disabled=not ai_warranted(st.session_state.dataset_type, peak_results)):
            try:
                ai_text, provider = generate_ai_summary(
                    dataset_type=st.session_state.dataset_type,
                    x_label=st.session_state.x_label,
                    y_label=st.session_state.y_label,
                    peak_results=peak_results,
                    fft_results=fft_results,
                    notes=st.session_state.status_notes,
                    topo_interpretation=topo_interpretation,
                    question=ai_question,
                    prefer_local=bool(st.session_state.ai_prefer_local),
                    ollama_model=st.session_state.ollama_model,
                )
                st.session_state.ai_last_provider = provider
                st.session_state.ai_last_response = ai_text
            except Exception as exc:
                st.warning(str(exc))
        if st.session_state.ai_last_response:
            st.success(f"AI provider used: {st.session_state.ai_last_provider}")
            st.markdown(st.session_state.ai_last_response)

with tabs[6]:
    st.subheader("Summary")
    report_lines = [
        f"Dataset Type: {st.session_state.dataset_type}",
        f"X Column: {st.session_state.x_col}",
        f"Y Column: {st.session_state.y_col}",
        f"Point Count: {len(x)}",
        f"Even X Spacing: {'Yes' if is_evenly_spaced(x) else 'No'}",
        f"X Range: {x.min():.4g} to {x.max():.4g}",
        f"Y Range: {y.min():.4g} to {y.max():.4g}",
    ]
    if peak_results is not None:
        report_lines.append(f"Detected Peaks: {len(peak_results.indices)}")
    if fft_results is not None and not fft_results.table.empty:
        dominant = fft_results.table.iloc[0]
        report_lines.append(f"Dominant Frequency: {dominant['Frequency']:.4g} | Dominant Period: {dominant['Period']:.4g} (Amplitude {dominant['Amplitude']:.4g})")
    if topo_interpretation is not None:
        if topo_interpretation.top_candidate is not None:
            report_lines.append(f"Top Landform Class: {topo_interpretation.top_candidate}")
        if not topo_interpretation.metrics.empty:
            for _, _mrow in topo_interpretation.metrics.iterrows():
                try:
                    _val = f"{float(_mrow['Value']):.4g}"
                except (TypeError, ValueError):
                    _val = str(_mrow["Value"])
                report_lines.append(f"{_mrow['Metric']}: {_val} ({_mrow['Used in score']})")

    st.markdown("\n".join([f"- {line}" for line in report_lines]))

    st.markdown("**Processing Notes**")
    if st.session_state.status_notes:
        st.markdown("\n".join([f"- {line}" for line in st.session_state.status_notes]))
    else:
        st.markdown("- No preprocessing notes yet.")

    if st.session_state.ai_last_response:
        st.markdown("**AI Summary**")
        st.markdown(f"- Provider: {st.session_state.ai_last_provider}")
        st.markdown(st.session_state.ai_last_response)

    export_1, export_2, export_3, export_4 = st.columns(4)
    with export_1:
        st.download_button(
            "Processed CSV",
            data=dataframe_download_bytes(processed_df),
            file_name="spectrumearth_processed_data.csv",
            mime="text/csv",
            width="stretch",
        )
    with export_2:
        if baseline_df is not None:
            st.download_button(
                "Baseline CSV",
                data=dataframe_download_bytes(baseline_df),
                file_name="spectrumearth_baseline.csv",
                mime="text/csv",
                width="stretch",
            )
    with export_3:
        if peak_results is not None and not peak_results.table.empty:
            st.download_button(
                "Peak Table CSV",
                data=dataframe_download_bytes(peak_results.table),
                file_name="spectrumearth_peaks.csv",
                mime="text/csv",
                width="stretch",
            )
    with export_4:
        if fft_results is not None and not fft_results.table.empty:
            st.download_button(
                "FFT Table CSV",
                data=dataframe_download_bytes(fft_results.table),
                file_name="spectrumearth_fft.csv",
                mime="text/csv",
                width="stretch",
            )

    if topo_interpretation is not None and not topo_interpretation.metrics.empty:
        topo_export_1, topo_export_2 = st.columns(2)
        with topo_export_1:
            st.download_button(
                "Landform Metrics CSV",
                data=dataframe_download_bytes(topo_interpretation.metrics),
                file_name="spectrumearth_landform_metrics.csv",
                mime="text/csv",
                width="stretch",
            )
        with topo_export_2:
            if not topo_interpretation.evidence.empty:
                st.download_button(
                    "Landform Evidence CSV",
                    data=dataframe_download_bytes(topo_interpretation.evidence),
                    file_name="spectrumearth_landform_evidence.csv",
                    mime="text/csv",
                    width="stretch",
                )


with tabs[7]:
    st.subheader("Shape register")
    st.markdown(
        "Idealized forms measured by the same code that measures your data, plotted in "
        "shape space. Every reference form is scale-free, so position on this grid depends "
        "on shape alone, not on how large the landform is."
    )
    st.info(
        "For any pure power-law wall the two form metrics are the SAME quantity: "
        "V-index = (b - 1) / (b + 1), verified to four decimals for b from 0.5 to 4.0. "
        "So the power-law family is not one axis of a grid - it is a reference manifold, and "
        "the informative quantity is departure from it: delta = V - (b - 1)/(b + 1). "
        "delta is 0 for self-similar walls, NEGATIVE for a flat floor, and POSITIVE for "
        "asymmetry or a tightly curved arc."
    )

    if st.session_state.dataset_type not in TOPO_DATASET_TYPES:
        st.caption(
            "The register is built for topographic profiles. Pick a Topography or Volcano "
            "dataset type to place your own data on it. The reference forms below are shown "
            "regardless."
        )

    try:
        register_df = build_shape_register()
    except Exception as exc:
        register_df = pd.DataFrame()
        st.warning(f"The shape register could not be built: {exc}")
    try:
        lattice_df = build_shape_lattice()
    except Exception as exc:
        lattice_df = pd.DataFrame()
        st.warning(f"The shape lattice could not be built: {exc}")

    if register_df.empty:
        st.info("No reference forms were produced.")
    else:
        measured_point = None
        measured_label = "This profile"
        _tm = st.session_state.get("topo_metrics") or {}
        if _tm:
            _v = _tm.get("cone_index", _tm.get("v_index"))
            _b = _tm.get("power_law_b")
            if _v is not None and _b is not None and np.isfinite(_v) and np.isfinite(_b):
                measured_point = {
                    "b (form)": float(_b),
                    "V-index": float(_v),
                    "Width / depth": float(_tm.get("width_depth_ratio", np.nan))
                    if "width_depth_ratio" in _tm else
                    (1.0 / float(_tm["height_basal_width_ratio"])
                     if _tm.get("height_basal_width_ratio") else np.nan),
                    "Asymmetry": float(_tm.get("asymmetry_index", np.nan)),
                    "delta (off-manifold)": shape_register_delta(float(_v), float(_b)),
                    "Conformity": float(st.session_state.get("conformity_value", np.nan)),
                    "Nearest ideal": st.session_state.get("conformity_best") or "",
                }
                if st.session_state.dataset_type == TOPO_VOLCANO:
                    measured_label = "This edifice"

        _cscores = st.session_state.get("conformity_scores") or {}
        if _cscores:
            st.markdown("**Shape name and departure from the ideal**")
            _cbest = st.session_state.get("conformity_best")
            _cval = float(st.session_state.get("conformity_value", np.nan))
            _c1, _c2 = st.columns([1.0, 1.4], gap="large")
            with _c1:
                _bfit = float((st.session_state.get("topo_metrics") or {}).get("power_law_b", np.nan))
                _fullname, _side = shape_name_with_side(_cbest, _bfit)
                if np.isfinite(_cval):
                    st.metric("Shape name", _fullname,
                              help="The nearest ideal section, and which side of it this profile "
                                   "falls on. Conformity is a magnitude and cannot supply the "
                                   "side on its own.")
                    st.metric("Conformity", f"{_cval:.3f}",
                              delta=f"{_cval - 1.0:+.3f} from ideal", delta_color="off")
                    st.caption(IDEAL_SECTION_NOTES.get(_cbest, ""))
                    if _side and _side != "on it":
                        st.caption(
                            f"b = {_bfit:.3f} against the ideal's {IDEAL_SECTION_B.get(_cbest, float('nan')):.2f}. "
                            "Two profiles can score the same conformity on opposite sides of an "
                            "ideal - a convex shield and a concave composite both read close to "
                            "Cone / V - so the side is the part that separates them."
                        )
                    if _cval >= 0.98:
                        st.success("Essentially indistinguishable from the ideal form.")
                    elif _cval >= 0.90:
                        st.info("Close to the ideal, with a measurable departure.")
                    elif _cval >= 0.70:
                        st.warning("Departs clearly from every ideal section.")
                    else:
                        st.warning(
                            "Low conformity to EVERY symmetric ideal. That is itself the reading: "
                            "the section is irregular or strongly asymmetric. Check the asymmetry "
                            "index before naming it."
                        )
            with _c2:
                st.plotly_chart(make_conformity_bars(_cscores, _cbest), width="stretch")

            st.markdown("**Ideal overlaid on the section**")
            st.caption(
                "The ideal is pinned to the same thalweg, the same base and the same height, then "
                "subtracted - the grain-shape construction. Blue is rock ABOVE the ideal, red is "
                "rock MISSING relative to it. Two numbers come out and they measure different "
                "things: NET is signed and describes gross form (it reproduces the V-index, which "
                "is this same subtraction against a triangle), while IRREGULARITY is unsigned and "
                "describes how ragged the section is. A cone carrying waves that preserve its area "
                "reads net +0.007 - a perfect cone - but irregularity 0.099."
            )
            _ov_names = list(IDEAL_SECTIONS.keys())
            _ov_default = _cbest if _cbest in _ov_names else _ov_names[0]
            _ov_pick = st.selectbox(
                "Overlay which ideal?", _ov_names,
                index=_ov_names.index(_ov_default), key="overlay_ideal_pick",
                help="Defaults to the nearest-fitting ideal. Pick another to see how the "
                     "difference changes.",
            )
            _ov_y = -y_metric if st.session_state.dataset_type == TOPO_VOLCANO else y_metric
            try:
                _ovm, _ovc = ideal_overlay(
                    x, _ov_y, _ov_pick,
                    height_fraction=float(st.session_state.topo_height_fraction),
                    spec_height=spec_h if "spec_h" in dir() else None,
                )
            except Exception:
                _ovm, _ovc = {}, {}
            if _ovc:
                st.plotly_chart(
                    make_overlay_plot(_ovc, _ovm,
                                      inverted=(st.session_state.dataset_type == TOPO_VOLCANO)),
                    width="stretch",
                )
                _o1, _o2, _o3, _o4 = st.columns(4)
                _o1.metric("Net (signed)", f"{_ovm.get('form_net', float('nan')):+.4f}",
                           help="Gross form. Positive means more material than the ideal.")
                _o2.metric("Irregularity", f"{_ovm.get('form_irregularity', float('nan')):.4f}",
                           help="How ragged. Zero only if the section IS the ideal.")
                _o3.metric("Excess area", f"{_ovm.get('form_excess', float('nan')):.4f}")
                _o4.metric("Deficit area", f"{_ovm.get('form_deficit', float('nan')):.4f}")
                st.caption(
                    "All four are fractions of the ideal's own area, so they are size-independent. "
                    "Net and irregularity are independent readings: a section can have net zero "
                    "and large irregularity, which is the difference between sphericity and "
                    "roundness in grain work."
                )
            else:
                st.info("The overlay could not be fitted for this profile.")

        st.markdown("---")

        _vcol, _bcol = st.columns([1.6, 1.0], gap="large")
        with _vcol:
            view = st.radio(
                "Register view",
                ["Form vs aspect", "Conformity vs form", "Off-manifold vs form",
                 "Off-manifold vs asymmetry"],
                horizontal=True,
                key="register_view",
            )
        with _bcol:
            backdrop = st.radio(
                "Backdrop",
                ["Shaded regions", "Lattice grid", "None"],
                horizontal=True,
                key="register_backdrop",
                help="Shaded regions replace the contour lattice with labelled fields. "
                     "Read values off the axes.",
            )
        _pinned = st.session_state.get("register_pinned") or []
        st.plotly_chart(
            make_shape_register_plot(register_df, view, measured_point, measured_label,
                                     _pinned, lattice_df, backdrop),
            width="stretch",
        )
        if backdrop == "Shaded regions":
            st.caption(
                "Each shaded field is the measured envelope of the ideal forms that produce it - "
                "the convex hull of where those families actually land, not a drawn box, so the "
                "fields move if the metrics change. A profile inside a field has that character; "
                "one in open space has none of them. Read values off the axes. The dotted grey "
                "track is the composite V-in-U family, drawn as a trajectory rather than a field "
                "because a re-incised valley MOVES between fields as its inner gorge grows."
            )
        elif view == "Form vs aspect" and backdrop == "Lattice grid":
            st.caption(
                "The grey lattice is the power-law manifold drawn as graph paper: iso-b verticals "
                "crossed with iso-aspect horizontals. It is rectangular because shape and aspect "
                "are independent - measured, not assumed. b_fit recovers its input at every aspect "
                "from W/D 2 to 40, and V-index does not move with aspect at all. That is also why "
                "each reference family is a whole vertical LINE rather than a point: a flat-floored "
                "section reads the same b whether it is narrow or wide."
            )
        if _pinned:
            st.caption(
                "Pinned samples are numbered black markers keyed to the table below; the open "
                "diamond is the profile currently loaded. Groups are separated by marker SHAPE, "
                "because the reference families already hold the colour palette."
            )

        if measured_point is None:
            st.caption(
                "Load a topographic profile and your data will appear on the grid as a black "
                "diamond."
            )
        else:
            _d = measured_point["delta (off-manifold)"]
            if np.isfinite(_d):
                _is_edifice = st.session_state.dataset_type == TOPO_VOLCANO
                _flat_word = "a flattened summit" if _is_edifice else "a flattened floor"
                if _d < -0.03:
                    _reading = ("below the manifold, which among the reference forms means "
                                + _flat_word)
                elif _d > 0.03:
                    _reading = ("above the manifold, which among the reference forms means "
                                "asymmetry or a tightly curved section - read it against the "
                                "asymmetry index before deciding which")
                else:
                    _reading = "on the manifold, so the section is close to a single power-law form"
                st.success(
                    f"This profile sits at b = {measured_point['b (form)']:.3f}, "
                    f"V-index = {measured_point['V-index']:+.3f}, delta = {_d:+.3f} - {_reading}."
                )

        if backdrop == "Shaded regions":
            _rcols = st.columns(len(SHAPE_REGIONS), gap="medium")
            for _c, (_rn, _rs) in zip(_rcols, SHAPE_REGIONS.items()):
                with _c:
                    st.markdown(
                        f"<div style='border-left:5px solid {_rs['colour']}; padding-left:8px;'>"
                        f"<b>{_rn}</b><br><span style='font-size:0.86em; color:#52514e;'>"
                        f"{_rs['note']}</span></div>",
                        unsafe_allow_html=True,
                    )


        st.markdown("---")
        st.markdown("**Pin samples to the register**")
        st.caption(
            "Add the loaded profile to the grid, then load another file and add that one too. "
            "Pinned samples persist for the session, so you can build up a population and see "
            "where the forms fall relative to each other and to the manifold."
        )

        pin_1, pin_2, pin_3, pin_4 = st.columns([1.5, 1.1, 0.7, 0.7], gap="medium")
        _default_label = (
            str(st.session_state.get("source_name", "")).rsplit(".", 1)[0]
            or f"sample {len(st.session_state.get('register_pinned') or []) + 1}"
        )
        with pin_1:
            _pin_label = st.text_input("Label", value=_default_label, key="register_pin_label")
        with pin_2:
            _pin_group = st.text_input(
                "Group (optional)", key="register_pin_group",
                help="Samples sharing a group get the same marker shape - e.g. glacial, fluvial, Mars.",
            )
        with pin_3:
            _add = st.button("Add to register", width="stretch", disabled=measured_point is None)
        with pin_4:
            _clear = st.button("Clear all", width="stretch",
                               disabled=not st.session_state.get("register_pinned"))

        if _add and measured_point is not None:
            _entry = dict(measured_point)
            _entry["Label"] = _pin_label or _default_label
            _entry["Group"] = _pin_group or ""
            _entry["Dataset"] = st.session_state.dataset_type
            _entry["Source"] = st.session_state.get("source_name", "")
            _pins = list(st.session_state.get("register_pinned") or [])
            _pins.append(_entry)
            st.session_state["register_pinned"] = _pins
            st.rerun()

        if _clear:
            st.session_state["register_pinned"] = []
            st.rerun()

        if measured_point is None:
            st.caption("Load a topographic profile to enable pinning.")

        _pins = st.session_state.get("register_pinned") or []
        if _pins:
            _prows = []
            for _i, _p in enumerate(_pins, start=1):
                _prows.append({
                    "#": _i,
                    "Label": _p.get("Label", ""),
                    "Group": _p.get("Group", ""),
                    "Dataset": _p.get("Dataset", ""),
                    "b (form)": _p.get("b (form)", float("nan")),
                    "V-index": _p.get("V-index", float("nan")),
                    "Width / depth": _p.get("Width / depth", float("nan")),
                    "Asymmetry": _p.get("Asymmetry", float("nan")),
                    "delta (off-manifold)": _p.get("delta (off-manifold)", float("nan")),
                    "Conformity": _p.get("Conformity", float("nan")),
                    "Nearest ideal": _p.get("Nearest ideal", ""),
                })
            _pdf = pd.DataFrame(_prows)
            _pshow = _pdf.copy()
            for _c in ("b (form)", "V-index", "Width / depth", "Asymmetry",
                       "delta (off-manifold)", "Conformity"):
                _pshow[_c] = _pshow[_c].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
            st.dataframe(_pshow, width="stretch", hide_index=True)

            _rm_1, _rm_2 = st.columns([2.0, 1.0], gap="medium")
            with _rm_1:
                _drop = st.multiselect(
                    "Remove pinned samples",
                    options=[f"{r['#']} - {r['Label']}" for r in _prows],
                    key="register_drop",
                )
            with _rm_2:
                if st.button("Remove selected", width="stretch", disabled=not _drop):
                    _keep_idx = {int(d.split(" - ", 1)[0]) for d in _drop}
                    st.session_state["register_pinned"] = [
                        p for i, p in enumerate(_pins, start=1) if i not in _keep_idx
                    ]
                    st.rerun()

            st.download_button(
                "Pinned Samples CSV",
                data=dataframe_download_bytes(_pdf),
                file_name="spectrumearth_pinned_samples.csv",
                mime="text/csv",
            )

        st.markdown("---")
        left_reg, right_reg = st.columns([1.0, 1.25], gap="large")
        with left_reg:
            fam = st.selectbox(
                "Reference family", list(SHAPE_REGISTER_FAMILIES.keys()), key="register_family"
            )
            st.caption(SHAPE_REGISTER_FAMILIES[fam]["note"])
        with right_reg:
            _mxy = None
            if st.session_state.dataset_type in TOPO_DATASET_TYPES:
                _yy = y_metric if "y_metric" in dir() else None
                if _yy is not None:
                    _mxy = (x, -_yy if st.session_state.dataset_type == TOPO_VOLCANO else _yy)
            st.plotly_chart(make_ideal_shapes_plot(fam, _mxy), width="stretch")

        st.markdown("---")
        st.markdown("**Ideal sections across the lattice**")
        st.caption(
            "Read across a row to see form change at fixed proportions; read down a column to see "
            "one form get wider and shallower. Each row is drawn to its own scale, so the aspect "
            "ratio you see is the real one."
        )
        gal_1, gal_2 = st.columns(2, gap="medium")
        with gal_1:
            _gal_b = st.multiselect(
                "Forms to show (b)", SHAPE_LATTICE_B,
                default=[0.5, 1.0, 1.5, 2.0, 3.0], key="gallery_b",
            )
        with gal_2:
            _gal_wd = st.multiselect(
                "Aspects to show (W/D)", SHAPE_LATTICE_WD,
                default=[2.0, 6.0, 15.0, 32.0], key="gallery_wd",
            )
        if _gal_b and _gal_wd:
            st.plotly_chart(
                make_shape_gallery(sorted(_gal_b), sorted(_gal_wd)), width="stretch"
            )
        else:
            st.info("Pick at least one form and one aspect to draw the gallery.")

        st.markdown("---")
        st.markdown("**Lattice table**")
        st.caption(
            "Every node measured by compute_cross_section_metrics. b_in and wd_in are what went "
            "in; the measured columns are what came back."
        )
        if not lattice_df.empty:
            _lshow = lattice_df.copy()
            for _c in ("b (form)", "Width / depth", "V-index"):
                _lshow[_c] = _lshow[_c].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
            st.dataframe(_lshow, width="stretch", hide_index=True)
            st.download_button(
                "Shape Lattice CSV",
                data=dataframe_download_bytes(lattice_df),
                file_name="spectrumearth_shape_lattice.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.markdown("**Register table**")
        st.caption(
            "Every row is an idealized form put through compute_cross_section_metrics, the same "
            "function your data goes through. The reference and the measurement cannot drift apart."
        )
        _show = register_df.copy()
        for _c in ("b (form)", "V-index", "Width / depth", "Asymmetry",
                   "delta (off-manifold)", "Fit r2", "Conformity"):
            _show[_c] = _show[_c].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
        st.dataframe(_show, width="stretch")
        st.download_button(
            "Shape Register CSV",
            data=dataframe_download_bytes(register_df),
            file_name="spectrumearth_shape_register.csv",
            mime="text/csv",
        )
