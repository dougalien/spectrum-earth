"""
The Ideal Shape and Shape Chart views of Spectrum Earth.

Everything measured here comes from ideal_shape.py; this module only draws it
and lays it out. Every landform gets the same two things: the plot of the fit
(the real profile, its ideal, and the extra and missing rock between them) and
its departure number. The ideal is either the best-fitting curve or the fixed
triangle (nothing fitted); the choice is made once, at the top of the app, and
every plot and number follows it. Extra and missing rock are a polarity, so
they take the diverging pair (blue / red); feature types on the shape chart
take the first three categorical slots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ideal_shape as ideal

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8981"
GRID = "#eceae4"
FAINT = "#d8d6ce"
EXCESS_FILL = "rgba(42,120,214,0.30)"
DEFICIT_FILL = "rgba(208,59,59,0.30)"
HIGHLIGHT = "#2a78d6"

FEATURE_COLOURS = {"hill": "#2a78d6", "valley": "#eb6834", "profile": "#1baf7a"}

FEATURE_CHOICES = {
    "Valley": ideal.VALLEY,
    "Hill or mountain": ideal.HILL,
    "One slope (long profile)": ideal.SLOPE,
    # A hill framed from its spreading axis, with the axis read too (ideal.read_ridge).
    "Spreading ridge": ideal.HILL,
}
RIDGE_FEATURE = "Spreading ridge"

DATASETS = {
    "Copernicus DEM GLO-30": (
        "Copernicus DEM GLO-30, 1 arc-second (about 30 m), read from the public copy on AWS "
        "(s3://copernicus-dem-30m). © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH "
        "2014-2018, provided under COPERNICUS by the European Union and ESA. "
        "doi:10.5270/ESA-c5d3d65"),
    "ETOPO1": (
        "ETOPO1 1 arc-minute global relief model (Amante and Eakins, 2009, NOAA Technical "
        "Memorandum NESDIS NGDC-24, doi:10.7289/V5C8276M), read from the AWS Terrain Tiles "
        "(s3://elevation-tiles-prod); each point is the median of 7 parallel lines across 30 km."),
}
RATE_SOURCE = ("Full spreading rates are approximate, from the MORVEL plate motion model "
               "(DeMets, Gordon and Argus, 2010, Geophysical Journal International 181, 1-80).")

# What every landform is compared with. The best fit is the curve that leaves
# the least rock out of place; the fixed triangle has nothing fitted.
BEST = "best"
TRIANGLE_AGAINST = ideal.TRIANGLE
AGAINST = {"Best-fit curve": BEST, "Fixed triangle": TRIANGLE_AGAINST}
AGAINST_HELP = (
    "Best-fit curve: the ideal curve that leaves the least rock out of place. Fixed triangle: one "
    "rigid ideal with nothing fitted, its base between the two feet and its apex at the full height "
    "midway between them, with straight sides. Every plot and number follows this choice."
)
TRIANGLE_LINE = "a perfect parabola"

DISTANCE_UNITS = {"metres": 1.0, "kilometres": 1000.0, "feet": 0.3048, "miles": 1609.344}
ELEVATION_UNITS = {"metres": 1.0, "feet": 0.3048, "kilometres": 1000.0}


@st.cache_data(show_spinner="Fitting the ideal...", max_entries=128)
def fit_profile(x: np.ndarray, z: np.ndarray, feature: str) -> tuple[dict, dict | None]:
    """(measurement, ridge reading or None) for one profile in metres, by the fixed procedure."""
    if feature == RIDGE_FEATURE:
        ridge = ideal.read_ridge(x, z)
        return ridge["whole"], ridge
    return ideal.measure(x, z, FEATURE_CHOICES[feature]), None


def read_against(item: dict, against: str) -> dict:
    """The measurement read against the chosen ideal: the best fit as measured, or the fixed triangle."""
    return ideal.as_triangle(item) if against == ideal.TRIANGLE else item


AGAINST_NAMES = {BEST: "best-fit curve", ideal.TRIANGLE: "fixed triangle"}


def average_grade(d: float, against: str) -> str:
    """The provisional grade of a best-fit departure; none against the fixed triangle."""
    return "" if against == ideal.TRIANGLE else ideal.departure_class(d)


def triangle_words(item: dict) -> str:
    """The fixed triangle in words, with where the real top sits against it."""
    if item["kind"] == ideal.SLOPE:
        return "Ideal: the fixed triangle's side, a straight line from the head to the mouth"
    where = offset_words(item["triangle"]["offset"])
    if item["kind"] == ideal.VALLEY:
        return ("Ideal: the fixed triangle upside down (a V), straight walls from each rim down to a point "
                f"at the full depth midway between the rims; the real valley floor is {where}")
    return ("Ideal: the fixed triangle, straight flanks from each foot up to an apex at the full height "
            f"midway between the feet; the real summit is {where}")


def ideal_label(against: str) -> str:
    return "fixed triangle" if against == ideal.TRIANGLE else "ideal"


def offset_words(offset: float) -> str:
    """Where the summit sits against the middle of the feet, e.g. '12% of the width left of the middle'."""
    if abs(offset) < 0.005:
        return "on the middle"
    return f"{abs(offset):.0%} of the width {'left' if offset < 0 else 'right'} of the middle"


def load_library(folder: Path) -> dict:
    """The real transect library: {menu label: feature row with its transects}, in file order."""
    try:
        feats = pd.read_csv(folder / "features.csv")
        trans = pd.read_csv(folder / "transects.csv")
    except Exception:
        return {}
    out = {}
    for f in feats.to_dict("records"):
        kind = {"volcano": "Volcano", "valley": "Valley", "ridge": "Ridge"}[f["feature"]]
        label = f"{kind}: {f['name']}, {f['place']}"
        if f["feature"] == "ridge":
            label = f"{kind}: {f['name']}, {f['notes'].rstrip('.').lower()} ({f['full_rate_mm_yr']:.0f} mm/yr)"
        f["transects"] = trans[trans.feature_id == f["id"]].to_dict("records")
        out[label] = f
    return out


def library_citation(entry: dict) -> str:
    """Where a library feature's transects come from, in words."""
    t = entry["transects"]
    dataset = DATASETS["ETOPO1" if entry["feature"] == "ridge" else "Copernicus DEM GLO-30"]
    placed = "; ".join(
        f"{r['column']} through {r['centre_lat']:.4f}, {r['centre_lon']:.4f} at {r['azimuth_deg']:.0f}°"
        for r in t)
    lines = [f"**{entry['name']}**, {entry['place']}. {entry['notes']}",
             f"{len(t)} transects, each {2 * entry['half_km']:g} km long (centre latitude, longitude and "
             f"azimuth): {placed}.",
             f"Data: {dataset}"]
    if entry["feature"] == "ridge":
        lines.append(f"Full spreading rate about {entry['full_rate_mm_yr']:.0f} mm/yr. {RATE_SOURCE}")
    return "  \n".join(lines)


def _layout(fig, title: str, height: int, **kw):
    fig.update_layout(
        title=title, template="plotly_white", height=height,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _pct(v: float, signed: bool = False) -> str:
    if not np.isfinite(v):
        return "-"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def _num(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.4g}"


# =====================================================================
# The plot of the fit
# =====================================================================
def overlay_figure(item: dict, x_label: str, y_label: str, title: str = "", height: int = 470,
                   axis: float | None = None, legend: bool = True):
    """The real profile with its ideal drawn on it, extra and missing rock shaded."""
    fig = go.Figure()
    x, zr, zi = ideal.drawn_halves(item)
    pad = 0.15 * (item["m"]["x_right"] - item["m"]["x_left"])
    sel = (item["x"] >= item["m"]["x_left"] - pad) & (item["x"] <= item["m"]["x_right"] + pad)
    fig.add_trace(go.Scatter(
        x=item["x"][sel], y=item["z"][sel], mode="lines", name="data beyond the feet",
        line=dict(width=1, color=FAINT), hoverinfo="skip", showlegend=legend,
    ))
    fig.add_trace(go.Scatter(x=x, y=np.maximum(zr, zi), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=zi, mode="lines", fill="tonexty", fillcolor=EXCESS_FILL,
                             line=dict(width=0), name="extra rock (above the ideal)", hoverinfo="skip",
                             showlegend=legend))
    fig.add_trace(go.Scatter(x=x, y=np.minimum(zr, zi), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=zi, mode="lines", fill="tonexty", fillcolor=DEFICIT_FILL,
                             line=dict(width=0), name="missing rock (below the ideal)", hoverinfo="skip",
                             showlegend=legend))
    name = ideal_label(item.get("against", BEST))
    fig.add_trace(go.Scatter(
        x=x, y=zi, mode="lines", name=name, line=dict(width=2, color=INK_2, dash="dash"),
        hovertemplate=name + " %{y:.4g}<extra></extra>", showlegend=legend,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=zr, mode="lines", name="profile", line=dict(width=3, color=INK),
        hovertemplate="profile %{y:.4g}<extra></extra>", showlegend=legend,
    ))
    if axis is not None:
        fig.add_vline(x=axis, line=dict(color=HIGHLIGHT, width=1.5, dash="dash"),
                      annotation_text="axis", annotation_position="top")
    _layout(fig, title, height, hovermode="x unified")
    if not legend:
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    return fig


def to_display_units(item: dict, sx: float, sz: float) -> dict:
    """Positions and elevations back in the file's own units, after a fit done in common units.

    Only the drawing changes: every share and departure is dimensionless.
    """
    if not item.get("ok") or (sx == 1.0 and sz == 1.0):
        return item
    out = dict(item)
    out["x"] = item["x"] / sx
    out["z"] = item["z"] / sz
    m = dict(item["m"])
    for k in ("x_left", "x_right", "x_apex", "width", "x_flat_left", "x_flat_right"):
        if k in m:
            m[k] = m[k] / sx
    for k in ("base_elev", "apex_elev", "height", "base_left", "base_right"):
        m[k] = m[k] / sz
    out["m"] = m
    out["halves"] = [dict(h, radius=h["radius"] / sx, base=h["base"] / sz, height=h["height"] / sz)
                     for h in item["halves"]]
    return out


def ridge_to_display_units(ridge: dict, sx: float, sz: float) -> dict:
    return dict(ridge, axis=ridge["axis"] / sx)


def default_feature(dataset_type: str, volcano_type: str, long_type: str) -> str:
    if dataset_type == volcano_type:
        return "Hill or mountain"
    if dataset_type == long_type:
        return "One slope (long profile)"
    return "Valley"


# =====================================================================
# The Ideal Shape tab
# =====================================================================
def render_explainer():
    with st.expander("How the departure is found - the one procedure for every landform", expanded=False):
        st.markdown(
            """
**The idea comes from grain shape.** Wadell fitted a sand grain with a perfect sphere and measured
the material left over; how far a grain departs from the sphere records how it has been worn. Here
the grain is a landform and the sphere is the best-fitting ideal curve.

Every landform, of every type and size, goes through the same steps, with no other rules:

1. **Frame.** Find the top - the summit of a hill, the floor of a valley (read upside down), the
   head of a long profile, the spreading axis of a ridge - and the foot on each side. Both feet sit
   at one base level, the higher of the two sides' lowest ground. Each side goes in a box as wide
   as that side and as tall as the landform, and the real profile and the ideal are compared in
   that box, so a 200 m valley and a 5 km volcano are compared at the same scale.
2. **Ideal.** Down each side runs one smooth curve, held level across the top for as far as fits
   best (a flat top, or a flat floor). The curve is one of three, each with one bend control:
   - **parabola**, g = (1 - u)(1 - k u): from concave (k = 1) through straight (k = 0) to a dome (k = -1);
   - **exponential**, g = (e^(-a u) - e^(-a)) / (1 - e^(-a)): steep at the top and still sloping at
     the foot when a > 0, like a graded river;
   - **superellipse**, g = (1 - u^e)^(1/e): deeply concave (e < 1), straight (e = 1), a quarter
     circle (e = 2), toward a box (e > 2).

   Here u runs from the top (0) to the foot (1) and g is the height as a share of the ideal's top.
   The ideal's top height - the same on both sides, anywhere up to the real top - and each side's
   level stretch, curve and bend are all chosen together to leave the least rock out of place.
3. **Departure.** The rock out of place - extra rock above the ideal (blue) plus missing rock
   below it (red) - as a share of the box. **0 means the landform is its ideal**; the further it
   departs, the bigger the number. It is exactly the quantity the fit makes as small as it can.
4. **Several transects.** Each transect of a landform goes through steps 1 to 3 and the departures
   are averaged, with their spread.

**The name** puts the ideal into words: its sides are concave (bowed in), straight or convex
(bowed out), and "with a flat top" (or floor) is added when its level stretch covers a tenth of the
width or more. The name never changes the fit or the number.

**The grades** (near-ideal under 1%, slightly modified under 2.5%, modified under 5%, strongly
modified above) are provisional: they were set from a few profiles before the real library was added,
not from a calibrated population. They apply to the best fit only.

**The fixed triangle.** With enough freedom a fitted curve might in the end match any profile, so
every landform can also be compared with one rigid ideal that has nothing fitted. Its base runs
between the two feet found in step 1, its apex stands at the landform's full height midway between
the feet, and its sides are straight: an isosceles triangle in real distance (upside down for a
valley, and one straight side from head to mouth for a long profile). The rock out of place is
measured as in step 3, as a share of the same box. Bent sides and a summit off the middle both
count. Landforms lie between two end members, the triangle and the parabola: a perfect parabola
centred between its feet, bowed in or domed, scores 16.7% against the triangle.
            """
        )


def _transects_from_table(df: pd.DataFrame, name: str) -> list:
    """(label, x, z) for every elevation column of a table whose first numeric column is distance."""
    num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    if num.shape[1] < 2:
        return []
    x = num.iloc[:, 0].to_numpy(dtype=float)
    out = []
    for col in num.columns[1:]:
        z = num[col].to_numpy(dtype=float)
        label = name if num.shape[1] == 2 else f"{name}: {col}"
        out.append((label, x, z))
    return out


def render_transects(feature: str, sx: float, sz: float, x_label: str, y_label: str, read_file,
                     library_entry: dict | None = None, library_table: pd.DataFrame | None = None,
                     against: str = BEST):
    """Step 4: several transects of one landform, each fitted the same way, then averaged.

    A landform from the library brings all its transects with it; otherwise they are uploaded.
    """
    st.markdown("#### Several transects of one landform")
    if library_entry is not None and library_table is not None:
        transects = _transects_from_table(library_table, library_entry["name"])
        st.caption(library_citation(library_entry))
        _render_transect_average(transects, feature, sx, sz, x_label, y_label, library_entry["name"], against)
        return
    files = st.file_uploader(
        "Transect files", type=["csv", "txt", "xlsx"], accept_multiple_files=True,
        key="ideal_transects",
        help="One transect per file (distance, elevation), or one distance column followed by "
             "several elevation columns, one per transect. All are read as the feature and in the "
             "units chosen above.",
    )
    if not files:
        st.caption("Load two or more transects across the same landform to average their departures.")
        return
    transects = []
    for f in files:
        try:
            transects += _transects_from_table(read_file(f), f.name.rsplit(".", 1)[0])
        except Exception as e:
            st.warning(f"{f.name} could not be read: {e}")
    _render_transect_average(transects, feature, sx, sz, x_label, y_label, "", against)


def _render_transect_average(transects: list, feature: str, sx: float, sz: float, x_label: str,
                             y_label: str, name: str, against: str = BEST):
    if not transects:
        return
    items = []
    for label, x, z in transects:
        it, _ = fit_profile(x * sx, z * sz, feature)
        items.append((label, read_against(to_display_units(it, sx, sz), against)))
    avg = ideal.average([it for _, it in items])
    if avg["n"] == 0:
        st.warning("None of the transects could be fitted.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Average departure", _pct(avg["mean"]),
              help="The mean of the transects' departures. Each is fitted on its own by the same "
                   "procedure.")
    c2.metric("Spread", f"± {avg['sd'] * 100:.1f}%",
              help="Standard deviation of the transects' departures.")
    c3.metric("Transects", f"{avg['n']}", help=f"Range {_pct(avg['lo'])} to {_pct(avg['hi'])}.")
    grade = average_grade(avg["mean"], against)
    st.caption((f"Grade of the average: **{grade}**. " if grade else "Against the fixed triangle. ")
               + f"Range {_pct(avg['lo'])} to {_pct(avg['hi'])}"
               + (f"; {avg['failed']} transect(s) could not be fitted." if avg["failed"] else "."))
    cols = st.columns(2)
    for i, (label, it) in enumerate(items):
        with cols[i % 2]:
            if it.get("ok"):
                st.plotly_chart(overlay_figure(it, x_label, y_label, f"{label}: {_pct(it['departure'])}",
                                               height=300, legend=False), width="stretch")
            else:
                st.warning(f"{label}: {it.get('problem') or 'could not be fitted.'}")
    g1, g2 = st.columns([2, 1])
    with g1:
        label = st.text_input("Label for the average", value=name or f"average of {avg['n']} transects",
                              key="ideal_transect_label")
    with g2:
        st.write("")
        if st.button("Add the average to the shape chart", width="stretch"):
            kind = FEATURE_CHOICES[feature]
            _add_rows([{"Label": label, "Group": "", "Feature": ideal.VOCAB[kind]["feature"],
                        "Shape": f"average of {avg['n']} transects", "Departure": avg["mean"],
                        "Grade": grade, "Spread (sd)": avg["sd"], "Transects": avg["n"],
                        "Ideal": AGAINST_NAMES[against]}], against)
            st.success("Added. See the Shape Chart tab.")


def render_ideal_tab(item: dict, x_label: str, y_label: str, source_name: str, ridge: dict | None = None,
                     feature: str = "", units: tuple = (1.0, 1.0), read_file=None,
                     library_entry: dict | None = None, library_table: pd.DataFrame | None = None,
                     against: str = BEST):
    if ridge is not None:
        st.caption(
            f"Spreading axis: **{ridge['axis_type']}**, read as **{ridge['rate_words']}** "
            "(provisional). The ridge is framed from its axis; the fit and the departure are the "
            "same as for any other landform."
        )
    if not item.get("ok"):
        st.warning(item.get("problem") or "This profile could not be fitted.")
    else:
        words = ideal.VOCAB[item["kind"]]
        st.markdown(f"### {item['name'][0].upper() + item['name'][1:]}")
        tri = item.get("against") == ideal.TRIANGLE
        c1, c2, c3 = st.columns([1.0, 1.2, 2.2])
        c1.metric("Departure", _pct(item["departure"]),
                  help=("Rock out of place against the fixed triangle" if tri else
                        "Rock out of place against the best-fitting ideal")
                       + " - extra plus missing - as a share of the landform's box (width times "
                       "height). 0 means the profile IS its ideal; the further it departs, the bigger "
                       "the number.")
        if tri and item["kind"] != ideal.SLOPE:
            c2.metric(f"{words['apex'].capitalize()} off the middle", f"{abs(item['triangle']['offset']):.0%}",
                      help=f"How far the {words['apex']} sits from the middle of the two feet, as a share "
                           "of the width. The triangle's apex is always on the middle.")
        elif not tri:
            c2.metric("Grade", item["grade"], help="Provisional: under 1% near-ideal, under 2.5% "
                                                   "slightly modified, under 5% modified.")
        c3.metric("Extra rock / missing rock", f"{_pct(item['excess'])} / {_pct(item['deficit'])}",
                  help="The two parts of the departure: rock above the ideal (blue) and rock missing "
                       "below it (red), each as a share of the box.")
        st.plotly_chart(overlay_figure(item, x_label, y_label, axis=ridge["axis"] if ridge else None),
                        width="stretch")
        if tri:
            st.caption(triangle_words(item) + ". Nothing is fitted. For scale, " + TRIANGLE_LINE
                       + " centred between its feet scores 16.7%.")
        else:
            st.caption(
                f"Ideal: {ideal.fit_words(item)}, its top at {item['fit']['top']:.0%} of the height of the "
                f"{words['feature']}"
                + (f", level across {item['level']:.0%} of the width" if item["flat"] else "") + "."
            )
        for n in item.get("notes", []):
            st.info(n)
    if read_file is not None and feature:
        render_transects(feature, units[0], units[1], x_label, y_label, read_file,
                         library_entry, library_table, against)
    render_explainer()


# =====================================================================
# The Shape Chart tab: landforms ranked by their departure
# =====================================================================
def departure_bands(fig, top: float, against: str):
    """The provisional grades behind a best-fit chart, or the parabola's mark on a triangle chart."""
    if against == ideal.TRIANGLE:
        fig.add_vline(x=ideal.TRIANGLE_PARABOLA * 100.0, line=dict(color=INK_2, width=1, dash="dot"),
                      annotation_text=TRIANGLE_LINE, annotation_position="top",
                      annotation_font=dict(size=11, color=INK_2))
        return
    lo = 0.0
    shades = ["rgba(27,175,122,0.07)", "rgba(237,161,0,0.06)", "rgba(235,104,52,0.06)",
              "rgba(208,59,59,0.06)"]
    for i, ((limit, label), shade) in enumerate(zip(ideal.DEPARTURE_CLASSES, shades)):
        hi = min(limit * 100.0, top)
        if lo >= top:
            break
        # Labels alternate top and bottom, so narrow neighbouring bands stay legible.
        fig.add_vrect(x0=lo, x1=hi, fillcolor=shade, line_width=0, layer="below",
                      annotation_text=label, annotation_position="top left" if i % 2 == 0 else "bottom left",
                      annotation_font=dict(size=11, color=INK_2))
        lo = hi


def ranking_figure(rows: list, current: dict | None, against: str = BEST):
    """Every profile added, ranked by departure from its ideal, lowest at the top."""
    pts = [dict(r) for r in rows if np.isfinite(r.get("Departure", np.nan))]
    if current and np.isfinite(current.get("Departure", np.nan)):
        pts.append(dict(current, Label="profile loaded now", **{"#": "now"}))
    pts.sort(key=lambda r: r["Departure"])
    labels = [f"{r.get('#', '')}. {r.get('Label', '')}" if r.get("#") != "now" else "profile loaded now"
              for r in pts]
    floor = ideal.TRIANGLE_PARABOLA * 1.15 if against == ideal.TRIANGLE else 0.06
    top = max([floor] + [r["Departure"] * 1.12 for r in pts]) * 100.0
    fig = go.Figure()
    departure_bands(fig, top, against)
    seen = set()
    for lab, r in zip(labels, pts):
        feat = r.get("Feature", "hill")
        now = r.get("#") == "now"
        fig.add_trace(go.Scatter(
            x=[r["Departure"] * 100.0], y=[lab], mode="markers",
            marker=dict(symbol="diamond-open" if now else "circle", size=14 if now else 11,
                        color=INK if now else FEATURE_COLOURS.get(feat, MUTED),
                        line=dict(width=2.5 if now else 2, color=INK if now else "#fcfcfb")),
            name="profile loaded now" if now else feat,
            showlegend=(feat not in seen) or now, legendgroup="now" if now else feat,
            customdata=[[r.get("Shape", ""), r.get("Grade", "")]],
            hovertemplate=("<b>%{y}</b><br>%{customdata[0]}<br>departure %{x:.2f}% of the box"
                           "<br>%{customdata[1]}<extra></extra>"),
        ))
        if not now:
            seen.add(feat)
    _layout(fig, "Landforms ranked by their departure from the "
            + ("fixed triangle" if against == ideal.TRIANGLE else "ideal"), 160 + 28 * max(len(pts), 3))
    fig.update_layout(margin=dict(l=20, r=20, t=70, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0))
    fig.update_xaxes(title="departure, % of the box (0 = the landform is its ideal)", range=[0, top])
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=labels)
    return fig


def _rows_key(against: str) -> str:
    """Each comparison keeps its own chart, so best-fit and triangle numbers never share an axis."""
    return "ideal_chart_rows" if against == BEST else f"ideal_chart_rows_{against}"


def _add_rows(new_rows: list, against: str = BEST):
    rows = list(st.session_state.get(_rows_key(against)) or [])
    for r in new_rows:
        if r:
            r = dict(r)
            r["#"] = len(rows) + 1
            rows.append(r)
    st.session_state[_rows_key(against)] = rows


def measure_library(library: dict, sample_feature, library_dir: Path, against: str = BEST) -> list:
    """One shape-chart row per library landform: the average of its transects."""
    out = []
    for label, entry in library.items():
        try:
            table = pd.read_csv(library_dir / f"{entry['id']}.csv")
        except Exception:
            continue
        feature = sample_feature(label)
        items = [read_against(fit_profile(x, z, feature)[0], against)
                 for _, x, z in _transects_from_table(table, entry["name"])]
        avg = ideal.average(items)
        if avg["n"] == 0:
            continue
        kind = FEATURE_CHOICES[feature]
        out.append({"Label": entry["name"], "Group": entry["feature"], "Feature": ideal.VOCAB[kind]["feature"],
                    "Shape": f"average of {avg['n']} transects", "Departure": avg["mean"],
                    "Grade": average_grade(avg["mean"], against), "Spread (sd)": avg["sd"],
                    "Transects": avg["n"], "Ideal": AGAINST_NAMES[against]})
    return out


def render_shape_chart(item: dict, source_name: str, library: dict, sample_feature,
                       library_dir: Path, against: str = BEST):
    tri = against == ideal.TRIANGLE
    st.markdown(
        "Landforms organised by their departure from "
        + ("the fixed triangle" if tri else "the ideal")
        + ", lowest first: a landform that IS its ideal sits at 0, and the further it departs, the "
        "further right it sits. Every number comes from the same procedure, so landforms of any type "
        "and size can be compared."
    )
    current = ideal.summary_row(item, "profile loaded now") if item.get("ok") else None
    rows = st.session_state.get(_rows_key(against)) or []
    st.plotly_chart(ranking_figure(rows, current, against), width="stretch")
    st.caption("Colour is the feature type. "
               + ("The dotted line is where " + TRIANGLE_LINE + " centred between its feet sits." if tri
                  else "The shaded bands are the provisional grades."))

    a1, a2, a3, a4, a5 = st.columns([1.4, 1.1, 0.9, 1.0, 0.7], gap="medium")
    default_label = str(source_name or "").rsplit(".", 1)[0] or f"profile {len(rows) + 1}"
    with a1:
        label = st.text_input("Label", value=default_label, key="ideal_pin_label")
    with a2:
        group = st.text_input("Group (optional)", key="ideal_pin_group",
                              help="A note kept with the profile in the table, e.g. glacial, fluvial, Mars.")
    with a3:
        if st.button("Add this profile", width="stretch", disabled=current is None):
            _add_rows([ideal.summary_row(item, label or default_label, group)], against)
            st.rerun()
    with a4:
        if st.button("Add the library", width="stretch",
                     help="Fit every transect of every real landform in the library and add each "
                          "landform's average to the chart."):
            have = {(r.get("Label"), r.get("Group")) for r in rows}
            new = [r for r in measure_library(library, sample_feature, library_dir, against)
                   if (r["Label"], r["Group"]) not in have]
            _add_rows(new, against)
            st.rerun()
    with a5:
        if st.button("Clear", width="stretch", disabled=not rows):
            st.session_state[_rows_key(against)] = []
            st.rerun()

    if rows:
        df = pd.DataFrame(rows).sort_values("Departure")
        show_cols = ["#", "Label", "Group", "Feature", "Shape", "Departure", "Spread (sd)"] + ([] if tri else ["Grade"])
        show = df[[c for c in show_cols if c in df.columns]].copy()
        show["Departure"] = show["Departure"].map(lambda v: _pct(v))
        if "Spread (sd)" in show:
            show["Spread (sd)"] = show["Spread (sd)"].map(lambda v: f"± {v * 100:.1f}%" if np.isfinite(v) else "")
        st.dataframe(show, width="stretch", hide_index=True)
        st.download_button(
            "Shape chart CSV (all columns)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="spectrumearth_shape_chart_triangle.csv" if tri else "spectrumearth_shape_chart.csv",
            mime="text/csv",
        )
    else:
        st.caption("Add the profile loaded now, or press Add the library to see the 30 real "
                   "landforms ranked.")
