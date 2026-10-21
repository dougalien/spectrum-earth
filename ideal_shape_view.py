"""
The Ideal Shape and Shape Chart views of Spectrum Earth.

Everything measured here comes from ideal_shape.py; this module only draws it
and lays it out. Excess and missing rock are a polarity, so they take the
diverging pair (blue / red). Feature types on the shape chart take the first
three categorical slots, the most a scatter can carry safely.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ideal_shape as ideal

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8981"
GRID = "#eceae4"
FAINT = "#d8d6ce"
EXCESS = "#2a78d6"
DEFICIT = "#d03b3b"
EXCESS_FILL = "rgba(42,120,214,0.30)"
DEFICIT_FILL = "rgba(208,59,59,0.30)"
HIGHLIGHT = "#2a78d6"
BAR_MUTED = "#b9c6d8"

FEATURE_COLOURS = {"hill": "#2a78d6", "valley": "#eb6834", "profile": "#1baf7a"}
GROUP_SYMBOLS = ["circle", "square", "triangle-up", "diamond", "triangle-down", "pentagon",
                 "hexagon", "star", "cross", "x"]

FEATURE_CHOICES = {
    "Valley": ideal.VALLEY,
    "Hill or mountain": ideal.HILL,
    "One slope (long profile)": ideal.SLOPE,
}

DISTANCE_UNITS = {"metres": 1.0, "kilometres": 1000.0, "feet": 0.3048, "miles": 1609.344}
ELEVATION_UNITS = {"metres": 1.0, "feet": 0.3048, "kilometres": 1000.0}

CHART_Y = {
    "Departure from the best-fitting ideal": "Departure",
    "Departure from a straight cone or V": "Departure from straight",
}

SHAPE_ORDER = list(ideal.PERFECT_SHAPES)
CHOICE_KEYS = list(ideal.FAMILIES) + [ideal.STRAIGHT]


def choice_label(key: str) -> str:
    if key == ideal.STRAIGHT:
        return "Straight cone or V (fixed)"
    return f"Best-fitting {ideal.FAMILIES[key]['name'].lower()}"


def choice_note(key: str) -> str:
    if key == ideal.STRAIGHT:
        return "A fixed straight line from apex to foot on each half: the perfect cone or V."
    return ideal.FAMILIES[key]["note"]


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


# =====================================================================
# Turning half profiles back into real coordinates
# =====================================================================
def _real_halves(item: dict, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(distance, real elevation, ideal elevation) across both halves, left to right."""
    m = item["m"]
    kind = item["kind"]
    ideal_g = item["comparisons"][key]["ideal"]
    xs, zr, zi = [], [], []
    for h in item["halves"]:
        x = m["x_apex"] + h["sign"] * ideal.U * h["radius"]
        z_real = h["base"] + h["g"] * h["height"]
        z_ideal = h["base"] + ideal_g[h["side"]] * h["height"]
        if kind == ideal.VALLEY:
            z_real, z_ideal = -z_real, -z_ideal
        if h["sign"] < 0:
            x, z_real, z_ideal = x[::-1], z_real[::-1], z_ideal[::-1]
        xs.append(x)
        zr.append(z_real)
        zi.append(z_ideal)
    order = np.argsort([a[0] for a in xs])
    return (np.concatenate([xs[i] for i in order]), np.concatenate([zr[i] for i in order]),
            np.concatenate([zi[i] for i in order]))


def overlay_figure(item: dict, key: str, x_label: str, y_label: str):
    """The real profile with the perfect shape drawn on it, extra and missing rock shaded."""
    fig = go.Figure()
    x, zr, zi = _real_halves(item, key)
    pad = 0.15 * (item["m"]["x_right"] - item["m"]["x_left"])
    sel = (item["x"] >= item["m"]["x_left"] - pad) & (item["x"] <= item["m"]["x_right"] + pad)
    fig.add_trace(go.Scatter(
        x=item["x"][sel], y=item["z"][sel], mode="lines", name="data",
        line=dict(width=1, color=FAINT), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(x=x, y=np.maximum(zr, zi), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=zi, mode="lines", fill="tonexty", fillcolor=EXCESS_FILL,
                             line=dict(width=0), name="extra rock (above the ideal)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=np.minimum(zr, zi), mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=zi, mode="lines", fill="tonexty", fillcolor=DEFICIT_FILL,
                             line=dict(width=0), name="missing rock (below the ideal)", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=x, y=zi, mode="lines", name=f"ideal: {choice_label(key)}",
        line=dict(width=2, color=INK_2, dash="dash"),
        hovertemplate="ideal %{y:.4g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=zr, mode="lines", name="profile (as measured)",
        line=dict(width=3, color=INK),
        hovertemplate="profile %{y:.4g}<extra></extra>",
    ))
    m = item["m"]
    fig.add_vline(x=m["x_apex"], line=dict(color=MUTED, width=1, dash="dot"))
    words = ideal.VOCAB[item["kind"]]
    ends = f"the {words['foot']}" if item["kind"] == ideal.SLOPE else f"each {words['foot']}"
    _layout(fig, f"{choice_label(key)} pinned to the {words['apex']} and to {ends}", 470,
            hovermode="x unified")
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    return fig


def half_figure(item: dict, key: str):
    """Both halves in the normalised frame, with the residual underneath."""
    kind = item["kind"]
    words = ideal.VOCAB[kind]
    comp = item["comparisons"][key]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32],
                        vertical_spacing=0.06)
    flip = kind == ideal.VALLEY
    first = True
    for h in item["halves"]:
        s = h["sign"] if kind != ideal.SLOPE else 1.0
        u = s * ideal.U
        g_real = 1.0 - h["g"] if flip else h["g"]
        g_ideal = 1.0 - comp["ideal"][h["side"]] if flip else comp["ideal"][h["side"]]
        fig.add_trace(go.Scatter(x=u, y=g_ideal, mode="lines", name=f"ideal: {choice_label(key)}",
                                 line=dict(width=2, color=INK_2, dash="dash"), showlegend=first,
                                 hovertemplate="ideal %{y:.3f}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=u, y=g_real, mode="lines", name="profile", showlegend=first,
                                 line=dict(width=3, color=INK),
                                 hovertemplate="profile %{y:.3f}<extra></extra>"), row=1, col=1)
        r = comp["residual"][h["side"]] * 100.0
        fig.add_trace(go.Scatter(x=u, y=np.where(r > 0, r, 0.0), mode="lines", fill="tozeroy",
                                 fillcolor=EXCESS_FILL, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=u, y=np.where(r < 0, r, 0.0), mode="lines", fill="tozeroy",
                                 fillcolor=DEFICIT_FILL, line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=u, y=r, mode="lines", line=dict(width=1.5, color=INK),
                                 showlegend=False,
                                 hovertemplate="rock vs ideal %{y:+.1f}% of height<extra></extra>"),
                      row=2, col=1)
        first = False
    for edge in ideal.ZONE_EDGES:
        for s in ((-1.0, 1.0) if kind != ideal.SLOPE else (1.0,)):
            fig.add_vline(x=s * edge, line=dict(color=FAINT, width=1, dash="dot"))
    fig.add_hline(y=0.0, line=dict(color=MUTED, width=1), row=2, col=1)
    _layout(fig, "Half profiles, each on its own width", 520, hovermode="x unified")
    if kind == ideal.SLOPE:
        ticks = ([0, ideal.ZONE_EDGES[0], ideal.ZONE_EDGES[1], 1], [words["apex"], "", "", words["foot"]])
    else:
        ticks = ([-1, -ideal.ZONE_EDGES[1], -ideal.ZONE_EDGES[0], 0, ideal.ZONE_EDGES[0],
                  ideal.ZONE_EDGES[1], 1],
                 [words["foot"], "", "", words["apex"], "", "", words["foot"]])
    fig.update_xaxes(tickvals=ticks[0], ticktext=ticks[1], row=2, col=1)
    fig.update_yaxes(title="share of height", range=[-0.04, 1.08], row=1, col=1)
    fig.update_yaxes(title="rock vs ideal, % of height", row=2, col=1)
    return fig


def fingerprint_figure(item: dict, key: str):
    """How well each ideal family fits at its best, and the fixed straight line for scale."""
    names, vals, colours = [], [], []
    for k in CHOICE_KEYS:
        if k not in item["comparisons"]:
            continue
        names.append(choice_label(k).replace("Best-fitting ", "Best ").replace(" (fixed)", ""))
        vals.append(100.0 * item["comparisons"][k]["departure"])
        colours.append(HIGHLIGHT if k == item["family"] else BAR_MUTED)
    fig = go.Figure(go.Bar(
        x=vals, y=names, orientation="h", marker=dict(color=colours, line=dict(width=0)),
        text=[f"{v:.1f}%" for v in vals], textposition="outside",
        textfont=dict(size=12, color=INK),
        hovertemplate="<b>%{y}</b><br>departure %{x:.2f}% of height<extra></extra>",
        showlegend=False,
    ))
    _layout(fig, "How well each ideal fits", 320, bargap=0.35)
    fig.update_layout(margin=dict(l=20, r=60, t=60, b=20))
    fig.update_xaxes(title="departure, % of the box (0 = this profile IS that ideal)",
                     range=[0, max(vals) * 1.25 + 0.5])
    fig.update_yaxes(autorange="reversed")
    return fig


def perfect_shapes_figure(kind: str):
    """The landmark shapes along the form axis, drawn the way up this feature is."""
    keys = SHAPE_ORDER
    fig = make_subplots(rows=1, cols=len(keys), horizontal_spacing=0.02,
                        subplot_titles=[ideal.PERFECT_SHAPES[k]["name"] for k in keys])
    u = np.concatenate([-ideal.U[::-1], ideal.U[1:]])
    for i, k in enumerate(keys, start=1):
        g = ideal.PERFECT_SHAPES[k]["g"](ideal.U)
        g2 = np.concatenate([g[::-1], g[1:]])
        if kind == ideal.SLOPE:
            xx, yy = ideal.U, g
        else:
            xx, yy = u, (1.0 - g2 if kind == ideal.VALLEY else g2)
        fig.add_trace(go.Scatter(x=xx, y=yy, mode="lines", line=dict(width=2.5, color=INK),
                                 fill="tozeroy" if kind != ideal.VALLEY else None,
                                 fillcolor="rgba(82,81,78,0.10)", showlegend=False,
                                 hoverinfo="skip"), row=1, col=i)
        fig.update_xaxes(visible=False, row=1, col=i)
        fig.update_yaxes(visible=False, range=[-0.05, 1.1], row=1, col=i)
    fig.update_layout(template="plotly_white", height=210, margin=dict(l=10, r=10, t=50, b=10))
    fig.update_annotations(font=dict(size=11, color=INK_2))
    return fig


def shape_chart_figure(rows: list, current: dict | None, y_col: str):
    """Powers' chart for landforms: form across, departure from the ideal up."""
    fig = go.Figure()
    y_all = [r.get(y_col, np.nan) for r in rows] + ([current.get(y_col, np.nan)] if current else [])
    y_all = [v for v in y_all if np.isfinite(v)]
    y_top = max([0.10] + [v * 1.15 for v in y_all]) * 100.0

    # Departure classes as faint bands, labelled at the right edge.
    lo = 0.0
    shades = ["rgba(27,175,122,0.07)", "rgba(237,161,0,0.06)", "rgba(235,104,52,0.06)",
              "rgba(208,59,59,0.06)"]
    if y_col == "Departure":
        for (limit, label), shade in zip(ideal.DEPARTURE_CLASSES, shades):
            hi = min(limit * 100.0, y_top)
            if lo >= y_top:
                break
            fig.add_hrect(y0=lo, y1=hi, fillcolor=shade, line_width=0, layer="below")
            fig.add_annotation(x=1.0, xref="x", y=0.5 * (lo + hi), text=label, showarrow=False,
                               xanchor="right", font=dict(size=11, color=INK_2))
            lo = hi

    # The perfect shapes sit on the floor of the chart: departure 0 by definition.
    for k in SHAPE_ORDER:
        f = ideal.fullness(ideal.PERFECT_SHAPES[k]["g"](ideal.U))
        name = ideal.PERFECT_SHAPES[k]["name"]
        fig.add_vline(x=f, line=dict(color=FAINT, width=1, dash="dot"), layer="below")
        fig.add_trace(go.Scatter(
            x=[f], y=[0.0], mode="markers+text", text=[name], textposition="bottom center",
            textfont=dict(size=11, color=INK_2), cliponaxis=False,
            marker=dict(symbol="diamond", size=11, color=INK, line=dict(width=2, color="#fcfcfb")),
            showlegend=False, hovertemplate=f"<b>{name}</b><br>perfect shape<extra></extra>",
        ))

    groups = []
    for r in rows:
        g = r.get("Group") or ""
        if g not in groups:
            groups.append(g)
    seen = set()
    for r in rows:
        y = r.get(y_col, np.nan)
        if not np.isfinite(y):
            continue
        feat = r.get("Feature", "hill")
        g = r.get("Group") or ""
        sym = GROUP_SYMBOLS[groups.index(g) % len(GROUP_SYMBOLS)]
        legend_name = feat + (f" - {g}" if g else "")
        fig.add_trace(go.Scatter(
            x=[r["Fullness"]], y=[y * 100.0], mode="markers+text",
            text=[str(r.get("#", ""))], textposition="middle right",
            textfont=dict(size=11, color=INK),
            marker=dict(symbol=sym, size=11, color=FEATURE_COLOURS.get(feat, MUTED),
                        line=dict(width=2, color="#fcfcfb")),
            name=legend_name, legendgroup=legend_name, showlegend=legend_name not in seen,
            customdata=[[r.get("Label", ""), r.get("Shape", ""), r.get("Grade", "")]],
            hovertemplate=("<b>#%{text} %{customdata[0]}</b><br>%{customdata[1]}<br>"
                           "fullness %{x:.3f}<br>" + y_col.lower() + " %{y:.2f}% of the box"
                           "<br>%{customdata[2]}<extra></extra>"),
        ))
        seen.add(legend_name)

    if current and np.isfinite(current.get(y_col, np.nan)):
        fig.add_trace(go.Scatter(
            x=[current["Fullness"]], y=[current[y_col] * 100.0], mode="markers",
            marker=dict(symbol="diamond-open", size=16, color=INK, line=dict(width=2.5)),
            name="profile loaded now",
            hovertemplate=(f"<b>profile loaded now</b><br>{current.get('Shape', '')}<br>"
                           "fullness %{x:.3f}<br>%{y:.2f}% of height<extra></extra>"),
        ))

    ylab = {"Departure": "departure from the best-fitting ideal"}.get(y_col, "departure from a straight cone or V")
    _layout(fig, "Shape chart", 560)
    fig.update_layout(margin=dict(l=20, r=20, t=60, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0))
    fig.update_xaxes(title="fullness: share of its box the half profile fills (sagging → full)",
                     range=[0.0, 1.0])
    # Room under the floor for the perfect shapes' names, clear of the points.
    fig.update_yaxes(title=f"{ylab}, % of the box", range=[-0.16 * y_top, y_top],
                     tickvals=[v for v in np.arange(0.0, y_top + 1e-9, 2.0 if y_top <= 20 else 5.0)])
    return fig


# =====================================================================
# Layout
# =====================================================================
def _pct(v: float, signed: bool = False) -> str:
    if not np.isfinite(v):
        return "-"
    return f"{v * 100:+.1f}%" if signed else f"{v * 100:.1f}%"


def to_display_units(item: dict, sx: float, sz: float) -> dict:
    """Positions and elevations back in the file's own units, after a fit done in common units.

    Only the drawing changes: every ratio and departure is dimensionless.
    """
    if not item.get("ok") or (sx == 1.0 and sz == 1.0):
        return item
    out = dict(item)
    out["x"] = item["x"] / sx
    out["z"] = item["z"] / sz
    m = dict(item["m"])
    for k in ("x_left", "x_right", "x_apex", "width", "x_flat_left", "x_flat_right"):
        m[k] = m[k] / sx
    for k in ("base_elev", "apex_elev", "height", "base_left", "base_right"):
        m[k] = m[k] / sz
    out["m"] = m
    out["halves"] = [dict(h, radius=h["radius"] / sx, base=h["base"] / sz, height=h["height"] / sz)
                     for h in item["halves"]]
    return out


def default_feature(dataset_type: str, volcano_type: str, long_type: str) -> str:
    if dataset_type == volcano_type:
        return "Hill or mountain"
    if dataset_type == long_type:
        return "One slope (long profile)"
    return "Valley"


def render_explainer():
    with st.expander("How the fit works - the fixed procedure", expanded=False):
        st.markdown(
            """
**The idea comes from grain shape.** Wadell fitted a sand grain with a perfect sphere and measured
the material left over; how far a grain departs from the sphere, and where, records how it has
been worn. Here the grain is a landform and the sphere is the best-fitting ideal curve.

Every profile goes through the same five steps:

1. **Locate.** Find the apex (the summit of a hill, the floor of a valley, the head of a long
   profile) and the foot of each side (the base of a hill, each rim of a valley, the mouth).
   Both feet sit at one level, the higher of the two sides' bases, so a volcano on a regional
   slope is read down to its own foot, not out across the slope. A valley with one rim lower
   than the other is read up to the lower rim.
2. **Half profiles.** Split at the apex. Each half is scaled to its own width, so a lopsided
   feature keeps its shape instead of being averaged.
3. **Fit four ideal curves to each half.** Each is pinned to the apex and the foot and has one
   free control, so they compete on equal terms:
   - **Parabola** (control *k*): every true parabola from the concave one with its vertex at the
     foot (*k* = 1) through the straight line (*k* = 0) to the dome or U with its vertex at the
     apex (*k* = -1).
   - **Exponential** (control *a*): steep at the apex and still sloping at the foot when *a* > 0,
     like a graded river or the flank of a stratovolcano; straight at *a* = 0, convex below it.
   - **Circle in true proportions** (control: bulge): a real circular arc, drawn without
     vertical exaggeration - Wadell's sphere.
   - **Superellipse** (control *e*): (1 - u^e)^(1/e), for shapes fuller or more sagging than any
     parabola - the strongly concave cone (e about 0.5), the semicircle (e = 2), the box.

   The best member of each family is the one that leaves the least rock out of place.
4. **Choose.** The family that leaves the least rock out of place is the ideal. On a near tie
   (within 0.1% of the box) the order above wins, parabola first.
5. **Measure.** Extra rock (blue) and missing rock (red) between the profile and that ideal, as a
   share of the feature's box (width times height). **Departure** is the two added together.
   The sign is about rock on a hill and in a valley alike.

**The name** comes from the fitted ideal's fullness - the share of its box it fills: 1/6 strongly
concave, 1/3 concave parabola, 5/12 gently concave, 1/2 straight, 7/12 gently convex, 2/3
parabola, 0.79 semicircle, 0.93 box - with a flat top or floor added when one is measured. A
profile is called straight only when its best parabola has |k| under 0.25.

The departure grades (near-ideal under 1%, slightly modified under 2.5%, modified under 5%,
strongly modified above) are provisional: they were set from the sample profiles that ship with
the app, not from a calibrated population.
            """
        )


def render_ideal_tab(item: dict, x_label: str, y_label: str, source_name: str):
    if not item.get("ok"):
        st.warning(item.get("problem") or "This profile could not be fitted.")
        render_explainer()
        return

    kind = item["kind"]
    words = ideal.VOCAB[kind]
    fam = item["family"]
    st.markdown(f"### {item['name'][0].upper() + item['name'][1:]}")
    st.caption(
        f"Best-fitting ideal: **{ideal.fit_words(item)}**. Departure {_pct(item['departure'])} "
        f"of the box, **{item['grade']}**."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Departure", _pct(item["departure"]),
              help="Extra plus missing rock between the profile and the best-fitting ideal, as a "
                   "share of the feature's width times its height. 0 means the profile IS that ideal.")
    c2.metric("Net rock", _pct(item["comparisons"][fam]["net"], signed=True),
              help="Extra minus missing rock against the best-fitting ideal. Positive means more "
                   "rock than the ideal has.")
    c3.metric("Fullness", f"{item['fullness']:.3f}",
              help="Share of its box the profile fills: 0.5 for a cone or V, 0.667 for a parabola.")
    c4.metric("Departure from straight", _pct(item["comparisons"][ideal.STRAIGHT]["departure"]),
              help="Against a fixed straight cone or V, for scale: how curved the profile is.")

    pick = st.selectbox(
        "Compare with", CHOICE_KEYS, index=CHOICE_KEYS.index(fam), format_func=choice_label,
        key="ideal_compare_with",
        help="Defaults to the best-fitting ideal. Pick another to see how it fits.",
    )
    comp = item["comparisons"][pick]
    st.plotly_chart(overlay_figure(item, pick, x_label, y_label), width="stretch")
    z1, z2, z3, z4 = st.columns(4)
    z1.metric("Extra rock", _pct(comp["excess"]))
    z2.metric("Missing rock", _pct(comp["deficit"]))
    z3.metric("Departure", _pct(comp["departure"]))
    z4.metric("Net", _pct(comp["net"], signed=True))
    st.markdown(ideal.zone_words(comp))
    st.caption(choice_note(pick))

    left, right = st.columns([1.35, 1.0], gap="large")
    with left:
        st.plotly_chart(half_figure(item, pick), width="stretch")
    with right:
        st.plotly_chart(fingerprint_figure(item, pick), width="stretch")
        rows = []
        side_name = {"L": "left", "R": "right", "D": "downslope"}
        spec = ideal.FAMILIES[fam]
        for h in item["halves"]:
            d = item["by_side"][h["side"]]
            v = item["comparisons"][fam]["params"][h["side"]]
            rows.append({
                "Half": side_name[h["side"]],
                "Width": _num(h["radius"]),
                "Height": _num(h["height"]),
                f"{spec['name'].split(' ')[0]} {spec['param']}": f"{v:+.2f}" if fam != "superellipse" else f"{v:.2f}",
                "Departure": _pct(d["departure"]),
                "Fullness": f"{d['fullness']:.3f}",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        m = item["m"]
        facts = [f"{words['feature'].capitalize()} {m['width_height']:.1f} times as wide as it is high."]
        if kind != ideal.SLOPE:
            facts.append(f"Asymmetry {m['asymmetry']:.3f} (0 = both halves hold the same area).")
            if item.get("flat"):
                facts.append(f"A {words['flat']} across {m['plateau']:.0%} of the width.")
        st.caption(" ".join(facts))

    for n in item.get("notes", []):
        st.info(n)

    st.markdown("**Landmark shapes along the form axis**")
    st.plotly_chart(perfect_shapes_figure(kind), width="stretch")
    st.caption(
        "Drawn the way up this feature is, each scaled to the same box. They name the fitted ideal "
        "by its fullness and sit on the floor of the shape chart."
    )
    render_explainer()


def _num(v: float) -> str:
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.4g}"


def _add_rows(new_rows: list):
    rows = list(st.session_state.get("ideal_chart_rows") or [])
    for r in new_rows:
        if r:
            r = dict(r)
            r["#"] = len(rows) + 1
            rows.append(r)
    st.session_state["ideal_chart_rows"] = rows


def measure_samples(sample_files: dict, sample_kind, samples_dir: Path) -> list:
    out = []
    for label, fname in sample_files.items():
        try:
            d = pd.read_csv(samples_dir / fname)
        except Exception:
            continue
        kind = sample_kind(label)
        it = ideal.measure(d.iloc[:, 0], d.iloc[:, 1], kind)
        row = ideal.summary_row(it, label.split(": ", 1)[-1], label.split(":", 1)[0])
        if row:
            out.append(row)
    return out


def render_shape_chart(item: dict, source_name: str, sample_files: dict, sample_kind,
                       samples_dir: Path):
    st.markdown(
        "Profiles organised by their departure from the ideal: across, how full the form is, from "
        "sagging to box-like; up, how much rock is out of place against the best-fitting ideal. The "
        "landmark shapes sit on the floor of the chart. This is Powers' roundness chart turned on "
        "landforms."
    )
    current = ideal.summary_row(item, "profile loaded now") if item.get("ok") else None
    rows = st.session_state.get("ideal_chart_rows") or []

    view = st.radio("Up the chart", list(CHART_Y), horizontal=True, key="ideal_chart_y")
    st.plotly_chart(shape_chart_figure(rows, current, CHART_Y[view]), width="stretch")
    st.caption(
        "Colour is the feature type and marker shape the group you give a profile. Numbers key the "
        "points to the table below. The shaded bands are the provisional departure grades."
    )

    a1, a2, a3, a4, a5 = st.columns([1.4, 1.1, 0.9, 1.0, 0.7], gap="medium")
    default_label = str(source_name or "").rsplit(".", 1)[0] or f"profile {len(rows) + 1}"
    with a1:
        label = st.text_input("Label", value=default_label, key="ideal_pin_label")
    with a2:
        group = st.text_input("Group (optional)", key="ideal_pin_group",
                              help="Profiles sharing a group get the same marker shape, e.g. glacial, "
                                   "fluvial, Mars.")
    with a3:
        if st.button("Add this profile", width="stretch", disabled=current is None):
            _add_rows([ideal.summary_row(item, label or default_label, group)])
            st.rerun()
    with a4:
        if st.button("Add all samples", width="stretch",
                     help="Fit every sample profile that ships with the app and put it on the chart."):
            have = {(r.get("Label"), r.get("Group")) for r in rows}
            new = [r for r in measure_samples(sample_files, sample_kind, samples_dir)
                   if (r["Label"], r["Group"]) not in have]
            _add_rows(new)
            st.rerun()
    with a5:
        if st.button("Clear", width="stretch", disabled=not rows):
            st.session_state["ideal_chart_rows"] = []
            st.rerun()

    if rows:
        df = pd.DataFrame(rows)
        show_cols = ["#", "Label", "Group", "Feature", "Shape", "Best-fitting ideal", "Departure",
                     "Grade", "Net rock", "Fullness"]
        show = df[[c for c in show_cols if c in df.columns]].copy()
        show["Departure"] = show["Departure"].map(lambda v: _pct(v))
        show["Net rock"] = show["Net rock"].map(lambda v: _pct(v, signed=True))
        show["Fullness"] = show["Fullness"].map(lambda v: f"{v:.3f}")
        st.dataframe(show, width="stretch", hide_index=True)
        st.download_button(
            "Shape chart CSV (all columns)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="spectrumearth_shape_chart.csv",
            mime="text/csv",
        )
    else:
        st.caption("Add the profile loaded now, or press Add all samples to see the sample profiles "
                   "organised on the chart.")
