"""
The library view of Spectrum Earth: every real transect, already fitted, shown together.

The fits come from samples/library/fits.json, written by tools/fit_library.py
with the same single procedure the rest of the app uses (ideal_shape.py), so
the view opens at once. By default it shows the whole library: every landform's
transects and their average departure, then every transect's fit. A group
(volcanoes, valleys or ridges), one landform, or one transect can be picked
instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import ideal_shape as ideal
import ideal_shape_view as ideal_view

GROUPS = {"volcano": "Volcanoes", "valley": "Valleys", "ridge": "Mid-ocean ridges"}
GROUP_COLOURS = {"volcano": "#2a78d6", "valley": "#eb6834", "ridge": "#1baf7a"}
ALL = "All"


@st.cache_data(show_spinner=False)
def load_fits(path: str) -> list:
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return []


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%" if np.isfinite(v) else "-"


def summary(library: dict, fits: list) -> pd.DataFrame:
    """One row per landform: its group, transects fitted and their average departure."""
    by_id = {}
    for r in fits:
        by_id.setdefault(r["feature_id"], []).append(r)
    rows = []
    for label, f in library.items():
        rs = [r for r in by_id.get(f["id"], []) if r.get("ok")]
        d = np.array([r["departure"] for r in rs], dtype=float)
        rows.append({
            "label": label, "id": f["id"], "Landform": f["name"], "group": f["feature"],
            "Group": GROUPS[f["feature"]], "Transects": len(d),
            "Average departure": float(d.mean()) if len(d) else np.nan,
            "Spread (sd)": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "Lowest": float(d.min()) if len(d) else np.nan,
            "Highest": float(d.max()) if len(d) else np.nan,
            "Grade": ideal.departure_class(float(d.mean())) if len(d) else "",
            "departures": d.tolist(), "columns": [r["column"] for r in rs],
        })
    return pd.DataFrame(rows)


def overview_figure(table: pd.DataFrame):
    """Every landform, lowest average departure at the top: each transect a dot, the average a bar."""
    t = table.dropna(subset=["Average departure"]).sort_values("Average departure")
    fig = go.Figure()
    top = max([0.06] + [max(d) * 1.08 for d in t["departures"] if d]) * 100.0
    lo = 0.0
    shades = ["rgba(27,175,122,0.07)", "rgba(237,161,0,0.06)", "rgba(235,104,52,0.06)",
              "rgba(208,59,59,0.06)"]
    for i, ((limit, words), shade) in enumerate(zip(ideal.DEPARTURE_CLASSES, shades)):
        if lo >= top:
            break
        hi = min(limit * 100.0, top)
        fig.add_vrect(x0=lo, x1=hi, fillcolor=shade, line_width=0, layer="below",
                      annotation_text=words, annotation_position="top left" if i % 2 == 0 else "bottom left",
                      annotation_font=dict(size=11, color=ideal_view.INK_2))
        lo = hi
    seen = set()
    for r in t.to_dict("records"):
        c = GROUP_COLOURS[r["group"]]
        avg, sd, name = r["Average departure"], r["Spread (sd)"], r["Landform"]
        fig.add_trace(go.Scatter(
            x=[v * 100 for v in r["departures"]], y=[name] * len(r["departures"]), mode="markers",
            marker=dict(size=7, color=c, opacity=0.45), legendgroup=r["group"], showlegend=False,
            customdata=r["columns"], hovertemplate=f"<b>{name}</b> %{{customdata}}: %{{x:.1f}}%<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[avg * 100], y=[name], mode="markers", name=r["Group"],
            marker=dict(size=13, color=c, symbol="line-ns", line=dict(width=3, color=c)),
            error_x=dict(type="data", array=[sd * 100], color=c, thickness=1.5, width=0),
            legendgroup=r["group"], showlegend=r["group"] not in seen,
            hovertemplate=(f"<b>{name}</b><br>average {avg * 100:.1f}% ± {sd * 100:.1f}% "
                           f"over {r['Transects']} transects<extra></extra>"),
        ))
        seen.add(r["group"])
    ideal_view._layout(fig, "Every landform by its departure from the ideal (dots: transects; bar: average ± sd)",
                       160 + 24 * max(len(t), 3))
    fig.update_layout(margin=dict(l=20, r=20, t=70, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0))
    fig.update_xaxes(title="departure, % of the box (0 = the landform is its ideal)", range=[0, top])
    fig.update_yaxes(autorange="reversed", categoryorder="array", categoryarray=list(t["Landform"]))
    return fig


def _fit_traces(r: dict, legend: bool):
    """The real profile, its ideal, and the extra and missing rock between them, in km and m."""
    x = np.asarray(r["x"]) / 1000.0
    zr, zi = np.asarray(r["real"]), np.asarray(r["ideal"])
    return [
        go.Scatter(x=np.asarray(r["profile_x"]) / 1000.0, y=r["profile_z"], mode="lines",
                   line=dict(width=1, color=ideal_view.FAINT), name="data beyond the feet",
                   hoverinfo="skip", showlegend=legend, legendgroup="beyond"),
        go.Scatter(x=x, y=np.maximum(zr, zi), mode="lines", line=dict(width=0), showlegend=False,
                   hoverinfo="skip"),
        go.Scatter(x=x, y=zi, mode="lines", fill="tonexty", fillcolor=ideal_view.EXCESS_FILL,
                   line=dict(width=0), name="extra rock (above the ideal)", hoverinfo="skip",
                   showlegend=legend, legendgroup="extra"),
        go.Scatter(x=x, y=np.minimum(zr, zi), mode="lines", line=dict(width=0), showlegend=False,
                   hoverinfo="skip"),
        go.Scatter(x=x, y=zi, mode="lines", fill="tonexty", fillcolor=ideal_view.DEFICIT_FILL,
                   line=dict(width=0), name="missing rock (below the ideal)", hoverinfo="skip",
                   showlegend=legend, legendgroup="missing"),
        go.Scatter(x=x, y=zi, mode="lines", line=dict(width=1.5, color=ideal_view.INK_2, dash="dash"),
                   name="ideal", showlegend=legend, legendgroup="ideal",
                   hovertemplate="ideal %{y:.0f} m<extra></extra>"),
        go.Scatter(x=x, y=zr, mode="lines", line=dict(width=2, color=ideal_view.INK), name="profile",
                   showlegend=legend, legendgroup="profile",
                   hovertemplate="%{x:.2f} km, %{y:.0f} m<extra></extra>"),
    ]


def grid_figure(rows: list, names: dict, cols: int = 4):
    """Every transect's fit in a grid, titled with its landform, transect and departure."""
    rows = [r for r in rows if r.get("ok")]
    n = max(len(rows), 1)
    nrows = int(np.ceil(n / cols))
    titles = [f"{names[r['feature_id']]}, {r['column']}: {_pct(r['departure'])}" for r in rows]
    fig = make_subplots(rows=nrows, cols=cols, subplot_titles=titles, horizontal_spacing=0.04,
                        vertical_spacing=min(0.25 / max(nrows - 1, 1), 0.08))
    for i, r in enumerate(rows):
        rr, cc = i // cols + 1, i % cols + 1
        for tr in _fit_traces(r, legend=(i == 0)):
            fig.add_trace(tr, row=rr, col=cc)
        if r.get("axis") is not None:
            fig.add_vline(x=r["axis"] / 1000.0, line=dict(color=ideal_view.HIGHLIGHT, width=1, dash="dot"),
                          row=rr, col=cc)
    fig.update_annotations(font_size=11)
    fig.update_layout(template="plotly_white", height=60 + 210 * nrows, margin=dict(l=10, r=10, t=50, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0 + 30 / (60 + 210 * nrows),
                                  xanchor="left", x=0))
    fig.update_xaxes(gridcolor=ideal_view.GRID, zeroline=False, tickfont_size=9)
    fig.update_yaxes(gridcolor=ideal_view.GRID, zeroline=False, tickfont_size=9)
    return fig


def single_figure(r: dict, title: str):
    fig = go.Figure(_fit_traces(r, legend=True))
    if r.get("axis") is not None:
        fig.add_vline(x=r["axis"] / 1000.0, line=dict(color=ideal_view.HIGHLIGHT, width=1.5, dash="dash"),
                      annotation_text="axis", annotation_position="top")
    ideal_view._layout(fig, title, 470, hovermode="x unified")
    fig.update_xaxes(title="distance, km")
    fig.update_yaxes(title="elevation, m")
    return fig


def _summary_table(t: pd.DataFrame):
    show = t.sort_values("Average departure")[["Landform", "Group", "Transects", "Average departure",
                                                "Spread (sd)", "Lowest", "Highest", "Grade"]].copy()
    for c in ("Average departure", "Lowest", "Highest"):
        show[c] = show[c].map(_pct)
    show["Spread (sd)"] = show["Spread (sd)"].map(lambda v: f"± {v * 100:.1f}%")
    st.dataframe(show, width="stretch", hide_index=True)


def render_library(library: dict, library_dir: Path):
    fits = load_fits(str(library_dir / "fits.json"))
    if not library or not fits:
        st.warning("The transect library could not be read.")
        return
    table = summary(library, fits)
    names = {f["id"]: f["name"] for f in library.values()}

    st.markdown(
        f"**{len(table)} real landforms, {sum(len(d) for d in table['departures'])} transects**, each "
        "fitted by the one procedure: the ideal curve that leaves the least rock out of place, and "
        "that rock as a share of the landform's box (its departure). Land is Copernicus DEM GLO-30, "
        "the sea floor ETOPO1; pick a landform to see its sources and transect positions."
    )
    c1, c2, c3 = st.columns([1.2, 1.6, 1.2], gap="medium")
    with c1:
        group = st.radio("Show", [ALL] + list(GROUPS.values()), horizontal=False, key="lib_group")
    in_group = table if group == ALL else table[table["Group"] == group]
    with c2:
        options = [ALL] + list(in_group.sort_values("Average departure")["Landform"])
        landform = st.selectbox("Landform", options, key="lib_landform",
                                help="All shows every landform in the chosen group.")
    one = None if landform == ALL else in_group[in_group["Landform"] == landform].iloc[0]
    transect = ALL
    with c3:
        if one is not None:
            transect = st.selectbox("Transect", [ALL] + one["columns"], key="lib_transect")

    rows = [r for r in fits if r["feature_id"] in set(in_group["id"] if one is None else [one["id"]])]
    order = {fid: i for i, fid in enumerate(in_group.sort_values("Average departure")["id"])}
    rows.sort(key=lambda r: order.get(r["feature_id"], 0))

    if one is None:
        g = [(name, in_group[in_group["Group"] == name]["Average departure"]) for name in GROUPS.values()
             if (in_group["Group"] == name).any()]
        cols = st.columns(len(g))
        for col, (name, vals) in zip(cols, g):
            col.metric(name, f"{vals.mean() * 100:.1f}%",
                       help=f"Mean of the {len(vals)} landforms' average departures; range "
                            f"{_pct(vals.min())} to {_pct(vals.max())}.")
        st.plotly_chart(overview_figure(in_group), width="stretch")
        st.caption("The shaded bands are the provisional grades. Hover a dot for its transect.")
        with st.expander("The numbers, landform by landform", expanded=False):
            _summary_table(in_group)
        st.markdown(f"#### Every transect ({len(rows)}), with its ideal")
        st.caption("Blue is extra rock above the ideal, red is missing rock below it, the grey line is "
                   "the ground beyond the feet, and on a ridge the dotted line is the spreading axis. "
                   "Distance in km, elevation in m.")
        st.plotly_chart(grid_figure(rows, names), width="stretch")
        return

    entry = library[one["label"]]
    a1, a2, a3 = st.columns(3)
    a1.metric("Average departure", _pct(one["Average departure"]))
    a2.metric("Spread", f"± {one['Spread (sd)'] * 100:.1f}%")
    a3.metric("Transects", f"{one['Transects']}", help=f"Range {_pct(one['Lowest'])} to {_pct(one['Highest'])}.")
    st.caption(ideal_view.library_citation(entry))
    if transect == ALL:
        st.plotly_chart(grid_figure(rows, names, cols=2), width="stretch")
        return
    r = next(r for r in rows if r["column"] == transect)
    st.plotly_chart(single_figure(r, f"{entry['name']}, {transect}: {_pct(r['departure'])}"), width="stretch")
    st.caption(f"{r['name'][0].upper() + r['name'][1:]}. Ideal: {r['ideal_words']}. Extra rock "
               f"{_pct(r['extra'])}, missing rock {_pct(r['missing'])}; grade {r['grade']}."
               + (f" Spreading axis: {r['axis_type']}, read as {r['rate_words']}." if r.get("axis_type") else ""))
