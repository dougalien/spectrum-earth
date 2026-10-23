"""
The library view of Spectrum Earth: every real transect, already fitted, shown together.

The fits come from samples/library/fits.json, written by tools/fit_library.py
with the same single procedure the rest of the app uses (ideal_shape.py), so
the view opens at once. By default it shows the whole library: every landform's
transects and their average departure, then every transect's fit. A group
(volcanoes, valleys or ridges), one landform, or one transect can be picked
instead. Every transect is stored against both ideals, its best fit and the
fixed triangle, and the view shows the one chosen at the top of the app.
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


def read_against(fits: list, against: str) -> list:
    """The library rows read against the chosen ideal: as stored for the best fit, or with the
    fixed triangle's line and numbers in place of the best fit's."""
    if against != ideal.TRIANGLE:
        return fits
    out = []
    for r in fits:
        if r.get("ok"):
            r = {**r, "departure": r["tri_departure"], "extra": r["tri_extra"], "missing": r["tri_missing"],
                 "ideal": r["triangle"], "grade": "", "against": ideal.TRIANGLE}
        out.append(r)
    return out


def _pct(v: float) -> str:
    return f"{v * 100:.1f}%" if np.isfinite(v) else "-"


def summary(library: dict, fits: list, against: str = ideal_view.BEST) -> pd.DataFrame:
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
            "Grade": ideal_view.average_grade(float(d.mean()), against) if len(d) else "",
            "departures": d.tolist(), "columns": [r["column"] for r in rs],
        })
    return pd.DataFrame(rows)


def overview_figure(table: pd.DataFrame, against: str = ideal_view.BEST):
    """Every landform, lowest average departure at the top: each transect a dot, the average a bar."""
    t = table.dropna(subset=["Average departure"]).sort_values("Average departure")
    fig = go.Figure()
    floor = ideal.TRIANGLE_PARABOLA * 1.15 if against == ideal.TRIANGLE else 0.06
    top = max([floor] + [max(d) * 1.08 for d in t["departures"] if d]) * 100.0
    ideal_view.departure_bands(fig, top, against)
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
    what = "the fixed triangle" if against == ideal.TRIANGLE else "the ideal"
    ideal_view._layout(fig, f"Every landform by its departure from {what} (dots: transects; bar: average ± sd)",
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
    name = ideal_view.ideal_label(r.get("against", ideal_view.BEST))
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
                   name=name, showlegend=legend, legendgroup="ideal",
                   hovertemplate=name + " %{y:.0f} m<extra></extra>"),
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


def two_ideals(library: dict, fits: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every transect's departure from both ideals, and each landform's averages of the two."""
    names = {f["id"]: (f["name"], f["feature"]) for f in library.values()}
    t = pd.DataFrame([{"id": r["feature_id"], "column": r["column"], "best": r["departure"],
                       "triangle": r["tri_departure"]} for r in fits if r.get("ok") and r["feature_id"] in names])
    t["Landform"] = t["id"].map(lambda i: names[i][0])
    t["group"] = t["id"].map(lambda i: names[i][1])
    lf = t.groupby(["id", "Landform", "group"], as_index=False)[["best", "triangle"]].mean()
    lf["best rank"] = lf["best"].rank(method="min").astype(int)
    lf["triangle rank"] = lf["triangle"].rank(method="min").astype(int)
    return t, lf


SHORT_NAMES = (("Mid-Atlantic Ridge", "MAR"), ("East Pacific Rise", "EPR"), ("SW Indian Ridge", "SWIR"),
               ("SE Indian Ridge", "SEIR"), ("Juan de Fuca Ridge", "Juan de Fuca"), ("Mohns Ridge", "Mohns"),
               ("Black Canyon of the Gunnison", "Black Canyon"), ("Piton de la Fournaise", "Fournaise"),
               ("Mount ", ""), (" Valley", ""))


def short_name(name: str) -> str:
    """A landform's name short enough to label a dot: 'Mid-Atlantic Ridge 23°N' becomes 'MAR 23°N'."""
    name = name.split(" (")[0]
    for long, short in SHORT_NAMES:
        name = name.replace(long, short)
    return name


# Where a label can sit beside its dot, as plotly names it and as a box offset in
# pixels from the dot (left, bottom), for a label w pixels wide and 12 high.
_PLACES = (("middle right", lambda w: (8, -6)), ("middle left", lambda w: (-8 - w, -6)),
           ("top center", lambda w: (-w / 2, 7)), ("bottom center", lambda w: (-w / 2, -19)),
           ("top right", lambda w: (4, 4)), ("bottom right", lambda w: (4, -16)),
           ("top left", lambda w: (-4 - w, 4)), ("bottom left", lambda w: (-4 - w, -16)))


def label_places(x, y, labels, x_range, y_range, size=(1150.0, 520.0)) -> list:
    """A place beside each dot for its label, chosen so labels cover as little as they can of the
    dots and of each other: each label takes the first place that is clear, or the least covered."""
    sx, sy = size[0] / (x_range[1] - x_range[0]), size[1] / (y_range[1] - y_range[0])
    px, py = (np.asarray(x) - x_range[0]) * sx, (np.asarray(y) - y_range[0]) * sy
    taken = [(a - 6, b - 6, a + 6, b + 6) for a, b in zip(px, py)]

    def covered(box):
        return sum(max(0.0, min(box[2], t[2]) - max(box[0], t[0])) * max(0.0, min(box[3], t[3]) - max(box[1], t[1]))
                   for t in taken)

    out = [""] * len(labels)
    for i in np.argsort(px):
        w = 5.6 * len(labels[i])
        best = None
        for name, offset in _PLACES:
            dx, dy = offset(w)
            box = (px[i] + dx, py[i] + dy, px[i] + dx + w, py[i] + dy + 12)
            if box[0] < 0 or box[2] > size[0]:
                continue
            c = covered(box)
            if best is None or c < best[0]:
                best = (c, name, box)
            if c == 0:
                break
        out[i] = best[1]
        taken.append(best[2])
    return out


def rank_agreement(lf: pd.DataFrame) -> float:
    """How closely the two orders agree: 1 the same order, 0 unrelated (Spearman's rank correlation)."""
    if len(lf) < 3:
        return float("nan")
    return float(lf["best"].rank().corr(lf["triangle"].rank()))


def two_ideals_figure(t: pd.DataFrame, lf: pd.DataFrame):
    """Each landform's departure from the best-fit curve against its departure from the fixed triangle.

    How far below the line where the two are equal a landform sits is the rock
    that bending the sides, levelling the top and moving the summit account for.
    """
    hi = max([ideal.TRIANGLE_PARABOLA * 1.15] + list(t["triangle"] * 1.05)) * 100.0
    top = max([0.06] + list(t["best"] * 1.1)) * 100.0
    names = [short_name(n) for n in lf["Landform"]]
    places = dict(zip(lf["id"], label_places(lf["triangle"] * 100, lf["best"] * 100, names, (0, hi), (0, top))))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, hi], y=[0, hi], mode="lines", line=dict(color=ideal_view.MUTED, width=1, dash="dash"),
                             name="the two equal", hoverinfo="skip"))
    fig.add_vline(x=ideal.TRIANGLE_PARABOLA * 100.0, line=dict(color=ideal_view.INK_2, width=1, dash="dot"),
                  annotation_text=ideal_view.TRIANGLE_LINE, annotation_position="top",
                  annotation_font=dict(size=11, color=ideal_view.INK_2))
    for g, words in GROUPS.items():
        c = GROUP_COLOURS[g]
        tg, lg = t[t["group"] == g], lf[lf["group"] == g]
        if lg.empty:
            continue
        fig.add_trace(go.Scatter(
            x=tg["triangle"] * 100, y=tg["best"] * 100, mode="markers", legendgroup=g, showlegend=False,
            marker=dict(size=6, color=c, opacity=0.3), customdata=tg[["Landform", "column"]],
            hovertemplate="%{customdata[0]}, %{customdata[1]}<br>triangle %{x:.1f}%, best fit %{y:.1f}%<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=lg["triangle"] * 100, y=lg["best"] * 100, mode="markers+text", name=words, legendgroup=g,
            text=[short_name(n) for n in lg["Landform"]], textposition=[places[i] for i in lg["id"]],
            textfont=dict(size=10, color=ideal_view.INK_2),
            marker=dict(size=11, color=c, line=dict(width=1.5, color="#fcfcfb")),
            customdata=lg[["Landform", "triangle rank", "best rank"]],
            hovertemplate=("<b>%{customdata[0]}</b><br>fixed triangle %{x:.1f}% (rank %{customdata[1]})"
                           "<br>best-fit curve %{y:.1f}% (rank %{customdata[2]})<extra></extra>"),
        ))
    ideal_view._layout(fig, "Departure from the best-fit curve against departure from the fixed triangle", 660)
    fig.update_layout(margin=dict(l=20, r=20, t=70, b=40),
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0))
    fig.update_xaxes(title="departure from the fixed triangle, % of the box", range=[0, hi])
    fig.update_yaxes(title="departure from the best-fit curve, % of the box", range=[0, top])
    return fig


def render_two_ideals(library: dict, fits: list):
    """The panel that sets the two ideals side by side: where each landform falls on both."""
    t, lf = two_ideals(library, fits)
    if lf.empty:
        return
    st.markdown("#### The two ideals, plotted against each other")
    st.plotly_chart(two_ideals_figure(t, lf), width="stretch")
    rho = rank_agreement(lf)
    st.caption(
        "Each large dot is a landform's average, each faint dot one transect. Every transect in the "
        "library is closer to its best fit than to the triangle, so all sit below the dashed line where "
        "the two are equal, and how far below it a landform sits is the rock that bending the sides, "
        "levelling the top and moving the summit account for. Low and to the right: smooth but bowed or lopsided "
        "shapes that a curve fits closely and the triangle does not. Higher up: rough or irregular "
        "profiles that no smooth curve fits."
        + (f" The two orders agree only loosely (rank agreement {rho:.2f}, where 1 is the same order and "
           "0 unrelated), so the two numbers measure different things." if np.isfinite(rho) and rho < 0.7 else
           f" Rank agreement between the two orders: {rho:.2f} (1 is the same order, 0 unrelated)."
           if np.isfinite(rho) else "")
    )
    with st.expander("Each landform's place in both orders", expanded=False):
        show = lf.sort_values("triangle")[["Landform", "triangle", "triangle rank", "best", "best rank"]].copy()
        d = show["best rank"] - show["triangle rank"]
        show["On the best fit"] = [("same place" if v == 0 else f"{abs(v)} place{'s' if abs(v) > 1 else ''} "
                                    + ("further down" if v > 0 else "further up")) for v in d]
        show["triangle"] = show["triangle"].map(_pct)
        show["best"] = show["best"].map(_pct)
        show.columns = ["Landform", "Fixed triangle", "Place (triangle)", "Best-fit curve", "Place (best fit)",
                        "On the best fit"]
        st.dataframe(show, width="stretch", hide_index=True)


def _summary_table(t: pd.DataFrame, against: str = ideal_view.BEST):
    cols = ["Landform", "Group", "Transects", "Average departure", "Spread (sd)", "Lowest", "Highest"]
    show = t.sort_values("Average departure")[cols + ([] if against == ideal.TRIANGLE else ["Grade"])].copy()
    for c in ("Average departure", "Lowest", "Highest"):
        show[c] = show[c].map(_pct)
    show["Spread (sd)"] = show["Spread (sd)"].map(lambda v: f"± {v * 100:.1f}%")
    st.dataframe(show, width="stretch", hide_index=True)


def render_library(library: dict, library_dir: Path, against: str = ideal_view.BEST):
    stored = load_fits(str(library_dir / "fits.json"))
    fits = read_against(stored, against)
    if not library or not fits:
        st.warning("The transect library could not be read.")
        return
    tri = against == ideal.TRIANGLE
    table = summary(library, fits, against)
    names = {f["id"]: f["name"] for f in library.values()}

    n = f"**{len(table)} real landforms, {sum(len(d) for d in table['departures'])} transects**, each "
    if tri:
        st.markdown(
            n + "measured against the fixed triangle, which has nothing fitted: its base runs between the "
            "two feet, its apex stands at the full height midway between them, and its sides are "
            "straight. The rock out of place, as a share of the landform's box, is its departure. "
            "Land is Copernicus DEM GLO-30, the sea floor ETOPO1; pick a landform to see its sources "
            "and transect positions."
        )
    else:
        st.markdown(
            n + "fitted by the one procedure: the ideal curve that leaves the least rock out of place, "
            "and that rock as a share of the landform's box (its departure). Land is Copernicus DEM "
            "GLO-30, the sea floor ETOPO1; pick a landform to see its sources and transect positions."
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
        st.plotly_chart(overview_figure(in_group, against), width="stretch")
        st.caption(("The dotted line is where " + ideal_view.TRIANGLE_LINE + " centred between its "
                    "feet sits: the other end member. " if tri else
                    "The shaded bands are the provisional grades. ") + "Hover a dot for its transect.")
        with st.expander("The numbers, landform by landform", expanded=False):
            _summary_table(in_group, against)
        render_two_ideals({k: v for k, v in library.items() if v["id"] in set(in_group["id"])}, stored)
        st.markdown(f"#### Every transect ({len(rows)}), with " + ("the fixed triangle" if tri else "its ideal"))
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
    if tri:
        words = ("Against the fixed triangle, nothing fitted. The real "
                 + ("valley floor" if one["group"] == "valley" else "summit" if one["group"] == "volcano"
                    else "axis")
                 + f" is {ideal_view.offset_words(r['tri_offset'])}. Extra rock {_pct(r['extra'])}, "
                 f"missing rock {_pct(r['missing'])}.")
    else:
        words = (f"{r['name'][0].upper() + r['name'][1:]}. Ideal: {r['ideal_words']}. Extra rock "
                 f"{_pct(r['extra'])}, missing rock {_pct(r['missing'])}; grade {r['grade']}.")
    st.caption(words + (f" Spreading axis: {r['axis_type']}, read as {r['rate_words']}." if r.get("axis_type") else ""))
