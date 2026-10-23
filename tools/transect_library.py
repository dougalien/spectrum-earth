"""Build the real transect library in samples/library/ from samples/library/features.csv.

Every transect is real elevation or depth read from a public global dataset:

- Land (valleys and volcanoes): Copernicus DEM GLO-30, 1 arc-second (about 30 m),
  the public copy on AWS (s3://copernicus-dem-30m). Sea is 0 m in it.
- Sea floor (spreading ridges): ETOPO1, 1 arc-minute, as composited in the AWS
  Terrain Tiles (tools/ridge_profile.py), each sample the median of 7 parallel
  lines across 30 km, exactly as the earlier ridge samples were made.

How each transect is placed (the measured centre of every transect goes in
samples/library/transects.csv):

- Volcano: through the highest ground within 1.5 km of the listed summit (the
  listed point for a caldera), along each listed azimuth.
- Valley: three places along the floor, each with the azimuth square to the
  valley there, picked by hand on a map of the same DEM and listed in
  samples/library/valley_points.csv. Each transect is centred on the lowest
  ground within a few hundred metres of its listed point along that azimuth.
- Ridge: across the listed azimuth through the listed axis point and through
  points one spacing either way along the axis, as in the ridge samples.

    py -m pip install rasterio numpy pandas pillow
    py tools/transect_library.py            # all features
    py tools/transect_library.py fuji etna  # some of them

Needs internet access.
"""
import math, os, sys, tempfile, urllib.error, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ridge_profile as rp  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIB = ROOT / "samples" / "library"
CACHE = Path(tempfile.gettempdir()) / "spectrum_earth_cop30"
COP_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/{n}/{n}.tif")
N_POINTS = 401
_tiles = {}


def _tile(ilon: int, ilat: int):
    """(elevation array, rasterio transform) of the 1° Copernicus tile whose SW corner is given."""
    key = (ilon, ilat)
    if key in _tiles:
        return _tiles[key]
    ns = f"{'N' if ilat >= 0 else 'S'}{abs(ilat):02d}_00"
    ew = f"{'E' if ilon >= 0 else 'W'}{abs(ilon):03d}_00"
    name = f"Copernicus_DSM_COG_10_{ns}_{ew}_DEM"
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{name}.tif"
    if not p.exists():
        try:
            with urllib.request.urlopen(COP_URL.format(n=name), timeout=300) as r:
                p.write_bytes(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):     # no tile: open sea
                _tiles[key] = None
                return None
            raise
    with rasterio.open(p) as ds:
        _tiles[key] = (ds.read(1).astype(float), ds.transform)
    return _tiles[key]


def cop_elev(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Copernicus GLO-30 elevation (m) at each point, bilinear; 0 where there is no tile (sea)."""
    lon, lat = np.asarray(lon, float), np.asarray(lat, float)
    out = np.zeros_like(lon)
    ilon, ilat = np.floor(lon).astype(int), np.floor(lat).astype(int)
    for a, b in set(zip(ilon.tolist(), ilat.tolist())):
        sel = (ilon == a) & (ilat == b)
        t = _tile(a, b)
        if t is None:
            continue
        arr, tr = t
        col = (lon[sel] - tr.c) / tr.a - 0.5
        row = (lat[sel] - tr.f) / tr.e - 0.5
        c0 = np.clip(np.floor(col).astype(int), 0, arr.shape[1] - 2)
        r0 = np.clip(np.floor(row).astype(int), 0, arr.shape[0] - 2)
        fc, fr = np.clip(col - c0, 0, 1), np.clip(row - r0, 0, 1)
        out[sel] = ((1 - fc) * (1 - fr) * arr[r0, c0] + fc * (1 - fr) * arr[r0, c0 + 1]
                    + (1 - fc) * fr * arr[r0 + 1, c0] + fc * fr * arr[r0 + 1, c0 + 1])
    return out


def line(lon0, lat0, az, d_m):
    """Points at signed distances d_m (m) from (lon0, lat0) along azimuth az."""
    pts = [rp.dest(lon0, lat0, az if d >= 0 else az + 180, abs(d)) for d in d_m]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def land_profile(lon0, lat0, az, half_km):
    d = np.linspace(-half_km, half_km, N_POINTS) * 1000.0
    lo, la = line(lon0, lat0, az, d)
    return d, cop_elev(lo, la)


def summit(lon0, lat0, radius_km=1.5):
    """The highest DEM point within radius_km of (lon0, lat0)."""
    best = (cop_elev([lon0], [lat0])[0], lon0, lat0)
    for az in range(0, 360, 10):
        d = np.linspace(0, radius_km, 31)[1:] * 1000.0
        lo, la = line(lon0, lat0, az, d)
        z = cop_elev(lo, la)
        k = int(np.argmax(z))
        if z[k] > best[0]:
            best = (z[k], lo[k], la[k])
    return best[1], best[2]


def lowest_across(lon0, lat0, across, search_km):
    """The lowest ground within search_km of the point along the azimuth `across`."""
    d = np.linspace(-search_km, search_km, 81) * 1000.0
    lo, la = line(lon0, lat0, across, d)
    k = int(np.argmin(cop_elev(lo, la)))
    return lo[k], la[k]


def build(f) -> tuple[pd.DataFrame, list]:
    rows, cols = [], {}
    if f.feature == "volcano":
        lon0, lat0 = (f.lon, f.lat) if f.id == "craterlake" else summit(f.lon, f.lat)
        for az in [float(a) for a in str(f.azimuths).split(";")]:
            d, z = land_profile(lon0, lat0, az, f.half_km)
            name = f"az{int(az):03d}"
            cols[name] = z
            rows.append(dict(column=name, centre_lon=round(lon0, 5), centre_lat=round(lat0, 5),
                             azimuth_deg=az, half_km=f.half_km, step_m=round(d[1] - d[0], 1),
                             dataset="Copernicus DEM GLO-30"))
    elif f.feature == "valley":
        plan = pd.read_csv(LIB / "valley_points.csv")
        for p in plan[plan.feature_id == f.id].itertuples():
            across = float(p.azimuth_deg)
            clon, clat = lowest_across(p.lon, p.lat, across, min(1.0, f.half_km / 6))
            d, z = land_profile(clon, clat, across, f.half_km)
            name = f"{p.label}_az{int(across):03d}"
            cols[name] = z
            rows.append(dict(column=name, centre_lon=round(clon, 5), centre_lat=round(clat, 5),
                             azimuth_deg=across, half_km=f.half_km, step_m=round(d[1] - d[0], 1),
                             dataset="Copernicus DEM GLO-30"))
    else:
        az = float(f.azimuths)
        for off in (-f.spacing_km, 0.0, f.spacing_km):
            plon, plat = rp.dest(f.lon, f.lat, az - 90 if off < 0 else az + 90, abs(off) * 1000.0) \
                if off else (f.lon, f.lat)
            d, z = rp.profile(plon, plat, az, f.half_km, 1.0, swath_km=30.0, nswath=7)
            name = f"along{off:+.0f}km" if off else "along0km"
            cols[name] = z
            rows.append(dict(column=name, centre_lon=round(plon, 4), centre_lat=round(plat, 4),
                             azimuth_deg=az, half_km=f.half_km, step_m=1000.0,
                             dataset="ETOPO1 (AWS Terrain Tiles), median of 7 lines over 30 km"))
    df = pd.DataFrame({"distance_m": np.round(d, 1), **{k: np.round(v, 1) for k, v in cols.items()}})
    return df, [dict(feature_id=f.id, **r) for r in rows]


def main(ids):
    feats = pd.read_csv(LIB / "features.csv")
    if ids:
        feats = feats[feats.id.isin(ids)]
    tpath = LIB / "transects.csv"
    meta = pd.read_csv(tpath) if tpath.exists() else pd.DataFrame()
    for f in feats.itertuples():
        df, rows = build(f)
        df.to_csv(LIB / f"{f.id}.csv", index=False)
        if len(meta):
            meta = meta[meta.feature_id != f.id]
        meta = pd.concat([meta, pd.DataFrame(rows)], ignore_index=True)
        print(f.id, len(rows), "transects")
    order = {k: i for i, k in enumerate(pd.read_csv(LIB / "features.csv").id)}
    meta = meta.sort_values(["feature_id"], key=lambda s: s.map(order), kind="stable")
    meta.to_csv(tpath, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
