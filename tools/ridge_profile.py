"""Make a ridge cross-section like the samples/sample_real_ridge_*.csv files.

Bathymetry is ETOPO1 (1 arc-minute, NOAA), read from the public AWS Terrain
Tiles (s3://elevation-tiles-prod, terrarium PNG). The profile runs along an
azimuth through a point on the spreading axis, and each sample is the median of
7 parallel lines spread across a swath, so a seamount on one line does not
become part of the ridge. Distance 0 is the axis.

    py -m pip install pillow numpy pandas
    py tools/ridge_profile.py -104.2444 9.5 90 out.csv            # EPR 9.5N
    py tools/ridge_profile.py LON LAT AZIMUTH OUT.csv --half 200 --swath 30

Needs internet access. GMRT (gmrt.org) has finer multibeam grids where they exist.
"""
import argparse, math, os, tempfile, urllib.request
import numpy as np
import pandas as pd
from PIL import Image

CACHE = os.path.join(tempfile.gettempdir(), "spectrum_earth_tiles")
URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
_mem = {}

def tile(z, x, y):
    key = (z, x % (2 ** z), y)
    if key in _mem:
        return _mem[key]
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{z}_{key[1]}_{y}.png")
    if not os.path.exists(p):
        with urllib.request.urlopen(URL.format(z=z, x=key[1], y=y), timeout=60) as r:
            data = r.read()
        with open(p, "wb") as f:
            f.write(data)
    a = np.asarray(Image.open(p).convert("RGB"), dtype=float)
    e = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
    _mem[key] = e
    return e

def px(lon, lat, z):
    n = 2 ** z * 256
    X = (lon + 180) / 360 * n
    lr = math.radians(lat)
    Y = (1 - math.log(math.tan(lr) + 1 / math.cos(lr)) / math.pi) / 2 * n
    return X, Y

def elev(lon, lat, z=8):
    X, Y = px(lon, lat, z)
    X -= 0.5; Y -= 0.5
    x0, y0 = math.floor(X), math.floor(Y)
    fx, fy = X - x0, Y - y0
    def v(xx, yy):
        return tile(z, xx // 256, yy // 256)[yy % 256, xx % 256]
    return ((1 - fx) * (1 - fy) * v(x0, y0) + fx * (1 - fy) * v(x0 + 1, y0)
            + (1 - fx) * fy * v(x0, y0 + 1) + fx * fy * v(x0 + 1, y0 + 1))

R = 6371000.0

def dest(lon, lat, az_deg, d_m):
    p1, l1, th, dl = math.radians(lat), math.radians(lon), math.radians(az_deg), d_m / R
    p2 = math.asin(math.sin(p1) * math.cos(dl) + math.cos(p1) * math.sin(dl) * math.cos(th))
    l2 = l1 + math.atan2(math.sin(th) * math.sin(dl) * math.cos(p1), math.cos(dl) - math.sin(p1) * math.sin(p2))
    return math.degrees(l2), math.degrees(p2)

def profile(lon0, lat0, az_deg, half_km, step_km=0.5, z=8, swath_km=0.0, nswath=1):
    """Profile centred on (lon0, lat0) along azimuth az; x in m, positive toward az.
    With swath_km > 0, the median of nswath parallel lines spread across that width."""
    d = np.arange(-half_km, half_km + 1e-9, step_km) * 1000.0
    offs = np.linspace(-swath_km / 2, swath_km / 2, nswath) * 1000.0 if nswath > 1 else [0.0]
    stack = []
    for o in offs:
        clon, clat = dest(lon0, lat0, az_deg + 90, o) if o else (lon0, lat0)
        row = []
        for di in d:
            lo, la = dest(clon, clat, az_deg if di >= 0 else az_deg + 180, abs(di))
            row.append(elev(lo, la, z))
        stack.append(row)
    return d, np.median(np.array(stack), axis=0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("lon", type=float); ap.add_argument("lat", type=float)
    ap.add_argument("azimuth", type=float, help="direction of the profile, degrees from north (across the axis)")
    ap.add_argument("out")
    ap.add_argument("--half", type=float, default=200.0, help="km either side of the axis")
    ap.add_argument("--step", type=float, default=1.0, help="km between samples")
    ap.add_argument("--swath", type=float, default=30.0, help="km across which the 7 lines are spread")
    a = ap.parse_args()
    d, z = profile(a.lon, a.lat, a.azimuth, a.half, a.step, swath_km=a.swath, nswath=7)
    pd.DataFrame({"distance_m": d.round(1), "elevation_m": z.round(1)}).to_csv(a.out, index=False)
