"""Fit every transect in samples/library/ and write the results to samples/library/fits.json.

The app's library view draws from this file, so it opens at once instead of
fitting 99 transects (about a second each) on every start. Every number in it
comes from ideal_shape.py by the one fixed procedure; tests/ re-fits every
transect and checks the file still matches, so it cannot drift from the engine.

    py tools/fit_library.py
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ideal_shape as ideal  # noqa: E402

LIB = ROOT / "samples" / "library"
N_DRAW = 300


def fit(x, z, feature: str) -> tuple[dict, dict | None]:
    """(measurement, ridge reading or None), exactly as the app fits a library transect."""
    if feature == "ridge":
        r = ideal.read_ridge(x, z)
        return r["whole"], r
    return ideal.measure(x, z, ideal.HILL if feature == "volcano" else ideal.VALLEY), None


def drawn(item: dict) -> dict:
    """The real profile and its ideal across both halves, left to right, thinned for drawing."""
    x, zr, zi = ideal.drawn_halves(item)
    keep = np.unique(np.linspace(0, len(x) - 1, N_DRAW).round().astype(int))
    return {"x": x[keep].round(1).tolist(), "real": zr[keep].round(1).tolist(),
            "ideal": zi[keep].round(1).tolist()}


def main():
    feats = pd.read_csv(LIB / "features.csv")
    out = []
    for f in feats.itertuples():
        table = pd.read_csv(LIB / f"{f.id}.csv")
        x = table.iloc[:, 0].to_numpy(dtype=float)
        for col in table.columns[1:]:
            z = table[col].to_numpy(dtype=float)
            item, ridge = fit(x, z, f.feature)
            row = {"feature_id": f.id, "column": col, "ok": bool(item.get("ok"))}
            if row["ok"]:
                pad = 0.15 * (item["m"]["x_right"] - item["m"]["x_left"])
                sel = (x >= item["m"]["x_left"] - pad) & (x <= item["m"]["x_right"] + pad)
                row.update({
                    "departure": item["departure"], "extra": item["excess"], "missing": item["deficit"],
                    "grade": item["grade"], "name": item["name"], "ideal_words": ideal.fit_words(item),
                    "profile_x": x[sel].round(1).tolist(), "profile_z": z[sel].round(1).tolist(),
                    **drawn(item),
                })
            if ridge is not None:
                row.update({"axis": ridge["axis"], "axis_type": ridge["axis_type"],
                            "rate_words": ridge["rate_words"]})
            out.append(row)
            print(f.id, col, f"{row.get('departure', float('nan')) * 100:.2f}%")
    (LIB / "fits.json").write_text(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main()
