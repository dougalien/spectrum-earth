# Spectrum Earth

Ideal-shape analysis of topographic profiles by Dougalien Systems.

The idea comes from grain shape: Wadell fitted a sand grain with a perfect sphere and measured the material left over, and how far a grain departs from the sphere records how it has been worn. Spectrum Earth does the same for landforms. Any profile, whether it crosses a valley, runs over a hill or mountain, or follows a river down its long profile, is fitted with perfect shapes. The app then measures how far the profile departs from them.

Every profile goes through the same fixed procedure:

1. **Locate** the apex (summit, valley floor or head of a long profile) and the foot of each side. Both feet sit at one level, the higher of the two sides' bases (the rule Spectrum Jax validated on 60 real transects), so a volcano on a regional slope is read down to its own foot, not out across the slope. A valley with one rim lower than the other is read up to the lower rim.
2. **Half profiles.** Split at the apex. Each half is scaled to its own width and pinned at the apex and the foot, so a lopsided feature keeps its shape instead of being averaged. A valley is measured as an upside-down hill, and a long profile is one half on its own. A flat top or floor is measured, not fitted: the ideal is flat out to its edge and the curve is fitted to the wall beyond.
3. **Fit four ideal curves to each half**, each with one free control, so they compete on equal terms:
   - **Parabola** `g = (1 - u)(1 - k u)`: every true parabola, from the concave one with its vertex at the foot (k = 1) through the straight line (k = 0) to the dome or U with its vertex at the apex (k = -1).
   - **Exponential** `g = (e^(-a u) - e^(-a)) / (1 - e^(-a))`: a = 0 is straight; a > 0 is concave, steep at the apex and still sloping at the foot, like a graded river long profile or the flank of a stratovolcano; a < 0 is convex.
   - **Circle in true proportions**: a real circular arc through apex and foot, drawn without vertical exaggeration (Wadell's sphere). Its control is the bulge of the arc.
   - **Superellipse** `g = (1 - u^e)^(1/e)`: shapes more sagging or fuller than any parabola, such as the strongly concave cone (e about 0.5), the semicircle (e = 2) and the box.

   The best member of each family is the one that leaves the least rock out of place, found by a fine grid and a bounded refinement.
4. **Choose** the family with the smallest total departure. On a near tie (within 0.1% of the box) the order above wins, parabola first.
5. **Measure.** **Departure** is the extra plus the missing rock between the profile and that ideal, as a share of the feature's width times its height. A fixed straight cone or V is kept alongside for scale.

**The name** comes from the fitted ideal's fullness (the share of its box it fills): strongly concave, concave parabola, gently concave, straight, gently convex, parabola, semicircle, box. A profile is called straight only when its best parabola has |k| under 0.25.

**Shape Chart.** Profiles are organised by form across (fullness) and by departure up, like Powers' roundness chart. Add your own profiles one at a time, or press *Add all samples*.

The departure grades (near-ideal under 1%, slightly modified under 2.5%, modified under 5%, strongly modified above) are provisional. They were set from the sample profiles, not from a calibrated population.

The fitting engine is `ideal_shape.py`. It has no Streamlit dependency and is ported from Spectrum Jax's validated volcano engine. `tests/` checks that each family recovers its own shapes (and that a curved profile is never left with a straight fit). The earlier analyses (peaks, FFT, landform screening, the power-law shape register) are still available under **Other analyses**.

## Mid-ocean ridges

Choose **Spreading ridge** under "What is the feature?" for a cross-section of a spreading centre, with the axis at distance 0 (or in the middle of the profile). The same fixed procedure reads it twice:

- **The whole ridge** is fitted as a hill. The sea floor deepens with the square root of its age away from the axis, so this regional ramp is the ridge itself, symmetric about the axis, and is fitted rather than removed. Its departure from the cooling curve `g = 1 - sqrt(u)` is shown alongside.
- **The axis** (the ground within 40 km of it) is fitted as a hill and as a valley. Whichever reading puts its apex on the axis says what the axis is: an **axial valley** or an **axial high**.

An axial valley reads as a slow or ultraslow ridge (under about 40 mm/yr). An axial high reads as intermediate (about 50 to 80 mm/yr) when it is full, a low straight-sided to domed rise (fullness 0.40 or more), and as fast (over about 80 mm/yr) when it is a peaked concave rise. On 27 profiles across the 9 sample ridges (three per ridge, 20 km apart along the axis), 25 were classed right; the two misses were both on the southern Mid-Atlantic Ridge. The fullness limit was set on these same profiles, so it is provisional. Removing a fitted square-root-of-age ramp before reading the axis blurred the intermediate and fast ridges together, so the profile is read as it is.

The nine "Real ridge" samples are swath medians (7 parallel lines over 30 km, 15 km for the southern Mid-Atlantic Ridge) of ETOPO1 1 arc-minute bathymetry (NOAA), read from the public AWS Terrain Tiles. `tools/ridge_profile.py` makes more. At about 2 km resolution the narrow summit of a fast ridge is smoothed; GMRT multibeam grids would sharpen it.

| Sample | Axis point (lon, lat) | Azimuth | Full rate |
|---|---|---|---|
| SW Indian Ridge 10°E | 9.743, -52.979 | 145° | ~14 mm/yr |
| Mohns Ridge 72°N | 1.744, 72.410 | 145° | ~16 mm/yr |
| Mid-Atlantic Ridge 26°N | -44.850, 26.130 | 105° | ~25 mm/yr |
| Mid-Atlantic Ridge 25.6°S | -13.750, -25.635 | 90° | ~34 mm/yr |
| Juan de Fuca 45°N | -130.200, 45.300 | 110° | ~56 mm/yr |
| SE Indian Ridge 98°E | 97.633, -47.490 | 39° | ~70 mm/yr |
| East Pacific Rise 12.5°N | -103.904, 12.504 | 85° | ~90 mm/yr |
| East Pacific Rise 9.5°N | -104.244, 9.500 | 90° | ~110 mm/yr |
| East Pacific Rise 17.5°S | -113.189, -17.519 | 100° | ~145 mm/yr |

XRD, GC and SEM/EDS analysis now lives in TraceCode.

Load your own x-y file, or pick one of the sample profiles in `samples/` from the "Or try a sample profile" menu. The menu includes five real volcano transects from Spectrum Jax and nine real mid-ocean ridge cross-sections, and choosing one sets the matching Dataset Type and feature.

## Run it on your own computer

```
py -m pip install -r requirements.txt
py -m streamlit run spectrumEarth.py
```

Run the checks with `py -m pip install pytest` and then `py -m pytest tests`.

## Optional AI help

The AI assistant uses free Ollama Cloud models. On Streamlit Community Cloud, add your key under the app's **Settings → Secrets**:

```
OLLAMA_API_KEY = "your key from https://ollama.com/settings/keys"
```

On your own computer it tries a local Ollama first. If no model can answer, the app gives a built-in summary of the analysis instead, so it always works. The core analysis never needs AI.
