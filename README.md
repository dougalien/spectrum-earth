# Spectrum Earth

Ideal-shape analysis of topographic profiles by Dougalien Systems.

The idea comes from grain shape: Wadell fitted a sand grain with a perfect sphere and measured the material left over, and how far a grain departs from the sphere records how it has been worn. Spectrum Earth does the same for landforms. Any profile, whether it crosses a valley, runs over a hill or mountain, or follows a river down its long profile, is fitted with perfect shapes. The app then measures how far the profile departs from them.

Every landform, of every type and size, goes through one fixed procedure, with no other rules:

1. **Frame (measured).** Find the top (the summit of a hill, the floor of a valley read upside down, the head of a long profile, the axis of a spreading ridge) and the foot on each side. Both feet sit at one base level, the higher of the two sides' bases (the rule Spectrum Jax validated on 60 real transects), so a volcano on a regional slope is read down to its own foot, not out across the slope, and a valley with one rim lower than the other is read up to the lower rim. Each side, from the top to its foot, goes in a box as wide as that side and as tall as the landform. The real profile and the ideal are compared in that box, so a 200 m valley and a 5 km volcano are compared at the same scale. Nothing assumes the two sides are alike.
2. **Ideal (fitted).** Down each side runs one smooth curve, held level across the top for as far as fits best (a flat top, or a flat floor). The curve is one of three, each with one bend control:
   - **Parabola** `g = (1 - u)(1 - k u)`: from the concave parabola with its vertex at the foot (k = 1) through the straight line (k = 0) to the dome or U with its vertex at the top (k = -1).
   - **Exponential** `g = (e^(-a u) - e^(-a)) / (1 - e^(-a))`: a > 0 is concave, steep at the top and still sloping at the foot, like a graded river long profile or the flank of a stratovolcano; a = 0 straight; a < 0 convex.
   - **Superellipse** `g = (1 - u^e)^(1/e)`: deeply concave (e < 1), straight (e = 1), the quarter circle (e = 2), toward a box (e > 2).

   Here u runs from the top (0) to the foot (1) and g is the height as a share of the ideal's top. The ideal's top height (shared by both sides, anywhere up to the real top), and each side's level stretch, curve and bend, are all chosen together to leave the least rock out of place: a grid search gives every pairing of curves a starting point, and each pairing is refined (Nelder-Mead within the same limits).
3. **Departure (the number).** The rock out of place, extra rock above the ideal plus missing rock below it, as a share of the box. 0 means the landform is its ideal; the further it departs, the bigger the number. It is exactly the quantity the fit makes as small as it can.
4. **Several transects** of one landform are each put through steps 1 to 3 and their departures averaged, with the spread. Load them under *Several transects of one landform* in the Ideal Shape tab: one file per transect, or one distance column followed by one elevation column per transect. `samples/transects_real_fuji_four_azimuths.csv` holds four real transects through the summit of Mount Fuji (azimuths 0°, 45°, 90° and 135°, AWS Terrain Tiles at zoom 12, 100 m steps); they average 1.1% ± 0.5%.

Earlier versions bent this procedure with special rules: thresholds deciding whether a top was flat, a split-summit rule that read a caldera from each rim, each side pinned at its own highest point, and a circle drawn in true proportions. They are gone. The circle went because its shape in the box changes with the map units and the vertical scale, so it could not be compared from one landform to the next; in the box, the superellipse with e = 2 is the quarter circle.

**The name** puts the ideal into words: its sides are concave (bowed in), straight or convex (bowed out), named after the landmark shape nearest the fitted curves, and "with a flat top" (or floor) is added when the level stretch covers a tenth of the width or more. The name never changes the fit or the number.

**Shape Chart.** Landforms ranked by their departure, lowest first. Add your own profiles one at a time, the average of several transects, or press *Add all samples*.

The departure grades (near-ideal under 1%, slightly modified under 2.5%, modified under 5%, strongly modified above) are provisional. They were set from the sample profiles, not from a calibrated population.

The fitting engine is `ideal_shape.py`. It has no Streamlit dependency and is ported from Spectrum Jax's validated volcano engine. `tests/` checks that each curve recovers its own shapes, that the departure does not change with the size of a landform, that the fitted ideal leaves the least rock out of place, and that every sample profile measures. The earlier analyses (peaks, FFT, landform screening, the power-law shape register) are still available under **Other analyses**.

## Mid-ocean ridges

Choose **Spreading ridge** under "What is the feature?" for a cross-section of a spreading centre, with the axis at distance 0 (or in the middle of the profile). The same fixed procedure reads it, with two things particular to a ridge:

- **The frame** is set from the axis rather than from the highest point, so both flanks are read even where a rift mountain stands higher than the axis. The fit and the departure are then exactly as for any other landform; a rift valley cut into the crest is missing rock. The sea floor deepens with the square root of its age away from the axis, so this regional ramp is the ridge itself, symmetric about the axis, and is fitted rather than removed.
- **The axis** (the ground within 40 km of it) is framed as a hill and as a valley. Whichever puts its top on the axis says what the axis is: an **axial valley** or an **axial high**.

An axial valley reads as a slow or ultraslow ridge (under about 40 mm/yr). An axial high reads as intermediate (about 50 to 80 mm/yr) when the ground fills 0.40 or more of its box, a low straight-sided to domed rise, and as fast (over about 80 mm/yr) when it is a peaked concave rise. On 27 profiles across the 9 sample ridges (three per ridge, 20 km apart along the axis), all 27 were classed right. A southern Mid-Atlantic Ridge profile (25.6°S) was tried first and dropped: in ETOPO1 its axis shifts between parallel lines and an off-axis trough cuts its flank. The 0.40 limit was set on these same profiles, so it is provisional. Removing a fitted square-root-of-age ramp before reading the axis blurred the intermediate and fast ridges together, so the profile is read as it is.

The nine "Real ridge" samples are swath medians (7 parallel lines over 30 km) of ETOPO1 1 arc-minute bathymetry (NOAA), read from the public AWS Terrain Tiles. `tools/ridge_profile.py` makes more. At about 2 km resolution the narrow summit of a fast ridge is smoothed; GMRT multibeam grids would sharpen it.

| Sample | Axis point (lon, lat) | Azimuth | Full rate |
|---|---|---|---|
| SW Indian Ridge 10°E | 9.743, -52.979 | 145° | ~14 mm/yr |
| Mohns Ridge 72°N | 1.744, 72.410 | 145° | ~16 mm/yr |
| Mid-Atlantic Ridge 26°N | -44.850, 26.130 | 105° | ~25 mm/yr |
| Mid-Atlantic Ridge 29°N | -43.160, 29.168 | 100° | ~24 mm/yr |
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
