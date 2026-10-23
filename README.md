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
4. **Several transects** of one landform are each put through steps 1 to 3 and their departures averaged, with the spread. A landform picked from the library brings all its transects with it. Your own go under *Several transects of one landform* in the Ideal Shape tab: one file per transect, or one distance column followed by one elevation column per transect.

Earlier versions bent this procedure with special rules: thresholds deciding whether a top was flat, a split-summit rule that read a caldera from each rim, each side pinned at its own highest point, and a circle drawn in true proportions. They are gone. The circle went because its shape in the box changes with the map units and the vertical scale, so it could not be compared from one landform to the next; in the box, the superellipse with e = 2 is the quarter circle.

**The name** puts the ideal into words: its sides are concave (bowed in), straight or convex (bowed out), named after the landmark shape nearest the fitted curves, and "with a flat top" (or floor) is added when the level stretch covers a tenth of the width or more. The name never changes the fit or the number.

**Shape Chart.** Landforms ranked by their departure, lowest first. Add your own profiles one at a time, the average of several transects, or press *Add the library* to rank all 30 real landforms by the average of their transects.

The departure grades (near-ideal under 1%, slightly modified under 2.5%, modified under 5%, strongly modified above) are provisional. They were set from a few profiles before the real library was added, not from a calibrated population.

The fitting engine is `ideal_shape.py`. It has no Streamlit dependency and is ported from Spectrum Jax's validated volcano engine. `tests/` checks that each curve recovers its own shapes, that the departure does not change with the size of a landform, that the fitted ideal leaves the least rock out of place, that every transect in the library measures, and that every library ridge transect reads as its spreading rate. The earlier analyses (peaks, FFT, landform screening, the power-law shape register) are still available under **Other analyses**. Their old ideal overlay and conformity scores (a fixed cone/V, parabola and so on pinned to the section) are gone, so the single best fit is the only ideal the app draws.

## The real transect library

Every landform in the menu is real ground, read from a public global dataset: 10 valleys, 10 volcanoes and 10 mid-ocean ridges, 100 transects in all. There are no synthetic examples.

**Explore the library** is the view the app opens on. Every transect is already fitted, so the whole library is shown at once: every landform ranked by its average departure (each transect a dot, the average a bar with its spread), then every transect drawn with its ideal. Choose a group (volcanoes, valleys or mid-ocean ridges), one landform, or one transect to look closer. The fits are stored in `samples/library/fits.json` by `tools/fit_library.py`, and the tests re-fit every transect and check the file still matches. Run `py tools/fit_library.py` after changing the library or the fitting engine. **Fit a profile** loads your own file, or one library landform, with all the controls.

- **Land** (valleys and volcanoes): Copernicus DEM GLO-30, 1 arc-second (about 30 m), from the public copy on AWS (`s3://copernicus-dem-30m`). © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018, provided under COPERNICUS by the European Union and ESA; doi:10.5270/ESA-c5d3d65. It is a surface model, so forest and buildings are included, and lakes and rivers are read at their water surface.
- **Sea floor** (ridges): ETOPO1 1 arc-minute global relief model, Amante, C. and Eakins, B.W., 2009, NOAA Technical Memorandum NESDIS NGDC-24, doi:10.7289/V5C8276M, read from the AWS Terrain Tiles (`s3://elevation-tiles-prod`).
- **Spreading rates** are approximate full rates from MORVEL: DeMets, C., Gordon, R.G. and Argus, D.F., 2010, Geophysical Journal International 181, 1-80.

Each volcano has 4 transects through its summit (the highest ground within 1.5 km of the listed point; the listed point for Crater Lake) at 0°, 45°, 90° and 135°. Each valley has 3 transects at three places along its floor, square to the valley, placed by hand on a map of the same DEM (`samples/library/valley_points.csv`) and centred on the lowest ground nearby. Each ridge has 3. The exact centre and azimuth of every transect are in `samples/library/transects.csv`, and `tools/transect_library.py` rebuilds the whole library from `samples/library/features.csv`.

| Landform | Type | Place | Point (lat, lon) | Transects | Length | Data |
|---|---|---|---|---|---|---|
| Mount Fuji | volcano | Honshu, Japan | 35.3606, 138.7274 | 4 | 44 km | Copernicus GLO-30 |
| Mayon | volcano | Luzon, Philippines | 13.2548, 123.6853 | 4 | 22 km | Copernicus GLO-30 |
| Mount Taranaki | volcano | North Island, New Zealand | -39.2963, 174.0634 | 4 | 28 km | Copernicus GLO-30 |
| Shishaldin | volcano | Unimak Island, Alaska, USA | 54.7554, -163.9711 | 4 | 30 km | Copernicus GLO-30 |
| Mount Rainier | volcano | Washington, USA | 46.8528, -121.7603 | 4 | 40 km | Copernicus GLO-30 |
| Kilimanjaro (Kibo) | volcano | Tanzania | -3.0674, 37.3556 | 4 | 80 km | Copernicus GLO-30 |
| Mount Etna | volcano | Sicily, Italy | 37.7510, 14.9934 | 4 | 50 km | Copernicus GLO-30 |
| Mauna Kea | volcano | Hawaii, USA | 19.8207, -155.4681 | 4 | 80 km | Copernicus GLO-30 |
| Piton de la Fournaise | volcano | Reunion, France | -21.2444, 55.7083 | 4 | 28 km | Copernicus GLO-30 |
| Crater Lake (Mount Mazama) | volcano | Oregon, USA | 42.9446, -122.1090 | 4 | 32 km | Copernicus GLO-30 |
| Yosemite Valley | valley | California, USA | 37.7400, -119.6000 | 3 | 7 km | Copernicus GLO-30 |
| Lauterbrunnen Valley | valley | Bernese Oberland, Switzerland | 46.5800, 7.9100 | 3 | 6 km | Copernicus GLO-30 |
| Kings Canyon (Cedar Grove) | valley | California, USA | 36.7950, -118.6200 | 3 | 8 km | Copernicus GLO-30 |
| Nant Ffrancon | valley | Snowdonia, Wales, UK | 53.1500, -4.0450 | 3 | 5 km | Copernicus GLO-30 |
| Great Langdale | valley | Lake District, England, UK | 54.4450, -3.0900 | 3 | 5 km | Copernicus GLO-30 |
| Grand Canyon (Phantom Ranch) | valley | Arizona, USA | 36.1000, -112.0950 | 3 | 24 km | Copernicus GLO-30 |
| Black Canyon of the Gunnison | valley | Colorado, USA | 38.5700, -107.7200 | 3 | 5 km | Copernicus GLO-30 |
| Hells Canyon | valley | Idaho and Oregon, USA | 45.4000, -116.6000 | 3 | 24 km | Copernicus GLO-30 |
| Colca Canyon | valley | Arequipa, Peru | -15.6000, -71.9000 | 3 | 16 km | Copernicus GLO-30 |
| Rhine Gorge (Loreley) | valley | Rhineland-Palatinate, Germany | 50.1400, 7.7300 | 3 | 5 km | Copernicus GLO-30 |
| SW Indian Ridge 10°E (14 mm/yr) | ridge | Southern Ocean | -52.9790, 9.7430 | 3 | 400 km | ETOPO1 |
| Mohns Ridge 72°N (16 mm/yr) | ridge | Norwegian-Greenland Sea | 72.4100, 1.7440 | 3 | 400 km | ETOPO1 |
| Mid-Atlantic Ridge 23°N (25 mm/yr) | ridge | North Atlantic | 23.3420, -44.9020 | 3 | 400 km | ETOPO1 |
| Mid-Atlantic Ridge 26°N (25 mm/yr) | ridge | North Atlantic | 26.1300, -44.8500 | 3 | 400 km | ETOPO1 |
| Mid-Atlantic Ridge 29°N (24 mm/yr) | ridge | North Atlantic | 29.1680, -43.1600 | 3 | 400 km | ETOPO1 |
| Juan de Fuca Ridge 45°N (56 mm/yr) | ridge | NE Pacific | 45.3000, -130.2000 | 3 | 400 km | ETOPO1 |
| SE Indian Ridge 98°E (70 mm/yr) | ridge | Southern Ocean | -47.4900, 97.6330 | 3 | 400 km | ETOPO1 |
| East Pacific Rise 12.5°N (90 mm/yr) | ridge | Eastern Pacific | 12.5040, -103.9040 | 3 | 400 km | ETOPO1 |
| East Pacific Rise 9.5°N (110 mm/yr) | ridge | Eastern Pacific | 9.5000, -104.2440 | 3 | 400 km | ETOPO1 |
| East Pacific Rise 17.5°S (145 mm/yr) | ridge | South Pacific | -17.5190, -113.1890 | 3 | 400 km | ETOPO1 |

## Mid-ocean ridges

Choose **Spreading ridge** under "What is the feature?" for a cross-section of a spreading centre, with the axis at distance 0 (or in the middle of the profile). The same fixed procedure reads it, with two things particular to a ridge:

- **The frame** is set from the axis rather than from the highest point, so both flanks are read even where a rift mountain stands higher than the axis. The fit and the departure are then exactly as for any other landform; a rift valley cut into the crest is missing rock. The sea floor deepens with the square root of its age away from the axis, so this regional ramp is the ridge itself, symmetric about the axis, and is fitted rather than removed.
- **The axis** (the ground within 40 km of it) is framed as a hill and as a valley. Whichever puts its top on the axis says what the axis is: an **axial valley** or an **axial high**.

An axial valley reads as a slow or ultraslow ridge (under about 40 mm/yr). An axial high reads as intermediate (about 50 to 80 mm/yr) when the ground fills 0.40 or more of its box, a low straight-sided to domed rise, and as fast (over about 80 mm/yr) when it is a peaked concave rise. On the 27 profiles of the first 9 ridges (three per ridge, 20 km apart along the axis), all 27 were classed right, and so were the 3 of the Mid-Atlantic Ridge at 23°N added with the library. A southern Mid-Atlantic Ridge profile (25.6°S) was tried first and dropped: in ETOPO1 its axis shifts between parallel lines and an off-axis trough cuts its flank. The 0.40 limit was set on these same profiles, so it is provisional. Removing a fitted square-root-of-age ramp before reading the axis blurred the intermediate and fast ridges together, so the profile is read as it is.

The library ridges are swath medians (7 parallel lines over 30 km) of ETOPO1 1 arc-minute bathymetry, three transects per ridge 20 km apart along the axis. At about 2 km resolution the narrow summit of a fast ridge is smoothed; GMRT multibeam grids would sharpen it.

XRD, GC and SEM/EDS analysis now lives in TraceCode.

Load your own x-y file, or pick a real landform from the library menu. Choosing one sets the matching Dataset Type and feature and loads all its transects.

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
