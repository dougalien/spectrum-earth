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

XRD, GC and SEM/EDS analysis now lives in TraceCode.

Load your own x-y file, or pick one of the sample profiles in `samples/` from the "Or try a sample profile" menu. The menu includes five real volcano transects from Spectrum Jax, and choosing one sets the matching Dataset Type.

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
