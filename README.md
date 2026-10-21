# Spectrum Earth

Ideal-shape analysis of topographic profiles by Dougalien Systems.

The idea comes from grain shape: Wadell fitted a sand grain with a perfect sphere and measured the material left over, and how far a grain departs from the sphere records how it has been worn. Spectrum Earth does the same for landforms. Any profile, whether it crosses a valley, runs over a hill or mountain, or follows a river down its long profile, is fitted with perfect shapes. The app then measures how far the profile departs from them.

- **Half profiles.** The profile is split at its apex (summit, valley floor or head), and each half is pinned at the apex and its own foot, so asymmetry is kept rather than averaged away. A valley is measured as an upside-down hill, and a long profile is one half on its own.
- **Perfect shapes**, from sagging to full: strongly concave, concave parabola, straight-sided (cone / V), parabola, the circle in true proportions (Wadell's sphere), semicircle and box.
- **Departure** is the extra plus the missing rock between the profile and the nearest perfect shape, as a share of the feature's width times its height. **Irregularity** is what is left after a smooth curve is fitted to each half. In grain work these are sphericity and roundness.
- **Shape Chart.** Profiles are organised by form across (fullness) and by departure up, like Powers' roundness chart. Add your own profiles one at a time, or press *Add all samples*.

The departure grades are provisional. They were set from the sample profiles, not from a calibrated population.

The fitting engine is `ideal_shape.py`. It has no Streamlit dependency and is ported from Spectrum Jax's validated volcano engine. `tests/` checks that it recovers every perfect shape. The earlier analyses (peaks, FFT, landform screening, the power-law shape register) are still available under **Other analyses**.

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
