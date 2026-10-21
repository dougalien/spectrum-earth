# Spectrum Earth

Landform and shape analysis by Dougalien Systems: measure topographic profiles (valley transects, river long profiles and volcano radial profiles), compare them with ideal shapes, and screen them against published landform classes. Peaks and FFT views are included for the profiles.

XRD, GC and SEM/EDS analysis now lives in TraceCode.

Load your own x-y file, or pick one of the sample profiles in `samples/` from the "Or try a sample profile" menu (choosing one sets the matching Dataset Type).

## Run it on your own computer

```
py -m pip install -r requirements.txt
py -m streamlit run spectrumEarth.py
```

## Optional AI help

The AI assistant uses free Ollama Cloud models. On Streamlit Community Cloud, add your key under the app's **Settings → Secrets**:

```
OLLAMA_API_KEY = "your key from https://ollama.com/settings/keys"
```

On your own computer it tries a local Ollama first. If no model can answer, the app gives a built-in summary of the analysis instead, so it always works. The core analysis never needs AI.
