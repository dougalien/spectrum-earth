"""Free AI calls through Ollama Cloud, with a fallback chain.

Settings come from Streamlit secrets or environment variables:
    OLLAMA_API_KEY  key from https://ollama.com/settings/keys (required for AI)
    OLLAMA_MODELS   comma-separated cloud models to try in order
                    (default: gpt-oss:120b, gpt-oss:20b, gemma3:4b)

chat() tries each model in turn. When a model is over its usage limit, needs a
paid plan, or errors, the next one is tried. If every model fails (or there is
no key), chat() returns None and the app shows its built-in non-AI fallback.
"""
import os
import time
from typing import List, Optional

import requests

OLLAMA_CHAT_URL = "https://ollama.com/api/chat"
DEFAULT_MODELS = ["gpt-oss:120b", "gpt-oss:20b", "gemma3:4b"]
COOL_DOWN_SECONDS = 600  # skip a model for 10 minutes after it hits a limit

_cooling: dict = {}  # model -> time it may be tried again
last_error: str = ""
last_model: str = ""  # the cloud model that gave the most recent reply


def _setting(name: str) -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        import streamlit as st
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


def models() -> List[str]:
    raw = _setting("OLLAMA_MODELS")
    return [m.strip() for m in raw.split(",") if m.strip()] or DEFAULT_MODELS


def available() -> bool:
    """True when an Ollama key is set, so AI features can be offered."""
    return bool(_setting("OLLAMA_API_KEY"))


def chat(messages: List[dict], temperature: float = 0.4, timeout: int = 60) -> Optional[str]:
    """Send chat messages ({'role', 'content'}) and return the reply text, or None if no model could answer."""
    global last_error, last_model
    key = _setting("OLLAMA_API_KEY")
    if not key:
        last_error = "no OLLAMA_API_KEY set"
        return None
    now = time.time()
    for model in models():
        if _cooling.get(model, 0) > now:
            continue
        try:
            response = requests.post(
                OLLAMA_CHAT_URL,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": model, "messages": messages, "stream": False,
                      "options": {"temperature": temperature}},
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = f"{model}: {type(exc).__name__}"
            continue
        if response.status_code in (402, 403, 429) or response.status_code >= 500:
            _cooling[model] = now + COOL_DOWN_SECONDS
            last_error = f"{model}: HTTP {response.status_code}"
            continue
        if not response.ok:
            last_error = f"{model}: HTTP {response.status_code}"
            continue
        try:
            text = (response.json().get("message") or {}).get("content", "").strip()
        except ValueError:
            text = ""
        if text:
            last_model = model
            return text
        last_error = f"{model}: empty reply"
    return None
