"""Minimal Groq chat client returning parsed JSON. Swap the model with GROQ_MODEL."""
import json
import os
import time

import requests

import qloo  # noqa: F401  (loads .env)

URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def chat_json(system, user, temperature=0.4, retries=8):
    """Send one chat turn and return the reply parsed as JSON, waiting out rate limits."""
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "response_format": {"type": "json_object"},
        "temperature": temperature,
    }
    if "gpt-oss" in MODEL:
        body["reasoning_effort"] = "low"
    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    for attempt in range(retries):
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=30)
        except requests.Timeout:
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(float(r.headers.get("retry-after", 5)) + 1)
            continue
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])
    raise RuntimeError(f"Groq still rate limited after {retries} attempts")
