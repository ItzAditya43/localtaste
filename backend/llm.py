"""Minimal Groq chat client. Swap the model with GROQ_MODEL."""
import json
import os
import time

import requests

import qloo  # noqa: F401  (loads .env)

URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def chat(messages, tools=None, as_json=False, temperature=0.4, retries=10):
    """Send a conversation and return the assistant message, waiting out rate limits."""
    body = {"model": MODEL, "messages": messages, "temperature": temperature}
    if tools:
        body["tools"] = tools
    if as_json:
        body["response_format"] = {"type": "json_object"}
    if "gpt-oss" in MODEL:
        body["reasoning_effort"] = "low"
    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    last = ""
    for _ in range(retries):
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=45)
        except requests.Timeout:
            last = "timeout"
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = r.text[:200]
            time.sleep(float(r.headers.get("retry-after", 5)) + 1)
            continue
        if r.status_code == 400 and "failed_generation" in r.text:
            last = r.text[:200]
            continue
        r.raise_for_status()
        return r.json()["choices"][0]["message"]
    raise RuntimeError(f"Groq request failed after {retries} attempts: {last}")


def chat_json(system, user, temperature=0.4):
    """Send one chat turn and return the reply parsed as JSON."""
    msg = chat([{"role": "system", "content": system}, {"role": "user", "content": user}], as_json=True, temperature=temperature)
    return json.loads(msg["content"])
