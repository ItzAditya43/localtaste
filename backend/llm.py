"""Minimal Groq chat client with an on-disk cache and token accounting.

Two models are used so research and writing draw on separate rate limits:
GROQ_RESEARCH_MODEL plans the tool calls, GROQ_MODEL writes the brief.
"""
import hashlib
import json
import os
import time
from pathlib import Path

import requests

import qloo  # noqa: F401  (loads .env)

URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
RESEARCH_MODEL = os.environ.get("GROQ_RESEARCH_MODEL", "qwen/qwen3.8-27b")
CACHE = Path(__file__).resolve().parent / ".cache" / "llm"
usage = {"calls": 0, "tokens": 0, "waited": 0.0, "cached": 0}


def chat(messages, tools=None, as_json=False, temperature=0.4, model=None, retries=10):
    """Send a conversation and return the assistant message, waiting out rate limits.

    Identical requests are answered from disk; set LLM_CACHE=0 to always call the API.
    """
    model = model or MODEL
    body = {"model": model, "messages": messages, "temperature": temperature}
    if tools:
        body["tools"] = tools
    if as_json:
        body["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    use_cache = os.environ.get("LLM_CACHE", "1") != "0"
    hit = CACHE / (hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest() + ".json")
    if use_cache and hit.exists():
        usage["cached"] += 1
        return json.loads(hit.read_text())
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
            wait = float(r.headers.get("retry-after", 5)) + 1
            usage["waited"] += wait
            time.sleep(wait)
            continue
        if r.status_code == 400 and "failed_generation" in r.text:
            last = r.text[:200]
            continue
        r.raise_for_status()
        data = r.json()
        usage["calls"] += 1
        usage["tokens"] += data.get("usage", {}).get("total_tokens", 0)
        msg = data["choices"][0]["message"]
        msg.pop("reasoning", None)
        if use_cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            hit.write_text(json.dumps(msg))
        return msg
    raise RuntimeError(f"Groq request failed after {retries} attempts: {last}")


def chat_json(system, user, temperature=0.4, model=None):
    """Send one chat turn and return the reply parsed as JSON."""
    msg = chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
               as_json=True, temperature=temperature, model=model)
    return json.loads(msg["content"])
