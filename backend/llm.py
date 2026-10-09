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
# Each model has its own daily token allowance; when one runs out, the next takes over.
FALLBACKS = [m for m in os.environ.get("GROQ_FALLBACK_MODELS", "openai/gpt-oss-20b,openai/gpt-oss-120b,qwen/qwen3.8-27b").split(",") if m]
LONG_WAIT = 70  # a longer retry-after than this means a daily limit, not a per-minute one
_exhausted = {}  # model -> time it becomes usable again
CACHE = Path(__file__).resolve().parent / ".cache" / "llm"
usage = {"calls": 0, "tokens": 0, "waited": 0.0, "cached": 0}
MAX_WAIT = 120  # seconds one request may spend waiting on rate limits before giving up
on_wait = None  # optional callback(seconds), called before each rate-limit pause


class Busy(RuntimeError):
    """The model provider is rate limited or unavailable for longer than we are willing to wait."""


def chat(messages, tools=None, as_json=False, temperature=0.4, model=None):
    """Send a conversation and return the assistant message, with the model that answered under "_model".

    Waits out per-minute rate limits, and moves to a fallback model when one has used its daily allowance.
    Identical requests are answered from disk; set LLM_CACHE=0 to always call the API.
    """
    first = model or MODEL
    chain = [first] + [m for m in FALLBACKS if m != first]
    live = [m for m in chain if _exhausted.get(m, 0) < time.time()] or chain
    last = ""
    for m in live:
        try:
            return _ask(m, messages, tools, as_json, temperature)
        except Busy as ex:
            last = str(ex)
    raise Busy(last)


def _ask(model, messages, tools, as_json, temperature, retries=6):
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
    waited, last = 0.0, ""
    for _ in range(retries):
        try:
            r = requests.post(URL, json=body, headers=headers, timeout=(10, 40))
        except requests.RequestException as ex:
            last = type(ex).__name__
            continue
        if r.status_code == 429 or r.status_code >= 500:
            last = r.text[:200]
            wait = float(r.headers.get("retry-after", 5)) + 1
            if wait > LONG_WAIT:
                _exhausted[model] = time.time() + wait
                break
            if waited + wait > MAX_WAIT:
                break
            if on_wait:
                on_wait(round(wait))
            waited += wait
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
        msg["_model"] = model
        if as_json:
            try:
                json.loads(msg.get("content") or "")
            except ValueError:
                last = "reply was not valid JSON"
                continue
        if use_cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            hit.write_text(json.dumps(msg))
        return msg
    raise Busy(f"{model} unavailable: {last}")


def chat_json(system, user, temperature=0.4, model=None):
    """Send one chat turn and return the reply parsed as JSON."""
    msg = chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
               as_json=True, temperature=temperature, model=model)
    out = json.loads(msg["content"])
    if isinstance(out, dict):
        out["_model"] = msg.get("_model")
    return out
