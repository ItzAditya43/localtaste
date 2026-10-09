"""LocalTaste web server: streams the agent's research and brief to the browser.

Finished runs are saved under runs/ and replayed instantly when the same question is asked again.
"""
import hashlib
import json
import queue
import re
import threading
import time
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse

import agent

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
FRONTEND = ROOT.parent / "frontend"
app = FastAPI(title="LocalTaste")
_live = threading.Semaphore(1)  # one live run at a time keeps us inside the LLM rate limit


def _slug(concept, location):
    norm = re.sub(r"\s+", " ", f"{concept}|{location}".lower()).strip()
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def _step_event(step):
    return {"title": agent.describe(step), "tool": step["tool"], "names": agent.preview(step),
            "failed": isinstance(step["result"], dict) and "error" in step["result"]}


def _package(concept, location, out, plain):
    ledger = out["ledger"]
    return {
        "concept": concept, "location": location, "brief": out["brief"], "plain": plain,
        "steps": [_step_event(s) for s in ledger],
        "kinds": agent.signal_kinds(ledger), "venues": agent.venue_facts(ledger),
        "stats": {"qloo_calls": len(ledger), "signals": len(set(c.lower() for c in agent.cited(out["brief"])))},
    }


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _replay(saved):
    yield _sse("start", {"saved": True})
    for step in saved["steps"]:
        time.sleep(0.35)
        yield _sse("step", step)
    yield _sse("result", saved)


def _live_run(concept, location):
    events = queue.Queue()

    def work():
        try:
            plain = {}
            side = threading.Thread(target=lambda: plain.update(agent.generic(concept, location)))
            side.start()
            out = agent.run(concept, location, on_step=lambda s: events.put(("step", _step_event(s))))
            events.put(("writing", {}))
            side.join()
            result = _package(concept, location, out, plain)
            RUNS.mkdir(exist_ok=True)
            (RUNS / f"{_slug(concept, location)}.json").write_text(json.dumps(result, ensure_ascii=False))
            events.put(("result", result))
        except Exception as ex:
            events.put(("error", {"message": str(ex)[:300]}))
        finally:
            _live.release()
            events.put(None)

    if not _live.acquire(timeout=1):
        yield _sse("error", {"message": "Another brief is being written right now. Try again in a minute, or open a saved example."})
        return
    yield _sse("start", {"saved": False})
    threading.Thread(target=work, daemon=True).start()
    while (item := events.get()) is not None:
        yield _sse(*item)


@app.get("/api/run")
def run(concept: str = Query(min_length=5, max_length=300), location: str = Query(min_length=3, max_length=80), fresh: bool = False):
    saved = RUNS / f"{_slug(concept, location)}.json"
    stream = _replay(json.loads(saved.read_text())) if saved.exists() and not fresh else _live_run(concept.strip(), location.strip())
    return StreamingResponse(stream, media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/examples")
def examples():
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))] if RUNS.exists() else []
    return [{"concept": r["concept"], "location": r["location"]} for r in runs]


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
