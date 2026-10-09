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

from collections import defaultdict, deque

from fastapi import FastAPI, Query, Request
from fastapi.responses import FileResponse, StreamingResponse

import agent

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
FRONTEND = ROOT.parent / "frontend"
app = FastAPI(title="LocalTaste")
_live = threading.Semaphore(1)  # one live run at a time keeps us inside the LLM rate limit
_recent = defaultdict(deque)  # visitor -> times of their live runs
HOURLY_LIMIT = 8


def _allowed(visitor):
    seen, now = _recent[visitor], time.time()
    while seen and now - seen[0] > 3600:
        seen.popleft()
    if len(seen) >= HOURLY_LIMIT:
        return False
    seen.append(now)
    return True


def _slug(concept, location, inspirations=""):
    norm = re.sub(r"\s+", " ", f"{concept}|{location}|{inspirations}".lower()).strip().rstrip("|")
    return hashlib.sha1(norm.encode()).hexdigest()[:16]


def _step_event(step):
    res = step["result"]
    return {"title": agent.describe(step), "tool": step["tool"], "names": agent.preview(step),
            "failed": isinstance(res, dict) and "error" in res,
            "resolved_as": res.get("resolved_as") if isinstance(res, dict) else None}


def _package(concept, location, inspirations, out, plain):
    ledger = out["ledger"]
    return {
        "concept": concept, "location": location, "inspirations": inspirations, "brief": out["brief"], "plain": plain,
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


def _live_run(concept, location, inspirations, visitor):
    events = queue.Queue()

    def work():
        try:
            plain = {}
            side = threading.Thread(target=lambda: plain.update(agent.generic(concept, location, inspirations)))
            side.start()
            out = agent.run(concept, location, inspirations, on_step=lambda s: events.put(("step", _step_event(s))))
            events.put(("writing", {}))
            side.join()
            result = _package(concept, location, inspirations, out, plain)
            RUNS.mkdir(exist_ok=True)
            (RUNS / f"{_slug(concept, location, inspirations)}.json").write_text(json.dumps(result, ensure_ascii=False))
            events.put(("result", result))
        except agent.UnknownPlace as ex:
            events.put(("error", {"message": str(ex)}))
        except Exception:
            events.put(("error", {"message": "Something went wrong while writing this brief. Please try again in a minute."}))
        finally:
            _live.release()
            events.put(None)

    if not _allowed(visitor):
        yield _sse("error", {"message": f"You have reached the limit of {HOURLY_LIMIT} new briefs an hour. Saved examples still work."})
        return
    if not _live.acquire(timeout=1):
        yield _sse("error", {"message": "Another brief is being written right now. Try again in a minute, or open a saved example."})
        return
    yield _sse("start", {"saved": False})
    threading.Thread(target=work, daemon=True).start()
    while (item := events.get()) is not None:
        yield _sse(*item)


@app.get("/api/run")
def run(request: Request, concept: str = Query(min_length=5, max_length=300), location: str = Query(min_length=3, max_length=80),
        inspirations: str = Query(default="", max_length=120), fresh: bool = False):
    concept, location, inspirations = concept.strip(), location.strip(), inspirations.strip()
    visitor = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
    saved = RUNS / f"{_slug(concept, location, inspirations)}.json"
    stream = (_replay(json.loads(saved.read_text())) if saved.exists() and not fresh
              else _live_run(concept, location, inspirations, visitor))
    return StreamingResponse(stream, media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/examples")
def examples():
    runs = [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))] if RUNS.exists() else []
    return [{"concept": r["concept"], "location": r["location"], "inspirations": r.get("inspirations", "")} for r in runs]


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
