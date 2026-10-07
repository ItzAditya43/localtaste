"""LocalTaste agent: turns a business concept and a location into a taste-grounded brief.

The model plans its own Qloo queries through tools. Everything the tools return is kept
in a ledger, which feeds the final brief and lets us check that cited signals are real.
"""
import json
import sys

import qloo
from llm import chat, chat_json
from qloo import _get, _insights

DOMAINS = ["artist", "brand", "movie", "tv_show"]
DETAIL_TAGS = ("ambience", "decor", "menu_highlight", "specialty_dish", "good_for", "cuisine", "setting")
MAX_STEPS = 6


def _names(items, n=6):
    return [i["name"] for i in items[:n] if i.get("name")]


def find_tags(query):
    res = _get("/v2/tags", **{"filter.query": query, "take": 8})
    return [{"id": t["id"], "name": t["name"]} for t in res["results"]["tags"]]


def area_taste(location):
    p = qloo.neighbourhood_profile(location, take=6)
    return {"popular_venues": _names(p["venues"], 8), **{k: _names(v) for k, v in p["taste"].items()},
            "leanings": [t["name"] for t in p["tags"][:12]]}


def competitors(location, tag_ids):
    ents = _insights(**{"filter.type": "urn:entity:place", "filter.location.query": location,
                        "filter.tags": ",".join(tag_ids), "take": 8}).get("entities", [])
    out = []
    for e in ents:
        p = e.get("properties", {})
        detail = [t["name"] for t in e.get("tags", []) if t.get("name") and any(f":{d}:" in t.get("id", "") for d in DETAIL_TAGS)]
        out.append({"id": e["entity_id"], "name": e.get("name"), "popularity": round(e.get("popularity") or 0, 2),
                    "rating": p.get("business_rating"), "price_level": p.get("price_level"), "known_for": detail[:8]})
    return out


def lookup(name):
    return [{"id": e["entity_id"], "name": e["name"], "type": (e.get("types") or [""])[0].split(":")[-1],
             "where": e.get("properties", {}).get("address") or e.get("disambiguation")}
            for e in _get("/search", query=name, take=4).get("results", [])]


def audience_taste(entity_ids, domain):
    return [{"name": i["name"], "affinity": round(i["affinity"] or 0, 2)}
            for i in qloo.taste_from(entity_ids[:8], domain, take=7) if i.get("name")]


def venues_for_taste(location, entity_ids):
    return [{"id": i["id"], "name": i["name"], "affinity": round(i["affinity"] or 0, 2), "rating": i["rating"]}
            for i in qloo.places_for_taste(location, entity_ids[:8], take=8) if i.get("name")]


def city_taste(city, domain):
    return _names(qloo.city_taste(city, domain, take=7), 7)


def _fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}


_S, _IDS = {"type": "string"}, {"type": "array", "items": {"type": "string"}}
_DOMAIN = {"type": "string", "enum": DOMAINS}
TOOLS = [
    _fn("area_taste", "Taste profile of a neighbourhood, derived from the audiences of its venues: music, brands, films, TV and leanings.", {"location": _S}, ["location"]),
    _fn("find_tags", "Find Qloo tag ids for a kind of place, cuisine or feature (e.g. 'coffee shop', 'wine bar'). Needed before competitors.", {"query": _S}, ["query"]),
    _fn("competitors", "Existing venues in an area matching tag ids, with popularity, rating, price level and what they are known for. Prefer urn:tag:category:place:* ids.", {"location": _S, "tag_ids": _IDS}, ["location", "tag_ids"]),
    _fn("lookup", "Find the Qloo entity id of a named venue, brand, artist or title.", {"name": _S}, ["name"]),
    _fn("audience_taste", "What the audience of given entities (by id) also loves in another domain. Use on competitor ids or on the owner's inspirations.", {"entity_ids": _IDS, "domain": _DOMAIN}, ["entity_ids", "domain"]),
    _fn("venues_for_taste", "Venues in an area ranked by affinity to given entity ids. Shows where fans of an inspiration already go.", {"location": _S, "entity_ids": _IDS}, ["location", "entity_ids"]),
    _fn("city_taste", "What a whole city over-indexes on in one domain. City names only, not neighbourhoods.", {"city": _S, "domain": _DOMAIN}, ["city", "domain"]),
]
IMPL = {f.__name__: f for f in (area_taste, find_tags, competitors, lookup, audience_taste, venues_for_taste, city_taste)}

RESEARCH = (
    "You research a neighbourhood for someone opening a business there, using Qloo taste data through tools. "
    "Call several tools per turn where they are independent. Always cover: the area's taste, the direct competitors "
    "(find_tags then competitors; if fewer than 4 come back, retry once with a broader category tag), and what the competitors' audience loves in at least two other domains "
    "(audience_taste with competitor ids). Use lookup and venues_for_taste for any inspirations the owner names. "
    "Stop calling tools once you have enough; then reply with the single word DONE."
)
BRIEF_SCHEMA = (
    '{"positioning": "two sentences on how this concept should position itself here", '
    '"area_read": "two sentences on what this neighbourhood\'s audience is like", '
    '"competitors": [{"name": "", "takeaway": ""}], '
    '"recommendations": [{"area": "menu|music|decor|partners|pricing|events", "idea": "", "why": "", "signals": ["exact names from the evidence"]}], '
    '"risks": [{"risk": "", "signals": []}]}'
)
BRIEF = (
    "You write a launch brief for a new local business from Qloo evidence. Every recommendation must follow from the "
    "evidence and list in `signals` the names it rests on: names of venues, artists, brands, films, shows or leanings, "
    "copied verbatim from the evidence. Section headings are not signals. Never cite a name that is not in the evidence. A signal must directly support "
    "its idea: music ideas rest on artists, partnerships on brands or non-competing venues, pricing on competitors' "
    "price levels, menu on what competitors are known for or the area's leanings. Do not decorate ideas with unrelated "
    "names, and drop any idea the evidence does not support. Give 6-8 recommendations covering at least menu, music, decor and partners, and 3-4 "
    "competitors. Be specific to this neighbourhood; avoid advice that would fit anywhere. price_level is a 1-4 scale "
    "(1 cheap, 4 luxury), not a currency amount. Partners must be non-competing brands or venues, never direct "
    "competitors. Give 2-3 risks, each resting on evidence such as a strong competitor or a mismatch with the area's "
    "taste, with its signals. Reply as JSON: " + BRIEF_SCHEMA
)
GENERIC = (
    "You write a launch brief for a new local business from your own general knowledge. Give 6-8 recommendations covering "
    "at least menu, music, decor and partners, and 3-4 competitors. Reply as JSON: " + BRIEF_SCHEMA
)


def research(concept, location, on_step=None):
    """Let the model query Qloo until it has enough. Returns the ledger of tool calls and results."""
    messages = [{"role": "system", "content": RESEARCH},
                {"role": "user", "content": f"Concept: {concept}\nLocation: {location}"}]
    ledger, handles = [], {}

    def shorten(x):
        """Swap Qloo ids for short handles so the model does not have to copy UUIDs."""
        if isinstance(x, list):
            return [shorten(v) for v in x]
        if isinstance(x, dict) and "id" in x and "name" in x and not str(x["id"]).startswith("urn:"):
            handle = next((h for h, i in handles.items() if i == x["id"]), f"e{len(handles) + 1}")
            handles[handle] = x["id"]
            return {**x, "id": handle}
        return x

    for _ in range(MAX_STEPS):
        msg = chat(messages, tools=TOOLS)
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
                real = {**args, "entity_ids": [handles.get(i, i) for i in args["entity_ids"]]} if "entity_ids" in args else args
                result = shorten(IMPL[name](**real))
            except Exception as ex:  # surfaced to the model so it can correct the call
                args, result = c["function"]["arguments"], {"error": str(ex)[:200]}
            ledger.append({"tool": name, "args": args, "result": result})
            if on_step:
                on_step(ledger[-1])
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(result, ensure_ascii=False)})
    return ledger


_LABELS = {"area_taste": "Taste of the area", "competitors": "Competing venues", "lookup": "Entity lookup",
           "audience_taste": "What that audience also loves", "venues_for_taste": "Where that audience already goes",
           "city_taste": "City-wide taste"}


def _evidence(ledger):
    lines = []
    for s in ledger:
        if s["tool"] == "find_tags" or (isinstance(s["result"], dict) and "error" in s["result"]):
            continue
        scope = ", ".join(str(v) for k, v in s["args"].items() if k != "entity_ids" and k != "tag_ids")
        lines.append(f"[{_LABELS[s['tool']]} - {scope}]\n{json.dumps(s['result'], ensure_ascii=False)}")
    return "\n\n".join(lines)


def run(concept, location, on_step=None):
    """Research with Qloo, then write the grounded brief. Returns {"brief": ..., "ledger": ...}.

    Signals the model cites that Qloo never returned are sent back once for correction, then removed.
    """
    ledger = research(concept, location, on_step)
    kinds = signal_kinds(ledger)
    known = set(kinds)
    prompt = f"Concept: {concept}\nLocation: {location}\n\nEVIDENCE\n{_evidence(ledger)}"
    brief = chat_json(BRIEF, prompt)
    bad = [c for c in cited(brief) if c.lower() not in known]
    weak = [r["idea"] for r in brief.get("recommendations", []) if misfit(r, kinds)]
    if bad or weak:
        fix = (f"{prompt}\n\nYOUR PREVIOUS BRIEF\n{json.dumps(brief, ensure_ascii=False)}\n\nProblems to fix. Signals not "
               f"found in the evidence: {json.dumps(bad, ensure_ascii=False)}. Ideas whose signals do not support them: "
               f"{json.dumps(weak, ensure_ascii=False)}. Rewrite the brief so every signal is an exact name from the "
               "evidence and is the right kind of evidence for its idea; replace or drop ideas you cannot support.")
        brief = chat_json(BRIEF, fix)
    for r in brief.get("recommendations", []) + brief.get("risks", []):
        r["signals"] = [c for c in r.get("signals", []) if c.lower() in known]
    brief["recommendations"] = [r for r in brief.get("recommendations", []) if r["signals"] and not misfit(r, kinds)]
    return {"brief": brief, "ledger": ledger}


def generic(concept, location):
    """The same brief from the model alone, as the no-Qloo comparison."""
    return chat_json(GENERIC, f"Concept: {concept}\nLocation: {location}")


FITTING = {"music": {"artist"}, "partners": {"brand", "venue"}, "pricing": {"venue"}, "menu": {"venue", "leaning"},
           "decor": {"venue", "leaning", "brand"}}


def signal_kinds(ledger):
    """Every name Qloo returned during research, lower-cased, mapped to what it is (venue, artist, brand...)."""
    kinds = {}

    def add(names, kind):
        for n in names:
            n = n.get("name") if isinstance(n, dict) else n
            if isinstance(n, str):
                kinds.setdefault(n.lower(), kind)

    for s in ledger:
        res, tool = s["result"], s["tool"]
        if isinstance(res, dict) and "error" in res:
            continue
        if tool == "area_taste":
            add(res["popular_venues"], "venue")
            add(res["leanings"], "leaning")
            for d in DOMAINS:
                add(res.get(d, []), d)
        elif tool in ("competitors", "venues_for_taste"):
            add(res, "venue")
            for v in res:
                add(v.get("known_for", []), "leaning")
        elif tool in ("audience_taste", "city_taste"):
            add(res, s["args"]["domain"])
        elif tool == "lookup":
            for e in res:
                add([e], "venue" if e["type"] == "place" else e["type"])
    return kinds


def known_names(ledger):
    return set(signal_kinds(ledger))


def misfit(rec, kinds):
    """True when none of a recommendation's signals is the kind of evidence its area calls for."""
    want = FITTING.get(rec.get("area"))
    return bool(want) and not any(kinds.get(c.lower()) in want for c in rec.get("signals", []))


def cited(brief):
    return [s for r in brief.get("recommendations", []) + brief.get("risks", []) for s in r.get("signals", [])]


if __name__ == "__main__":
    concept, location = sys.argv[1], sys.argv[2]
    out = run(concept, location, on_step=lambda s: print(f"  > {s['tool']}({json.dumps(s['args'])})", file=sys.stderr))
    print(json.dumps(out["brief"], indent=2, ensure_ascii=False))
