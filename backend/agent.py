"""LocalTaste agent: turns a business concept and a location into a taste-grounded brief.

The model plans its own Qloo queries through tools. Everything the tools return is kept
in a ledger, which feeds the final brief and lets us check that cited signals are real.
"""
import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import qloo
from llm import RESEARCH_MODEL, chat, chat_json
from qloo import _get, _insights

DOMAINS = ["artist", "brand", "movie", "tv_show"]
ATLAS = json.loads(Path(__file__).with_name("atlas.json").read_text())
MAX_STEPS = 6
MIN_CALLS, MAX_CALLS = 4, 8
MIN_RECS = 5
MIN_SIGNALS = {"events": 2}


def _names(items, n=6):
    return [i["name"] for i in items[:n] if i.get("name")]


def find_tags(query, take=8):
    res = _get("/v2/tags", **{"filter.query": query, "take": take})
    return [{"id": t["id"], "name": t["name"]} for t in res["results"]["tags"]]


TAG_GROUPS = {"music_genres": "urn:tag:genre:music", "food_scene": "urn:tag:genre:place", "ambience": "urn:tag:ambience:qloo"}


class ThinBrief(RuntimeError):
    """Raised when too few recommendations survive verification to be worth showing."""


class UnknownPlace(ValueError):
    """Raised when Qloo cannot place the location or has no venues there."""


def area_taste(location):
    resolved = qloo.resolve_location(location)
    venues = qloo.places(location, take=8) if resolved else []
    if not venues:
        raise UnknownPlace(f"Qloo has no venue data for \u201c{location}\u201d. Try the form \u201cNeighbourhood, City\u201d, for example \u201cShoreditch, London\u201d.")
    ids = [v["id"] for v in venues]
    city = location.split(",")[-1].strip()
    jobs = {d: (qloo.taste_from, ids, d, 6) for d in DOMAINS}
    jobs["city_artists"] = (qloo.city_taste, city, "artist", 6)
    jobs.update({k: (qloo.tags_from, ids, 8, t) for k, t in TAG_GROUPS.items()})
    with ThreadPoolExecutor(3) as pool:
        got = {k: f.result() for k, f in {k: pool.submit(*j) for k, j in jobs.items()}.items()}
    spots = [(v["lat"], v["lon"]) for v in venues if v["lat"] is not None]
    return {
        "resolved_as": resolved,
        "in_named_city": qloo.same_city(location, resolved),
        "centre": [statistics.median(p[0] for p in spots), statistics.median(p[1] for p in spots)] if spots else None,
        "popular_venues": _names(venues, 8),
        "local_venue_artists": _names(got["artist"]),
        "city_artists": _names(got["city_artists"]),
        "music_genres": [t["name"].strip() for t in got["music_genres"]],
        "brands": _names(got["brand"]),
        "movies": _names(got["movie"]),
        "tv_shows": _names(got["tv_show"]),
        "food_scene": [t["name"] for t in got["food_scene"]],
        "ambience": [t["name"] for t in got["ambience"]],
    }


FOOD_TAGS, FEEL_TAGS = ("menu_highlight", "specialty_dish", "cuisine"), ("ambience", "decor", "setting", "good_for")
PLACE_TAG_TYPES = "urn:tag:category:place,urn:tag:genre:place"


def _tags_for(kind):
    """Qloo place tags for a plain description such as 'coffee shop'."""
    res = _get("/v2/tags", **{"filter.query": kind, "filter.tag.types": PLACE_TAG_TYPES, "take": 6})
    return [t["id"] for t in res["results"]["tags"] if ":hotel:" not in t["id"]][:2]


def competitors(location, kinds):
    seen, out = set(), []
    tagged = {kind: _tags_for(kind) for kind in kinds[:3]}
    if not any(tagged.values()):
        raise ValueError(f"Qloo has no place category matching {kinds}. Try more common names, e.g. 'book store', 'bar', 'cafe'.")
    for kind, tags in tagged.items():
        for tag in tags:
            ents = _insights(**{"filter.type": "urn:entity:place", "filter.location.query": location,
                                "filter.tags": tag, "take": 10}).get("entities", [])
            for e in ents:
                if e["entity_id"] in seen or not e.get("name"):
                    continue
                seen.add(e["entity_id"])
                p = e.get("properties", {})
                tags = [t for t in e.get("tags", []) if t.get("name")]
                pick = lambda types: [t["name"] for t in tags if any(f":{d}:" in t.get("id", "") for d in types)][:5]
                out.append(_venue(e, kind))
    return sorted(out, key=lambda v: -v["popularity"])[:12]


def _venue(e, kind):
    p = e.get("properties", {})
    tags = [t for t in e.get("tags", []) if t.get("name")]

    def pick(types):
        return [t["name"] for t in tags if any(f":{d}:" in t.get("id", "") for d in types)][:5]

    return {"id": e["entity_id"], "name": e["name"], "category": kind, "popularity": round(e.get("popularity") or 0, 2),
            "rating": p.get("business_rating"), "price_level": p.get("price_level"), "known_for": pick(FOOD_TAGS),
            "feel": pick(FEEL_TAGS), "lat": (e.get("location") or {}).get("lat"), "lon": (e.get("location") or {}).get("lon")}


def _city(location):
    city = qloo._plain(location.split(",")[-1].strip())
    return qloo.CITY_ALIASES.get(city, city)


def taste_twins(location, top=4):
    """Neighbourhoods in other cities where this area's audience would feel most at home.

    This area's venues are used as taste signals; each atlas area is scored by the mean affinity of its
    ten best-matching venues.
    """
    ids = [v["id"] for v in qloo.places(location, take=8)]
    others = [a for a in ATLAS if _city(a["name"]) != _city(location)]

    def score(area):
        ents = _insights(**{"filter.type": "urn:entity:place", "filter.location.query": area["name"],
                            "signal.interests.entities": ",".join(ids), "take": 10}).get("entities", [])
        affinities = [e["query"]["affinity"] for e in ents if e.get("query", {}).get("affinity") is not None]
        return {"name": area["name"], "match": round(100 * statistics.mean(affinities)) if affinities else 0,
                "lat": area["lat"], "lon": area["lon"]}

    with ThreadPoolExecutor(3) as pool:
        scored = sorted(pool.map(score, others), key=lambda a: -a["match"])
    return {"twins": scored[:top], "least_alike": scored[-1], "compared": len(scored)}


def borrow_from(location, twin, kinds):
    """Venues of the given kinds in the twin area, ranked by how strongly this area's audience would take to them."""
    ids = [v["id"] for v in qloo.places(location, take=8)]
    seen, out = set(), []
    for kind in kinds[:2]:
        for tag in _tags_for(kind)[:1]:
            ents = _insights(**{"filter.type": "urn:entity:place", "filter.location.query": twin, "filter.tags": tag,
                                "signal.interests.entities": ",".join(ids), "take": 5}).get("entities", [])
            for e in ents:
                if e["entity_id"] not in seen and e.get("name"):
                    seen.add(e["entity_id"])
                    out.append({**_venue(e, kind), "affinity": round(e.get("query", {}).get("affinity") or 0, 2)})
    return sorted(out, key=lambda v: -v["affinity"])[:6]


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
    _fn("area_taste", "Taste profile of a neighbourhood as 'Neighbourhood, City': artists favoured by its venues' audiences and by the city, music genres, brands, films, TV, food scene and ambience.", {"location": _S}, ["location"]),
    _fn("competitors", "Existing venues of given kinds in an area, with popularity, rating, price tier, what they serve and how they feel. Pass 1-3 plain kinds, e.g. ['coffee shop', 'record store'].", {"location": _S, "kinds": _IDS}, ["location", "kinds"]),
    _fn("lookup", "Find the Qloo entity id of a named venue, brand, artist or title.", {"name": _S}, ["name"]),
    _fn("audience_taste", "What the audience of given entities (by id) also loves in another domain. Use on competitor ids or on the owner's inspirations.", {"entity_ids": _IDS, "domain": _DOMAIN}, ["entity_ids", "domain"]),
    _fn("venues_for_taste", "Venues in an area ranked by affinity to given entity ids. Shows where fans of an inspiration already go.", {"location": _S, "entity_ids": _IDS}, ["location", "entity_ids"]),
    _fn("city_taste", "What a whole city over-indexes on in one domain. City names only, not neighbourhoods.", {"city": _S, "domain": _DOMAIN}, ["city", "domain"]),
]
IMPL = {f.__name__: f for f in (area_taste, competitors, lookup, audience_taste, venues_for_taste, city_taste)}

RESEARCH = (
    "You research a neighbourhood for someone opening a business there, using Qloo taste data through tools. "
    "Call several tools per turn where they are independent. The area's taste is given to you. Always cover: the direct competitors "
    "(competitors with the kinds of place this concept is; if fewer than 4 come back, retry once with broader kinds), and what the competitors' audience loves in at least two other domains "
    "(audience_taste with the ids of the 3-5 closest competitors). If the owner names places, brands or artists they admire, "
    "lookup each one, then use venues_for_taste to see where their fans already go locally and audience_taste to see what "
    "else those fans love. "
    "Stop calling tools once you have enough; then reply with the single word DONE."
)
BRIEF_SCHEMA = (
    '{"positioning": "two sentences on how this concept should position itself here", '
    '"area_read": "two sentences on what this neighbourhood\'s audience is like", '
    '"competitors": [{"name": "", "takeaway": ""}], '
    '"recommendations": [{"area": "menu|music|decor|partners|pricing|events|borrow", "idea": "", "why": "", "signals": ["exact names from the evidence"]}], '
    '"risks": [{"risk": "", "signals": []}]}'
)
BRIEF = (
    "You write a launch brief for a new local business from Qloo evidence. Every recommendation must follow from the "
    "evidence and list in `signals` the names it rests on: names of venues, artists, brands, films, shows or leanings, "
    "genres or other items copied verbatim from the evidence. Section headings are not signals. Never cite a name that is not in the evidence. A signal must directly support "
    "its idea: music ideas rest on artists, partnerships on brands or non-competing venues, pricing on competitors' "
    "price levels, menu on what competitors are known for or the area's food scene, decor on ambience. For music, "
    "the artists favoured by local venues' audiences reflect the neighbourhood and the city-wide artists the wider "
    "city; use whichever suits the concept, and prefer the city-wide list when the venue list looks unrelated to the "
    "local culture. Say only what the evidence shows about a name: that this audience favours it, what kind of place a venue is, "
    "what it serves and how it feels. Do not call anything local, popular, famous, iconic or beloved unless a number in "
    "the evidence shows it. Never add facts about a named artist, brand or venue from your own knowledge (where they are from, "
    "what they sell, that they are local). Write for a business owner in plain language; never mention data field "
    "names, tools or the word 'evidence'. Do not decorate ideas with unrelated "
    "names, and drop any idea the evidence does not support. Give 6-8 recommendations covering at least menu, music, decor and partners, and 3-4 "
    "competitors. If the evidence lists venues in a taste twin, add 2 recommendations with area \"borrow\": an idea taken "
    "from one of those venues (what it serves or how it feels) and adapted to this neighbourhood, citing that venue; "
    "these venues are abroad, so they are inspiration, never competitors or partners. "
    "Be specific to this neighbourhood; avoid advice that would fit anywhere. Price tier is a 1-4 scale "
    "(1 cheap, 4 luxury), not a currency amount. Partners must be non-competing brands or venues, never direct "
    "competitors. Give 2-3 risks, each resting on evidence such as a strong competitor or a mismatch with the area's "
    "taste, with its signals. Reply as JSON: " + BRIEF_SCHEMA
)
GENERIC = (
    "You write a launch brief for a new local business from your own general knowledge. Give 6-8 recommendations covering "
    "at least menu, music, decor and partners, and 3-4 competitors. Reply as JSON: " + BRIEF_SCHEMA
)


def _twin_steps(location, ledger, record):
    """After the model's research: find this area's taste twins and what could be borrowed from the closest one."""
    found = taste_twins(location)
    record("taste_twins", {"location": location}, found)
    kinds = next((s["args"]["kinds"] for s in ledger if s["tool"] == "competitors" and isinstance(s["result"], list) and s["result"]), None)
    if kinds and found["twins"]:
        twin = found["twins"][0]["name"]
        record("borrow_from", {"twin": twin, "kinds": kinds[:2]}, borrow_from(location, twin, kinds))


def _lite(tool, result):
    """What the planning model sees: enough to choose the next call, without the detail kept for the brief."""
    if tool == "competitors" and isinstance(result, list):
        return [{k: v[k] for k in ("id", "name", "category", "popularity")} for v in result]
    if tool == "area_taste":
        return {k: v for k, v in result.items() if k not in ("centre", "in_named_city")}
    return result


def research(concept, location, inspirations="", on_step=None):
    """Read the area, then let the model query Qloo until it has enough. Returns the ledger of tool calls and results."""
    ledger, handles, names, models = [], {}, {}, set()

    def shorten(x):
        """Swap Qloo ids for short handles so the model does not have to copy UUIDs."""
        if isinstance(x, list):
            return [shorten(v) for v in x]
        if isinstance(x, dict) and "id" in x and "name" in x and not str(x["id"]).startswith("urn:"):
            handle = next((h for h, i in handles.items() if i == x["id"]), f"e{len(handles) + 1}")
            handles[handle], names[handle] = x["id"], x["name"]
            return {**x, "id": handle}
        return x

    def record(tool, args, result):
        step = {"tool": tool, "args": args, "result": result}
        if isinstance(args, dict) and "entity_ids" in args:
            step["about"] = [names[i] for i in args["entity_ids"] if i in names]
        ledger.append(step)
        if on_step:
            on_step(step)

    area = area_taste(location)  # always first; fails fast on a place Qloo cannot read
    record("area_taste", {"location": location}, area)
    ask = f"Concept: {concept}\nLocation: {location}"
    if inspirations:
        ask += f"\nThe owner admires: {inspirations}"
    messages = [{"role": "system", "content": RESEARCH},
                {"role": "user", "content": f"{ask}\n\narea_taste is already done:\n{json.dumps(area, ensure_ascii=False)}"}]

    def missing():
        ok = [s for s in ledger if not (isinstance(s["result"], dict) and "error" in s["result"])]
        todo = []
        if inspirations and not any(s["tool"] == "lookup" and s["result"] for s in ok):
            todo.append(f"lookup for what the owner admires ({inspirations}), then audience_taste or venues_for_taste with those ids")
        if not any(s["tool"] == "competitors" and s["result"] for s in ok):
            todo.append("competitors, with broader kinds if needed")
        elif len({s["args"].get("domain") for s in ok if s["tool"] == "audience_taste"}) < 2:
            todo.append("audience_taste on the competitor ids for two different domains")
        return todo

    nudges = 0
    for _ in range(MAX_STEPS + 2):
        msg = chat(messages, tools=TOOLS, model=RESEARCH_MODEL)
        calls = msg.get("tool_calls") or []
        if not calls:
            todo = missing()
            if not todo or nudges == 2:
                break
            nudges += 1
            messages.append({"role": "user", "content": "Still required before you finish: " + "; ".join(todo) + "."})
            continue
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls})
        models.add(msg.get("_model"))
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"]["arguments"] or "{}")
                if any(s["tool"] == name and s["args"] == args for s in ledger):
                    messages.append({"role": "tool", "tool_call_id": c["id"], "content": '"Already asked; use the earlier answer."'})
                    continue
                real = {**args, "entity_ids": [handles.get(i, i) for i in args["entity_ids"]]} if "entity_ids" in args else args
                result = shorten(IMPL[name](**real))
            except Exception as ex:  # surfaced to the model so it can correct the call
                args, result = c["function"]["arguments"], {"error": str(ex)[:200]}
            record(name, args, result)
            messages.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(_lite(name, result), ensure_ascii=False)})
        if len(ledger) >= MAX_CALLS or (len(ledger) >= MIN_CALLS and not missing()):
            break
    try:
        _twin_steps(location, ledger, record)
    except Exception:  # twins are an extra; the brief still stands without them
        pass
    return ledger


_LABELS = {"area_taste": "Taste of the area", "competitors": "Competing venues", "lookup": "Entity lookup",
           "audience_taste": "What that audience also loves", "venues_for_taste": "Where that audience already goes",
           "city_taste": "City-wide taste", "taste_twins": "Taste twins: areas in other cities whose venues this audience has the highest affinity for",
           "borrow_from": "Venues in the closest taste twin, ranked by this area's audience's affinity for them"}


_AREA_KEYS = {"popular_venues": "most popular venues", "local_venue_artists": "artists favoured by local venues' audiences",
              "city_artists": "artists favoured across the city", "music_genres": "music genres", "brands": "brands",
              "movies": "films", "tv_shows": "TV shows", "food_scene": "food scene", "ambience": "ambience"}


_VENUE_KEYS = {"price_level": "price tier (1 cheap - 4 luxury)", "known_for": "serves", "feel": "feels", "rating": "rating out of 5",
               "category": "what it is"}
FIELD_NAMES = ("price_level", "known_for", "what it is", "city_artists", "local_venue_artists", "popular_venues", "food_scene",
               "music_genres", "audience_taste", "area_taste", "venues_for_taste", "tv_shows")


def _evidence(ledger):
    lines = []
    for s in ledger:
        if isinstance(s["result"], dict) and "error" in s["result"]:
            continue
        scope = ", ".join(", ".join(v) if isinstance(v, list) else str(v) for k, v in s["args"].items() if k != "entity_ids")
        res = s["result"]
        if s["tool"] == "area_taste":
            res = {_AREA_KEYS[k]: v for k, v in res.items() if k in _AREA_KEYS}
        elif s["tool"] in ("competitors", "venues_for_taste", "borrow_from"):
            res = [{_VENUE_KEYS.get(k, k): x for k, x in v.items() if k not in ("id", "lat", "lon")} for v in res]
        elif s["tool"] == "taste_twins":
            res = [{"area": t["name"], "match out of 100": t["match"]} for t in res["twins"]]
        lines.append(f"[{_LABELS[s['tool']]} - {scope}]\n{json.dumps(res, ensure_ascii=False)}")
    return "\n\n".join(lines)


def run(concept, location, inspirations="", on_step=None):
    """Research with Qloo, then write the grounded brief. Returns {"brief": ..., "ledger": ...}.

    Signals the model cites that Qloo never returned are sent back once for correction, then removed.
    """
    ledger = research(concept, location, inspirations, on_step)
    kinds = signal_kinds(ledger)
    known = set(kinds)
    admires = f"\nThe owner admires: {inspirations}" if inspirations else ""
    prompt = f"Concept: {concept}\nLocation: {location}{admires}\n\nEVIDENCE\n{_evidence(ledger)}"
    brief = chat_json(BRIEF, prompt)

    def clean(b):
        for r in b.get("recommendations", []) + b.get("risks", []):
            r["signals"] = [c for c in dict.fromkeys(r.get("signals", [])) if c.lower() in known]
        rivals = {c.get("name", "").lower() for c in b.get("competitors", [])}
        for r in b.get("recommendations", []):
            if r.get("area") == "partners":  # a venue cannot be both a rival and a partner
                r["signals"] = [c for c in r["signals"] if c.lower() not in rivals]
        b["competitors"] = [c for c in b.get("competitors", []) if kinds.get(c.get("name", "").lower()) == "venue"]
        b["recommendations"] = [r for r in b.get("recommendations", []) if len(r["signals"]) >= MIN_SIGNALS.get(r.get("area"), 1) and not misfit(r, kinds)]
        return b

    def problems(b):
        bad = [c for c in cited(b) if c.lower() not in known]
        weak = [r["idea"] for r in b.get("recommendations", []) if misfit(r, kinds)]
        return bad, weak

    writer = brief.pop("_model", None)
    bad, weak = problems(brief)
    if len(brief.get("recommendations", [])) - len(weak) < MIN_RECS or len(bad) > 3:
        fix = (f"{prompt}\n\nYOUR PREVIOUS BRIEF\n{json.dumps(brief, ensure_ascii=False)}\n\nProblems to fix. Signals not "
               f"found in the evidence: {json.dumps(bad, ensure_ascii=False)}. Ideas whose signals do not support them: "
               f"{json.dumps(weak, ensure_ascii=False)}. Rewrite the brief so every signal is an exact name from the "
               "evidence and is the right kind of evidence for its idea; replace or drop ideas you cannot support.")
        brief = chat_json(BRIEF, fix)
        writer = brief.pop("_model", writer)
    brief = clean(brief)
    if len(brief["recommendations"]) < MIN_RECS or len(brief.get("competitors", [])) < 2:
        raise ThinBrief(f"only {len(brief['recommendations'])} supported recommendations")
    return {"brief": brief, "ledger": ledger, "writer": writer}


def generic(concept, location, inspirations=""):
    """The same brief from the model alone, as the no-Qloo comparison."""
    admires = f"\nThe owner admires: {inspirations}" if inspirations else ""
    return chat_json(GENERIC, f"Concept: {concept}\nLocation: {location}{admires}")


FITTING = {"music": {"artist", "genre"}, "partners": {"brand", "venue"}, "pricing": {"venue"},
           "menu": {"venue", "food"}, "decor": {"venue", "ambience", "brand"}, "borrow": {"twin_venue"}}


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
            for key, kind in (("popular_venues", "venue"), ("local_venue_artists", "artist"), ("city_artists", "artist"),
                              ("music_genres", "genre"), ("brands", "brand"), ("movies", "movie"), ("tv_shows", "tv_show"),
                              ("food_scene", "food"), ("ambience", "ambience")):
                add(res[key], kind)
        elif tool in ("competitors", "venues_for_taste"):
            add(res, "venue")
            for v in res:
                add(v.get("known_for", []), "food")
                add(v.get("feel", []), "ambience")
        elif tool in ("audience_taste", "city_taste"):
            add(res, s["args"]["domain"])
        elif tool == "taste_twins":
            add(res["twins"], "twin")
        elif tool == "borrow_from":
            add(res, "twin_venue")
        elif tool == "lookup":
            for e in res[:1]:  # only the top match counts as found
                add([e], "venue" if e["type"] == "place" else e["type"])
    return kinds


def known_names(ledger):
    return set(signal_kinds(ledger))


def misfit(rec, kinds):
    """True when none of a recommendation's signals is the kind of evidence its area calls for."""
    want = FITTING.get(rec.get("area"))
    return bool(want) and not any(kinds.get(c.lower()) in want for c in rec.get("signals", []))


def describe(step):
    """A one-line, human description of a research step, for showing the agent's work."""
    a = step["args"] if isinstance(step["args"], dict) else {}
    about = step.get("about") or []
    who = ", ".join(about[:3]) + (f" and {len(about) - 3} more" if len(about) > 3 else "") or "those venues"
    domain = {"artist": "music", "brand": "brands", "movie": "films", "tv_show": "TV"}.get(a.get("domain"), "")
    return {
        "area_taste": f"Reading the taste of {a.get('location', 'the area')}",
        "competitors": f"Finding {' and '.join(k + 's' for k in a.get('kinds', [])[:3]) or 'comparable venues'} in {a.get('location', 'the area')}",
        "lookup": f"Finding \u201c{a.get('name', '')}\u201d in Qloo",
        "audience_taste": f"Asking what fans of {who} also love: {domain}",
        "venues_for_taste": f"Finding where fans of {who} already go in {a.get('location', 'the area')}",
        "city_taste": f"Reading {a.get('city', 'the city')}-wide taste: {domain}",
        "taste_twins": f"Searching {step['result'].get('compared', '') if isinstance(step['result'], dict) else ''} neighbourhoods worldwide for this area's taste twins",
        "borrow_from": f"Finding what this crowd would love most in {a.get('twin', 'the twin')}",
    }.get(step["tool"], step["tool"])


def preview(step, n=8):
    """A few names from a step's result."""
    res = step["result"]
    if isinstance(res, dict) and "error" in res:
        return []
    if step["tool"] == "taste_twins":
        return [f"{t['name']} · {t['match']}" for t in res["twins"]]
    if step["tool"] == "area_taste":
        return res["popular_venues"][:3] + res["city_artists"][:2] + res["brands"][:2] + res["music_genres"][:2]
    return list(dict.fromkeys((x.get("name") if isinstance(x, dict) else x) for x in res))[:n]


def venue_facts(ledger):
    """Rating, price tier, popularity and highlights for every venue seen during research, keyed by lower-case name."""
    facts = {}
    for s in ledger:
        if s["tool"] in ("competitors", "venues_for_taste", "borrow_from") and isinstance(s["result"], list):
            for v in s["result"]:
                facts.setdefault(v["name"].lower(), {k: v.get(k) for k in ("category", "popularity", "rating", "price_level", "known_for", "feel", "affinity", "lat", "lon")})
                if s["tool"] == "borrow_from":
                    facts[v["name"].lower()]["in"] = s["args"]["twin"]
    return facts


def check_rivals(brief, location):
    """How many of a brief's named competitors Qloo lists as a place in that city. Returns (found, total)."""
    names = [c.get("name", "") for c in brief.get("competitors", []) if c.get("name")]
    found = 0
    for n in names:
        try:
            found += any(r["type"] == "place" and qloo.same_city(location, r.get("where") or "") for r in lookup(n))
        except Exception:
            pass
    return found, len(names)


def prose(brief):
    """All free text in a brief, for checks on what it says."""
    parts = [brief.get("positioning", ""), brief.get("area_read", "")]
    parts += [c.get("takeaway", "") for c in brief.get("competitors", [])]
    parts += [f"{r.get('idea', '')} {r.get('why', '')}" for r in brief.get("recommendations", [])]
    parts += [r.get("risk", "") for r in brief.get("risks", [])]
    return "\n".join(parts)


def cited(brief):
    return [s for r in brief.get("recommendations", []) + brief.get("risks", []) for s in r.get("signals", [])]


if __name__ == "__main__":
    concept, location = sys.argv[1], sys.argv[2]
    out = run(concept, location, sys.argv[3] if len(sys.argv) > 3 else "", on_step=lambda s: print(f"  > {s['tool']}({json.dumps(s['args'])})", file=sys.stderr))
    print(json.dumps(out["brief"], indent=2, ensure_ascii=False))
