"""Thin client for the Qloo hackathon API.

Every function here maps to a query shape that was verified to return data
with a hackathon key. Neighbourhood-level `signal.location` is sparse, so
neighbourhood taste is derived from the area's venues used as signals.
"""
import os
from pathlib import Path

import requests

_ENV = Path(__file__).resolve().parent.parent / ".env"
if _ENV.exists():
    for line in _ENV.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

BASE = os.environ.get("QLOO_API_URL", "https://hackathon.api.qloo.com")
TASTE_TYPES = ("artist", "brand", "movie", "tv_show")

_session = requests.Session()
_session.headers["X-Api-Key"] = os.environ.get("QLOO_API_KEY", "")


def _get(path, **params):
    r = _session.get(BASE + path, params=params, timeout=40)
    r.raise_for_status()
    return r.json()


def _insights(**params):
    return _get("/v2/insights", **params).get("results", {})


def _slim(e):
    p = e.get("properties", {})
    q = e.get("query", {})
    return {
        "id": e["entity_id"],
        "name": e.get("name"),
        "type": e.get("subtype"),
        "popularity": e.get("popularity"),
        "affinity": q.get("affinity"),
        "explain": q.get("explainability"),
        "rating": p.get("business_rating"),
        "price_level": p.get("price_level"),
        "description": p.get("description"),
        "tags": [t["name"] for t in e.get("tags", []) if t.get("name")],
    }


def search(query, take=5):
    """Look up entities by name."""
    return [
        {"id": e["entity_id"], "name": e.get("name"), "types": e.get("types", [])}
        for e in _get("/search", query=query, take=take).get("results", [])
    ]


def places(location, tags=None, take=20, page=1):
    """Venues in an area, most popular first. `tags` is a comma-separated tag id list."""
    params = {"filter.type": "urn:entity:place", "filter.location.query": location, "take": take, "page": page}
    if tags:
        params["filter.tags"] = tags
    return [_slim(e) for e in _insights(**params).get("entities", [])]


def taste_from(entity_ids, type, take=10, explain=False):
    """Cross-domain taste of the audience behind a set of entities (e.g. venues -> artists)."""
    params = {
        "filter.type": f"urn:entity:{type}",
        "signal.interests.entities": ",".join(entity_ids),
        "take": take,
    }
    if explain:
        params["feature.explainability"] = "true"
    return [_slim(e) for e in _insights(**params).get("entities", [])]


def city_taste(city, type, take=10):
    """What a city over-indexes on. Works at city level; neighbourhoods mostly return nothing."""
    params = {"filter.type": f"urn:entity:{type}", "signal.location.query": city, "take": take}
    return [_slim(e) for e in _insights(**params).get("entities", [])]


def places_for_taste(location, entity_ids, take=10):
    """Venues in an area ranked by affinity to a set of taste signals."""
    params = {
        "filter.type": "urn:entity:place",
        "filter.location.query": location,
        "signal.interests.entities": ",".join(entity_ids),
        "take": take,
    }
    return [_slim(e) for e in _insights(**params).get("entities", [])]


def demographics(entity_ids):
    """Age and gender skew per entity, as {entity_id: {"age": {...}, "gender": {...}}}."""
    res = _insights(**{"filter.type": "urn:demographics", "signal.interests.entities": ",".join(entity_ids)})
    return {d["entity_id"]: d["query"] for d in res.get("demographics", [])}


def tags_from(entity_ids, take=20):
    """Descriptive tags the audience behind a set of entities leans towards."""
    res = _insights(**{"filter.type": "urn:tag", "signal.interests.entities": ",".join(entity_ids), "take": take})
    return [
        {"id": t.get("tag_id") or t.get("id"), "name": t["name"], "type": t.get("subtype") or t.get("type"),
         "affinity": t.get("query", {}).get("affinity")}
        for t in res.get("tags", [])
    ]


def neighbourhood_profile(location, venues=8, take=8):
    """Venue-derived taste profile of an area: its venues, their audience skew and cross-domain taste."""
    top = places(location, take=venues)
    ids = [p["id"] for p in top]
    demo = demographics(ids) if ids else {}
    for p in top:
        p["demographics"] = demo.get(p["id"])
    return {
        "location": location,
        "venues": top,
        "taste": {t: taste_from(ids, t, take=take) for t in TASTE_TYPES} if ids else {},
        "tags": tags_from(ids) if ids else [],
    }


if __name__ == "__main__":
    import sys

    prof = neighbourhood_profile(sys.argv[1] if len(sys.argv) > 1 else "Indiranagar, Bangalore")
    print("Venues:", ", ".join(p["name"] for p in prof["venues"]))
    for t, items in prof["taste"].items():
        print(f"{t}:", ", ".join(i["name"] for i in items))
    print("tags:", ", ".join(t["name"] for t in prof["tags"]))
