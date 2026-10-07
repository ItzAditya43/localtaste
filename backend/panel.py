"""Audience segments for an area, built only from Qloo data.

Venues are grouped by the age band and gender their audience skews towards;
each group's venues are then used as signals to pull that segment's taste.
"""
from collections import defaultdict

import qloo
from llm import chat_json

GENDER_LEAN = 0.15


def _segment_key(demo):
    age = max(demo["age"], key=demo["age"].get)
    male = demo["gender"]["male"]
    gender = "male" if male > GENDER_LEAN else "female" if male < -GENDER_LEAN else "mixed"
    return age, gender


def build_segments(location, venues=40, max_segments=4, min_venues=3, take=5):
    """Return the area's largest audience segments with the venues and tastes that define them."""
    pool = qloo.places(location, take=venues)
    demo = qloo.demographics([p["id"] for p in pool])
    groups = defaultdict(list)
    for p in pool:
        if p["id"] in demo:
            p["demographics"] = demo[p["id"]]
            groups[_segment_key(demo[p["id"]])].append(p)

    segments = []
    for (age, gender), members in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(members) < min_venues or len(segments) == max_segments:
            continue
        ids = [m["id"] for m in members[:5]]
        segments.append({
            "age": age,
            "gender": gender,
            "share": round(len(members) / len(pool), 2),
            "venues": [m["name"] for m in members],
            "venue_ids": [m["id"] for m in members],
            "taste": {t: [x["name"] for x in qloo.taste_from(ids, t, take=take)] for t in qloo.TASTE_TYPES},
            "tags": [t["name"] for t in qloo.tags_from(ids, take=12)],
        })
    return segments


def _persona(i, seg, area, grounded):
    who = f"Persona {i}: {seg['age'].replace('_', ' ')}, {'any gender' if seg['gender'] == 'mixed' else seg['gender']}, spends time in {area}."
    if not grounded:
        return who
    t = seg["taste"]
    return (
        f"{who}\n  Regular at: {', '.join(seg['venues'][:6])}\n  Listens to: {', '.join(t['artist'])}\n"
        f"  Buys from: {', '.join(t['brand'])}\n  Watches: {', '.join(t['movie'] + t['tv_show'])}\n"
        f"  Leans towards: {', '.join(seg['tags'])}"
    )


def react(concept, segments, area, grounded=True):
    """Have each segment's persona judge a business concept. Returns one reaction per segment, in order.

    With grounded=False the personas get only age and gender, as a generic-LLM baseline.
    """
    system = (
        "You simulate distinct local customers reacting to a new business concept. Judge strictly from each "
        "persona's profile; personas must differ where their profiles differ. Reply as JSON: "
        '{"reactions": [{"persona": 1, "visit_likelihood": 0-10, "reason": "one sentence", '
        '"dealbreaker": "one short phrase or null"}]} with one entry per persona.'
    )
    personas = "\n\n".join(_persona(i, s, area, grounded) for i, s in enumerate(segments, 1))
    out = chat_json(system, f"CONCEPT\n{concept}\n\nPERSONAS\n{personas}")
    by_id = {r["persona"]: r for r in out["reactions"]}
    return [by_id[i] for i in range(1, len(segments) + 1)]


if __name__ == "__main__":
    import sys

    for s in build_segments(sys.argv[1] if len(sys.argv) > 1 else "Indiranagar, Bangalore"):
        print(f"\n{s['age']} / {s['gender']} ({s['share']:.0%} of venues): {', '.join(s['venues'][:5])}")
        for t, names in s["taste"].items():
            print(f"  {t}: {', '.join(names)}")
        print(f"  tags: {', '.join(s['tags'])}")
