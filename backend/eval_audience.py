"""Audience-recovery test: can the panel tell who a real venue is for?

Real venues the panel has never seen are described as anonymous concepts. Each persona's
visit likelihood is compared with the venue's actual age and gender skew from Qloo, for
Qloo-grounded personas versus personas given only age and gender.
"""
import json
import statistics as st
import sys

import panel
import qloo


def _actual(demo, seg):
    gender = {"male": demo["gender"]["male"], "female": demo["gender"]["female"], "mixed": 0}[seg["gender"]]
    return demo["age"][seg["age"]] + gender


def _pearson(a, b):
    ma, mb = st.mean(a), st.mean(b)
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else float("nan")


def run(area, n=12):
    segments = panel.build_segments(area)
    seen = {i for s in segments for i in s["venue_ids"]}
    held = [p for p in qloo.places(area, take=40, page=2) if p["id"] not in seen and p["description"]]
    demo = qloo.demographics([p["id"] for p in held])
    held = [p for p in held if p["id"] in demo][:n]
    rows = []
    for p in held:
        concept = f"{p['description']} Price level: {p['price_level'] or 'unknown'}. Features: {', '.join(p['tags'][:25])}."
        actual = [_actual(demo[p["id"]], s) for s in segments]
        row = {"venue": p["name"], "actual": actual}
        for mode, grounded in (("grounded", True), ("generic", False)):
            row[mode] = [r["visit_likelihood"] for r in panel.react(concept, segments, area, grounded)]
        rows.append(row)
        print(json.dumps(row), flush=True)

    print(f"\n## {area}: {len(rows)} held-out venues, {len(segments)} segments (chance top-1 = {1 / len(segments):.0%})")
    for mode in ("grounded", "generic"):
        pred, act, hits = [], [], 0
        for r in rows:
            mp, ma = st.mean(r[mode]), st.mean(r["actual"])
            pred += [x - mp for x in r[mode]]
            act += [x - ma for x in r["actual"]]
            hits += r[mode].index(max(r[mode])) == r["actual"].index(max(r["actual"]))
        print(f"   {mode:9s} correlation with real audience skew: {_pearson(pred, act):+.2f} | top-1 hit rate {hits / len(rows):.0%}")


if __name__ == "__main__":
    for a in sys.argv[1:] or ["Williamsburg, Brooklyn"]:
        run(a)
