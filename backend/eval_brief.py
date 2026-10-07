"""Rates the agent's briefs on measurable grounds: are cited signals real, and are briefs area-specific?"""
import json
import sys
import time

import agent

CASES = [
    ("A specialty coffee shop and vinyl listening room", "Williamsburg, Brooklyn"),
    ("A specialty coffee shop and vinyl listening room", "Shoreditch, London"),
    ("A specialty coffee shop and vinyl listening room", "Indiranagar, Bangalore"),
    ("A natural wine bar with small plates", "Bandra West, Mumbai"),
]


def rate(out):
    brief, ledger = out["brief"], out["ledger"]
    known, cites = agent.known_names(ledger), agent.cited(brief)
    recs = brief.get("recommendations", [])
    return {
        "tool_calls": len(ledger),
        "tools_used": sorted({s["tool"] for s in ledger}),
        "tool_errors": sum("error" in s["result"] for s in ledger if isinstance(s["result"], dict)),
        "recommendations": len(recs),
        "areas": sorted({r.get("area") for r in recs}),
        "competitors": len(brief.get("competitors", [])),
        "unsupported_risks": sum(not r.get("signals") for r in brief.get("risks", [])),
        "cited": len(cites),
        "cited_examples": cites[:6],
    }


if __name__ == "__main__":
    results = []
    for concept, location in CASES:
        t = time.time()
        out = agent.run(concept, location)
        r = rate(out)
        r["seconds"] = round(time.time() - t)
        results.append((concept, location, out, r))
        print(f"\n## {concept} @ {location}\n{json.dumps(r, ensure_ascii=False)}", flush=True)
    same = [set(c.lower() for c in agent.cited(o["brief"])) for c_, _, o, _ in results[:3]]
    for i, j in ((0, 1), (0, 2), (1, 2)):
        print(f"signal overlap {results[i][1]} vs {results[j][1]}: {len(same[i] & same[j])} shared of {len(same[i] | same[j])}")
    json.dump([{"concept": c, "location": l, **o, "rating": r} for c, l, o, r in results], open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"), ensure_ascii=False, indent=1)
