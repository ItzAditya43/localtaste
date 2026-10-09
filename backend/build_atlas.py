"""Builds atlas.json: well-known neighbourhoods Qloo can read, with coordinates, used to find taste twins."""
import json
import statistics as st
from pathlib import Path

import qloo
from qloo import _insights

CANDIDATES = [
    "Shoreditch, London", "Soho, London", "Williamsburg, Brooklyn", "Lower East Side, New York", "Le Marais, Paris",
    "Kreuzberg, Berlin", "Shibuya, Tokyo", "Harajuku, Tokyo", "Bandra West, Mumbai", "Indiranagar, Bangalore",
    "Koramangala, Bangalore", "Hauz Khas, Delhi", "Silver Lake, Los Angeles", "Hayes Valley, San Francisco",
    "Wicker Park, Chicago", "Capitol Hill, Seattle", "Fitzroy, Melbourne", "Newtown, Sydney", "Roma Norte, Mexico City",
    "Palermo, Buenos Aires", "Gracia, Barcelona", "Trastevere, Rome", "De Pijp, Amsterdam", "Norrebro, Copenhagen",
    "Hongdae, Seoul", "Tiong Bahru, Singapore", "Kadikoy, Istanbul", "Chiado, Lisbon", "Plateau Mont-Royal, Montreal",
    "East Austin, Austin",
]

if __name__ == "__main__":
    atlas = []
    for name in CANDIDATES:
        resolved = qloo.resolve_location(name)
        ents = _insights(**{"filter.type": "urn:entity:place", "filter.location.query": name, "take": 8}).get("entities", []) if resolved else []
        coords = [(e["location"]["lat"], e["location"]["lon"]) for e in ents if e.get("location")]
        if len(ents) < 8 or not coords or not qloo.same_city(name, resolved):
            print("skipped", name, "->", resolved, len(ents))
            continue
        atlas.append({"name": name, "resolved_as": resolved, "lat": round(st.median(c[0] for c in coords), 4),
                      "lon": round(st.median(c[1] for c in coords), 4)})
        print("ok", name, "->", resolved[:70], atlas[-1]["lat"], atlas[-1]["lon"])
    Path(__file__).with_name("atlas.json").write_text(json.dumps(atlas, ensure_ascii=False, indent=1))
    print(len(atlas), "areas")
