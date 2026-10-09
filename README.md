# LocalTaste

**Know the neighbourhood before you sign the lease.**

Describe the business you want to open and where. LocalTaste's agent reads the area's taste with [Qloo](https://www.qloo.com/), studies the venues you would compete with, and writes a launch brief (menu, music, decor, partners, pricing, events, risks) in which every idea points to the Qloo signals behind it. A toggle shows the same brief written by the same model without Qloo.

Built for the Qloo Agentic Hackathon.

## How it uses Qloo

Qloo's neighbourhood-level location signal is sparse, so LocalTaste derives an area's taste from the audiences of its venues. The agent has seven tools, all backed by the Qloo API:

| Tool | Qloo query | Purpose |
|---|---|---|
| `area_taste` | Places filtered by location, then those venues as `signal.interests.entities` for artists, brands, films, TV and typed tags; city-level `signal.location` for artists | What this neighbourhood's audience favours |
| `find_tags` | `/v2/tags` search | Map a concept ("wine bar") to Qloo tag ids |
| `competitors` | Places filtered by location and tags | Comparable venues with rating, price tier, popularity and what they are known for |
| `audience_taste` | Competitor ids as signals into another domain | What the competitors' audience also loves |
| `venues_for_taste` | Entity signals ranked over places in the area | Where fans of an inspiration already go |
| `lookup` | `/search` | Resolve a named venue, brand or artist |
| `city_taste` | City-level `signal.location` | City-wide context |

## How the agent works

1. **Research.** A planning model chooses which Qloo tools to call. Code enforces a checklist (area taste, competitors, two audience domains) before it may stop.
2. **Ledger.** Every tool result is recorded, with each returned name typed as venue, artist, brand, film, show, genre, food or ambience.
3. **Writing.** A second model writes the brief from the ledger only, citing the names each idea rests on.
4. **Verification in code.** Any cited name Qloo did not return is removed. An idea is dropped unless it cites the right kind of signal (music needs artists or genres, partnerships need brands, pricing needs competitors).
5. **Comparison.** The same writing model answers with no tools, for the side-by-side view.

## Run it locally

Requires Python 3.11+.

```bash
git clone https://github.com/ItzAditya43/localtaste.git
cd localtaste
pip install -r backend/requirements.txt
cp .env.example .env        # then add your keys
cd backend
uvicorn server:app --port 8000
```

Open http://localhost:8000. The four saved examples load without any API calls.

| Variable | Purpose |
|---|---|
| `QLOO_API_KEY` | Qloo hackathon key |
| `QLOO_API_URL` | Defaults to `https://hackathon.api.qloo.com` |
| `GROQ_API_KEY` | Groq key for both models |
| `GROQ_MODEL`, `GROQ_RESEARCH_MODEL` | Optional model overrides |

From the command line: `python agent.py "A natural wine bar with small plates" "Bandra West, Mumbai"`.

## Evaluation

`python eval_brief.py out.json` runs four concept and city cases and reports tool calls, signals cited, run time, an audit for unsupported factual claims, and how many of each brief's named competitors Qloo can find in that city.

In the last run, all 16 competitors in the grounded briefs were found (they come from Qloo), against 7 to 9 of 16 for the model without Qloo across runs. "Not found" includes real venues Qloo lacks, so this is not a hallucination rate.

## What it does not do

LocalTaste describes who an area's audience is and what they favour. It does not predict whether a business will succeed. During development we tested whether taste fit predicts a venue's popularity or rating, and whether simulated local personas match a venue's real audience; neither held up, so neither is in the product. `panel.py` and `eval_audience.py` are kept as the record of those tests.

Other limits: results are thinner outside major cities, and the free LLM tier allows one live run at a time.

## Layout

```
backend/qloo.py        Qloo client with caching and retry
backend/agent.py       Tools, research loop, brief writer, verification
backend/llm.py         Groq client
backend/server.py      FastAPI server with streaming
backend/runs/          Saved example runs
backend/eval_brief.py  Brief evaluation
frontend/index.html    Web app
```

## Licence

MIT. See [LICENSE](LICENSE).
