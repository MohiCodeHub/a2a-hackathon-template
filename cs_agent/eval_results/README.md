# Retrieval-Isolation Benchmark — Baseline

Fast feedback loop for tuning tool/doc retrieval **without** running the full
agent harness. For each task we take the user scenario as the query, run *only*
the retriever (`rag_tools.kb_search_vector` / `kb_search_bm25`), and check
whether the task's ground-truth `required_documents` land in the top-k. No LLM,
no env, no agent loop — runs in seconds.

Harness: [`cs_agent/eval_retrieval.py`](../eval_retrieval.py)

## Ground truth (from the task files)

| field | source | what it scores |
|---|---|---|
| query | `user_scenario.instructions` | the prompt |
| gold docs | `required_documents` | current KB-doc retrieval (what `rag_tools` does today) |
| gold tools | action `arguments.agent_tool_name` | the **tool**-retrieval target for the B/C pipelines |

Splits come from `banking_hackathon_splits.json` (train=61, test=18,
feedback=3). **Tune on `train`; leave `test` untouched** — mining edges or
tuning thresholds against `test` is eval leakage.

## How to run

Redis must be up with the KB index built (`docker compose up -d redis cs-agent`).

```bash
cid=$(docker compose ps -q cs-agent)
docker cp cs_agent/eval_retrieval.py "$cid":/app/eval_retrieval.py
docker compose exec -T -e TASKS_DIR=/app/_eval/tasks -e SPLITS_PATH=/app/_eval/splits.json \
  -e A2A_HACK_TRACE=0 cs-agent python /app/eval_retrieval.py --split train --show-failures 8 2>/dev/null
```

(BM25-only needs no Google creds and can run on host via `harness/.venv`.)

## Baseline: current single-stage RAG (doc retrieval)

Raw output: [`baseline_train.txt`](baseline_train.txt) · [`baseline_test.txt`](baseline_test.txt)

**TRAIN (n=61)**

| retriever | hit@1 | hit@5 | hit@10 | recall@5 | MRR |
|---|---|---|---|---|---|
| vector | 0.705 | 0.885 | **1.00** | 0.271 | **0.799** |
| bm25   | 0.279 | 0.738 | 0.836 | 0.197 | 0.498 |

**TEST (n=18, held out)**

| retriever | hit@1 | hit@5 | hit@10 | recall@5 | MRR |
|---|---|---|---|---|---|
| vector | 0.722 | 1.00 | 1.00 | 0.198 | **0.852** |
| bm25   | 0.389 | 0.667 | 0.722 | 0.102 | 0.481 |

### Read of the baseline
- **Vector dominates BM25** on every metric — invest in the semantic path; BM25 is a fallback.
- **hit@10 ≈ 1.0 but recall@5 ≈ 0.2–0.27.** The right doc is almost always *findable*, but multi-doc tasks (e.g. comparing gold/silver/bronze/platinum cards) can't fit the full gold set in top-5. This is the **completeness** gap (the problem COLT / graph-fusion target) — the headline thing the new pipelines should move.
- BM25 misses are overwhelmingly "keyword pulled the generic `_(general)_` doc instead of the specific product doc."

## Caveats (read before trusting the numbers)
1. **Query = full roleplay scenario, not the runtime user turn.** At runtime the
   retriever sees a single conversation message; here it sees the whole persona
   script. These numbers are a *proxy*. For a faithful signal, replace the query
   with the real first user utterance (or replay turns from
   `harness/results/*/simulations`).
2. **This scores KB-doc retrieval (the current system), not tool retrieval.**
   The harness already extracts `gold_tools`; the B/C pipelines should add a
   `--target tools` path (see below) that scores against those instead.

## Extending for the three proposed pipelines (B / C / D)

The harness is the shared yardstick. To benchmark a new pipeline, point
`evaluate()` at your retriever and (for tool retrieval) score against
`gold_tools`:

- **Approach B (Graph RAG-Tool Fusion):** build the tool index + dependency
  graph, add `--target tools`, score `gold_tools`. Watch whether graph expansion
  lifts **recall** (the completeness gap above) on multi-tool tasks.
- **Approach C (Tool Wiki + ToolShed):** same target; the enriched-description /
  decomposition changes should move **hit@1 / MRR**. Measure the per-turn LLM
  calls (decompose, self-reflect) against the latency budget separately.
- **Approach D (lexicon):** offline query expansion; re-run doc retrieval with
  expansion on vs off and confirm **recall@5 / hit@5** rises on colloquial
  phrasings without hurting precision. (Open question: does C's enrichment
  already close this gap? Measure before building.)

Keep every pipeline's results as a new `*.txt` here so comparisons stay honest.
