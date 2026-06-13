# Retrieval Pipeline Bake-off — Decision

> **Status: SHIPPED.** Approach B is now the standard KB retrieval pipeline.
> `rag_tools.kb_search_graph` is the CS agent's semantic search tool; the graph
> is built by `cs_agent/precompute_cooccurrence.py` (train-only) and shipped as
> `kb/cooccurrence_graph.json`. Tune via `KB_GRAPH_SEED_K` / `KB_GRAPH_NBRS`.

**Verdict: pursue Approach B (co-occurrence graph expansion).** It is the only
pipeline that improves the target metric (recall / completeness) without
degrading ranking quality, it generalizes to the held-out test split, and it
adds **zero runtime LLM cost**.

Harness: [`cs_agent/eval_pipelines.py`](../eval_pipelines.py) ·
raw output: [`pipeline_comparison_test.txt`](pipeline_comparison_test.txt)

## Results — held-out TEST split (n=18, the fair comparison surface)

| pipeline | recall@5 | recall@10 | hit@5 | hit@10 | MRR | runtime |
|---|---|---|---|---|---|---|
| baseline (vector) | 0.198 | 0.271 | **1.00** | 1.00 | **0.852** | 34s |
| **B — graph expansion** | **0.302** | **0.413** | 0.944 | 1.00 | 0.843 | **9s** |
| C — query decomposition | 0.162 | 0.262 | 0.778 | 0.889 | 0.632 | 130s |
| D — query rewrite/synonyms | 0.212 | 0.304 | 0.889 | 1.00 | 0.587 | 115s |

`recall@5` is the headline metric: baseline `hit@10` is already ~1.0, so the
real problem is **completeness** — getting the *full* gold set (e.g. all four
cards a customer must compare) into the window, not finding *a* relevant doc.

## Why each landed where it did

- **B (+53% recall@5, +52% recall@10)** — mines doc co-occurrence edges from the
  **train** gold sets and grafts a retrieved seed's co-occurring docs into the
  top-k window. When vector search finds one card doc, B pulls its siblings up.
  Tiny ranking cost (hit@5 0.944 vs 1.0, MRR 0.843 vs 0.852) for a large recall
  gain. Edges are built from train and scored on test, so this is the
  **non-leaky** number and it generalizes. No LLM → fastest pipeline.
- **C (worse than baseline on every metric)** — decomposing the message into
  sub-intents and round-robin-merging *dilutes* the ranking: most of these tasks
  are effectively single-category lookups, so 2 of 3 sub-queries surface
  off-target docs that displace the good vector hit (hit@1 0.5 vs 0.722). It also
  costs a per-task LLM call (130s vs 9s). Decomposition is counterproductive when
  the task isn't genuinely multi-intent.
- **D (marginal recall, wrecked ranking)** — query rewrite/synonym expansion
  broadens reach (recall@10 0.304 > 0.271) but buries the top hit (MRR 0.587).
  Net negative for a router that acts on the top results.

## Honest caveats (read before over-trusting this)

1. **Query = full roleplay scenario, not the runtime user turn.** This likely
   *penalizes* C/D (the verbose scenario is noisy to decompose/rewrite) and the
   ranking is a proxy. Re-running with real first-turn utterances (from
   `harness/results/*/simulations`) could shift C/D — but B wins on the available
   signal and its mechanism doesn't depend on query phrasing.
2. **This benchmark scores DOC retrieval, not TOOL retrieval.** It validates B's
   *core* mechanism (co-occurrence graph) and refutes C's *core* differentiator
   (decomposition). It does **not** test B/C's tool *dependency* traversal
   (`unlock → call`) or C's self-reflection — those need a tool index scored
   against `gold_tools` (the documented next milestone). Expectation: B's
   dependency-graph edges should help *more* on tool retrieval than here, since
   tools have real prerequisite structure that docs lack.
3. **B is leaky if scored on train** (graph built from train gold). Always score
   B on a held-out split. The table above is test-only for this reason.

## Recommendation

1. **Adopt B's co-occurrence graph expansion** as the retrieval upgrade.
2. **Drop C's query decomposition and D's rewrite** — both hurt here and cost
   per-turn latency. (C's *self-reflection* and *enriched offline descriptions*
   remain untested and are cheap/offline — worth a separate look, but not the
   per-turn LLM decomposition.)
3. **Next milestone:** build the tool index + `--target tools` path and re-run
   B vs baseline against `gold_tools`, where B's dependency edges should pay off
   most. That is the deployment-faithful test.

## Reproduce

```bash
cid=$(docker compose ps -q cs-agent)
docker cp cs_agent/eval_pipelines.py "$cid":/app/eval_pipelines.py
docker cp cs_agent/eval_results/cache "$cid":/app/_eval/cache   # deterministic LLM cache
docker compose exec -T -e TASKS_DIR=/app/_eval/tasks -e SPLITS_PATH=/app/_eval/splits.json \
  -e A2A_HACK_TRACE=0 cs-agent python /app/eval_pipelines.py --split test 2>/dev/null
```
