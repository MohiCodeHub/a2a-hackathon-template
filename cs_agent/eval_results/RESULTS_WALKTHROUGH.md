# Retrieval Pipeline Tests — Results Walkthrough

Plain-language summary of the B/C/D bake-off, for iterating on the pipeline.
Raw numbers: [`pipeline_comparison_test.txt`](pipeline_comparison_test.txt) ·
decision + caveats: [`PIPELINE_DECISION.md`](PIPELINE_DECISION.md).

## Results — held-out TEST split (n=18), identical ground truth

| pipeline | recall@5 | recall@10 | hit@5 | MRR | runtime |
|----------|----------|-----------|-------|-----|---------|
| baseline (vector) | 0.198 | 0.271 | 1.00 | 0.852 | 34s |
| **B (graph)** | **0.302** | **0.413** | 0.944 | 0.843 | **9s** |
| C (decomposition) | 0.162 | 0.262 | 0.778 | 0.632 | 130s |
| D (rewrite/synonyms) | 0.212 | 0.304 | 0.889 | 0.587 | 115s |

## Reading the results

- **The real problem is completeness, not findability.** Baseline already hits
  *a* relevant doc ~100% of the time by k=10, but recall@5 of 0.198 means it
  surfaces only ~1 in 5 of the *full* gold set a task needs (e.g. all four cards
  a customer must compare).
- **B is the only candidate that improves the target metric** (recall@5 +53%,
  recall@10 +52%) at a negligible ranking cost (hit@5 0.944 vs 1.0, MRR 0.843 vs
  0.852) — and it's the fastest, since it adds no LLM call.
- **C made things worse on every metric.** Decomposing mostly-single-intent
  queries dilutes the ranking (hit@1 collapses to 0.5) and adds a ~14× latency
  tax.
- **D bought marginal recall by wrecking ranking** (MRR 0.587) — broadening the
  query surfaces more docs deeper but buries the top hit.

## What B does (one sentence)

B runs ordinary vector search, then grafts in each top result's most-frequently
co-occurring documents (edges mined offline from the training tasks' gold sets),
so it returns the *complete* cluster of related documents a query needs rather
than just the single closest match.

## Why B is SOTA of the tested candidates (one sentence)

It's the only strategy that directly attacks the actual failure mode (multi-doc
completeness) with a learned co-occurrence prior instead of query-side guessing,
so it lifts recall the most while preserving top-rank precision *and* costing
zero extra latency — whereas C and D pay a per-query LLM cost to make ranking
worse.

## Caveat to keep in view

These numbers use the verbose roleplay scenarios as queries (a proxy for real
runtime turns), which likely penalizes C/D — but B wins on the available signal
and its mechanism doesn't depend on query phrasing. The deployment-faithful next
test is tool retrieval (`--target tools` vs `gold_tools`), where B's dependency
edges should pay off even more than on doc retrieval.
