"""Benchmark the three proposed retrieval pipelines on the isolation harness.

Each pipeline is implemented as a retriever variant over the existing Redis KB
index and scored with the same evaluate() loop as eval_retrieval.py (gold =
required_documents). LLM outputs (decompositions, rewrites) are cached to disk
so reruns are deterministic and free.

Pipelines (doc-retrieval-faithful slice of each MD proposal):
  baseline : current single-stage vector search (kb_search_vector).
  B        : vector seeds + co-occurrence-graph expansion. Edges mined from the
             TRAIN gold sets (Approach B's "task-derived co-occurrence edges").
             NOTE: scoring B on train is leaky (graph built from train gold);
             the fair number is the TEST split.
  C        : LLM query decomposition into sub-intents -> one vector search per
             sub-intent -> round-robin merge (Approach C Phase 2). Attacks the
             multi-doc recall gap directly.
  D        : LLM query rewrite into canonical banking terminology + synonyms,
             then vector search (Approach D, query-expansion slice).

What this canNOT measure: B/C's tool dependency-graph traversal (unlock->call)
— that is tool-retrieval-specific and needs a tool index scored against
gold_tools. Documented as the next milestone.

Usage (inside cs-agent container; needs Redis index + Google creds):
    python eval_pipelines.py --split test
    python eval_pipelines.py --split train --pipelines baseline,C,D
"""

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from eval_retrieval import load_split, load_tasks
from rag_tools import DOC_PREFIX, _get_genai_client, kb_search_vector

CACHE_DIR = Path(os.environ.get("EVAL_CACHE_DIR", "/app/_eval/cache"))
LLM_MODEL = os.environ.get("EVAL_LLM_MODEL", "gemini-3.5-flash")


# ---------------------------------------------------------------- LLM helpers
def _gen(prompt: str, retries: int = 4) -> str:
    """One generate call with simple backoff for 429s on the shared key."""
    client = _get_genai_client()
    delay = 2.0
    for attempt in range(retries):
        try:
            r = client.models.generate_content(model=LLM_MODEL, contents=prompt)
            return (r.text or "").strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"[gen] retry {attempt+1} after {type(e).__name__}", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
    return ""


def _cache(name: str) -> dict:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = CACHE_DIR / name
    return json.loads(p.read_text()) if p.exists() else {}


def _cache_save(name: str, data: dict) -> None:
    (CACHE_DIR / name).write_text(json.dumps(data, indent=0))


# ----------------------------------------------------------- pipeline pieces
def _wrap(ids: list[str]) -> list[dict]:
    """doc ids (no prefix) -> hit dicts shaped like rag_tools output."""
    return [{"doc_id": f"{DOC_PREFIX}{i}"} for i in ids]


def _strip(hits: list[dict]) -> list[str]:
    out = []
    for h in hits:
        d = h.get("doc_id", "")
        out.append(d[len(DOC_PREFIX):] if d.startswith(DOC_PREFIX) else d)
    return out


def build_cooccurrence_graph(train_tasks: list[dict]) -> dict:
    """A<->B edge weighted by how often docs co-occur in a task's gold set."""
    edges: dict[str, Counter] = defaultdict(Counter)
    for t in train_tasks:
        gold = list(t["gold_docs"])
        for a in gold:
            for b in gold:
                if a != b:
                    edges[a][b] += 1
    return {k: dict(v) for k, v in edges.items()}


def decompose(query: str, cache: dict) -> list[str]:
    """Split a scenario into distinct retrieval intents (cached)."""
    if query in cache:
        return cache[query]
    prompt = (
        "A bank customer's message is below. List the DISTINCT information-"
        "retrieval intents as short search queries (banking product/policy "
        "lookups). If there is only one intent, return one. Reply with ONLY a "
        "JSON array of strings.\n\nMESSAGE:\n" + query[:4000]
    )
    raw = _gen(prompt)
    try:
        s = raw[raw.index("["): raw.rindex("]") + 1]
        subs = [x for x in json.loads(s) if isinstance(x, str) and x.strip()]
    except Exception:
        subs = []
    subs = subs or [query]
    cache[query] = subs
    return subs


def rewrite(query: str, cache: dict) -> str:
    """Rewrite into a concise canonical-terminology retrieval query (cached)."""
    if query in cache:
        return cache[query]
    prompt = (
        "Rewrite this bank customer's message as a CONCISE retrieval query "
        "using canonical banking terminology, and append 5-8 relevant synonyms/"
        "related terms (product names, jargon). Reply with ONLY the query "
        "string.\n\nMESSAGE:\n" + query[:4000]
    )
    out = _gen(prompt) or query
    cache[query] = out
    return out


def merge_roundrobin(per_query: list[list[dict]], cap: int) -> list[dict]:
    """Interleave ranked lists by rank, dedup, keep first `cap`+buffer."""
    seen, out = set(), []
    for rank in range(max((len(p) for p in per_query), default=0)):
        for lst in per_query:
            if rank < len(lst):
                d = lst[rank]["doc_id"]
                if d not in seen:
                    seen.add(d)
                    out.append(lst[rank])
    return out


# ---------------------------------------------------------------- pipelines
def make_baseline():
    return lambda q, top_k: kb_search_vector(q, top_k)


def make_B(graph: dict, seed_k: int = 3, nbrs: int = 3):
    """Vector seeds, then INSERT each seed's co-occurrence neighbors right after
    it (pushing graph-related docs up into the top-k window), then backfill with
    the remaining vector results. dedup, preserve order."""
    def retrieve(q, top_k):
        base = _strip(kb_search_vector(q, top_k))
        out, seen = [], set()
        for i, s in enumerate(base):
            if s not in seen:
                seen.add(s)
                out.append(s)
            if i < seed_k:  # graft this strong seed's neighbors into the window
                for nbr, _ in Counter(graph.get(s, {})).most_common(nbrs):
                    if nbr not in seen:
                        seen.add(nbr)
                        out.append(nbr)
        return _wrap(out)
    return retrieve


def make_C(cache: dict):
    def retrieve(q, top_k):
        subs = decompose(q, cache)
        per = [kb_search_vector(s, top_k) for s in subs]
        return merge_roundrobin(per, top_k)
    return retrieve


def make_D(cache: dict):
    return lambda q, top_k: kb_search_vector(rewrite(q, cache), top_k)


# ---------------------------------------------------------------- scoring
def evaluate(tasks: list[dict], retriever, ks: list[int]) -> dict:
    max_k = max(ks)
    rows = []
    for t in tasks:
        ranked = _strip(retriever(t["query"], max_k))
        gold = t["gold_docs"]
        first = next((i + 1 for i, d in enumerate(ranked) if d in gold), None)
        rows.append({
            "recall": {k: len(set(ranked[:k]) & gold) / len(gold) for k in ks},
            "hit": {k: bool(set(ranked[:k]) & gold) for k in ks},
            "first": first,
        })
    n = len(rows)
    return {
        "n": n,
        "recall@k": {k: round(sum(r["recall"][k] for r in rows) / n, 3) for k in ks},
        "hit@k": {k: round(sum(r["hit"][k] for r in rows) / n, 3) for k in ks},
        "mrr": round(sum(1 / r["first"] for r in rows if r["first"]) / n, 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", default="1,3,5,10")
    ap.add_argument("--pipelines", default="baseline,B,C,D")
    args = ap.parse_args()
    ks = sorted(int(x) for x in args.k.split(","))
    tasks = load_tasks(args.split)

    graph = build_cooccurrence_graph(load_tasks("train"))
    dec_cache = _cache("decompose.json")
    rew_cache = _cache("rewrite.json")
    builders = {
        "baseline": lambda: make_baseline(),
        "B": lambda: make_B(graph),
        "C": lambda: make_C(dec_cache),
        "D": lambda: make_D(rew_cache),
    }
    wanted = [p.strip() for p in args.pipelines.split(",")]

    print(f"split={args.split}  n={len(tasks)}  k={ks}  model={LLM_MODEL}\n", file=sys.stderr)
    results = {}
    for name in wanted:
        t0 = time.perf_counter()
        agg = evaluate(tasks, builders[name](), ks)
        results[name] = agg
        leak = "  [LEAKY on train]" if (name == "B" and args.split == "train") else ""
        print(f"=== {name} (n={agg['n']}, {time.perf_counter()-t0:.0f}s){leak} ===")
        print("  recall@k:", "  ".join(f"@{k}={agg['recall@k'][k]}" for k in ks))
        print("  hit@k:   ", "  ".join(f"@{k}={agg['hit@k'][k]}" for k in ks))
        print(f"  MRR:      {agg['mrr']}\n")
        _cache_save("decompose.json", dec_cache)
        _cache_save("rewrite.json", rew_cache)

    # compact comparison line on the headline metrics
    print("=== summary (recall@5 / hit@5 / MRR) ===")
    for name in wanted:
        a = results[name]
        print(f"  {name:9s} recall@5={a['recall@k'].get(5,'-')}  hit@5={a['hit@k'].get(5,'-')}  MRR={a['mrr']}")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
