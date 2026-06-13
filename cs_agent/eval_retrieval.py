"""Retrieval-isolation benchmark: does the retriever surface the right docs?

Fast feedback loop for tuning RAG WITHOUT running the full agent harness.
For each task we take the user scenario as the query, run ONLY the retriever
(kb_search_vector / kb_search_bm25 from rag_tools), and check whether the
task's ground-truth `required_documents` land in the top-k. No LLM, no env,
no agent loop — just embed + KNN/BM25 + set membership.

Ground truth comes straight from the task files:
  query     = user_scenario.instructions
  gold docs = required_documents            (what current KB retrieval targets)
  gold tool = action arguments.agent_tool_name (the B/C tool-retrieval target)

Splits come from banking_hackathon_splits.json so you tune on `train` and
leave `test` untouched (no eval leakage when mining/tuning).

Usage (Redis must be up with the KB index built, e.g. `docker compose up -d
redis cs-agent` once, or run ingest.py):

    REDIS_URL=redis://localhost:6379/0 python eval_retrieval.py            # train split, docs
    python eval_retrieval.py --split test --k 1,3,5,10
    python eval_retrieval.py --show-failures 10
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

from rag_tools import DOC_PREFIX, kb_search_bm25, kb_search_vector

REPO = Path(__file__).resolve().parent.parent
TASKS_DIR = Path(os.environ.get("TASKS_DIR", REPO / "harness/src/a2a_hack/data/tasks"))
SPLITS_PATH = Path(
    os.environ.get("SPLITS_PATH", REPO / "harness/src/a2a_hack/data/banking_hackathon_splits.json")
)


def load_split(split: str) -> set[str]:
    """Task ids for a named split (train/test/feedback), or all if 'all'."""
    splits = json.loads(SPLITS_PATH.read_text())
    if split == "all":
        return {tid for ids in splits.values() for tid in ids}
    if split not in splits:
        sys.exit(f"unknown split {split!r}; have {list(splits)} + 'all'")
    return set(splits[split])


def gold_tools(task: dict) -> set[str]:
    """Discoverable tool names the task expects to be unlocked/called."""
    tools = set()
    for a in (task.get("evaluation_criteria") or {}).get("actions") or []:
        name = a.get("name", "")
        args = a.get("arguments") or {}
        if "agent_tool_name" in args:
            tools.add(args["agent_tool_name"])
        elif name not in ("log_verification", "transfer_to_human_agents", "request_human_agent_transfer"):
            tools.add(name)  # direct tool (e.g. apply_for_credit_card)
    return tools


def load_tasks(split: str) -> list[dict]:
    """Tasks in the split that carry retrieval ground truth, normalized."""
    keep = load_split(split)
    out = []
    for f in sorted(glob.glob(str(TASKS_DIR / "*.json"))):
        t = json.load(open(f))
        if t["id"] not in keep:
            continue
        query = ((t.get("user_scenario") or {}).get("instructions") or "").strip()
        gold_docs = set(t.get("required_documents") or [])
        if not query or not gold_docs:
            continue  # nothing to score against
        out.append(
            {"id": t["id"], "query": query, "gold_docs": gold_docs, "gold_tools": gold_tools(t)}
        )
    return out


def _doc_id(hit: dict) -> str:
    """Strip the 'doc:' index prefix so ids match required_documents."""
    raw = hit.get("doc_id", "")
    return raw[len(DOC_PREFIX):] if raw.startswith(DOC_PREFIX) else raw


def evaluate(tasks: list[dict], retriever, ks: list[int]) -> dict:
    """Run the retriever once per task at max(k) and score every k from the
    same ranked list. Returns aggregate metrics + per-task detail."""
    max_k = max(ks)
    per_task = []
    for t in tasks:
        hits = retriever(t["query"], top_k=max_k)
        if hits and "error" in hits[0]:
            sys.exit(f"retriever error (is Redis up + index built with embeddings?): {hits[0]['error']}")
        ranked = [_doc_id(h) for h in hits]
        gold = t["gold_docs"]
        # rank (1-based) of first gold doc, for MRR
        first = next((i + 1 for i, d in enumerate(ranked) if d in gold), None)
        per_task.append(
            {
                "id": t["id"],
                "ranked": ranked,
                "gold": gold,
                "first_gold_rank": first,
                "recall": {k: len(set(ranked[:k]) & gold) / len(gold) for k in ks},
                "hit": {k: bool(set(ranked[:k]) & gold) for k in ks},
            }
        )
    n = len(per_task)
    agg = {
        "n": n,
        "recall@k": {k: round(sum(p["recall"][k] for p in per_task) / n, 3) for k in ks},
        "hit@k": {k: round(sum(p["hit"][k] for p in per_task) / n, 3) for k in ks},
        "mrr": round(sum(1 / p["first_gold_rank"] for p in per_task if p["first_gold_rank"]) / n, 3),
    }
    return {"agg": agg, "per_task": per_task}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="train", help="train|test|feedback|all (default: train)")
    ap.add_argument("--k", default="1,3,5,10", help="comma-separated cutoffs (default: 1,3,5,10)")
    ap.add_argument("--retriever", default="both", choices=["vector", "bm25", "both"])
    ap.add_argument("--show-failures", type=int, default=0, metavar="N", help="print N worst misses")
    args = ap.parse_args()

    ks = sorted(int(x) for x in args.k.split(","))
    tasks = load_tasks(args.split)
    print(f"split={args.split}  tasks_with_gold={len(tasks)}  k={ks}\n", file=sys.stderr)

    retrievers = (
        [("vector", kb_search_vector), ("bm25", kb_search_bm25)]
        if args.retriever == "both"
        else [(args.retriever, {"vector": kb_search_vector, "bm25": kb_search_bm25}[args.retriever])]
    )

    for name, fn in retrievers:
        res = evaluate(tasks, fn, ks)
        a = res["agg"]
        print(f"=== {name} (n={a['n']}) ===")
        print("  recall@k:", "  ".join(f"@{k}={a['recall@k'][k]}" for k in ks))
        print("  hit@k:   ", "  ".join(f"@{k}={a['hit@k'][k]}" for k in ks))
        print(f"  MRR:      {a['mrr']}")
        if args.show_failures:
            misses = sorted(
                (p for p in res["per_task"] if not p["hit"][max(ks)]),
                key=lambda p: p["id"],
            )[: args.show_failures]
            if misses:
                print(f"  --- {len(misses)} miss(es) @{max(ks)} (no gold doc retrieved) ---")
            for p in misses:
                print(f"    {p['id']}: gold={sorted(p['gold'])[:3]}... got={p['ranked'][:5]}")
        print()


if __name__ == "__main__":
    main()
