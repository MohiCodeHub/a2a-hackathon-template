"""Precompute the co-occurrence graph for graph-expanded retrieval (Approach B).

Mines doc-doc co-occurrence edges from the TRAIN split's task gold sets
(`required_documents`) and writes them to kb/cooccurrence_graph.json. rag_tools
loads that artifact at runtime and `kb_search_graph` grafts a retrieved seed's
co-occurring docs into the top-k window — closing the multi-doc completeness gap
(validated: recall@5 +53% on the held-out test split, see
cs_agent/eval_results/PIPELINE_DECISION.md).

TRAIN-ONLY by construction: edges never see test/feedback gold, so using the
graph at inference is not eval leakage. The build context (Dockerfile) does not
include the harness, so this runs offline and the JSON is committed, exactly
like the kb/embeddings.json pattern.

Stdlib only — run on the host whenever the task files change:

    python cs_agent/precompute_cooccurrence.py
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TASKS_DIR = Path(os.environ.get("TASKS_DIR", REPO / "harness/src/a2a_hack/data/tasks"))
SPLITS_PATH = Path(
    os.environ.get("SPLITS_PATH", REPO / "harness/src/a2a_hack/data/banking_hackathon_splits.json")
)
OUT_PATH = Path(os.environ.get("COOCCURRENCE_GRAPH_PATH", REPO / "kb/cooccurrence_graph.json"))


def build() -> dict:
    """A<->B edge weighted by how often two docs share a train task's gold set.

    Mirrors eval_pipelines.build_cooccurrence_graph so the shipped graph matches
    what the benchmark scored."""
    splits = json.loads(SPLITS_PATH.read_text())
    train = set(splits["train"])
    edges: dict[str, Counter] = defaultdict(Counter)
    n_tasks = 0
    for f in sorted(TASKS_DIR.glob("*.json")):
        t = json.loads(Path(f).read_text())
        if t["id"] not in train:
            continue
        gold = list(t.get("required_documents") or [])
        if len(gold) < 2:
            continue
        n_tasks += 1
        for a in gold:
            for b in gold:
                if a != b:
                    edges[a][b] += 1
    graph = {k: dict(v) for k, v in edges.items()}
    print(
        f"[cooccurrence] {n_tasks} multi-doc train tasks -> {len(graph)} nodes, "
        f"{sum(len(v) for v in graph.values())} directed edges",
        file=sys.stderr,
    )
    return graph


def main() -> None:
    graph = build()
    OUT_PATH.write_text(json.dumps(graph))
    print(f"[cooccurrence] wrote {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
