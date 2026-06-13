"""Knowledge-base search tools backed by Redis (RediSearch).

kb_search_bm25: full-text BM25 search (OR-semantics keyword query).
kb_search_vector: HNSW vector search over gemini-embedding-001 embeddings
(available only when the index was built with embeddings).

Replies are parsed via execute_command so both the classic array reply and
the Redis 8 map-style reply work regardless of redis-py version."""

import json
import os
import re
import struct
import sys
import time

import redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
KB_INDEX = "kb_idx"
DOC_PREFIX = "doc:"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768

# Co-occurrence graph for graph-expanded retrieval (Approach B, the standard
# pipeline — see cs_agent/eval_results/PIPELINE_DECISION.md). Built offline from
# the TRAIN split by precompute_cooccurrence.py and shipped in the image.
COOCCURRENCE_GRAPH_PATH = os.environ.get(
    "COOCCURRENCE_GRAPH_PATH", "/app/kb/cooccurrence_graph.json"
)
GRAPH_SEED_K = int(os.environ.get("KB_GRAPH_SEED_K", "3"))  # seeds whose neighbors graft in
GRAPH_NBRS = int(os.environ.get("KB_GRAPH_NBRS", "3"))      # neighbors grafted per seed

_client = redis.Redis.from_url(REDIS_URL, decode_responses=False)
_genai_client = None
_cooccurrence_graph = None

# Opt-in RAG tracing: set A2A_HACK_TRACE=1 to log every kb_search to stderr
# (visible in `docker compose logs cs-agent`). Off by default so marked runs
# stay quiet and unaffected.
_TRACE = os.environ.get("A2A_HACK_TRACE", "").lower() in ("1", "true", "yes", "on")


def _trace(message: str) -> None:
    if _TRACE:
        print(f"[rag] {message}", file=sys.stderr, flush=True)


def _summarize(docs: list[dict]) -> list[str]:
    """Compact one-line view of search hits: doc_id (+ score for vector)."""
    summary = []
    for doc in docs:
        doc_id = doc.get("doc_id", "?")
        if "score" in doc:
            summary.append(f"{doc_id}(score={doc['score']})")
        else:
            summary.append(doc_id)
    return summary


def _get_genai_client():
    """Reused genai client (one connection pool, not a new one per search)."""
    global _genai_client
    if _genai_client is None:
        from google import genai

        _genai_client = genai.Client()
    return _genai_client


def _embed(texts: list[str]) -> list[list[float]]:
    """Embed texts with gemini-embedding-001 via google-genai."""
    from google.genai import types

    # Reduced-dim output is unnormalized; the index uses COSINE, so that's fine.
    result = _get_genai_client().models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(output_dimensionality=EMBEDDING_DIM),
    )
    return [e.values for e in result.embeddings]


def _decode(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _parse_search_reply(reply) -> list[dict]:
    """Normalize an FT.SEARCH reply (array or map shape) to result dicts."""
    if isinstance(reply, dict):
        results = reply.get(b"results", reply.get("results")) or []
        out = []
        for row in results:
            attrs = row.get(b"extra_attributes", row.get("extra_attributes")) or {}
            doc = {"doc_id": _decode(row.get(b"id", row.get("id", "")))}
            doc.update({_decode(k): _decode(v) for k, v in attrs.items()})
            out.append(doc)
        return out
    out = []
    for i in range(1, len(reply) - 1, 2):
        doc = {"doc_id": _decode(reply[i])}
        fields = reply[i + 1]
        for j in range(0, len(fields) - 1, 2):
            doc[_decode(fields[j])] = _decode(fields[j + 1])
        out.append(doc)
    return out


def _strip_score(docs: list[dict]) -> list[dict]:
    for doc in docs:
        doc.pop("score", None)
    return docs


def kb_search_bm25(query: str, top_k: int = 5) -> list[dict]:
    """Full-text (BM25) search over the Rho-Bank knowledge base.

    Args:
        query: Keywords or a short phrase to search for. Matching is ranked,
            so extra keywords help rather than hurt.
        top_k: Number of documents to return.

    Returns:
        Matching documents with doc_id, title, and full content.
    """
    terms = re.findall(r"\w+", query.lower())
    if not terms:
        _trace(f"bm25 query={query!r} -> no searchable terms, returning []")
        return []
    # OR-join: RediSearch defaults to AND, which zeroes out long queries.
    or_query = "|".join(dict.fromkeys(terms))
    start = time.perf_counter()
    reply = _client.execute_command(
        "FT.SEARCH", KB_INDEX, or_query,
        "LIMIT", "0", str(top_k),
        "RETURN", "2", "title", "content",
    )
    docs = _parse_search_reply(reply)
    _trace(
        f"bm25 query={query!r} top_k={top_k} hits={len(docs)} "
        f"elapsed_ms={(time.perf_counter() - start) * 1000:.0f} "
        f"docs={_summarize(docs)}"
    )
    return docs


def kb_search_vector(query: str, top_k: int = 5) -> list[dict]:
    """Semantic (vector) search over the Rho-Bank knowledge base.

    Better than kb_search_bm25 when the query is a natural-language question
    rather than exact keywords.

    Args:
        query: A natural-language question or description.
        top_k: Number of documents to return.

    Returns:
        Matching documents with doc_id, title, and full content; or an error
        entry telling you to fall back to kb_search_bm25.
    """
    start = time.perf_counter()
    try:
        vector = struct.pack(f"{EMBEDDING_DIM}f", *_embed([query])[0])
        reply = _client.execute_command(
            "FT.SEARCH", KB_INDEX, f"*=>[KNN {top_k} @embedding $vec AS score]",
            "PARAMS", "2", "vec", vector,
            "SORTBY", "score",
            "LIMIT", "0", str(top_k),
            "RETURN", "3", "title", "content", "score",
            "DIALECT", "2",
        )
        docs = _parse_search_reply(reply)
        _trace(
            f"vector query={query!r} top_k={top_k} hits={len(docs)} "
            f"elapsed_ms={(time.perf_counter() - start) * 1000:.0f} "
            f"docs={_summarize(docs)}"
        )
        return _strip_score(docs)
    except Exception as e:
        _trace(
            f"vector query={query!r} FAILED ({type(e).__name__}: {e}) "
            f"elapsed_ms={(time.perf_counter() - start) * 1000:.0f} "
            f"-> instructing fallback to bm25"
        )
        return [
            {
                "error": f"Vector search unavailable ({type(e).__name__}). "
                "Use kb_search_bm25 with keywords instead."
            }
        ]


def _get_graph() -> dict:
    """Lazily load the co-occurrence graph (empty dict if the artifact is
    missing, so retrieval degrades gracefully to plain vector search)."""
    global _cooccurrence_graph
    if _cooccurrence_graph is None:
        try:
            with open(COOCCURRENCE_GRAPH_PATH) as fp:
                _cooccurrence_graph = json.load(fp)
            _trace(f"cooccurrence graph loaded: {len(_cooccurrence_graph)} nodes")
        except (FileNotFoundError, ValueError) as e:
            _trace(f"cooccurrence graph unavailable ({type(e).__name__}); graph expansion off")
            _cooccurrence_graph = {}
    return _cooccurrence_graph


def _fetch_doc(doc_id: str) -> dict:
    """Title + content for a doc id (with DOC_PREFIX), straight from the hash."""
    reply = _client.execute_command("HMGET", doc_id, "title", "content")
    title = _decode(reply[0]) if reply and reply[0] is not None else ""
    content = _decode(reply[1]) if reply and len(reply) > 1 and reply[1] is not None else ""
    return {"doc_id": doc_id, "title": title, "content": content}


def kb_search_graph(query: str, top_k: int = 5) -> list[dict]:
    """Semantic search with co-occurrence expansion (the standard KB search).

    Runs vector search, then grafts each top seed's most-co-occurring documents
    into the result window. This surfaces the *full* set of related documents a
    question needs (e.g. all comparable card/account products), not just the
    single closest match — the validated win over plain vector search.

    Args:
        query: A natural-language question or description.
        top_k: Number of documents to return.

    Returns:
        Matching documents with doc_id, title, and full content; or an error
        entry telling you to fall back to kb_search_bm25.
    """
    seeds = kb_search_vector(query, top_k)
    if seeds and "error" in seeds[0]:
        return seeds  # propagate the fallback-to-bm25 signal unchanged
    graph = _get_graph()
    ranked, seen = [], set()
    for i, hit in enumerate(seeds):
        did = hit["doc_id"]
        if did not in seen:
            seen.add(did)
            ranked.append(hit)
        if i < GRAPH_SEED_K:
            bare = did[len(DOC_PREFIX):] if did.startswith(DOC_PREFIX) else did
            neighbors = sorted(graph.get(bare, {}).items(), key=lambda kv: kv[1], reverse=True)
            for nbr, _w in neighbors[:GRAPH_NBRS]:
                nbr_id = f"{DOC_PREFIX}{nbr}"
                if nbr_id not in seen:
                    seen.add(nbr_id)
                    ranked.append(_fetch_doc(nbr_id))
    ranked = ranked[:top_k]
    _trace(f"graph query={query!r} top_k={top_k} hits={len(ranked)} docs={_summarize(ranked)}")
    return ranked
