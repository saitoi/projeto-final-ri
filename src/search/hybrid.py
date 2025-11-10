# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb",
#     "sentence-transformers",
#     "bm25s",
#     "nltk",
#     "pydantic-settings",
# ]
# ///

import sys
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import EmbeddingVariant, get_logger

# Import from search modules
if __name__ == "__main__":
    from search.generate_bm25 import query_bm25, build_bm25
    from search.generate_embeddings import query_embeddings
    from search.utils import create_embedding_model
else:
    from .generate_bm25 import query_bm25, build_bm25
    from .generate_embeddings import query_embeddings
    from .utils import create_embedding_model

import duckdb

logger = get_logger(__name__)


def normalize_scores(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize scores to [0, 1] range using min-max normalization."""
    if not results:
        return results

    scores = [r["score"] for r in results]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        for r in results:
            r["score"] = 1.0
    else:
        for r in results:
            r["score"] = (r["score"] - min_score) / (max_score - min_score)

    return results


def query_hybrid(
    query: str,
    k: int,
    bm25_variant: str,
    embedding_variant: EmbeddingVariant,
    db_filepath: str,
    bm25_dir: str,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """
    Perform hybrid search combining BM25 and embeddings using RRF.

    Args:
        query: Search query
        k: Number of results to return
        bm25_variant: BM25 variant to use
        embedding_variant: Embedding model variant to use
        db_filepath: Path to DuckDB database
        bm25_dir: Directory containing BM25 models
        k1: BM25 k1 parameter
        b: BM25 b parameter

    Returns:
        List of hybrid search results
    """
    logger.info("Performing hybrid search...")
    logger.info(f"BM25 variant: {bm25_variant}, Embedding variant: {embedding_variant}")

    # Get BM25 results
    conn = duckdb.connect(db_filepath, read_only=True)
    try:
        retriever = build_bm25(
            conn=conn,
            model_dir=bm25_dir,
            variant=bm25_variant,
            build=False,
            k1=k1,
            b=b,
        )
    finally:
        conn.close()

    bm25_results = query_bm25(retriever=retriever, query=query, k=k)
    logger.info(f"BM25 returned {len(bm25_results)} results")

    # Get embedding results
    model = create_embedding_model(variant=embedding_variant)
    emb_results = query_embeddings(
        query=query,
        k=k,
        model=model,
        variant=embedding_variant,
        db_filepath=db_filepath,
    )
    logger.info(f"Embeddings returned {len(emb_results)} results")

    # Normalize scores
    bm25_results = normalize_scores(bm25_results)
    emb_results = normalize_scores(emb_results)

    # Combine using simple score fusion (average)
    doc_scores: dict[int, float] = {}
    doc_data: dict[int, dict[str, Any]] = {}

    for result in bm25_results:
        docid = result["docid"]
        doc_scores[docid] = result["score"] * 0.5  # Weight 0.5 for BM25
        doc_data[docid] = result

    for result in emb_results:
        docid = result["docid"]
        if docid in doc_scores:
            doc_scores[docid] += result["score"] * 0.5  # Weight 0.5 for embeddings
        else:
            doc_scores[docid] = result["score"] * 0.5
            doc_data[docid] = result

    # Sort by combined score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Build final results
    hybrid_results = []
    for rank, (docid, score) in enumerate(sorted_docs[:k], start=1):
        result = doc_data[docid].copy()
        result["rank"] = rank
        result["score"] = score
        hybrid_results.append(result)

    logger.info(f"Hybrid search returned {len(hybrid_results)} results")
    return hybrid_results
