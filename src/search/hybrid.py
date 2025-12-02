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
    from search.utils_hybrid import fusion_results, rerank_results, create_ranker
else:
    from .generate_bm25 import query_bm25, build_bm25
    from .generate_embeddings import query_embeddings
    from .utils import create_embedding_model
    from .utils_hybrid import fusion_results, rerank_results, create_ranker

import duckdb

logger = get_logger(__name__)


# Não é necessário
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
    hybrid_variant: str = "fusion",
    fusion_method: str = "rrf",
    fusion_norm: str = "min-max",
    fusion_k: int = 60,
) -> list[dict[str, Any]]:
    """
    Perform hybrid search combining BM25 and embeddings using fusion or reranking.

    Args:
        query: Search query
        k: Number of results to return
        bm25_variant: BM25 variant to use
        embedding_variant: Embedding model variant to use
        db_filepath: Path to DuckDB database
        bm25_dir: Directory containing BM25 models
        k1: BM25 k1 parameter
        b: BM25 b parameter
        hybrid_variant: Hybrid method to use ("fusion" or "ranker")
        fusion_method: Fusion method from ranx (rrf, min, max, sum, etc.)
        fusion_norm: Normalization method (min-max, max, sum, zmuv, rank, borda)
        fusion_k: K parameter for fusion algorithms (e.g., RRF constant)

    Returns:
        List of hybrid search results
    """
    logger.info("Performing hybrid search...")
    logger.info(f"BM25 variant: {bm25_variant}, Embedding variant: {embedding_variant}")
    logger.info(f"Hybrid variant: {hybrid_variant}")

    if hybrid_variant == "fusion":
        logger.info(f"Fusion method: {fusion_method}, Norm: {fusion_norm}, K: {fusion_k}")
    elif hybrid_variant == "ranker":
        logger.info("Using neural reranker for hybrid search")

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

    # Combine results based on hybrid variant
    if hybrid_variant == "fusion":
        # Combine using ranx fusion
        hybrid_results = fusion_results(
            bm25_results=bm25_results,
            embedding_results=emb_results,
            k=k,
            method=fusion_method,
            norm=fusion_norm,
            fusion_k=fusion_k
        )
    elif hybrid_variant == "ranker":
        # Combine using neural reranker
        ranker = create_ranker()
        hybrid_results = rerank_results(
            query=query,
            bm25_results=bm25_results,
            embedding_results=emb_results,
            ranker=ranker,
            k=k
        )
    else:
        raise ValueError(f"Unknown hybrid variant: {hybrid_variant}")

    logger.info(f"Hybrid search returned {len(hybrid_results)} results")
    return hybrid_results
