from typing import Any
import flashrank
from flashrank import Ranker, RerankRequest
from settings import get_settings, get_logger

logger = get_logger(__name__)


def create_ranker() -> Ranker:
    """Create and configure a Flashrank reranker model."""
    settings = get_settings()

    # Configure flashrank model file mappings
    flashrank.Config.model_file_map["mmarco-mMiniLM-L12-H384-v1"] = "flashrank-mmarco-mMiniLM-L12.onnx"
    flashrank.Config.model_file_map["gte-multilingual-reranker-base"] = "flashrank-gte-multilingual-base.onnx"

    logger.info(f"Loading ranker model: {settings.ranker_model}")
    model = Ranker(model_name=settings.ranker_model, cache_dir=settings.ranker_cache_dir)

    return model


def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    embedding_results: list[dict[str, Any]],
    k: int = 10,
    k_rrf: int = 60
) -> list[dict[str, Any]]:
    """
    Combine BM25 and embedding results using Reciprocal Rank Fusion (RRF).

    Args:
        bm25_results: Results from BM25 search (must have 'rank' field)
        embedding_results: Results from embedding search (must have 'rank' field)
        k: Number of top results to return
        k_rrf: RRF constant (default 60)

    Returns:
        Combined and reranked results
    """
    doc_lookup: dict[int, dict[str, Any]] = dict()
    for entry in bm25_results:
        docid = entry["docid"]
        doc_lookup[docid] = {"bm25_rank": entry.get("rank"), "emb_rank": None, "data": entry}

    for entry in embedding_results:
        docid = entry["docid"]
        if docid not in doc_lookup:
            doc_lookup[docid] = {"bm25_rank": None, "emb_rank": None, "data": entry}
        doc_lookup[docid]["emb_rank"] = entry.get("rank")

    def _rrf(docid: int) -> float:
        # Documento faltante: Penalidade de 1.5 * k
        bm25_r = doc_lookup[docid]["bm25_rank"] if doc_lookup[docid]["bm25_rank"] is not None else 3/2 * k
        emb_r = doc_lookup[docid]["emb_rank"] if doc_lookup[docid]["emb_rank"] is not None else 3/2 * k
        return 1 / (k_rrf + bm25_r) + 1 / (k_rrf + emb_r)

    hybrid_scores = []
    for docid, doc_info in doc_lookup.items():
        entry = doc_info["data"]
        hybrid_scores.append({
            "docid": docid,
            "texto": entry.get("texto", entry.get("text", "")),
            "tema": entry.get("tema"),
            "subtema": entry.get("subtema"),
            "enunciado": entry.get("enunciado"),
            "excerto": entry.get("excerto"),
            "score": _rrf(docid),
            "variant": f"rrf - {entry.get('variant', '')}",
        })

    # Sort by RRF score (descending)
    hybrid_scores.sort(key=lambda x: x["score"], reverse=True)

    # Add rank and return top k
    for idx, item in enumerate(hybrid_scores[:k], start=1):
        item["rank"] = idx

    return hybrid_scores[:k]


def rerank_results(
    query: str,
    bm25_results: list[dict[str, Any]],
    embedding_results: list[dict[str, Any]],
    ranker: Ranker,
    k: int = 10
) -> list[dict[str, Any]]:
    """
    Rerank combined BM25 and embedding results using a neural reranker.

    Args:
        query: The search query
        bm25_results: Results from BM25 search
        embedding_results: Results from embedding search
        ranker: Flashrank Ranker instance
        k: Number of top results to return

    Returns:
        Reranked results
    """
    # Combine results and remove duplicates
    all_results: dict[str, dict[str, Any]] = {}
    for entry in bm25_results + embedding_results:
        docid = entry["docid"]
        # potencialmente sobrescreve o bm25
        if docid not in all_results:
            all_results[docid] = entry

    passages = []
    for docid, entry in all_results.items():
        text = entry.get("texto") or entry.get("text") or ""
        # if not text and entry.get("enunciado"):
        #     text = entry.get("enunciado", "")
        passages.append({
            "id": docid,
            "text": text,
            "meta": entry
        })

    if not passages:
        logger.warning("No passages to rerank")
        return []

    # Rerank
    logger.info(f"Reranking {len(passages)} passages")
    rerank_request = RerankRequest(query=query, passages=passages)
    reranked = ranker.rerank(rerank_request)

    # Format results
    results = []
    for idx, item in enumerate(reranked[:k], start=1):
        meta = item.get("meta", {})
        results.append({
            "rank": idx,
            "docid": item.get("id"),
            "score": float(item.get("score", 0.0)),  # Convert numpy float32 to Python float
            "texto": item.get("text", ""),
            "tema": meta.get("tema"),
            "subtema": meta.get("subtema"),
            "enunciado": meta.get("enunciado"),
            "excerto": meta.get("excerto"),
            "variant": f"reranker - {meta.get('variant')}",
        })

    return results
