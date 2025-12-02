from typing import Any
from rerankers import Reranker
from ranx import Run, fuse
from settings import get_settings, get_logger

logger = get_logger(__name__)


def create_ranker() -> Reranker:
    """Create and configure a reranker model using rerankers library."""
    settings = get_settings()

    logger.info(f"Loading ranker model: {settings.ranker_model}")

    # Use reranker with GPU support
    # trust_remote_code=True is needed for some models like Alibaba GTE
    # batch_size=1 to avoid padding token issues with some models (e.g., Qwen)
    model = Reranker(
        settings.ranker_model,
        model_type='cross-encoder',
        device=f'cuda:{settings.ranker_gpu_id}' if settings.ranker_use_gpu else 'cpu',
        model_kwargs={'trust_remote_code': True},
        batch_size=1  # Process one document at a time to avoid padding token errors
    )

    return model


def fusion_results(
    bm25_results: list[dict[str, Any]],
    embedding_results: list[dict[str, Any]],
    k: int = 10,
    method: str = "rrf",
    norm: str = "min-max",
    fusion_k: int = 60,
    extra_params: dict | None = None
) -> list[dict[str, Any]]:
    """
    Combine BM25 and embedding results using ranx fusion algorithms.

    Args:
        bm25_results: Results from BM25 search (must have 'rank' and 'score' fields)
        embedding_results: Results from embedding search (must have 'rank' and 'score' fields)
        k: Number of top results to return
        method: Fusion method from ranx (rrf, min, max, sum, etc.)
        norm: Normalization method (min-max, max, sum, zmuv, rank, borda)
        fusion_k: K parameter for fusion algorithms (e.g., RRF constant)
        extra_params: Extra parameters for fusion methods (gamma, weights, phi, etc.)

    Returns:
        Combined and reranked results
    """
    settings = get_settings()

    # Build doc metadata lookup
    doc_lookup: dict[int, dict[str, Any]] = {}
    for entry in bm25_results + embedding_results:
        docid = entry["docid"]
        if docid not in doc_lookup:
            doc_lookup[docid] = entry

    # Convert results to ranx Run format
    bm25_run_dict = {}
    emb_run_dict = {}

    # Use a dummy query ID since we're only doing one query at a time
    query_id = "q1"

    for entry in bm25_results:
        docid = str(entry["docid"])
        score = entry.get("score", 1.0 / (entry.get("rank", 1)))
        if query_id not in bm25_run_dict:
            bm25_run_dict[query_id] = {}
        bm25_run_dict[query_id][docid] = score

    for entry in embedding_results:
        docid = str(entry["docid"])
        score = entry.get("score", 1.0 / (entry.get("rank", 1)))
        if query_id not in emb_run_dict:
            emb_run_dict[query_id] = {}
        emb_run_dict[query_id][docid] = score

    # Create Run objects
    bm25_run = Run(bm25_run_dict, name="bm25")
    emb_run = Run(emb_run_dict, name="embedding")

    # Prepare fusion parameters
    params = {}

    # Use extra_params if provided, otherwise use defaults
    if extra_params:
        params = extra_params.copy()
    else:
        if method == "rrf":
            params = {"k": fusion_k}
        elif method == "gmnz":
            # gamma parameter for CombGMNZ (typical range: 0.1 to 2.0)
            params = {"gamma": 1.0}
        elif method in ["wsum", "wmnz", "mixed"]:
            # Equal weights for both retrievers
            params = {"weights": [0.5, 0.5]}
        elif method in ["w_bordafuse", "w_condorcet"]:
            # Equal weights for weighted variants
            params = {"weights": [0.5, 0.5]}
        elif method == "rbc":
            # phi parameter for RBC (typical range: 0.1 to 1.0)
            params = {"phi": 0.5}
        # Methods requiring training (bayesfuse, mapfuse, posfuse, probfuse, segfuse, slidefuse)
        # are not supported without qrels training data
        elif method in ["bayesfuse", "mapfuse", "posfuse", "probfuse", "segfuse", "slidefuse"]:
            logger.warning(f"Method {method} requires training data (qrels), skipping...")
            return []

    # Fuse the runs
    logger.info(f"Fusing results using method={method}, norm={norm}")
    combined_run = fuse(
        runs=[bm25_run, emb_run],
        norm=norm,
        method=method,
        params=params if params else None
    )

    # Extract results and format them
    # Use __getitem__ which returns dict, not the numba TypedDict
    fused_scores = combined_run[query_id] if query_id in combined_run.run else {}

    # Sort by score descending
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

    # Format results
    results = []
    for idx, (docid_str, score) in enumerate(sorted_results[:k], start=1):
        docid = int(docid_str)
        entry = doc_lookup[docid]
        results.append({
            "rank": idx,
            "docid": docid,
            "score": float(score),
            "texto": entry.get("texto", entry.get("text", "")),
            "tema": entry.get("tema"),
            "subtema": entry.get("subtema"),
            "enunciado": entry.get("enunciado"),
            "excerto": entry.get("excerto"),
            "variant": f"{method} - {entry.get('variant', '')}",
        })

    return results


def rerank_results(
    query: str,
    bm25_results: list[dict[str, Any]],
    embedding_results: list[dict[str, Any]],
    ranker: Reranker,
    k: int = 10
) -> list[dict[str, Any]]:
    """
    Rerank combined BM25 and embedding results using a neural reranker.

    Args:
        query: The search query
        bm25_results: Results from BM25 search
        embedding_results: Results from embedding search
        ranker: Reranker instance from rerankers library
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
    docid_to_meta = {}
    for docid, entry in all_results.items():
        text = entry.get("texto") or entry.get("text") or ""
        # if not text and entry.get("enunciado"):
        #     text = entry.get("enunciado", "")
        passages.append(text)
        docid_to_meta[len(passages) - 1] = {"docid": docid, "entry": entry}

    if not passages:
        logger.warning("No passages to rerank")
        return []

    # Rerank using rerankers library
    logger.info(f"Reranking {len(passages)} passages")
    reranked = ranker.rank(query, passages)

    # Get top k results
    top_results = reranked.top_k(k)

    # Normalize scores to [0, 1] using sigmoid for better interpretability
    import math

    # Format results
    results = []
    for idx, item in enumerate(top_results, start=1):
        # item.doc_id is the index in the original passages list
        meta_info = docid_to_meta[item.doc_id]
        docid = meta_info["docid"]
        entry = meta_info["entry"]

        # Apply sigmoid to convert logits to probabilities [0, 1]
        # sigmoid(x) = 1 / (1 + e^(-x))
        normalized_score = 1.0 / (1.0 + math.exp(-float(item.score)))

        results.append({
            "rank": idx,
            "docid": docid,
            "score": normalized_score,  # Now between 0 and 1
            "texto": item.text,
            "tema": entry.get("tema"),
            "subtema": entry.get("subtema"),
            "enunciado": entry.get("enunciado"),
            "excerto": entry.get("excerto"),
            "variant": f"reranker - {entry.get('variant')}",
        })

    return results
