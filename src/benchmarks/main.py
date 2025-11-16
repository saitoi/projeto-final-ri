# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bm25s",
#     "duckdb",
#     "einops",
#     "flashrank",
#     "langchain-text-splitters",
#     "nltk",
#     "pydantic-settings",
#     "pyserini",
#     "sentence-transformers",
#     "tiktoken",
#     "torch",
#     "tqdm",
#     "baguetter",
# ]
# ///

import duckdb
import sys
from pathlib import Path
from collections import defaultdict
from typing import Callable
import itertools
import json

# Add src to path (parent of benchmarks directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import EmbeddingVariant, BM25Variant, get_logger, get_settings, EMBEDDING_MODELS
from search.generate_bm25 import build_bm25, query_bm25, load_corpus
from search.generate_embeddings import query_embeddings
from search.utils import create_embedding_model
import bm25s

logger = get_logger(__name__)
RetrievalFn = Callable[[str, int], list[str]]

def create_bm25_retriever(
    variant: str,
    db_path: str,
    k1: float = 1.5,
    b: float = 0.75,
    delta: float = 0.5,
    alpha: float | None = None,
    beta: float | None = None
) -> RetrievalFn:
    """Create BM25 retriever with custom parameters."""
    with duckdb.connect(db_path, read_only=True) as conn:
        corpus = load_corpus(conn=conn)

    # Handle pyserini separately
    if variant == "pyserini":
        from search._pyserini import build_pyserini, query_pyserini
        # Pyserini doesn't support custom k1, b, delta parameters via Python API
        retriever = build_pyserini(corpus=corpus, model_dir="bm25_models/bm25", build=False)

        def retrieve(query: str, k: int) -> list[str]:
            results = query_pyserini(retriever=retriever, query=query, k=k)
            return [str(result["docid"]) for result in results]

        return retrieve

    # Handle bmx separately
    if variant == "bmx":
        from search._bmx import build_bmx, query_bmx
        retriever = build_bmx(corpus=corpus, model_dir="bm25_models/bm25", build=False, k1=k1, b=b, alpha=alpha, beta=beta)

        def retrieve(query: str, k: int) -> list[str]:
            results = query_bmx(retriever=retriever, query=query, k=k)
            return [str(result["docid"]) for result in results]

        return retrieve

    # Handle bm25s variants
    corpus_text = [doc["texto"] for doc in corpus]

    # Tokenize corpus
    from search.generate_bm25 import stemmer
    corpus_tokens = bm25s.tokenize(corpus_text, stopwords="pt", stemmer=stemmer)

    # Create retriever with custom parameters
    retriever = bm25s.BM25(corpus=corpus, method=variant, k1=k1, b=b, delta=delta)
    retriever.index(corpus_tokens)

    # Wrapper de query_bm25
    def retrieve(query: str, k: int) -> list[str]:
        results = query_bm25(retriever=retriever, query=query, k=k)
        return [str(result["docid"]) for result in results]

    return retrieve


def create_embedding_retriever(variant: EmbeddingVariant, db_path: str) -> RetrievalFn:
    model = create_embedding_model(variant=variant)

    # Wrapper de query_embeddings
    def retrieve(query: str, k: int) -> list[str]:
        results = query_embeddings(
            query=query,
            k=k,
            model=model,
            variant=variant,
            db_filepath=db_path
        )
        return [str(result["docid"]) for result in results]

    return retrieve


def create_hybrid_retriever(
    bm25_variant: str,
    embedding_variant: EmbeddingVariant,
    db_path: str,
    hybrid_variant: str = "rrf",
    k1: float = 1.5,
    b: float = 0.75,
    alpha: float | None = None,
    beta: float | None = None,
) -> RetrievalFn:
    """Create hybrid retriever combining BM25 and embeddings."""
    from search.utils_hybrid import reciprocal_rank_fusion, rerank_results, create_ranker

    # Load BM25 retriever
    with duckdb.connect(db_path, read_only=True) as conn:
        corpus = load_corpus(conn=conn)

    # Handle pyserini separately
    if bm25_variant == "pyserini":
        from search._pyserini import build_pyserini, query_pyserini
        bm25_retriever = build_pyserini(corpus=corpus, model_dir="bm25_models/bm25", build=False)

        def query_bm25_fn(query: str, k: int):
            return query_pyserini(retriever=bm25_retriever, query=query, k=k)
    # Handle bmx separately
    elif bm25_variant == "bmx":
        from search._bmx import build_bmx, query_bmx
        bm25_retriever = build_bmx(corpus=corpus, model_dir="bm25_models/bm25", build=False, k1=k1, b=b, alpha=alpha, beta=beta)

        def query_bm25_fn(query: str, k: int):
            return query_bmx(retriever=bm25_retriever, query=query, k=k)
    else:
        # Handle bm25s variants
        corpus_text = [doc["texto"] for doc in corpus]
        from search.generate_bm25 import stemmer
        corpus_tokens = bm25s.tokenize(corpus_text, stopwords="pt", stemmer=stemmer)
        bm25_retriever = bm25s.BM25(corpus=corpus, method=bm25_variant, k1=k1, b=b, delta=0.5)
        bm25_retriever.index(corpus_tokens)

        def query_bm25_fn(query: str, k: int):
            return query_bm25(retriever=bm25_retriever, query=query, k=k)

    # Load embedding model
    embedding_model = create_embedding_model(variant=embedding_variant)

    # Create ranker if needed
    ranker = create_ranker() if hybrid_variant == "ranker" else None

    # Wrapper for hybrid search
    def retrieve(query: str, k: int) -> list[str]:
        # Get BM25 results
        bm25_results = query_bm25_fn(query, k)
        for item in bm25_results:
            item["variant"] = bm25_variant

        # Get embedding results
        emb_results = query_embeddings(
            query=query,
            k=k,
            model=embedding_model,
            variant=embedding_variant,
            db_filepath=db_path
        )
        for item in emb_results:
            item["variant"] = embedding_variant

        # Combine results
        if hybrid_variant == "rrf":
            results = reciprocal_rank_fusion(
                bm25_results=bm25_results,
                embedding_results=emb_results,
                k=k
            )
        elif hybrid_variant == "ranker":
            results = rerank_results(
                query=query,
                bm25_results=bm25_results,
                embedding_results=emb_results,
                ranker=ranker,
                k=k
            )
        else:
            raise ValueError(f"Unknown hybrid variant: {hybrid_variant}")

        return [str(result["docid"]) for result in results]

    return retrieve


def get_all_query_groups(db_path: str) -> list[int]:
    """Get all distinct query groups from database."""
    conn = duckdb.connect(db_path, read_only=True)
    try:
        groups = conn.execute("""
            SELECT DISTINCT groupid FROM queries ORDER BY groupid
        """).fetchall()
        return [int(group) for group, in groups]
    finally:
        conn.close()


def get_group_name(group_id: int) -> str:
    """Map group ID to human-readable name."""
    names = {0: "LLM", 1: "search log", 2: "expression from LLM question"}
    return names.get(group_id, f"group-{group_id}")


def load_data_from_duckdb(
    db_path: str,
    query_group: int | None = None
) -> tuple[dict[str, str], dict[str, set[str]], dict[str, dict[str, float]]]:
    """
    Load queries and relevance judgments from DuckDB.

    Args:
        db_path: Path to database
        query_group: Optional filter by query group (0=LLM, 1=search log, 2=expression from LLM question)

    Returns:
        queries: dict mapping qid (str) to query text (str)
        relevant_docs: dict mapping qid (str) to set of relevant docids (str)
        qrels_scores: dict mapping qid (str) to dict of docid (str) to relevance score (float)
    """
    conn = duckdb.connect(db_path, read_only=True)

    try:
        # Load queries (optionally filtered by group)
        if query_group is not None:
            logger.info(f"Loading queries from group {query_group} ({get_group_name(query_group)})...")
            queries_data = conn.execute("""
                SELECT qid, text FROM queries WHERE groupid = ?
            """, [query_group]).fetchall()
        else:
            logger.info("Loading all queries...")
            queries_data = conn.execute("""
                SELECT qid, text FROM queries
            """).fetchall()

        queries = {str(qid): text for qid, text in queries_data}
        logger.info(f"Loaded {len(queries)} queries")

        # Load relevance judgments (qrels) with graded relevance scores
        logger.info("Loading relevance judgments...")
        qrels_data = conn.execute("""
            SELECT qid, docid, score
            FROM queries_rel
            WHERE score > 0
        """).fetchall()

        relevant_docs: dict[str, set[str]] = {}
        qrels_scores: dict[str, dict[str, float]] = {}

        for qid, docid, score in qrels_data:
            qid_str = str(qid)
            docid_str = str(docid)

            # Build set of relevant docs
            if qid_str not in relevant_docs:
                relevant_docs[qid_str] = set()
            relevant_docs[qid_str].add(docid_str)

            # Build graded relevance scores
            if qid_str not in qrels_scores:
                qrels_scores[qid_str] = {}
            qrels_scores[qid_str][docid_str] = float(score)

        logger.info(f"Loaded {len(relevant_docs)} queries with relevance judgments")

        return queries, relevant_docs, qrels_scores

    finally:
        conn.close()


def compute_ir_metrics(
    queries: dict[str, str],
    relevant_docs: dict[str, set[str]],
    retrieve_fn: RetrievalFn,
    qrels_scores: dict[str, dict[str, float]],
    k_values: list[int] = [1, 3, 5, 10, 100, 1000]
) -> dict[str, float]:
    """
    Compute Information Retrieval metrics using a retrieval function.

    This function is agnostic to the retrieval method (BM25, embeddings, etc.)
    Uses graded relevance scores for nDCG calculation.

    Args:
        queries: dict mapping qid to query text
        relevant_docs: dict mapping qid to set of relevant docids
        retrieve_fn: retrieval function that takes (query, k) and returns list of docids
        qrels_scores: dict mapping qid to dict of docid to relevance score (for graded relevance)
        k_values: list of k values to evaluate

    Returns metrics:
    - Shallow metrics (top of ranking): P@10, MRR@10, nDCG@10
    - Deep metrics (global consistency): MAP, Recall@1000, nDCG@1000
    """
    from tqdm import tqdm
    import math

    metrics: dict[str, list[float]] = defaultdict(list)

    for qid, query_text in tqdm(queries.items(), desc="Evaluating"):
        if qid not in relevant_docs:
            continue

        # Retrieve documents using the provided retrieval function
        retrieved_docs = retrieve_fn(query_text, k=max(k_values))

        # Ground truth
        ground_truth = relevant_docs[qid]
        doc_scores = qrels_scores.get(qid, {})

        # Calculate metrics for each k
        for k in k_values:
            retrieved_at_k = set(retrieved_docs[:k])

            # Precision@k
            if k > 0:
                precision = len(retrieved_at_k & ground_truth) / k
                metrics[f"precision@{k}"].append(precision)

            # Recall@k (métricas profundas: Recall@1000)
            if len(ground_truth) > 0:
                recall = len(retrieved_at_k & ground_truth) / len(ground_truth)
                metrics[f"recall@{k}"].append(recall)

            # NDCG@k with graded relevance (métricas rasas: nDCG@10, métricas profundas: nDCG@1000)
            # DCG: Discounted Cumulative Gain (usa aproximação TREC: rel / log2(rank + 1))
            dcg = 0.0
            for i, docid in enumerate(retrieved_docs[:k], start=1):
                if docid in ground_truth:
                    rel = doc_scores.get(docid, 1.0)  # Default 1.0 se não encontrar
                    dcg += rel / math.log2(i + 1)

            # IDCG: Ideal DCG (ordena documentos relevantes por score decrescente)
            relevant_with_scores = [(docid, doc_scores.get(docid, 1.0)) for docid in ground_truth]
            relevant_sorted = sorted(relevant_with_scores, key=lambda x: -x[1])  # Ordena por score decrescente

            idcg = 0.0
            for i, (docid, rel) in enumerate(relevant_sorted[:k], start=1):
                idcg += rel / math.log2(i + 1)

            ndcg = dcg / idcg if idcg > 0 else 0.0
            metrics[f"ndcg@{k}"].append(ndcg)

        # MRR@10 (Mean Reciprocal Rank limitado ao top-10 - métrica rasa)
        reciprocal_rank_10 = 0.0
        for i, docid in enumerate(retrieved_docs[:10], start=1):
            if docid in ground_truth:
                reciprocal_rank_10 = 1.0 / i
                break
        metrics["mrr@10"].append(reciprocal_rank_10)

        # MAP (Mean Average Precision - métrica profunda de consistência global)
        average_precision = 0.0
        relevant_retrieved = 0
        for i, docid in enumerate(retrieved_docs, start=1):
            if docid in ground_truth:
                relevant_retrieved += 1
                average_precision += relevant_retrieved / i
        if len(ground_truth) > 0:
            average_precision /= len(ground_truth)
        metrics["map"].append(average_precision)

    # Aggregate metrics (mean)
    aggregated_metrics = {}
    for metric_name, values in metrics.items():
        aggregated_metrics[metric_name] = sum(values) / len(values) if values else 0.0

    return aggregated_metrics


def run_benchmark_for_group(
    retrieve_fn: RetrievalFn,
    db_path: str,
    method_name: str,
    query_group: int | None = None
):
    """Run benchmark for a specific query group."""
    # Load data
    queries, relevant_docs, qrels_scores = load_data_from_duckdb(db_path, query_group)

    if len(queries) == 0:
        logger.warning(f"No queries found for group {query_group}")
        return

    # Run evaluation
    logger.info("Running evaluation...")
    results = compute_ir_metrics(
        queries=queries,
        relevant_docs=relevant_docs,
        retrieve_fn=retrieve_fn,
        qrels_scores=qrels_scores,
        k_values=[1, 3, 5, 10, 100, 1000]
    )

    # Print results
    group_suffix = f" - {get_group_name(query_group)}" if query_group is not None else ""
    print("\n" + "="*80)
    print(f"Information Retrieval Evaluation: {method_name}{group_suffix}")
    print("="*80)
    print(f"Queries: {len(queries)}")
    print(f"Queries with relevance judgments: {len(relevant_docs)}")
    if query_group is not None:
        print(f"Query Group: {query_group} ({get_group_name(query_group)})")
    print("\nResults:")
    print("-"*80)

    # Métricas rasas (topo do ranking)
    print("Shallow Metrics (top of ranking):")
    shallow_metrics = ["precision@10", "mrr@10", "ndcg@10"]
    for metric in shallow_metrics:
        if metric in results:
            print(f"  {metric}: {results[metric]:.4f}")

    print("\nDeep Metrics (global consistency):")
    deep_metrics = ["map", "recall@1000", "ndcg@1000"]
    for metric in deep_metrics:
        if metric in results:
            print(f"  {metric}: {results[metric]:.4f}")

    print("\nAll Metrics:")
    metric_order = ["precision@1", "precision@3", "precision@5", "precision@10",
                   "recall@1", "recall@3", "recall@5", "recall@10", "recall@100", "recall@1000",
                   "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10", "ndcg@100", "ndcg@1000",
                   "mrr@10", "map"]

    for metric in metric_order:
        if metric in results:
            print(f"  {metric}: {results[metric]:.4f}")

    print("-"*80)
    print("\nPrimary Metric: map")
    print(f"Primary Score: {results['map']:.4f}")
    print("="*80 + "\n")


def get_bm25_parameter_grid(variant: BM25Variant) -> dict:
    """
    Get parameter grid for each BM25 variant.

    Based on: Kamphuis et al. 2020 - Which BM25 Do You Mean?
    https://link.springer.com/chapter/10.1007/978-3-030-45442-5_4

    For BMX: Li et al. 2024 - BMX: Entropy-weighted Similarity and Semantic-enhanced Lexical Search
    https://arxiv.org/abs/2408.06643
    """
    # Common parameters for all variants
    k1_values = [0.5, 0.9, 1.2, 1.5, 2.0]
    b_values = [0.0, 0.3, 0.5, 0.75, 1.0]

    if variant == "bmx":
        # BMX-specific parameters: alpha (entropy normalization) and beta (semantic similarity)
        alpha_values = [0.5, 0.75, 1.0, 1.25, 1.5]
        beta_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        return {
            "k1": k1_values,
            "b": b_values,
            "delta": [0.5],  # Not used by BMX
            "alpha": alpha_values,
            "beta": beta_values
        }
    elif variant in ("bm25l", "bm25+"):
        # BM25L and BM25+ require delta parameter
        delta_values = [0.0, 0.5, 1.0, 1.5]
        return {
            "k1": k1_values,
            "b": b_values,
            "delta": delta_values,
            "alpha": [None],  # Not used
            "beta": [None]    # Not used
        }
    else:
        # Robertson, Lucene, ATIRE don't use delta, alpha, or beta
        return {
            "k1": k1_values,
            "b": b_values,
            "delta": [0.5],   # Single value (will be ignored)
            "alpha": [None],  # Not used
            "beta": [None]    # Not used
        }


def grid_search_bm25(
    variant: BM25Variant,
    db_path: str,
    query_group: int | None = None,
    top_n: int = 5
) -> list[dict]:
    """
    Perform grid search over BM25 parameters for a given variant.

    Args:
        variant: BM25 variant to evaluate
        db_path: Database path
        query_group: Optional query group filter
        top_n: Number of top configurations to return

    Returns:
        List of top configurations with their metrics
    """
    logger.info(f"Starting grid search for BM25 variant: {variant}")

    # Get parameter grid
    param_grid = get_bm25_parameter_grid(variant)
    logger.info(f"Parameter grid: {param_grid}")

    # Generate all combinations
    param_combinations = list(itertools.product(
        param_grid["k1"],
        param_grid["b"],
        param_grid["delta"],
        param_grid["alpha"],
        param_grid["beta"]
    ))

    total_combinations = len(param_combinations)
    logger.info(f"Total parameter combinations to evaluate: {total_combinations}")

    # Load data once
    queries, relevant_docs, qrels_scores = load_data_from_duckdb(db_path, query_group)

    if len(queries) == 0:
        logger.warning(f"No queries found for group {query_group}")
        return []

    # Evaluate each combination
    results = []
    from tqdm import tqdm

    for i, (k1, b, delta, alpha, beta) in enumerate(tqdm(param_combinations, desc=f"Grid Search ({variant})")):
        param_str = f"k1={k1}, b={b}, delta={delta}"
        if alpha is not None:
            param_str += f", alpha={alpha}"
        if beta is not None:
            param_str += f", beta={beta}"
        logger.info(f"[{i+1}/{total_combinations}] Evaluating {param_str}")

        try:
            # Create retriever with these parameters
            retrieve_fn = create_bm25_retriever(variant, db_path, k1=k1, b=b, delta=delta, alpha=alpha, beta=beta)

            # Evaluate
            metrics = compute_ir_metrics(
                queries=queries,
                relevant_docs=relevant_docs,
                retrieve_fn=retrieve_fn,
                qrels_scores=qrels_scores,
                k_values=[1, 3, 5, 10, 100, 1000]
            )

            # Store results
            config = {
                "variant": variant,
                "k1": k1,
                "b": b,
                "delta": delta if variant in ("bm25l", "bm25+") else None,
                "alpha": alpha if variant == "bmx" else None,
                "beta": beta if variant == "bmx" else None,
                "metrics": metrics,
                "primary_score": metrics.get("map", 0.0)  # Use MAP as primary metric
            }
            results.append(config)

        except Exception as e:
            logger.error(f"Error evaluating {param_str}: {e}")
            continue

    # Sort by primary score (MAP)
    results.sort(key=lambda x: x["primary_score"], reverse=True)

    # Print top results
    print("\n" + "="*100)
    print(f"Grid Search Results: BM25 ({variant})")
    print("="*100)
    print(f"Total configurations evaluated: {len(results)}")
    print(f"Query group: {query_group if query_group is not None else 'All'}")
    print("\n" + f"Top {top_n} Configurations (ranked by MAP):")
    print("-"*100)

    for i, config in enumerate(results[:top_n], 1):
        params = f"k1={config['k1']}, b={config['b']}"
        if config['delta'] is not None:
            params += f", delta={config['delta']}"
        if config['alpha'] is not None:
            params += f", alpha={config['alpha']}"
        if config['beta'] is not None:
            params += f", beta={config['beta']}"

        print(f"\n#{i} - {params}")
        print(f"  MAP: {config['metrics']['map']:.4f}")
        print(f"  MRR@10: {config['metrics']['mrr@10']:.4f}")
        print(f"  P@10: {config['metrics']['precision@10']:.4f}")
        print(f"  nDCG@10: {config['metrics']['ndcg@10']:.4f}")
        print(f"  Recall@1000: {config['metrics']['recall@1000']:.4f}")

    print("="*100 + "\n")

    # Save results to JSON
    output_file = Path(f"results/grid_search_{variant}_group{query_group if query_group is not None else 'all'}.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Detailed results saved to {output_file}")

    return results[:top_n]


def run_grid_search_all_variants(
    db_path: str,
    query_group: int | None = None
):
    """Run grid search for all BM25 variants and compare."""
    variants: list[BM25Variant] = [
        # "robertson",
        # "lucene",
        # "atire",
        # "bm25l",
        # "bm25+",
        "bmx"
    ]

    all_best_configs = {}

    for variant in variants:
        logger.info(f"\n{'='*100}")
        logger.info(f"Grid Search: {variant}")
        logger.info(f"{'='*100}\n")

        best_configs = grid_search_bm25(
            variant=variant,
            db_path=db_path,
            query_group=query_group,
            top_n=3
        )

        if best_configs:
            all_best_configs[variant] = best_configs[0]  # Store best config

    # Print comparison
    print("\n" + "="*100)
    print("COMPARISON: Best Configuration for Each Variant")
    print("="*100)

    if all_best_configs:
        # Sort by MAP
        sorted_variants = sorted(
            all_best_configs.items(),
            key=lambda x: x[1]["primary_score"],
            reverse=True
        )

        for rank, (variant, config) in enumerate(sorted_variants, 1):
            print(f"\n#{rank} {variant.upper()}")
            params = f"k1={config['k1']}, b={config['b']}"
            if config['delta'] is not None:
                params += f", delta={config['delta']}"
            if config.get('alpha') is not None:
                params += f", alpha={config['alpha']}"
            if config.get('beta') is not None:
                params += f", beta={config['beta']}"
            print(f"  Parameters: {params}")
            print(f"  MAP: {config['metrics']['map']:.4f}")
            print(f"  MRR@10: {config['metrics']['mrr@10']:.4f}")
            print(f"  P@10: {config['metrics']['precision@10']:.4f}")
            print(f"  nDCG@10: {config['metrics']['ndcg@10']:.4f}")

        print("\n" + "="*100)
        print(f"WINNER: {sorted_variants[0][0].upper()}")
        print(f"Best MAP: {sorted_variants[0][1]['primary_score']:.4f}")
        print("="*100 + "\n")

    # Save comparison
    output_file = Path(f"results/comparison_all_variants_group{query_group if query_group is not None else 'all'}.json")
    with open(output_file, 'w') as f:
        json.dump(all_best_configs, f, indent=2)

    logger.info(f"Comparison saved to {output_file}")


def main():
    settings = get_settings()

    # Check if grid search mode is enabled (via environment variable or flag)
    import os
    run_grid_search = os.environ.get("BM25_GRID_SEARCH", "").lower() in ("1", "true", "yes")

    if run_grid_search and settings.variant == "bm25":
        logger.info("Grid search mode enabled")
        run_grid_search_all_variants(settings.database, settings.query_group)
        return

    # Determine retrieval method and variant
    if settings.variant == "bm25":
        variant = settings.bm25_variant
        method_name = f"BM25 ({variant})"
        logger.info(f"Starting benchmark for {method_name}")

        retrieve_fn = create_bm25_retriever(
            variant, settings.database,
            k1=settings.k1,
            b=settings.b
        )

    elif settings.variant == "embeddings":
        variant = settings.embedding_variant
        method_name = f"Embeddings ({variant})"
        model_name = EMBEDDING_MODELS[variant]
        logger.info(f"Starting benchmark for {method_name}")
        logger.info(f"Model: {model_name}")

        retrieve_fn = create_embedding_retriever(variant, settings.database)

    elif settings.variant == "hybrid":
        bm25_variant = settings.bm25_variant
        embedding_variant = settings.embedding_variant
        hybrid_variant = settings.hybrid_variant
        method_name = f"Hybrid ({hybrid_variant}: BM25-{bm25_variant} + Emb-{embedding_variant})"
        logger.info(f"Starting benchmark for {method_name}")
        logger.info(f"BM25 variant: {bm25_variant}")
        logger.info(f"Embedding variant: {embedding_variant}")
        logger.info(f"Hybrid strategy: {hybrid_variant}")

        retrieve_fn = create_hybrid_retriever(
            bm25_variant=bm25_variant,
            embedding_variant=embedding_variant,
            db_path=settings.database,
            hybrid_variant=hybrid_variant,
            k1=settings.k1,
            b=settings.b
        )

    else:
        raise ValueError(f"Unsupported model: {settings.variant}. Use 'bm25', 'embeddings', or 'hybrid'")

    # Run benchmarks
    if settings.query_group is not None:
        # Single group
        run_benchmark_for_group(retrieve_fn, settings.database, method_name, settings.query_group)
    else:
        # All groups sequentially
        groups = get_all_query_groups(settings.database)
        logger.info(f"Running benchmarks for {len(groups)} query groups: {groups}")

        for group in groups:
            run_benchmark_for_group(retrieve_fn, settings.database, method_name, group)


if __name__ == "__main__":
    main()
