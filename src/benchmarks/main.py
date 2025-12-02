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
#     "rerankers[transformers]",
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
    hybrid_variant: str = "fusion",
    fusion_method: str = "rrf",
    fusion_norm: str = "min-max",
    fusion_k: int = 60,
    k1: float = 1.5,
    b: float = 0.75,
    alpha: float | None = None,
    beta: float | None = None,
) -> RetrievalFn:
    """Create hybrid retriever combining BM25 and embeddings."""
    from search.utils_hybrid import fusion_results, rerank_results, create_ranker

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
        if hybrid_variant == "fusion":
            results = fusion_results(
                bm25_results=bm25_results,
                embedding_results=emb_results,
                k=k,
                method=fusion_method,
                norm=fusion_norm,
                fusion_k=fusion_k
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


def get_group_name(group_id: int | list[int]) -> str:
    """Map group ID to human-readable name."""
    names = {0: "LLM", 1: "search log", 2: "expression from LLM question", 3: "LLM"}

    if isinstance(group_id, list):
        # Multiple groups - concatenate names
        group_names = [names.get(gid, f"group-{gid}") for gid in group_id]
        return ", ".join(group_names)
    else:
        return names.get(group_id, f"group-{group_id}")


def load_data_from_duckdb(
    db_path: str,
    query_group: int | None | list[int] = None
) -> tuple[dict[str, str], dict[str, set[str]], dict[str, dict[str, float]]]:
    """
    Load queries and relevance judgments from DuckDB.

    Args:
        db_path: Path to database
        query_group: Optional filter by query group (0=LLM, 1=search log, 2=expression from LLM question)
                    Can be a single int, a list of ints, or None for all groups

    Returns:
        queries: dict mapping qid (str) to query text (str)
        relevant_docs: dict mapping qid (str) to set of relevant docids (str)
        qrels_scores: dict mapping qid (str) to dict of docid (str) to relevance score (float)
    """
    conn = duckdb.connect(db_path, read_only=True)

    try:
        # Load queries (optionally filtered by group)
        if query_group is not None:
            if isinstance(query_group, list):
                logger.info(f"Loading queries from groups {query_group}...")
                placeholders = ','.join(['?'] * len(query_group))
                queries_data = conn.execute(f"""
                    SELECT qid, text FROM queries WHERE groupid IN ({placeholders})
                """, query_group).fetchall()
            else:
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
    query_group: int | list[int] | None = None,
    save_results: bool = False,
    variant_info: dict | None = None
):
    """Run benchmark for a specific query group."""
    # Load data
    queries, relevant_docs, qrels_scores = load_data_from_duckdb(db_path, query_group)

    if len(queries) == 0:
        logger.warning(f"No queries found for group {query_group}")
        return None

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
        if isinstance(query_group, list):
            print(f"Query Groups: {query_group} ({get_group_name(query_group)})")
        else:
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

    # Save results to JSON if requested
    if save_results:
        output_data = {
            "method": method_name,
            "query_group": query_group,
            "query_group_name": get_group_name(query_group) if query_group is not None else "all",
            "num_queries": len(queries),
            "num_queries_with_judgments": len(relevant_docs),
            "metrics": results,
            "primary_score": results.get("map", 0.0)
        }

        # Add variant-specific information
        if variant_info:
            output_data.update(variant_info)

        # Generate filename based on method and group
        method_slug = method_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").replace(":", "").replace("+", "_")

        # Handle list of groups in filename
        if query_group is None:
            group_suffix = "all"
        elif isinstance(query_group, list):
            group_suffix = "".join(str(g) for g in query_group)
        else:
            group_suffix = str(query_group)

        output_file = Path(f"results/benchmark_{method_slug}_group{group_suffix}.json")
        output_file.parent.mkdir(exist_ok=True, parents=True)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Benchmark results saved to {output_file}")

    return results


def get_bm25_parameter_grid(variant: BM25Variant, search_type: str = "grid") -> dict:
    """
    Get parameter grid for each BM25 variant.

    Based on: Kamphuis et al. 2020 - Which BM25 Do You Mean?
    https://link.springer.com/chapter/10.1007/978-3-030-45442-5_4

    For BMX: Li et al. 2024 - BMX: Entropy-weighted Similarity and Semantic-enhanced Lexical Search
    https://arxiv.org/abs/2408.06643

    Args:
        variant: BM25 variant name
        search_type: "base" for default params, "random" for random search, "grid" for full grid
    """
    if search_type == "base":
        # Default parameters only
        if variant == "bmx":
            return {
                "k1": [1.2],
                "b": [0.75],
                "delta": [0.5],
                "alpha": [1.0],
                "beta": [0.5]
            }
        elif variant in ("bm25l", "bm25+"):
            return {
                "k1": [1.2],
                "b": [0.75],
                "delta": [0.5],
                "alpha": [None],
                "beta": [None]
            }
        else:
            return {
                "k1": [1.2],
                "b": [0.75],
                "delta": [0.5],
                "alpha": [None],
                "beta": [None]
            }

    # Expanded parameter ranges for grid and random search
    # More granular values based on literature
    k1_values = [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.5, 1.8, 2.0, 2.5, 3.0]
    b_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]

    if variant == "bmx":
        # BMX-specific parameters: alpha (entropy normalization) and beta (semantic similarity)
        # Alpha: controls entropy-based weighting (0.5-2.0 range)
        # Beta: controls semantic similarity contribution (0.0-1.0 range)
        alpha_values = [0.3, 0.5, 0.7, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
        beta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        return {
            "k1": k1_values,
            "b": b_values,
            "delta": [0.5],  # Not used by BMX
            "alpha": alpha_values,
            "beta": beta_values
        }
    elif variant in ("bm25l", "bm25+"):
        # BM25L and BM25+ require delta parameter
        # Delta: controls document length normalization (0.0-2.0 range)
        delta_values = [0.0, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0]
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
    top_n: int = 5,
    search_type: str = "grid",
    n_random_samples: int = 100
) -> list[dict]:
    """
    Perform hyperparameter search over BM25 parameters for a given variant.

    Uses scikit-optimize (skopt) for efficient random search and grid search.

    Args:
        variant: BM25 variant to evaluate
        db_path: Database path
        query_group: Optional query group filter
        top_n: Number of top configurations to return
        search_type: "base" for default params, "random" for random search, "grid" for full grid
        n_random_samples: Number of random samples to try (only for search_type="random")

    Returns:
        List of top configurations with their metrics
    """
    logger.info(f"Starting {search_type} search for BM25 variant: {variant}")

    # Get parameter grid
    param_grid = get_bm25_parameter_grid(variant, search_type)
    logger.info(f"Parameter grid: {param_grid}")

    # Generate parameter combinations based on search type
    if search_type == "random":
        # Use sklearn's ParameterSampler for proper random sampling
        from sklearn.model_selection import ParameterSampler
        import numpy as np

        # Convert to sklearn format
        sklearn_grid = {}
        for key, values in param_grid.items():
            if None not in values:
                sklearn_grid[key] = values
            elif len(values) == 1:
                sklearn_grid[key] = values  # Keep single None value
            else:
                sklearn_grid[key] = [v for v in values if v is not None]

        # Sample random combinations
        param_sampler = ParameterSampler(
            sklearn_grid,
            n_iter=n_random_samples,
            random_state=42
        )

        param_combinations = []
        for params in param_sampler:
            k1 = params.get("k1", 1.2)
            b = params.get("b", 0.75)
            delta = params.get("delta", 0.5)
            alpha = params.get("alpha", None)
            beta = params.get("beta", None)
            param_combinations.append((k1, b, delta, alpha, beta))
    else:
        # Generate all combinations (base or grid)
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
    search_name = {
        "base": "Baseline Evaluation",
        "random": "Random Search",
        "grid": "Grid Search"
    }.get(search_type, "Parameter Search")

    print("\n" + "="*100)
    print(f"{search_name} Results: BM25 ({variant})")
    print("="*100)
    print(f"Total configurations evaluated: {len(results)}")
    print(f"Query group: {query_group if query_group is not None else 'All'}")
    print(f"Search type: {search_type}")
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
    output_file = Path(f"results/{search_type}_search_{variant}_group{query_group if query_group is not None else 'all'}.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Detailed results saved to {output_file}")

    return results[:top_n]


def grid_search_fusion(
    bm25_variant: str,
    embedding_variant: str,
    db_path: str,
    query_group: int | None = None,
    k1: float = 1.2,
    b: float = 0.75,
    top_n: int = 5
) -> list[dict]:
    """
    Perform grid search over fusion methods for hybrid search.

    Args:
        bm25_variant: BM25 variant to use
        embedding_variant: Embedding variant to use
        db_path: Database path
        query_group: Optional query group filter
        k1: BM25 k1 parameter
        b: BM25 b parameter
        top_n: Number of top configurations to return

    Returns:
        List of top configurations with their metrics
    """
    from settings import FusionMethod, NormMethod
    import typing

    logger.info(f"Starting fusion grid search: BM25-{bm25_variant} + Emb-{embedding_variant}")

    # All fusion methods from settings
    all_fusion_methods = typing.get_args(FusionMethod)

    # Filter out methods that require training data (not supported in grid search)
    methods_requiring_training = {"bayesfuse", "mapfuse", "posfuse", "probfuse", "segfuse", "slidefuse"}
    fusion_methods = [m for m in all_fusion_methods if m not in methods_requiring_training]

    norm_methods = typing.get_args(NormMethod)

    # Parameter grids for different fusion methods
    fusion_k_values = [20, 40, 60, 80, 100]  # For RRF and others
    gamma_values = [0.5, 1.0, 1.5, 2.0]  # For gmnz
    weight_values = [[0.3, 0.7], [0.5, 0.5], [0.7, 0.3]]  # For wsum, wmnz, mixed, w_bordafuse, w_condorcet
    phi_values = [0.3, 0.5, 0.7, 0.9]  # For rbc

    logger.info(f"Testing {len(fusion_methods)} fusion methods × {len(norm_methods)} normalization methods")
    logger.info(f"Fusion methods: {fusion_methods}")
    logger.info(f"Excluded (require training): {methods_requiring_training}")
    logger.info(f"Normalization methods: {norm_methods}")

    # Load data once
    queries, relevant_docs, qrels_scores = load_data_from_duckdb(db_path, query_group)

    if len(queries) == 0:
        logger.warning(f"No queries found for group {query_group}")
        return []

    # Build retriever for BM25 and embeddings once
    logger.info("Loading BM25 and embedding models...")

    # Evaluate each combination
    results = []
    from tqdm import tqdm

    # Calculate total combinations based on method-specific parameters
    total = 0
    for method in fusion_methods:
        if method == "gmnz":
            total += len(norm_methods) * len(gamma_values)
        elif method in ["wsum", "wmnz", "mixed", "w_bordafuse", "w_condorcet"]:
            total += len(norm_methods) * len(weight_values)
        elif method == "rbc":
            total += len(norm_methods) * len(phi_values)
        else:
            total += len(norm_methods) * len(fusion_k_values)

    logger.info(f"Total configurations to test: {total}")
    pbar = tqdm(total=total, desc="Fusion Grid Search")

    for fusion_method in fusion_methods:
        for norm_method in norm_methods:
            # Determine which parameter to iterate over based on method
            if fusion_method == "gmnz":
                param_iter = [(gamma, None, None) for gamma in gamma_values]
            elif fusion_method in ["wsum", "wmnz", "mixed", "w_bordafuse", "w_condorcet"]:
                param_iter = [(None, weights, None) for weights in weight_values]
            elif fusion_method == "rbc":
                param_iter = [(None, None, phi) for phi in phi_values]
            else:
                param_iter = [(None, None, None, k) for k in fusion_k_values]

            for param_tuple in param_iter:
                if len(param_tuple) == 3:
                    gamma, weights, phi = param_tuple
                    fusion_k = 60  # default
                else:
                    gamma, weights, phi, fusion_k = None, None, None, param_tuple[3]

                # Build config string for logging
                config_parts = [f"method={fusion_method}", f"norm={norm_method}"]
                if gamma is not None:
                    config_parts.append(f"gamma={gamma}")
                if weights is not None:
                    config_parts.append(f"weights={weights}")
                if phi is not None:
                    config_parts.append(f"phi={phi}")
                if fusion_method not in ["gmnz", "wsum", "wmnz", "mixed", "w_bordafuse", "w_condorcet", "rbc"]:
                    config_parts.append(f"k={fusion_k}")

                config_str = ", ".join(config_parts)

                try:
                    # Create hybrid retriever with these parameters
                    # We need to pass extra params through the fusion function
                    from search.utils_hybrid import fusion_results_with_params

                    # Get BM25 and embedding results
                    conn = duckdb.connect(db_path, read_only=True)
                    try:
                        from search.generate_bm25 import load_corpus
                        corpus = load_corpus(conn=conn)
                    finally:
                        conn.close()

                    # Build BM25 retriever
                    bm25_retriever = create_bm25_retriever(bm25_variant, db_path, k1=k1, b=b)

                    # Build embedding retriever
                    from search.utils import create_embedding_model
                    emb_model = create_embedding_model(variant=embedding_variant)

                    # Create a custom retrieve function
                    def retrieve_fn_custom(query_text: str, k_param: int) -> list[str]:
                        from search.generate_bm25 import query_bm25
                        from search.generate_embeddings import query_embeddings
                        from search.utils_hybrid import fusion_results

                        bm25_results = query_bm25(retriever=bm25_retriever, query=query_text, k=k_param)
                        emb_results = query_embeddings(
                            query=query_text, k=k_param, model=emb_model,
                            variant=embedding_variant, db_filepath=db_path
                        )

                        # Prepare extra params
                        extra_params = {}
                        if gamma is not None:
                            extra_params["gamma"] = gamma
                        if weights is not None:
                            extra_params["weights"] = weights
                        if phi is not None:
                            extra_params["phi"] = phi

                        hybrid_results = fusion_results(
                            bm25_results=bm25_results,
                            embedding_results=emb_results,
                            k=k_param,
                            method=fusion_method,
                            norm=norm_method,
                            fusion_k=fusion_k,
                            extra_params=extra_params
                        )

                        return [str(r["docid"]) for r in hybrid_results]

                    # Evaluate
                    metrics = compute_ir_metrics(
                        queries=queries,
                        relevant_docs=relevant_docs,
                        retrieve_fn=retrieve_fn_custom,
                        qrels_scores=qrels_scores,
                        k_values=[1, 3, 5, 10, 100, 1000]
                    )

                    # Store results
                    config = {
                        "fusion_method": fusion_method,
                        "fusion_norm": norm_method,
                        "fusion_k": fusion_k,
                        "bm25_variant": bm25_variant,
                        "embedding_variant": embedding_variant,
                        "bm25_k1": k1,
                        "bm25_b": b,
                        "metrics": metrics,
                        "primary_score": metrics.get("map", 0.0)
                    }

                    # Add method-specific params to config
                    if gamma is not None:
                        config["gamma"] = gamma
                    if weights is not None:
                        config["weights"] = weights
                    if phi is not None:
                        config["phi"] = phi

                    results.append(config)

                except Exception as e:
                    logger.error(f"Error evaluating {config_str}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
                finally:
                    pbar.update(1)

    pbar.close()

    # Sort by primary score (MAP)
    results.sort(key=lambda x: x["primary_score"], reverse=True)

    # Print top results
    print("\n" + "="*100)
    print(f"Fusion Grid Search Results: BM25-{bm25_variant} + Emb-{embedding_variant}")
    print("="*100)
    print(f"Total configurations evaluated: {len(results)}")
    print(f"Query group: {query_group if query_group is not None else 'All'}")
    print(f"\nTop {top_n} Configurations (ranked by MAP):")
    print("-"*100)

    for i, config in enumerate(results[:top_n], 1):
        print(f"\n#{i} - method={config['fusion_method']}, norm={config['fusion_norm']}, k={config['fusion_k']}")
        print(f"  MAP: {config['metrics']['map']:.4f}")
        print(f"  MRR@10: {config['metrics']['mrr@10']:.4f}")
        print(f"  P@10: {config['metrics']['precision@10']:.4f}")
        print(f"  nDCG@10: {config['metrics']['ndcg@10']:.4f}")
        print(f"  Recall@1000: {config['metrics']['recall@1000']:.4f}")

    print("="*100 + "\n")

    # Save results to JSON
    output_file = Path(f"results/fusion_grid_search_{bm25_variant}_{embedding_variant}_group{query_group if query_group is not None else 'all'}.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Detailed results saved to {output_file}")

    return results[:top_n]


def run_grid_search_all_variants(
    db_path: str,
    query_group: int | None = None,
    search_type: str = "grid",
    n_random_samples: int = 100
):
    """Run hyperparameter search for all BM25 variants and compare."""
    variants: list[BM25Variant] = [
        "robertson",
        "lucene",
        "atire",
        "bm25l",
        "bm25+",
        "bmx"
    ]

    all_best_configs = {}

    for variant in variants:
        logger.info(f"\n{'='*100}")
        logger.info(f"{search_type.capitalize()} Search: {variant}")
        logger.info(f"{'='*100}\n")

        best_configs = grid_search_bm25(
            variant=variant,
            db_path=db_path,
            query_group=query_group,
            top_n=3,
            search_type=search_type,
            n_random_samples=n_random_samples
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
    output_file = Path(f"results/{search_type}_comparison_all_variants_group{query_group if query_group is not None else 'all'}.json")
    with open(output_file, 'w') as f:
        json.dump(all_best_configs, f, indent=2)

    logger.info(f"Comparison saved to {output_file}")


def main():
    settings = get_settings()

    # Check if fusion grid search is enabled for hybrid
    if settings.fusion_grid_search and settings.variant == "hybrid":
        logger.info("Fusion grid search mode enabled")

        grid_search_fusion(
            bm25_variant=settings.bm25_variant,
            embedding_variant=settings.embedding_variant,
            db_path=settings.database,
            query_group=settings.query_group,
            k1=settings.k1,
            b=settings.b,
            top_n=10
        )
        return

    # Check if hyperparameter search is enabled
    if settings.grid_search and settings.variant == "bm25":
        logger.info(f"Hyperparameter search mode enabled: {settings.grid_search}")

        # Number of random samples (only used for random search)
        n_random_samples = 100

        run_grid_search_all_variants(
            db_path=settings.database,
            query_group=settings.query_group,
            search_type=settings.grid_search,
            n_random_samples=n_random_samples
        )
        return

    # Determine retrieval method and variant
    save_results = False  # Flag to save results automatically
    variant_info = {}     # Additional variant information for JSON

    if settings.variant == "bm25":
        variant = settings.bm25_variant
        method_name = f"BM25 ({variant})"
        logger.info(f"Starting benchmark for {method_name}")

        retrieve_fn = create_bm25_retriever(
            variant, settings.database,
            k1=settings.k1,
            b=settings.b,
            delta=settings.delta,
        )

        variant_info = {
            "variant_type": "bm25",
            "bm25_variant": variant,
            "k1": settings.k1,
            "b": settings.b,
            "delta": settings.delta
        }

    elif settings.variant == "embeddings":
        variant = settings.embedding_variant
        method_name = f"Embeddings ({variant})"
        model_name = EMBEDDING_MODELS[variant]
        logger.info(f"Starting benchmark for {method_name}")
        logger.info(f"Model: {model_name}")

        retrieve_fn = create_embedding_retriever(variant, settings.database)

        # Enable automatic saving for embeddings
        save_results = True
        variant_info = {
            "variant_type": "embeddings",
            "embedding_variant": variant,
            "model_name": model_name
        }

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

        # Enable automatic saving for hybrid
        save_results = True
        variant_info = {
            "variant_type": "hybrid",
            "hybrid_variant": hybrid_variant,
            "bm25_variant": bm25_variant,
            "embedding_variant": embedding_variant,
            "bm25_k1": settings.k1,
            "bm25_b": settings.b,
            "fusion_method": settings.fusion_method if hybrid_variant == "fusion" else None,
            "fusion_norm": settings.fusion_norm if hybrid_variant == "fusion" else None,
            "fusion_k": settings.fusion_k if hybrid_variant == "fusion" else None,
            "ranker_model": settings.ranker_model if hybrid_variant == "ranker" else None
        }

    else:
        raise ValueError(f"Unsupported model: {settings.variant}. Use 'bm25', 'embeddings', or 'hybrid'")

    # Run benchmarks
    if settings.query_group is not None:
        # Single group
        run_benchmark_for_group(
            retrieve_fn,
            settings.database,
            method_name,
            settings.query_group,
            save_results=save_results,
            variant_info=variant_info
        )
    else:
        # All groups sequentially
        groups = get_all_query_groups(settings.database)
        logger.info(f"Running benchmarks for {len(groups)} query groups: {groups}")

        for group in groups:
            run_benchmark_for_group(
                retrieve_fn,
                settings.database,
                method_name,
                group,
                save_results=save_results,
                variant_info=variant_info
            )


if __name__ == "__main__":
    main()
