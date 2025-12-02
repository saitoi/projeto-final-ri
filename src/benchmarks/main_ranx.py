"""
Benchmark script using ranx library for metric computation.

This version uses ranx's built-in evaluation functions instead of custom implementations,
providing access to more metrics and standardized computation.

Supported metrics by ranx:
- hits@k: Number of hits at cutoff k
- hit_rate@k: Hit rate at cutoff k
- precision@k: Precision at cutoff k
- recall@k: Recall at cutoff k
- f1@k: F1 score at cutoff k
- r-precision: R-precision
- mrr: Mean Reciprocal Rank (MRR@10 by default)
- map: Mean Average Precision
- ndcg@k: Normalized Discounted Cumulative Gain at cutoff k
- ndcg_burges@k: NDCG using Burges formula
- bpref: Binary preference
- rbp.p: Rank-biased precision with persistence p (e.g., rbp.95)
"""

import duckdb
import sys
from pathlib import Path
from typing import Callable
import json
import numpy as np

# Add src to path (parent of benchmarks directory)
sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import EmbeddingVariant, get_logger, get_settings, EMBEDDING_MODELS
from search.generate_bm25 import query_bm25, load_corpus
from search.generate_embeddings import query_embeddings
from search.utils import create_embedding_model
import bm25s

# Import ranx for evaluation
from ranx import Qrels, Run, evaluate

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
) -> tuple[dict[str, str], dict[str, dict[str, float]]]:
    """
    Load queries and relevance judgments from DuckDB for ranx.

    Args:
        db_path: Path to database
        query_group: Optional filter by query group (single int, list of ints, or None)

    Returns:
        queries: dict mapping qid (str) to query text (str)
        qrels_dict: dict mapping qid (str) to dict of docid (str) to relevance score (float)
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

        # Build qrels in ranx format: {qid: {docid: score}}
        qrels_dict: dict[str, dict[str, float]] = {}
        for qid, docid, score in qrels_data:
            qid_str = str(qid)
            docid_str = str(docid)

            if qid_str not in qrels_dict:
                qrels_dict[qid_str] = {}
            qrels_dict[qid_str][docid_str] = float(score)

        logger.info(f"Loaded {len(qrels_dict)} queries with relevance judgments")

        return queries, qrels_dict

    finally:
        conn.close()


def compute_interpolated_precision_recall(
    queries: dict[str, str],
    qrels_dict: dict[str, dict[str, float]],
    retrieve_fn: RetrievalFn,
    max_k: int = 1000,
    recall_levels: list[float] | None = None
) -> dict[str, float]:
    """
    Compute 11-point interpolated precision-recall (classic TREC metric).

    For each query:
    1. Compute precision and recall at each rank position
    2. Interpolate precision at standard recall levels (0.0, 0.1, ..., 1.0)
    3. Average across all queries

    Interpolated precision at recall level r:
        P_interp(r) = max_{r' >= r} P(r')

    Args:
        queries: dict mapping qid to query text
        qrels_dict: dict mapping qid to dict of docid to relevance score
        retrieve_fn: retrieval function that takes (query, k) and returns list of docids
        max_k: maximum k value for retrieval
        recall_levels: list of recall levels (default: [0.0, 0.1, ..., 1.0])

    Returns:
        dict with keys like 'interp_p@0.0', 'interp_p@0.1', ..., 'interp_p@1.0'
    """
    from tqdm import tqdm

    if recall_levels is None:
        # Standard 11 recall levels
        recall_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # Store interpolated precisions for each query at each recall level
    all_interp_precisions = {r: [] for r in recall_levels}

    for qid, query_text in tqdm(queries.items(), desc="Computing interpolated P-R"):
        if qid not in qrels_dict:
            continue

        # Get relevant docs and their count
        relevant_docs = set(qrels_dict[qid].keys())
        num_relevant = len(relevant_docs)

        if num_relevant == 0:
            continue

        # Retrieve documents
        retrieved_docs = retrieve_fn(query_text, k=max_k)

        # Compute precision and recall at each rank
        precisions = []
        recalls = []
        num_relevant_retrieved = 0

        for rank, docid in enumerate(retrieved_docs, start=1):
            if docid in relevant_docs:
                num_relevant_retrieved += 1

            precision = num_relevant_retrieved / rank
            recall = num_relevant_retrieved / num_relevant

            precisions.append(precision)
            recalls.append(recall)

        # If no documents retrieved, skip
        if not precisions:
            continue

        # Interpolate precision at each recall level
        for recall_level in recall_levels:
            # Find all precisions where recall >= recall_level
            valid_precisions = [
                p for p, r in zip(precisions, recalls) if r >= recall_level
            ]

            # Interpolated precision is the maximum of all valid precisions
            if valid_precisions:
                interp_precision = max(valid_precisions)
            else:
                interp_precision = 0.0

            all_interp_precisions[recall_level].append(interp_precision)

    # Average interpolated precision across all queries for each recall level
    results = {}
    for recall_level in recall_levels:
        if all_interp_precisions[recall_level]:
            avg_precision = np.mean(all_interp_precisions[recall_level])
            # Format recall level as string (e.g., "0.0", "0.1")
            key = f"interp_p@{recall_level:.1f}"
            results[key] = float(avg_precision)
        else:
            key = f"interp_p@{recall_level:.1f}"
            results[key] = 0.0

    return results


def compute_ir_metrics_ranx(
    queries: dict[str, str],
    qrels_dict: dict[str, dict[str, float]],
    retrieve_fn: RetrievalFn,
    metrics: list[str] | None = None,
    max_k: int = 1000,
    compute_interp_pr: bool = False
) -> dict[str, float]:
    """
    Compute Information Retrieval metrics using ranx library.

    This function uses ranx's native evaluation which supports:
    - hits@k, hit_rate@k, precision@k, recall@k, f1@k
    - r-precision, mrr, map
    - ndcg@k, ndcg_burges@k, dcg@k, dcg_burges@k
    - bpref, rbp.p (e.g., rbp.95)

    Args:
        queries: dict mapping qid to query text
        qrels_dict: dict mapping qid to dict of docid to relevance score
        retrieve_fn: retrieval function that takes (query, k) and returns list of docids
        metrics: list of metrics to compute (if None, uses default set)
        max_k: maximum k value for retrieval
        compute_interp_pr: if True, also compute 11-point interpolated precision-recall

    Returns:
        dict of metric names to scores
    """
    from tqdm import tqdm

    if metrics is None:
        # Default comprehensive metric set based on best practices
        # Shallow metrics (top-k): focus on early rank positions
        # Deep metrics: focus on global consistency and recall
        metrics = [
            # Hits (shallow and deep)
            "hits@1", "hits@3", "hits@5", "hits@10", "hits@100", "hits@500", "hits@1000",

            # Precision (shallow and deep)
            "precision@1", "precision@3", "precision@5", "precision@10",
            "precision@100", "precision@500", "precision@1000",

            # Recall (shallow and deep)
            "recall@1", "recall@3", "recall@5", "recall@10",
            "recall@100", "recall@500", "recall@1000",

            # NDCG (shallow and deep)
            "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
            "ndcg@100", "ndcg@500", "ndcg@1000",

            # Ranking metrics
            "mrr",  # Mean Reciprocal Rank
            "map",  # Mean Average Precision (DEEP - most important)

            # Advanced metrics
            "r-precision",  # R-precision
            "bpref",  # Binary preference
            "hit_rate@10",  # Hit rate at 10
            "f1@10",  # F1 score at 10
        ]

    # Build run in ranx format: {qid: {docid: score}}
    import time
    import numpy as np

    run_dict: dict[str, dict[str, float]] = {}
    query_times: list[float] = []

    for qid, query_text in tqdm(queries.items(), desc="Retrieving"):
        if qid not in qrels_dict:
            continue

        # Measure retrieval time
        start_time = time.perf_counter()
        retrieved_docs = retrieve_fn(query_text, k=max_k)
        end_time = time.perf_counter()

        query_time = (end_time - start_time) * 1000  # Convert to milliseconds
        query_times.append(query_time)

        # Build run scores (use 1/rank as score for ranking)
        run_dict[qid] = {}
        for rank, docid in enumerate(retrieved_docs, start=1):
            run_dict[qid][docid] = 1.0 / rank

    # Create ranx objects
    qrels = Qrels(qrels_dict)
    run = Run(run_dict)

    # Filter qrels to only include queries that have results in run
    # This matches the behavior of main.py which skips queries without results
    filtered_qrels_dict = {qid: rels for qid, rels in qrels_dict.items() if qid in run_dict}

    logger.info(f"Queries with qrels: {len(qrels_dict)}")
    logger.info(f"Queries with results: {len(run_dict)}")
    logger.info(f"Queries with both: {len(filtered_qrels_dict)}")

    # Recreate qrels and run with matching query IDs
    qrels = Qrels(filtered_qrels_dict)
    run_filtered_dict = {qid: run_dict[qid] for qid in filtered_qrels_dict.keys()}
    run = Run(run_filtered_dict)

    # Evaluate using ranx
    logger.info(f"Evaluating {len(metrics)} metrics...")
    results = evaluate(
        qrels=qrels,
        run=run,
        metrics=metrics,
        return_mean=True,
        threads=0  # Use all available threads
    )

    # Add timing statistics to results
    if query_times:
        results["query_time_mean_ms"] = float(np.mean(query_times))
        results["query_time_median_ms"] = float(np.median(query_times))
        results["query_time_std_ms"] = float(np.std(query_times))
        results["query_time_min_ms"] = float(np.min(query_times))
        results["query_time_max_ms"] = float(np.max(query_times))
        results["query_time_p95_ms"] = float(np.percentile(query_times, 95))
        results["query_time_p99_ms"] = float(np.percentile(query_times, 99))
        results["total_queries_timed"] = len(query_times)

    # Compute 11-point interpolated precision-recall if requested
    if compute_interp_pr:
        logger.info("Computing 11-point interpolated precision-recall...")
        interp_results = compute_interpolated_precision_recall(
            queries=queries,
            qrels_dict=qrels_dict,
            retrieve_fn=retrieve_fn,
            max_k=max_k
        )
        results.update(interp_results)

    return results


def run_benchmark_for_group(
    retrieve_fn: RetrievalFn,
    db_path: str,
    method_name: str,
    query_group: int | list[int] | None = None,
    save_results: bool = False,
    variant_info: dict | None = None,
    metrics: list[str] | None = None,
    compute_interp_pr: bool = False
):
    """Run benchmark for a specific query group using ranx."""
    # Load data
    queries, qrels_dict = load_data_from_duckdb(db_path, query_group)

    if len(queries) == 0:
        logger.warning(f"No queries found for group {query_group}")
        return None

    # Run evaluation with ranx
    logger.info("Running evaluation with ranx...")
    results = compute_ir_metrics_ranx(
        queries=queries,
        qrels_dict=qrels_dict,
        retrieve_fn=retrieve_fn,
        metrics=metrics,
        max_k=1000,
        compute_interp_pr=compute_interp_pr
    )

    # Print results
    group_suffix = f" - {get_group_name(query_group)}" if query_group is not None else ""
    print("\n" + "="*80)
    print(f"Information Retrieval Evaluation (ranx): {method_name}{group_suffix}")
    print("="*80)
    print(f"Queries: {len(queries)}")
    print(f"Queries with relevance judgments: {len(qrels_dict)}")
    if query_group is not None:
        if isinstance(query_group, list):
            print(f"Query Groups: {query_group} ({get_group_name(query_group)})")
        else:
            print(f"Query Group: {query_group} ({get_group_name(query_group)})")
    print("\nResults:")
    print("-"*80)

    # Separate timing metrics and interpolated P-R from IR metrics
    timing_metrics = {k: v for k, v in results.items() if "query_time" in k or "total_queries_timed" in k}
    interp_metrics = {k: v for k, v in results.items() if k.startswith("interp_p@")}
    ir_metrics = {k: v for k, v in results.items() if k not in timing_metrics and k not in interp_metrics}

    # Organize IR metrics by category (shallow vs deep)
    shallow_metrics = [m for m in ir_metrics.keys() if any(x in m for x in ["@1", "@3", "@5", "@10"]) or m in ["mrr"]]
    deep_metrics = [m for m in ir_metrics.keys() if any(x in m for x in ["@100", "@500", "@1000", "map", "r-precision", "bpref"])]

    print("\nShallow Metrics (top of ranking - k=1,3,5,10):")
    # Group by metric type
    for metric_type in ["hits", "precision", "recall", "ndcg", "hit_rate", "f1", "mrr"]:
        relevant = sorted([m for m in shallow_metrics if metric_type in m])
        if relevant:
            values = [f"{m}: {results[m]:.4f}" for m in relevant]
            print(f"  {', '.join(values)}")

    print("\nDeep Metrics (global consistency - k=100,500,1000, MAP):")
    # Group by metric type
    for metric_type in ["precision", "recall", "ndcg", "map", "r-precision", "bpref"]:
        relevant = sorted([m for m in deep_metrics if metric_type in m])
        if relevant:
            values = [f"{m}: {results[m]:.4f}" for m in relevant]
            print(f"  {', '.join(values)}")

    # Print timing statistics
    if timing_metrics:
        print("\nQuery Time Statistics:")
        print(f"  Mean query time: {timing_metrics.get('query_time_mean_ms', 0):.2f} ms")
        print(f"  Median query time: {timing_metrics.get('query_time_median_ms', 0):.2f} ms")
        print(f"  Std dev: {timing_metrics.get('query_time_std_ms', 0):.2f} ms")
        print(f"  Min: {timing_metrics.get('query_time_min_ms', 0):.2f} ms")
        print(f"  Max: {timing_metrics.get('query_time_max_ms', 0):.2f} ms")
        print(f"  P95: {timing_metrics.get('query_time_p95_ms', 0):.2f} ms")
        print(f"  P99: {timing_metrics.get('query_time_p99_ms', 0):.2f} ms")
        print(f"  Queries/sec (throughput): {1000 / timing_metrics.get('query_time_mean_ms', 1):.2f}")

    # Print 11-point interpolated precision-recall
    if interp_metrics:
        print("\n11-Point Interpolated Precision-Recall:")
        # Sort by recall level
        sorted_interp = sorted(interp_metrics.items(), key=lambda x: float(x[0].split('@')[1]))
        for metric, value in sorted_interp:
            recall_level = metric.split('@')[1]
            print(f"  Recall {recall_level}: {value:.4f}")

    print("\nAll Metrics:")
    for metric in sorted(ir_metrics.keys()):
        print(f"  {metric}: {results[metric]:.4f}")

    print("-"*80)
    print("\nPrimary Metric: map")
    print(f"Primary Score: {results.get('map', 0.0):.4f}")
    if timing_metrics:
        print(f"Average Query Time: {timing_metrics.get('query_time_mean_ms', 0):.2f} ms")
    print("="*80 + "\n")

    # Save results to JSON if requested
    if save_results:
        output_data = {
            "method": method_name,
            "query_group": query_group,
            "query_group_name": get_group_name(query_group) if query_group is not None else "all",
            "num_queries": len(queries),
            "num_queries_with_judgments": len(qrels_dict),
            "metrics": results,
            "primary_score": results.get("map", 0.0),
            "evaluation_library": "ranx"
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

        output_file = Path(f"results/ranx_{method_slug}_group{group_suffix}.json")
        output_file.parent.mkdir(exist_ok=True, parents=True)

        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Benchmark results saved to {output_file}")

    return results


def main():
    settings = get_settings()

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
        save_results = True

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
            variant_info=variant_info,
            compute_interp_pr=settings.compute_interp_pr
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
                variant_info=variant_info,
                compute_interp_pr=settings.compute_interp_pr
            )


if __name__ == "__main__":
    main()
