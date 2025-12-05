#!/usr/bin/env python3
"""
End-to-end optimization pipeline for hybrid search using ranx library.

This script:
1. Tests all embedding variants on a query group
2. Tests all BM25 variants with random search
3. Runs fusion grid search with the best models
4. Saves comprehensive results to JSON

Uses ranx library for metric computation, providing:
- More metrics (hits, bpref, r-precision, etc.)
- Query time statistics
- Standardized evaluation
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Any
from settings import (
    EmbeddingVariant, BM25Variant, FusionMethod, NormMethod,
    get_logger, EMBEDDING_MODELS
)
import typing

# Import from benchmarks/main_ranx.py
from benchmarks.main_ranx import (
    create_embedding_retriever,
    create_bm25_retriever,
    create_hybrid_retriever,
    load_data_from_duckdb,
    compute_ir_metrics_ranx,
    get_group_name
)

logger = get_logger(__name__)


def build_metrics_list(k_values: list[int] | None = None) -> list[str]:
    """
    Build metrics list based on specified k values.

    Args:
        k_values: List of k values to include (e.g., [500] or [100, 500])
                  If None, uses default: [1, 3, 5, 10, 100, 500, 1000]

    Returns:
        List of metric names
    """
    if k_values is None:
        k_values = [1, 3, 5, 10, 100, 500, 1000]

    metrics = []

    # Add k-based metrics
    for k in k_values:
        metrics.extend([
            f"hits@{k}",
            f"precision@{k}",
            f"recall@{k}",
            f"ndcg@{k}",
        ])

    # Add non-k metrics (always included)
    metrics.extend([
        "mrr",  # Mean Reciprocal Rank
        "map",  # Mean Average Precision
        "r-precision",  # R-precision
        "bpref",  # Binary preference
    ])

    # Add fixed-k metrics
    if 10 in k_values:
        metrics.extend([
            "hit_rate@10",
            "f1@10",
        ])

    return metrics


def test_all_embeddings(
    db_path: str,
    query_group: int | list[int],
    metrics: list[str] | None = None,
    compute_interp_pr: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Test all embedding variants and return the best one plus all results.

    Args:
        db_path: Database path
        query_group: Single group ID or list of group IDs
        metrics: List of metrics to compute (if None, uses default set)

    Returns:
        tuple of (best_result, all_results)
    """
    logger.info("="*100)
    logger.info("STEP 1: Testing all embedding variants")
    logger.info("="*100)
    if metrics:
        logger.info(f"Computing {len(metrics)} custom metrics")

    embedding_variants = typing.get_args(EmbeddingVariant)
    queries, qrels_dict = load_data_from_duckdb(db_path, query_group)

    results = []

    for variant in embedding_variants:
        logger.info(f"\nTesting embedding variant: {variant}")
        model_name = EMBEDDING_MODELS[variant]
        logger.info(f"Model: {model_name}")

        try:
            retrieve_fn = create_embedding_retriever(variant, db_path)

            computed_metrics = compute_ir_metrics_ranx(
                queries=queries,
                qrels_dict=qrels_dict,
                retrieve_fn=retrieve_fn,
                metrics=metrics,
                max_k=1000,
                compute_interp_pr=compute_interp_pr
            )

            result = {
                "variant": variant,
                "model_name": model_name,
                "metrics": computed_metrics,
                "primary_score": computed_metrics.get("map", 0.0)
            }
            results.append(result)

            logger.info(f"  MAP: {computed_metrics['map']:.4f}")
            logger.info(f"  MRR: {computed_metrics['mrr']:.4f}")
            if 'ndcg@10' in computed_metrics:
                logger.info(f"  nDCG@10: {computed_metrics['ndcg@10']:.4f}")
            logger.info(f"  Avg query time: {computed_metrics.get('query_time_mean_ms', 0):.2f} ms")

        except Exception as e:
            logger.error(f"Error testing {variant}: {e}")
            import traceback
            logger.debug(traceback.format_exc())

            # Add error result to track failures
            result = {
                "variant": variant,
                "model_name": model_name,
                "error": str(e),
                "metrics": None,
                "primary_score": 0.0
            }
            results.append(result)
            continue

    # Sort by MAP and get best (filter out errors)
    valid_results = [r for r in results if r.get("metrics") is not None]

    if not valid_results:
        raise RuntimeError("All embedding variants failed!")

    valid_results.sort(key=lambda x: x["primary_score"], reverse=True)
    best = valid_results[0]

    logger.info("\n" + "="*100)
    logger.info(f"BEST EMBEDDING: {best['variant']} (MAP: {best['primary_score']:.4f})")
    logger.info("="*100 + "\n")

    return best, results


def test_all_bm25_random_search(
    db_path: str,
    query_group: int | list[int],
    n_samples: int = 50,
    metrics: list[str] | None = None,
    compute_interp_pr: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Test all BM25 variants with random search and return the best one plus all results.

    Args:
        db_path: Database path
        query_group: Single group ID or list of group IDs
        n_samples: Number of random samples per variant
        metrics: List of metrics to compute (if None, uses default set)

    Returns:
        tuple of (best_result, all_results)
    """
    from sklearn.model_selection import ParameterSampler

    logger.info("="*100)
    logger.info("STEP 2: Testing all BM25 variants with random search")
    logger.info("="*100)
    if metrics:
        logger.info(f"Computing {len(metrics)} custom metrics")

    bm25_variants = typing.get_args(BM25Variant)
    queries, qrels_dict = load_data_from_duckdb(db_path, query_group)

    all_results = []

    for variant in bm25_variants:
        logger.info(f"\nTesting BM25 variant: {variant}")

        # Define parameter grid for this variant
        if variant == "bmx":
            param_grid = {
                "k1": [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.5, 1.8, 2.0, 2.5, 3.0],
                "b": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
                "alpha": [0.3, 0.5, 0.7, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
                "beta": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        elif variant in ("bm25l", "bm25+"):
            param_grid = {
                "k1": [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.5, 1.8, 2.0, 2.5, 3.0],
                "b": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0],
                "delta": [0.0, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0]
            }
        else:
            param_grid = {
                "k1": [0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.4, 1.5, 1.8, 2.0, 2.5, 3.0],
                "b": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
            }

        # Sample random combinations
        param_sampler = ParameterSampler(param_grid, n_iter=n_samples, random_state=42)

        variant_results = []

        for params in param_sampler:
            k1 = params.get("k1", 1.2)
            b = params.get("b", 0.75)
            delta = params.get("delta", 0.5)
            alpha = params.get("alpha", None)
            beta = params.get("beta", None)

            try:
                retrieve_fn = create_bm25_retriever(
                    variant, db_path, k1=k1, b=b, delta=delta, alpha=alpha, beta=beta
                )

                computed_metrics = compute_ir_metrics_ranx(
                    queries=queries,
                    qrels_dict=qrels_dict,
                    retrieve_fn=retrieve_fn,
                    metrics=metrics,
                    max_k=1000,
                    compute_interp_pr=compute_interp_pr
                )

                result = {
                    "variant": variant,
                    "k1": k1,
                    "b": b,
                    "delta": delta if variant in ("bm25l", "bm25+") else None,
                    "alpha": alpha if variant == "bmx" else None,
                    "beta": beta if variant == "bmx" else None,
                    "metrics": computed_metrics,
                    "primary_score": computed_metrics.get("map", 0.0)
                }
                variant_results.append(result)

            except Exception as e:
                logger.error(f"Error with params k1={k1}, b={b}: {e}")
                continue

        # Get best for this variant
        if variant_results:
            variant_results.sort(key=lambda x: x["primary_score"], reverse=True)
            best_for_variant = variant_results[0]
            all_results.append(best_for_variant)

            logger.info(f"  Best MAP for {variant}: {best_for_variant['primary_score']:.4f}")
            logger.info(f"  Parameters: k1={best_for_variant['k1']}, b={best_for_variant['b']}")

    # Sort all results and get overall best
    all_results.sort(key=lambda x: x["primary_score"], reverse=True)
    best = all_results[0]

    logger.info("\n" + "="*100)
    logger.info(f"BEST BM25: {best['variant']} (MAP: {best['primary_score']:.4f})")
    logger.info(f"Parameters: k1={best['k1']}, b={best['b']}")
    if best['delta'] is not None:
        logger.info(f"  delta={best['delta']}")
    if best['alpha'] is not None:
        logger.info(f"  alpha={best['alpha']}")
    if best['beta'] is not None:
        logger.info(f"  beta={best['beta']}")
    logger.info("="*100 + "\n")

    return best, all_results


def test_fusion_grid_search(
    bm25_config: dict[str, Any],
    embedding_config: dict[str, Any],
    db_path: str,
    query_group: int | list[int],
    metrics: list[str] | None = None,
    compute_interp_pr: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Test all fusion methods with the best BM25 and embedding models.

    Args:
        bm25_config: Best BM25 configuration
        embedding_config: Best embedding configuration
        db_path: Database path
        query_group: Single group ID or list of group IDs
        metrics: List of metrics to compute (if None, uses default set)

    Returns:
        tuple of (best_result, all_results)
    """
    logger.info("="*100)
    logger.info("STEP 3: Testing all fusion methods")
    logger.info("="*100)
    logger.info(f"BM25: {bm25_config['variant']}")
    logger.info(f"Embeddings: {embedding_config['variant']}")
    if metrics:
        logger.info(f"Computing {len(metrics)} custom metrics")

    fusion_methods = typing.get_args(FusionMethod)
    norm_methods = typing.get_args(NormMethod)
    fusion_k_values = [20, 40, 60, 80, 100]

    queries, qrels_dict = load_data_from_duckdb(db_path, query_group)

    results = []

    total = len(fusion_methods) * len(norm_methods) * len(fusion_k_values)
    logger.info(f"Testing {total} fusion combinations...")

    from tqdm import tqdm
    pbar = tqdm(total=total, desc="Fusion Grid Search")

    for fusion_method in fusion_methods:
        for norm_method in norm_methods:
            for fusion_k in fusion_k_values:
                try:
                    retrieve_fn = create_hybrid_retriever(
                        bm25_variant=bm25_config['variant'],
                        embedding_variant=embedding_config['variant'],
                        db_path=db_path,
                        hybrid_variant="fusion",
                        fusion_method=fusion_method,
                        fusion_norm=norm_method,
                        fusion_k=fusion_k,
                        k1=bm25_config['k1'],
                        b=bm25_config['b'],
                        alpha=bm25_config.get('alpha'),
                        beta=bm25_config.get('beta')
                    )

                    computed_metrics = compute_ir_metrics_ranx(
                        queries=queries,
                        qrels_dict=qrels_dict,
                        retrieve_fn=retrieve_fn,
                        metrics=metrics,
                        max_k=1000,
                        compute_interp_pr=compute_interp_pr
                    )

                    result = {
                        "fusion_method": fusion_method,
                        "fusion_norm": norm_method,
                        "fusion_k": fusion_k,
                        "metrics": computed_metrics,
                        "primary_score": computed_metrics.get("map", 0.0)
                    }
                    results.append(result)

                except Exception as e:
                    logger.error(f"Error with {fusion_method}/{norm_method}/{fusion_k}: {e}")
                    continue
                finally:
                    pbar.update(1)

    pbar.close()

    # Sort and get best
    results.sort(key=lambda x: x["primary_score"], reverse=True)
    best = results[0]

    logger.info("\n" + "="*100)
    logger.info(f"BEST FUSION: {best['fusion_method']} (MAP: {best['primary_score']:.4f})")
    logger.info(f"Normalization: {best['fusion_norm']}")
    logger.info(f"Fusion K: {best['fusion_k']}")
    logger.info("="*100 + "\n")

    return best, results


def get_group_display_name(query_group: int | list[int]) -> str:
    """Get display name for single group or multiple groups."""
    if isinstance(query_group, list):
        names = [get_group_name(g) for g in query_group]
        return f"groups {query_group} ({', '.join(names)})"
    else:
        return f"group {query_group} ({get_group_name(query_group)})"


def run_optimization_pipeline(
    db_path: str,
    query_group: int | list[int],
    bm25_random_samples: int = 50,
    output_file: str | None = None,
    metrics_k: list[int] | None = None,
    compute_interp_pr: bool = False
):
    """
    Run the complete optimization pipeline using ranx library.

    Args:
        db_path: Database path
        query_group: Query group to optimize for (single int or list of ints)
        bm25_random_samples: Number of random samples for BM25 search
        output_file: Output JSON file path (optional)
        metrics_k: Specific k values for metrics (e.g., [500] or [100, 500])
                   If None, uses all: [1, 3, 5, 10, 100, 500, 1000]
        compute_interp_pr: Compute 11-point interpolated precision-recall curve
    """
    logger.info("="*100)
    logger.info(f"STARTING OPTIMIZATION PIPELINE (RANX) FOR {get_group_display_name(query_group).upper()}")
    logger.info("="*100 + "\n")

    # Build metrics list based on k values
    metrics = build_metrics_list(metrics_k) if metrics_k else None
    if metrics_k:
        logger.info(f"Using custom k values: {metrics_k}")
        logger.info(f"Total metrics to compute: {len(metrics)}")

    if compute_interp_pr:
        logger.info("11-point interpolated precision-recall will be computed")

    # Step 1: Find best embedding variant
    best_embedding, all_embeddings = test_all_embeddings(db_path, query_group, metrics, compute_interp_pr)

    # Step 2: Find best BM25 variant with random search
    best_bm25, all_bm25 = test_all_bm25_random_search(db_path, query_group, bm25_random_samples, metrics, compute_interp_pr)

    # Step 3: Find best fusion method
    best_fusion, all_fusions = test_fusion_grid_search(best_bm25, best_embedding, db_path, query_group, metrics, compute_interp_pr)

    # Compile final results with ALL data
    final_results = {
        "query_group": query_group,
        "query_group_name": get_group_display_name(query_group) if isinstance(query_group, list) else get_group_name(query_group),
        "evaluation_library": "ranx",
        "metrics_k_values": metrics_k if metrics_k else [1, 3, 5, 10, 100, 500, 1000],
        "optimization_summary": {
            "best_embedding_variant": best_embedding['variant'],
            "best_embedding_map": best_embedding['primary_score'],
            "best_bm25_variant": best_bm25['variant'],
            "best_bm25_map": best_bm25['primary_score'],
            "best_fusion_method": best_fusion['fusion_method'],
            "best_fusion_map": best_fusion['primary_score'],
            "total_embeddings_tested": len(all_embeddings),
            "total_bm25_configs_tested": len(all_bm25),
            "total_fusion_configs_tested": len(all_fusions)
        },
        "best_embedding": best_embedding,
        "best_bm25": best_bm25,
        "best_fusion": best_fusion,
        "all_embeddings_results": all_embeddings,
        "all_bm25_results": all_bm25,
        "all_fusion_results": all_fusions
    }

    # Print final summary
    print("\n" + "="*100)
    print("FINAL OPTIMIZATION RESULTS (using ranx)")
    print("="*100)
    print(f"Query Group: {get_group_display_name(query_group)}")
    print("\n--- Best Embedding ---")
    print(f"Variant: {best_embedding['variant']}")
    print(f"Model: {best_embedding['model_name']}")
    print(f"MAP: {best_embedding['primary_score']:.4f}")
    print(f"MRR: {best_embedding['metrics']['mrr']:.4f}")
    if 'ndcg@10' in best_embedding['metrics']:
        print(f"nDCG@10: {best_embedding['metrics']['ndcg@10']:.4f}")
    print(f"Avg query time: {best_embedding['metrics'].get('query_time_mean_ms', 0):.2f} ms")

    print("\n--- Best BM25 ---")
    print(f"Variant: {best_bm25['variant']}")
    print(f"k1: {best_bm25['k1']}")
    print(f"b: {best_bm25['b']}")
    if best_bm25['delta'] is not None:
        print(f"delta: {best_bm25['delta']}")
    if best_bm25['alpha'] is not None:
        print(f"alpha: {best_bm25['alpha']}")
    if best_bm25['beta'] is not None:
        print(f"beta: {best_bm25['beta']}")
    print(f"MAP: {best_bm25['primary_score']:.4f}")
    print(f"MRR: {best_bm25['metrics']['mrr']:.4f}")
    if 'ndcg@10' in best_bm25['metrics']:
        print(f"nDCG@10: {best_bm25['metrics']['ndcg@10']:.4f}")
    print(f"Avg query time: {best_bm25['metrics'].get('query_time_mean_ms', 0):.2f} ms")

    print("\n--- Best Fusion ---")
    print(f"Method: {best_fusion['fusion_method']}")
    print(f"Normalization: {best_fusion['fusion_norm']}")
    print(f"Fusion K: {best_fusion['fusion_k']}")
    print(f"MAP: {best_fusion['primary_score']:.4f}")
    print(f"MRR: {best_fusion['metrics']['mrr']:.4f}")
    if 'ndcg@10' in best_fusion['metrics']:
        print(f"nDCG@10: {best_fusion['metrics']['ndcg@10']:.4f}")
    print(f"Avg query time: {best_fusion['metrics'].get('query_time_mean_ms', 0):.2f} ms")
    print("="*100 + "\n")

    # Save to JSON
    if output_file is None:
        if isinstance(query_group, list):
            group_str = "".join(str(g) for g in sorted(query_group))
            output_file = f"results/optimizing_ranx_group{group_str}.json"
        else:
            output_file = f"results/optimizing_ranx_group{query_group}.json"

    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    with open(output_path, 'w') as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Total data points saved: {len(all_embeddings)} embeddings, {len(all_bm25)} BM25 configs, {len(all_fusions)} fusion configs")

    return final_results


if __name__ == "__main__":
    from settings import get_settings

    settings = get_settings()

    # Check if optimization pipeline should run
    if not settings.optimize_run:
        logger.error("Please use --optimize_run True to run the optimization pipeline")
        logger.info("Usage: uv run src/benchmarks/optimize_pipeline_ranx.py --optimize_run True --query_group 3")
        logger.info("For multiple groups: --query_group 1,2,3")
        sys.exit(1)

    if settings.query_group is None:
        logger.error("Please specify --query_group (0, 1, 2, 3, or comma-separated like 1,2,3)")
        sys.exit(1)

    # Parse query_group - support comma-separated values for multiple groups
    run_optimization_pipeline(
        db_path=settings.database,
        query_group=settings.query_group,
        bm25_random_samples=settings.optimize_bm25_samples,
        output_file=settings.optimize_output,
        metrics_k=settings.optimize_metrics_k,
        compute_interp_pr=settings.compute_interp_pr
    )
