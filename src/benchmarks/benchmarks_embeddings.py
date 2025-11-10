# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb",
#     "pydantic-settings",
#     "sentence-transformers",
#     "langchain-text-splitters",
#     "torch",
#     "tqdm",
# ]
# ///

import duckdb
import sys
from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer

# Add src to path to import settings
sys.path.insert(0, str(Path(__file__).parent))

from settings import EmbeddingVariant, EMBEDDING_MODELS, get_logger, get_settings
from utils import create_embedding_model

logger = get_logger(__name__)


def load_data_from_duckdb(
    db_path: str, embedding_variant: EmbeddingVariant
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """
    Load queries and relevance judgments from DuckDB.

    Returns:
        queries: dict mapping qid (str) to query text (str)
        relevant_docs: dict mapping qid (str) to set of relevant docids (str)
    """
    conn = duckdb.connect(db_path, read_only=True)

    try:
        # Count documents with embeddings
        logger.info(f"Counting documents with {embedding_variant} embeddings...")
        doc_count = conn.execute("""
            SELECT COUNT(*) FROM doc_embeddings WHERE model = ?
        """, [embedding_variant]).fetchone()[0]
        logger.info(f"Found {doc_count} documents with embeddings")

        # Load queries
        logger.info("Loading queries...")
        queries_data = conn.execute("""
            SELECT qid, text FROM queries
        """).fetchall()

        queries = {str(qid): text for qid, text in queries_data}
        logger.info(f"Loaded {len(queries)} queries")

        # Load relevance judgments (qrels)
        logger.info("Loading relevance judgments...")
        qrels_data = conn.execute("""
            SELECT qid, docid, score
            FROM queries_rel
            WHERE score > 0
        """).fetchall()

        relevant_docs: dict[str, set[str]] = {}
        for qid, docid, score in qrels_data:
            qid_str = str(qid)
            docid_str = str(docid)
            if qid_str not in relevant_docs:
                relevant_docs[qid_str] = set()
            relevant_docs[qid_str].add(docid_str)

        logger.info(f"Loaded {len(relevant_docs)} queries with relevance judgments")

        return queries, relevant_docs

    finally:
        conn.close()


def retrieve_with_embeddings(
    db_path: str,
    query_embedding: list[float],
    embedding_variant: EmbeddingVariant,
    k: int
) -> list[str]:
    """
    Retrieve top-k documents using DuckDB's array_cosine_similarity.

    Returns:
        List of docids ordered by similarity (descending)
    """
    conn = duckdb.connect(db_path, read_only=True)

    try:
        results = conn.execute("""
            SELECT de.docid
            FROM doc_embeddings de
            WHERE de.model = ?
            ORDER BY array_cosine_similarity(de.embedding, ?::float[768]) DESC
            LIMIT ?
        """, [embedding_variant, query_embedding[:768], k]).fetchall()

        return [str(docid) for docid, in results]

    finally:
        conn.close()


def compute_ir_metrics(
    queries: dict[str, str],
    relevant_docs: dict[str, set[str]],
    model: SentenceTransformer,
    db_path: str,
    embedding_variant: EmbeddingVariant,
    k_values: list[int] = [1, 3, 5, 10, 100, 1000]
) -> dict[str, float]:
    """
    Compute Information Retrieval metrics for embeddings.

    Returns metrics like Precision@k, Recall@k, NDCG@k, MAP, MRR@10
    Shallow metrics (top of ranking): P@10, MRR@10, nDCG@10
    Deep metrics (global consistency): MAP, Recall@1000, nDCG@1000
    """
    from tqdm import tqdm
    import math

    metrics: dict[str, list[float]] = defaultdict(list)

    for qid, query_text in tqdm(queries.items(), desc="Evaluating"):
        if qid not in relevant_docs:
            continue

        # Encode query
        query_embedding = model.encode(query_text, normalize_embeddings=True)
        query_embedding_list = query_embedding.tolist()

        # Retrieve using DuckDB (with cosine similarity computed in SQL)
        retrieved_docs = retrieve_with_embeddings(
            db_path=db_path,
            query_embedding=query_embedding_list,
            embedding_variant=embedding_variant,
            k=max(k_values)
        )

        # Ground truth
        ground_truth = relevant_docs[qid]

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

            # NDCG@k (métricas rasas: nDCG@10, métricas profundas: nDCG@1000)
            dcg = 0.0
            idcg = 0.0
            for i, docid in enumerate(retrieved_docs[:k], start=1):
                if docid in ground_truth:
                    dcg += 1.0 / math.log2(i + 1)
            for i in range(1, min(k, len(ground_truth)) + 1):
                idcg += 1.0 / math.log2(i + 1)

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


def main():
    settings = get_settings()
    variant: EmbeddingVariant = settings.embedding_variant

    logger.info(f"Starting Embeddings benchmark for variant: {variant}")
    logger.info(f"Model: {EMBEDDING_MODELS[variant]}")

    logger.info("Loading embedding model...")
    model = create_embedding_model()

    queries, relevant_docs = load_data_from_duckdb(settings.database, variant)

    logger.info("Running evaluation...")
    results = compute_ir_metrics(
        queries=queries,
        relevant_docs=relevant_docs,
        model=model,
        db_path=settings.database,
        embedding_variant=variant,
        k_values=[1, 3, 5, 10, 100]
    )

    print("\n" + "="*80)
    print(f"Embeddings Information Retrieval Evaluation: {variant}")
    print("="*80)
    print(f"Model: {EMBEDDING_MODELS[variant]}")
    print(f"Queries: {len(queries)}")
    print(f"Queries with relevance judgments: {len(relevant_docs)}")
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


if __name__ == "__main__":
    main()
