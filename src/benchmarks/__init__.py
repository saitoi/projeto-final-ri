"""Benchmarks module - IR evaluation for all search methods."""

from .main import (
    create_bm25_retriever,
    create_embedding_retriever,
    create_hybrid_retriever,
    compute_ir_metrics,
    load_data_from_duckdb,
)

__all__ = [
    "create_bm25_retriever",
    "create_embedding_retriever",
    "create_hybrid_retriever",
    "compute_ir_metrics",
    "load_data_from_duckdb",
]
