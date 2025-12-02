from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

from functools import lru_cache
from pathlib import Path
from typing import Literal
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("script.log"),
        logging.StreamHandler()
    ]
)

EmbeddingVariant = Literal["jina", "alibaba", "lamdec", "gemma", "qwen", "lamdec-qwen", "lamdec-gemma", "lamdec-gte"]
BM25Variant = Literal["robertson", "atire", "bm25l", "bm25+", "lucene", "pyserini", "bmx"]
ModelType = Literal["bm25", "embeddings", "hybrid"]
FusionMethod = Literal[
    "rrf", "min", "max", "med", "sum", "anz", "mnz", "gmnz",
    "isr", "log_isr", "logn_isr", "bordafuse", "w_bordafuse",
    "condorcet", "w_condorcet", "bayesfuse", "mapfuse", "posfuse",
    "mixed", "wmnz", "wsum", "rbc"
]
NormMethod = Literal["min-max", "max", "sum", "zmuv", "rank", "borda"]

EMBEDDING_MODELS: dict[str, str] = {
    "jina": "jinaai/jina-embeddings-v3",
    "alibaba": "Alibaba-NLP/gte-multilingual-base",
    "lamdec": "LAMDEC/gte-finetune-pgm",
    "gemma": "google/embeddinggemma-300m",
    "qwen": "Qwen/Qwen3-Embedding-0.6B",
    "lamdec-qwen": "LAMDEC/qwen-pgm-pairs",
    "lamdec-gemma": "LAMDEC/gemma-pgm-pairs",
    "lamdec-gte": "LAMDEC/gte-pgm-pairs",
}

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parent / ".env",
        env_ignore_empty=True,
        cli_parse_args=True,
        extra="ignore",
    )

    # Database
    database: Literal["dataset.duckdb", "ulysses.duckdb"] = Field(default="dataset.duckdb", description="Database file path")

    # Pre-processing
    preprocess: bool = False

    # Universal
    variant: ModelType = "bm25"
    k: int = 10
    query: str = ""
    build: bool = False

    # Hybrid search
    hybrid_variant: Literal["fusion", "ranker"] = "fusion"
    fusion_method: FusionMethod = "rrf"
    fusion_norm: NormMethod = "min-max"
    fusion_k: int = Field(default=60, description="K parameter for RRF and other fusion algorithms")

    # Embeddings
    embedding_variant: EmbeddingVariant = "lamdec"
    embedding_variants: list[EmbeddingVariant] | None = Field(
        default=None,
        description="Build multiple embedding variants at once (comma-separated: jina,alibaba,lamdec,gemma,qwen,lamdec-qwen,lamdec-gemma,lamdec-gte)"
    )
    embedding_dim: int = 768
    embedding_batch_size: int = 100
    embedding_gpu_id: int = 0

    chunk_size: int = 1024
    chunk_overlap: int = 0

    # Bm25 Related
    bm25_dir: str = "./bm25_models/bm25"
    bm25_variant: BM25Variant = "bm25l"
    bm25_variants: list[BM25Variant] | None = Field(
        default=None,
        description="Build multiple BM25 variants at once (comma-separated: robertson,atire,bm25l,bm25+,lucene,pyserini,bmx)"
    )
    k1: float = Field(default=2.0, description="BM25 k1 parameter")
    b: float = Field(default=.75, description="BM25 b parameter")
    delta: float = Field(default=1.5, description="BM25 delta parameter")

    # RM3 Query Expansion
    use_rm3: bool = Field(default=False, description="Enable RM3 pseudo-relevance feedback")
    rm3_fb_docs: int = Field(default=10, description="Number of feedback documents for RM3")
    rm3_fb_terms: int = Field(default=10, description="Number of expansion terms for RM3")
    rm3_original_query_weight: float = Field(default=0.5, description="Weight of original query in RM3 (0.0-1.0)")

    # Ranker (for hybrid search)
    ranker_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Reranker model to use. Options: BAAI/bge-reranker-base, Alibaba-NLP/gte-multilingual-reranker-base"
    )
    ranker_use_gpu: bool = True
    ranker_gpu_id: int = 0

    # Benchmarks
    query_group: int | list[int] | None = Field(
        default=None,
        choices=[1, 2, 3],
        description="Filter queries by group (3=LLM, 1=search log, 2=expression from LLM question). If not specified, runs for all groups."
    )
    grid_search: Literal["base", "random", "grid"] | None = Field(
        default=None,
        description="Hyperparameter search strategy: 'base' for default params, 'random' for random search, 'grid' for exhaustive grid search"
    )
    fusion_grid_search: bool = Field(
        default=False,
        description="Enable grid search over all fusion methods for hybrid search (only works with --variant hybrid)"
    )

    # Optimization pipeline parameters
    optimize_run: bool = Field(
        default=False,
        description="Run end-to-end optimization pipeline (tests all embeddings, BM25 variants, and fusion methods)"
    )
    optimize_bm25_samples: int = Field(
        default=50,
        description="Number of random samples for BM25 search in optimization pipeline"
    )
    optimize_output: str | None = Field(
        default=None,
        description="Output JSON file path for optimization pipeline results"
    )
    optimize_metrics_k: list[int] | None = Field(
        default=None,
        description="Specific k values for metrics (e.g., 500 or 100,500). If not specified, uses all: 1,3,5,10,100,500,1000"
    )
    compute_interp_pr: bool = Field(
        default=False,
        description="Compute 11-point interpolated precision-recall curve (classic TREC metric)"
    )

    hf_token: str

@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
