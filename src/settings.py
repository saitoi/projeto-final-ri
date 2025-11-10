from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import SecretStr, Field

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
BM25Variant = Literal["robertson", "atire", "bm25l", "bm25+", "lucene", "pyserini"]
ModelType = Literal["bm25", "embeddings", "hybrid"]

EMBEDDING_MODELS: dict[str, str] = {
    "jina": "jinaai/jina-embeddings-v3",
    "alibaba": "Alibaba-NLP/gte-multilingual-base",
    "lamdec": "LAMDEC/gte-finetune-pgm", # trocar para lamdec-alibaba
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
    database: str = Field(default="dataset.duckdb", description="Database file path")

    # Pre-processing
    preprocess: bool = False

    # Universal
    variant: ModelType = "bm25"
    k: int = 10
    query: str = ""
    build: bool = False

    # Hybrid search
    hybrid_variant: Literal["rrf", "ranker"] = "rrf"

    # Embeddings
    embedding_variant: EmbeddingVariant = "lamdec"
    embedding_variants: list[EmbeddingVariant] | None = Field(
        default=None,
        description="Build multiple embedding variants at once (comma-separated: jina,alibaba,lamdec,gemma,qwen,lamdec-qwen,lamdec-gemma,lamdec-gte)"
    )
    embedding_dim: int = 768
    embedding_batch_size: int = 100
    embedding_gpu_id: int = 0
    hf_token: SecretStr | None = None

    chunk_size: int = 1024
    chunk_overlap: int = 0

    # Bm25 Related
    bm25_dir: str = "./bm25_models/bm25"
    bm25_variant: BM25Variant = "lucene"
    bm25_variants: list[BM25Variant] | None = Field(
        default=None,
        description="Build multiple BM25 variants at once (comma-separated: robertson,atire,bm25l,bm25+,lucene)"
    )
    k1: float = Field(default=.5, description="BM25 k1 parameter")
    b: float = Field(default=.75, description="BM25 b parameter")
    # delta: float = Field(default=0.5, description="BM25 delta parameter")

    # RM3 Query Expansion
    use_rm3: bool = Field(default=False, description="Enable RM3 pseudo-relevance feedback")
    rm3_fb_docs: int = Field(default=10, description="Number of feedback documents for RM3")
    rm3_fb_terms: int = Field(default=10, description="Number of expansion terms for RM3")
    rm3_original_query_weight: float = Field(default=0.5, description="Weight of original query in RM3 (0.0-1.0)")

    # Ranker (for hybrid search)
    ranker_model: str = "gte-multilingual-reranker-base"
    ranker_cache_dir: str = "./rankers/gte-multilingual-onnx"

    # Benchmarks
    query_group: int | None = Field(
        default=None,
        choices=[1, 2, 3],
        description="Filter queries by group (3=LLM, 1=search log, 2=expression from LLM question). If not specified, runs for all groups."
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
