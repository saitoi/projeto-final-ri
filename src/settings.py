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

EmbeddingVariant = Literal["jina", "alibaba", "lamdec"]
BM25Variant = Literal["robertson", "atire", "bm25l", "bm25+", "lucene"]
ModelType = Literal["bm25", "embeddings", "hybrid"]

EMBEDDING_MODELS: dict[str, str] = {
    "jina": "jinaai/jina-embeddings-v3",
    "alibaba": "Alibaba-NLP/gte-multilingual-base",
    "lamdec": "LAMDEC/gte-finetune-pgm"
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
    model: ModelType = "bm25"
    alpha: float = 0.5
    k: int = 10
    query: str = ""
    build: bool = False

    # Embeddings
    embedding_variant: EmbeddingVariant = "lamdec"
    embedding_dim: int = 768
    embedding_batch_size: int = 100
    embedding_gpu_id: int = 1
    hf_token: SecretStr | None = None

    chunk_size: int = 1024
    chunk_overlap: int = 0

    # Bm25 Related
    bm25_dir: str = "./bm25-model"
    bm25_variant: BM25Variant = "lucene"

@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
