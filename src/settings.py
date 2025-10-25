from pydantic_settings import BaseSettings
from pydantic import SecretStr

from functools import lru_cache
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("script.log"),
        logging.StreamHandler()
    ]
)

model_map: dict[str, str] = {
    "jina": "jinaai/jina-embeddings-v3",
    "alibaba": "Alibaba-NLP/gte-multilingual-base",
    "lamdec": "LAMDEC/gte-finetune-pgm"
}

class Settings(BaseSettings):
    DUCKDB_FILE: str = "dataset.duckdb"

    MODEL: str = "lamdec"
    EMBEDDING_MODEL: str = model_map[MODEL]
    EMBEDDING_DIM: int = 768
    HF_TOKEN: SecretStr

    CHUNK_SIZE: int = 1024
    CHUNK_OVERLAP: int = 0

    EMBEDDING_BATCH_SIZE: int = 100

    EMBEDDINGS_GPU_ID: int = 1

@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
