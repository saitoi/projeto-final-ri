from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import SecretStr
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("script.log"),
        logging.StreamHandler()
    ]
)

class Settings(BaseSettings):
    DUCKDB_FILE: str = "dataset.duckdb"

    EMBEDDING_MODEL: str = "LAMDEC/gte-finetune-pgm"
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
