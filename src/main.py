# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb",
#     "langchain-text-splitters",
#     "sentence-transformers",
#     "tiktoken",
#     "tqdm",
#     "torch",
#     "pydantic-settings",
# ]
# ///

import tiktoken
import duckdb
from duckdb import DuckDBPyConnection
from tqdm import tqdm

import queries
from settings import get_settings, Settings, get_logger
from utils import (
    create_chunks,
    create_embedding_model,
    create_embeddings
)

logger = get_logger(__name__)

def chunking(conn: DuckDBPyConnection):
    logger.info("Buscando textos...")
    texts = conn.execute(queries.GET_TEXTS).fetchall()
    logger.info(f"{len(texts)} textos para processar")

    encoding = tiktoken.encoding_for_model("gpt-4o")

    batch = []
    batch_size = 100_000
    for id_auto, text in tqdm(texts, desc="Segmentando os textos"):
        chunks = create_chunks(text)
        for i, chunk in enumerate(chunks):
            tokens = len(encoding.encode(chunk))
            batch.append(
                {"id_auto": id_auto, "id_chunk": i, "tokens": tokens, "texto": chunk}
            )

        if len(batch) >= batch_size:
            conn.execute(queries.INSERT_CHUNK, params=batch)
            batch.clear()

    if batch:
        conn.execute(queries.INSERT_CHUNK, params=batch)


def chunk_embedding(conn: DuckDBPyConnection):
    model = create_embedding_model()
    chunks = conn.execute(queries.GET_CHUNKS).fetchall()

    def process_batch(batch: list[dict], texts: list[str]):
        embeddings = create_embeddings(texts, model)
        for i, embedding in enumerate(embeddings):
            batch[i]["embedding"] = embedding

        conn.execute(queries.INSERT_CHUNK_EMBEDDING, params=batch)
        batch.clear()
        texts.clear()

    batch = []
    texts = []
    batch_size = 100

    for id_auto, id_chunk, text in tqdm(chunks, desc="Processando chunks"):
        texts.append(text)
        batch.append({"id_auto": id_auto, "id_chunk": id_chunk})

        if len(batch) >= batch_size:
            process_batch(batch, texts)

    if batch:
        logger.info("Processando últimos chunks...")
        process_batch(batch, texts)


def main():
    settings: Settings = get_settings()
    conn: DuckDBPyConnection = duckdb.connect(settings.DUCKDB_FILE)

    logger.info("Criando tabelas caso não existam...")
    conn.execute(queries.CREATE_CHUNKS_TABLE)
    conn.execute(queries.CREATE_CHUNKS_EMBEDDINGS_TABLE)
    conn.execute(queries.CREATE_EMBEDDINGS_TABLE)

    logger.info("Iniciando a segmentação...")
    chunking(conn)
    logger.info("Iniciando o embedding dos segmentos...")
    chunk_embedding(conn)

    logger.info("Agregando os embeddings...")
    conn.execute(queries.AGGREGATE_MEAN_POOLING)
    logger.info("Finalizando...")
    conn.close()


if __name__ == "__main__":
    main()
