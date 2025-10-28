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

import duckdb
import tiktoken
from duckdb import DuckDBPyConnection
from tqdm import tqdm

import queries
from settings import Settings, get_logger, get_settings
from utils import create_chunks, create_embedding_model, create_embeddings, _embed_text
from sentence_transformers import SentenceTransformer

logger = get_logger(__name__)


def chunking(conn: DuckDBPyConnection):
    logger.info("Buscando textos para segmentação...")
    texts = conn.execute(queries.GET_DOCS_FOR_CHUNKING).fetchall()
    logger.info("Processando %s documentos", len(texts))

    encoding = tiktoken.encoding_for_model("gpt-4o")
    batch: list[tuple[int, int, int, str]] = []
    batch_size = 100_000

    for docid, text in tqdm(texts, desc="Segmentando os textos"):
        if not text or not text.strip():
            continue
        chunks = create_chunks(text)
        for idx, chunk in enumerate(chunks):
            tokens = len(encoding.encode(chunk))
            batch.append((int(docid), idx, tokens, chunk))

        if len(batch) >= batch_size:
            conn.executemany(queries.INSERT_CHUNK, batch)
            batch.clear()

    if batch:
        conn.executemany(queries.INSERT_CHUNK, batch)


def chunk_embedding(model: SentenceTransformer, conn: DuckDBPyConnection):
    chunks = conn.execute(queries.GET_PENDING_CHUNKS).fetchall()

    def process_batch(entries: list[tuple[int, int]], texts: list[str]):
        if not entries:
            return
        embeddings = create_embeddings(texts, model)
        rows = [
            (int(docid), int(chunk_index), embedding)
            for (docid, chunk_index), embedding in zip(entries, embeddings, strict=True)
        ]
        conn.executemany(queries.INSERT_CHUNK_EMBEDDING, rows)
        entries.clear()
        texts.clear()

    batch: list[tuple[int, int]] = []
    texts: list[str] = []
    batch_size = 100

    for docid, chunk_index, text in tqdm(chunks, desc="Gerando embeddings"):
        texts.append(text)
        batch.append((docid, chunk_index))

        if len(batch) >= batch_size:
            process_batch(batch, texts)

    if batch:
        logger.info("Processando batch restante..")
        process_batch(batch, texts)

def build_embeddings(model: SentenceTransformer, db_filepath: str):
    conn: DuckDBPyConnection = duckdb.connect(db_filepath)

    try:
        logger.info("Criando tabelas caso não existam...")
        conn.execute(queries.CREATE_CHUNKS_TABLE)
        conn.execute(queries.CREATE_CHUNKS_EMBEDDINGS_TABLE)
        conn.execute(queries.CREATE_EMBEDDINGS_TABLE)

        logger.info("Iniciando a segmentação...")
        chunking(conn)
        logger.info("Iniciando o embedding dos segmentos...")
        chunk_embedding(model, conn)
        logger.info("Agregando os embeddings...")
        conn.execute(queries.AGGREGATE_MEAN_POOLING)
    finally:
        conn.close()

def query_embeddings(query: str, model: SentenceTransformer, db_filepath: str):
    conn: DuckDBPyConnection = duckdb.connect(db_filepath)

    try:
        logger.info("Pesquisando embeddings mais similares...")
        query_embedding: list[float] = _embed_text(query, model)
        res = conn.execute(queries.SEARCH_EMBEDDING_TEXTO, [query_embedding]).fetchall()
        return res
    finally:
        conn.close()

if __name__ == "__main__":
    settings: Settings = get_settings()

    model = create_embedding_model()

    if settings.build:
        build_embeddings(model=model, db_filepath=settings.database)

    if settings.query:
        res = query_embeddings(query=settings.query, model=model, db_filepath=settings.database)
        print(res)
