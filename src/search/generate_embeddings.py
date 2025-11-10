# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb",
#     "langchain-text-splitters",
#     "sentence-transformers",
#     "tiktoken",
#     "tqdm",
#     "torch>=2.2",
#     "pydantic-settings",
#     "prettytable",
#     "einops",
#     "nltk",
# ]
# ///

import duckdb
import tiktoken
import sys
from pathlib import Path
from duckdb import DuckDBPyConnection
from tqdm import tqdm
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import queries
from settings import EmbeddingVariant
from settings import Settings, get_logger, get_settings

# Import directly from module to avoid circular imports
if __name__ == "__main__":
    from search.utils import create_chunks, create_embedding_model, create_embeddings, _embed_text
else:
    from .utils import create_chunks, create_embedding_model, create_embeddings, _embed_text

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


def chunk_embedding(
    model: SentenceTransformer, variant: EmbeddingVariant, conn: DuckDBPyConnection
):
    chunks = conn.execute(queries.GET_PENDING_CHUNKS, [variant]).fetchall()

    def process_batch(entries: list[tuple[int, int]], texts: list[str]):
        if not entries:
            return
        embeddings = create_embeddings(texts, model)
        print(len(embeddings))
        rows = [
            (int(docid), int(chunk_index), embedding[:768], variant)
            for (docid, chunk_index), embedding in zip(entries, embeddings, strict=True)
        ]
        conn.executemany(queries.INSERT_CHUNK_EMBEDDING, rows)
        entries.clear()
        texts.clear()

    batch: list[tuple[int, int]] = []
    texts: list[str] = []
    batch_size = 1_000

    for docid, chunk_index, text in tqdm(chunks, desc="Gerando embeddings"):
        texts.append(text)
        batch.append((docid, chunk_index))

        if len(batch) >= batch_size:
            process_batch(batch, texts)

    if batch:
        logger.info("Processando batch restante..")
        process_batch(batch, texts)


def build_embeddings(
    model: SentenceTransformer, variant: EmbeddingVariant, db_filepath: str
):
    conn: DuckDBPyConnection = duckdb.connect(db_filepath)

    try:
        logger.info("Criando tabelas caso não existam...")
        conn.execute(queries.CREATE_CHUNKS_TABLE)
        conn.execute(queries.CREATE_CHUNKS_EMBEDDINGS_TABLE)
        conn.execute(queries.CREATE_EMBEDDINGS_TABLE)

        logger.info("Iniciando a segmentação...")
        chunking(conn)
        logger.info("Iniciando o embedding dos segmentos...")
        chunk_embedding(model=model, variant=variant, conn=conn)
        logger.info("Agregando os embeddings...")
        conn.execute(queries.AGGREGATE_MEAN_POOLING, [variant, variant])
    finally:
        conn.close()


def query_embeddings(
        query: str, k: int, model: SentenceTransformer, variant: EmbeddingVariant, db_filepath: str
) -> list[dict[str, Any]]:
    conn: DuckDBPyConnection = duckdb.connect(db_filepath, read_only=True)

    try:
        logger.info("Pesquisando embeddings mais similares...")
        query_embedding: list[float] = _embed_text(query, model)
        res = conn.execute(
                queries.SEARCH_EMBEDDING_TEXTO, [query_embedding[:768], variant, k]
        ).fetchall()
        # similaridade sendo ignorada por enquanto
        res_dict: list[dict[str, Any]] = [
            {"rank": i, "docid": d, "texto": t, "score": s}
            for i, (d, t, s) in enumerate(res, start=1)
        ]
        return res_dict
    finally:
        conn.close()


def show_results(results: list[dict[str, Any]], variant: str):
    from prettytable import PrettyTable
    if not results:
        logger.warning("Nenhum resultado encontrado.")
        return

    table = PrettyTable()
    table.field_names = ["DocID", "Conteúdo"]
    logger.info("Results for %s embeddings..", variant)
    for item in results:
        content: str = (t := item.get("texto") or "")[:100] + ("..." if len(t) > 100 else "")
        table.add_row([item.get("docid"), content])

    print(table)


if __name__ == "__main__":
    settings: Settings = get_settings()

    if settings.build:
        # Build multiple variants if embedding_variants is specified
        if settings.embedding_variants:
            logger.info(f"Building {len(settings.embedding_variants)} embedding variants: {settings.embedding_variants}")
            for variant in settings.embedding_variants:
                logger.info(f"\n{'='*60}")
                logger.info(f"Building variant: {variant}")
                logger.info(f"{'='*60}")
                # Create model for each variant
                model: SentenceTransformer = create_embedding_model(variant=variant)
                build_embeddings(
                    model=model,
                    variant=variant,
                    db_filepath=settings.database,
                )
            logger.info(f"\n{'='*60}")
            logger.info(f"All {len(settings.embedding_variants)} variants built successfully!")
            logger.info(f"{'='*60}")
        else:
            # Build single variant (default)
            logger.info("Iniciando a construção dos embeddings...")
            model: SentenceTransformer = create_embedding_model()
            build_embeddings(
                model=model,
                variant=settings.embedding_variant,
                db_filepath=settings.database,
            )

    if settings.query:
        variant: str = settings.embedding_variant
        model: SentenceTransformer = create_embedding_model(variant=variant)
        results: list[dict[str, Any]] = query_embeddings(
            query=settings.query,
            k=settings.k,
            model=model,
            variant=variant,
            db_filepath=settings.database,
        )
        show_results(results, variant)
