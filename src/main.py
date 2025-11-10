# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bm25s",
#     "duckdb",
#     "flashrank",
#     "langchain-text-splitters",
#     "nltk",
#     "prettytable",
#     "pydantic-settings",
#     "sentence-transformers",
#     "tiktoken",
#     "torch>=2.2",
# ]
# ///

import sys
from pathlib import Path
from typing import Sequence, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import duckdb
from prettytable import PrettyTable
from sentence_transformers import SentenceTransformer

from search.generate_bm25 import build_bm25, query_bm25
from search.generate_embeddings import query_embeddings
from search.utils import create_embedding_model
from search.hybrid import query_hybrid
from settings import get_logger, get_settings, Settings

logger = get_logger("cli")


def show_results(results: Sequence[dict]) -> None:
    if not results:
        logger.info("Nenhum resultado encontrado.")
        return

    table = PrettyTable()
    table.field_names = ["Rank", "DocID", "Variante", "Tema", "Snippet"]
    table.align["Snippet"] = "l"
    table.align["Tema"] = "l"
    table.align["Subtema"] = "l"
    table.max_width["Tema"] = 25
    table.max_width["Subtema"] = 25

    for item in results:
        snippet = item.get("enunciado") or item.get("excerto") or item.get("text") or ""
        snippet = snippet[:100] + "..." if len(snippet) > 100 else snippet

        table.add_row([
            item.get("rank", ""),
            item.get("docid", ""),
            item.get("variant", ""),
            (item.get("tema") or "")[:25],
            snippet,
        ])

    print(table)


def main() -> None:
    settings: Settings = get_settings()

    # Use read_only=True when not preprocessing (only reading corpus for BM25/queries)
    read_only = not settings.preprocess
    conn = duckdb.connect(settings.database, read_only=read_only)

    try:

        # ******** Pré-processamento ********

        if settings.preprocess:
            logger.info("Executando preprocessamento...")
            from preprocess import run as preprocess_run
            preprocess_run(settings.database)
            logger.info("Preprocessamento concluído.")

        # ******** Variantes ********

        if settings.variant == "embeddings":
            model: SentenceTransformer = create_embedding_model()
            logger.info("Modelo de embeddings instanciado...")
        elif settings.variant == "bm25":
            retriever = build_bm25(
                conn=conn,
                model_dir=settings.bm25_dir,
                variant=settings.bm25_variant,
                build=settings.build,
                k1=settings.k1,
                b=settings.b,
            )
        elif settings.variant == "hybrid":
            # Hybrid search will load models internally
            logger.info("Hybrid search selected - models will be loaded on query...")
        else:
            logger.info("Nenhum modelo selecionado.")

        # ******** Consulta ********

        if settings.query:
            logger.info(f"Executando query: {settings.query}")

            if settings.variant == "bm25":
                results: list[dict[str, Any]] = query_bm25(
                    retriever=retriever,
                    query=settings.query,
                    k=settings.k
                )
                for item in results:
                    item["variant"] = settings.bm25_variant

            elif settings.variant == "embeddings":
                results = query_embeddings(
                    query=settings.query,
                    k=settings.k,
                    model=model,
                    variant=settings.embedding_variant,
                    db_filepath=settings.database
                )
                for item in results:
                    item["variant"] = settings.embedding_variant

            elif settings.variant == "hybrid":
                results = query_hybrid(
                    query=settings.query,
                    k=settings.k,
                    bm25_variant=settings.bm25_variant,
                    embedding_variant=settings.embedding_variant,
                    db_filepath=settings.database,
                    bm25_dir=settings.bm25_dir,
                    k1=settings.k1,
                    b=settings.b,
                )
                for item in results:
                    item["variant"] = f"{settings.bm25_variant}+{settings.embedding_variant}"
            else:
                raise ValueError(f"Modelo desconhecido: {settings.model}")

            show_results(results)

    except RuntimeError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc
    finally:
        conn.close()


if __name__ == "__main__":
    main()
