# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bm25s",
#     "duckdb",
#     "pydantic-settings",
#     "nltk",
#     "prettytable",
# ]
# ///

import math
from typing import Iterable, Sequence, Any

import duckdb
from prettytable import PrettyTable

import queries
from generate_bm25 import build_bm25, query_bm25
from generate_embeddings import build_embeddings
from settings import get_logger, get_settings, Settings
from utils import create_embedding_model, create_embeddings

logger = get_logger("cli")


def show_results(results: Sequence[dict]) -> None:
    if not results:
        logger.info("Nenhum resultado encontrado.")
        return

    table = PrettyTable()
    table.field_names = ["Rank", "DocID", "Score", "Strategy", "Tema", "Subtema", "Snippet"]
    table.align["Snippet"] = "l"
    table.align["Tema"] = "l"
    table.align["Subtema"] = "l"
    table.max_width["Snippet"] = 60
    table.max_width["Tema"] = 25
    table.max_width["Subtema"] = 25

    for item in results:
        snippet = item.get("enunciado") or item.get("excerto") or item.get("text") or ""
        snippet = snippet[:100] + "..." if len(snippet) > 100 else snippet

        table.add_row([
            item.get("rank", ""),
            item.get("docid", ""),
            f"{item.get('score', 0.0):.4f}",
            item.get("strategy", "")[:10],
            (item.get("tema") or "")[:25],
            (item.get("subtema") or "")[:25],
            snippet,
        ])

    print(table)


def normalize_scores(results: Iterable[dict]) -> dict[int, float]:
    scores = [float(item["score"]) for item in results if "score" in item]
    if not scores:
        return {}

    max_score = max(scores)
    min_score = min(scores)
    if math.isclose(max_score, min_score):
        return {int(item["docid"]): 1.0 for item in results}

    span = max_score - min_score
    return {int(item["docid"]): (float(item["score"]) - min_score) / span for item in results}


def query_embeddings_docs(
    conn: duckdb.DuckDBPyConnection,
    query_text: str,
    k: int,
    *,
    model,
    query_vector: list[float] | None = None,
) -> list[dict]:
    rows = conn.execute(queries.GET_DOC_EMBEDDINGS).fetchall()
    if not rows:
        raise RuntimeError("Nenhum embedding encontrado. Execute com --build primeiro.")

    if query_vector is None:
        query_vector = create_embeddings([query_text], model)[0]

    results: list[dict] = []
    for (
        docid,
        embedding,
        texto,
        tema,
        subtema,
        enunciado,
        excerto,
    ) in rows:
        score = sum(q * d for q, d in zip(query_vector, embedding, strict=True))
        results.append(
            {
                "docid": int(docid),
                "score": float(score),
                "text": texto or "",
                "tema": tema,
                "subtema": subtema,
                "enunciado": enunciado,
                "excerto": excerto,
                "strategy": "embeddings",
            }
        )

    results.sort(key=lambda item: item["score"], reverse=True)
    for idx, item in enumerate(results[:k], start=1):
        item["rank"] = idx
    return results[:k]


def query_hybrid(
    conn: duckdb.DuckDBPyConnection,
    query_text: str,
    *,
    retriever,
    model,
    alpha: float,
    k: int,
    augment_factor: int = 3,
) -> list[dict]:
    if not 0 <= alpha <= 1:
        raise ValueError("O parâmetro alpha deve estar entre 0 e 1.")

    bm25_k = max(k, 1) * augment_factor
    bm25_results = query_bm25(retriever, query_text, k=bm25_k)
    for item in bm25_results:
        item["strategy"] = "bm25"

    query_vector = create_embeddings([query_text], model)[0]
    embedding_results = query_embeddings_docs(
        conn,
        query_text,
        bm25_k,
        model=model,
        query_vector=query_vector,
    )

    bm25_scores = normalize_scores(bm25_results)
    embedding_scores = normalize_scores(embedding_results)

    combined: dict[int, dict] = {}
    for docid, score in bm25_scores.items():
        combined.setdefault(docid, {"bm25": 0.0, "embeddings": 0.0})
        combined[docid]["bm25"] = score
    for docid, score in embedding_scores.items():
        combined.setdefault(docid, {"bm25": 0.0, "embeddings": 0.0})
        combined[docid]["embeddings"] = score

    ranked = sorted(
        (
            (
                docid,
                alpha * scores.get("bm25", 0.0) + (1 - alpha) * scores.get("embeddings", 0.0),
            )
            for docid, scores in combined.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    top_docids = [docid for docid, _ in ranked[:k]]
    score_map = dict(ranked)

    # Buscar metadados dos resultados BM25 e embeddings
    metadata_map = {}
    for item in bm25_results + embedding_results:
        if item["docid"] not in metadata_map:
            metadata_map[item["docid"]] = {
                "text": item.get("text", ""),
                "tema": item.get("tema"),
                "subtema": item.get("subtema"),
                "enunciado": item.get("enunciado"),
                "excerto": item.get("excerto"),
            }

    hybrid_results: list[dict] = []
    for rank, docid in enumerate(top_docids, start=1):
        meta = metadata_map.get(docid, {})
        hybrid_results.append(
            {
                "docid": docid,
                "rank": rank,
                "score": score_map.get(docid, 0.0),
                "text": meta.get("text", ""),
                "tema": meta.get("tema"),
                "subtema": meta.get("subtema"),
                "enunciado": meta.get("enunciado"),
                "excerto": meta.get("excerto"),
                "bm25_component": bm25_scores.get(docid, 0.0),
                "embedding_component": embedding_scores.get(docid, 0.0),
                "strategy": "hybrid",
            }
        )

    return hybrid_results


def main() -> None:
    settings: Settings = get_settings()
    conn = duckdb.connect(settings.database)

    try:
        # Preprocessamento
        if settings.preprocess:
            logger.info("Executando preprocessamento...")
            from preprocess import run as preprocess_run
            preprocess_run(settings.database)
            logger.info("Preprocessamento concluído.")

        if settings.build:
            logger.info(f"Construindo índice {settings.model}...")
            if settings.model == "bm25":
                build_bm25(
                    conn=conn,
                    model_dir=settings.bm25_dir,
                    variant=settings.bm25_variant,
                    build=True,
                )
                logger.info("Índice BM25 construído.")
            elif settings.model == "embeddings":
                build_embeddings(settings.database)
                logger.info("Embeddings gerados.")
            else:
                raise ValueError(f"Modelo desconhecido: {settings.model}")

        # Query
        if settings.query:
            logger.info(f"Executando query: {settings.query}")

            if settings.model == "bm25":
                retriever = build_bm25(
                    conn=conn,
                    model_dir=settings.bm25_dir,
                    variant=settings.bm25_variant,
                    build=False,
                )
                results: list[dict[str, Any]] = query_bm25(retriever, settings.query, k=settings.k)
                show_results(results)
                for item in results:
                    item["strategy"] = "bm25"

            elif settings.model == "embeddings":
                model = create_embedding_model()
                results = query_embeddings_docs(conn, settings.query, settings.k, model=model)

            elif settings.model == "hybrid":
                model = create_embedding_model()
                retriever = build_bm25(
                    conn=conn,
                    model_dir=settings.bm25_dir,
                    variant=settings.bm25_variant,
                    build=False,
                )
                results = query_hybrid(
                    conn,
                    settings.query,
                    retriever=retriever,
                    model=model,
                    alpha=settings.alpha,
                    k=settings.k,
                )
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
