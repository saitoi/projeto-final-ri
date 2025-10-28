# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bm25s",
#     "nltk",
#     "duckdb",
#     "tqdm",
#     "prettytable",
#     "pydantic-settings",
# ]
# ///

import pathlib
import shutil
from typing import Any

import bm25s
import duckdb
from duckdb import DuckDBPyConnection
from nltk.stem import SnowballStemmer
from tqdm.auto import tqdm

import queries
from settings import Settings, get_logger, get_settings

# Wrapper do SnowballStemmer para o bm25s
class StemmerWrapper:
    def __init__(self, stemmer):
        self.stemmer = stemmer

    def stemWord(self, word):
        return self.stemmer.stem(word)

    def __call__(self, words):
        if isinstance(words, list):
            return [self.stemmer.stem(word) for word in words]
        return self.stemmer.stem(words)

stemmer = StemmerWrapper(SnowballStemmer("portuguese"))
logger = get_logger(__name__)

def load_corpus(conn: DuckDBPyConnection) -> list[dict]:
    rows = conn.execute(queries.GET_DOC_TEXTS).fetchall()
    total = len(rows)

    corpus: list[dict] = []
    pbar = tqdm(rows, total=total or None, desc="Loading corpus", unit="doc")
    for docid, texto, tema, subtema, enunciado, excerto in pbar:
        corpus.append({
            "docid": int(docid),
            "text": texto,
            "tema": tema,
            "subtema": subtema,
            "enunciado": enunciado,
            "excerto": excerto,
        })

    return corpus


def build_bm25(
    conn: DuckDBPyConnection, model_dir: str, variant: str = "lucene", *, build: bool = False
) -> bm25s.BM25:
    model_path = pathlib.Path(model_dir)

    if not build and model_path.exists():
        logger.info("Loading BM25 retriever from %s", model_path)
        return bm25s.BM25.load(model_dir, load_corpus=True)

    if model_path.exists():
        logger.info("Removing existing BM25 model at %s", model_path)
        shutil.rmtree(model_path)

    logger.info("Building BM25 retriever (variant=%s)...", variant)
    corpus = load_corpus(conn=conn)
    corpus_text = [doc["text"] for doc in corpus]
    corpus_tokens = bm25s.tokenize(corpus_text, stopwords="pt", stemmer=stemmer)

    retriever = bm25s.BM25(corpus=corpus, method=variant)
    retriever.index(corpus_tokens)
    model_path.mkdir(parents=True, exist_ok=True)
    retriever.save(model_dir)
    logger.info("BM25 retriever saved to %s", model_path)

    return retriever


def query_bm25(retriever, query_text: str, k: int = 10) -> list[dict[str, Any]]:
    query_tokens = bm25s.tokenize(query_text, stopwords="pt", stemmer=stemmer)
    results, scores = retriever.retrieve(query_tokens, k=k)

    matches: list[dict] = []
    for rank in range(results.shape[1]):
        doc = results[0, rank]
        score = float(scores[0, rank])
        matches.append({
            "rank": rank + 1,
            "docid": doc["docid"],
            "score": score,
            "text": doc.get("text", ""),
            "tema": doc.get("tema"),
            "subtema": doc.get("subtema"),
            "enunciado": doc.get("enunciado"),
            "excerto": doc.get("excerto"),
        })

    return matches


def show_results(results: list[dict]) -> None:
    from prettytable import PrettyTable
    if not results:
        logger.info("Nenhum resultado encontrado.")
        return

    table = PrettyTable()
    table.field_names = ["Rank", "DocID", "Score", "Conteúdo"]
    # table.max_width["Conteúdo"] = 80

    for item in results:
        content: str = item.get("text") or ""
        content: str = content[:100] + "..." if len(content) > 100 else content

        table.add_row([
            item.get("rank", ""),
            item.get("docid", ""),
            f"{item.get('score', 0.0):.4f}",
            # item.get("tema", "")[:30] if item.get("tema") else "",
            # item.get("subtema", "")[:30] if item.get("subtema") else "",
            content,
        ])

    print(table)

if __name__ == "__main__":
    settings: Settings = get_settings()
    conn: DuckDBPyConnection = duckdb.connect(settings.database)

    try:
        if settings.build:
            build_bm25(
                conn=conn,
                model_dir=settings.bm25_dir,
                variant=settings.bm25_variant,
                build=True,
            )

        if settings.query:
            retriever: bm25s.BM25 = build_bm25(
                conn=conn,
                model_dir=settings.bm25_dir,
                variant=settings.bm25_variant,
                build=False,
            )
            results: list[dict[str, Any]] = query_bm25(retriever, settings.query, k=settings.k)
            show_results(results)
    finally:
        conn.close()
