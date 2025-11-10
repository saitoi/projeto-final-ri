# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "bm25s",
#     "nltk",
#     "duckdb",
#     "tqdm",
#     "prettytable",
#     "pydantic-settings",
#     "pyserini",
# ]
# ///

import pathlib
import shutil
import sys
from pathlib import Path
from typing import Any, Union

import nltk
import bm25s
import duckdb
from duckdb import DuckDBPyConnection
from nltk.stem import RSLPStemmer
from tqdm.auto import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import queries
from settings import Settings, get_logger, get_settings

nltk.download('rslp')

# Wrapper do RSLPStemmer para o bm25s
class StemmerWrapper:
    def __init__(self, stemmer):
        self.stemmer = stemmer

    def stemWord(self, word):
        return self.stemmer.stem(word)

    def __call__(self, words):
        if isinstance(words, list):
            return [self.stemmer.stem(word) for word in words]
        return self.stemmer.stem(words)

stemmer = StemmerWrapper(RSLPStemmer())
logger = get_logger(__name__)

def load_corpus(conn: DuckDBPyConnection) -> list[dict]:
    rows = conn.execute(queries.GET_DOC_TEXTS).fetchall()
    total = len(rows)

    corpus: list[dict] = []
    pbar = tqdm(rows, total=total or None, desc="Loading corpus", unit="doc")
    for docid, texto, tema, subtema, enunciado, excerto in pbar:
        corpus.append({
            "docid": int(docid),
            "texto": texto,
            "tema": tema,
            "subtema": subtema,
            "enunciado": enunciado,
            "excerto": excerto,
        })

    return corpus


def build_bm25(
    conn: DuckDBPyConnection, model_dir: str, variant: str = "lucene", *, build: bool = False,
    k1: float = 1.5, b: float = 0.75, delta: float = 0.5
) -> bm25s.BM25 | Any:
    """
    Build or load BM25 retriever. Supports both bm25s and pyserini variants.

    For pyserini variant, returns a dict with:
        - 'searcher': pyserini LuceneSearcher instance
        - 'corpus': corpus data
        - 'doc_ids': document IDs mapping

    Args:
        conn: DuckDB connection
        model_dir: Directory to save/load model
        variant: BM25 variant (robertson, lucene, atire, bm25l, bm25+, pyserini)
        build: Force rebuild even if model exists
        k1: BM25 k1 parameter (term frequency saturation)
        b: BM25 b parameter (length normalization)
        delta: BM25 delta parameter (for bm25l and bm25+ only)
    """
    if variant == "pyserini":
        from _pyserini import build_pyserini
        corpus = load_corpus(conn=conn)
        return build_pyserini(corpus=corpus, model_dir=model_dir, build=build)

    model_path = pathlib.Path(model_dir + "_" + variant)

    if not build and model_path.exists():
        logger.info("Loading BM25 retriever from %s", model_path)
        return bm25s.BM25.load(str(model_path), load_corpus=True)

    if model_path.exists():
        logger.info("Removing existing BM25 model at %s", model_path)
        shutil.rmtree(model_path)

    logger.info("Building BM25 retriever (variant=%s, k1=%s, b=%s, delta=%s)...", variant, k1, b, delta)
    corpus = load_corpus(conn=conn)
    corpus_text = [doc["texto"] for doc in corpus]
    corpus_tokens = bm25s.tokenize(corpus_text, stopwords="pt", stemmer=stemmer)

    retriever = bm25s.BM25(corpus=corpus, method=variant, k1=k1, b=b, delta=delta)
    retriever.index(corpus_tokens)
    model_path.mkdir(parents=True, exist_ok=True)
    retriever.save(model_path)
    logger.info("BM25 retriever saved to %s", model_path)

    return retriever


def query_bm25(retriever: Union[bm25s.BM25, dict], query: str, k: int = 10,
               use_rm3: bool = False, rm3_fb_docs: int = 10, rm3_fb_terms: int = 10,
               rm3_original_query_weight: float = 0.5) -> list[dict[str, Any]]:
    """
    Query BM25 index. Supports both bm25s and pyserini retrievers.

    Args:
        retriever: BM25 retriever (bm25s.BM25 or pyserini dict)
        query: Search query
        k: Number of results
        use_rm3: Enable RM3 for pyserini variant
        rm3_fb_docs: Number of feedback documents for RM3
        rm3_fb_terms: Number of expansion terms for RM3
        rm3_original_query_weight: Original query weight for RM3

    Returns:
        List of matched documents
    """
    # Check if pyserini
    if isinstance(retriever, dict) and 'searcher' in retriever:
        from _pyserini import query_pyserini
        return query_pyserini(
            retriever=retriever,
            query=query,
            k=k,
            use_rm3=use_rm3,
            rm3_fb_docs=rm3_fb_docs,
            rm3_fb_terms=rm3_fb_terms,
            rm3_original_query_weight=rm3_original_query_weight
        )

    # bm25s variant
    query_tokens = bm25s.tokenize(query, stopwords="pt", stemmer=stemmer)
    results, scores = retriever.retrieve(query_tokens, k=k)

    matches: list[dict] = []
    for rank in range(results.shape[1]):
        doc = results[0, rank]
        score = float(scores[0, rank])
        matches.append({
            "rank": rank + 1,
            "docid": doc["docid"],
            "score": score,
            "texto": doc.get("texto", ""),
            "tema": doc.get("tema"),
            "subtema": doc.get("subtema"),
            "enunciado": doc.get("enunciado"),
            "excerto": doc.get("excerto"),
        })

    return matches


def show_results(results: list[dict]) -> None:
    from prettytable import PrettyTable
    if not results:
        logger.warning("Nenhum resultado encontrado.")
        return

    table = PrettyTable()
    table.field_names = ["Rank", "DocID", "Score", "Conteúdo"]
    for item in results:
        content: str = (t := item.get("texto") or "")[:100] + ("..." if len(t) > 100 else "")
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
    conn: DuckDBPyConnection = duckdb.connect(settings.database, read_only=True)

    try:
        if settings.build:
            # Build multiple variants if bm25_variants is specified
            if settings.bm25_variants:
                logger.info(f"Building {len(settings.bm25_variants)} BM25 variants: {settings.bm25_variants}")
                for variant in settings.bm25_variants:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Building variant: {variant}")
                    logger.info(f"{'='*60}")
                    build_bm25(
                        conn=conn,
                        model_dir=settings.bm25_dir,
                        variant=variant,
                        build=True,
                        k1=settings.k1,
                        b=settings.b,
                    )
                logger.info(f"\n{'='*60}")
                logger.info(f"All {len(settings.bm25_variants)} variants built successfully!")
                logger.info(f"{'='*60}")
            else:
                # Constrói uma variante (default)
                build_bm25(
                    conn=conn,
                    model_dir=settings.bm25_dir,
                    variant=settings.bm25_variant,
                    build=True,
                    k1=settings.k1,
                    b=settings.b,
                )

        if settings.query:
            retriever = build_bm25(
                conn=conn,
                model_dir=settings.bm25_dir,
                variant=settings.bm25_variant,
                build=False,
                k1=settings.k1,
                b=settings.b,
            )
            results: list[dict[str, Any]] = query_bm25(
                retriever=retriever,
                query=settings.query,
                k=settings.k,
                use_rm3=settings.use_rm3,
                rm3_fb_docs=settings.rm3_fb_docs,
                rm3_fb_terms=settings.rm3_fb_terms,
                rm3_original_query_weight=settings.rm3_original_query_weight,
            )
            show_results(results)
    finally:
        conn.close()
