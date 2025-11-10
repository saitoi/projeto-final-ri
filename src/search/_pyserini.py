import pathlib
import pickle
import shutil
import subprocess
import sys
from typing import Any

from tqdm.auto import tqdm

from settings import get_logger

logger = get_logger(__name__)


def build_pyserini(corpus: list[dict], model_dir: str, *, build: bool = False) -> dict:
    """
    Build or load pyserini BM25 retriever.

    Args:
        corpus: List of documents with fields (docid, texto, tema, subtema, enunciado, excerto)
        model_dir: Base directory for models
        build: If True, force rebuild even if index exists

    Returns:
        Dict with searcher, corpus, and doc_ids
    """
    from pyserini.search.lucene import LuceneSearcher
    import json

    model_path = pathlib.Path(model_dir + "-pyserini")
    index_path = model_path / "index"
    corpus_file = model_path / "corpus.pkl"

    # Load existing index
    if not build and index_path.exists() and corpus_file.exists():
        logger.info("Loading pyserini retriever from %s", model_path)
        with open(corpus_file, 'rb') as f:
            corpus_data = pickle.load(f)

        searcher = LuceneSearcher(str(index_path))
        logger.info("Loaded pyserini index with %d documents", searcher.num_docs)

        return {
            'searcher': searcher,
            'corpus': corpus_data['corpus'],
            'doc_ids': corpus_data['doc_ids'],
        }

    # Clean existing index
    if model_path.exists():
        logger.info("Removing existing pyserini model at %s", model_path)
        shutil.rmtree(model_path)

    logger.info("Building pyserini BM25 retriever...")

    # Prepare corpus for pyserini (JSONL format)
    doc_ids = [doc["docid"] for doc in corpus]

    # Create collection directory
    model_path.mkdir(parents=True, exist_ok=True)
    collection_path = model_path / "collection"
    collection_path.mkdir(parents=True, exist_ok=True)

    jsonl_file = collection_path / "docs.jsonl"

    # Write documents in JSONL format
    logger.info("Writing %d documents to JSONL...", len(corpus))
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for doc in tqdm(corpus, desc="Preparing documents"):
            json_doc = {
                "id": str(doc["docid"]),
                "contents": doc["texto"],
            }
            f.write(json.dumps(json_doc, ensure_ascii=False) + '\n')

    # Build index using pyserini CLI
    logger.info("Building Lucene index with Portuguese stemmer...")
    index_path.mkdir(parents=True, exist_ok=True)

    # Use python -m to invoke pyserini indexer
    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(collection_path),
        "--index", str(index_path),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--storePositions", "--storeDocvectors", "--storeRaw",
        "--language", "pt",
        "--stemmer", "porter"
    ]

    logger.info(f"Running indexer...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Indexing failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        raise RuntimeError(f"Pyserini indexing failed: {result.stderr}")

    logger.info("Index built successfully")
    logger.info(f"Indexer output:\n{result.stdout}")

    # Create searcher
    searcher = LuceneSearcher(str(index_path))
    logger.info("Searcher created with %d documents", searcher.num_docs)

    # Save corpus metadata
    corpus_data = {
        'corpus': corpus,
        'doc_ids': doc_ids,
    }

    with open(corpus_file, 'wb') as f:
        pickle.dump(corpus_data, f)

    logger.info("Pyserini retriever saved to %s", model_path)

    return {
        'searcher': searcher,
        'corpus': corpus,
        'doc_ids': doc_ids,
    }


def query_pyserini(
    retriever: dict,
    query: str,
    k: int = 10,
    use_rm3: bool = False,
    rm3_fb_docs: int = 10,
    rm3_fb_terms: int = 10,
    rm3_original_query_weight: float = 0.5
) -> list[dict[str, Any]]:
    """
    Query using pyserini retriever with optional RM3.

    Args:
        retriever: Pyserini retriever dict from build_pyserini()
        query: Search query string
        k: Number of results to return
        use_rm3: Enable RM3 pseudo-relevance feedback
        rm3_fb_docs: Number of feedback documents for RM3
        rm3_fb_terms: Number of expansion terms for RM3
        rm3_original_query_weight: Weight of original query (0.0-1.0)

    Returns:
        List of matched documents with scores
    """
    searcher = retriever['searcher']
    corpus = retriever['corpus']

    # Configure RM3 if requested
    if use_rm3:
        logger.info(
            f"Using RM3 with pyserini "
            f"(fb_docs={rm3_fb_docs}, fb_terms={rm3_fb_terms}, weight={rm3_original_query_weight})"
        )
        searcher.set_rm3(
            fb_terms=rm3_fb_terms,
            fb_docs=rm3_fb_docs,
            original_query_weight=rm3_original_query_weight
        )
    else:
        searcher.unset_rm3()

    # Search
    hits = searcher.search(query, k=k)

    # Create doc_id to corpus mapping
    doc_id_map = {str(doc["docid"]): doc for doc in corpus}

    # Format results
    matches: list[dict] = []
    for rank, hit in enumerate(hits, start=1):
        doc_id = hit.docid
        score = hit.score

        # Get document from corpus
        doc = doc_id_map.get(doc_id, {})

        if not doc:
            logger.warning(f"Document {doc_id} not found in corpus")

        matches.append({
            "rank": rank,
            "docid": int(doc_id) if doc else None,
            "score": float(score),
            "texto": doc.get("texto", ""),
            "tema": doc.get("tema"),
            "subtema": doc.get("subtema"),
            "enunciado": doc.get("enunciado"),
            "excerto": doc.get("excerto"),
        })

    return matches
