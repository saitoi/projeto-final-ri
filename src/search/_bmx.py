import pathlib
import pickle
import shutil
from typing import Any

from baguetter.indices import BMXSparseIndex
from baguetter.indices.sparse.text_preprocessor.text_processor import TextPreprocessorConfig
from nltk.stem import RSLPStemmer
from tqdm.auto import tqdm

from settings import get_logger

logger = get_logger(__name__)


def build_bmx(corpus: list[dict], model_dir: str, *, build: bool = False,
              k1: float = 1.5, b: float = 0.75, alpha: float | None = None,
              beta: float | None = None) -> dict:
    """
    Build or load BMX retriever using baguetter library.

    Args:
        corpus: List of documents with fields (docid, texto, tema, subtema, enunciado, excerto)
        model_dir: Base directory for models
        build: If True, force rebuild even if index exists
        k1: BM25 k1 parameter (term frequency saturation)
        b: BM25 b parameter (length normalization)
        alpha: BMX alpha parameter (entropy normalization weight, auto-calculated if None)
        beta: BMX beta parameter (semantic similarity weight, auto-calculated if None)

    Returns:
        Dict with bmx_index, corpus, and doc_ids
    """
    model_path = pathlib.Path(model_dir + "-bmx")
    index_path = model_path / "index"
    corpus_file = model_path / "corpus.pkl"

    # Load existing index
    if not build and index_path.exists() and corpus_file.exists():
        logger.info("Loading BMX retriever from %s", model_path)

        # Load BMX index
        bmx = BMXSparseIndex.load(str(index_path))

        # Load corpus metadata
        with open(corpus_file, 'rb') as f:
            corpus_data = pickle.load(f)

        logger.info("Loaded BMX index with %d documents", len(corpus_data['corpus']))

        return {
            'bmx_index': bmx,
            'corpus': corpus_data['corpus'],
            'doc_ids': corpus_data['doc_ids'],
        }

    # Clean existing index
    if model_path.exists():
        logger.info("Removing existing BMX model at %s", model_path)
        shutil.rmtree(model_path)

    logger.info("Building BMX retriever (k1=%s, b=%s)...", k1, b)

    # Prepare corpus
    doc_ids = [doc["docid"] for doc in corpus]
    corpus_text = [doc["texto"] for doc in corpus]

    # Create RSLP stemmer for Portuguese
    rslp_stemmer = RSLPStemmer()

    # Configure for Portuguese language with RSLP stemmer
    preprocessor_config = TextPreprocessorConfig(
        custom_stemmer=rslp_stemmer.stem,
        stopwords="portuguese"
    )

    # Initialize BMX index with Portuguese configuration
    bmx = BMXSparseIndex(
        index_name="bmx-ri",
        preprocessor_or_config=preprocessor_config,
        k1=k1,
        b=b,
        alpha=alpha,
        beta=beta,
    )

    # Index documents
    logger.info("Indexing %d documents...", len(corpus))
    bmx.add_many(
        keys=doc_ids,
        values=corpus_text,
        show_progress=True,
    )

    # Save index and corpus
    model_path.mkdir(parents=True, exist_ok=True)
    index_path.mkdir(parents=True, exist_ok=True)

    bmx.save(str(index_path))
    logger.info("BMX index saved to %s", index_path)

    # Save corpus metadata
    corpus_data = {
        'corpus': corpus,
        'doc_ids': doc_ids,
    }

    with open(corpus_file, 'wb') as f:
        pickle.dump(corpus_data, f)

    logger.info("BMX retriever saved to %s", model_path)

    return {
        'bmx_index': bmx,
        'corpus': corpus,
        'doc_ids': doc_ids,
    }


def query_bmx(
    retriever: dict,
    query: str,
    k: int = 10,
) -> list[dict[str, Any]]:
    """
    Query using BMX retriever.

    Args:
        retriever: BMX retriever dict from build_bmx()
        query: Search query string
        k: Number of results to return

    Returns:
        List of matched documents with scores
    """
    bmx_index = retriever['bmx_index']
    corpus = retriever['corpus']

    # Create doc_id to corpus mapping
    doc_id_map = {doc["docid"]: doc for doc in corpus}

    # Search
    results = bmx_index.search(query, top_k=k)

    # Format results
    matches: list[dict] = []
    for rank, (doc_id, score) in enumerate(zip(results.keys, results.scores), start=1):
        # Get document from corpus
        doc = doc_id_map.get(doc_id, {})

        if not doc:
            logger.warning(f"Document {doc_id} not found in corpus")

        matches.append({
            "rank": rank,
            "docid": doc_id,
            "score": float(score),
            "texto": doc.get("texto", ""),
            "tema": doc.get("tema"),
            "subtema": doc.get("subtema"),
            "enunciado": doc.get("enunciado"),
            "excerto": doc.get("excerto"),
        })

    return matches
