import pathlib
import pickle
import shutil
import subprocess
import sys
import json
from typing import Any

from tqdm.auto import tqdm
from pyserini.search.lucene import LuceneSearcher

# Se você não tiver o get_logger configurado, substitua por print ou logging padrão
try:
    from settings import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


def build_pyserini(corpus: list[dict], model_dir: str, *, build: bool = False) -> dict:
    model_path = pathlib.Path(model_dir + "-pyserini")
    index_path = model_path / "index"
    corpus_file = model_path / "corpus.pkl"

    # --- 1. Tentar carregar índice existente ---
    if not build and index_path.exists() and corpus_file.exists():
        logger.info("Loading pyserini retriever from %s", model_path)
        with open(corpus_file, 'rb') as f:
            corpus_data = pickle.load(f)

        searcher = LuceneSearcher(str(index_path))
        searcher.set_language('pt')
        logger.info("Loaded pyserini index with %d documents", searcher.num_docs)

        return {
            'searcher': searcher,
            'corpus': corpus_data['corpus'],
            'doc_ids': corpus_data['doc_ids'],
        }

    # --- 2. Limpar índice antigo se existir ---
    if model_path.exists():
        logger.info("Removing existing pyserini model at %s", model_path)
        shutil.rmtree(model_path)

    logger.info("Building pyserini BM25 retriever (Text Only)...")

    doc_ids = [doc["docid"] for doc in corpus]

    # Criar diretórios
    model_path.mkdir(parents=True, exist_ok=True)
    collection_path = model_path / "collection"
    collection_path.mkdir(parents=True, exist_ok=True)

    jsonl_file = collection_path / "docs.jsonl"

    # --- 3. Escrever JSONL (Apenas Texto) ---
    logger.info("Writing %d documents to JSONL...", len(corpus))
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for doc in tqdm(corpus, desc="Preparing documents"):
            # Limpeza básica para evitar erros de nulos, se houver
            conteudo = doc.get("texto")
            if conteudo is None:
                conteudo = ""
            
            json_doc = {
                "id": str(doc["docid"]),
                "contents": conteudo, # Apenas o texto, conforme solicitado
            }
            f.write(json.dumps(json_doc, ensure_ascii=False) + '\n')

    # --- 4. Construir Índice (Configuração PT-BR) ---
    logger.info("Building Lucene index with Portuguese settings...")
    index_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", str(collection_path),
        "--index", str(index_path),
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "4",
        "--storePositions", "--storeDocvectors", "--storeRaw",
        "--language", "pt"
    ]

    logger.info("Running indexer...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Indexing failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")
        raise RuntimeError(f"Pyserini indexing failed: {result.stderr}")

    logger.info("Index built successfully")

    # --- 5. Inicializar Searcher ---
    searcher = LuceneSearcher(str(index_path))
    searcher.set_language('pt') # Importante definir aqui também
    logger.info("Searcher created with %d documents", searcher.num_docs)

    # Salvar metadados
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
    Realiza a busca no índice Pyserini.
    """
    searcher = retriever['searcher']
    corpus = retriever['corpus']

    # Configurar RM3 (Query Expansion)
    if use_rm3:
        # RM3 é útil se o vocabulário da query for muito diferente do texto jurídico
        searcher.set_rm3(
            fb_terms=rm3_fb_terms,
            fb_docs=rm3_fb_docs,
            original_query_weight=rm3_original_query_weight
        )
    else:
        searcher.unset_rm3()

    # Busca
    try:
        hits = searcher.search(query, k=k)
    except Exception as e:
        logger.error(f"Error searching for query '{query}': {e}")
        return []

    # Mapa para recuperação rápida do documento original
    # Nota: Se o corpus for gigante, isso pode consumir muita RAM. 
    # Idealmente, criar esse mapa fora da função de query ou usar o índice reverso.
    doc_id_map = {str(doc["docid"]): doc for doc in corpus}

    matches: list[dict] = []
    for rank, hit in enumerate(hits, start=1):
        doc_id = hit.docid
        score = hit.score
        
        # Recuperar documento original
        doc = doc_id_map.get(doc_id)
        
        # Se o doc não estiver no mapa (erro de sincronia), criamos um placeholder
        if not doc:
            logger.warning(f"Document {doc_id} found in index but missing in corpus map")
            continue

        matches.append({
            "rank": rank,
            "docid": doc["docid"],
            "score": float(score),
            "texto": doc.get("texto", ""),
            "tema": doc.get("tema"),
            "subtema": doc.get("subtema"),
            "enunciado": doc.get("enunciado"),
            "excerto": doc.get("excerto"),
        })

    return matches