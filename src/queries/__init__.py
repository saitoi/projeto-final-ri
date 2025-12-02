from .embedding_queries import (
    CREATE_CHUNKS_EMBEDDINGS_TABLE,
    CREATE_CHUNKS_TABLE,
    CREATE_EMBEDDINGS_TABLE,
    GET_DOCS_FOR_CHUNKING,
    INSERT_CHUNK,
    GET_PENDING_CHUNKS,
    INSERT_CHUNK_EMBEDDING,
    AGGREGATE_MEAN_POOLING,
    SEARCH_EMBEDDING_TEXTO,
)

from .bm25_queries import (
    TOTAL_COUNT,
    GET_DOC_TEXTS,
    get_doc_texts_query,
)

# Note: preprocess queries are imported directly in their respective preprocessors
# - tcu.py imports from queries.preprocess_queries_tcu
# - ulysses.py imports from queries.preprocess_queries_ulysses
# This avoids redefinition conflicts in __init__.py

__all__ = [
    "CREATE_CHUNKS_TABLE",
    "CREATE_CHUNKS_EMBEDDINGS_TABLE",
    "CREATE_EMBEDDINGS_TABLE",
    "GET_DOCS_FOR_CHUNKING",
    "INSERT_CHUNK",
    "GET_PENDING_CHUNKS",
    "INSERT_CHUNK_EMBEDDING",
    "AGGREGATE_MEAN_POOLING",
    "TOTAL_COUNT",
    "GET_DOC_TEXTS",
    "get_doc_texts_query",
    "SEARCH_EMBEDDING_TEXTO",
]

