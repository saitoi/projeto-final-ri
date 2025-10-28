CREATE_CHUNKS_TABLE = """
    CREATE TABLE IF NOT EXISTS doc_chunks (
        docid BIGINT,
        chunk_index INTEGER NOT NULL,
        tokens INTEGER NOT NULL,
        text TEXT NOT NULL,
        PRIMARY KEY (docid, chunk_index)
    );
"""

CREATE_CHUNKS_EMBEDDINGS_TABLE = """
    CREATE TABLE IF NOT EXISTS doc_chunk_embeddings (
        docid BIGINT,
        chunk_index INTEGER NOT NULL,
        embedding FLOAT[],
        PRIMARY KEY (docid, chunk_index)
    );
"""

CREATE_EMBEDDINGS_TABLE = """
    CREATE TABLE IF NOT EXISTS doc_embeddings (
        docid BIGINT,
        embedding FLOAT[],
        PRIMARY KEY (docid)
    );
"""

GET_DOCS_FOR_CHUNKING = """
    SELECT docid, texto
    FROM docs
    WHERE texto IS NOT NULL
      AND length(trim(texto)) > 0
    ORDER BY docid;
"""

INSERT_CHUNK = """
    INSERT OR REPLACE INTO doc_chunks (docid, chunk_index, tokens, text)
    VALUES (?, ?, ?, ?);
"""

GET_PENDING_CHUNKS = """
    SELECT docid, chunk_index, text
    FROM doc_chunks
    WHERE (docid, chunk_index) NOT IN (
        SELECT docid, chunk_index FROM doc_chunk_embeddings
    )
    ORDER BY docid, chunk_index;
"""

INSERT_CHUNK_EMBEDDING = """
    INSERT OR REPLACE INTO doc_chunk_embeddings (docid, chunk_index, embedding)
    VALUES (?, ?, ?);
"""

SELECT_CHUNK_EMBEDDINGS = """
    SELECT docid, chunk_index, embedding
    FROM doc_chunk_embeddings
    ORDER BY docid, chunk_index;
"""

DELETE_DOC_EMBEDDING = """
    DELETE FROM doc_embeddings WHERE docid = ?;
"""

INSERT_DOC_EMBEDDING = """
    INSERT OR REPLACE INTO doc_embeddings (docid, embedding)
    VALUES (?, ?);
"""

GET_DOC_EMBEDDINGS = """
    SELECT
        e.docid,
        e.embedding,
        d.texto,
        d.tema,
        d.subtema,
        d.enunciado,
        d.excerto
    FROM doc_embeddings e
    JOIN docs d ON d.docid = e.docid
    WHERE e.embedding IS NOT NULL;
"""

AGGREGATE_MEAN_POOLING = """
    INSERT OR REPLACE INTO doc_embeddings (docid, embedding)
    SELECT
        docid,
        avg(embedding) AS embedding
    FROM doc_chunk_embeddings
    GROUP BY docid;
"""

SEARCH_EMBEDDING_TEXTO = """
    from doc_embeddings de
    inner join docs d on d.docid = de.docid
    select
        d.docid,
        d.texto,
        sim:array_cosine_similarity(de.embedding, ?) as similarity
    order by sim desc
"""