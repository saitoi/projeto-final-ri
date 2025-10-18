CREATE_CHUNKS_TABLE = """
    CREATE TABLE IF NOT EXISTS chunks_teses (
        id_auto VARCHAR,
        id_chunk INT NOT NULL,
        tokens SMALLINT NOT NULL,
        texto TEXT NOT NULL,
        PRIMARY KEY (id_auto, id_chunk),
        FOREIGN KEY (id_auto) REFERENCES autos_judiciais(id_auto)
    );
"""

CREATE_CHUNKS_EMBEDDINGS_TABLE = """
    CREATE TABLE IF NOT EXISTS chunks_embeddings_teses (
        id_auto VARCHAR,
        id_chunk INT NOT NULL,
        embedding FLOAT(768) NOT NULL,
        PRIMARY KEY (id_auto, id_chunk),
        FOREIGN KEY (id_auto, id_chunk) REFERENCES chunks_teses(id_auto, id_chunk) ON DELETE CASCADE
    )
"""

CREATE_EMBEDDINGS_TABLE = """
    CREATE TABLE IF NOT EXISTS embeddings_teses (
        id_auto VARCHAR,
        embedding FLOAT(768) NOT NULL,
        PRIMARY KEY (id_auto)
    )
"""

# adaptar para contexto do tcu
GET_TEXTS = """
    SELECT ta.id_auto, ta.texto_auto
    FROM texto_autos ta
    WHERE NOT EXISTS (
        SELECT 1 FROM chunks_teses ct WHERE ct.id_auto = ta.id_auto
    )
"""

INSERT_CHUNK = """
    INSERT INTO chunks_teses (id_auto, id_chunk, tokens, texto)
    VALUES (:id_auto, :id_chunk, :tokens, :texto)
"""

GET_CHUNKS = """
    SELECT id_auto, id_chunk, texto
    FROM chunks_teses ct
    WHERE NOT EXISTS (
        SELECT 1 FROM chunks_embeddings_teses ce
        WHERE ct.id_auto = ce.id_auto AND ct.id_chunk = ce.id_chunk
    ) 
"""

INSERT_CHUNK_EMBEDDING = """
    INSERT INTO chunks_embeddings_teses (id_auto, id_chunk, embedding)
    VALUES (:id_auto, :id_chunk, :embedding)
"""

AGGREGATE_MEAN_POOLING = """
    INSERT INTO embeddings_teses(id_auto, embedding)
    SELECT
        cet.id_auto,
        AVG(cet.embedding) as embedding
    FROM
        chunks_embeddings_teses cet
    WHERE NOT EXISTS (
        SELECT 1 FROM embeddings_teses et WHERE et.id_auto = cet.id_auto
    )
    GROUP BY cet.id_auto
"""