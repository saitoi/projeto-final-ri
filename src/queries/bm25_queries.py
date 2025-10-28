TOTAL_COUNT = """
    SELECT COUNT(*)
    FROM docs;
"""

GET_DOC_TEXTS = """
    SELECT
        docid,
        texto,
        tema,
        subtema,
        enunciado,
        excerto
    FROM docs
    WHERE texto IS NOT NULL
      AND length(trim(texto)) > 0
    ORDER BY docid;
"""
