TOTAL_COUNT = """
    SELECT COUNT(*)
    FROM docs;
"""

def get_doc_texts_query(conn) -> str:
    """
    Build GET_DOC_TEXTS query dynamically based on available columns.
    This handles different datasets (TCU with tema/subtema/summary/excerto vs Ulysses with just docid/texto).
    """
    # Get available columns
    columns_result = conn.execute("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'docs'
    """).fetchall()

    available_columns = {col[0] for col in columns_result}

    # Base columns (required)
    select_parts = ["docid", "texto"]

    # Optional columns
    optional_cols = ["tema", "subtema", "summary", "excerto"]
    for col in optional_cols:
        if col in available_columns:
            select_parts.append(col)
        else:
            select_parts.append(f"NULL AS {col}")

    return f"""
        SELECT
            {', '.join(select_parts)}
        FROM docs
        WHERE texto IS NOT NULL
          AND length(trim(texto)) > 0
        ORDER BY docid;
    """

# Legacy constant for backwards compatibility
GET_DOC_TEXTS = """
    SELECT
        docid,
        texto,
        tema,
        subtema,
        summary,
        excerto
    FROM docs
    WHERE texto IS NOT NULL
      AND length(trim(texto)) > 0
    ORDER BY docid;
"""
