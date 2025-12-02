CREATE_DOCS_TABLE = """
    create or replace table docs as
    from read_csv('./ulysses-data/bills_dataset.csv', max_line_size=10000000)
    select
        original_docid:name,
        docid:code,
        texto:text,
        summary:text
"""

CREATE_QUERIES_TABLE = """
    create or replace table queries as
    from './ulysses-data/relevance_feedback_dataset.csv'
    select
        qid:id,
        text:query,
        groupid:1;
"""

CREATE_QUERIES_REL_TABLE = """
    create or replace table queries_rel as
    WITH raw_data AS (
        SELECT 
            id AS query_id,
            REPLACE(REPLACE(user_feedback, '''', '"'), ' None', ' null') AS user_feedback_json,
            REPLACE(REPLACE(extra_results, '''', '"'), ' None', ' null') AS extra_results_json
        FROM './ulysses-data/relevance_feedback_dataset.csv'
    ),
    parsed_feedback AS (
        SELECT 
            query_id,
            UNNEST(CAST(user_feedback_json AS JSON[])) AS item
        FROM raw_data
        WHERE user_feedback_json IS NOT NULL 
    ),
    extracted_feedback AS (
        SELECT
            query_id,
            item->>'id' AS doc_id,
            item->>'class' AS relevance_class,
            (item->>'score')::DOUBLE AS score,
            (item->>'score_normalized')::DOUBLE AS score_normalized
        FROM parsed_feedback
    ),
    parsed_extra AS (
        SELECT 
            query_id,
            UNNEST(CAST(extra_results_json AS VARCHAR[])) AS doc_id,
            'r' AS relevance_class,
            NULL::DOUBLE AS score,
            NULL::DOUBLE AS score_normalized
            -- Removido 'ranking' daqui para alinhar as colunas com extracted_feedback
        FROM raw_data
        WHERE extra_results_json IS NOT NULL AND extra_results_json != '[]'
    ),
    combined_raw AS (
        SELECT * FROM extracted_feedback
        UNION ALL
        SELECT * FROM parsed_extra
    ),
    combined_with_relevance AS (
        SELECT 
            *,
            CASE 
                WHEN relevance_class = 'i' THEN 0 
                WHEN relevance_class = 'pr' THEN 1
                WHEN relevance_class = 'r' THEN 2
                ELSE 0 
            END AS relevance
        FROM combined_raw
    )
    SELECT
        qid:cr.query_id,
        docid:d.docid,
        score:cr.relevance,
        -- relevance_class,
        rank:DENSE_RANK() OVER (PARTITION BY cr.query_id ORDER BY cr.relevance DESC),
        -- score,
        -- score_normalized
    FROM combined_with_relevance as cr
    join docs as d on d.original_docid = cr.doc_id
"""

# Consultas de Pré-processamento aplicadas à consulta também

REPLACEMENTS = r"""
    ;update docs
    set texto = replace(texto, 'arts.', 'artigos'),
        summary = replace(summary, 'arts.', 'artigos');

    ;update docs
    set texto = replace(texto, 'art.', 'artigo'),
        summary = replace(summary, 'art.', 'artigo');
    
    ;update docs
    set texto = replace(texto, 'nº', 'numero'),
        summary = replace(summary, 'nº', 'numero');

    ;update docs
    set texto = regexp_replace(texto, '([0-9]+)[ºª]', '\1'),
        summary = regexp_replace(summary, '([0-9]+)[ºª]', '\1');

    ;update docs
    set texto = regexp_replace(texto, '§\s*([0-9]+)', 'paragrafo \1'),
        summary = regexp_replace(summary, '§\s*([0-9]+)', 'paragrafo \1');
"""

REMOVALS = r"""
    ;update docs
    set texto = replace(texto, '[...]', ' '),
        summary = replace(summary, '[...]', ' ');

    ;update docs
    set texto = regexp_replace(texto, 'artigos\s*([0-9]+)', 'artigos \1'),
        summary = regexp_replace(summary, 'artigos\s*([0-9]+)', 'artigos \1');

    ;update docs
    set texto = replace(replace(replace(texto, E'\r', ' '), E'\n', ' '), E'\t', ' '),
        summary = replace(replace(replace(summary, E'\r', ' '), E'\n', ' '), E'\t', ' ');
"""
