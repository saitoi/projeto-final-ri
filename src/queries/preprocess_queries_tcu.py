CREATE_DOCS_TABLE = """
    create or replace table docs as
    from './tcu-data/doc.csv'
    select
        docid:try_cast(string_split(KEY, '-')[-1] as bigint),
        summary:ENUNCIADO, -- embeddings
        excerto:EXCERTO,
        tema:TEMA,
        subtema:SUBTEMA,
        indexacao:INDEXACAO,
        texto:trim( -- bm25
            concat_ws(
                ' ',
                coalesce(ENUNCIADO, ''),
                coalesce(EXCERTO, '')
                -- coalesce(TEMA, ''),
                -- coalesce(SUBTEMA, ''),
                -- coalesce(INDEXACAO, '')
            )
        )
    where key is not null;
"""

CREATE_QUERIES_TABLE = """
    create or replace table queries as
    from './tcu-data/query.csv'
    select
        qid:ID,
        text:TEXT,
        source:SOURCE,
        groupid:case SOURCE when 'LLM' then 3 when 'search log' then 1 else 2 end;
"""

CREATE_QUERIES_REL_TABLE = """
    create or replace table queries_rel as
    from './tcu-data/qrel.csv'
    select
        qid:QUERY_ID,
        docid:DOC_ID,
        score:SCORE,
        rank:RANK;
"""

EXTRACT_TEXT = """
    ;update docs
    set summary = extract_text(summary),
        excerto = extract_text(excerto),
        texto = extract_text(texto);
"""

# Consultas de Pré-processamento aplicadas à consulta também

REPLACEMENTS = r"""
    ;update docs
    set summary = replace(summary, 'arts.', 'artigos'),
        excerto = replace(excerto, 'arts.', 'artigos'),
        texto = replace(texto, 'arts.', 'artigos');

    ;update docs
    set summary = replace(summary, 'art.', 'artigo'),
        excerto = replace(excerto, 'art.', 'artigo'),
        texto = replace(texto, 'art.', 'artigo');

    ;update docs
    set summary = replace(summary, 'nº', 'numero'),
        excerto = replace(excerto, 'nº', 'numero'),
        texto = replace(texto, 'nº', 'numero');

    ;update docs
    set summary = regexp_replace(summary, '([0-9]+)[ºª]', '\1'),
        excerto = regexp_replace(excerto, '([0-9]+)[ºª]', '\1'),
        texto = regexp_replace(texto, '([0-9]+)[ºª]', '\1');

    ;update docs
    set summary = regexp_replace(summary, '§\s*([0-9]+)', 'paragrafo \1'),
        excerto = regexp_replace(excerto, '§\s*([0-9]+)', 'paragrafo \1'),
        texto = regexp_replace(texto, '§\s*([0-9]+)', 'paragrafo \1');
"""

REMOVALS = r"""
    ;update docs
    set summary = replace(summary, '[...]', ' '),
        excerto = replace(excerto, '[...]', ' '),
        texto = replace(texto, '[...]', ' ');

    ;update docs
    set summary = regexp_replace(summary, 'artigos\s*([0-9]+)', 'artigos \1'),
        excerto = regexp_replace(excerto, 'artigos\s*([0-9]+)', 'artigos \1'),
        texto = regexp_replace(texto, 'artigos\s*([0-9]+)', 'artigos \1');
"""

# Não mexo mais em summary

NORMALIZATIONS = r"""
    ;update docs
    set excerto = remove_accents(excerto),
    texto = remove_accents(texto);
    
    ;update docs
    set excerto = remove_non_ascii(excerto),
    texto = remove_non_ascii(texto);
    
    ;update docs
    set excerto = clean_quotes(excerto),
    texto = clean_quotes(texto);
    
    ;update docs
    set excerto = clean_commas(excerto),
        texto = clean_commas(texto);

    ;update docs
    set excerto = clean_dashes(excerto),
        texto = clean_dashes(texto);

    ;update docs
    set excerto = clean_symbols(excerto),
        texto = clean_symbols(texto);
"""
