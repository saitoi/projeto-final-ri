CREATE_DOCS_TABLE = """
    create or replace table docs as
    from './tcu-data/doc.csv'
    select
        docid:try_cast(string_split(KEY, '-')[-1] as bigint),
        enunciado:ENUNCIADO,
        excerto:EXCERTO,
        tema:TEMA,
        subtema:SUBTEMA,
        indexacao:INDEXACAO,
        texto:trim(
            concat_ws(
                ' ',
                coalesce(ENUNCIADO, ''),
                coalesce(EXCERTO, ''),
                coalesce(TEMA, ''),
                coalesce(SUBTEMA, ''),
                coalesce(INDEXACAO, '')
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
        source:SOURCE;
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
    set enunciado = extract_text(enunciado),
        excerto = extract_text(excerto),
        texto = extract_text(texto);
"""

REFRESH_TEXT = """
    ;update docs
    set texto = trim(
        concat_ws(
            ' ',
            coalesce(enunciado, ''),
            coalesce(excerto, ''),
            coalesce(tema, ''),
            coalesce(subtema, ''),
            coalesce(indexacao, '')
        )
    );
"""

# Consultas de Pré-processamento aplicadas à consulta também

REPLACEMENTS = r"""
    ;update docs
    set enunciado = replace(enunciado, 'arts.', 'artigos'),
        excerto = replace(excerto, 'arts.', 'artigos'),
        texto = replace(texto, 'arts.', 'artigos');

    ;update docs
    set enunciado = replace(enunciado, 'art.', 'artigo'),
        excerto = replace(excerto, 'art.', 'artigo'),
        texto = replace(texto, 'art.', 'artigo');

    ;update docs
    set enunciado = replace(enunciado, 'nº', 'numero'),
        excerto = replace(excerto, 'nº', 'numero'),
        texto = replace(texto, 'nº', 'numero');

    ;update docs
    set enunciado = regexp_replace(enunciado, '([0-9]+)[ºª]', '\1'),
        excerto = regexp_replace(excerto, '([0-9]+)[ºª]', '\1'),
        texto = regexp_replace(texto, '([0-9]+)[ºª]', '\1');

    ;update docs
    set enunciado = regexp_replace(enunciado, '§\s*([0-9]+)', 'paragrafo \1'),
        excerto = regexp_replace(excerto, '§\s*([0-9]+)', 'paragrafo \1'),
        texto = regexp_replace(texto, '§\s*([0-9]+)', 'paragrafo \1');
"""

REMOVALS = r"""
    ;update docs
    set enunciado = replace(enunciado, '[...]', ' '),
        excerto = replace(excerto, '[...]', ' '),
        texto = replace(texto, '[...]', ' ');

    ;update docs
    set enunciado = regexp_replace(enunciado, 'artigos\s*([0-9]+)', 'artigos \1'),
        excerto = regexp_replace(excerto, 'artigos\s*([0-9]+)', 'artigos \1'),
        texto = regexp_replace(texto, 'artigos\s*([0-9]+)', 'artigos \1');
"""
