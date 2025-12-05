CREATE_DOCS_TABLE = """
    create or replace table docs as
    from (
    from (
        from './votos-data/stj_sts.csv'
        union
        from './votos-data/tcu-sts.csv'
    )
    select sentence_A texto
    union
    from (
        from './votos-data/stj_sts.csv'
        union
        from './votos-data/tcu-sts.csv'
    )
    select sentence_B texto
    )
    select
        row_number() over(order by length(doc) asc) id,
        texto
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
