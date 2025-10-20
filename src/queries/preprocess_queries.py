CREATE_DOCS_TABLE = """
    create table if not exists docs as
    from './tcu-data/doc.csv'
    select KEY as key, 
"""
