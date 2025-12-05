# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "duckdb",
#     "bs4",
#     "pydantic-settings",
#     "numpy",
# ]
# ///

import duckdb
from duckdb import DuckDBPyConnection
from duckdb.sqltypes import VARCHAR

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import get_settings, Settings, get_logger
import queries
from preprocess._utils import (
    remove_accents,
    remove_non_ascii,
    clean_quotes,
    clean_commas,
    clean_dashes,
    clean_symbols,
)

logger = get_logger(__name__)

def run(db_filepath: str):
    conn: DuckDBPyConnection = duckdb.connect(db_filepath)

    try:
        logger.info("Registering UDFs...")
        conn.create_function("remove_accents", remove_accents, [VARCHAR], VARCHAR)
        conn.create_function("remove_non_ascii", remove_non_ascii, [VARCHAR], VARCHAR)
        conn.create_function("clean_quotes", clean_quotes, [VARCHAR], VARCHAR)
        conn.create_function("clean_commas", clean_commas, [VARCHAR], VARCHAR)
        conn.create_function("clean_dashes", clean_dashes, [VARCHAR], VARCHAR)
        conn.create_function("clean_symbols", clean_symbols, [VARCHAR], VARCHAR)

        # DDLs

        logger.info("Creating tables...")
        conn.execute(queries.CREATE_DOCS_TABLE)
        conn.execute(queries.CREATE_QUERIES_TABLE)
        conn.execute(queries.CREATE_QUERIES_REL_TABLE)

        # DMLs

        logger.info("Parsing HTML fields...")
        conn.execute(queries.EXTRACT_TEXT)

        logger.info("Applying replacements...")
        conn.execute(queries.REPLACEMENTS)

        logger.info("Applying removals...")
        conn.execute(queries.REMOVALS)

        # Text Normalizations: Não funcionou como esperado para BM25

        # logger.info("Applying normalizations...")
        # conn.execute(queries.NORMALIZATIONS)
    finally:
        conn.close()


if __name__ == "__main__":
    settings: Settings = get_settings()
    run(settings.database)
