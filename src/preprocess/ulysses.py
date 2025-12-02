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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from settings import get_settings, Settings, get_logger
from queries.preprocess_queries_ulysses import (
    CREATE_DOCS_TABLE,
    CREATE_QUERIES_TABLE,
    CREATE_QUERIES_REL_TABLE,
    REPLACEMENTS,
    REMOVALS,
)

logger = get_logger(__name__)

def run(db_filepath: str):
    conn: DuckDBPyConnection = duckdb.connect(db_filepath)

    try:
        # DDLs

        logger.info("Creating tables...")
        conn.execute(CREATE_DOCS_TABLE)
        conn.execute(CREATE_QUERIES_TABLE)
        conn.execute(CREATE_QUERIES_REL_TABLE)

        # DMLs

        logger.info("Applying replacements...")
        conn.execute(REPLACEMENTS)

        logger.info("Applying removals...")
        conn.execute(REMOVALS)
    finally:
        conn.close()


if __name__ == "__main__":
    settings: Settings = get_settings()
    run(settings.database)
