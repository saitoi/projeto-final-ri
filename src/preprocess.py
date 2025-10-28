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
from bs4 import BeautifulSoup

from settings import get_settings, Settings, get_logger
import queries

logger = get_logger(__name__)

def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    return text

def run(db_filepath: str):
    conn: DuckDBPyConnection = duckdb.connect(db_filepath)

    try:
        logger.info("Registering extract_text UDF...")
        conn.create_function("extract_text", extract_text, [VARCHAR], VARCHAR)

        logger.info("Creating tables...")
        conn.execute(queries.CREATE_DOCS_TABLE)
        conn.execute(queries.CREATE_QUERIES_TABLE)
        conn.execute(queries.CREATE_QUERIES_REL_TABLE)

        logger.info("Parsing HTML fields...")
        conn.execute(queries.EXTRACT_TEXT)

        logger.info("Applying replacements...")
        conn.execute(queries.REPLACEMENTS)

        logger.info("Applying removals...")
        conn.execute(queries.REMOVALS)
    finally:
        conn.close()


if __name__ == "__main__":
    settings: Settings = get_settings()
    run(settings.database)
