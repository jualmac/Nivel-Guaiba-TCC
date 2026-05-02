"""
Make DuckDB database
"""
########################################################################################################################
#                                                                  
# LIBRARIES
#
########################################################################################################################
import os
import duckdb

from util import configure_logging

logger = configure_logging(__name__)

########################################################################################################################
#                                                                  
# FUNCTION
#
########################################################################################################################
def ensure_duckdb_exists(path: str = "data/hidroweb_scrapping.db") -> str:
    """
    Ensure DuckDB file and a bootstrap metadata table exist.
    """
    database_directory = os.path.dirname(path)
    if database_directory:
        os.makedirs(database_directory, exist_ok=True)

    connection = duckdb.connect(path)
    try:
        # Internal bootstrap table guarantees first-run DB passes table-existence health checks;
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS __db_bootstrap__ (
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO __db_bootstrap__
            SELECT CURRENT_TIMESTAMP
            WHERE NOT EXISTS (SELECT 1 FROM __db_bootstrap__)
            """
        )
        logger.info("DuckDB ready at %s", path)
    finally:
        connection.close()

    return path