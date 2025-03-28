import argparse
import psycopg2
import logging
from app.storage.database import init_db, reset_database
from app.chat import chat_loop, initialize_embeddings, run_test_queries
from app.core.config import POSTGRES_CONN_STR
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def main():
    """
    Main entry point for the JSON RAG system.

    Command-line Arguments:
        --new: Reset the database and start fresh
        --test: Run test queries from test_queries file

    Flow:
        1. Connects to PostgreSQL database
        2. Optionally resets and initializes database
        3. Updates embeddings for any new/changed files
        4. Either runs test queries or starts interactive chat loop
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--new", action="store_true", help="Reset the database and start fresh"
    )
    parser.add_argument(
        "--test", action="store_true", help="Run test queries from test_queries file"
    )
    args = parser.parse_args()

    # Set logging level to ERROR only for test runs
    if args.test:
        logging.getLogger().setLevel(logging.ERROR)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.ERROR)

    try:
        conn = psycopg2.connect(POSTGRES_CONN_STR)
        logger.info("Connected to PostgreSQL database successfully")
    except Exception as e:
        logger.error(f"Failed to connect to PostgreSQL database: {str(e)}")
        logger.info("Using demo mode without database persistence")
        from unittest.mock import MagicMock
        conn = MagicMock()
        # Set up mock cursor
        mock_cursor = MagicMock()
        conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = []
        mock_cursor.fetchone.return_value = None

    if args.new:
        logger.info("Initializing new database...")
        reset_database(conn)
        init_db(conn)
        initialize_embeddings(conn)
    else:
        # Check for changes in JSON files and update embeddings if needed
        initialize_embeddings(conn)

    if args.test:
        run_test_queries(conn)
    else:
        chat_loop(conn)


if __name__ == "__main__":
    main()
