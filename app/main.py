import argparse
import os
import psycopg2
from dotenv import load_dotenv
from app.chat import chat_loop, initialize_embeddings, run_test_queries, answer_query
from app.storage.database import init_db
from app.storage.chroma import init_chroma
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def main():
    """Main entry point for the chat application."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="JSON RAG System")
    parser.add_argument("--test", action="store_true", help="Run test queries")
    parser.add_argument("--vector-store", choices=["postgres", "chroma"], 
                       default=os.getenv("VECTOR_STORE", "postgres"),
                       help="Vector store to use (postgres or chroma)")
    args = parser.parse_args()
    
    try:
        # Initialize vector store based on selection
        if args.vector_store == "postgres":
            logger.info("Using PostgreSQL as vector store")
            
            # Get connection parameters from environment variables
            db_name = os.getenv("POSTGRES_DB", "postgres")
            db_user = os.getenv("POSTGRES_USER", "postgres")
            db_password = os.getenv("POSTGRES_PASSWORD", "postgres")
            db_host = os.getenv("POSTGRES_HOST", "localhost")
            db_port = os.getenv("POSTGRES_PORT", "5432")
            
            # Establish database connection
            conn_string = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            logger.info(f"Connecting to PostgreSQL at {db_host}:{db_port}/{db_name}")
            
            conn = psycopg2.connect(conn_string)
            
            # Initialize database schema
            init_db(conn)
            
            store = conn
        else:
            logger.info("Using ChromaDB as vector store")
            store = init_chroma()

        if args.test:
            # Run test queries mode
            run_test_queries(store)
            return

        # Normal interactive mode
        logger.info("Checking for new data to embed...")

        # Process any new JSON files
        initialize_embeddings(store)

        # Interactive query loop
        while True:
            try:
                query = input("\nYou: ")
                if query.lower() in ["exit", "quit", "q"]:
                    break

                response = answer_query(store, query)
                print(f"\nAssistant: {response}")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print("\nAssistant: I encountered an error. Please try again.")

    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        if "conn" in locals() and args.vector_store == "postgres":
            conn.close()

if __name__ == "__main__":
    main()
