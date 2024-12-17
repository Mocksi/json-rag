import argparse
from app.database import init_db, reset_database
from app.chat import chat_loop, initialize_embeddings
import psycopg2
from app.config import POSTGRES_CONN_STR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--new', action='store_true', help='Reset the database and start fresh')
    args = parser.parse_args()

    conn = psycopg2.connect(POSTGRES_CONN_STR)
    
    if args.new:
        print("Initializing new database...")
        reset_database(conn)
        init_db(conn)
        initialize_embeddings(conn)
    else:
        # Check for changes in JSON files and update embeddings if needed
        initialize_embeddings(conn)
    
    chat_loop(conn)

if __name__ == "__main__":
    main()
