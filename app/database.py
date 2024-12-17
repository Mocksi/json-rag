import os
import psycopg2
from datetime import datetime

def get_file_info(conn):
    """
    Get stored file metadata from database.
    
    Args:
        conn: PostgreSQL database connection
        
    Returns:
        dict: Mapping of filenames to (hash, modified_time) tuples
    """
    cur = conn.cursor()
    cur.execute("SELECT filename, file_hash, last_modified FROM file_metadata;")
    rows = cur.fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}

def upsert_file_metadata(conn, filename, file_hash, mod_time):
    """
    Update or insert file metadata.
    
    Args:
        conn: PostgreSQL database connection
        filename (str): Name of file
        file_hash (str): Hash of file contents
        mod_time (datetime): Last modification time
    """
    cur = conn.cursor()
    query = """
    INSERT INTO file_metadata (filename, file_hash, last_modified)
    VALUES (%s, %s, %s)
    ON CONFLICT (filename) DO UPDATE SET
        file_hash = EXCLUDED.file_hash,
        last_modified = EXCLUDED.last_modified;
    """
    cur.execute(query, (filename, file_hash, mod_time))
    conn.commit()

def get_files_to_process(conn, compute_file_hash, get_json_files):
    """
    Get list of files that need processing.
    
    Args:
        conn: PostgreSQL database connection
        compute_file_hash: Function to compute file hash
        get_json_files: Function to get list of JSON files
        
    Returns:
        list: Tuples of (filename, hash, modified_time) for files needing processing
    """
    existing_info = get_file_info(conn)
    files = get_json_files()
    to_process = []
    
    for f in files:
        f_hash = compute_file_hash(f)
        f_mtime = datetime.utcfromtimestamp(os.path.getmtime(f))
        if f not in existing_info:
            to_process.append((f, f_hash, f_mtime))
        else:
            old_hash, old_mtime = existing_info[f]
            if old_hash != f_hash:
                to_process.append((f, f_hash, f_mtime))
    return to_process

def reset_database(conn):
    """
    Reset the database by truncating all tables.
    
    Args:
        conn: PostgreSQL database connection
    """
    print("Warning: This will delete all stored embeddings and metadata.")
    confirmation = input("Are you sure you want to reset the database? (yes/no): ")
    if confirmation.lower() != 'yes':
        print("Database reset cancelled.")
        return
    
    print("Resetting database...")
    cur = conn.cursor()
    try:
        # Execute truncate statements individually
        cur.execute("TRUNCATE TABLE json_chunks CASCADE;")
        cur.execute("TRUNCATE TABLE file_metadata CASCADE;")
        cur.execute("TRUNCATE TABLE schema_evolution CASCADE;")
        cur.execute("TRUNCATE TABLE chunk_key_values CASCADE;")
        # Commit the changes
        conn.commit()
        print("Database reset complete.")
    except Exception as e:
        conn.rollback()
        print(f"Error resetting database: {str(e)}")
    finally:
        cur.close()

def init_db(conn):
    cur = conn.cursor()
    # Create tables if not exist, indexing, etc.
    # This was partially in the original code; adapt as needed.
    cur.execute("""
        CREATE TABLE IF NOT EXISTS json_chunks (
            id TEXT PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            embedding vector(1536)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS file_metadata (
            filename TEXT PRIMARY KEY,
            file_hash TEXT,
            last_modified TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunk_key_values (
            chunk_id TEXT,
            key TEXT,
            value TEXT
        )
    """)
    # Add any other tables you created (e.g. chunk_hierarchy, chunk_metrics, etc.)
    # from the main code if needed.
    # For brevity, only essential ones shown here.

    conn.commit()