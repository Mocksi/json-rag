from typing import List, Dict, Optional, Tuple
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path

def get_file_info(conn):
    """
    Retrieves stored file metadata from the database.
    
    Args:
        conn: PostgreSQL database connection
        
    Returns:
        dict: Mapping of filenames to tuples of (hash, modified_time)
        
    Note:
        Used to detect changes in JSON files between runs
    """
    cur = conn.cursor()
    cur.execute("SELECT filename, file_hash, last_modified FROM file_metadata;")
    rows = cur.fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}

def upsert_file_metadata(conn, filename, file_hash, mod_time):
    """
    Updates or inserts file metadata in the database.
    
    Args:
        conn: PostgreSQL database connection
        filename (str): Name of file
        file_hash (str): Hash of file contents
        mod_time (datetime): Last modification time
        
    Note:
        Uses UPSERT (INSERT ... ON CONFLICT) to handle both new and existing files
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
    Gets list of files that need processing based on changes.
    
    Args:
        conn: PostgreSQL database connection
        compute_file_hash: Function to compute file hash
        get_json_files: Function to get list of JSON files
        
    Returns:
        list: Tuples of (filename, hash, modified_time) for files needing processing
        
    Process:
        1. Gets existing file metadata from database
        2. Compares with current files on disk
        3. Returns files that are new or have changed
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
    Resets the database by truncating all tables.
    
    Args:
        conn: PostgreSQL database connection
        
    Note:
        Requires user confirmation before proceeding
        Handles errors with rollback
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
    """
    Initializes database schema by creating required tables.
    
    Args:
        conn: PostgreSQL database connection
        
    Tables Created:
        - json_chunks: Stores document chunks and their embeddings
        - file_metadata: Tracks file changes
        - chunk_key_values: Stores searchable key-value pairs
        - chunk_archetypes: Stores detected archetypes for chunks
        - chunk_relationships: Stores relationships between chunks
    """
    cur = conn.cursor()
    
    # Create tables if not exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS json_chunks (
            id TEXT PRIMARY KEY,
            chunk_text TEXT NOT NULL,
            chunk_json JSONB NOT NULL,
            embedding vector(384),
            metadata JSONB DEFAULT '{}'::jsonb
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
    
    # New table for chunk archetypes
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunk_archetypes (
            chunk_id TEXT,
            archetype TEXT,
            confidence FLOAT,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (chunk_id, archetype),
            FOREIGN KEY (chunk_id) REFERENCES json_chunks(id)
        )
    """)
    
    # New table for chunk relationships
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunk_relationships (
            source_chunk TEXT,
            target_chunk TEXT,
            relationship_type TEXT,
            metadata JSONB DEFAULT '{}'::jsonb,
            detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (source_chunk, target_chunk, relationship_type),
            FOREIGN KEY (source_chunk) REFERENCES json_chunks(id),
            FOREIGN KEY (target_chunk) REFERENCES json_chunks(id)
        )
    """)
    
    conn.commit()

def save_chunk_archetypes(conn, chunk_id: str, archetypes: List[Tuple[str, float]]):
    """Save chunk archetypes to database."""
    cur = conn.cursor()
    for archetype, confidence in archetypes:
        cur.execute("""
            INSERT INTO chunk_archetypes (chunk_id, archetype, confidence)
            VALUES (%s, %s, %s)
            ON CONFLICT (chunk_id, archetype) DO UPDATE 
            SET confidence = EXCLUDED.confidence
        """, (chunk_id, archetype, confidence))
    conn.commit()
    cur.close()

def save_chunk_relationships(conn, relationships: List[Dict]):
    """
    Save detected relationships between chunks.
    
    Args:
        conn: PostgreSQL database connection
        relationships: List of relationship dictionaries with:
            - source: Source chunk ID
            - target: Target chunk ID
            - type: Relationship type
            - metadata (optional): Additional relationship data
            
    Example:
        >>> relationships = [{
        ...     'source': 'chunk123',
        ...     'target': 'chunk456',
        ...     'type': 'key-based',
        ...     'metadata': {'key_name': 'user_id'}
        ... }]
        >>> save_chunk_relationships(conn, relationships)
    """
    cur = conn.cursor()
    query = """
    INSERT INTO chunk_relationships 
        (source_chunk, target_chunk, relationship_type, metadata)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (source_chunk, target_chunk, relationship_type)
    DO UPDATE SET
        metadata = EXCLUDED.metadata,
        detected_at = CURRENT_TIMESTAMP;
    """
    
    try:
        for rel in relationships:
            metadata = rel.get('metadata', {})
            cur.execute(query, (
                rel['source'],
                rel['target'],
                rel['type'],
                json.dumps(metadata)
            ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error saving relationships: {e}")
    finally:
        cur.close()

def get_chunk_archetypes(conn, chunk_id: str) -> List[Tuple[str, float]]:
    """
    Retrieve archetypes for a specific chunk.
    
    Args:
        conn: PostgreSQL database connection
        chunk_id: ID of the chunk
        
    Returns:
        list: List of (archetype, confidence) tuples
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT archetype, confidence 
        FROM chunk_archetypes 
        WHERE chunk_id = %s 
        ORDER BY confidence DESC
    """, (chunk_id,))
    return cur.fetchall()

def get_chunk_relationships(conn, chunk_id: str) -> List[Dict]:
    """
    Retrieve relationships for a specific chunk.
    
    Args:
        conn: PostgreSQL database connection
        chunk_id: ID of the chunk
        
    Returns:
        list: List of relationship dictionaries
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT source_chunk, target_chunk, relationship_type, metadata
        FROM chunk_relationships
        WHERE source_chunk = %s OR target_chunk = %s
    """, (chunk_id, chunk_id))
    
    relationships = []
    for row in cur.fetchall():
        relationships.append({
            'source': row[0],
            'target': row[1],
            'type': row[2],
            'metadata': row[3]
        })
    return relationships

def get_chunk_by_id(conn, chunk_id: str) -> Optional[Dict]:
    """Get chunk data by ID."""
    cur = conn.cursor()
    cur.execute("""
        SELECT chunk_json, metadata 
        FROM json_chunks 
        WHERE id = %s
    """, (chunk_id,))
    result = cur.fetchone()
    return json.loads(result[0]) if result else None

def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of file contents."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
