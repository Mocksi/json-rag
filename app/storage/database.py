"""
Database Management Module for JSON RAG System

This module provides database operations for the JSON RAG (Retrieval-Augmented Generation) system,
handling storage, retrieval, and management of JSON document chunks, embeddings, and metadata.
It implements a hierarchical storage system that maintains relationships between document chunks
and supports efficient vector similarity search.

Key Features:
    - Chunk Management: Store and retrieve document chunks with embeddings
    - File Tracking: Monitor changes in JSON files for incremental updates
    - Relationship Mapping: Track parent-child and cross-reference relationships
    - Archetype Detection: Store and retrieve detected data patterns
    - Vector Search: Support for semantic similarity search
    - Schema Evolution: Track and manage database schema changes

Database Schema:
    - json_chunks: Main table for document chunks and embeddings
    - file_metadata: Track file changes and processing status
    - chunk_key_values: Store searchable key-value pairs
    - chunk_archetypes: Store detected data patterns
    - chunk_relationships: Track relationships between chunks

Usage:
    >>> import psycopg2
    >>> from app.storage.database import init_db, store_chunk
    >>> conn = psycopg2.connect(DATABASE_URL)
    >>> init_db(conn)  # Initialize schema
    >>> chunk_data = {'id': 'chunk1', 'content': {...}}
    >>> store_chunk(conn, chunk_data, embedding)

Note:
    - All functions expect a valid PostgreSQL connection object
    - Vector operations require the pgvector extension
    - Functions handle database transactions internally
"""

from typing import List, Dict, Optional, Tuple
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from app.core.config import POSTGRES_CONN_STR
from app.utils.logging_config import get_logger
import uuid

def get_file_info(conn) -> Dict[str, Tuple[str, datetime]]:
    """
    Retrieve stored file metadata from the database.
    
    This function queries the file_metadata table to get information about
    previously processed files. It's used to detect changes in JSON files
    between runs of the system.
    
    Args:
        conn: PostgreSQL database connection
        
    Returns:
        dict: Mapping of filenames to tuples of (hash, modified_time), where:
            - filename (str): Path to the JSON file
            - hash (str): SHA-256 hash of file contents
            - modified_time (datetime): Last modification timestamp
            
    Example:
        >>> conn = psycopg2.connect(DATABASE_URL)
        >>> file_info = get_file_info(conn)
        >>> for filename, (file_hash, mod_time) in file_info.items():
        ...     print(f"{filename}: {mod_time}")
        
    Note:
        - Used by the change detection system
        - Returns empty dict if no files have been processed
        - Timestamps are in UTC
    """
    cur = conn.cursor()
    cur.execute("SELECT filename, file_hash, last_modified FROM file_metadata;")
    rows = cur.fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}

def upsert_file_metadata(conn, filename: str, file_hash: str, mod_time: datetime) -> None:
    """
    Update or insert file metadata in the database.
    
    This function maintains the file_metadata table, which tracks the state
    of processed JSON files. It uses an UPSERT operation to handle both new
    and existing files efficiently.
    
    Args:
        conn: PostgreSQL database connection
        filename (str): Path to the JSON file
        file_hash (str): SHA-256 hash of file contents
        mod_time (datetime): Last modification timestamp (UTC)
        
    Example:
        >>> from datetime import datetime
        >>> conn = psycopg2.connect(DATABASE_URL)
        >>> upsert_file_metadata(
        ...     conn,
        ...     "data/users.json",
        ...     "abc123def456",
        ...     datetime.utcnow()
        ... )
        
    Note:
        - Uses ON CONFLICT DO UPDATE for atomic operations
        - Automatically commits the transaction
        - Timestamps should be in UTC
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

def get_files_to_process(conn, compute_file_hash, get_json_files) -> List[Tuple[str, str, datetime]]:
    """
    Get list of files that need processing based on detected changes.
    
    This function implements the change detection system, comparing current
    file states with stored metadata to identify files that are new or
    have been modified since their last processing.
    
    Process Flow:
        1. Retrieve stored file metadata from database
        2. Scan directory for current JSON files
        3. Compare file hashes and timestamps
        4. Return list of files needing processing
        
    Args:
        conn: PostgreSQL database connection
        compute_file_hash: Function to compute file hash
        get_json_files: Function to get list of JSON files
        
    Returns:
        list: Tuples of (filename, hash, modified_time) for files needing
            processing, where:
            - filename (str): Path to the JSON file
            - hash (str): Current SHA-256 hash of file contents
            - modified_time (datetime): Current modification timestamp
            
    Example:
        >>> conn = psycopg2.connect(DATABASE_URL)
        >>> files = get_files_to_process(
        ...     conn,
        ...     compute_file_hash,
        ...     get_json_files
        ... )
        >>> for f, h, t in files:
        ...     print(f"Processing {f} modified at {t}")
        
    Note:
        - Only returns files that are new or modified
        - Uses SHA-256 hashing for change detection
        - All timestamps are in UTC
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

def reset_database(conn) -> None:
    """
    Reset the database by truncating all tables.
    
    This function provides a way to completely reset the database state
    by truncating all tables. It requires explicit user confirmation
    before proceeding with the destructive operation.
    
    Process Flow:
        1. Display warning message
        2. Request user confirmation
        3. If confirmed, truncate all tables
        4. If error occurs, rollback changes
        
    Args:
        conn: PostgreSQL database connection
        
    Tables Affected:
        - json_chunks
        - file_metadata
        - schema_evolution
        - chunk_key_values
        - chunk_archetypes
        - chunk_relationships
        
    Example:
        >>> conn = psycopg2.connect(DATABASE_URL)
        >>> reset_database(conn)
        Warning: This will delete all stored embeddings and metadata.
        Are you sure you want to reset the database? (yes/no): yes
        Database reset complete.
        
    Note:
        - Requires explicit 'yes' confirmation
        - Operation is atomic (all-or-nothing)
        - Handles errors with automatic rollback
        - Cannot be undone after completion
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

def init_db(conn) -> None:
    """Initialize database schema."""
    cur = conn.cursor()
    
    # Existing tables...
    
    # Update json_summaries table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS json_summaries (
        id SERIAL PRIMARY KEY,
        group_key TEXT,
        group_type TEXT,
        count INTEGER,
        total_value NUMERIC,
        metadata JSONB,
        chunk_ids TEXT[] DEFAULT '{}',
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT unique_group_summary UNIQUE (group_key, group_type)
    )
    """)
    
    # Add new columns and indexes
    cur.execute("ALTER TABLE json_chunks ADD COLUMN IF NOT EXISTS timestamp TIMESTAMP")
    cur.execute("ALTER TABLE json_chunks ADD COLUMN IF NOT EXISTS group_key TEXT")
    cur.execute("ALTER TABLE json_chunks ADD COLUMN IF NOT EXISTS group_type TEXT")
    
    # Create indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_summaries_group_type ON json_summaries(group_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_summaries_chunks ON json_summaries USING GIN(chunk_ids)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_group ON json_chunks(group_key, group_type)")
    
    conn.commit()

def save_chunk_archetypes(conn, chunk_id: str, archetypes: List[Tuple[str, float]]) -> None:
    """
    Save detected archetypes for a document chunk.
    
    This function stores or updates the archetype classifications for a given
    document chunk. Each archetype represents a detected data pattern with an
    associated confidence score.
    
    Args:
        conn: PostgreSQL database connection
        chunk_id (str): Unique identifier of the chunk
        archetypes: List of tuples (archetype_name, confidence_score), where:
            - archetype_name (str): Name of the detected pattern
            - confidence_score (float): Confidence level (0.0 to 1.0)
            
    Example:
        >>> conn = psycopg2.connect(DATABASE_URL)
        >>> archetypes = [
        ...     ('event_log', 0.95),
        ...     ('metric_data', 0.75)
        ... ]
        >>> save_chunk_archetypes(conn, 'chunk123', archetypes)
        
    Note:
        - Uses UPSERT to handle updates
        - Automatically commits changes
        - Timestamps are set automatically
        - Existing archetypes are updated with new confidence scores
    """
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

def save_chunk_relationships(conn, relationships: List[Dict]) -> None:
    """
    Save detected relationships between document chunks.
    
    This function stores or updates relationships between chunks, maintaining
    a graph-like structure of connections in the document collection. Each
    relationship has a type and optional metadata.
    
    Args:
        conn: PostgreSQL database connection
        relationships: List of relationship dictionaries containing:
            - source (str): ID of source chunk
            - target (str): ID of target chunk
            - type (str): Relationship type (e.g., 'parent', 'reference')
            - metadata (dict, optional): Additional relationship data
            
    Example:
        >>> conn = psycopg2.connect(DATABASE_URL)
        >>> relationships = [{
        ...     'source': 'chunk123',
        ...     'target': 'chunk456',
        ...     'type': 'key-based',
        ...     'metadata': {'key_name': 'user_id', 'confidence': 0.95}
        ... }]
        >>> save_chunk_relationships(conn, relationships)
        
    Note:
        - Uses UPSERT to handle updates
        - Automatically commits changes
        - Timestamps are updated on each save
        - Handles errors with automatic rollback
        - Metadata is stored as JSONB for flexible querying
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

def get_chunk_relationships(conn, chunk_id: str) -> List[Dict]:
    """Get relationships for a chunk."""
    cur = None
    try:
        cur = conn.cursor()
        
        query = """
        SELECT 
            r.source_chunk_id,
            r.target_chunk_id,
            r.relationship_type,
            r.metadata,
            s.chunk_json as source_data,
            t.chunk_json as target_data
        FROM chunk_relationships r
        JOIN json_chunks s ON r.source_chunk_id = s.id
        JOIN json_chunks t ON r.target_chunk_id = t.id
        WHERE r.source_chunk_id = %s OR r.target_chunk_id = %s
        """
        
        cur.execute(query, (chunk_id, chunk_id))
        results = cur.fetchall()
        
        relationships = []
        for row in results:
            source_id, target_id, rel_type, metadata, source_data, target_data = row
            relationships.append({
                'source_id': source_id,
                'target_id': target_id,
                'type': rel_type,
                'metadata': metadata,
                'source_data': source_data,
                'target_data': target_data
            })
            
        return relationships
        
    except Exception as e:
        logger.error(f"Error getting relationships for chunk {chunk_id}: {e}")
        if conn:
            conn.rollback()
        return []
        
    finally:
        if cur and not cur.closed:
            cur.close()

def get_chunk_archetypes(conn, chunk_id: str) -> List[Tuple[str, float]]:
    """Get archetypes for a chunk."""
    cur = None
    try:
        cur = conn.cursor()
        
        query = """
        SELECT 
            archetype,
            confidence
        FROM chunk_archetypes
        WHERE chunk_id = %s
        ORDER BY confidence DESC
        """
        
        cur.execute(query, (chunk_id,))
        return cur.fetchall()
        
    except Exception as e:
        logger.error(f"Error getting archetypes for chunk {chunk_id}: {e}")
        return []
        
    finally:
        if cur and not cur.closed:
            cur.close()

def get_chunk_by_id(conn, chunk_id: str) -> Optional[Dict]:
    """
    Retrieve a document chunk by its unique identifier.
    
    This function fetches a specific chunk's data and metadata from the
    database. It's commonly used to resolve references and retrieve
    detailed chunk information.
    
    Args:
        conn: PostgreSQL database connection
        chunk_id (str): Unique identifier of the chunk
        
    Returns:
        dict or None: Chunk data if found, containing:
            - content: Original JSON content of the chunk
            - metadata: Additional chunk information
            Returns None if chunk not found
            
    Example:
        >>> conn = psycopg2.connect(DATABASE_URL)
        >>> chunk = get_chunk_by_id(conn, 'chunk123')
        >>> if chunk:
        ...     print(f"Content: {chunk['content']}")
        ...     print(f"Metadata: {chunk['metadata']}")
        
    Note:
        - Returns None if chunk doesn't exist
        - JSON content is automatically parsed
        - Efficient single-row query
        - Does not include embedding vector
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT chunk_json, metadata 
        FROM json_chunks 
        WHERE id = %s
    """, (chunk_id,))
    result = cur.fetchone()
    return json.loads(result[0]) if result else None

def compute_file_hash(filepath: str) -> str:
    """
    Compute SHA-256 hash of a file's contents.
    
    This function generates a cryptographic hash of a file's contents,
    used for detecting changes in files between processing runs. It reads
    the file in chunks to efficiently handle large files.
    
    Args:
        filepath (str): Path to the file to hash
        
    Returns:
        str: Hexadecimal string of the SHA-256 hash
        
    Example:
        >>> file_hash = compute_file_hash('data/users.json')
        >>> print(f"File hash: {file_hash}")
        File hash: 8d969eef6ecad3c29a3a629280e686cf0c3f5d5a86aff3ca12020c923adc6c92
        
    Note:
        - Uses SHA-256 algorithm
        - Processes file in 4KB chunks
        - Returns lowercase hex string
        - Handles large files efficiently
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def store_chunk(conn, chunk: Dict, timestamp: Optional[str] = None,
                group_key: Optional[str] = None, 
                group_type: Optional[str] = None) -> str:
    """Store chunk with grouping information."""
    chunk_id = str(uuid.uuid4())
    
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO json_chunks 
            (id, chunk_json, timestamp, group_key, group_type)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """, (
            chunk_id,
            json.dumps(chunk),
            timestamp,
            group_key,
            group_type
        ))
        
    conn.commit()
    return chunk_id

def get_parent_chunks(conn, chunk_id: str, max_depth: int = 3) -> List[Dict]:
    """
    Retrieve parent chunks up to a specified depth in the hierarchy.
    
    This function traverses up the chunk hierarchy, collecting parent chunks
    until it reaches either the root or the specified maximum depth. The
    results are ordered from root to direct parent.
    
    Args:
        conn: PostgreSQL database connection
        chunk_id (str): Unique identifier of the starting chunk
        max_depth (int, optional): Maximum number of parent levels to traverse.
            Defaults to 3.
            
    Returns:
        list: List of parent chunks ordered from root to direct parent,
            each containing:
            - id (str): Chunk identifier
            - content (dict): JSON content
            - metadata (dict): Additional information
            
    Example:
        >>> conn = psycopg2.connect(DATABASE_URL)
        >>> parents = get_parent_chunks('chunk123', max_depth=2)
        >>> for p in parents:
        ...     print(f"Parent {p['id']}: {p['content']}")
        Parent root_chunk: {'type': 'root'}
        Parent mid_chunk: {'type': 'section'}
        
    Note:
        - Returns empty list if chunk has no parents
        - Respects max_depth parameter
        - Orders from most distant to closest parent
        - Includes full content and metadata
        - Stops at root node (no parent_id)
    """
    parents = []
    current_id = chunk_id
    depth = 0
    
    while depth < max_depth:
        cur = conn.cursor()
        cur.execute("""
            SELECT parent_id, chunk_json, metadata 
            FROM json_chunks 
            WHERE id = %s AND parent_id IS NOT NULL
        """, (current_id,))
        result = cur.fetchone()
        
        if not result:
            break
            
        parent_id, chunk_json, metadata = result
        parents.append({
            'id': parent_id,
            'content': json.loads(chunk_json),
            'metadata': metadata
        })
        
        current_id = parent_id
        depth += 1
        
    return list(reversed(parents))  # Return from root to direct parent

def get_child_chunks(conn, parent_id: str, max_depth: int = 2) -> List[Dict]:
    """
    Retrieve child chunks up to a specified depth in the hierarchy.
    
    This function recursively traverses down the chunk hierarchy, collecting
    all descendant chunks until it reaches the specified maximum depth. The
    results are ordered by level and depth.
    
    Args:
        conn: PostgreSQL database connection
        parent_id (str): Unique identifier of the parent chunk
        max_depth (int, optional): Maximum depth to traverse. Defaults to 2.
            
    Returns:
        list: List of child chunks ordered by level and depth, each containing:
            - id (str): Chunk identifier
            - content (dict): JSON content
            - metadata (dict): Additional information
            - depth (int): Nesting level from parent
            
    Example:
        >>> conn = psycopg2.connect(DATABASE_URL)
        >>> children = get_child_chunks('parent123', max_depth=2)
        >>> for child in children:
        ...     print(f"Child at depth {child['depth']}: {child['id']}")
        Child at depth 1: chunk456
        Child at depth 2: chunk789
        
    Note:
        - Uses recursive CTE for efficient traversal
        - Returns empty list if no children found
        - Orders by level (breadth) then depth
        - Includes full content and metadata
        - Respects max_depth parameter
    """
    cur = conn.cursor()
    cur.execute("""
        WITH RECURSIVE chunk_tree AS (
            -- Base case: direct children
            SELECT id, chunk_json, metadata, depth, 1 as level
            FROM json_chunks
            WHERE parent_id = %s
            
            UNION ALL
            
            -- Recursive case: children of children
            SELECT c.id, c.chunk_json, c.metadata, c.depth, ct.level + 1
            FROM json_chunks c
            INNER JOIN chunk_tree ct ON c.parent_id = ct.id
            WHERE ct.level < %s
        )
        SELECT id, chunk_json, metadata, depth
        FROM chunk_tree
        ORDER BY level, depth;
    """, (parent_id, max_depth))
    
    return [{
        'id': row[0],
        'content': json.loads(row[1]),
        'metadata': row[2],
        'depth': row[3]
    } for row in cur.fetchall()]

def get_chunk_with_context(conn, chunk_id: str, max_parent_depth: int = 3, max_child_depth: int = 2) -> Dict:
    """
    Retrieve a chunk with its complete hierarchical context.
    
    This function provides a comprehensive view of a chunk within its document
    hierarchy, including its own data, parent context, and child chunks. It
    combines upward and downward traversal to build a complete picture.
    
    Args:
        conn: PostgreSQL database connection
        chunk_id (str): Unique identifier of the chunk
        max_parent_depth (int, optional): Maximum levels to traverse upward.
            Defaults to 3.
        max_child_depth (int, optional): Maximum levels to traverse downward.
            Defaults to 2.
            
    Returns:
        dict or None: Complete chunk context if found, containing:
            - id (str): Chunk identifier
            - content (dict): JSON content
            - metadata (dict): Additional information
            - depth (int): Nesting level in document
            - path (str): JSON path to chunk
            - parents (list): Parent chunks from root
            - children (list): Child chunks by level
            Returns None if chunk not found
            
    Example:
        >>> conn = psycopg2.connect(DATABASE_URL)
        >>> context = get_chunk_with_context('chunk123')
        >>> if context:
        ...     print(f"Chunk: {context['id']}")
        ...     print(f"Parents: {len(context['parents'])}")
        ...     print(f"Children: {len(context['children'])}")
        
    Note:
        - Returns None if chunk doesn't exist
        - Includes complete hierarchical context
        - Parents ordered from root to direct parent
        - Children ordered by level and depth
        - Efficient recursive queries for traversal
    """
    # Get the main chunk
    cur = conn.cursor()
    cur.execute("""
        SELECT chunk_json, metadata, depth, path
        FROM json_chunks
        WHERE id = %s
    """, (chunk_id,))
    result = cur.fetchone()
    
    if not result:
        return None
        
    chunk_json, metadata, depth, path = result
    
    # Get parent and child chunks
    parents = get_parent_chunks(conn, chunk_id, max_parent_depth)
    children = get_child_chunks(conn, chunk_id, max_child_depth)
    
    return {
        'id': chunk_id,
        'content': json.loads(chunk_json),
        'metadata': metadata,
        'depth': depth,
        'path': path,
        'parents': parents,
        'children': children
    }

def update_summary(conn, group_key: str, group_type: str, data: Dict):
    """Update or insert summary record."""
    with conn.cursor() as cur:
        cur.execute("""
        INSERT INTO json_summaries (group_key, group_type, count, total_value, metadata)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (group_key, group_type) DO UPDATE
        SET count = %s,
            total_value = %s,
            metadata = %s,
            last_updated = CURRENT_TIMESTAMP
        """, (
            group_key, group_type, 
            data['count'], data['total_value'], json.dumps(data.get('metadata', {})),
            data['count'], data['total_value'], json.dumps(data.get('metadata', {}))
        ))
    conn.commit()
