"""
PostgreSQL Retrieval Module

This module provides retrieval functions specifically for PostgreSQL.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from app.utils.logging_config import get_logger
from app.retrieval.embedding import get_embedding

logger = get_logger(__name__)

def get_relevant_chunks(conn, query: str, limit: int = 10) -> List[Dict]:
    """
    Get relevant chunks from PostgreSQL based on vector similarity.
    
    Args:
        conn: PostgreSQL connection
        query: The search query
        limit: Maximum number of results to return
        
    Returns:
        List of matching chunks with metadata
    """
    try:
        # Generate embedding for the query
        query_embedding = get_embedding(query)
        
        # Convert numpy array to list
        embedding_list = query_embedding.tolist()
        
        # Prepare the embedding string for PostgreSQL
        embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
        
        with conn.cursor() as cur:
            # Query with vector similarity search
            query_sql = """
            SELECT 
                id,
                chunk_json,
                metadata,
                file_path,
                embedding <=> %s::vector as distance
            FROM json_chunks
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            
            cur.execute(query_sql, [embedding_str, embedding_str, limit])
            results = cur.fetchall()
            
            # Format the results
            return [
                {
                    "id": r[0],
                    "content": json.loads(r[1]) if isinstance(r[1], str) else r[1],
                    "metadata": json.loads(r[2]) if isinstance(r[2], str) else r[2],
                    "source_file": r[3],
                    "distance": float(r[4]),
                }
                for r in results
            ]
            
    except Exception as e:
        logger.error(f"Error retrieving chunks from PostgreSQL: {e}")
        return [] 