"""
ChromaDB Retrieval Module

This module provides retrieval functions for ChromaDB collections.
"""

import logging
from typing import List, Dict, Any, Optional
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def search_chunks(collection, query: str, limit: int = 5) -> List[Dict]:
    """
    Search for chunks in ChromaDB collection using the query.
    
    Args:
        collection: ChromaDB collection
        query: Search query
        limit: Maximum number of results
        
    Returns:
        List of matching chunks
    """
    try:
        # Import here to avoid circular imports
        from app.storage.chroma import query_chunks
        from app.retrieval.embedding import get_embedding
        
        # Generate embedding for the query
        embedding = get_embedding(query)
        
        # Convert NumPy array to Python list (ChromaDB requires this)
        embedding_list = embedding.tolist()
        
        # Use storage.chroma query_chunks function with list embedding
        chunks = query_chunks(collection, embedding_list, limit)
        
        logger.info(f"Retrieved {len(chunks)} chunks from ChromaDB")
        return chunks
    except Exception as e:
        logger.error(f"Error searching ChromaDB: {e}")
        return [] 