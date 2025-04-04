"""
ChromaDB Storage Module

This module provides ChromaDB integration for vector storage and retrieval.
It implements the same interface as the PostgreSQL storage module for compatibility.
"""

import os
import json
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def init_chroma():
    """
    Initialize ChromaDB client and collection.
    
    Returns:
        ChromaDB client with initialized collection
    """
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "9091"))
    collection_name = os.getenv("CHROMA_COLLECTION", "json_chunks")
    
    client = chromadb.PersistentClient(
        path="/tmp/chroma",
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            is_persistent=True
        )
    )
    
    # Get or create collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    logger.info(f"Initialized ChromaDB collection: {collection_name}")
    return collection

def upsert_chunks(collection, chunks: List[Dict[str, Any]], embeddings: List[List[float]]):
    """
    Upsert chunks and their embeddings into ChromaDB.
    
    Args:
        collection: ChromaDB collection
        chunks: List of chunk dictionaries
        embeddings: List of embeddings for each chunk
    """
    ids = [chunk.get("id", str(i)) for i, chunk in enumerate(chunks)]
    documents = [json.dumps(chunk) for chunk in chunks]
    metadatas = [chunk.get("metadata", {}) for chunk in chunks]
    
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

def query_chunks(collection, query_embedding: List[float], n_results: int = 5, filter_criteria: Dict = None, exclude_ids: List[str] = None) -> List[Dict[str, Any]]:
    """
    Query chunks using similarity search.
    
    Args:
        collection: ChromaDB collection
        query_embedding: Query embedding vector
        n_results: Number of results to return
        filter_criteria: Optional filter to apply (metadata filter)
        exclude_ids: Optional list of IDs to exclude from results
        
    Returns:
        List of chunk dictionaries with their metadata
    """
    # Convert string query to embedding if needed
    if isinstance(query_embedding, str):
        from app.retrieval.embedding import get_embedding
        query_embedding = get_embedding(query_embedding).tolist()
    
    # Prepare query parameters
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_results
    }
    
    # Add filter if provided, properly formatted with operators
    if filter_criteria:
        # ChromaDB requires operators like $eq for filtering
        where_clause = {}
        
        # Convert simple key-value pairs to proper ChromaDB filter format
        for key, value in filter_criteria.items():
            if key.startswith("metadata."):
                field_name = key
            else:
                # Assume it's a metadata field if not specified
                field_name = f"metadata.{key}"
            
            where_clause[field_name] = {"$eq": value}
        
        query_params["where"] = where_clause
    
    # Execute the query
    results = collection.query(**query_params)
    
    chunks = []
    for i in range(len(results["documents"][0])):
        chunk_id = results["ids"][0][i]
        
        # Skip excluded IDs if specified
        if exclude_ids and chunk_id in exclude_ids:
            continue
            
        chunk = {
            "id": chunk_id,
            "content": json.loads(results["documents"][0][i]),
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i]
        }
        chunks.append(chunk)
    
    return chunks

def delete_chunks(collection, chunk_ids: List[str]):
    """
    Delete chunks by their IDs.
    
    Args:
        collection: ChromaDB collection
        chunk_ids: List of chunk IDs to delete
    """
    collection.delete(ids=chunk_ids)

def reset_collection(collection):
    """
    Reset the collection by deleting all chunks.
    
    Args:
        collection: ChromaDB collection
    """
    collection.delete(where={})

def get_chunks(collection, filter_dict: Dict = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get chunks by filter criteria without using embeddings.
    
    Args:
        collection: ChromaDB collection
        filter_dict: Dictionary of filter criteria for metadata fields
        limit: Maximum number of chunks to return
        
    Returns:
        List of chunk dictionaries with their metadata
    """
    try:
        # Format the where clause properly with operators
        where_clause = None
        if filter_dict:
            # ChromaDB requires operators like $eq for filtering
            where_clause = {}
            
            # Convert simple key-value pairs to proper ChromaDB filter format
            # For metadata fields, we need to prefix with "metadata."
            for key, value in filter_dict.items():
                if key.startswith("metadata."):
                    field_name = key
                else:
                    # Assume it's a metadata field if not specified
                    field_name = f"metadata.{key}"
                
                where_clause[field_name] = {"$eq": value}
        
        # Query using properly formatted where clause
        results = collection.get(
            where=where_clause,
            limit=limit
        )
        
        chunks = []
        if results["ids"]:
            for i in range(len(results["ids"])):
                chunk = {
                    "id": results["ids"][i],
                    "content": json.loads(results["documents"][i]),
                    "metadata": results["metadatas"][i]
                }
                chunks.append(chunk)
        
        return chunks
    except Exception as e:
        logger.error(f"Error getting chunks from ChromaDB: {e}")
        return [] 