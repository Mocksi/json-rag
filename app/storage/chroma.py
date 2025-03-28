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

def query_chunks(collection, query_embedding: List[float], n_results: int = 5) -> List[Dict[str, Any]]:
    """
    Query chunks using similarity search.
    
    Args:
        collection: ChromaDB collection
        query_embedding: Query embedding vector
        n_results: Number of results to return
        
    Returns:
        List of chunk dictionaries with their metadata
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    chunks = []
    for i in range(len(results["documents"][0])):
        chunk = {
            "id": results["ids"][0][i],
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