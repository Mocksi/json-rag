"""
Embedding Generation Module

This module handles the generation and manipulation of vector embeddings for text chunks.
It provides archetype-aware embedding generation and vector similarity search functionality
using the Sentence Transformers library.

The module uses the all-MiniLM-L6-v2 model which provides a good balance between:
- Performance (384-dimensional embeddings)
- Quality (state-of-the-art semantic similarity)
- Speed (efficient inference)

Key Features:
    - Archetype-aware text preprocessing
    - Vector similarity search with filtering
    - Batch embedding generation
    - Entity-specific embedding optimization
"""

from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str, archetype: Optional[Dict] = None) -> np.ndarray:
    """
    Generate embedding vector for text with optional archetype-specific preprocessing.
    
    This function applies specialized preprocessing based on the content archetype
    before generating embeddings. This improves semantic search for specific types
    of content like entities, events, or metrics.
    
    Args:
        text (str): Text to generate embedding for
        archetype (Optional[Dict]): Content archetype information containing:
            - type: The archetype type (entity_definition, event, metric)
            - Additional archetype-specific parameters
            
    Returns:
        np.ndarray: 384-dimensional embedding vector
        
    Examples:
        >>> # Basic embedding
        >>> vec = get_embedding("some text")
        
        >>> # Entity-aware embedding
        >>> vec = get_embedding(json.dumps({"id": "123", "name": "Example"}),
        ...                    {"type": "entity_definition"})
    
    Note:
        Preprocessing strategies:
        - entity_definition: Emphasizes identifier fields
        - event: Prioritizes temporal and state information
        - metric: Highlights numerical values and measurements
    """
    if not archetype:
        return model.encode(text)
        
    # Preprocess based on archetype
    processed_text = text
    if isinstance(text, (dict, list)):
        text = json.dumps(text)
        
    try:
        if archetype['type'] == 'entity_definition':
            data = json.loads(text) if isinstance(text, str) else text
            key_fields = ['id', 'name', 'type', 'code']
            identifiers = " ".join(f"{k}:{data.get(k)}" for k in key_fields if k in data)
            processed_text = f"{identifiers} {text}"
            
        elif archetype['type'] == 'event':
            data = json.loads(text) if isinstance(text, str) else text
            time_fields = ['timestamp', 'date', 'created_at']
            state_fields = ['status', 'state', 'type']
            temporal = " ".join(f"{k}:{data.get(k)}" for k in time_fields if k in data)
            states = " ".join(f"{k}:{data.get(k)}" for k in state_fields if k in data)
            processed_text = f"{temporal} {states} {text}"
            
        elif archetype['type'] == 'metric':
            data = json.loads(text) if isinstance(text, str) else text
            numeric_fields = {k: v for k, v in data.items() if isinstance(v, (int, float))}
            metrics = " ".join(f"{k}:{v}" for k, v in numeric_fields.items())
            processed_text = f"{metrics} {text}"
    except Exception as e:
        print(f"WARNING: Failed to preprocess text for embedding: {e}")
        processed_text = text
        
    return model.encode(processed_text)

def vector_search_with_filter(conn, query: str, allowed_chunk_ids: List[str], top_k: int = 5) -> List[Dict]:
    """
    Perform vector similarity search with optional ID filtering.
    
    This function combines vector similarity search with ID-based filtering,
    allowing for efficient retrieval of semantically similar chunks from a
    subset of the database.
    
    Args:
        conn: PostgreSQL database connection
        query (str): Search query text
        allowed_chunk_ids (List[str]): List of chunk IDs to search within
        top_k (int, optional): Maximum number of results to return. Defaults to 5.
        
    Returns:
        List[Dict]: List of matching chunks, each containing:
            - id: Chunk identifier
            - content: Chunk content
            - score: Similarity score
            - metadata: Additional chunk information
            
    Note:
        The function uses the pgvector extension for efficient
        similarity search in the PostgreSQL database.
    """
    print(f"\nDEBUG: Filtered search for query: '{query}'")
    print(f"DEBUG: Filtering on {len(allowed_chunk_ids) if allowed_chunk_ids else 0} chunk IDs")
    
    query_embedding = embedding_model.encode([query])[0]
    print(f"DEBUG: Generated embedding of size {len(query_embedding)}")
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    cur = conn.cursor()
    if allowed_chunk_ids and len(allowed_chunk_ids) > 0:
        format_ids = ",".join(["%s"] * len(allowed_chunk_ids))
        sql = f"""
        SELECT id, chunk_text::text, embedding <-> '{embedding_str}' as score
        FROM json_chunks
        WHERE id IN ({format_ids})
        ORDER BY score
        LIMIT {top_k};
        """
        cur.execute(sql, tuple(allowed_chunk_ids))
    else:
        sql = f"""
        SELECT id, chunk_text::text, embedding <-> '{embedding_str}' as score
        FROM json_chunks
        ORDER BY score
        LIMIT {top_k};
        """
        cur.execute(sql)

    results = cur.fetchall()
    
    # Log retrieval results
    print("\nDEBUG: Retrieved chunks:")
    retrieved_texts = []
    for chunk_id, chunk_text, score in results:
        print(f"Chunk {chunk_id}: score = {score:.4f}")
        retrieved_texts.append(chunk_text)
    
    cur.close()
    return retrieved_texts

def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1 (list): First embedding vector
        embedding2 (list): Second embedding vector
        
    Returns:
        float: Similarity score between 0 and 1
    """
    # Convert to numpy arrays
    v1 = np.array(embedding1)
    v2 = np.array(embedding2)
    
    # Compute cosine similarity
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
