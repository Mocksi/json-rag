from typing import List, Dict, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str, archetype: Optional[Dict] = None) -> np.ndarray:
    """Get embedding vector for text, with archetype-specific preprocessing."""
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

def vector_search_with_filter(conn, query, allowed_chunk_ids, top_k):
    """
    Perform vector similarity search with optional ID filtering.
    
    Args:
        conn: PostgreSQL database connection
        query (str): Query string to embed and search
        allowed_chunk_ids (list): List of chunk IDs to filter results
        top_k (int): Maximum number of results to return
        
    Returns:
        list: Retrieved chunks ordered by relevance score
        
    Process:
        1. Generates embedding for query text
        2. Performs cosine similarity search in database
        3. Optionally filters by allowed chunk IDs
        4. Returns top_k most similar chunks
        
    Note:
        Uses PGVector for efficient similarity search
        Includes debug output for query processing
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
