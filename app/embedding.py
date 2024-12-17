from app.config import embedding_model
import psycopg2

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

def get_embedding(text):
    """
    Generate embedding vector for input text.
    
    Args:
        text (str): Text to embed
        
    Returns:
        list: Embedding vector of fixed dimension
        
    Note:
        Handles invalid input by returning zero vector
        Includes debug output for embedding generation
        Uses sentence-transformers model from config
        
    Example:
        >>> text = "What suppliers are in the East region?"
        >>> embedding = get_embedding(text)
        DEBUG: Generated embedding of size 384 for text: What suppliers...
    """
    try:
        # Ensure text is string and handle empty input
        if not text or not isinstance(text, str):
            print(f"WARNING: Invalid text for embedding: {text}")
            text = str(text) if text else ""
            
        # Generate embedding
        embedding = embedding_model.encode([text])[0]
        
        print(f"DEBUG: Generated embedding of size {len(embedding)} for text: {text[:100]}...")
        return embedding.tolist()  # Convert numpy array to list
        
    except Exception as e:
        print(f"ERROR: Failed to generate embedding: {e}")
        # Return zero vector of correct dimension as fallback
        return [0.0] * embedding_model.get_sentence_embedding_dimension()
