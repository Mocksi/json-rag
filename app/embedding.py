from app.config import embedding_model
import psycopg2

def vector_search_with_filter(conn, query, allowed_chunk_ids, top_k):
    """
    Perform vector similarity search with optional ID filtering.
    
    Args:
        conn: PostgreSQL database connection
        query (str): Query string
        allowed_chunk_ids (list): List of allowed chunk IDs
        top_k (int): Number of results to return
        
    Returns:
        list: Retrieved chunks ordered by relevance
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
