"""
Advanced Retrieval Module

This module provides enhanced retrieval functionality that leverages the hierarchical
and index-based structure created by the advanced processor. It enables more efficient
and accurate retrievals from large JSON files like adMetrics.json.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple

from app.retrieval.embedding import get_embedding

logger = logging.getLogger(__name__)

def retrieve_from_hierarchical_chunks(
    store: Any,
    query: str,
    top_k: int = 10,
    filter_criteria: Optional[Dict] = None,
    file_path: Optional[str] = None
) -> List[Dict]:
    """
    Retrieve chunks using hierarchical retrieval strategy.
    
    This function:
    1. First retrieves summary chunks that match the query
    2. Follows those to retrieve detailed chunks
    3. Optionally, retrieves related chunks based on relationships
    
    Args:
        store: Database backend (PostgreSQL or ChromaDB)
        query: The search query
        top_k: Number of top results to retrieve (initial)
        filter_criteria: Optional filters to apply to the search
        file_path: Optional file path to narrow search
        
    Returns:
        List of retrieved chunks matching the query
    """
    is_postgres = hasattr(store, 'cursor')
    
    # Start by looking for matching summary chunks
    summary_filter = {"chunk_type": "summary"}
    if file_path:
        summary_filter["source_file"] = file_path
    
    # Combine with any user-provided filter criteria
    if filter_criteria:
        summary_filter.update(filter_criteria)
    
    # Get summary chunks first - they provide an overview
    if is_postgres:
        # Using PostgreSQL - construct vector query
        query_embedding = get_embedding(query).tolist()
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        
        # Construct filter conditions
        filter_conditions = []
        filter_values = []
        
        for k, v in summary_filter.items():
            filter_conditions.append(f"metadata->>'{k}' = %s")
            filter_values.append(str(v))
        
        filter_clause = ""
        if filter_conditions:
            filter_clause = "AND " + " AND ".join(filter_conditions)
        
        cur = store.cursor()
        try:
            # Query to find matching summary chunks
            cur.execute(
                f"""
                SELECT id, chunk_json, metadata, embedding <=> %s::vector AS distance 
                FROM json_chunks
                WHERE (metadata->>'chunk_type' = 'summary' OR metadata->>'chunk_type' = 'index') {filter_clause}
                ORDER BY distance
                LIMIT %s
                """,
                [embedding_str] + filter_values + [top_k]
            )
            
            summary_results = []
            for row in cur.fetchall():
                chunk_id, chunk_json, metadata, distance = row
                summary_results.append({
                    "id": chunk_id,
                    "content": json.loads(chunk_json) if isinstance(chunk_json, str) else chunk_json,
                    "metadata": json.loads(metadata) if isinstance(metadata, str) else metadata,
                    "distance": float(distance)
                })
        except Exception as e:
            logger.error(f"Error retrieving summary chunks: {e}")
            cur.close()
            return []
    else:
        # Using ChromaDB
        from app.storage.chroma import query_chunks
        
        # We can directly pass the filter to ChromaDB
        summary_results = query_chunks(store, query, top_k, summary_filter)
    
    # If no summaries found, fall back to standard retrieval
    if not summary_results:
        logger.info("No summary chunks found, falling back to standard retrieval")
        if is_postgres:
            # Standard retrieval with PostgreSQL
            query_embedding = get_embedding(query).tolist()
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            cur = store.cursor()
            try:
                cur.execute(
                    """
                    SELECT id, chunk_json, metadata, embedding <=> %s::vector AS distance 
                    FROM json_chunks
                    ORDER BY distance
                    LIMIT %s
                    """,
                    [embedding_str, top_k * 2]  # Get more results when falling back
                )
                
                standard_results = []
                for row in cur.fetchall():
                    chunk_id, chunk_json, metadata, distance = row
                    standard_results.append({
                        "id": chunk_id,
                        "content": json.loads(chunk_json) if isinstance(chunk_json, str) else chunk_json,
                        "metadata": json.loads(metadata) if isinstance(metadata, str) else metadata,
                        "distance": float(distance)
                    })
                return standard_results
            finally:
                cur.close()
        else:
            # Standard retrieval with ChromaDB
            from app.storage.chroma import query_chunks
            return query_chunks(store, query, top_k * 2)
    
    # Process summary results to get detail chunks
    detail_chunks = []
    seen_ids = set()
    
    for summary in summary_results:
        # Add the summary itself for context
        if summary["id"] not in seen_ids:
            detail_chunks.append(summary)
            seen_ids.add(summary["id"])
        
        # Handle index results specially
        if summary["metadata"].get("chunk_type") == "index":
            # For index chunks, we need to find relevant paths and then fetch those chunks
            index_content = summary["content"]
            # Find paths that might be relevant to the query
            relevant_paths = find_relevant_paths_in_index(index_content, query)
            
            # Fetch chunks for these paths
            for path in relevant_paths:
                path_chunks = get_chunks_by_path(store, path, summary["metadata"].get("source_file"))
                for chunk in path_chunks:
                    if chunk["id"] not in seen_ids:
                        detail_chunks.append(chunk)
                        seen_ids.add(chunk["id"])
        else:
            # For summary chunks, find the detail chunks they refer to
            if "original_path" in summary["metadata"]:
                original_path = summary["metadata"]["original_path"]
                
                # Get detailed chunks for this path
                path_chunks = get_chunks_by_path(store, original_path, summary["metadata"].get("source_file"))
                for chunk in path_chunks:
                    if chunk["id"] not in seen_ids:
                        detail_chunks.append(chunk)
                        seen_ids.add(chunk["id"])
            
            # For batch summaries, find their item chunks
            if summary["metadata"].get("chunk_type") == "batch":
                batch_path = summary["metadata"].get("path")
                if batch_path:
                    # Get all detail chunks within this batch
                    batch_chunks = get_chunks_by_parent_path(store, batch_path, summary["metadata"].get("source_file"))
                    for chunk in batch_chunks:
                        if chunk["id"] not in seen_ids:
                            detail_chunks.append(chunk)
                            seen_ids.add(chunk["id"])
    
    # Combine results and score against query
    combined_results = detail_chunks.copy()
    
    # If we have very few chunks, try standard retrieval for this query
    if len(combined_results) < top_k:
        logger.info(f"Hierarchical retrieval returned only {len(combined_results)} chunks, supplementing with standard retrieval")
        
        # Get standard search results
        if is_postgres:
            query_embedding = get_embedding(query).tolist()
            embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
            
            cur = store.cursor()
            try:
                # Exclude already seen chunks
                excluding_ids = ', '.join([f"'{id}'" for id in seen_ids])
                exclude_clause = f"AND id NOT IN ({excluding_ids})" if seen_ids else ""
                
                cur.execute(
                    f"""
                    SELECT id, chunk_json, metadata, embedding <=> %s::vector AS distance 
                    FROM json_chunks
                    WHERE 1=1 {exclude_clause}
                    ORDER BY distance
                    LIMIT %s
                    """,
                    [embedding_str, top_k]
                )
                
                for row in cur.fetchall():
                    chunk_id, chunk_json, metadata, distance = row
                    if chunk_id not in seen_ids:
                        combined_results.append({
                            "id": chunk_id,
                            "content": json.loads(chunk_json) if isinstance(chunk_json, str) else chunk_json,
                            "metadata": json.loads(metadata) if isinstance(metadata, str) else metadata,
                            "distance": float(distance)
                        })
                        seen_ids.add(chunk_id)
            finally:
                cur.close()
        else:
            # ChromaDB version
            from app.storage.chroma import query_chunks
            exclude_ids = list(seen_ids)
            standard_results = query_chunks(store, query, top_k, exclude_ids=exclude_ids)
            for chunk in standard_results:
                if chunk["id"] not in seen_ids:
                    combined_results.append(chunk)
                    seen_ids.add(chunk["id"])
    
    # Return the combined results
    return combined_results[:top_k*2]  # Limit to avoid returning too many chunks

def find_relevant_paths_in_index(index: Dict, query: str) -> List[str]:
    """
    Find paths in the index that are relevant to the query.
    
    Args:
        index: The index structure from advanced processing
        query: The search query
        
    Returns:
        List of relevant paths in the index
    """
    # Simplistic approach: look for keywords in paths
    # In a real-world scenario, this would use more sophisticated NLP
    query_terms = set(query.lower().split())
    relevant_paths = []
    
    for path, path_info in index.get("paths", {}).items():
        # Check if any query term appears in the path
        if any(term in path.lower() for term in query_terms):
            relevant_paths.append(path)
        
        # Check fields for object paths
        if isinstance(path_info, dict) and "fields" in path_info:
            if any(term in str(path_info["fields"]).lower() for term in query_terms):
                relevant_paths.append(path)
    
    # Return a reasonable number of paths
    return relevant_paths[:10]

def get_chunks_by_path(store: Any, path: str, file_path: Optional[str] = None) -> List[Dict]:
    """
    Get chunks with a specific path.
    
    Args:
        store: Database backend
        path: The path to look for
        file_path: Optional file path to narrow search
        
    Returns:
        List of chunks with the given path
    """
    is_postgres = hasattr(store, 'cursor')
    
    if is_postgres:
        cur = store.cursor()
        try:
            conditions = ["path = %s"]
            params = [path]
            
            if file_path:
                conditions.append("file_path = %s")
                params.append(file_path)
            
            where_clause = " AND ".join(conditions)
            
            cur.execute(
                f"""
                SELECT id, chunk_json, metadata 
                FROM json_chunks
                WHERE {where_clause}
                LIMIT 10
                """,
                params
            )
            
            results = []
            for row in cur.fetchall():
                chunk_id, chunk_json, metadata = row
                results.append({
                    "id": chunk_id,
                    "content": json.loads(chunk_json) if isinstance(chunk_json, str) else chunk_json,
                    "metadata": json.loads(metadata) if isinstance(metadata, str) else metadata
                })
            return results
        finally:
            cur.close()
    else:
        # ChromaDB version
        from app.storage.chroma import get_chunks
        
        filter_dict = {"path": path}
        if file_path:
            filter_dict["source_file"] = file_path
            
        return get_chunks(store, filter_dict=filter_dict)

def get_chunks_by_parent_path(store: Any, parent_path: str, file_path: Optional[str] = None) -> List[Dict]:
    """
    Get chunks with a specific parent path.
    
    Args:
        store: Database backend
        parent_path: The parent path to look for
        file_path: Optional file path to narrow search
        
    Returns:
        List of chunks with the given parent path
    """
    is_postgres = hasattr(store, 'cursor')
    
    if is_postgres:
        cur = store.cursor()
        try:
            conditions = ["metadata->>'parent_path' = %s"]
            params = [parent_path]
            
            if file_path:
                conditions.append("file_path = %s")
                params.append(file_path)
            
            where_clause = " AND ".join(conditions)
            
            cur.execute(
                f"""
                SELECT id, chunk_json, metadata 
                FROM json_chunks
                WHERE {where_clause}
                LIMIT 50
                """,
                params
            )
            
            results = []
            for row in cur.fetchall():
                chunk_id, chunk_json, metadata = row
                results.append({
                    "id": chunk_id,
                    "content": json.loads(chunk_json) if isinstance(chunk_json, str) else chunk_json,
                    "metadata": json.loads(metadata) if isinstance(metadata, str) else metadata
                })
            return results
        finally:
            cur.close()
    else:
        # ChromaDB version
        from app.storage.chroma import get_chunks
        
        filter_dict = {"metadata.parent_path": parent_path}
        if file_path:
            filter_dict["metadata.source_file"] = file_path
            
        return get_chunks(store, filter_dict=filter_dict)

def hierarchical_search(store: Any, query: str, file_path: Optional[str] = None) -> List[Dict]:
    """
    Perform hierarchical search using summaries and details.
    
    Args:
        store: Database backend
        query: Search query
        file_path: Optional file to limit search to
        
    Returns:
        List of relevant chunks
    """
    # First get matching summary chunks
    summary_results = retrieve_from_hierarchical_chunks(
        store, 
        query, 
        top_k=5,
        filter_criteria={"chunk_type": "summary"},
        file_path=file_path
    )
    
    # Extract relevant information
    search_results = []
    for chunk in summary_results:
        # Clean up and format the chunk for display
        search_results.append({
            "id": chunk["id"],
            "content": chunk["content"],
            "metadata": chunk["metadata"],
            "relevance": 1.0 - chunk.get("distance", 0)
        })
    
    return search_results 