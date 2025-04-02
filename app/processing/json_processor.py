"""
JSON Processing Module

This module provides streamlined functionality for processing JSON files into chunks
and storing them in a vector database (either PostgreSQL or ChromaDB). It handles
large JSON files efficiently using streaming techniques when possible.
"""

import os
import json
import ijson
import logging
import decimal
from typing import List, Dict, Any, Union
from pathlib import Path
from tqdm import tqdm

# Import necessary functions from other modules
from app.processing.json_parser import json_to_path_chunks, generate_chunk_id
from app.retrieval.embedding import get_embedding

logger = logging.getLogger(__name__)

# Custom JSON encoder to handle Decimal types and other non-standard JSON types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        return super().default(obj)

def process_json_files(directory: str, store) -> List[Dict]:
    """
    Process all JSON files in a directory and store them in the provided database.
    
    This function handles:
    1. Discovering JSON files in the directory (recursive)
    2. Parsing JSON files - using streaming for large arrays when possible
    3. Chunking the content into semantic units
    4. Generating embeddings for each chunk
    5. Storing the chunks and embeddings in the database
    
    For exceptionally large files like adMetrics.json, it automatically uses
    advanced hierarchical and index-based processing techniques.
    
    Args:
        directory (str): Path to directory containing JSON files
        store: Database connection (PostgreSQL connection or ChromaDB collection)
        
    Returns:
        List[Dict]: List of processed chunks with their metadata and IDs
        
    Note:
        - Uses streaming processing for files with large arrays when possible
        - Falls back to full file loading when streaming is not applicable
        - Compatible with both PostgreSQL and ChromaDB backends
        - Returns chunk data needed for subsequent relationship processing
        - Uses advanced techniques for very large files (>10MB)
    """
    # Import here to avoid circular imports
    try:
        from app.processing.advanced_processor import process_json_files_advanced
        advanced_processing_available = True
    except ImportError:
        advanced_processing_available = False
        logger.warning("Advanced processing module not available, falling back to standard processing")
    
    # For large directories or if adMetrics.json is present, use advanced processing if available
    if advanced_processing_available:
        # Check if directory contains any very large files or adMetrics.json
        has_large_files = False
        has_ad_metrics = False
        
        for root, _, files in os.walk(directory):
            for file in files:
                if not file.endswith(".json"):
                    continue
                    
                file_path = os.path.join(root, file)
                if file == "adMetrics.json":
                    has_ad_metrics = True
                    break
                    
                try:
                    if os.path.getsize(file_path) > 10_000_000:  # 10MB
                        has_large_files = True
                        break
                except:
                    pass
            
            if has_large_files or has_ad_metrics:
                break
        
        if has_large_files or has_ad_metrics:
            logger.info("Using advanced processing due to large files or adMetrics.json")
            # To avoid circular imports, we'll import here
            from app.processing.advanced_processor import process_json_files_advanced
            
            # Get the full file paths of large files for targeted advanced processing
            large_file_paths = []
            for root, _, files in os.walk(directory):
                for file in files:
                    if not file.endswith(".json"):
                        continue
                    
                    file_path = os.path.join(root, file)
                    if file == "adMetrics.json" or (os.path.getsize(file_path) > 10_000_000):
                        large_file_paths.append(file_path)
            
            # Use advanced processing only for the large files
            advanced_chunks = []
            for file_path in large_file_paths:
                logger.info(f"Using advanced processing for large file: {file_path}")
                # Process each large file individually with advanced processing
                chunks = process_json_files_advanced([file_path], store)
                advanced_chunks.extend(chunks)
            
            return advanced_chunks
    
    # Otherwise, continue with standard processing
    # Determine if we're using PostgreSQL or ChromaDB
    is_postgres = hasattr(store, 'cursor')
    logger.info(f"Starting JSON processing in directory: {directory} with {'PostgreSQL' if is_postgres else 'ChromaDB'}")
    
    # List to store all processed chunks (for returning and relationship processing)
    processed_chunks = []
    
    # Prepare for ChromaDB batch insertion if needed
    chroma_chunks = []
    chroma_embeddings = []
    
    # Initialize cursor for PostgreSQL if needed
    cur = None
    if is_postgres:
        cur = store.cursor()
    
    try:
        # Walk through directory and process all JSON files
        for root, _, files in os.walk(directory):
            for file in tqdm(files, desc="Processing JSON files"):
                # Skip non-JSON files or specific files
                if not file.endswith(".json") or file == "test_queries":
                    continue
                
                file_path = os.path.join(root, file)
                logger.info(f"Processing file: {file_path}")
                
                try:
                    # Try streaming processing first - assume structure with arrays
                    chunks_from_file = []
                    streaming_worked = False
                    
                    # Attempt to stream from common JSON array patterns
                    array_paths = ['metrics.item', 'data.item', 'results.item', 'items.item', 'records.item', 'item']
                    
                    for array_path in array_paths:
                        try:
                            with open(file_path, 'rb') as f:
                                base_path = array_path.split('.')[0]  # Extract base path (e.g., 'metrics' from 'metrics.item')
                                items = ijson.items(f, array_path)
                                
                                # Process each item in the array
                                for i, item in enumerate(items):
                                    item_base_path = f'{base_path}[{i}]'  # Create base path like metrics[0], metrics[1], etc.
                                    
                                    try:
                                        # Generate chunks for this item
                                        item_chunks = json_to_path_chunks(
                                            item,
                                            file_path=str(file_path),
                                            base_path=item_base_path
                                        )
                                        
                                        # Process and store these chunks
                                        for chunk in item_chunks:
                                            # Generate a unique ID for this chunk
                                            chunk_id = generate_chunk_id(str(file_path), chunk.get("path", f"{item_base_path}_{i}"))
                                            
                                            # Store the chunks
                                            chunks_from_file.append((chunk_id, chunk, file_path))
                                    except Exception as e:
                                        logger.error(f"Error chunking item {i} in {file_path}: {e}")
                                        continue
                                
                                streaming_worked = True
                                logger.info(f"Successfully streamed from '{array_path}' in {file_path}")
                                break  # Exit the array_paths loop if successful
                                
                        except ijson.JSONError:
                            # This array path didn't work, try the next one
                            continue
                        except Exception as e:
                            logger.error(f"Unexpected error trying to stream from '{array_path}' in {file_path}: {e}")
                            continue
                    
                    # If streaming didn't work, fall back to full file loading
                    if not streaming_worked:
                        logger.info(f"Falling back to full file load for {file_path}")
                        try:
                            with open(file_path, 'r') as f:
                                data = json.load(f)
                                
                            # Generate chunks for the entire document
                            file_chunks = json_to_path_chunks(
                                data,
                                file_path=str(file_path)
                            )
                            
                            # Process and store these chunks
                            for chunk in file_chunks:
                                chunk_id = generate_chunk_id(str(file_path), chunk.get("path", ""))
                                chunks_from_file.append((chunk_id, chunk, file_path))
                                
                        except Exception as e:
                            logger.error(f"Error processing full file {file_path}: {e}")
                            continue
                    
                    # Process all chunks collected from this file
                    for chunk_id, chunk, file_path in chunks_from_file:
                        # Create metadata for the chunk
                        chunk_metadata = {
                            "source_file": file_path,
                            "path": chunk.get("path", ""),
                        }
                        
                        # If we have any archetype info, include it
                        if "archetypes" in chunk:
                            chunk_metadata["archetypes"] = chunk["archetypes"]
                        
                        # Generate embedding for the chunk
                        try:
                            # Serialize the chunk for embedding - use custom encoder to handle Decimal types
                            embedding_text = json.dumps(chunk, cls=CustomJSONEncoder)
                            
                            # Generate the embedding
                            embedding = get_embedding(embedding_text)
                            
                            # Convert to a format suitable for storage
                            embedding_list = embedding.tolist()
                            
                            # Store in appropriate database
                            if is_postgres:
                                # Format for PostgreSQL vector type
                                embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
                                
                                # Insert into database - use custom encoder for JSON serialization
                                cur.execute(
                                    """
                                    INSERT INTO json_chunks (
                                        id, chunk_json, embedding, metadata,
                                        path, file_path
                                    )
                                    VALUES (%s, %s::jsonb, %s::vector, %s::jsonb, %s, %s)
                                    ON CONFLICT (id) DO UPDATE SET
                                        chunk_json = EXCLUDED.chunk_json,
                                        embedding = EXCLUDED.embedding,
                                        metadata = EXCLUDED.metadata,
                                        path = EXCLUDED.path,
                                        file_path = EXCLUDED.file_path
                                    """,
                                    (
                                        chunk_id,
                                        json.dumps(chunk, cls=CustomJSONEncoder),
                                        embedding_str,
                                        json.dumps(chunk_metadata, cls=CustomJSONEncoder),
                                        chunk.get("path", ""),
                                        file_path,
                                    ),
                                )
                            else:
                                # Format for ChromaDB
                                # Use custom encoder to handle Decimal types in both content and metadata
                                chroma_chunk = {
                                    "id": chunk_id,
                                    "content": json.loads(json.dumps(chunk, cls=CustomJSONEncoder)),
                                    "metadata": json.loads(json.dumps(chunk_metadata, cls=CustomJSONEncoder))
                                }
                                
                                chroma_chunks.append(chroma_chunk)
                                chroma_embeddings.append(embedding_list)
                            
                            # Add to processed chunks for return and relationship processing
                            # Use custom encoder to handle Decimal types by converting to standard JSON types
                            processed_chunk = {
                                "id": chunk_id,
                                "content": json.loads(json.dumps(chunk, cls=CustomJSONEncoder)),
                                "metadata": json.loads(json.dumps(chunk_metadata, cls=CustomJSONEncoder))
                            }
                            processed_chunks.append(processed_chunk)
                            
                        except Exception as e:
                            logger.error(f"Error processing chunk {chunk_id}: {e}")
                            continue
                
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue
        
        # For ChromaDB, batch insert all chunks with their embeddings
        if not is_postgres and chroma_chunks:
            from app.storage.chroma import upsert_chunks
            upsert_chunks(store, chroma_chunks, chroma_embeddings)
            logger.info(f"Added {len(chroma_chunks)} chunks to ChromaDB")
        
        # Commit PostgreSQL transaction if needed
        if is_postgres:
            store.commit()
            if cur:
                cur.close()
        
        logger.info(f"Successfully processed {len(processed_chunks)} chunks from JSON files in {directory}")
        return processed_chunks
        
    except Exception as e:
        logger.error(f"Unexpected error in process_json_files: {e}", exc_info=True)
        if is_postgres:
            store.rollback()
            if cur:
                cur.close()
        return [] 