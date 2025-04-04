"""
Advanced JSON Processing Module

This module provides enhanced JSON processing strategies for large and complex JSON files:
1. Hierarchical Chunking: Creates multi-level chunks including summaries and detailed data
2. Index-based Approach: Creates index structures for efficient retrieval without storing entire objects

These approaches are particularly useful for large files like adMetrics.json that exceed normal
processing capabilities.
"""

import os
import json
import ijson
import logging
import decimal
import statistics
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Set, Optional
from pathlib import Path
from tqdm import tqdm

# Import necessary functions from other modules
from app.processing.json_parser import generate_chunk_id
from app.retrieval.embedding import get_embedding

logger = logging.getLogger(__name__)

# Custom JSON encoder to handle Decimal types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, decimal.Decimal):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

def compute_statistics(values: List[Any]) -> Dict:
    """Compute statistical summaries for a list of numeric values."""
    if not values or not all(isinstance(v, (int, float, decimal.Decimal)) for v in values):
        return {"count": len(values)}
    
    try:
        numeric_values = [float(v) for v in values]
        stats = {
            "count": len(numeric_values),
            "min": min(numeric_values),
            "max": max(numeric_values),
            "mean": statistics.mean(numeric_values),
            "median": statistics.median(numeric_values)
        }
        # Add standard deviation if we have enough values
        if len(numeric_values) > 1:
            stats["std_dev"] = statistics.stdev(numeric_values)
        return stats
    except:
        return {"count": len(values)}

def extract_sample(items: List[Any], max_samples: int = 3) -> List[Any]:
    """Extract representative samples from a list of items."""
    if not items:
        return []
    if len(items) <= max_samples:
        return items
    
    # Take first, middle and last items for a representative sample
    sample_indices = [0, len(items) // 2, len(items) - 1]
    return [items[i] for i in sample_indices]

def generate_array_summary(array: List[Any], path: str) -> Dict:
    """Generate a summary of an array with statistics and samples."""
    if not array:
        return {"type": "array", "count": 0, "path": path, "samples": []}
    
    # Determine if array contains objects or primitive values
    if all(isinstance(item, dict) for item in array):
        # For arrays of objects, summarize structure
        keys_freq = defaultdict(int)
        for item in array:
            for key in item.keys():
                keys_freq[key] += 1
        
        # Find common keys (present in >50% of items)
        common_keys = {k for k, v in keys_freq.items() if v > len(array) * 0.5}
        
        samples = extract_sample(array)
        
        return {
            "type": "object_array",
            "count": len(array),
            "path": path,
            "common_fields": list(common_keys),
            "samples": samples
        }
    elif all(isinstance(item, (int, float, decimal.Decimal)) for item in array):
        # For numeric arrays, compute statistics
        stats = compute_statistics(array)
        samples = extract_sample(array)
        
        return {
            "type": "numeric_array",
            "path": path,
            "statistics": stats,
            "samples": samples
        }
    else:
        # For mixed arrays, just take samples
        samples = extract_sample(array)
        
        return {
            "type": "mixed_array",
            "count": len(array),
            "path": path,
            "samples": samples
        }

def generate_object_summary(obj: Dict, path: str) -> Dict:
    """Generate a summary of a complex object."""
    if not obj:
        return {"type": "empty_object", "path": path}
    
    summary = {
        "type": "object_summary",
        "path": path,
        "field_count": len(obj),
        "fields": {}
    }
    
    # Summarize each field
    for key, value in obj.items():
        field_path = f"{path}.{key}" if path else key
        
        if isinstance(value, dict):
            summary["fields"][key] = {
                "type": "object",
                "field_count": len(value),
                "path": field_path
            }
        elif isinstance(value, list):
            if value:
                if all(isinstance(item, dict) for item in value):
                    summary["fields"][key] = {
                        "type": "object_array",
                        "count": len(value),
                        "path": field_path
                    }
                else:
                    summary["fields"][key] = {
                        "type": "array",
                        "count": len(value),
                        "path": field_path,
                        "element_type": type(value[0]).__name__ if value else "unknown"
                    }
            else:
                summary["fields"][key] = {
                    "type": "empty_array",
                    "path": field_path
                }
        else:
            # For primitive values, store the actual value if short enough
            if isinstance(value, str) and len(value) > 100:
                val_summary = value[:100] + "..."
            else:
                val_summary = value
                
            summary["fields"][key] = {
                "type": type(value).__name__,
                "path": field_path,
                "value": val_summary
            }
    
    return summary

def create_hierarchical_chunks(data: Dict, file_path: str, base_path: str = "") -> Tuple[List[Dict], Dict]:
    """
    Create hierarchical chunks from JSON data with summaries and details.
    
    Args:
        data: The JSON data to chunk
        file_path: Path to the source file
        base_path: Base path for this chunk
        
    Returns:
        Tuple of (all_chunks, index_data) where:
            - all_chunks is a list of all created chunks
            - index_data is the index information for this data
    """
    all_chunks = []
    chunk_index = {}
    
    # Handle arrays specially
    if isinstance(data, list):
        # Create a summary chunk for the array
        array_summary = generate_array_summary(data, base_path)
        summary_id = generate_chunk_id(file_path, f"{base_path}_summary")
        
        summary_chunk = {
            "id": summary_id,
            "path": f"{base_path}_summary",
            "content": array_summary,
            "metadata": {
                "source_file": file_path,
                "chunk_type": "summary",
                "item_count": len(data),
                "original_path": base_path
            }
        }
        all_chunks.append(summary_chunk)
        
        # Add to index
        chunk_index[base_path] = {
            "type": "array",
            "summary_chunk_id": summary_id,
            "count": len(data),
            "item_paths": []
        }
        
        # Process individual items if it's an array of objects
        if all(isinstance(item, dict) for item in data) and data:
            # Process items in batches for large arrays
            batch_size = 50
            for batch_idx in range(0, len(data), batch_size):
                batch_items = data[batch_idx:batch_idx+batch_size]
                batch_path = f"{base_path}[{batch_idx}:{batch_idx+len(batch_items)-1}]"
                
                # Create a batch summary
                batch_summary = {
                    "type": "batch",
                    "count": len(batch_items),
                    "start_index": batch_idx,
                    "end_index": batch_idx + len(batch_items) - 1,
                    "path": batch_path,
                    "content_sample": extract_sample(batch_items, 1)
                }
                
                batch_id = generate_chunk_id(file_path, batch_path)
                batch_chunk = {
                    "id": batch_id,
                    "path": batch_path,
                    "content": batch_summary,
                    "metadata": {
                        "source_file": file_path,
                        "chunk_type": "batch",
                        "parent_path": base_path
                    }
                }
                all_chunks.append(batch_chunk)
                chunk_index[base_path]["item_paths"].append(batch_path)
                
                # Process each item in the batch
                for i, item in enumerate(batch_items):
                    item_idx = batch_idx + i
                    item_path = f"{base_path}[{item_idx}]"
                    
                    # Create detailed chunk for the item
                    item_id = generate_chunk_id(file_path, item_path)
                    item_chunk = {
                        "id": item_id,
                        "path": item_path,
                        "content": item,
                        "metadata": {
                            "source_file": file_path,
                            "chunk_type": "detail",
                            "parent_path": base_path,
                            "batch_path": batch_path,
                            "index": item_idx
                        }
                    }
                    all_chunks.append(item_chunk)
                    
                    # If the item is complex, also create summaries and hierarchies for it
                    if isinstance(item, dict) and len(item) > 5:
                        item_summary = generate_object_summary(item, item_path)
                        item_summary_id = generate_chunk_id(file_path, f"{item_path}_summary")
                        item_summary_chunk = {
                            "id": item_summary_id,
                            "path": f"{item_path}_summary",
                            "content": item_summary,
                            "metadata": {
                                "source_file": file_path,
                                "chunk_type": "item_summary",
                                "parent_path": item_path
                            }
                        }
                        all_chunks.append(item_summary_chunk)
                        
                        # Process nested structures in complex items
                        for key, value in item.items():
                            if isinstance(value, (dict, list)) and value:
                                nested_path = f"{item_path}.{key}"
                                nested_chunks, nested_index = create_hierarchical_chunks(
                                    value, file_path, nested_path
                                )
                                all_chunks.extend(nested_chunks)
                                chunk_index.update(nested_index)
    
    # Handle objects
    elif isinstance(data, dict):
        # Create a summary chunk for the object
        obj_summary = generate_object_summary(data, base_path)
        summary_id = generate_chunk_id(file_path, f"{base_path}_summary")
        
        summary_chunk = {
            "id": summary_id,
            "path": f"{base_path}_summary",
            "content": obj_summary,
            "metadata": {
                "source_file": file_path,
                "chunk_type": "summary",
                "field_count": len(data),
                "original_path": base_path
            }
        }
        all_chunks.append(summary_chunk)
        
        # Add to index
        chunk_index[base_path] = {
            "type": "object",
            "summary_chunk_id": summary_id,
            "field_count": len(data),
            "fields": {}
        }
        
        # Create a detail chunk for the full object
        detail_id = generate_chunk_id(file_path, base_path)
        detail_chunk = {
            "id": detail_id,
            "path": base_path,
            "content": data,
            "metadata": {
                "source_file": file_path,
                "chunk_type": "detail",
                "summary_id": summary_id
            }
        }
        all_chunks.append(detail_chunk)
        chunk_index[base_path]["detail_chunk_id"] = detail_id
        
        # Process nested structures recursively
        for key, value in data.items():
            field_path = f"{base_path}.{key}" if base_path else key
            
            # Only create hierarchies for complex structures
            if isinstance(value, (dict, list)) and value:
                nested_chunks, nested_index = create_hierarchical_chunks(
                    value, file_path, field_path
                )
                all_chunks.extend(nested_chunks)
                chunk_index.update(nested_index)
                chunk_index[base_path]["fields"][key] = field_path
    
    return all_chunks, chunk_index

def build_json_index(file_path: str, sample_rate: float = 0.1) -> Dict:
    """
    Build an index of a JSON file without loading the entire file.
    
    Args:
        file_path: Path to the JSON file
        sample_rate: Percentage of array items to sample for the index
        
    Returns:
        An index structure mapping paths to metadata about the JSON content
    """
    index = {
        "file_path": file_path,
        "paths": {},
        "structure": {},
        "root_type": None
    }
    
    try:
        # Try to determine root structure
        with open(file_path, 'rb') as f:
            # Read the first chunk to determine if root is object or array
            prefix = f.read(10).decode('utf-8').strip()
            if prefix.startswith('['):
                index["root_type"] = "array"
            elif prefix.startswith('{'):
                index["root_type"] = "object"
            else:
                logger.error(f"Unknown root type in {file_path}")
                return index
            
        # Reset file position
        with open(file_path, 'rb') as f:
            # If root is an array, sample array items
            if index["root_type"] == "array":
                try:
                    # First count approximate items (fast pass)
                    item_count = 0
                    for _ in ijson.items(f, 'item'):
                        item_count += 1
                        # Stop counting after a reasonable number to estimate
                        if item_count > 1000:
                            item_count = -1  # Indicate we didn't finish counting
                            break
                    
                    index["structure"]["count"] = item_count
                    index["structure"]["sample_count"] = 0
                    index["structure"]["samples"] = []
                    
                    # Reset file position
                    f.seek(0)
                    
                    # Sample items based on sample_rate
                    sample_threshold = int(1 / sample_rate)
                    sample_count = 0
                    
                    for i, item in enumerate(ijson.items(f, 'item')):
                        # Take samples at regular intervals
                        if i % sample_threshold == 0:
                            item_path = f"[{i}]"
                            index["paths"][item_path] = {
                                "index": i,
                                "type": type(item).__name__
                            }
                            
                            # Add field information for objects
                            if isinstance(item, dict):
                                index["paths"][item_path]["fields"] = list(item.keys())
                                
                                # Record field types
                                for k, v in item.items():
                                    field_path = f"{item_path}.{k}"
                                    index["paths"][field_path] = {
                                        "parent": item_path,
                                        "type": type(v).__name__
                                    }
                            
                            # Add to samples
                            index["structure"]["samples"].append({
                                "path": item_path,
                                "preview": str(item)[:100] + "..." if len(str(item)) > 100 else str(item)
                            })
                            sample_count += 1
                            
                            # Limit total samples
                            if sample_count >= 10:
                                break
                    
                    index["structure"]["sample_count"] = sample_count
                    
                except Exception as e:
                    logger.error(f"Error sampling array in {file_path}: {e}")
            
            # If root is an object, extract structure using prefix-based parsing
            elif index["root_type"] == "object":
                # Reset file position
                f.seek(0)
                
                # Collect top-level keys and their types
                for prefix, event, value in ijson.parse(f):
                    if event == 'map_key':
                        key_path = value
                        index["paths"][key_path] = {"type": "unknown"}
                    
                    # Only go one level deep to keep it fast
                    elif '.' in prefix and prefix.count('.') == 1 and event in ('string', 'number', 'boolean'):
                        path = prefix
                        index["paths"][path] = {"type": event}
                    
                    # For nested objects and arrays, just note their existence 
                    elif '.' in prefix and prefix.count('.') == 1 and event in ('start_map', 'start_array'):
                        path = prefix
                        index["paths"][path] = {"type": "object" if event == 'start_map' else "array"}
                    
    except Exception as e:
        logger.error(f"Error building index for {file_path}: {e}")
        
    return index

def process_json_file_with_index(
    file_path: str, 
    store: Any,
    index: Optional[Dict] = None
) -> Tuple[List[Dict], Dict]:
    """
    Process a single JSON file using index-based and hierarchical approaches.
    
    Args:
        file_path: Path to the JSON file
        store: Database backend (PostgreSQL or ChromaDB)
        index: Optional pre-built index. If None, one will be created.
        
    Returns:
        Tuple of (processed_chunks, file_index) where:
            - processed_chunks is a list of all chunks stored in the database
            - file_index is the index structure for the file
    """
    logger.info(f"Processing file with advanced techniques: {file_path}")
    
    # Determine if we're using PostgreSQL or ChromaDB
    is_postgres = hasattr(store, 'cursor')
    
    # Build index if not provided
    if index is None:
        logger.info(f"Building index for {file_path}")
        index = build_json_index(file_path)
    
    # Initialize cursor for PostgreSQL if needed
    cur = None
    if is_postgres:
        cur = store.cursor()
    
    # Store the index itself as a special chunk
    index_id = generate_chunk_id(file_path, "_index")
    index_chunk = {
        "id": index_id,
        "content": index,
        "metadata": {
            "source_file": file_path,
            "chunk_type": "index"
        }
    }
    
    # Prepare for processing
    processed_chunks = [index_chunk]
    chroma_chunks = []
    chroma_embeddings = []
    
    # Full file processing depends on root type
    root_chunks = []
    if index["root_type"] == "array":
        # Process the file as a large array
        with open(file_path, 'r') as f:
            try:
                # Load the first few items for creating summaries
                data = json.load(f)
                if isinstance(data, list):
                    # Create hierarchical chunks for batches of the array
                    # For very large arrays, we'll just summarize and sample
                    array_sample = data[:100] if len(data) > 100 else data
                    chunks, chunk_index = create_hierarchical_chunks(
                        array_sample, file_path, ""
                    )
                    root_chunks.extend(chunks)
                    
                    # If the array is too large, add a note in the index
                    if len(data) > 100:
                        index["structure"]["note"] = f"Array too large, only first 100 items processed in detail. Total items: {len(data)}"
                else:
                    logger.error(f"Expected array, found {type(data).__name__} in {file_path}")
            except json.JSONDecodeError:
                # If the file is too large to load fully, use streaming
                logger.info(f"File too large for full loading, using streaming: {file_path}")
                
                # Create a summary chunk based on the index
                summary = {
                    "type": "large_array",
                    "path": "",
                    "count_estimate": index["structure"].get("count", "unknown"),
                    "samples": index["structure"].get("samples", [])
                }
                
                summary_id = generate_chunk_id(file_path, "_summary")
                summary_chunk = {
                    "id": summary_id,
                    "path": "_summary",
                    "content": summary,
                    "metadata": {
                        "source_file": file_path,
                        "chunk_type": "summary",
                        "note": "Large array processed via streaming"
                    }
                }
                root_chunks.append(summary_chunk)
                
                # Process batches using streaming
                try:
                    with open(file_path, 'rb') as f:
                        batch_size = 50
                        batch_items = []
                        batch_idx = 0
                        
                        for i, item in enumerate(ijson.items(f, 'item')):
                            batch_items.append(item)
                            
                            if len(batch_items) >= batch_size:
                                # Process this batch
                                batch_path = f"[{batch_idx}:{batch_idx+len(batch_items)-1}]"
                                batch_chunks, batch_index = create_hierarchical_chunks(
                                    batch_items, file_path, batch_path
                                )
                                root_chunks.extend(batch_chunks)
                                
                                # Update counters and reset batch
                                batch_idx += len(batch_items)
                                batch_items = []
                                
                                # Limit total number of chunks for very large files
                                if len(root_chunks) > 1000:
                                    logger.warning(f"Reached chunk limit for {file_path}, stopping processing")
                                    break
                        
                        # Process final batch if any items remain
                        if batch_items:
                            batch_path = f"[{batch_idx}:{batch_idx+len(batch_items)-1}]"
                            batch_chunks, batch_index = create_hierarchical_chunks(
                                batch_items, file_path, batch_path
                            )
                            root_chunks.extend(batch_chunks)
                except Exception as e:
                    logger.error(f"Error streaming array in {file_path}: {e}")
                    
    elif index["root_type"] == "object":
        # Process the file as an object
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                chunks, chunk_index = create_hierarchical_chunks(
                    data, file_path, ""
                )
                root_chunks.extend(chunks)
        except json.JSONDecodeError:
            # If the file is too large to load fully, create a summary based on the index
            logger.info(f"Object too large for full loading, using index-based summary: {file_path}")
            
            summary = {
                "type": "large_object",
                "path": "",
                "fields": list(index["paths"].keys()),
                "structure": index["structure"]
            }
            
            summary_id = generate_chunk_id(file_path, "_summary")
            summary_chunk = {
                "id": summary_id,
                "path": "_summary",
                "content": summary,
                "metadata": {
                    "source_file": file_path,
                    "chunk_type": "summary",
                    "note": "Large object processed via indexing"
                }
            }
            root_chunks.append(summary_chunk)
    
    # Generate embeddings and store chunks
    for chunk in root_chunks:
        try:
            # Generate embedding for the chunk
            embedding_text = json.dumps(chunk["content"], cls=CustomJSONEncoder)
            embedding = get_embedding(embedding_text)
            embedding_list = embedding.tolist()
            
            # Store in the appropriate database
            if is_postgres:
                # Format for PostgreSQL
                embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
                
                # Insert into database
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
                        chunk["id"],
                        json.dumps(chunk["content"], cls=CustomJSONEncoder),
                        embedding_str,
                        json.dumps(chunk.get("metadata", {}), cls=CustomJSONEncoder),
                        chunk.get("path", ""),
                        file_path,
                    ),
                )
            else:
                # Format for ChromaDB
                chroma_chunk = {
                    "id": chunk["id"],
                    "content": json.loads(json.dumps(chunk["content"], cls=CustomJSONEncoder)),
                    "metadata": json.loads(json.dumps(chunk.get("metadata", {}), cls=CustomJSONEncoder))
                }
                
                chroma_chunks.append(chroma_chunk)
                chroma_embeddings.append(embedding_list)
            
            # Add to processed chunks
            processed_chunks.append(chunk)
            
        except Exception as e:
            logger.error(f"Error processing chunk {chunk.get('id')}: {e}")
    
    # For ChromaDB, batch insert all chunks
    if not is_postgres and chroma_chunks:
        from app.storage.chroma import upsert_chunks
        upsert_chunks(store, chroma_chunks, chroma_embeddings)
        logger.info(f"Added {len(chroma_chunks)} chunks to ChromaDB")
    
    # Commit PostgreSQL transaction
    if is_postgres and cur:
        store.commit()
        cur.close()
    
    logger.info(f"Successfully processed {len(processed_chunks)} chunks from {file_path}")
    return processed_chunks, index

def process_json_files_advanced(directory: str, store: Any) -> List[Dict]:
    """
    Process all JSON files in a directory using advanced techniques.
    
    Args:
        directory: Path to directory containing JSON files
        store: Database connection (PostgreSQL or ChromaDB)
        
    Returns:
        List of processed chunks
    """
    logger.info(f"Starting advanced JSON processing in directory: {directory}")
    
    # Determine if we're using PostgreSQL or ChromaDB
    is_postgres = hasattr(store, 'cursor')
    
    all_processed_chunks = []
    file_indices = {}
    
    try:
        # Allow input to be either a directory or a list of specific files
        files_to_process = []
        
        if isinstance(directory, list):
            # If directory is a list, assume it's a list of file paths
            files_to_process = directory
        else:
            # Otherwise walk the directory to find JSON files
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.endswith(".json"):
                        files_to_process.append(os.path.join(root, file))
        
        # Process each file
        for file_path in tqdm(files_to_process, desc="Processing JSON files"):
            # Skip non-JSON files
            if not file_path.endswith(".json"):
                continue
            
            # Check if this is a large file that needs advanced processing
            use_advanced = (os.path.basename(file_path) == "adMetrics.json" or 
                           os.path.getsize(file_path) > 10_000_000)  # 10MB
            
            if use_advanced:
                logger.info(f"Using advanced processing for large file: {file_path}")
                processed_chunks, file_index = process_json_file_with_index(file_path, store)
                all_processed_chunks.extend(processed_chunks)
                file_indices[file_path] = file_index
            else:
                # For smaller files, process directly without using json_processor's function
                # This avoids the circular dependency
                logger.info(f"Processing smaller file directly: {file_path}")
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # Process the file using hierarchical chunking
                    chunks, _ = create_hierarchical_chunks(data, file_path, "")
                    
                    # Store chunks in the database
                    for chunk in chunks:
                        # Generate embedding
                        embedding_text = json.dumps(chunk["content"], cls=CustomJSONEncoder)
                        embedding = get_embedding(embedding_text)
                        embedding_list = embedding.tolist()
                        
                        if is_postgres:
                            # Store in PostgreSQL
                            cur = store.cursor()
                            embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
                            
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
                                    chunk["id"],
                                    json.dumps(chunk["content"], cls=CustomJSONEncoder),
                                    embedding_str,
                                    json.dumps(chunk.get("metadata", {}), cls=CustomJSONEncoder),
                                    chunk.get("path", ""),
                                    file_path,
                                ),
                            )
                            cur.close()
                        else:
                            # Store in ChromaDB
                            from app.storage.chroma import upsert_chunks
                            chroma_chunk = {
                                "id": chunk["id"],
                                "content": json.loads(json.dumps(chunk["content"], cls=CustomJSONEncoder)),
                                "metadata": json.loads(json.dumps(chunk.get("metadata", {}), cls=CustomJSONEncoder))
                            }
                            upsert_chunks(store, [chroma_chunk], [embedding_list])
                        
                        all_processed_chunks.append(chunk)
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
        
        if is_postgres:
            store.commit()
        
        logger.info(f"Successfully processed {len(all_processed_chunks)} chunks from JSON files")
        return all_processed_chunks
        
    except Exception as e:
        logger.error(f"Unexpected error in advanced JSON processing: {e}", exc_info=True)
        if is_postgres:
            store.rollback()
        return [] 