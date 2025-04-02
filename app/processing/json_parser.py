"""
JSON Parser Module

This module handles the parsing and chunking of JSON documents for RAG processing.
It includes utilities for generating unique chunk IDs, normalizing JSON paths,
and converting JSON structures into processable chunks while preserving hierarchy
and relationships.
"""

from typing import List, Dict, Tuple
import hashlib
import json
import orjson
import ijson
from pathlib import Path
from app.utils.logging_config import get_logger
from collections import defaultdict
import os

logger = get_logger(__name__)


def generate_chunk_id(source_file: str, path: str) -> str:
    """
    Generate a unique chunk ID based on source file and path.

    Args:
        source_file (str): Source file path
        path (str): JSON path within the file

    Returns:
        str: MD5 hash of the combined string

    Note:
        The function normalizes paths to ensure consistent IDs across
        different operating systems and path representations.
    """
    logger.debug(
        f"DEBUG: generate_chunk_id input - source_file: {source_file}, path: {path}"
    )

    # Clean up the source file path - remove duplicates and normalize
    source_file = str(Path(source_file).resolve())
    if source_file.startswith("/"):
        source_file = source_file[1:]  # Remove leading slash

    # Clean up the path
    path = path.strip().strip(".")

    # Remove any duplicate file paths that might have been added to the path
    if source_file in path:
        logger.debug(f"DEBUG: Found duplicate file path in path, removing: {path}")
        path = path.replace(source_file + ":", "")
        path = path.replace(source_file + "/", "")
        logger.debug(f"DEBUG: After removing duplicate: {path}")

    # Create a unique string combining file and path
    unique_str = f"{source_file}:{path}"
    logger.debug(f"DEBUG: Final unique string: {unique_str}")

    # Generate MD5 hash
    chunk_id = hashlib.md5(unique_str.encode()).hexdigest()
    logger.debug(f"DEBUG: Generated chunk ID: {chunk_id}")

    return chunk_id


def normalize_json_path(path: str) -> str:
    """
    Normalize a JSON path to ensure consistent formatting.

    Args:
        path (str): Raw JSON path string

    Returns:
        str: Normalized path string with consistent separators and formatting

    Example:
        >>> normalize_json_path("root[0].items.name")
        "root.0.items.name"
    """
    # Remove any leading/trailing dots or spaces
    path = path.strip().strip(".")

    # Replace array notation with dot notation
    while "[" in path and "]" in path:
        start = path.find("[")
        end = path.find("]", start)
        if start != -1 and end != -1:
            array_index = path[start + 1 : end]
            path = path[:start] + "." + array_index + path[end + 1 :]

    # Clean up any double dots
    while ".." in path:
        path = path.replace("..", ".")

    return path


def is_entity_like(obj: Dict) -> bool:
    """
    Check if an object represents a coherent unit that should be kept as a single chunk.

    The goal is to identify objects that form a logical unit based on their structure:
    - Objects with a small number of direct fields (2-5) are likely coherent units
    - Objects with too many fields or that are too nested are likely too big
    - Objects with too few fields might be too granular
    """
    if not isinstance(obj, dict):
        logger.debug(f"Object is not a dict: {type(obj)}")
        return False

    # Count direct fields and check their types
    field_count = len(obj)
    nested_count = sum(1 for v in obj.values() if isinstance(v, (dict, list)))
    has_nested = nested_count > 0

    # Estimate total content size
    content_size = sum(
        len(str(v)) for v in obj.values() if not isinstance(v, (dict, list))
    )

    logger.debug(
        f"Checking object with {field_count} fields, nested={nested_count}, size={content_size}"
    )

    # Allow slightly more fields and nesting
    max_fields = 7 # Increased from 5
    max_nested = 3 # Increased from 2
    min_fields = 2 # Keep at 2
    max_content = 1200 # Increased from 800

    # Too few fields might be too granular
    if field_count < min_fields:
        logger.debug(f"Too few fields (<{min_fields}) to be a coherent unit")
        return False

    # Too many direct fields suggests this might be too large
    if field_count > max_fields:
        logger.debug(f"Too many fields (>{max_fields}) to be a coherent unit")
        return False

    # Too many nested structures make this too complex
    if nested_count > max_nested:
        logger.debug(f"Too many nested structures (>{max_nested})")
        return False

    # Content size too large
    if content_size > max_content:
        logger.debug(f"Content size too large (>{max_content})")
        return False

    # If it has nested structures but also fits within field limits,
    # it might be a good unit
    if has_nested and min_fields <= field_count <= max_fields:
        logger.debug(f"Good size ({min_fields}-{max_fields} fields) with some nested structure (<= {max_nested})")
        return True

    # If it has no nested structures but fits within field limits,
    # it's likely a good unit
    if not has_nested and min_fields <= field_count <= max_fields:
        logger.debug(f"Good size ({min_fields}-{max_fields} fields) with flat structure")
        return True

    logger.debug("Does not meet criteria for a coherent unit")
    return False


def json_to_path_chunks(
    json_obj: Dict,
    file_path: str = "",
    max_chunks: int = 10000,
    entities: Dict = None,
    archetypes: List[Tuple[str, float]] = None,
    chunk_strategy: str = "hybrid",
    base_path: str = "",
) -> List[Dict]:
    """
    Convert a JSON object into path-based chunks, with a configurable strategy.

    Args:
        json_obj (dict): The JSON object to chunk
        file_path (str): The file path for logging
        max_chunks (int): Max number of chunks to produce
        entities (dict): (Optional) Additional entity detection config
        archetypes (List[Tuple[str, float]]): (Optional) Archetype info
        chunk_strategy (str): 'full' (old behavior) or 'hybrid' (less granular)
        base_path (str): Optional base path to prefix generated chunk paths
    """
    chunks = []
    logger.debug(
        f"=== Starting JSON chunking for file: {file_path} with strategy={chunk_strategy} ==="
    )

    def process_value(obj, path="", parent_path=None, context=None):
        if context is None:
            context = {}

        # Construct the full path including the base path
        full_path = f"{base_path}{path}" if base_path else path
        full_parent_path = f"{base_path}{parent_path}" if base_path and parent_path else parent_path
        full_parent_path = full_parent_path if full_parent_path is not None else base_path # Handle root case

        if len(chunks) >= max_chunks:
            if len(chunks) == max_chunks: # Log only once
                 logger.warning(f"Reached max chunks limit ({max_chunks}) for {file_path} (base: {base_path})")
            return

        if isinstance(obj, dict):
            # logger.debug(f"=== Checking if object at path '{full_path}' is entity-like ===")
            # logger.debug(f"Object keys: {list(obj.keys())}")

            if chunk_strategy == "hybrid" and is_entity_like(obj):
                # logger.debug(f"=== FOUND ENTITY-LIKE OBJECT at path: {full_path} ===")
                chunk = {
                    "path": full_path,
                    "value": obj,
                    "context": context,
                    "parent_path": full_parent_path,
                    "metadata": {
                        "has_children": any(
                            isinstance(v, (dict, list)) for v in obj.values()
                        ),
                        "chunked_as_entity": True,
                        "field_count": len(obj),
                        "file_path": file_path # Add file path to metadata
                    },
                }
                if entities:
                    chunk["entities"] = entities
                if archetypes:
                    chunk["archetypes"] = archetypes
                chunks.append(chunk)
                return

            # logger.debug(
            #     f"=== Object at path '{full_path}' was NOT entity-like, processing fields individually ==="
            # )

            has_any_children = False
            for k, v in obj.items():
                child_path = f"{path}.{k}" if path else k
                new_context = {**context}
                for ck, cv in obj.items():
                    if not isinstance(cv, (dict, list)):
                         # Prefix context keys as well
                        context_key_prefix = f"{path}_" if path else ""
                        new_context[f"{context_key_prefix}{ck}"] = cv
                process_value(v, child_path, path, new_context)
                has_any_children = True

            if not has_any_children:
                chunk = {
                    "path": full_path,
                    "value": obj, # Empty dict
                    "context": context,
                    "parent_path": full_parent_path,
                    "metadata": {"has_children": False, "file_path": file_path},
                }
                if entities:
                    chunk["entities"] = entities
                if archetypes:
                    chunk["archetypes"] = archetypes
                chunks.append(chunk)

        elif isinstance(obj, list):
            # logger.debug(f"=== Processing list at path '{full_path}' ===")
            if obj and all(isinstance(item, dict) for item in obj):
                # logger.debug(
                #     "=== List contains all dict items, checking if they are entity-like ==="
                # )
                if any(is_entity_like(item) for item in obj):
                    # logger.debug(
                    #     f"=== FOUND ARRAY OF ENTITY-LIKE OBJECTS at path: {full_path} ==="
                    # )
                    for i, item in enumerate(obj):
                        item_path = f"{path}[{i}]"
                        full_item_path = f"{base_path}{item_path}"
                        chunk = {
                            "path": full_item_path,
                            "value": item,
                            "context": context,
                            "parent_path": full_path, # The list path is the parent
                            "metadata": {
                                "is_collection_item": True,
                                "collection_path": full_path,
                                "item_index": i,
                                "chunked_as_entity": True,
                                "file_path": file_path
                            },
                        }
                        if entities:
                            chunk["entities"] = entities
                        if archetypes:
                            chunk["archetypes"] = archetypes
                        chunks.append(chunk)
                    return

                # logger.debug(
                #     "=== List items were NOT entity-like, processing individually ==="
                # )

            for i, item in enumerate(obj):
                item_path = f"{path}[{i}]"
                process_value(item, item_path, path, context)
        else:
            chunk = {
                "path": full_path,
                "value": obj,
                "context": context,
                "parent_path": full_parent_path,
                "metadata": {"is_primitive": True, "value_type": type(obj).__name__, "file_path": file_path},
            }
            if entities:
                chunk["entities"] = entities
            if archetypes:
                chunk["archetypes"] = archetypes
            chunks.append(chunk)

    process_value(json_obj, path=base_path) # Start with base_path if provided
    logger.debug(f"Generated {len(chunks)} chunks for {file_path} (base: {base_path})")

    # Log chunk statistics
    path_depths = [len(c["path"].split(".")) for c in chunks]
    if path_depths:
        logger.debug(
            f"Path depth stats: min={min(path_depths)}, max={max(path_depths)}, avg={sum(path_depths) / len(path_depths):.1f}"
        )

    value_types = defaultdict(int)
    for c in chunks:
        value_types[type(c["value"]).__name__] += 1
    logger.debug(f"Value type distribution: {dict(value_types)}")

    return chunks # Return all chunks, limit applied within recursion


def process_json_files(directory: str) -> List[Dict]:
    """Process all JSON files in directory, streaming items from the 'metrics' array."""
    all_chunks = []
    logger.info(f"Starting JSON processing in directory: {directory}")

    # Walk through directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Skip non-JSON files or specific files
            if not file.endswith(".json") or file == "test_queries":
                continue

            file_path = Path(root) / file
            logger.info(f"Processing file: {file_path}")

            try:
                # Assume the structure is { "metrics": [...] } and stream items from the array
                logger.info(f"Attempting to stream items from 'metrics' array in {file_path}")
                processed_items = 0
                with open(file_path, 'rb') as f:
                    # Use ijson.items to iterate over elements in the 'metrics' array
                    items = ijson.items(f, 'metrics.item') # Target the array under the 'metrics' key
                    for i, item in enumerate(items):
                        item_base_path = f'metrics[{i}]' # Construct base path like metrics[0], metrics[1]
                        try:
                            item_chunks = json_to_path_chunks(
                                item,
                                file_path=str(file_path),
                                base_path=item_base_path
                            )
                            all_chunks.extend(item_chunks)
                            processed_items += 1
                        except Exception as item_e: # Catch errors during chunking of a single item
                            logger.error(f"Error chunking item {i} (path: {item_base_path}) in {file_path}: {item_e}")
                            # Optionally continue to the next item
                            continue
                logger.info(f"Finished streaming {processed_items} items from {file_path}")

            except ijson.JSONError as e:
                # Handle cases where the structure might not be { "metrics": [...] } 
                # or other ijson parsing errors.
                # Log the specific ijson error message
                logger.error(f"ijson.JSONError processing {file_path}: {e}. Falling back to full load.") 
                logger.info(f"Falling back to full object load for {file_path}")
                try:
                    with open(file_path, 'rb') as f_obj:
                        data = orjson.loads(f_obj.read())
                    # Call json_to_path_chunks for the entire object
                    object_chunks = json_to_path_chunks(data, file_path=str(file_path), base_path="") 
                    all_chunks.extend(object_chunks)
                    logger.info(f"Finished fallback full object load for {file_path}")
                except Exception as fallback_e:
                     logger.error(f"Error during fallback load for {file_path}: {fallback_e}")
            except FileNotFoundError:
                logger.error(f"File not found: {file_path}")
            except Exception as e:
                logger.error(f"An unexpected error occurred processing {file_path}: {e}", exc_info=True)

    logger.info(f"Finished processing all files. Total chunks generated: {len(all_chunks)}")
    return all_chunks


def get_chunk_hierarchy_info(chunk: Dict) -> Dict:
    """
    Extract hierarchical information from a chunk.

    Args:
        chunk (Dict): The chunk to analyze

    Returns:
        Dict: Hierarchy information including:
            - depth: Nesting level in the JSON structure
            - parent_path: Path to parent node
            - child_paths: List of immediate child paths
            - siblings: List of sibling paths

    Note:
        This information is used for context assembly and
        relationship tracking in the RAG system.
    """
    return {
        "id": chunk["id"],
        "parent_id": chunk.get("parent_id"),
        "depth": chunk["depth"],
        "path": chunk["path"],
        "has_children": chunk["metadata"]["has_children"],
    }


def combine_chunk_with_context(
    chunk: Dict, parents: List[Dict], children: List[Dict]
) -> str:
    """
    Combine a chunk with its contextual information.

    Creates a text representation of a chunk that includes relevant
    context from parent and child nodes for better semantic understanding.

    Args:
        chunk (Dict): The main chunk to process
        parents (List[Dict]): List of parent chunks
        children (List[Dict]): List of child chunks

    Returns:
        str: A formatted string containing the chunk content with
             hierarchical context suitable for embedding generation

    Note:
        The context assembly is optimized for:
        - Semantic completeness
        - Relationship preservation
        - Embedding model token limits
    """
    context_parts = []

    # Add parent context
    for parent in parents:
        context_parts.append(
            f"Parent ({parent['path']}): {json.dumps(parent['content'])}"
        )

    # Add main chunk
    context_parts.append(f"Current: {json.dumps(chunk['content'])}")

    # Add child context
    for child in children:
        context_parts.append(f"Child ({child['path']}): {json.dumps(child['content'])}")

    return "\n".join(context_parts)
