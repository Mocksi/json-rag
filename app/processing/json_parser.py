"""
JSON Parser Module

This module handles the parsing and chunking of JSON documents for RAG processing.
It includes utilities for generating unique chunk IDs, normalizing JSON paths,
and converting JSON structures into processable chunks while preserving hierarchy
and relationships.
"""

from typing import List, Dict, Any, Optional, Tuple
import hashlib
import json
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
    logger.debug(f"DEBUG: generate_chunk_id input - source_file: {source_file}, path: {path}")
    
    # Clean up the source file path - remove duplicates and normalize
    source_file = str(Path(source_file).resolve())
    if source_file.startswith('/'):
        source_file = source_file[1:]  # Remove leading slash
    
    # Clean up the path
    path = path.strip().strip('.')
    
    # Remove any duplicate file paths that might have been added to the path
    if source_file in path:
        logger.debug(f"DEBUG: Found duplicate file path in path, removing: {path}")
        path = path.replace(source_file + ':', '')
        path = path.replace(source_file + '/', '')
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
    path = path.strip().strip('.')
    
    # Replace array notation with dot notation
    while '[' in path and ']' in path:
        start = path.find('[')
        end = path.find(']', start)
        if start != -1 and end != -1:
            array_index = path[start+1:end]
            path = path[:start] + '.' + array_index + path[end+1:]
    
    # Clean up any double dots
    while '..' in path:
        path = path.replace('..', '.')
    
    return path

def is_entity_like(obj: Dict) -> bool:
    """
    Check if an object represents a coherent unit that should be kept as a single chunk.
    
    The goal is to identify objects that form a logical unit based on their structure:
    - Objects with a small number of direct fields (3-7) are likely coherent units
    - Objects with too many fields or that are too nested are likely too big
    - Objects with too few fields might be too granular
    """
    if not isinstance(obj, dict):
        logger.debug(f"Object is not a dict: {type(obj)}")
        return False
        
    # Count direct fields and check their types
    field_count = len(obj)
    has_nested = any(isinstance(v, (dict, list)) for v in obj.values())
    
    logger.debug(f"Checking object with {field_count} fields, has_nested={has_nested}")
    
    # Too few fields might be too granular
    if field_count < 3:
        logger.debug("Too few fields to be a coherent unit")
        return False
        
    # Too many direct fields suggests this might be too large
    if field_count > 7:
        logger.debug("Too many fields to be a coherent unit")
        return False
    
    # If it has nested structures but also has 3-7 direct fields, 
    # it might be a good unit
    if has_nested and 3 <= field_count <= 7:
        logger.debug("Good size with some nested structure")
        return True
        
    # If it has no nested structures but has 3-7 fields,
    # it's likely a good unit
    if not has_nested and 3 <= field_count <= 7:
        logger.debug("Good size with flat structure")
        return True
    
    logger.debug("Does not meet criteria for a coherent unit")
    return False

def json_to_path_chunks(
    json_obj: Dict,
    file_path: str = '',
    max_chunks: int = 100,
    entities: Dict = None,
    archetypes: List[Tuple[str, float]] = None,
    chunk_strategy: str = 'hybrid'
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
    """
    chunks = []
    logger.debug(f"=== Starting JSON chunking for file: {file_path} with strategy={chunk_strategy} ===")
    
    def process_value(obj, path='', parent_path=None, context=None):
        if context is None:
            context = {}
            
        if len(chunks) >= max_chunks:
            logger.warning(f"Reached max chunks limit ({max_chunks})")
            return
            
        if isinstance(obj, dict):
            # Check if this is an entity-like object we should keep together
            logger.debug(f"=== Checking if object at path '{path}' is entity-like ===")
            logger.debug(f"Object keys: {list(obj.keys())}")
            
            if chunk_strategy == 'hybrid' and is_entity_like(obj):
                logger.debug(f"=== FOUND ENTITY-LIKE OBJECT at path: {path} ===")
                chunk = {
                    'path': path,
                    'value': obj,
                    'context': context,
                    'parent_path': parent_path,
                    'metadata': {
                        'has_children': any(isinstance(v, (dict, list)) for v in obj.values()),
                        'chunked_as_entity': True,
                        'field_count': len(obj)
                    }
                }
                if entities:
                    chunk['entities'] = entities
                if archetypes:
                    chunk['archetypes'] = archetypes
                chunks.append(chunk)
                return  # Stop recursion - entire object is one chunk
                
            logger.debug(f"=== Object at path '{path}' was NOT entity-like, processing fields individually ===")
            
            # Process as normal dictionary if not entity-like or using full strategy
            has_any_children = False
            for k, v in obj.items():
                child_path = f"{path}.{k}" if path else k
                new_context = {**context}
                # Add current object's fields to context
                for ck, cv in obj.items():
                    if not isinstance(cv, (dict, list)):
                        new_context[f"{path}_{ck}" if path else ck] = cv
                process_value(v, child_path, path, new_context)
                has_any_children = True
                
            # Create chunk for empty dict
            if not has_any_children:
                chunk = {
                    'path': path,
                    'value': obj,
                    'context': context,
                    'parent_path': parent_path,
                    'metadata': {'has_children': False}
                }
                if entities:
                    chunk['entities'] = entities
                if archetypes:
                    chunk['archetypes'] = archetypes
                chunks.append(chunk)
                
        elif isinstance(obj, list):
            logger.debug(f"=== Processing list at path '{path}' ===")
            # Handle arrays of objects - check each object in the array
            if obj and all(isinstance(item, dict) for item in obj):
                logger.debug(f"=== List contains all dict items, checking if they are entity-like ===")
                # If any object in the array is entity-like, treat each object as a chunk
                if any(is_entity_like(item) for item in obj):
                    logger.debug(f"=== FOUND ARRAY OF ENTITY-LIKE OBJECTS at path: {path} ===")
                    for i, item in enumerate(obj):
                        chunk = {
                            'path': f"{path}[{i}]",
                            'value': item,
                            'context': context,
                            'parent_path': parent_path,
                            'metadata': {
                                'is_collection_item': True,
                                'collection_path': path,
                                'item_index': i,
                                'chunked_as_entity': True
                            }
                        }
                        if entities:
                            chunk['entities'] = entities
                        if archetypes:
                            chunk['archetypes'] = archetypes
                        chunks.append(chunk)
                    return  # Stop recursion - each object is its own chunk
                
                logger.debug(f"=== List items were NOT entity-like, processing individually ===")
            
            # Process list items individually if not entity-like objects
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                process_value(item, new_path, path, context)
        else:
            # Only create chunks for primitive values if they're not part of an entity
            chunk = {
                'path': path,
                'value': obj,
                'context': context,
                'parent_path': parent_path,
                'metadata': {
                    'is_primitive': True,
                    'value_type': type(obj).__name__
                }
            }
            if entities:
                chunk['entities'] = entities
            if archetypes:
                chunk['archetypes'] = archetypes
            chunks.append(chunk)
            
    process_value(json_obj)
    logger.debug(f"Generated {len(chunks)} chunks")
    
    # Log chunk statistics
    path_depths = [len(c['path'].split('.')) for c in chunks]
    if path_depths:
        logger.debug(f"Path depth stats: min={min(path_depths)}, max={max(path_depths)}, avg={sum(path_depths)/len(path_depths):.1f}")
    
    value_types = defaultdict(int)
    for c in chunks:
        value_types[type(c['value']).__name__] += 1
    logger.debug(f"Value type distribution: {dict(value_types)}")
    
    return chunks[:max_chunks]

def process_json_files(directory: str) -> List[Dict]:
    """Process all JSON files in directory."""
    all_chunks = []
    
    # Walk through directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Skip test queries file
            if file == 'test_queries':
                continue
                
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                with open(file_path) as f:
                    data = json.load(f)
                    all_chunks.extend(json_to_path_chunks(data, str(file_path)))
    
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
        'id': chunk['id'],
        'parent_id': chunk.get('parent_id'),
        'depth': chunk['depth'],
        'path': chunk['path'],
        'has_children': chunk['metadata']['has_children']
    }

def combine_chunk_with_context(chunk: Dict, parents: List[Dict], children: List[Dict]) -> str:
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
        context_parts.append(f"Parent ({parent['path']}): {json.dumps(parent['content'])}")
    
    # Add main chunk
    context_parts.append(f"Current: {json.dumps(chunk['content'])}")
    
    # Add child context
    for child in children:
        context_parts.append(f"Child ({child['path']}): {json.dumps(child['content'])}")
    
    return "\n".join(context_parts) 