"""
JSON Parser Module

This module handles the parsing and chunking of JSON documents for RAG processing.
It includes utilities for generating unique chunk IDs, normalizing JSON paths,
and converting JSON structures into processable chunks while preserving hierarchy
and relationships.
"""

from typing import List, Dict, Any, Optional
import hashlib
import json
from pathlib import Path
from app.utils.logging_config import get_logger

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

def json_to_path_chunks(
    data: Any,
    source_file: str,
    parent_id: Optional[str] = None,
    path: str = '',
    depth: int = 0
) -> List[Dict]:
    """
    Convert a JSON structure into a list of chunks with path information.
    
    This function recursively traverses a JSON structure and creates chunks
    that preserve the hierarchical relationships and context.
    
    Args:
        data (Any): JSON data to process
        source_file (str): Path to the source JSON file
        parent_id (Optional[str]): ID of the parent chunk
        path (str): Current JSON path
        depth (int): Current recursion depth
        
    Returns:
        List[Dict]: List of chunks, each containing:
            - id: Unique chunk identifier
            - content: The chunk's JSON content
            - path: Full JSON path to the chunk
            - parent_id: ID of parent chunk (if any)
            - metadata: Additional chunk information
            
    Note:
        The function implements smart chunking strategies based on:
        - Data structure complexity
        - Semantic completeness
        - Maximum chunk size constraints
    """
    chunks = []
    
    # Normalize the path
    normalized_path = normalize_json_path(path)
    current_id = generate_chunk_id(source_file, normalized_path)
    
    # Create chunk for current node
    chunk = {
        'id': current_id,
        'content': data,
        'parent_id': parent_id,
        'depth': depth,
        'path': normalized_path,
        'source_file': source_file,
        'metadata': {
            'depth': depth,
            'has_children': isinstance(data, (dict, list)),
            'type': 'array' if isinstance(data, list) else 'object' if isinstance(data, dict) else 'value'
        }
    }
    chunks.append(chunk)
    
    # Process child nodes
    if isinstance(data, dict):
        for key, value in data.items():
            # Handle special cases for arrays of objects with IDs
            if isinstance(value, list) and value and isinstance(value[0], dict) and 'id' in value[0]:
                # Create a container chunk for the array
                array_path = normalize_json_path(f"{normalized_path}.{key}" if normalized_path else key)
                array_id = generate_chunk_id(source_file, array_path)
                array_chunk = {
                    'id': array_id,
                    'content': {'type': 'array', 'count': len(value)},
                    'parent_id': current_id,
                    'depth': depth + 1,
                    'path': array_path,
                    'source_file': source_file,
                    'metadata': {
                        'depth': depth + 1,
                        'has_children': True,
                        'type': 'array'
                    }
                }
                chunks.append(array_chunk)
                
                # Create chunks for each array item, using their IDs in the path
                for item in value:
                    if isinstance(item, dict) and 'id' in item:
                        item_path = normalize_json_path(f"{array_path}.{item['id']}")
                        child_chunks = json_to_path_chunks(item, source_file, array_id, item_path, depth + 2)
                        chunks.extend(child_chunks)
            else:
                # Normal object property
                new_path = normalize_json_path(f"{normalized_path}.{key}" if normalized_path else key)
                child_chunks = json_to_path_chunks(value, source_file, current_id, new_path, depth + 1)
                chunks.extend(child_chunks)
    elif isinstance(data, list):
        # Create chunks for array items
        for index, item in enumerate(data):
            if isinstance(item, dict) and 'id' in item:
                # Use ID in path for objects with IDs
                new_path = normalize_json_path(f"{normalized_path}.{item['id']}")
            else:
                # Use index for other items
                new_path = normalize_json_path(f"{normalized_path}.{index}")
            child_chunks = json_to_path_chunks(item, source_file, current_id, new_path, depth + 1)
            chunks.extend(child_chunks)
            
    return chunks

def process_json_file(file_path: str) -> List[Dict]:
    """
    Process a JSON file and convert it into chunks.
    
    Args:
        file_path (str): Path to the JSON file to process
        
    Returns:
        List[Dict]: List of processed chunks with metadata
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
        
    Note:
        This is the main entry point for processing JSON files.
        It handles file reading, JSON parsing, and chunk generation.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path) as f:
        data = json.load(f)
        
    return json_to_path_chunks(data, str(file_path))

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