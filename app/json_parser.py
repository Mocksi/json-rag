from typing import List, Dict, Any, Optional
import hashlib
import json
from pathlib import Path
from .logging_config import get_logger

logger = get_logger(__name__)

def generate_chunk_id(source_file: str, path: str) -> str:
    """
    Generate a unique chunk ID based on source file and path.
    
    Args:
        source_file: Source file path
        path: JSON path within the file
        
    Returns:
        str: MD5 hash of the combined string
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
        path: JSON path to normalize
        
    Returns:
        str: Normalized path
        
    Example:
        >>> normalize_json_path("data.users[0].profile")
        'data.users.0.profile'
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

def json_to_path_chunks(data: Any, source_file: str, parent_id: Optional[str] = None, path: str = '', depth: int = 0) -> List[Dict]:
    """
    Convert JSON data to chunks while preserving hierarchical relationships.
    
    Args:
        data: JSON data to process
        source_file: Source file path
        parent_id: ID of parent chunk
        path: Current JSON path
        depth: Current depth in the hierarchy
        
    Returns:
        list: List of chunks with hierarchical information
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
    Process a JSON file and convert it to chunks with hierarchical information.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        list: List of chunks with hierarchical information
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
        
    with open(file_path) as f:
        data = json.load(f)
        
    return json_to_path_chunks(data, str(file_path))

def get_chunk_hierarchy_info(chunk: Dict) -> Dict:
    """
    Extract hierarchy information from a chunk.
    
    Args:
        chunk: Chunk dictionary
        
    Returns:
        dict: Hierarchy information including depth and path
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
    Combine chunk content with its hierarchical context for embedding generation.
    
    Args:
        chunk: Main chunk
        parents: List of parent chunks
        children: List of child chunks
        
    Returns:
        str: Combined content for embedding
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