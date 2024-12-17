import hashlib
import os
from datetime import datetime
from dateutil.parser import parse

def compute_file_hash(filepath):
    """
    Computes SHA-256 hash of a file.
    
    Args:
        filepath (str): Path to the file to hash
        
    Returns:
        str: Hexadecimal string of the file's SHA-256 hash
    """
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def parse_timestamp(timestamp_str):
    """
    Parses timestamp strings in various formats.
    
    Args:
        timestamp_str (str): Timestamp string to parse
        
    Returns:
        datetime: Parsed datetime object, or None if parsing fails
        
    Supported Formats:
        - ISO format: 2024-03-10T14:30:00
        - Common date-time: 2024-03-10 14:30:00
        - Date only: 2024-03-10
        - Fallback to dateutil.parser for other formats
    """
    if not timestamp_str:
        return None
    # Try common formats, then fallback to dateutil
    formats = [
        '%Y-%m-%dT%H:%M:%S',
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d'
    ]
    for fmt in formats:
        try:
            return datetime.strptime(timestamp_str, fmt)
        except ValueError:
            continue
    try:
        return parse(timestamp_str)
    except:
        return None

def classify_path(path):
    """
    Classifies a JSON path by its access pattern.
    
    Args:
        path (str): JSON path to classify
        
    Returns:
        str: Path classification:
            - 'root': Root path
            - 'array_access': Contains array indexing
            - 'nested_object': Contains dot notation
            - 'direct_access': Direct property access
    """
    if path == 'root':
        return 'root'
    if '[' in path:
        return 'array_access'
    if '.' in path:
        return 'nested_object'
    return 'direct_access'

def get_json_files():
    """
    Gets list of JSON files from the configured data directory.
    
    Returns:
        list: List of paths to JSON files in DATA_DIR
        
    Note:
        Uses DATA_DIR from app.config
    """
    from app.config import DATA_DIR
    import glob
    return glob.glob(os.path.join(DATA_DIR, "*.json"))
