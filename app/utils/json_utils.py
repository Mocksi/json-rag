"""
JSON Utility Functions

This module provides common utility functions for JSON processing that are used
across multiple modules in the system. It helps avoid circular dependencies by
centralizing shared functionality.
"""

from typing import Dict, List, Any, Tuple
import re
from datetime import datetime

def extract_key_value_pairs(data: Any, path: str = "") -> List[Tuple[str, Any]]:
    """
    Extract key-value pairs from a JSON structure with their full paths.
    
    Args:
        data: JSON data to process
        path: Current path in the structure (for recursion)
        
    Returns:
        List of (path, value) tuples
        
    Example:
        >>> data = {"user": {"id": 123, "name": "Alice"}}
        >>> pairs = extract_key_value_pairs(data)
        >>> for path, value in pairs:
        ...     print(f"{path}: {value}")
        user.id: 123
        user.name: Alice
    """
    pairs = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            
            # Add the current key-value pair
            pairs.append((current_path, value))
            
            # Recurse into nested structures
            if isinstance(value, (dict, list)):
                pairs.extend(extract_key_value_pairs(value, current_path))
                
    elif isinstance(data, list):
        for i, item in enumerate(data):
            current_path = f"{path}[{i}]"
            
            # Add the current index-value pair
            pairs.append((current_path, item))
            
            # Recurse into nested structures
            if isinstance(item, (dict, list)):
                pairs.extend(extract_key_value_pairs(item, current_path))
                
    return pairs

def normalize_json_path(path: str) -> str:
    """
    Normalize a JSON path for consistent formatting.
    
    Args:
        path: Raw JSON path string
        
    Returns:
        Normalized path string
        
    Example:
        >>> normalize_json_path("root[0].items[1].name")
        "root.0.items.1.name"
    """
    # Remove leading/trailing dots and spaces
    path = path.strip().strip('.')
    
    # Replace array notation with dot notation
    while '[' in path and ']' in path:
        start = path.find('[')
        end = path.find(']', start)
        if start != -1 and end != -1:
            array_index = path[start+1:end]
            path = path[:start] + '.' + array_index + path[end+1:]
    
    # Clean up double dots
    while '..' in path:
        path = path.replace('..', '.')
    
    return path

def classify_path(path: str) -> str:
    """
    Classify a JSON path based on its structure and naming.
    
    Args:
        path: JSON path to classify
        
    Returns:
        Classification string indicating the likely data type/purpose
        
    Example:
        >>> classify_path("users.0.email")
        "contact_info"
        >>> classify_path("order.items.3.price")
        "financial"
    """
    # Common path patterns
    patterns = {
        'temporal': r'(date|time|timestamp|created|updated|deleted)(_at)?$',
        'identifier': r'(id|uuid|key|ref)$',
        'financial': r'(price|cost|amount|total|tax|discount)$',
        'quantity': r'(count|quantity|stock|inventory|available)$',
        'status': r'(status|state|condition|phase|stage)$',
        'contact': r'(email|phone|address|contact)$',
        'metadata': r'(type|category|tag|label|group)$'
    }
    
    # Check path against patterns
    path_parts = path.split('.')
    for part in path_parts:
        for category, pattern in patterns.items():
            if re.search(pattern, part, re.IGNORECASE):
                return category
                
    return 'general'

def parse_timestamp(value: Any) -> datetime:
    """
    Parse a timestamp value into a datetime object.
    
    Args:
        value: Timestamp value to parse
        
    Returns:
        datetime object
        
    Raises:
        ValueError: If the timestamp format is not recognized
        
    Example:
        >>> parse_timestamp("2024-03-16T10:30:00Z")
        datetime(2024, 3, 16, 10, 30)
    """
    if isinstance(value, datetime):
        return value
        
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
        
    if isinstance(value, str):
        # Try common formats
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",  # ISO 8601
            "%Y-%m-%d %H:%M:%S",    # Standard datetime
            "%Y-%m-%d",             # Date only
            "%d/%m/%Y %H:%M:%S",    # UK format
            "%m/%d/%Y %H:%M:%S"     # US format
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
                
    raise ValueError(f"Unrecognized timestamp format: {value}") 