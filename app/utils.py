"""
Utility Functions Module

This module provides common utility functions used throughout the JSON RAG system.
It includes functions for file operations, timestamp parsing, and path analysis.

Key Function Categories:
    - File Operations: Hash computation, JSON file discovery
    - Time Handling: Timestamp parsing with multiple format support
    - Path Analysis: JSON path classification and context extraction
    - Data Validation: Input validation and normalization
"""

import hashlib
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re

def compute_file_hash(filepath: str) -> str:
    """
    Compute SHA-256 hash of file contents.
    
    Args:
        filepath (str): Path to the file to hash
        
    Returns:
        str: Hexadecimal string of the SHA-256 hash
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        PermissionError: If the file can't be read
        
    Note:
        Uses 4KB blocks for memory-efficient processing of large files
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_json_files(data_dir: str = "data") -> List[str]:
    """
    Get list of JSON files in data directory and its subdirectories.
    
    Args:
        data_dir (str, optional): Root directory to search. Defaults to "data".
        
    Returns:
        List[str]: List of paths to JSON files
        
    Example:
        >>> files = get_json_files("data/documents")
        >>> print(files)
        ['data/documents/doc1.json', 'data/documents/subfolder/doc2.json']
    """
    json_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """
    Parse timestamp string into datetime object with multiple format support.
    
    Supported formats:
        - ISO format: YYYY-MM-DDTHH:MM:SS
        - SQL format: YYYY-MM-DD HH:MM:SS
        - Slash format: YYYY/MM/DD HH:MM:SS
        - Date only: YYYY-MM-DD
    
    Args:
        timestamp_str (str): Timestamp string to parse
        
    Returns:
        Optional[datetime]: Parsed datetime object, or None if parsing fails
        
    Example:
        >>> dt = parse_timestamp("2023-12-18T15:30:00")
        >>> print(dt)
        2023-12-18 15:30:00
    """
    timestamp_patterns = [
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # SQL format
        r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}',  # Slash format
        r'\d{4}-\d{2}-\d{2}'                      # Date only
    ]
    
    for pattern in timestamp_patterns:
        match = re.search(pattern, timestamp_str)
        if match:
            try:
                return datetime.fromisoformat(match.group().replace('/', '-'))
            except ValueError:
                continue
    return None

def classify_path(path: str) -> Dict[str, str]:
    """
    Classify JSON path components for semantic context extraction.
    
    This function analyzes a JSON path to determine its semantic role
    and hierarchical context within the document structure.
    
    Args:
        path (str): JSON path string (e.g., "users.0.profile.name")
        
    Returns:
        Dict[str, str]: Classification containing:
            - type: Path type (array_item, object_property, etc.)
            - context: Contextual description of the path
            - depth: Number of path segments
            
    Example:
        >>> info = classify_path("users.0.profile.name")
        >>> print(info)
        {'type': 'object_property', 'context': 'user profile', 'depth': '4'}
    """
    parts = path.split('.')
    classification = {
        'type': 'unknown',
        'context': '',
        'depth': str(len(parts))
    }
    
    # Identify common patterns
    if any(key in path.lower() for key in ['id', 'key', 'code']):
        classification['type'] = 'identifier'
    elif any(key in path.lower() for key in ['time', 'date', 'created', 'updated']):
        classification['type'] = 'temporal'
    elif any(key in path.lower() for key in ['name', 'title', 'label']):
        classification['type'] = 'descriptor'
    elif any(key in path.lower() for key in ['count', 'total', 'sum', 'avg']):
        classification['type'] = 'metric'
        
    return classification
