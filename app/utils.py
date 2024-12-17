import hashlib
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re

def compute_file_hash(filepath: str) -> str:
    """Compute SHA-256 hash of file contents."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_json_files(data_dir: str = "data") -> List[str]:
    """Get list of JSON files in data directory."""
    json_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def parse_timestamp(timestamp_str: str) -> Optional[datetime]:
    """Parse timestamp string into datetime object."""
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
    """Classify JSON path components for context."""
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
