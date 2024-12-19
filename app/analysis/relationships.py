# relationships.py
"""
Relationship Analysis Module

This module handles the detection and processing of relationships between JSON chunks,
including explicit references (like ID matches), semantic relationships, and temporal
connections.
"""

from typing import List, Dict, Optional, Tuple
import json
from datetime import datetime
import re
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def detect_explicit_references(chunks: List[Dict]) -> List[Dict]:
    """
    Detect explicit references between chunks based on ID matching.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of relationship dictionaries
    """
    relationships = []
    id_map = {}
    
    # First pass: Build ID map
    for chunk in chunks:
        content = chunk.get('content', {})
        if isinstance(content, dict):
            # Map any ID fields to their chunks
            for key, value in content.items():
                if isinstance(value, str) and (
                    key.endswith('_id') or 
                    key == 'id' or
                    re.match(r'^[A-Z]+-\d+$', str(value))  # Match patterns like PROD-002
                ):
                    id_map[value] = chunk['id']
    
    # Second pass: Detect references
    for chunk in chunks:
        content = chunk.get('content', {})
        if isinstance(content, dict):
            for key, value in content.items():
                if isinstance(value, str) and value in id_map:
                    target_id = id_map[value]
                    if target_id != chunk['id']:  # Avoid self-references
                        relationships.append({
                            'source_chunk_id': chunk['id'],
                            'target_chunk_id': target_id,
                            'relationship_type': 'explicit',
                            'confidence': 1.0,
                            'metadata': {
                                'field': key,
                                'value': value
                            }
                        })
    
    return relationships

def detect_semantic_relationships(chunks: List[Dict]) -> List[Dict]:
    """
    Detect semantic relationships between chunks based on content analysis.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of relationship dictionaries
    """
    relationships = []
    
    # Define semantic patterns
    patterns = {
        'invoice_ledger': {
            'keywords': ['invoice', 'payment', 'transaction', 'ledger', 'accounting'],
            'confidence': 0.7
        },
        'product_inventory': {
            'keywords': ['product', 'inventory', 'stock', 'warehouse', 'supply'],
            'confidence': 0.8
        },
        'customer_order': {
            'keywords': ['customer', 'order', 'purchase', 'buyer'],
            'confidence': 0.75
        }
    }
    
    # Group chunks by their content type
    chunk_types = {}
    for chunk in chunks:
        content = json.dumps(chunk.get('content', {})).lower()
        for pattern_type, pattern in patterns.items():
            if any(kw in content for kw in pattern['keywords']):
                if pattern_type not in chunk_types:
                    chunk_types[pattern_type] = []
                chunk_types[pattern_type].append(chunk)
    
    # Create relationships between related types
    related_types = {
        'invoice_ledger': ['customer_order'],
        'product_inventory': ['customer_order'],
        'customer_order': ['invoice_ledger', 'product_inventory']
    }
    
    for type1, related in related_types.items():
        if type1 in chunk_types:
            for type2 in related:
                if type2 in chunk_types:
                    for chunk1 in chunk_types[type1]:
                        for chunk2 in chunk_types[type2]:
                            relationships.append({
                                'source_chunk_id': chunk1['id'],
                                'target_chunk_id': chunk2['id'],
                                'relationship_type': 'semantic',
                                'confidence': patterns[type1]['confidence'],
                                'metadata': {
                                    'source_type': type1,
                                    'target_type': type2
                                }
                            })
    
    return relationships

def detect_temporal_relationships(chunks: List[Dict]) -> List[Dict]:
    """
    Detect temporal relationships between chunks based on dates and times.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of relationship dictionaries
    """
    relationships = []
    dated_chunks = []
    
    # Extract dates from chunks
    for chunk in chunks:
        content = chunk.get('content', {})
        dates = []
        
        # Look for date fields recursively
        def extract_dates(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        # Try to parse as date
                        try:
                            date = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            dates.append((key, date))
                        except ValueError:
                            pass
                    elif isinstance(value, (dict, list)):
                        extract_dates(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_dates(item)
        
        extract_dates(content)
        if dates:
            dated_chunks.append((chunk, dates))
    
    # Sort chunks by date
    dated_chunks.sort(key=lambda x: x[1][0][1])  # Sort by first date found
    
    # Create temporal relationships
    for i, (chunk1, dates1) in enumerate(dated_chunks):
        for j, (chunk2, dates2) in enumerate(dated_chunks[i+1:], i+1):
            # Find the closest dates between chunks
            min_diff = min(
                abs(d1[1] - d2[1]) 
                for d1 in dates1 
                for d2 in dates2
            )
            
            # If dates are within 30 days, create relationship
            if min_diff.days <= 30:
                relationships.append({
                    'source_chunk_id': chunk1['id'],
                    'target_chunk_id': chunk2['id'],
                    'relationship_type': 'temporal',
                    'confidence': 1.0 - (min_diff.days / 30),
                    'metadata': {
                        'date_difference_days': min_diff.days
                    }
                })
    
    return relationships

def process_relationships(chunks: List[Dict]) -> List[Dict]:
    """
    Process all types of relationships for a set of chunks.
    
    Args:
        chunks: List of chunk dictionaries
        
    Returns:
        List of all detected relationships
    """
    relationships = []
    
    # Detect explicit references
    explicit = detect_explicit_references(chunks)
    relationships.extend(explicit)
    logger.debug(f"Found {len(explicit)} explicit references")
    
    # Detect semantic relationships
    semantic = detect_semantic_relationships(chunks)
    relationships.extend(semantic)
    logger.debug(f"Found {len(semantic)} semantic relationships")
    
    # Detect temporal relationships
    temporal = detect_temporal_relationships(chunks)
    relationships.extend(temporal)
    logger.debug(f"Found {len(temporal)} temporal relationships")
    
    return relationships
