# relationships.py
"""
Relationship Detection and Analysis Module

This module provides functionality for detecting and analyzing relationships between
JSON document chunks. It uses semantic similarity and pattern matching to identify
connections between entities, events, and data structures.

Key Features:
    - Entity Relationship Detection
    - Semantic Similarity Analysis
    - Pattern-based Relationship Matching
    - Relationship Type Classification
    - Confidence Scoring

Usage:
    >>> from app.analysis.relationships import detect_relationships
    >>> chunks = [{'id': 'chunk1', 'content': {...}}, ...]
    >>> relationships = detect_relationships(chunks)
    >>> for rel in relationships:
    ...     print(f"{rel['source']} -> {rel['target']}: {rel['type']}")
"""

from typing import List, Dict, Optional, Tuple
import json
from collections import defaultdict
import re
import numpy as np

from app.retrieval.embedding import get_embedding
from app.analysis.archetype import ArchetypeDetector
from app.core.config import SIMILARITY_THRESHOLD
from app.storage.database import get_chunk_by_id
from app.utils.json_utils import extract_key_value_pairs
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def validate_relationship(source_archetype: str, target_archetype: str, relationship_type: str) -> Tuple[bool, float]:
    """
    Validate and score a relationship based on archetype patterns.
    
    This function determines whether a proposed relationship between two archetypes
    is valid according to predefined rules and assigns a confidence score to the
    relationship.
    
    Args:
        source_archetype (str): Archetype of the source entity
        target_archetype (str): Archetype of the target entity
        relationship_type (str): Type of relationship to validate
        
    Returns:
        Tuple[bool, float]: A tuple containing:
            - bool: Whether the relationship is valid
            - float: Confidence score (0.0 to 1.0) for the relationship
            
    Validation Rules:
        entity_definition:
            - Can reference other entities and metric data
            - Can contain metric data and event logs
            - Maximum traversal depth of 4
            
        event_log:
            - Can have temporal relationships with other events
            - Can trigger metric data updates
            - Maximum traversal depth of 3
            
        metric_data:
            - Can aggregate other metric data
            - Can be derived from event logs
            - Maximum traversal depth of 2
            
    Example:
        >>> is_valid, score = validate_relationship(
        ...     'entity_definition',
        ...     'metric_data',
        ...     'reference'
        ... )
        >>> print(f"Valid: {is_valid}, Score: {score}")
        Valid: True, Score: 0.9
    """
    # Safety check for None values
    if not source_archetype or not target_archetype:
        return True, 0.5
        
    # Define valid relationship patterns with traversal rules
    archetype_patterns = {
        'entity_definition': {
            'valid_relationships': {
                'reference': ['entity_definition', 'metric_data'],
                'contains': ['metric_data', 'event_log']
            },
            'max_depth': 4,
            'score': 0.9
        },
        'event_log': {
            'valid_relationships': {
                'before': ['event_log'],
                'after': ['event_log'],
                'triggers': ['metric_data']
            },
            'max_depth': 3,
            'score': 0.8
        },
        'metric_data': {
            'valid_relationships': {
                'aggregates': ['metric_data'],
                'derives_from': ['event_log']
            },
            'max_depth': 2,
            'score': 0.7
        }
    }

    pattern = archetype_patterns.get(source_archetype, {})
    valid_targets = pattern.get('valid_relationships', {}).get(relationship_type, [])
    
    if target_archetype in valid_targets:
        return True, pattern.get('score', 0.5)
    return False, 0.0

def compute_similarity(embedding1, embedding2) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    This function calculates the cosine similarity between two embedding vectors,
    handling various input formats and edge cases. It's used primarily for
    detecting semantic relationships between entities.
    
    Args:
        embedding1: First embedding vector (numpy array or list)
        embedding2: Second embedding vector (numpy array or list)
        
    Returns:
        float: Cosine similarity score between 0 and 1, where:
            1.0 = Identical embeddings
            0.0 = Completely dissimilar or invalid embeddings
            
    Edge Cases:
        - Returns 0.0 for None or empty embeddings
        - Handles conversion from list to numpy array
        - Protects against zero-norm vectors
        
    Example:
        >>> v1 = [0.1, 0.2, 0.3]
        >>> v2 = [0.2, 0.3, 0.4]
        >>> similarity = compute_similarity(v1, v2)
        >>> print(f"Similarity: {similarity:.2f}")
        
    Note:
        The function uses numpy for efficient computation and
        automatically handles type conversion and validation.
    """
    # Check for None or empty embeddings
    if embedding1 is None or embedding2 is None:
        return 0.0
        
    # Convert to numpy arrays if needed
    if isinstance(embedding1, list):
        embedding1 = np.array(embedding1)
    if isinstance(embedding2, list):
        embedding2 = np.array(embedding2)
        
    # Check for empty arrays
    if embedding1.size == 0 or embedding2.size == 0:
        return 0.0
        
    # Compute cosine similarity
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
        
    return float(dot_product / (norm1 * norm2))

def detect_relationships(chunks, conn):
    """
    Enhanced relationship detection with archetype validation and semantic analysis.
    
    This function performs a comprehensive analysis of relationships between JSON
    document chunks, detecting both direct (explicit) and semantic (implicit)
    relationships. It uses a multi-pass approach to build an entity registry
    and identify various types of relationships.
    
    Process Flow:
        1. First Pass - Entity Registration:
            - Extract key-value pairs from chunks
            - Build entity registry with metadata
            - Map entity IDs to chunk IDs
            - Detect and validate direct relationships
            
        2. Second Pass - Semantic Analysis:
            - Generate embeddings for entities
            - Compute semantic similarities
            - Identify related entities
            - Score relationship confidence
            
    Args:
        chunks: List of JSON document chunks to analyze
        conn: Database connection for additional context lookup
        
    Returns:
        dict: Comprehensive relationship analysis containing:
            - direct: Dictionary of direct relationships by chunk ID
            - semantic: Dictionary of semantic relationships by entity ID
            - entity_registry: Registry of all detected entities
            - chunk_id_map: Mapping between entity IDs and chunk IDs
            
    Example:
        >>> chunks = [
        ...     {"id": "1", "content": {"user_id": "U123", "name": "Alice"}},
        ...     {"id": "2", "content": {"order_id": "O456", "user_id": "U123"}}
        ... ]
        >>> results = detect_relationships(chunks, conn)
        >>> print(f"Direct relationships: {len(results['direct'])}")
        >>> print(f"Semantic relationships: {len(results['semantic'])}")
        
    Note:
        - Uses archetype detection for relationship validation
        - Considers both structural and semantic relationships
        - Maintains bidirectional relationship mapping
        - Includes confidence scores for relationships
        - Handles errors gracefully with logging
    """
    # Initialize relationship tracking and detector
    detector = ArchetypeDetector()
    direct_relations = defaultdict(list)
    semantic_relations = defaultdict(list)
    entity_registry = {}
    chunk_id_map = {}  # Map entity IDs to chunk IDs
    
    logger.debug("Processing chunks for relationships...")
    
    # First pass: extract key-value pairs and build entity registry
    for chunk in chunks:
        try:
            if not chunk:  # Skip None chunks
                continue
                
            chunk_id = chunk.get('id') or f"chunk_{len(chunk_id_map)}"
            
            # Get archetype for the chunk - pass the whole chunk dict
            source_archetype = None
            chunk_archetypes = detector.detect_archetypes(chunk)
            if chunk_archetypes:
                source_archetype = chunk_archetypes[0][0]  # Get first archetype type
            
            kv_pairs = extract_key_value_pairs(chunk)
            if not kv_pairs:  # Skip if no key-value pairs found
                continue
                
            # Look for any ID or name fields
            entity_pairs = {k: v for k, v in kv_pairs.items() 
                          if any(pattern in k.lower() for pattern in [
                              '_id', 'id_', 'id',
                              '_name', 'name_', 'name'
                          ])}
            
            logger.debug(f"Found entity pairs: {entity_pairs}")
            
            for key, value in entity_pairs.items():
                entity_id = str(value)
                chunk_id_map[entity_id] = chunk_id
                
                if entity_id not in entity_registry:
                    # Get display name from key-value pairs
                    display_name = None
                    name_keys = ['name', 'title', 'label', 'display_name']
                    for name_key in name_keys:
                        if name_key in kv_pairs:
                            display_name = kv_pairs[name_key]
                            break
                    
                    # Create new entity entry
                    entity_registry[entity_id] = {
                        'type': key.split('_')[0] if '_' in key else 'unknown',
                        'name': display_name or entity_id,  # Use ID if no name found
                        'chunk_id': chunk_id,
                        'embedding': None,
                        'source_chunks': [],
                        'archetype': source_archetype  # Store the archetype
                    }
                    
                    if display_name:
                        logger.debug(f"Registered entity {entity_id} with name '{display_name}'")
                
                # Store relationship using chunk IDs
                if isinstance(chunk, dict):
                    entity_registry[entity_id]['source_chunks'].append(chunk_id)
                
                # Look for direct ID references
                for ref_key, ref_value in kv_pairs.items():
                    if '_id' in ref_key.lower() and ref_value != entity_id:
                        # Get target archetype from registry if available
                        target_archetype = entity_registry.get(str(ref_value), {}).get('archetype')
                        
                        # Validate relationship
                        is_valid, confidence = validate_relationship(
                            source_archetype,
                            target_archetype,
                            'reference'  # Default type for ID references
                        )
                        
                        if is_valid:
                            relation = {
                                'type': 'reference',
                                'source': chunk_id,
                                'source_name': entity_registry[entity_id]['name'],
                                'target': chunk_id_map.get(str(ref_value), chunk_id),
                                'target_name': entity_registry.get(str(ref_value), {}).get('name', str(ref_value)),
                                'context': {
                                    'reference_type': ref_key,
                                    'confidence': confidence,
                                    'source_archetype': source_archetype,
                                    'target_archetype': target_archetype
                                }
                            }
                            direct_relations[chunk_id].append(relation)
                
        except Exception as e:
            logger.error(f"Failed to process chunk for relationships: {e}")
            continue
    
    logger.debug(f"Found {len(direct_relations)} entities with direct relationships")
    
    # Second pass: compute semantic relationships
    for entity_id, entity_data in entity_registry.items():
        if not entity_data['embedding']:
            # Get archetype from entity type
            archetype = {
                'type': 'entity_definition' if entity_data['type'] != 'unknown' else None,
                'confidence': 0.8 if entity_data['type'] != 'unknown' else 0.5
            }
            
            # Create proper JSON structure for embedding
            entity_text = {
                'id': entity_id,
                'type': entity_data['type'],
                'name': entity_data['name']
            }
            entity_data['embedding'] = get_embedding(json.dumps(entity_text), archetype)
    
    # Find semantic similarities between entities
    for entity_id, entity_data in entity_registry.items():
        for other_id, other_data in entity_registry.items():
            if entity_id != other_id and entity_data['type'] == other_data['type']:
                similarity = compute_similarity(entity_data['embedding'], other_data['embedding'])
                if similarity > SIMILARITY_THRESHOLD:
                    relation = {
                        'type': 'semantic',
                        'source': entity_id,
                        'source_name': entity_data['name'],
                        'target': other_id,
                        'target_name': other_data['name'],
                        'confidence': similarity
                    }
                    semantic_relations[entity_id].append(relation)
    
    return {
        'direct': direct_relations,
        'semantic': semantic_relations,
        'entity_registry': entity_registry,
        'chunk_id_map': chunk_id_map
    }
