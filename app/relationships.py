# relationships.py
from typing import List, Dict, Optional, Tuple
import json
from collections import defaultdict
import re
from app.embedding import get_embedding
from app.archetype import ArchetypeDetector
from app.config import SIMILARITY_THRESHOLD
from app.database import get_chunk_by_id
from app.parsing import extract_key_value_pairs
import numpy as np
from .logging_config import get_logger

logger = get_logger(__name__)

def validate_relationship(source_archetype: str, target_archetype: str, relationship_type: str) -> Tuple[bool, float]:
    """Validate and score a relationship based on archetype patterns."""
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
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        float: Cosine similarity score between 0 and 1
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
    """Enhanced relationship detection with archetype validation."""
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
