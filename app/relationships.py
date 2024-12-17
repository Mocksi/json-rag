# relationships.py
from app.parsing import extract_key_value_pairs
from app.embedding import get_embedding
from collections import defaultdict
import re

def detect_relationships(chunks):
    """
    Detect relationships between entities in chunks using direct ID matching
    and semantic similarity.
    
    Args:
        chunks (list): List of chunk dictionaries containing entity information
        
    Returns:
        dict: Detected relationships containing:
            - direct: Dictionary of direct ID-based relationships
            - semantic: Dictionary of similarity-based relationships
            - entity_registry: Registry of all found entities
            
    Process:
        1. First pass: Extract key-value pairs and build entity registry
        2. Second pass: Compute semantic relationships using embeddings
        3. Combines both types of relationships with confidence scores
        
    Note:
        Direct relationships are found through ID references
        Semantic relationships use embedding similarity above SIMILARITY_THRESHOLD
    """
    # Initialize relationship tracking
    direct_relations = defaultdict(list)
    semantic_relations = defaultdict(list)
    entity_registry = {}
    
    print("\nDEBUG: Processing chunks for relationships...")
    
    # First pass: extract key-value pairs and build entity registry
    for chunk in chunks:
        try:
            # Extract searchable attributes using existing parser
            kv_pairs = extract_key_value_pairs(chunk)
            
            # Look for entity identifiers and names
            entity_pairs = {k: v for k, v in kv_pairs.items() 
                          if any(pattern in k.lower() for pattern in ['entity_', 'id_', '_id', 'name_'])}
            
            for key, value in entity_pairs.items():
                entity_id = str(value)
                if entity_id not in entity_registry:
                    # Get display name from various possible sources
                    display_name = None
                    if 'name' in kv_pairs:
                        display_name = kv_pairs['name']
                    elif f"{key}_name" in kv_pairs:
                        display_name = kv_pairs[f"{key}_name"]
                    elif 'display_name' in kv_pairs:
                        display_name = kv_pairs['display_name']
                    
                    # Create new entity entry with enhanced name handling
                    entity_registry[entity_id] = {
                        'type': key.split('_')[0] if '_' in key else 'unknown',
                        'name': display_name or entity_id,  # Prioritize display name
                        'description': kv_pairs.get(f"{key}_description", ''),
                        'context': {k: v for k, v in kv_pairs.items() if k.startswith('context_')},
                        'embedding': None,  # Will compute later
                        'source_chunks': [],
                        'display_name': display_name  # Store separately for reference
                    }
                    
                    if display_name:
                        print(f"DEBUG: Registered entity {entity_id} with name '{display_name}'")
                    
                entity_registry[entity_id]['source_chunks'].append(chunk)
                
                # Look for direct ID references with enhanced context
                for ref_key, ref_value in kv_pairs.items():
                    if '_id' in ref_key.lower() and ref_value != entity_id:
                        relation = {
                            'type': 'reference',
                            'source': entity_id,
                            'source_name': entity_registry[entity_id]['name'],
                            'target': str(ref_value),
                            'target_name': entity_registry.get(str(ref_value), {}).get('name', str(ref_value)),
                            'context': {'reference_type': ref_key.replace('_id', '')}
                        }
                        direct_relations[entity_id].append(relation)
                        print(f"DEBUG: Found direct relationship: {relation['source_name']} -> {relation['target_name']} ({ref_key})")
                        
        except Exception as e:
            print(f"ERROR: Failed to process chunk for relationships: {e}")
            continue
    
    print(f"\nDEBUG: Found {len(direct_relations)} entities with direct relationships")
    
    # Second pass: compute semantic relationships using embeddings
    for entity_id, entity_data in entity_registry.items():
        if not entity_data['embedding']:
            # Generate embedding from entity attributes including display name
            text_to_embed = f"{entity_data['name']} {entity_data['description']} {' '.join(str(v) for v in entity_data['context'].values())}"
            entity_data['embedding'] = get_embedding(text_to_embed)
    
    # Find semantic similarities between entities
    from app.config import SIMILARITY_THRESHOLD
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
                        'confidence': similarity,
                        'context': {'relationship_type': 'similar_entity'}
                    }
                    semantic_relations[entity_id].append(relation)
                    print(f"DEBUG: Found semantic relationship: {relation['source_name']} -> {relation['target_name']} (confidence: {similarity:.2f})")
    
    print(f"\nDEBUG: Found {len(semantic_relations)} entities with semantic relationships")
    
    # Combine and return all relationships with enhanced name information
    return {
        'direct': dict(direct_relations),
        'semantic': dict(semantic_relations),
        'entity_registry': entity_registry
    }

def compute_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1 (list): First embedding vector
        embedding2 (list): Second embedding vector
        
    Returns:
        float: Similarity score between 0 and 1
        
    Note:
        Uses numpy's dot product and norm functions for efficient computation
        Higher scores indicate more similar embeddings
    """
    import numpy as np
    
    # Convert to numpy arrays
    v1 = np.array(embedding1)
    v2 = np.array(embedding2)
    
    # Compute cosine similarity
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
