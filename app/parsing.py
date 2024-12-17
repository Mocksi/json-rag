import json
import math
import re
from datetime import datetime
from collections import defaultdict
from app.utils import parse_timestamp, classify_path
from app.models import FlexibleModel

def serialize_value(value):
    """
    Serializes a Python value into a structured dictionary with type information.
    
    Args:
        value: Any Python value (None, primitive, dict, list, or other)
        
    Returns:
        dict: Structured representation containing:
            - type: 'null', 'primitive', 'complex', or 'other'
            - value: The actual value or sample for complex types
            - additional metadata based on type
    """
    if value is None:
        return {'type': 'null', 'value': None}
    if isinstance(value, (str, int, float, bool)):
        return {
            'type': 'primitive',
            'value': value,
            'python_type': type(value).__name__
        }
    if isinstance(value, dict):
        return {
            'type': 'complex',
            'structure': 'dict',
            'size': len(value),
            'keys': list(value.keys()),
            'sample': {k: serialize_value(v) for k, v in list(value.items())[:3]},
            'summary': {
                'total_keys': len(value),
                'value_types': list(set(type(v).__name__ for v in value.values()))
            }
        }
    if isinstance(value, list):
        return {
            'type': 'complex',
            'structure': 'list',
            'size': len(value),
            'sample': [serialize_value(v) for v in value[:3]],
            'summary': {
                'total_items': len(value),
                'value_types': list(set(type(v).__name__ for v in value))
            }
        }
    return {
        'type': 'other',
        'python_type': type(value).__name__,
        'string_repr': str(value)
    }

def serialize_context(context):
    """
    Serializes a context dictionary into a format suitable for storage.
    
    Args:
        context (dict): Context dictionary to serialize
        
    Returns:
        dict: Serialized context with primitive values preserved and
             complex values converted to strings
        
    Note:
        Handles serialization errors by converting problematic values
        to error messages.
    """
    if not isinstance(context, dict):
        return {'error': 'Invalid context type'}
    serialized = {}
    for key, value in context.items():
        try:
            if isinstance(value, (str, int, float, bool, type(None))):
                serialized[key] = value
            else:
                serialized[key] = str(value)
        except Exception as e:
            serialized[key] = f"<Error serializing: {str(e)}>"
    return serialized

def serialize_entities(entities):
    """
    Serializes entity information into a structured format.
    
    Args:
        entities (dict): Dictionary of entity information
        
    Returns:
        dict: Serialized entities with:
            - type: Entity type
            - attributes: Serialized context
            - relationships: Serialized relationships
    """
    if not isinstance(entities, dict):
        return {'error': 'Invalid entities type'}
    serialized = {}
    for entity_id, entity_data in entities.items():
        try:
            serialized[str(entity_id)] = {
                'type': entity_data.get('type', 'unknown'),
                'attributes': serialize_context(entity_data.get('attributes', {})),
                'relationships': serialize_context(entity_data.get('relationships', {}))
            }
        except Exception as e:
            serialized[str(entity_id)] = {'error': f"Serialization failed: {str(e)}"}
    return serialized

def enrich_chunk_metadata(chunk_data, value):
    """
    Enriches chunk data with additional metadata based on value type.
    
    Args:
        chunk_data (dict): Chunk dictionary to enrich
        value: Value to analyze for metadata
        
    Adds:
        - numeric_metadata for numbers
        - temporal_metadata for timestamps
        - string_metadata for text
    """
    if isinstance(value, (int, float)):
        val = value
        chunk_data['numeric_metadata'] = {
            'value_exact': val,
            'value_range': {
                'min': val - abs(val * 0.1),
                'max': val + abs(val * 0.1)
            },
            'magnitude': math.floor(math.log10(abs(val))) if val != 0 else 0
        }
    if isinstance(value, str):
        try:
            parsed_date = parse_timestamp(value)
            if parsed_date:
                chunk_data['temporal_metadata'] = {
                    'timestamp': parsed_date.isoformat(),
                    'year': parsed_date.year,
                    'month': parsed_date.month,
                    'day': parsed_date.day,
                    'weekday': parsed_date.weekday()
                }
        except:
            pass
        chunk_data['string_metadata'] = {
            'length': len(value),
            'contains_numbers': bool(re.search(r'\d', value)),
            'contains_urls': bool(re.search(r'https?://\S+', value))
        }

def create_enhanced_chunk(path, value, context=None, entities=None):
    """
    Creates an enhanced chunk with metadata, context, and entity information.
    
    Args:
        path (str): JSONPath to the value
        value: The value at the path
        context (dict, optional): Additional context
        entities (dict, optional): Related entity information
        
    Returns:
        str: JSON string containing the enhanced chunk
        
    Note:
        Handles serialization errors gracefully by returning error chunk.
    """
    from app.config import MAX_CHUNKS
    chunk_data = {
        'path': path,
        'value': serialize_value(value),
        'context': serialize_context(context) if context else {},
        'entities': serialize_entities(entities) if entities else {},
        'metadata': {
            'created_at': datetime.now().isoformat(),
            'chunk_version': '2.0',
            'path_depth': len(path.split('.')),
            'path_type': classify_path(path)
        }
    }
    enrich_chunk_metadata(chunk_data, value)
    try:
        return json.dumps(chunk_data, default=str)
    except TypeError as e:
        print(f"Error serializing chunk data: {e}")
        return json.dumps({
            'path': path,
            'error': 'Serialization failed',
            'metadata': chunk_data['metadata']
        })

def track_entity_relationships(json_obj, current_path="$", parent_context=None):
    """
    Tracks relationships between entities in a JSON object.
    
    Args:
        json_obj: JSON object to analyze
        current_path (str): Current path in the JSON (default: "$")
        parent_context (dict, optional): Context from parent nodes
        
    Returns:
        list: Found relationships with:
            - entity information
            - relationship type
            - context
            - path information
    """
    relationships = []
    def process_node(node, path, context):
        if isinstance(node, dict):
            entity_types = {
                'name': 'person',
                'id': 'identifier',
                'title': 'document',
                'key': 'reference',
                'uuid': 'unique_id',
                'email': 'contact'
            }
            found_entity = {}
            for key, entity_type in entity_types.items():
                if key in node:
                    found_entity['type'] = entity_type
                    found_entity['value'] = node[key]
                    found_entity['path'] = path
                    found_entity['context'] = {k: v for k, v in node.items() if k != key}
                    break
            
            if found_entity:
                path_parts = path.split('.')
                for i, part in enumerate(path_parts):
                    if part in ['member', 'owner', 'participant', 'author', 'assignee']:
                        found_entity['role'] = part
                    elif part in ['project', 'team', 'organization', 'department']:
                        found_entity['group_type'] = part
                        if i > 0:
                            found_entity['group_context'] = path_parts[i-1]
                if 'actor' in node:
                    actor = node['actor']
                    if isinstance(actor, dict):
                        if 'role' in actor:
                            found_entity['actor_role'] = actor['role']
                        if 'permissions' in actor:
                            found_entity['permissions'] = actor['permissions']
                if 'organization' in context:
                    found_entity['org_context'] = context['organization']
                if 'project' in context:
                    found_entity['project_context'] = context['project']
                relationships.append(found_entity)
            
            for key, value in node.items():
                new_path = f"{path}.{key}"
                new_context = {**context} if context else {}
                new_context.update({k: v for k, v in node.items() if not isinstance(v, (dict, list))})
                process_node(value, new_path, new_context)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                new_path = f"{path}[{i}]"
                process_node(item, new_path, context)
    process_node(json_obj, current_path, parent_context)
    return relationships

def extract_entities(json_obj, current_path="$"):
    """
    Extracts entities and their relationships from a JSON object.
    
    Args:
        json_obj: JSON object to analyze
        current_path (str): Current path in the JSON (default: "$")
        
    Returns:
        dict: Extracted entities with:
            - path: Location in JSON
            - type: Entity type
            - context: Related context
            - role: Entity role if applicable
            - group information if applicable
    """
    relationships = track_entity_relationships(json_obj, current_path)
    entities = {}
    for rel in relationships:
        entity_id = str(rel.get('value'))
        if entity_id:
            entities[entity_id] = {
                'path': rel['path'],
                'type': rel['type'],
                'context': rel.get('context', {}),
                'role': rel.get('role'),
                'group_type': rel.get('group_type'),
                'group_context': rel.get('group_context'),
                'actor_role': rel.get('actor_role'),
                'permissions': rel.get('permissions'),
                'org_context': rel.get('org_context'),
                'project_context': rel.get('project_context')
            }
    return entities

def iterate_paths(json_obj, current_path="$"):
    """
    Recursively iterates through a JSON object yielding path-value pairs.
    
    Args:
        json_obj: JSON object to iterate
        current_path (str): Current path in the JSON (default: "$")
        
    Yields:
        tuple: (path, value) pairs using JSONPath notation
        
    Example paths:
        - $.data.suppliers[0].name
        - $.inventory_levels[1].quantity
    """
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            new_path = f"{current_path}.{key}" if current_path else key
            yield new_path, value
            if isinstance(value, (dict, list)):
                yield from iterate_paths(value, new_path)
                
    elif isinstance(json_obj, list):
        for i, value in enumerate(json_obj):
            new_path = f"{current_path}[{i}]"
            yield new_path, value
            if isinstance(value, (dict, list)):
                yield from iterate_paths(value, new_path)

def json_to_path_chunks(json_obj, entities=None):
    """
    Converts a JSON object into a list of path-based chunks.
    
    Args:
        json_obj: JSON object to chunk
        entities (dict, optional): Pre-extracted entity information
        
    Returns:
        list: Chunks with:
            - path: JSONPath location
            - value: Serialized value
            - context: Hierarchical context
            - display_names: Human-readable names
            - entities: Related entity information
    """
    chunks = []
    for path, value in iterate_paths(json_obj):
        chunk = {
            "path": path,
            "value": serialize_value(value),
            "context": extract_context(json_obj, path),
            "display_names": extract_display_names(json_obj, path),
            "entities": extract_entities_with_names(entities, path) if entities else {}
        }
        chunks.append(chunk)
    return chunks

def extract_key_value_pairs(chunk_text):
    """
    Extract searchable key-value pairs from structured JSON chunk.
    
    Args:
        chunk_text (str): JSON string containing chunk data
        
    Returns:
        dict: Extracted key-value pairs for indexing
    """
    pairs = {}
    try:
        data = json.loads(chunk_text)
        
        # Extract path information
        path = data.get('path', '')
        pairs['path'] = path
        path_parts = path.split('.')
        if len(path_parts) > 1:
            pairs['path_root'] = path_parts[0]
            pairs['path_leaf'] = path_parts[-1]
        
        # Extract value information
        if 'value' in data:
            value_data = data['value']
            pairs['value_type'] = value_data.get('type', 'unknown')
            
            if value_data['type'] == 'primitive':
                val = value_data.get('value')
                if isinstance(val, (int, float, str)):
                    pairs['value'] = str(val)
                    pairs['python_type'] = value_data.get('python_type', '')
                    
                    if isinstance(val, (int, float)):
                        pairs['value_exact'] = val
                        pairs[f"value_gt_{val-1}"] = True
                        pairs[f"value_lt_{val+1}"] = True
        
        # Extract temporal metadata
        if 'temporal_metadata' in data:
            temporal = data['temporal_metadata']
            for key in ['timestamp', 'year', 'month', 'day', 'weekday']:
                if key in temporal:
                    pairs[key] = temporal[key]
            
            # Add date-based filters
            if 'timestamp' in temporal:
                try:
                    date = parse_timestamp(temporal['timestamp'])
                    if date:
                        pairs['date_year'] = date.year
                        pairs['date_month'] = date.month
                        pairs['date_day'] = date.day
                        pairs['is_weekend'] = date.weekday() >= 5
                except Exception:
                    pass
        
        # Extract numeric metadata
        if 'numeric_metadata' in data:
            numeric = data['numeric_metadata']
            if 'value_exact' in numeric:
                pairs['numeric_value'] = numeric['value_exact']
                if 'value_range' in numeric:
                    range_data = numeric['value_range']
                    pairs['value_min'] = range_data.get('min', '')
                    pairs['value_max'] = range_data.get('max', '')
            if 'magnitude' in numeric:
                pairs['magnitude'] = numeric['magnitude']
        
        # Extract string metadata
        if 'string_metadata' in data:
            string_meta = data['string_metadata']
            pairs['string_length'] = string_meta.get('length', 0)
            pairs['has_numbers'] = string_meta.get('contains_numbers', False)
            pairs['has_urls'] = string_meta.get('contains_urls', False)
        
        # Extract context information
        if 'context' in data:
            for key, value in data['context'].items():
                if isinstance(value, (str, int, float)):
                    pairs[f"context_{key}"] = str(value)
        
        # Extract entity information
        if 'entities' in data:
            for entity_id, entity_data in data['entities'].items():
                entity_type = entity_data.get('type', 'unknown')
                pairs[f"entity_{entity_id}"] = entity_type
                pairs[f"has_entity_type_{entity_type}"] = True
                
                for attr_key, attr_value in entity_data.get('attributes', {}).items():
                    if isinstance(attr_value, (str, int, float)):
                        pairs[f"entity_{entity_id}_{attr_key}"] = str(attr_value)
    
    except json.JSONDecodeError as e:
        print(f"Error decoding chunk JSON: {e}")
    except Exception as e:
        print(f"Error extracting key-value pairs: {e}")
    
    return pairs

def process_json_document(json_obj):
    """
    Process a JSON document through the complete pipeline.
    
    Args:
        json_obj: Parsed JSON object
        
    Returns:
        tuple: (chunks, relationships)
    """
    # Step 1: Extract entities
    entities = extract_entities(json_obj)
    
    # Step 2: Create chunks with entity context
    chunks = json_to_path_chunks(json_obj, entities=entities)
    
    # Step 3: Detect relationships
    from app.relationships import detect_relationships
    relationships = detect_relationships(chunks)
    
    return chunks, relationships

def extract_context(json_obj, path, max_depth=3):
    """
    Extracts hierarchical context for a path in a JSON object.
    
    Args:
        json_obj: JSON object to extract context from
        path (str): JSONPath to extract context for
        max_depth (int): Maximum depth of parent context (default: 3)
        
    Returns:
        dict: Context information with parent paths as keys
        
    Note:
        Truncates large values to prevent context bloat.
    """
    context = {}
    
    # Split path into parts
    parts = path.replace('][', '.').replace('[', '.').replace(']', '').split('.')
    
    # Build increasingly specific paths and their values
    current = "$"
    context[current] = str(json_obj)[:100] + '...' if len(str(json_obj)) > 100 else str(json_obj)
    
    for i, part in enumerate(parts[1:], 1):  # Skip the root '$'
        try:
            # Build the path up to this point
            current_obj = json_obj
            path_parts = parts[1:i+1]  # Skip root
            
            # Navigate to current position
            for p in path_parts:
                if p.isdigit():  # Handle array indices
                    current_obj = current_obj[int(p)]
                else:
                    current_obj = current_obj[p]
            
            # Build the path string
            if i == 1:
                current = f"$.{part}"
            else:
                prev = parts[i-1]
                if prev.isdigit():
                    current = f"{current}[{prev}]"
                    if not part.isdigit():
                        current = f"{current}.{part}"
                else:
                    current = f"{current}.{part}"
            
            # Add to context with truncation for large values
            context[current] = str(current_obj)[:100] + '...' if len(str(current_obj)) > 100 else str(current_obj)
            
        except (KeyError, IndexError, TypeError):
            continue
            
        # Stop if we've reached max depth
        if i >= max_depth:
            break
    
    return context

def extract_display_names(json_obj, path):
    """
    Extracts human-readable names and labels from a JSON object.
    
    Args:
        json_obj: JSON object to extract names from
        path (str): JSONPath to the current location
        
    Returns:
        dict: Display names and labels found in the object
        
    Looks for:
        - name fields
        - titles
        - labels
        - display names
        - ID-name pairs
    """
    names = {}
    
    try:
        # Navigate to current position
        current_obj = json_obj
        parts = path.replace('][', '.').replace('[', '.').replace(']', '').split('.')
        
        for part in parts[1:]:  # Skip root
            if part.isdigit():
                current_obj = current_obj[int(part)]
            else:
                current_obj = current_obj[part]
        
        # Look for name fields in current object
        if isinstance(current_obj, dict):
            for key in ['name', 'title', 'label', 'display_name', 'description']:
                if key in current_obj:
                    names[key] = str(current_obj[key])
            
            # Look for ID-name pairs
            for key, value in current_obj.items():
                if '_id' in key.lower() and isinstance(value, str):
                    name_key = key.replace('_id', '_name')
                    if name_key in current_obj:
                        names[f"{key}_display"] = str(current_obj[name_key])
                        
    except (KeyError, IndexError, TypeError):
        pass
        
    return names

def extract_entities_with_names(entities, path):
    """
    Extracts entity information with display names for a path.
    
    Args:
        entities (dict): Entity registry
        path (str): JSONPath to match against
        
    Returns:
        dict: Entity information including:
            - type: Entity type
            - name: Display name
            - attributes: Entity attributes
            - relationships: Related entities
    """
    if not entities:
        return {}
        
    result = {}
    for entity_id, info in entities.items():
        if info['path'] == path:
            result[entity_id] = {
                'type': info['type'],
                'name': info.get('name', entity_id),
                'attributes': info.get('attributes', {}),
                'relationships': info.get('relationships', {})
            }
    return result
