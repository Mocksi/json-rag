import json
import math
import re
from datetime import datetime
from collections import defaultdict
from app.utils import parse_timestamp, classify_path
from app.models import FlexibleModel

def serialize_value(value):
    # (Same code as before for serialize_value)
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
    # Move the track_entity_relationships code here from original
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

def json_to_path_chunks(json_obj, current_path="$", entities=None, parent_context=None, path_contexts=None):
    from app import parsing  # If needed, careful with circular imports
    if path_contexts is None:
        path_contexts = {}
    chunks = []
    if isinstance(json_obj, dict):
        path_contexts[current_path] = {
            key: str(value) if not isinstance(value, (dict, list)) else "..."
            for key, value in json_obj.items()
        }
        chunks.append(create_enhanced_chunk(
            current_path,
            json_obj,
            context=path_contexts,
            entities=entities
        ))
        from app import models  # If needed
        for key, value in json_obj.items():
            from app import config
            from app import parsing
            # Keep your original logic
            new_path = f"{current_path}.{key}"
            chunks.extend(json_to_path_chunks(value, current_path=new_path, entities=entities, parent_context=json_obj, path_contexts=path_contexts))
    elif isinstance(json_obj, list):
        path_contexts[current_path + "[]"] = "..."
        for i, item in enumerate(json_obj):
            new_path = f"{current_path}[{i}]"
            chunks.extend(json_to_path_chunks(item, current_path=new_path, entities=entities, parent_context=parent_context, path_contexts=path_contexts))
    else:
        chunks.append(create_enhanced_chunk(
            current_path,
            json_obj,
            context=path_contexts,
            entities=entities
        ))
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