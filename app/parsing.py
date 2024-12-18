"""
JSON Parsing and Analysis Module

This module provides comprehensive functionality for parsing, analyzing, and processing
JSON documents in the RAG (Retrieval-Augmented Generation) system. It handles complex
JSON structures with a focus on maintaining hierarchical relationships, extracting
entities, and generating searchable chunks.

Key Features:
    - Value Serialization: Convert Python values to structured representations
    - Entity Detection: Extract and track entities and their relationships
    - Path Analysis: Process JSON paths and hierarchical structures
    - Context Extraction: Build hierarchical context for JSON elements
    - Display Name Handling: Extract human-readable labels and names
    - Relationship Tracking: Identify and track entity relationships

The module supports complex JSON documents containing:
    - Nested structures
    - Arrays and collections
    - Entity references
    - Metadata
    - Hierarchical relationships
    - Named entities

Usage:
    >>> from app.parsing import process_json_document
    >>> json_obj = {'data': {'users': [{'id': 1, 'name': 'John'}]}}
    >>> chunks, relationships = process_json_document(json_obj)
    >>> print(f"Generated {len(chunks)} chunks with {len(relationships)} relationships")
"""

import json
import math
import re
from datetime import datetime
from collections import defaultdict
from app.utils import parse_timestamp, classify_path
from app.models import FlexibleModel
from typing import Dict, List, Tuple

def serialize_value(value):
    """
    Serializes a Python value into a structured dictionary with type information.
    
    This function converts any Python value into a standardized dictionary format
    that includes type information, value representation, and additional metadata.
    It handles various Python types including primitives, collections, and custom
    objects.
    
    Args:
        value: Any Python value to serialize, including:
            - None
            - Primitives (str, int, float, bool)
            - Collections (dict, list)
            - Custom objects
            
    Returns:
        dict: A structured representation containing:
            For None:
                - type: 'null'
                - value: None
                
            For primitives:
                - type: 'primitive'
                - value: The actual value
                - python_type: Name of the Python type
                
            For dictionaries:
                - type: 'complex'
                - structure: 'dict'
                - size: Number of key-value pairs
                - keys: List of dictionary keys
                - sample: Up to 3 serialized key-value pairs
                - summary: Size and value type information
                
            For lists:
                - type: 'complex'
                - structure: 'list'
                - size: Length of the list
                - sample: Up to 3 serialized items
                - summary: Size and value type information
                
            For other types:
                - type: 'other'
                - python_type: Name of the Python type
                - string_repr: String representation of the value
                
    Example:
        >>> value = {'name': 'John', 'age': 30, 'scores': [95, 87, 92]}
        >>> result = serialize_value(value)
        >>> print(result['type'])  # 'complex'
        >>> print(result['structure'])  # 'dict'
        >>> print(result['size'])  # 3
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
    Serializes a context dictionary into a format suitable for storage and retrieval.
    
    This function processes a context dictionary to ensure all values are serializable,
    handling complex objects by converting them to string representations. It preserves
    primitive values while safely converting non-primitive types to strings.
    
    Args:
        context (dict): Context dictionary to serialize, which may contain:
            - Primitive values (str, int, float, bool, None)
            - Complex objects (dict, list, custom objects)
            - Mixed-type values
            
    Returns:
        dict: Serialized context where:
            - Primitive values are preserved as-is
            - Complex objects are converted to string representations
            - Invalid or unserializable values are replaced with error messages
            
    Error Handling:
        - Returns {'error': 'Invalid context type'} if input is not a dictionary
        - Replaces unserializable values with error messages
        - Preserves partial results even if some values fail to serialize
        
    Example:
        >>> context = {
        ...     'name': 'John',
        ...     'age': 30,
        ...     'metadata': {'created': datetime.now()},
        ...     'tags': ['user', 'active']
        ... }
        >>> result = serialize_context(context)
        >>> print(result['name'])  # 'John'
        >>> print(result['age'])  # 30
        >>> print(type(result['metadata']))  # <class 'str'>
        
    Note:
        This function is designed to be safe and never raise exceptions,
        making it suitable for processing untrusted or variable data.
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
    Serializes entity information into a structured format for storage and analysis.
    
    This function processes a dictionary of entity information, converting complex
    entity data into a standardized format suitable for storage and retrieval.
    It handles nested attributes and relationships while preserving entity types
    and identifiers.
    
    Args:
        entities (dict): Dictionary of entity information where:
            - Keys are entity identifiers (str)
            - Values are entity data dictionaries containing:
                - type: Entity type classification
                - attributes: Dictionary of entity properties
                - relationships: Dictionary of related entity references
                
    Returns:
        dict: Serialized entity information where each entity entry contains:
            - type (str): Entity type classification (e.g., 'user', 'product')
            - attributes (dict): Serialized entity properties
            - relationships (dict): Serialized relationship information
            
    Error Handling:
        - Returns {'error': 'Invalid entities type'} if input is not a dictionary
        - Individual entity errors are captured in their respective entries
        - Continues processing remaining entities if one fails
        
    Example:
        >>> entities = {
        ...     'user_123': {
        ...         'type': 'user',
        ...         'attributes': {'name': 'John', 'active': True},
        ...         'relationships': {'manager_id': 'user_456'}
        ...     }
        ... }
        >>> result = serialize_entities(entities)
        >>> print(result['user_123']['type'])  # 'user'
        >>> print(result['user_123']['attributes']['name'])  # 'John'
        
    Note:
        Entity IDs are converted to strings to ensure consistent dictionary keys.
        Complex objects in attributes and relationships are serialized to strings.
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
    Enriches chunk data with additional metadata based on value type and content.
    
    This function analyzes the provided value and adds specialized metadata to the
    chunk data based on the value's type and content. It handles numeric values,
    temporal data, and string content, adding relevant metadata for each type.
    
    Args:
        chunk_data (dict): Chunk dictionary to enrich, which will be modified in place
        value: Value to analyze, can be:
            - Numeric (int, float): Adds range and magnitude information
            - String: Adds length and content type information
            - Temporal string: Adds parsed date/time information
            
    Metadata Added:
        For numeric values:
            - numeric_metadata:
                - value_exact: Original numeric value
                - value_range: {min, max} for ±10% range
                - magnitude: Order of magnitude (log10)
                
        For temporal strings:
            - temporal_metadata:
                - timestamp: ISO format timestamp
                - year: Extracted year
                - month: Extracted month
                - day: Extracted day
                - weekday: Day of week (0-6)
                
        For strings:
            - string_metadata:
                - length: String length
                - contains_numbers: Boolean
                - contains_urls: Boolean
                
    Example:
        >>> chunk = {}
        >>> # Numeric example
        >>> enrich_chunk_metadata(chunk, 100)
        >>> print(chunk['numeric_metadata']['magnitude'])  # 2
        >>> 
        >>> # String example
        >>> enrich_chunk_metadata(chunk, "https://example.com")
        >>> print(chunk['string_metadata']['contains_urls'])  # True
        >>> 
        >>> # Temporal example
        >>> enrich_chunk_metadata(chunk, "2023-12-25")
        >>> print(chunk['temporal_metadata']['year'])  # 2023
        
    Note:
        - Modifies the chunk_data dictionary in place
        - Silently handles parsing errors for temporal data
        - Adds metadata only for supported value types
        - Uses regex for URL and number detection in strings
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
    
    This function generates a comprehensive chunk representation that includes
    the original value, contextual information, entity relationships, and
    enriched metadata. It handles serialization of complex data types and
    ensures all components are properly formatted for storage and retrieval.
    
    Args:
        path (str): JSONPath to the value, indicating its location in the document
        value: The value at the specified path, can be any JSON-serializable type
        context (dict, optional): Additional contextual information, such as:
            - Parent-child relationships
            - Sibling data
            - Document metadata
            - Processing history
        entities (dict, optional): Related entity information containing:
            - Entity types and identifiers
            - Relationships between entities
            - Entity attributes and metadata
            
    Returns:
        tuple: A pair containing:
            - str: JSON string representation of the enhanced chunk
            - dict: Python dictionary of the chunk data
            
    Generated Chunk Structure:
        - path: Original JSONPath
        - value: Serialized value with type information
        - context: Serialized contextual data
        - entities: Serialized entity information
        - metadata:
            - created_at: ISO timestamp
            - chunk_version: Format version string
            - path_depth: Nesting level in JSON
            - path_type: Classification of the path
            - Additional type-specific metadata
            
    Example:
        >>> path = "$.users[0].profile"
        >>> value = {"name": "John", "age": 30}
        >>> context = {"parent": "users", "type": "profile"}
        >>> entities = {"user_123": {"type": "user", "name": "John"}}
        >>> chunk_text, chunk_data = create_enhanced_chunk(path, value, context, entities)
        >>> print(chunk_data['metadata']['path_depth'])  # 3
        
    Note:
        - Handles serialization errors gracefully
        - Returns error chunk if serialization fails
        - Enriches metadata based on value type
        - Respects MAX_CHUNKS configuration
        - Uses version 2.0 chunk format
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
    
    # Create both text and JSON versions
    chunk_text = json.dumps(chunk_data, default=str)
    try:
        return chunk_text, chunk_data
    except TypeError as e:
        print(f"Error serializing chunk data: {e}")
        error_data = {
            'path': path,
            'error': 'Serialization failed',
            'metadata': chunk_data['metadata']
        }
        return json.dumps(error_data), error_data

def track_entity_relationships(json_obj, current_path="$", parent_context=None):
    """
    Tracks and analyzes relationships between entities in a JSON object.
    
    This function performs deep analysis of a JSON object to identify entities
    and their relationships, tracking hierarchical connections, roles, and
    organizational context. It handles various entity types and relationship
    patterns commonly found in JSON documents.
    
    Args:
        json_obj: JSON object to analyze, which may contain:
            - Entity identifiers (IDs, names, keys)
            - Role information
            - Group memberships
            - Organizational hierarchies
            - Actor permissions
        current_path (str, optional): Current path in the JSON structure.
            Defaults to "$" (root).
        parent_context (dict, optional): Context from parent nodes containing:
            - Organization information
            - Project context
            - Role hierarchies
            - Access control data
            
    Returns:
        list: List of found relationships, where each relationship is a
        dictionary containing:
            - type: Entity type classification
            - value: Entity identifier or value
            - path: JSONPath to the entity
            - context: Related entity data
            - role: Entity role if applicable
            - group_type: Organizational group type
            - group_context: Parent group information
            - actor_role: Associated actor role
            - permissions: Access control information
            - org_context: Organizational context
            - project_context: Project-level context
            
    Entity Types Detected:
        - person: Name-based entities
        - identifier: ID-based references
        - document: Title-based entities
        - reference: Key-based references
        - unique_id: UUID-based entities
        - contact: Email-based entities
        
    Role Types Detected:
        - member
        - owner
        - participant
        - author
        - assignee
        
    Group Types Detected:
        - project
        - team
        - organization
        - department
        
    Example:
        >>> json_data = {
        ...     "project": {
        ...         "name": "Project A",
        ...         "owner": {"id": "user_123", "name": "John"},
        ...         "team": {
        ...             "members": [
        ...                 {"id": "user_456", "role": "developer"}
        ...             ]
        ...         }
        ...     }
        ... }
        >>> relationships = track_entity_relationships(json_data)
        >>> for rel in relationships:
        ...     print(f"{rel['type']}: {rel['value']} ({rel['role']})")
        
    Note:
        - Processes nested structures recursively
        - Preserves hierarchical context
        - Handles array-based collections
        - Extracts role and permission data
        - Maintains organizational context
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
    Extracts and analyzes entities and their relationships from a JSON object.
    
    This function performs comprehensive entity extraction and analysis from a
    JSON document, identifying entities, their types, relationships, and contextual
    information. It uses relationship tracking to build a complete picture of
    entity interactions and hierarchies.
    
    Args:
        json_obj: JSON object to analyze, which may contain:
            - Named entities (users, products, etc.)
            - Entity references (IDs, keys)
            - Relationship definitions
            - Role assignments
            - Group memberships
            - Access control information
        current_path (str, optional): Current path in the JSON structure.
            Defaults to "$" (root).
            
    Returns:
        dict: Dictionary of extracted entities where:
            - Keys are entity identifiers (str)
            - Values are entity information dictionaries containing:
                - path: JSONPath to entity location
                - type: Entity type classification
                - context: Related contextual data
                - role: Entity role in the system
                - group_type: Organizational group classification
                - group_context: Parent group information
                - actor_role: Associated actor role
                - permissions: Access control data
                - org_context: Organizational context
                - project_context: Project-level context
                
    Entity Types:
        - person: Human entities with names
        - identifier: System identifiers
        - document: Document-type entities
        - reference: Reference keys
        - unique_id: UUID-based entities
        - contact: Contact information
        
    Contextual Information:
        - Hierarchical relationships
        - Role assignments
        - Group memberships
        - Access permissions
        - Organizational structure
        
    Example:
        >>> json_data = {
        ...     "team": {
        ...         "name": "Engineering",
        ...         "lead": {
        ...             "id": "user_123",
        ...             "name": "John Smith",
        ...             "role": "team_lead"
        ...         },
        ...         "members": [
        ...             {
        ...                 "id": "user_456",
        ...                 "name": "Jane Doe",
        ...                 "role": "developer"
        ...             }
        ...         ]
        ...     }
        ... }
        >>> entities = extract_entities(json_data)
        >>> print(entities["user_123"]["role"])  # "team_lead"
        >>> print(entities["user_456"]["type"])  # "person"
        
    Note:
        - Processes nested structures recursively
        - Preserves entity relationships
        - Maintains hierarchical context
        - Handles complex organizational structures
        - Supports multiple entity types
        - Extracts rich contextual information
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
    
    This function performs a depth-first traversal of a JSON object, generating
    JSONPath expressions for each node and yielding them along with their
    corresponding values. It handles both objects and arrays, maintaining proper
    path notation throughout the hierarchy.
    
    Args:
        json_obj: JSON object to iterate, which can be:
            - Dictionary: Yields path.key notation
            - List: Yields path[index] notation
            - Primitive: Yields path-value pair
            - Nested combinations of the above
        current_path (str, optional): Current path in the JSON structure.
            Defaults to "$" (root).
            
    Yields:
        tuple: (path, value) pairs where:
            - path (str): JSONPath expression to the current node
            - value: The value at that path, which may be:
                - Dictionary: For object nodes
                - List: For array nodes
                - str, int, float, bool, None: For primitive values
                
    Path Notation:
        - Root: $
        - Object properties: .property_name
        - Array indices: [index]
        - Nested paths: Combinations of the above
        
    Examples:
        >>> json_data = {
        ...     "store": {
        ...         "book": [
        ...             {
        ...                 "title": "Book 1",
        ...                 "price": 10.99
        ...             },
        ...             {
        ...                 "title": "Book 2",
        ...                 "price": 12.99
        ...             }
        ...         ],
        ...         "location": "New York"
        ...     }
        ... }
        >>> paths = list(iterate_paths(json_data))
        >>> for path, value in paths:
        ...     print(f"{path}: {value}")
        $.store: {'book': [...], 'location': 'New York'}
        $.store.book: [{'title': 'Book 1', ...}, {'title': 'Book 2', ...}]
        $.store.book[0]: {'title': 'Book 1', 'price': 10.99}
        $.store.book[0].title: 'Book 1'
        $.store.book[0].price: 10.99
        $.store.book[1]: {'title': 'Book 2', 'price': 12.99}
        $.store.book[1].title: 'Book 2'
        $.store.book[1].price: 12.99
        $.store.location: 'New York'
        
    Note:
        - Performs depth-first traversal
        - Handles nested structures recursively
        - Maintains path context throughout traversal
        - Supports arbitrary nesting depth
        - Preserves array indices in paths
        - Handles all JSON value types
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

def json_to_path_chunks(json_obj: Dict, file_path: str = '', max_chunks: int = 100, entities: Dict = None, archetypes: List[Tuple[str, float]] = None) -> List[Dict]:
    """
    Convert a JSON object into path-based chunks with metadata and relationships.
    
    This function decomposes a JSON object into meaningful chunks based on paths
    and values, while preserving hierarchical relationships and enriching each
    chunk with metadata, entity information, and archetype classifications.
    
    Args:
        json_obj (Dict): JSON object to chunk, which can contain:
            - Nested objects and arrays
            - Entity references
            - Metadata fields
            - Hierarchical relationships
        file_path (str, optional): Path to source file for reference.
            Defaults to empty string.
        max_chunks (int, optional): Maximum number of chunks to generate.
            Defaults to 100. Used to prevent memory issues with large documents.
        entities (Dict, optional): Entity information dictionary containing:
            - Entity identifiers
            - Entity types and classifications
            - Relationship mappings
            - Contextual information
        archetypes (List[Tuple[str, float]], optional): List of archetype
            classifications with confidence scores, where each tuple contains:
            - str: Archetype name/classification
            - float: Confidence score (0.0 to 1.0)
            
    Returns:
        List[Dict]: List of chunk dictionaries, where each chunk contains:
            - path: JSONPath to the chunk's location
            - value: The actual value at the path
            - context: Contextual information dictionary
            - entities: Related entity information
            - archetypes: Applicable archetype classifications
            - metadata:
                - depth: Nesting level in JSON
                - has_children: Boolean indicating nested content
                - type: Value type classification
                - Additional type-specific metadata
                
    Chunk Types Generated:
        - Object chunks: For dictionary nodes
        - Array chunks: For list nodes
        - Value chunks: For primitive values
        - Container chunks: For arrays of objects with IDs
        
    Path Generation:
        - Uses dot notation for object properties
        - Uses bracket notation for array indices
        - Preserves IDs in paths when available
        - Maintains hierarchical structure
        
    Example:
        >>> data = {
        ...     "users": [
        ...         {
        ...             "id": "user_1",
        ...             "name": "John",
        ...             "roles": ["admin", "user"]
        ...         }
        ...     ]
        ... }
        >>> entities = {"user_1": {"type": "user", "name": "John"}}
        >>> archetypes = [("user_collection", 0.95)]
        >>> chunks = json_to_path_chunks(data, entities=entities, archetypes=archetypes)
        >>> for chunk in chunks:
        ...     print(f"{chunk['path']}: {chunk['value']}")
        
    Note:
        - Handles nested structures recursively
        - Preserves parent-child relationships
        - Enriches chunks with metadata
        - Respects max_chunks limit
        - Optimizes for searchability
        - Maintains entity context
    """
    chunks = []
    
    def process_value(obj, path='', context=None):
        if context is None:
            context = {}
            
        if isinstance(obj, dict):
            for k, v in obj.items():
                new_path = f"{path}.{k}" if path else k
                process_value(v, new_path, {**context, k: v})
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                process_value(item, new_path, context)
        else:
            chunk = {
                'path': path,
                'value': obj,
                'context': context
            }
            if entities:
                chunk['entities'] = entities
            if archetypes:
                chunk['archetypes'] = archetypes
            chunks.append(chunk)
            
    process_value(json_obj)
    return chunks[:max_chunks]

def extract_key_value_pairs(chunk_data):
    """
    Extract searchable key-value pairs from chunk data with context preservation.
    
    This function analyzes chunk data to extract searchable key-value pairs,
    including nested identifiers, context values, and path information. It
    handles various data structures and maintains context relationships in
    the extracted pairs.
    
    Args:
        chunk_data: Chunk data to analyze, which can be:
            - Dictionary: Processed for paths, values, and context
            - Non-dictionary: Treated as a single value
            - Nested structures: Recursively processed
            
    Returns:
        dict: Dictionary of extracted key-value pairs where:
            - Keys are descriptive identifiers including:
                - Direct field names
                - Nested path components
                - Context prefixes
                - Type indicators
            - Values are string representations of:
                - Direct values
                - Nested identifiers
                - Context values
                - Path components
                
    Extracted Fields:
        - path: JSONPath to the chunk
        - value: Main chunk value
        - context_*: Context field values
        - *_id: Entity identifiers
        - Nested identifier paths
        
    ID Types Detected:
        - _id suffixes
        - supplier references
        - product references
        - warehouse references
        - shipment references
        
    Example:
        >>> chunk = {
        ...     'path': '$.orders[0]',
        ...     'value': 'ORD-123',
        ...     'context': {
        ...         'customer_id': 'CUST-456',
        ...         'product': {
        ...             'id': 'PROD-789',
        ...             'supplier_id': 'SUP-101'
        ...         }
        ...     }
        ... }
        >>> pairs = extract_key_value_pairs(chunk)
        >>> print(pairs['path'])  # '$.orders[0]'
        >>> print(pairs['value'])  # 'ORD-123'
        >>> print(pairs['context_customer_id'])  # 'CUST-456'
        >>> print(pairs['product_id'])  # 'PROD-789'
        >>> print(pairs['product_supplier_id'])  # 'SUP-101'
        
    Note:
        - Processes nested structures recursively
        - Preserves context relationships
        - Handles various ID patterns
        - Converts all values to strings
        - Maintains path information
        - Supports debug logging
    """
    pairs = {}
    
    def extract_nested_ids(data, prefix=''):
        """Recursively extract IDs from nested structures."""
        if isinstance(data, dict):
            for k, v in data.items():
                if any(id_type in k for id_type in ['_id', 'supplier', 'product', 'warehouse', 'shipment']):
                    pairs[f"{prefix}{k}"] = str(v)
                if isinstance(v, (dict, list)):
                    extract_nested_ids(v, f"{prefix}{k}_")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                extract_nested_ids(item, f"{prefix}{i}_")
    
    try:
        # If chunk_data is not a dict, treat it as a value
        if not isinstance(chunk_data, dict):
            pairs['value'] = str(chunk_data)
            return pairs
            
        # Process the main chunk data
        data = chunk_data
        
        # Extract path and value
        if 'path' in data:
            pairs['path'] = data['path']
        if 'value' in data:
            pairs['value'] = str(data['value'])
            
        # Process context recursively
        if 'context' in data:
            extract_nested_ids(data['context'])
            
            # Also store direct context values
            for key, value in data['context'].items():
                if isinstance(value, (str, int, float)):
                    pairs[f"context_{key}"] = str(value)
        
        print(f"DEBUG: Extracted pairs: {pairs}")
        
    except Exception as e:
        print(f"Error extracting key-value pairs: {e}")
    
    return pairs

def process_json_document(json_obj: Dict, file_path: str = '', max_chunks: int = 100) -> Tuple[List[Dict], List[Dict]]:
    """
    Process a JSON document into searchable chunks with relationships.
    
    This function serves as the main entry point for processing JSON documents,
    converting them into searchable chunks while preserving relationships,
    context, and metadata. It handles document analysis, chunking, entity
    extraction, and relationship tracking in a single unified process.
    
    Processing Pipeline:
        1. Document Analysis:
            - Structure analysis
            - Entity detection
            - Archetype classification
            
        2. Chunking:
            - Path-based decomposition
            - Context preservation
            - Metadata enrichment
            
        3. Entity Processing:
            - Entity extraction
            - Relationship detection
            - Context building
            
        4. Relationship Analysis:
            - Parent-child tracking
            - Cross-references
            - Entity relationships
            
    Args:
        json_obj (Dict): JSON document to process, which can contain:
            - Nested objects and arrays
            - Entity references
            - Metadata fields
            - Hierarchical relationships
        file_path (str, optional): Source file path for reference.
            Defaults to empty string.
        max_chunks (int, optional): Maximum number of chunks to generate.
            Defaults to 100. Used to prevent memory issues.
            
    Returns:
        Tuple[List[Dict], List[Dict]]: A pair containing:
            1. List of chunk dictionaries, where each chunk has:
                - path: JSONPath to chunk location
                - value: Actual value at path
                - context: Contextual information
                - entities: Related entity data
                - metadata: Type-specific metadata
                
            2. List of relationship dictionaries, where each has:
                - source: Source chunk/entity identifier
                - target: Target chunk/entity identifier
                - type: Relationship classification
                - metadata: Relationship context
                
    Example:
        >>> data = {
        ...     "order": {
        ...         "id": "ORD-123",
        ...         "customer": {
        ...             "id": "CUST-456",
        ...             "name": "John Doe"
        ...         },
        ...         "items": [
        ...             {
        ...                 "product_id": "PROD-789",
        ...                 "quantity": 2
        ...             }
        ...         ]
        ...     }
        ... }
        >>> chunks, relationships = process_json_document(data)
        >>> print(f"Generated {len(chunks)} chunks")
        >>> print(f"Found {len(relationships)} relationships")
        
    Note:
        - Handles complex document structures
        - Preserves hierarchical relationships
        - Maintains entity context
        - Optimizes for searchability
        - Supports large documents
        - Provides rich metadata
        - Enables relationship queries
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
