"""
Smart Archetype Detection module for identifying JSON data patterns.
Analyzes structure and content to classify JSON documents into common patterns.
"""

from datetime import datetime
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any, Set
import json

class ArchetypeDetector:
    """
    Detects common JSON data patterns and structures.
    
    Patterns Detected:
    - Event Logs: Timestamped entries with actions/states
    - API Responses: Standard REST patterns with metadata
    - Metric Data: Numerical time series or measurements
    - Entity Collections: Lists of business objects
    - State Machines: Objects with status transitions
    - Configuration: Setting key-value pairs
    """
    
    def __init__(self):
        self.patterns = defaultdict(int)
        self.field_frequencies = defaultdict(int)
        self.type_frequencies = defaultdict(int)
        self.value_patterns = defaultdict(set)
        self.temporal_fields = set()
        self.numerical_fields = set()
        
    def detect_dataset_patterns(self, structures: List[Dict]) -> Dict:
        """Analyze entire dataset to detect common patterns and archetypes."""
        print("\nAnalyzing dataset-wide patterns...")
        
        # Collect patterns across all structures
        for data in structures:
            self._analyze_structure(data)
            
        # Core JSON archetypes that inform data usage patterns
        dataset_archetypes = {
            'record': self._detect_record_pattern(),      # Individual entities with properties
            'event': self._detect_event_pattern(),        # Time-based occurrences with context
            'collection': self._detect_collection_pattern(), # Lists of similar items
            'reference': self._detect_reference_pattern(), # Objects that primarily link to others
            'metric': self._detect_metric_pattern(),      # Numerical measurements/stats
            'state': self._detect_state_pattern(),        # Current status/configuration
            'document': self._detect_document_pattern()    # Nested, hierarchical content
        }
        
        print("\nDataset analysis results:")
        for archetype, details in dataset_archetypes.items():
            if details['confidence'] > 0.3:
                print(f"- {archetype}: {details['confidence']:.2f} confidence")
                print(f"  Common fields: {', '.join(details['common_fields'])}")
        
        return dataset_archetypes
    
    def _detect_record_pattern(self) -> Dict:
        """Detect record patterns - objects representing distinct entities."""
        record_indicators = {
            'id', 'uuid', 'name', 'type', 'code', 'key',
            'properties', 'attributes', 'metadata'
        }
        found = set(self.field_frequencies.keys()) & record_indicators
        
        # Check for strong identity fields
        identity_fields = {'id', 'uuid'}
        has_identity = bool(found & identity_fields)
        
        # Check for descriptive fields
        descriptive_fields = {'name', 'type', 'code'}
        has_description = bool(found & descriptive_fields)
        
        # Check for stable value patterns
        has_stable_values = any(len(values) == 1 
                              for values in self.value_patterns.values())
        
        confidence = len(found) / len(record_indicators)
        if has_identity and has_description:
            confidence *= 2.0  # Boost confidence if we have both identity and description
        elif has_identity and has_stable_values:
            confidence *= 1.5  # Boost if we have identity and stable values
        elif len(found) < 2:
            confidence *= 0.25  # Heavily penalize having just one indicator
        
        return {
            'confidence': confidence,
            'common_fields': list(found),
            'pattern': 'entity_definition',
            'usage': 'Represents distinct objects with stable properties'
        }
    
    def _detect_event_pattern(self) -> Dict:
        """Detect event patterns - time-based occurrences."""
        event_indicators = {
            'timestamp', 'time', 'date', 'created_at', 'updated_at',
            'occurred_at', 'status', 'state', 'type', 'action',
            'order_date', 'delivery_date', 'eta'
        }
        found = set(self.field_frequencies.keys()) & event_indicators
        
        # Check temporal fields we detected
        has_temporal = bool(self.temporal_fields)
        
        # Check status/state values
        has_status = bool(self.value_patterns.get('status') or 
                         self.value_patterns.get('state'))
        
        confidence = len(found) / len(event_indicators)
        if has_temporal and has_status:
            confidence *= 2.0  # Boost confidence if we have both time and status
        elif has_temporal:
            confidence *= 1.5  # Boost confidence if we have time fields
        
        return {
            'confidence': confidence,
            'common_fields': list(found | self.temporal_fields),  # Include detected time fields
            'pattern': 'temporal_occurrence',
            'usage': 'Represents things that happen at specific times'
        }
    
    def _detect_collection_pattern(self) -> Dict:
        """Detect collection patterns - lists of similar items."""
        # Look for array structures and plural names
        array_fields = set()
        for type_info, count in self.type_frequencies.items():
            field, type_name = type_info.split(':')
            if type_name == 'list' and count > 0:
                array_fields.add(field)
                # Also check if field name is plural
                if field.endswith('s'):  # Simple plural check
                    array_fields.add(field)
        
        # Collection confidence based on number of array fields and their usage
        confidence = min(len(array_fields) / 2, 1.0)
        
        return {
            'confidence': confidence,
            'common_fields': list(array_fields),
            'pattern': 'item_collection',
            'usage': 'Represents groups of related items'
        }
    
    def _detect_reference_pattern(self) -> Dict:
        """Detect reference patterns - objects that link to others."""
        # Look for fields that reference other objects
        reference_indicators = [f for f in self.field_frequencies.keys() 
                              if f.endswith('_id') or f.endswith('_ref')]
        return {
            'confidence': min(len(reference_indicators) / 2, 1.0),
            'common_fields': reference_indicators,
            'pattern': 'object_reference',
            'usage': 'Represents relationships between objects'
        }
    
    def _detect_metric_pattern(self) -> Dict:
        """Detect metric patterns - numerical measurements."""
        metric_indicators = {
            'value', 'amount', 'count', 'total', 'quantity',
            'rate', 'percentage', 'score', 'ratio', 'stock',
            'reserved', 'available', 'balance', 'level'
        }
        found = set(self.field_frequencies.keys()) & metric_indicators
        
        # Check numerical fields we detected
        has_numbers = bool(self.numerical_fields)
        
        # Check for quantity-specific fields
        quantity_fields = {'quantity', 'stock', 'reserved', 'available'}
        has_quantities = bool(found & quantity_fields)
        
        confidence = len(found) / len(metric_indicators)
        if has_numbers and has_quantities:
            confidence *= 2.0  # Boost confidence if we have both numbers and quantity fields
        elif has_numbers:
            confidence *= 1.5  # Boost confidence if we have numerical fields
        
        return {
            'confidence': confidence,
            'common_fields': list(found | self.numerical_fields),  # Include detected number fields
            'pattern': 'numerical_measurement',
            'usage': 'Represents quantitative measurements'
        }
    
    def _detect_state_pattern(self) -> Dict:
        """Detect state patterns - current status/configuration."""
        state_indicators = {
            'status', 'state', 'enabled', 'active', 'config',
            'settings', 'preferences', 'mode', 'flags'
        }
        found = set(self.field_frequencies.keys()) & state_indicators
        return {
            'confidence': len(found) / len(state_indicators),
            'common_fields': list(found),
            'pattern': 'current_state',
            'usage': 'Represents configuration or status'
        }
    
    def _detect_document_pattern(self) -> Dict:
        """Detect document patterns - nested, hierarchical content."""
        # Look for deep nesting and content-related fields
        doc_indicators = {
            'content', 'data', 'body', 'text', 'description',
            'details', 'sections', 'elements'
        }
        found = set(self.field_frequencies.keys()) & doc_indicators
        nested_depth = self._calculate_max_nesting()
        confidence = (len(found) / len(doc_indicators) + (nested_depth > 2)) / 2
        return {
            'confidence': confidence,
            'common_fields': list(found),
            'pattern': 'hierarchical_content',
            'usage': 'Represents structured document content'
        }
    
    def _find_array_patterns(self) -> Set[str]:
        """Find fields that contain arrays of similar items."""
        array_fields = set()
        for type_info, count in self.type_frequencies.items():
            field, type_name = type_info.split(':')
            if type_name == 'list' and count > 1:
                array_fields.add(field)
        return array_fields
    
    def _calculate_max_nesting(self) -> int:
        """Calculate the maximum nesting depth in the data."""
        max_depth = 0
        for path in self.field_frequencies.keys():
            depth = path.count('.')
            max_depth = max(max_depth, depth)
        return max_depth

    def analyze_structure(self, json_obj: dict) -> Dict:
        """
        Perform detailed structural analysis of a JSON object.
        
        Args:
            json_obj: JSON object to analyze
            
        Returns:
            dict: Structural analysis including:
                - depth: Maximum nesting depth
                - array_paths: Paths containing arrays
                - value_types: Distribution of value types
                - field_patterns: Common field naming patterns
        """
        analysis = {
            'depth': 0,
            'array_paths': [],
            'value_types': Counter(),
            'field_patterns': Counter()
        }
        
        def analyze_recursive(obj, path='', depth=0):
            analysis['depth'] = max(analysis['depth'], depth)
            
            if isinstance(obj, dict):
                for k, v in obj.items():
                    # Track field naming patterns
                    if '_' in k:
                        pattern = k.split('_')[-1]
                        analysis['field_patterns'][pattern] += 1
                    
                    new_path = f"{path}.{k}" if path else k
                    analyze_recursive(v, new_path, depth + 1)
                    
            elif isinstance(obj, list):
                analysis['array_paths'].append(path)
                for i, item in enumerate(obj):
                    analyze_recursive(item, f"{path}[{i}]", depth + 1)
                    
            else:
                analysis['value_types'][type(obj).__name__] += 1
        
        analyze_recursive(json_obj)
        return analysis

    def detect_archetype(self, data: Dict) -> Tuple[str, float]:
        """
        Detect the archetype of a JSON object and return type and confidence.
        
        Args:
            data: JSON object to analyze
            
        Returns:
            tuple: (archetype_type, confidence_score)
            
        Example:
            >>> detector = ArchetypeDetector()
            >>> data = {
            ...     "timestamp": "2024-03-16T10:30:00Z",
            ...     "event": "user_login",
            ...     "status": "success"
            ... }
            >>> archetype, confidence = detector.detect_archetype(data)
            >>> print(f"Detected {archetype} with {confidence:.2f} confidence")
            Detected event_log with 0.90 confidence
        """
        scores = {
            'event_log': self._score_event_log(data),
            'metric_data': self._score_metric_data(data),
            'api_response': self._score_api_response(data),
            'state_machine': self._score_state_machine(data),
            'entity_collection': self._score_entity_collection(data)
        }
        
        best_archetype = max(scores.items(), key=lambda x: x[1])
        return best_archetype[0], best_archetype[1]

    def _score_event_log(self, data: Dict) -> float:
        """Score likelihood of being an event log entry."""
        score = 0.0
        
        # Check for timestamp field
        has_timestamp = False
        for key, value in data.items():
            if any(re.search(pattern, str(value)) for pattern in self.timestamp_patterns):
                has_timestamp = True
                score += 0.4
                break
        
        # Check for event-related fields
        event_keywords = {'event', 'action', 'type', 'status', 'level', 'severity'}
        if any(key in event_keywords for key in data.keys()):
            score += 0.3
        
        # Check for sequential indicators
        sequence_keywords = {'sequence', 'id', 'index', 'order'}
        if any(key in sequence_keywords for key in data.keys()):
            score += 0.2
        
        # Must have timestamp for minimum viable event log
        return score if has_timestamp else 0.0

    def _score_metric_data(self, data: Dict) -> float:
        """Score likelihood of being metric data."""
        score = 0.0
        
        # Check for numerical values
        numerical_values = sum(1 for v in data.values() 
                             if isinstance(v, (int, float)))
        if numerical_values > 0:
            score += 0.3 * (numerical_values / len(data))
        
        # Check for metric-related fields
        metric_keywords = {'value', 'metric', 'measure', 'count', 'rate', 'total'}
        if any(key in metric_keywords for key in data.keys()):
            score += 0.3
        
        # Check for time series indicators
        if any(re.search(pattern, str(v)) for k, v in data.items() 
               for pattern in self.timestamp_patterns):
            score += 0.2
        
        # Check for units or dimensions
        unit_keywords = {'unit', 'dimension', 'scale', 'period'}
        if any(key in unit_keywords for key in data.keys()):
            score += 0.2
            
        return score

    def _score_api_response(self, data: Dict) -> float:
        """Score likelihood of being an API response."""
        score = 0.0
        
        # Check for common API response fields
        api_keywords = {'status', 'code', 'message', 'error', 'data', 'response'}
        matches = sum(1 for key in data.keys() if key in api_keywords)
        if matches > 0:
            score += 0.3 * (matches / len(api_keywords))
        
        # Check for HTTP status codes
        for value in data.values():
            if isinstance(value, (int, str)) and \
               any(re.match(pattern, str(value)) for pattern in self.http_status_patterns):
                score += 0.4
                break
        
        # Check for response metadata
        metadata_keywords = {'headers', 'timestamp', 'version', 'api'}
        if any(key in metadata_keywords for key in data.keys()):
            score += 0.3
            
        return score

    def _score_state_machine(self, data: Dict) -> float:
        """Score likelihood of being a state machine entry."""
        score = 0.0
        
        # Check for state-related fields
        state_keywords = {'state', 'status', 'current', 'previous', 'next'}
        if any(key in state_keywords for key in data.keys()):
            score += 0.4
        
        # Check for transition-related fields
        transition_keywords = {'transition', 'from', 'to', 'action'}
        if any(key in transition_keywords for key in data.keys()):
            score += 0.3
        
        # Check for timestamps (state changes usually logged)
        if any(re.search(pattern, str(v)) for k, v in data.items() 
               for pattern in self.timestamp_patterns):
            score += 0.2
        
        # Check for validation or rule fields
        rule_keywords = {'valid', 'allowed', 'rules', 'constraints'}
        if any(key in rule_keywords for key in data.keys()):
            score += 0.1
            
        return score

    def _score_entity_collection(self, data: Dict) -> float:
        """Score likelihood of being an entity collection."""
        score = 0.0
        
        # Check for identity fields
        id_keywords = {'id', 'uuid', 'key', 'name'}
        if any(key in id_keywords for key in data.keys()):
            score += 0.3
        
        # Check for relationship fields
        relationship_keywords = {'parent', 'child', 'related', 'owner', 'group'}
        if any(key in relationship_keywords for key in data.keys()):
            score += 0.3
        
        # Check for metadata fields
        metadata_keywords = {'type', 'category', 'class', 'attributes'}
        if any(key in metadata_keywords for key in data.keys()):
            score += 0.2
        
        # Check for collection indicators
        collection_keywords = {'items', 'elements', 'entries', 'records'}
        if any(key in collection_keywords for key in data.keys()):
            score += 0.2
            
        return score

    def get_matching_archetype(self, value: Dict, path: str) -> Dict:
        """Get the best matching archetype for a specific value based on dataset patterns."""
        best_match = None
        highest_confidence = 0
        
        # Check value against known patterns
        for archetype, details in self.dataset_archetypes.items():
            if details['confidence'] > highest_confidence:
                # Check if value matches this archetype's pattern
                if self._matches_pattern(value, details['common_fields']):
                    best_match = {
                        'type': archetype,
                        'confidence': details['confidence'],
                        'pattern': details['pattern']
                    }
                    highest_confidence = details['confidence']
        
        return best_match
    
    def _matches_pattern(self, value: Dict, pattern_fields: List[str]) -> bool:
        """Check if a value matches a pattern's expected fields."""
        value_fields = set(value.keys())
        return bool(value_fields & set(pattern_fields))

    def _analyze_structure(self, data: Any, path: str = "") -> None:
        """Recursively analyze data structure and collect patterns."""
        if isinstance(data, dict):
            # Record field names and their types
            for key, value in data.items():
                self.field_frequencies[key] += 1
                
                # Track both field name and type
                type_name = type(value).__name__
                self.type_frequencies[f"{key}:{type_name}"] += 1
                
                # Track actual values for specific fields
                if key in {'status', 'state', 'type'}:
                    self.value_patterns[key].add(str(value))
                
                # Track date/time strings
                if isinstance(value, str):
                    if any(pattern in value.lower() for pattern in 
                          ['date', 'time', '2024-', '2023-']):
                        self.temporal_fields.add(key)
                
                # Track numerical fields
                if isinstance(value, (int, float)):
                    self.numerical_fields.add(key)
                
                # Recursively analyze nested structures
                if isinstance(value, (dict, list)):
                    new_path = f"{path}.{key}" if path else key
                    self._analyze_structure(value, new_path)
                    
        elif isinstance(data, list) and data:
            # For lists, analyze the first few items as representative samples
            for item in data[:3]:  # Limit to first 3 items for efficiency
                if isinstance(item, (dict, list)):
                    new_path = f"{path}[*]"
                    self._analyze_structure(item, new_path)

    def analyze_patterns(self, data: Dict) -> None:
        """
        Analyze patterns in a single JSON document.
        
        Args:
            data: JSON object to analyze
        """
        # Analyze structure recursively
        self._analyze_structure(data)
        
        # Update pattern frequencies
        if self._detect_record_pattern()['confidence'] > 0.5:
            self.patterns['record'] += 1
        if self._detect_event_pattern()['confidence'] > 0.5:
            self.patterns['event'] += 1
        if self._detect_metric_pattern()['confidence'] > 0.5:
            self.patterns['metric'] += 1
        if self._detect_state_pattern()['confidence'] > 0.5:
            self.patterns['state'] += 1
        if self._detect_collection_pattern()['confidence'] > 0.5:
            self.patterns['collection'] += 1

    def _detect_transaction_pattern(self) -> Dict:
        """Detect transaction patterns - business events with multiple entities."""
        transaction_indicators = {
            'order', 'payment', 'transaction', 'invoice',
            'purchase', 'sale', 'transfer', 'shipment'
        }
        found = set(self.field_frequencies.keys()) & transaction_indicators
        
        # Check for transaction-specific fields
        has_amount = bool(self.numerical_fields & {'amount', 'total', 'price'})
        has_parties = bool(set(self.field_frequencies.keys()) & 
                          {'buyer', 'seller', 'customer', 'supplier'})
        
        confidence = len(found) / len(transaction_indicators)
        if has_amount and has_parties:
            confidence *= 2.0  # Boost if we have both amount and parties
        elif has_amount or has_parties:
            confidence *= 1.5  # Boost if we have either amount or parties
        
        return {
            'confidence': confidence,
            'common_fields': list(found),
            'pattern': 'transaction_pattern',
            'usage': 'Represents business events with multiple entities'
        }

def detect_data_patterns(json_obj: dict) -> Dict:
    """
    High-level function to detect and analyze data patterns in JSON.
    
    Args:
        json_obj: JSON object to analyze
        
    Returns:
        dict: Complete analysis including:
            - archetype: Detected data pattern type
            - confidence: Confidence score for detection
            - structure: Structural analysis
            - metadata: Additional pattern-specific metadata
            
    Example:
        >>> data = {
        ...     "events": [
        ...         {
        ...             "timestamp": "2024-03-10T14:30:00Z",
        ...             "event": "user_login",
        ...             "level": "info"
        ...         }
        ...     ]
        ... }
        >>> analysis = detect_data_patterns(data)
        >>> print(f"Detected {analysis['archetype']} pattern")
        >>> print(f"Confidence: {analysis['confidence']:.2f}")
    """
    detector = ArchetypeDetector()
    
    # Get basic archetype detection
    archetype, confidence = detector.detect_archetype(json_obj)
    
    # Perform structural analysis
    structure = detector.analyze_structure(json_obj)
    
    # Extract pattern-specific metadata
    metadata = {}
    if archetype == 'event_log':
        metadata['timestamp_format'] = _detect_timestamp_format(json_obj)
        metadata['event_types'] = _extract_event_types(json_obj)
    elif archetype == 'metric_data':
        metadata['units'] = _extract_units(json_obj)
        metadata['dimensions'] = _extract_dimensions(json_obj)
    elif archetype == 'api_response':
        metadata['pagination'] = _extract_pagination_info(json_obj)
        metadata['response_structure'] = _analyze_response_structure(json_obj)
    
    return {
        'archetype': archetype,
        'confidence': confidence,
        'structure': structure,
        'metadata': metadata
    }

# Helper functions for metadata extraction
def _detect_timestamp_format(json_obj: dict) -> Optional[str]:
    """
    Detect the format of timestamp fields in the JSON object.
    
    Args:
        json_obj: JSON object to analyze
        
    Returns:
        str: Detected timestamp format or None if not found
        
    Detects formats:
        - ISO 8601
        - Unix timestamp
        - Custom formats with common separators
    """
    timestamp_fields = ['timestamp', 'created_at', 'date', 'time', 'datetime']
    formats = {
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z': 'ISO8601_UTC',
        r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}': 'ISO8601_OFFSET',
        r'\d{10}': 'UNIX_TIMESTAMP',
        r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}': 'DATETIME',
        r'\d{4}-\d{2}-\d{2}': 'DATE'
    }
    
    def check_format(value: str) -> Optional[str]:
        for pattern, format_name in formats.items():
            if re.match(pattern, str(value)):
                return format_name
        return None
    
    # Recursively search for timestamp fields
    def search_timestamps(obj, found_formats=None):
        if found_formats is None:
            found_formats = set()
            
        if isinstance(obj, dict):
            for key, value in obj.items():
                if any(field in key.lower() for field in timestamp_fields):
                    fmt = check_format(value)
                    if fmt:
                        found_formats.add(fmt)
                search_timestamps(value, found_formats)
        elif isinstance(obj, list):
            for item in obj:
                search_timestamps(item, found_formats)
                
        return found_formats
    
    formats_found = search_timestamps(json_obj)
    return list(formats_found) if formats_found else None

def _extract_event_types(json_obj: dict) -> List[str]:
    """
    Extract unique event types from log data.
    
    Args:
        json_obj: JSON object to analyze
        
    Returns:
        list: Unique event types found
        
    Example:
        >>> data = {
        ...     "events": [
        ...         {"event": "user_login"},
        ...         {"event": "data_update"},
        ...         {"event": "user_login"}
        ...     ]
        ... }
        >>> _extract_event_types(data)
        ['user_login', 'data_update']
    """
    event_types = set()
    
    def extract_events(obj):
        if isinstance(obj, dict):
            # Check common event field names
            for field in ['event', 'event_type', 'action', 'type']:
                if field in obj:
                    event_types.add(str(obj[field]))
            # Recurse through nested objects
            for value in obj.values():
                extract_events(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_events(item)
    
    extract_events(json_obj)
    return sorted(list(event_types))

def _extract_units(json_obj: dict) -> Dict[str, str]:
    """
    Extract measurement units from metric data.
    
    Args:
        json_obj: JSON object to analyze
        
    Returns:
        dict: Mapping of metric names to their units
        
    Example:
        >>> data = {
        ...     "metrics": [
        ...         {"name": "cpu_usage", "unit": "percent"},
        ...         {"name": "memory", "unit": "bytes"}
        ...     ]
        ... }
        >>> _extract_units(data)
        {'cpu_usage': 'percent', 'memory': 'bytes'}
    """
    units = {}
    
    def extract_metric_units(obj, current_metric=None):
        if isinstance(obj, dict):
            # Check if this object defines a metric
            metric_name = obj.get('name') or obj.get('metric') or current_metric
            unit = obj.get('unit') or obj.get('units')
            
            if metric_name and unit:
                units[metric_name] = unit
            
            # Recurse through nested objects
            for key, value in obj.items():
                extract_metric_units(value, metric_name)
        elif isinstance(obj, list):
            for item in obj:
                extract_metric_units(item, current_metric)
    
    extract_metric_units(json_obj)
    return units

def _extract_dimensions(json_obj: dict) -> List[str]:
    """
    Extract metric dimensions (tags, labels, categories).
    
    Args:
        json_obj: JSON object to analyze
        
    Returns:
        list: Unique dimension names found
        
    Example:
        >>> data = {
        ...     "metrics": [
        ...         {
        ...             "value": 95,
        ...             "dimensions": {
        ...                 "region": "us-east",
        ...                 "service": "api"
        ...             }
        ...         }
        ...     ]
        ... }
        >>> _extract_dimensions(data)
        ['region', 'service']
    """
    dimensions = set()
    
    def extract_dims(obj):
        if isinstance(obj, dict):
            # Check common dimension field names
            for field in ['dimensions', 'tags', 'labels']:
                if field in obj and isinstance(obj[field], dict):
                    dimensions.update(obj[field].keys())
            # Recurse through nested objects
            for value in obj.values():
                extract_dims(value)
        elif isinstance(obj, list):
            for item in obj:
                extract_dims(item)
    
    extract_dims(json_obj)
    return sorted(list(dimensions))

def _extract_pagination_info(json_obj: dict) -> Dict:
    """
    Extract pagination metadata from API responses.
    
    Args:
        json_obj: JSON object to analyze
        
    Returns:
        dict: Pagination information including:
            - type: Type of pagination (offset, cursor, page)
            - fields: Pagination fields found
            - structure: Pagination structure details
    """
    pagination_info = {
        'type': None,
        'fields': [],
        'structure': {}
    }
    
    # Common pagination field patterns
    patterns = {
        'offset': ['offset', 'limit', 'total'],
        'cursor': ['next', 'previous', 'cursor'],
        'page': ['page', 'per_page', 'total_pages']
    }
    
    def analyze_pagination(obj):
        if isinstance(obj, dict):
            # Check for pagination section
            pagination_section = obj.get('pagination') or obj.get('meta') or obj
            
            # Detect pagination type
            for p_type, fields in patterns.items():
                if any(f in pagination_section for f in fields):
                    pagination_info['type'] = p_type
                    pagination_info['fields'].extend(
                        f for f in fields if f in pagination_section
                    )
                    pagination_info['structure'].update({
                        f: pagination_section[f] 
                        for f in fields 
                        if f in pagination_section
                    })
            
            # Recurse through nested objects
            for value in obj.values():
                analyze_pagination(value)
    
    analyze_pagination(json_obj)
    return pagination_info

def _analyze_response_structure(json_obj: dict) -> Dict:
    """
    Analyze API response structure patterns.
    
    Args:
        json_obj: JSON object to analyze
        
    Returns:
        dict: Analysis of response structure including:
            - root_keys: Top-level structure
            - data_location: Path to main data
            - metadata_location: Path to metadata
            - error_handling: Error response pattern if present
    """
    analysis = {
        'root_keys': list(json_obj.keys()),
        'data_location': None,
        'metadata_location': None,
        'error_handling': None
    }
    
    # Detect data location
    data_keys = ['data', 'results', 'items', 'records']
    for key in data_keys:
        if key in json_obj:
            analysis['data_location'] = key
            break
    
    # Detect metadata location
    meta_keys = ['meta', 'metadata', '_metadata']
    for key in meta_keys:
        if key in json_obj:
            analysis['metadata_location'] = key
            break
    
    # Analyze error handling pattern
    error_keys = ['error', 'errors', 'fault']
    for key in error_keys:
        if key in json_obj:
            analysis['error_handling'] = {
                'location': key,
                'structure': json_obj[key]
            }
            break
    
    return analysis 

def analyze_dataset_archetypes(files):
    """
    Analyze all files to determine dataset-wide patterns.
    
    Args:
        files (list): List of file paths to analyze
        
    Returns:
        dict: Dataset-wide archetype patterns and their confidence scores
    """
    detector = ArchetypeDetector()
    
    # Analyze patterns across all files
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                detector.analyze_patterns(data)
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            continue
    
    # Get detected patterns with confidence scores
    patterns = {
        'record': detector._detect_record_pattern(),
        'transaction': detector._detect_transaction_pattern(),
        'event': detector._detect_event_pattern(),
        'reference': detector._detect_reference_pattern()
    }
    
    return patterns 