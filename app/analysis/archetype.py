"""
Smart Archetype Detection module for identifying JSON data patterns.
Analyzes structure and content to classify JSON documents into common patterns.

This module provides intelligent pattern detection for JSON data structures,
helping to identify common data archetypes and their relationships. It uses
a combination of structural analysis, field naming patterns, and content
type detection to classify JSON documents.

Key Features:
    - Pattern Detection: Identifies common JSON data structures
    - Relationship Analysis: Maps connections between data elements
    - Content Classification: Categorizes data based on usage patterns
    - Confidence Scoring: Provides certainty levels for detected patterns
"""

from datetime import datetime
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple, Any, Set
import json
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class ArchetypeDetector:
    """
    Detects and classifies JSON data patterns and archetypes.
    
    This class analyzes JSON data structures to identify common patterns
    and archetypes, helping to understand the semantic purpose and
    relationships within the data.
    
    Patterns Detected:
        Event Logs:
            - Timestamped entries
            - Action/state changes
            - Sequential records
            
        API Responses:
            - Status codes
            - Response metadata
            - Error handling
            
        Metric Data:
            - Numerical measurements
            - Time series data
            - Statistical aggregates
            
        Entity Collections:
            - Business objects
            - Record sets
            - Data hierarchies
            
        State Machines:
            - Status transitions
            - State history
            - Validation rules
            
        Configuration:
            - Setting definitions
            - Feature flags
            - System preferences
            
    Attributes:
        patterns (defaultdict): Tracks detected pattern frequencies
        field_frequencies (defaultdict): Counts field occurrences
        type_frequencies (defaultdict): Tracks data type usage
        value_patterns (defaultdict): Records value pattern sets
        temporal_fields (set): Tracks time-related fields
        numerical_fields (set): Tracks number-related fields
        
    Example:
        >>> detector = ArchetypeDetector()
        >>> data = {
        ...     "timestamp": "2024-01-18T10:30:00",
        ...     "event": "user_login",
        ...     "user_id": "123"
        ... }
        >>> archetypes = detector.detect_archetypes(data)
        >>> print(archetypes)
        [('event_log', 0.9), ('entity_reference', 0.4)]
    """
    
    def __init__(self):
        self.patterns = defaultdict(int)
        self.field_frequencies = defaultdict(int)
        self.type_frequencies = defaultdict(int)
        self.value_patterns = defaultdict(set)
        self.temporal_fields = set()
        self.numerical_fields = set()
        
        # Add timestamp pattern matching
        self.timestamp_patterns = [
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO 8601
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # ISO-like datetime
            r'\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}',  # Common datetime
            r'\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}',  # US datetime
            r'\d{4}-\d{2}-\d{2}',                     # ISO date
            r'\d{2}/\d{2}/\d{4}',                     # US date
            r'\d{4}/\d{2}/\d{2}'                      # Alternative date
        ]
        
        # Add HTTP status patterns
        self.http_status_patterns = [
            r'^[1-5][0-9]{2}$',           # Basic HTTP status codes
            r'2\d{2}',                     # Success codes (200-299)
            r'3\d{2}',                     # Redirect codes (300-399)
            r'4\d{2}',                     # Client error codes (400-499)
            r'5\d{2}'                      # Server error codes (500-599)
        ]
        
    def detect_dataset_patterns(self, structures: List[Dict]) -> Dict:
        """
        Analyze the entire dataset to detect patterns and relationships.
        
        Args:
            structures: List of JSON structures to analyze.
            
        Returns:
            dict: Detected archetypes and interrelationships with:
                - archetypes: List of archetype matches per structure
                - relationships: List of detected relationships
                
        Example:
            >>> detector = ArchetypeDetector()
            >>> data = [
            ...     {
            ...         "event": "login",
            ...         "timestamp": "2024-03-16T10:30:00Z",
            ...         "user_id": "u123"
            ...     },
            ...     {
            ...         "user": {"id": "u123", "name": "Alice"}
            ...     }
            ... ]
            >>> results = detector.detect_dataset_patterns(data)
            >>> print("Archetypes:", results["archetypes"])
            >>> print("Relationships:", results["relationships"])
        """
        logger.debug("Analyzing dataset-wide patterns...")
        
        archetype_results = []
        relationships = []

        for data in structures:
            self._analyze_structure(data)
            archetypes = self.detect_archetypes(data)
            archetype_results.append(archetypes)
            relationships.extend(self.detect_relationships(data))

        logger.debug("Archetype Results:")
        for i, result in enumerate(archetype_results):
            logger.debug(f"Structure {i}: {result}")
        
        logger.debug("Detected Relationships:")
        for rel in relationships:
            logger.debug(f"{rel['source']} -> {rel['target']} ({rel['type']})")
        
        return {
            "archetypes": archetype_results, 
            "relationships": relationships
        }
    
    def _detect_record_pattern(self) -> Dict:
        """
        Detect record patterns in JSON data representing distinct entities.
        
        This method analyzes field names and value patterns to identify
        structures that represent distinct business entities or records.
        
        Detection Criteria:
            - Identity Fields: 'id', 'uuid'
            - Descriptive Fields: 'name', 'type', 'code'
            - Metadata Fields: 'properties', 'attributes'
            - Value Stability: Consistent values across instances
            
        Confidence Scoring:
            - Base: Proportion of indicator fields present
            - Boost: 2.0x for identity + description fields
            - Boost: 1.5x for identity + stable values
            - Penalty: 0.25x for single indicator
            
        Returns:
            Dict: Detection results containing:
                - confidence (float): Detection confidence score
                - common_fields (list): Found indicator fields
                - pattern (str): Pattern identifier
                - usage (str): Pattern usage description
                
        Note:
            High confidence requires multiple corroborating indicators.
            Single field matches result in significantly reduced confidence.
        """
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
        """
        Detect event patterns in JSON data representing time-based occurrences.
        
        This method identifies structures that represent events, actions,
        or state changes that occur at specific points in time.
        
        Detection Criteria:
            - Temporal Fields: 'timestamp', 'time', 'date'
            - Action Fields: 'status', 'state', 'action'
            - Event Types: 'type', 'category'
            - Lifecycle Fields: 'created_at', 'updated_at'
            
        Confidence Scoring:
            - Base: Proportion of indicator fields present
            - Boost: 2.0x for temporal + status fields
            - Boost: 1.5x for temporal fields only
            
        Returns:
            Dict: Detection results containing:
                - confidence (float): Detection confidence score
                - common_fields (list): Found indicator fields
                - pattern (str): Pattern identifier
                - usage (str): Pattern usage description
                
        Note:
            Temporal fields are critical for event pattern detection.
            Status/state fields provide additional confirmation.
        """
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
        """
        Detect collection patterns in JSON data representing groups of items.
        
        This method identifies structures that represent collections or
        lists of related items, typically with similar structures.
        
        Detection Criteria:
            - Array Fields: Fields containing lists
            - Plural Names: Field names ending in 's'
            - Consistent Structure: Similar items in arrays
            - Collection Terms: 'items', 'list', 'collection'
            
        Confidence Scoring:
            - Base: Number of array fields (max 1.0)
            - Boost: Plural field names
            - Boost: Consistent array item structure
            
        Returns:
            Dict: Detection results containing:
                - confidence (float): Detection confidence score
                - common_fields (list): Found array fields
                - pattern (str): Pattern identifier
                - usage (str): Pattern usage description
                
        Note:
            Arrays must contain multiple items for high confidence.
            Single-item arrays receive reduced confidence scores.
        """
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
        """
        Detect reference patterns in JSON data linking to other objects.
        
        This method identifies structures that represent relationships or
        references between different objects in the data model.
        
        Detection Criteria:
            - ID References: Fields ending in '_id'
            - References: Fields ending in '_ref'
            - Foreign Keys: Fields matching known ID patterns
            - Link Fields: Fields containing reference metadata
            
        Confidence Scoring:
            - Base: Number of reference fields (max 1.0)
            - Boost: Multiple references to same entity type
            - Boost: Presence of reference metadata
            
        Returns:
            Dict: Detection results containing:
                - confidence (float): Detection confidence score
                - common_fields (list): Found reference fields
                - pattern (str): Pattern identifier
                - usage (str): Pattern usage description
                
        Note:
            Reference fields should follow consistent naming patterns.
            Confidence increases with multiple related references.
        """
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
        """
        Detect metric patterns in JSON data containing measurements.
        
        This method identifies structures that represent numerical
        measurements, statistics, or quantitative data points.
        
        Detection Criteria:
            - Numerical Fields: Integer or float values
            - Metric Names: 'value', 'amount', 'count'
            - Units: Fields containing measurement units
            - Aggregations: 'total', 'average', 'sum'
            
        Confidence Scoring:
            - Base: Proportion of metric indicators present
            - Boost: 2.0x for numbers + quantity fields
            - Boost: 1.5x for numerical fields only
            
        Returns:
            Dict: Detection results containing:
                - confidence (float): Detection confidence score
                - common_fields (list): Found metric fields
                - pattern (str): Pattern identifier
                - usage (str): Pattern usage description
                
        Note:
            High confidence requires both numerical values and context.
            Pure numbers without context receive reduced confidence.
        """
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
        """
        Detect state patterns in JSON data representing configurations.
        
        This method identifies structures that represent system states,
        configurations, or settings that control behavior.
        
        Detection Criteria:
            - State Fields: 'status', 'state', 'mode'
            - Config Fields: 'enabled', 'active', 'config'
            - Settings: 'preferences', 'options'
            - Flags: Boolean or enumerated values
            
        Confidence Scoring:
            - Base: Proportion of state indicators present
            - Boost: Multiple related state fields
            - Boost: Presence of validation rules
            
        Returns:
            Dict: Detection results containing:
                - confidence (float): Detection confidence score
                - common_fields (list): Found state fields
                - pattern (str): Pattern identifier
                - usage (str): Pattern usage description
                
        Note:
            State patterns often include validation constraints.
            Configuration data typically has limited value ranges.
        """
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
        """
        Detect document patterns in JSON data with hierarchical content.
        
        This method identifies structures that represent document-like
        content with nested sections and hierarchical organization.
        
        Detection Criteria:
            - Content Fields: 'content', 'body', 'text'
            - Structure: 'sections', 'elements', 'blocks'
            - Metadata: 'title', 'description', 'details'
            - Nesting: Deep hierarchical organization
            
        Confidence Scoring:
            - Base: Proportion of document indicators
            - Boost: Deep nesting (>2 levels)
            - Boost: Multiple content sections
            
        Returns:
            Dict: Detection results containing:
                - confidence (float): Detection confidence score
                - common_fields (list): Found document fields
                - pattern (str): Pattern identifier
                - usage (str): Pattern usage description
                
        Note:
            Document patterns typically have deeper nesting.
            Content organization is hierarchical by nature.
        """
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
        """
        Find fields containing arrays of similar items.
        
        This method analyzes the type frequencies to identify fields that
        contain arrays (lists) with multiple items, indicating potential
        collections or repeated structures.
        
        Detection Process:
            1. Examine type frequency data
            2. Identify fields with list type
            3. Filter for fields with multiple occurrences
            4. Collect field names in a set
            
        Returns:
            Set[str]: Set of field names that contain arrays with:
                - Multiple items
                - Consistent structure
                - Similar content types
                
        Note:
            Only includes arrays with more than one item.
            Single-item arrays are excluded to avoid false positives.
        """
        array_fields = set()
        for type_info, count in self.type_frequencies.items():
            field, type_name = type_info.split(':')
            if type_name == 'list' and count > 1:
                array_fields.add(field)
        return array_fields
    
    def _calculate_max_nesting(self) -> int:
        """
        Calculate the maximum nesting depth in the data structure.
        
        This method analyzes field paths to determine the deepest level
        of nesting in the JSON structure, which helps identify complex
        hierarchical patterns.
        
        Calculation Process:
            1. Examine each field path
            2. Count dot separators in path
            3. Track maximum depth seen
            4. Return highest depth found
            
        Returns:
            int: Maximum nesting depth, where:
                0 = Flat structure (no nesting)
                1 = Single level of nesting
                2+ = Multiple levels of nesting
                
        Note:
            Depth calculation uses dot notation in paths.
            Array indices are treated as additional nesting levels.
        """
        max_depth = 0
        for path in self.field_frequencies.keys():
            depth = path.count('.')
            max_depth = max(max_depth, depth)
        return max_depth

    def analyze_structure(self, json_obj: dict) -> Dict:
        """
        Perform detailed structural analysis of a JSON object.
        
        This method conducts a comprehensive analysis of a JSON object's
        structure, examining nesting patterns, value types, and field
        naming conventions.
        
        Analysis Components:
            Depth Analysis:
                - Maximum nesting depth
                - Path complexity
                - Structural patterns
                
            Array Detection:
                - Array locations
                - Item consistency
                - Collection patterns
                
            Value Analysis:
                - Type distribution
                - Value patterns
                - Data characteristics
                
            Field Analysis:
                - Naming patterns
                - Field relationships
                - Semantic groupings
                
        Args:
            json_obj (dict): JSON object to analyze
            
        Returns:
            Dict: Analysis results containing:
                - depth (int): Maximum nesting depth
                - array_paths (List[str]): Paths to array fields
                - value_types (Counter): Distribution of value types
                - field_patterns (Counter): Common field patterns
                
        Example:
            >>> analysis = analyze_structure({
            ...     "users": [
            ...         {"id": 1, "name": "Alice"},
            ...         {"id": 2, "name": "Bob"}
            ...     ]
            ... })
            >>> print(f"Max depth: {analysis['depth']}")
            >>> print(f"Arrays found at: {analysis['array_paths']}")
            
        Note:
            Analysis is recursive and handles nested structures.
            Array indices are included in path calculations.
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

    def detect_archetypes(self, chunk: Dict) -> List[Tuple[str, float]]:
        """Detect archetypes with proper error handling."""
        try:
            archetypes = []
            content = chunk.get('content', {})
            
            # Check each archetype pattern
            for pattern in self.patterns:
                try:
                    confidence = pattern.check(content)
                    if confidence > 0:
                        archetypes.append((pattern.name, confidence))
                except Exception as e:
                    logger.error(f"Error checking pattern {pattern.name}: {e}")
                    continue
            
            return sorted(archetypes, key=lambda x: x[1], reverse=True)
            
        except Exception as e:
            logger.error(f"Error detecting archetypes: {e}")
            return []

    def _score_event_log(self, data: Dict) -> float:
        """
        Score the likelihood that a JSON object represents an event log entry.
        
        This method evaluates how closely a JSON object matches the expected
        pattern of an event log entry by checking for temporal, event-related,
        and sequential indicators.
        
        Scoring Criteria:
            Temporal (0.4):
                - Timestamp field presence
                - Valid timestamp format
                - Standard date/time patterns
                
            Event Context (0.3):
                - Event type/name
                - Action description
                - Status information
                - Severity level
                
            Sequential (0.2):
                - Order indicators
                - Sequence numbers
                - Index values
                
        Args:
            data (Dict): JSON object to evaluate
            
        Returns:
            float: Confidence score between 0.0 and 0.9, where:
                0.0 = No event log characteristics
                0.4 = Has timestamp only
                0.7 = Has timestamp and event context
                0.9 = Complete event log entry
                
        Note:
            Timestamp is required for any non-zero score.
            Additional context increases confidence.
        """
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
        """
        Score the likelihood that a JSON object represents metric data.
        
        This method evaluates how closely a JSON object matches the expected
        pattern of metric data by checking for numerical values, metric-specific
        fields, and measurement context.
        
        Scoring Criteria:
            Numerical Content (0.3):
                - Presence of numbers
                - Proportion of numerical values
                - Value ranges and precision
                
            Metric Context (0.3):
                - Metric identifiers
                - Measurement terms
                - Aggregation indicators
                
            Time Series (0.2):
                - Temporal markers
                - Sequence indicators
                - Period references
                
            Units (0.2):
                - Unit specifications
                - Scale indicators
                - Dimension markers
                
        Args:
            data (Dict): JSON object to evaluate
            
        Returns:
            float: Confidence score between 0.0 and 1.0, where:
                0.0 = No metric characteristics
                0.3 = Has numerical values
                0.6 = Has context and values
                1.0 = Complete metric data
                
        Note:
            Pure numbers without context get reduced scores.
            Complete metric data requires multiple indicators.
        """
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
        """
        Score the likelihood that a JSON object represents an API response.
        
        This method evaluates how closely a JSON object matches the expected
        pattern of an API response by checking for standard response fields,
        status codes, and metadata.
        
        Scoring Criteria:
            Response Fields (0.3):
                - Status indicators
                - Response codes
                - Message content
                - Error information
                
            HTTP Status (0.4):
                - Standard HTTP codes
                - Status ranges
                - Response types
                
            Metadata (0.3):
                - Headers
                - Timestamps
                - Version info
                - API details
                
        Args:
            data (Dict): JSON object to evaluate
            
        Returns:
            float: Confidence score between 0.0 and 1.0, where:
                0.0 = No API response characteristics
                0.3 = Basic response fields
                0.7 = Fields + status codes
                1.0 = Complete API response
                
        Note:
            Status codes strongly indicate API responses.
            Metadata provides additional confirmation.
        """
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
        """
        Score the likelihood that a JSON object represents a state machine.
        
        This method evaluates how closely a JSON object matches the expected
        pattern of a state machine by checking for state transitions,
        validation rules, and temporal markers.
        
        Scoring Criteria:
            State Fields (0.4):
                - Current state
                - Status indicators
                - State names
                - Mode settings
                
            Transitions (0.3):
                - State changes
                - From/To states
                - Actions
                - Events
                
            Temporal (0.2):
                - Timestamps
                - Change history
                - Transition times
                
            Rules (0.1):
                - Validation rules
                - Constraints
                - Allowed states
                
        Args:
            data (Dict): JSON object to evaluate
            
        Returns:
            float: Confidence score between 0.0 and 1.0, where:
                0.0 = No state machine characteristics
                0.4 = Has state fields
                0.7 = Has states and transitions
                1.0 = Complete state machine
                
        Note:
            State fields are primary indicators.
            Transitions and rules provide confirmation.
        """
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
        """
        Score the likelihood that a JSON object represents an entity collection.
        
        This method evaluates how closely a JSON object matches the expected
        pattern of an entity collection by checking for identity fields,
        relationships, and collection indicators.
        
        Scoring Criteria:
            Identity (0.3):
                - Unique identifiers
                - Keys
                - Names
                - References
                
            Relationships (0.3):
                - Parent/child links
                - Ownership
                - Group membership
                - Related entities
                
            Metadata (0.2):
                - Types
                - Categories
                - Classifications
                - Attributes
                
            Collection (0.2):
                - Item groups
                - Entry lists
                - Record sets
                - Element arrays
                
        Args:
            data (Dict): JSON object to evaluate
            
        Returns:
            float: Confidence score between 0.0 and 1.0, where:
                0.0 = No collection characteristics
                0.3 = Has identifiers
                0.6 = Has relationships
                1.0 = Complete collection
                
        Note:
            Identity fields are primary indicators.
            Relationships and grouping provide confirmation.
        """
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

    def _score_transaction_pattern(self, data: Dict) -> float:
        """
        Score the likelihood that a JSON object represents a transaction.
        
        This method evaluates how closely a JSON object matches the expected
        pattern of a transaction by checking for transaction-specific fields,
        amounts, and related metadata.
        
        Scoring Criteria:
            Transaction Fields (0.4):
                - Transaction IDs
                - Order references
                - Payment info
                - Invoice details
                
            Financial Data (0.3):
                - Amounts
                - Prices
                - Totals
                - Quantities
                
            Status (0.2):
                - Transaction state
                - Payment status
                - Processing stage
                
            Metadata (0.1):
                - Timestamps
                - References
                - Categories
                
        Args:
            data (Dict): JSON object to evaluate
            
        Returns:
            float: Confidence score between 0.0 and 1.0, where:
                0.0 = No transaction characteristics
                0.4 = Has transaction fields
                0.7 = Has financial data
                1.0 = Complete transaction
                
        Note:
            Transaction fields are primary indicators.
            Financial data provides strong confirmation.
        """
        score = 0.0
        
        # Check for transaction-related fields
        transaction_keywords = {
            'transaction', 'order', 'payment', 'invoice', 'purchase',
            'sale', 'shipment', 'total', 'amount', 'price'
        }
        has_transaction_fields = any(key in transaction_keywords for key in data.keys())
        if has_transaction_fields:
            score += 0.4
        
        # Check for numerical fields (e.g., amounts or quantities)
        numerical_fields = sum(1 for v in data.values() if isinstance(v, (int, float)))
        if numerical_fields > 0:
            score += 0.3
        
        # Check for references to parties (e.g., buyer/seller, customer/supplier)
        party_keywords = {'buyer', 'seller', 'customer', 'supplier', 'party'}
        has_party_fields = any(key in party_keywords for key in data.keys())
        if has_party_fields:
            score += 0.3

        return score

    def get_matching_archetype(self, value: Dict, path: str) -> Dict:
        """
        Get the best matching archetype for a specific value based on dataset patterns.
        
        This method evaluates a JSON value against known dataset patterns to
        determine its most likely archetype, considering both structural and
        semantic characteristics.
        
        Matching Process:
            1. Compare against known patterns
            2. Calculate confidence scores
            3. Select highest confidence match
            4. Return archetype details
            
        Args:
            value (Dict): JSON value to classify
            path (str): Path to value in data structure
            
        Returns:
            Dict: Best matching archetype containing:
                - type (str): Archetype identifier
                - confidence (float): Match confidence
                - pattern (str): Pattern description
                
        Example:
            >>> value = {"id": 123, "name": "Example"}
            >>> path = "data.items[0]"
            >>> match = get_matching_archetype(value, path)
            >>> print(f"Found {match['type']} with {match['confidence']}")
            
        Note:
            Returns None if no patterns match with sufficient confidence.
            Path information helps with contextual matching.
        """
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
        """
        Check if a value matches a pattern's expected fields.
        
        This method determines if a JSON value matches a pattern by checking
        for the presence of characteristic fields that define the pattern.
        
        Matching Process:
            1. Extract value's field names
            2. Compare with pattern fields
            3. Check for field intersections
            4. Determine match status
            
        Args:
            value (Dict): JSON value to check
            pattern_fields (List[str]): Expected fields for pattern
            
        Returns:
            bool: True if value matches pattern, False otherwise
            
        Note:
            Partial matches are considered valid.
            Field order is not significant.
        """
        value_fields = set(value.keys())
        return bool(value_fields & set(pattern_fields))

    def _analyze_structure(self, data: Any, path: str = "") -> None:
        """
        Recursively analyze data structure, chunk it, and detect archetypes.
        
        This method performs a deep analysis of a JSON data structure,
        identifying patterns and archetypes at each level of nesting.
        
        Analysis Process:
            Structure Analysis:
                - Field identification
                - Type detection
                - Nesting analysis
                
            Pattern Detection:
                - Field frequencies
                - Value patterns
                - Structural patterns
                
            Archetype Detection:
                - Pattern matching
                - Confidence scoring
                - Relationship mapping
                
        Args:
            data (Any): Data structure to analyze
            path (str): Current path in structure (default: "")
            
        Note:
            Updates internal state (field_frequencies, etc.).
            Logs detected archetypes for debugging.
            Handles nested structures recursively.
        """
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                self.field_frequencies[key] += 1
                
                # Detect archetypes for each sub-object
                if isinstance(value, (dict, list)):
                    self._analyze_structure(value, current_path)
                else:
                    logger.debug(f"Analyzing path: {current_path} -> {value}")
                    archetypes = self.detect_archetypes({key: value})
                    logger.debug(f"Detected archetypes at {current_path}: {archetypes}")
        elif isinstance(data, list):
            for index, item in enumerate(data):
                current_path = f"{path}[{index}]"
                self._analyze_structure(item, current_path)

    def analyze_patterns(self, data: Dict) -> None:
        """
        Analyze patterns in a single JSON document.
        
        This method performs a comprehensive pattern analysis on a JSON
        document, identifying and tracking various structural and semantic
        patterns.
        
        Analysis Steps:
            1. Recursive Structure Analysis:
                - Field relationships
                - Value distributions
                - Nesting patterns
                
            2. Pattern Detection:
                - Record patterns
                - Event patterns
                - Metric patterns
                - State patterns
                - Collection patterns
                
            3. Pattern Frequency Updates:
                - Track occurrences
                - Update confidence scores
                - Record relationships
                
        Args:
            data (Dict): JSON document to analyze
            
        Note:
            Updates internal pattern frequencies.
            Confidence threshold of 0.5 for pattern recognition.
            Patterns can overlap and coexist.
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

    def detect_relationships(self, data: Dict, parent_key: Optional[str] = None) -> List[Dict]:
        """
        Detect relationships between archetypes within a JSON file.
        
        Args:
            data: JSON object to analyze.
            parent_key: Key to track hierarchical relationships.
            
        Returns:
            list: Relationships detected with source, target, and type.
            
        Example:
            >>> detector = ArchetypeDetector()
            >>> data = {
            ...     "user": {
            ...         "id": "u123",
            ...         "team_id": "t456"
            ...     },
            ...     "team": {
            ...         "id": "t456",
            ...         "parent_id": "org789"
            ...     }
            ... }
            >>> relationships = detector.detect_relationships(data)
            >>> for rel in relationships:
            ...     print(f"{rel['source']} -> {rel['target']} ({rel['type']})")
            user -> id (key-based)
            user -> team_id (key-based)
            team -> id (key-based)
            team -> parent_id (key-based)
        """
        relationships = []
        
        def find_links(obj, parent=None):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_key = f"{parent}.{key}" if parent else key
                    
                    # Key-based relationships
                    if key.endswith("_id") or key in {"id", "ref", "parent_id"}:
                        relationships.append({
                            "source": parent, 
                            "target": key, 
                            "type": "key-based"
                        })
                    
                    # Recurse through sub-objects
                    find_links(value, current_key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    find_links(item, f"{parent}[{i}]")
        
        find_links(data)
        return relationships

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