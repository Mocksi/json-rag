import re
from datetime import datetime, timedelta
from app.utils import parse_timestamp
# From your original code: extract_time_range, extract_metric_conditions, extract_pagination_info, extract_entity_references, analyze_query_intent

def extract_time_range(query):
    """
    Extract temporal range information from a query string.
    
    Args:
        query (str): User query string
        
    Returns:
        dict: Time range information containing:
            - start: Start datetime
            - end: End datetime
            - type: 'exact', 'relative', or 'named'
            - unit: Time unit for relative ranges
            - amount: Numeric amount for relative ranges
            - period: Named period (e.g., 'week', 'month')
            
    Patterns Recognized:
        - Exact: "between 2024-01-01 and 2024-02-01"
        - Relative: "last 7 days", "last 2 weeks"
        - Named: "this week", "this month"
    """
    query_lower = query.lower()
    # exact date range
    match = re.search(r'between\s+(\d{4}-\d{2}-\d{2})\s+and\s+(\d{4}-\d{2}-\d{2})', query_lower)
    if match:
        return {
            'start': parse_timestamp(match.group(1)),
            'end': parse_timestamp(match.group(2)),
            'type': 'exact'
        }
    # last X day/week etc.
    match = re.search(r'last\s+(\d+)\s+(day|week|month|year)s?', query_lower)
    if match:
        amount = int(match.group(1))
        unit = match.group(2)
        end_date = datetime.now()
        if unit == 'day':
            delta = timedelta(days=amount)
        elif unit == 'week':
            delta = timedelta(weeks=amount)
        elif unit == 'month':
            delta = timedelta(days=amount * 30)
        else:
            delta = timedelta(days=amount * 365)
        return {
            'start': end_date - delta,
            'end': end_date,
            'type': 'relative',
            'unit': unit,
            'amount': amount
        }
    # "this week"/"this month"
    if 'this week' in query_lower:
        today = datetime.now()
        start = today - timedelta(days=today.weekday())
        return {
            'start': start,
            'end': start + timedelta(days=7),
            'type': 'named',
            'period': 'week'
        }
    elif 'this month' in query_lower:
        today = datetime.now()
        start = today.replace(day=1)
        if today.month == 12:
            end = today.replace(year=today.year+1, month=1, day=1)
        else:
            end = today.replace(month=today.month+1, day=1)
        return {
            'start': start,
            'end': end,
            'type': 'named',
            'period': 'month'
        }
    return None

def extract_metric_conditions(query):
    """
    Extract metric-related conditions from a query string.
    
    Args:
        query (str): User query string
        
    Returns:
        dict: Metric conditions containing:
            - aggregation: Type of aggregation (average, maximum, etc.)
            - metric: Name of the metric
            - comparison: Comparison details (type and value)
            
    Patterns Recognized:
        - Aggregations: "average", "maximum", "minimum", "sum", "count"
        - Metrics: "of [metric]", "[metric] values", "[metric] is"
        - Comparisons: "greater than", "less than", "equal to"
    """
    conditions = {}
    query_lower = query.lower()
    agg_patterns = {
        'average': ['average', 'mean', 'avg'],
        'maximum': ['maximum', 'max', 'highest', 'peak'],
        'minimum': ['minimum', 'min', 'lowest', 'bottom'],
        'sum': ['sum', 'total'],
        'count': ['count', 'number of', 'how many']
    }
    for agg_type, patterns in agg_patterns.items():
        if any(p in query_lower for p in patterns):
            conditions['aggregation'] = agg_type
            break
    metric_patterns = [
        r'(?:of|in|for)\s+(\w+)',
        r'(\w+)\s+(?:values|measurements)',
        r'(\w+)\s+(?:is|are|was|were)'
    ]
    for pattern in metric_patterns:
        match = re.search(pattern, query_lower)
        if match:
            metric = match.group(1)
            if metric not in ['the', 'a', 'an']:
                conditions['metric'] = metric
                break
    comp_patterns = {
        'greater_than': [r'greater than (\d+)', r'more than (\d+)', r'above (\d+)'],
        'less_than': [r'less than (\d+)', r'under (\d+)', r'below (\d+)'],
        'equals': [r'equal to (\d+)', r'exactly (\d+)']
    }
    for comp_type, pats in comp_patterns.items():
        for pat in pats:
            match = re.search(pat, query_lower)
            if match:
                conditions['comparison'] = {
                    'type': comp_type,
                    'value': float(match.group(1))
                }
                break
        if 'comparison' in conditions:
            break
    return conditions if conditions else None

def extract_pagination_info(query):
    """
    Extract pagination-related information from a query string.
    
    Args:
        query (str): User query string
        
    Returns:
        dict: Pagination information containing:
            - type: 'absolute', 'relative', or 'limit'
            - page: Page number for absolute pagination
            - direction: 'next' or 'previous' for relative pagination
            - size: Number of results for limit pagination
            
    Patterns Recognized:
        - Absolute: "page 5"
        - Relative: "next page", "previous page"
        - Limit: "show 10 results"
    """
    query_lower = query.lower()
    match = re.search(r'page\s+(\d+)', query_lower)
    if match:
        return {'type': 'absolute', 'page': int(match.group(1))}
    if 'next page' in query_lower:
        return {'type': 'relative', 'direction': 'next'}
    if 'previous page' in query_lower or 'prev page' in query_lower:
        return {'type': 'relative', 'direction': 'previous'}
    match = re.search(r'show\s+(\d+)\s+(?:results|items)', query_lower)
    if match:
        return {'type': 'limit', 'size': int(match.group(1))}
    return None

def extract_entity_references(query):
    """
    Extract entity references and filters from a query string.
    
    Args:
        query (str): User query string
        
    Returns:
        dict: Entity references containing:
            - Key-value pairs from explicit assignments
            - Typed references (type:value)
            - ID references (#123)
            - Entity-specific patterns (user, status, category)
            
    Patterns Recognized:
        - Key-value: "key=value"
        - Typed: "type:value"
        - ID: "#123"
        - Entity: "user john", "status active", "category books"
    """
    references = {}
    for token in query.split():
        if '=' in token:
            key, value = token.split('=', 1)
            references[key.strip()] = value.strip()
    typed_refs = re.finditer(r'(\w+):(\w+)', query)
    for m in typed_refs:
        references[m.group(1)] = m.group(2)
    id_match = re.search(r'#(\d+)', query)
    if id_match:
        references['id'] = id_match.group(1)
    entity_patterns = {
        'user': [r'user (\w+)', r'by (\w+)'],
        'status': [r'status (\w+)', r'state (\w+)'],
        'category': [r'in (\w+)', r'category (\w+)']
    }
    for entity_type, pats in entity_patterns.items():
        for pat in pats:
            match = re.search(pat, query.lower())
            if match:
                references[entity_type] = match.group(1)
                break
    return references if references else None

def analyze_query_intent(query):
    """Analyze query to determine primary and secondary intents."""
    query_lower = query.lower()
    intents = set()
    
    patterns = {
        'relationship': [
            # Direct relationships
            r'influence', r'impact', r'affect',
            r'relationship', r'connection', r'linked',
            r'between', r'among', r'across',
            # Indirect relationships
            r'through', r'via', r'chain',
            r'path', r'flow', r'network',
            # Comparative relationships
            r'compare', r'versus', r'vs',
            r'relative to', r'compared to',
            r'discrepanc(y|ies)', r'difference',
            r'actual vs', r'against',
            # Connected entities
            r'considering', r'based on', r'related to',
            r'involving', r'including', r'with'
        ],
        'temporal': [
            # Time points
            r'when', r'time', r'date',
            r'current', r'latest', r'recent',
            # Time ranges
            r'during', r'period', r'interval',
            r'between.*and', r'from.*to',
            # Relative time
            r'next', r'last', r'previous',
            r'upcoming', r'future', r'past'
        ],
        'metric': [
            # Basic metrics
            r'how many', r'amount', r'number',
            r'value', r'level', r'quantity',
            # Aggregations
            r'average', r'mean', r'median',
            r'total', r'sum', r'aggregate',
            # Comparisons
            r'more than', r'less than',
            r'greater', r'higher', r'lower',
            r'maximum', r'minimum', r'peak'
        ],
        'risk': [
            # Risk indicators
            r'risk', r'threat', r'danger',
            r'warning', r'alert', r'critical',
            # Analysis terms
            r'analysis', r'assessment',
            r'evaluation', r'measure',
            # Discrepancies
            r'gap', r'deviation', r'variance',
            r'discrepancy', r'difference'
        ],
        'comparison': [
            r'compare', r'vs', r'versus',
            r'actual.*vs', r'contracted.*vs',
            r'discrepanc(y|ies)', r'difference',
            r'deviation', r'variance',
            r'expected.*actual', 'planned.*actual'
        ]
    }
    
    # Check patterns and add intents
    for intent_type, pattern_list in patterns.items():
        matches = [p for p in pattern_list if re.search(p, query_lower)]
        if matches:
            intents.add(intent_type)
            print(f"DEBUG: {intent_type.title()} matches: {matches}")
    
    # Add 'general' intent if no others found
    if not intents:
        intents.add('general')
        print("DEBUG: No specific intent detected, using 'general'")
    
    # Determine primary intent based on combinations
    primary = 'general'
    if intents - {'general'}:
        if 'comparison' in intents:
            primary = 'relationship'
        elif 'risk' in intents:
            # Risk queries usually need relationship traversal
            primary = 'relationship'
            if 'temporal' in intents:
                # Add temporal for future predictions
                intents.add('temporal')
        elif 'relationship' in intents:
            primary = 'relationship'
        elif 'temporal' in intents and 'metric' in intents:
            primary = 'temporal'
        elif 'metric' in intents:
            primary = 'metric'
        elif 'temporal' in intents:
            primary = 'temporal'
    
    # Force relationship intent when multiple factors need consideration
    if 'considering' in query_lower or 'based on' in query_lower:
        primary = 'relationship'
        intents.add('relationship')
    
    result = {
        'primary_intent': primary,
        'all_intents': list(intents)
    }
    
    print(f"DEBUG: Final intent analysis: {result}")
    return result

def extract_filters_from_query(query):
    """
    Extract key-value filters from query string.
    
    Args:
        query (str): Query string to parse
        
    Returns:
        dict: Extracted filters as key-value pairs
        
    Example:
        >>> filters = extract_filters_from_query("show items where status=active and region=east")
        {'status': 'active', 'region': 'east'}
    """
    filters = {}
    tokens = query.split()
    for token in tokens:
        if "=" in token:
            k, v = token.split("=", 1)
            filters[k.strip()] = v.strip()
    return filters

def get_system_prompt(query_intent):
    """
    Get the appropriate system prompt based on query intent.
    
    Args:
        query_intent (dict): Dictionary containing primary_intent and all_intents
        
    Returns:
        str: Customized system prompt for the query type
        
    Intents:
        - temporal: Focus on time-based relationships
        - aggregation: Focus on numerical analysis
        - entity: Focus on relationships
        - state: Focus on transitions
        - general: Default helpful assistant
    """
    base_prompt = """You are a precise data analyst assistant that answers questions based ONLY on the provided context.
    
    CRITICAL RULES:
    1. NEVER make up or hallucinate information
    2. If you don't have enough information to answer, say "I don't have enough information to answer that question"
    3. Only reference data that is explicitly present in the provided chunks
    4. Do not infer relationships or connections unless they are directly evidenced in the data
    5. If asked about specific metrics, only cite exact numbers from the data"""

    # Add intent-specific guidance
    if query_intent['primary_intent'] == 'temporal':
        return base_prompt + " Focus on temporal relationships and event sequences that are explicitly documented in the data."
    elif query_intent['primary_intent'] == 'relationship':
        return base_prompt + " Focus on relationships and connections that are directly evidenced in the data structure."
    elif query_intent['primary_intent'] == 'metric':
        return base_prompt + " Focus on numerical patterns and trends that are explicitly present in the data."
    elif query_intent['primary_intent'] == 'risk':
        return base_prompt + " Focus on risk indicators and warning signs that are directly present in the data."
    
    return base_prompt
