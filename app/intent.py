import re
from datetime import datetime, timedelta
from app.utils import parse_timestamp
# From your original code: extract_time_range, extract_metric_conditions, extract_pagination_info, extract_entity_references, analyze_query_intent

def extract_time_range(query):
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
    """
    Analyze query to determine primary and secondary intents.
    
    Args:
        query (str): User query string
        
    Returns:
        dict: Intent analysis with primary and all intents
    """
    query_lower = query.lower()
    intents = set()
    
    # Enhanced temporal patterns
    temporal_patterns = [
        r'during', r'progression', r'trend',
        r'when', r'time', r'date', 
        r'recent', r'latest', r'today', 
        r'yesterday', r'week', r'month',
        r'history', r'period', r'interval'
    ]
    
    # Enhanced metric/aggregation patterns
    metric_patterns = [
        r'metric', r'measure', r'value',
        r'how many', r'average', r'mean',
        r'total', r'peak', r'maximum', 
        r'minimum', r'count', r'sum',
        r'usage', r'level', r'rate',
        r'progression', r'trend', r'change'
    ]
    
    # Enhanced entity relationship patterns
    entity_patterns = [
        r'related', r'connected', r'between',
        r'relationship', r'dependency', r'link',
        r'associated', r'correlation'
    ]
    
    # Enhanced state transition patterns
    state_patterns = [
        r'state', r'status', r'changed',
        r'transition', r'switch', r'moved',
        r'became', r'turned'
    ]
    
    print(f"\nDEBUG: Analyzing query: '{query}'")
    
    # Check temporal patterns
    temporal_matches = [p for p in temporal_patterns if p in query_lower]
    if temporal_matches:
        intents.add('temporal')
        print(f"DEBUG: Temporal matches: {temporal_matches}")
    
    # Check metric patterns
    metric_matches = [p for p in metric_patterns if p in query_lower]
    if metric_matches:
        intents.add('aggregation')
        print(f"DEBUG: Metric matches: {metric_matches}")
    
    # Check entity patterns
    entity_matches = [p for p in entity_patterns if p in query_lower]
    if entity_matches:
        intents.add('entity')
        print(f"DEBUG: Entity matches: {entity_matches}")
    
    # Check state patterns
    state_matches = [p for p in state_patterns if p in query_lower]
    if state_matches:
        intents.add('state')
        print(f"DEBUG: State matches: {state_matches}")
    
    # Determine primary intent with enhanced priority logic
    if not intents:
        primary = 'general'
        intents.add('general')
        print("DEBUG: No specific intent detected, using 'general'")
    else:
        # Priority order considers combined intents
        if 'temporal' in intents and 'aggregation' in intents:
            # For queries about metric changes over time
            primary = 'temporal' if 'progression' in query_lower else 'aggregation'
        else:
            # Standard priority order
            for intent_type in ['temporal', 'aggregation', 'entity', 'state', 'general']:
                if intent_type in intents:
                    primary = intent_type
                    break
    
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
        dict: Extracted filters
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
        str: Customized system prompt
    """
    base_prompt = "You are a helpful assistant that answers questions based on the provided context."
    
    if query_intent['primary_intent'] == 'temporal':
        return base_prompt + " Focus on temporal relationships and event sequences."
    elif query_intent['primary_intent'] == 'aggregation':
        return base_prompt + " Focus on numerical patterns and trends."
    elif query_intent['primary_intent'] == 'entity':
        return base_prompt + " Focus on entity relationships and attributes."
    elif query_intent['primary_intent'] == 'state':
        return base_prompt + " Focus on state transitions and system conditions."
        
    return base_prompt
