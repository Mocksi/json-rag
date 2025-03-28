"""
Query Intent Analysis Module

This module handles the analysis and classification of natural language queries
to determine their intent, extract relevant parameters, and guide the retrieval
process. It supports multiple types of queries and can extract various filters
and conditions.

Features:
    - Intent Classification
    - Multi-intent classification
    - Parameter Extraction
    - Filter Detection
    - Time Range Analysis
    - Entity Reference Detection

Usage:
    >>> from app.analysis.intent import analyze_query_intent
    >>> query = "Show me all orders from last week with status pending"
    >>> intent = analyze_query_intent(query)
    >>> print(f"Primary intent: {intent['primary']}")
"""

import re
from datetime import datetime
from app.utils.logging_config import get_logger
from typing import Optional, Dict
# From your original code: extract_time_range, extract_metric_conditions, extract_pagination_info, extract_entity_references, analyze_query_intent

logger = get_logger(__name__)

# Define query intent patterns
patterns = {
    "relationship": [
        r"related to",
        r"connected with",
        r"linked to",
        r"associated with",
        r"depends on",
        r"references",
        r"relationship between",
    ],
    "temporal": [
        r"(?:on\s+)?\d{4}-\d{2}-\d{2}",
        r"before\s+\d{1,2}:\d{2}\s*(?:am|pm)?",
        r"after\s+\d{1,2}:\d{2}\s*(?:am|pm)?",
        r"between\s+\d{1,2}:\d{2}\s*(?:am|pm)?\s+and\s+\d{1,2}:\d{2}\s*(?:am|pm)?",
        r"between\s+\d{4}-\d{2}-\d{2}\s+and\s+\d{4}-\d{2}-\d{2}",
        r"last\s+\d+\s+(?:day|week|month|year)s?",
        r"this (?:week|month|year)",
        r"since\s+\d{4}-\d{2}-\d{2}",
        r"before\s+\d{4}-\d{2}-\d{2}",
        r"after\s+\d{4}-\d{2}-\d{2}",
        r"on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?",
        r"from\s+\d{4}-\d{2}-\d{2}",
        r"to\s+\d{4}-\d{2}-\d{2}",
    ],
    "metric": [
        r"average|mean|avg",
        r"maximum|max|highest|peak",
        r"minimum|min|lowest|bottom",
        r"sum|total",
        r"count|number of|how many",
        r"greater than|more than|above",
        r"less than|under|below",
        r"equal to|exactly",
        r"value|amount|cost",
    ],
    "risk": [
        r"risk|danger|hazard",
        r"warning|alert|critical",
        r"threshold|limit|boundary",
        r"violation|breach|exceed",
        r"compliance|conform|adhere",
    ],
}


def determine_primary_intent(intents: dict) -> str:
    """
    Determine the primary intent from a set of detected intents.

    This function implements a priority-based approach where certain intents
    take precedence over others based on their importance and specificity.

    Args:
        intents (dict): Dictionary of detected intents and their confidence scores

    Returns:
        str: The primary intent category
    """
    # Check for combined temporal-metric queries
    if "temporal" in intents and "metric" in intents:
        return "temporal_metric"

    # Priority order for single intents
    priority_order = ["risk", "temporal", "metric", "relationship"]

    # Return the first matching intent based on priority
    for intent in priority_order:
        if intent in intents:
            return intent

    return "general"


def extract_time_range(query: str) -> Optional[Dict]:
    """Extract temporal range information from a query string."""
    query_lower = query.lower()

    # Add support for more natural language date formats
    month_names = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    # First try to extract the date
    extracted_date = None

    # Enhanced pattern for natural language dates
    for month_name, month_num in month_names.items():
        patterns = [
            f"{month_name}\\s+(\\d{{1,2}})(?:st|nd|rd|th)?,?\\s*(\\d{{4}})?",  # March 18, 2024
            f"{month_name}\\s+(\\d{{1,2}})(?:st|nd|rd|th)?",  # March 18
            f"(\\d{{1,2}})(?:st|nd|rd|th)?\\s+{month_name}",  # 18 March
        ]

        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                day = int(match.group(1))
                year = (
                    int(match.group(2))
                    if len(match.groups()) > 1 and match.group(2)
                    else datetime.now().year
                )

                try:
                    extracted_date = datetime(year, month_num, day)
                    break
                except ValueError:
                    continue
        if extracted_date:
            break

    if not extracted_date:
        return None

    # Now look for time constraints
    time_pattern = r"(before|after)\s+(\d{1,2}):?(\d{2})?\s*(am|pm|AM|PM)?"
    time_match = re.search(time_pattern, query_lower)

    if time_match:
        direction = time_match.group(1)  # 'before' or 'after'
        hour = int(time_match.group(2))
        minutes = int(time_match.group(3)) if time_match.group(3) else 0
        meridiem = time_match.group(4)

        # Convert to 24-hour format if AM/PM specified
        if meridiem:
            if meridiem.lower() == "pm" and hour != 12:
                hour += 12
            elif meridiem.lower() == "am" and hour == 12:
                hour = 0

        target_time = extracted_date.replace(hour=hour, minute=minutes)

        if direction == "before":
            return {
                "start": extracted_date.replace(
                    hour=0, minute=0, second=0, microsecond=0
                ),
                "end": target_time.replace(second=59, microsecond=999999),
                "type": "exact",
            }
        else:  # after
            return {
                "start": target_time,
                "end": extracted_date.replace(
                    hour=23, minute=59, second=59, microsecond=999999
                ),
                "type": "exact",
            }

    # If no time constraint, return full day range
    return {
        "start": extracted_date.replace(hour=0, minute=0, second=0, microsecond=0),
        "end": extracted_date.replace(
            hour=23, minute=59, second=59, microsecond=999999
        ),
        "type": "exact",
    }


def extract_metric_conditions(query: str) -> dict:
    """
    Extract metric-related conditions from a query string.

    This function analyzes natural language queries to identify metric-related
    operations, including aggregations, specific metrics, and comparison
    conditions. It supports various ways of expressing metric operations
    in natural language.

    Args:
        query (str): User query string containing metric-related expressions

    Returns:
        dict: Metric conditions containing:
            - aggregation (str): Type of aggregation operation
                - 'average': Mean value calculation
                - 'maximum': Highest value
                - 'minimum': Lowest value
                - 'sum': Total value
                - 'count': Number of items
            - metric (str): Name or identifier of the metric
            - comparison (dict): Comparison criteria
                - 'type': Comparison operator (gt, lt, eq, etc.)
                - 'value': Numeric value for comparison
                - 'unit': Optional unit of measurement

    Supported Patterns:
        Aggregations:
            - "average/mean/avg response time"
            - "maximum/max/highest/peak value"
            - "minimum/min/lowest/bottom score"
            - "sum/total of values"
            - "count/number of records"

        Metric Identification:
            - "of [metric_name]"
            - "[metric_name] values"
            - "[metric_name] is"
            - "where [metric_name]"

        Comparisons:
            - "greater than/more than/above X"
            - "less than/under/below X"
            - "equal to/exactly X"

    Examples:
        >>> extract_metric_conditions("Show average response time above 100ms")
        {
            'aggregation': 'average',
            'metric': 'response_time',
            'comparison': {
                'type': 'gt',
                'value': 100,
                'unit': 'ms'
            }
        }

        >>> extract_metric_conditions("Count number of errors per day")
        {
            'aggregation': 'count',
            'metric': 'errors',
            'group_by': 'day'
        }

    Note:
        - The function attempts to identify the most specific metric name possible
        - Comparison values are extracted with their units when present
        - Multiple conditions in the same query are resolved by priority
        - If no specific metric is identified, returns None for that field
    """
    conditions = {}
    query_lower = query.lower()
    agg_patterns = {
        "average": ["average", "mean", "avg"],
        "maximum": ["maximum", "max", "highest", "peak"],
        "minimum": ["minimum", "min", "lowest", "bottom"],
        "sum": ["sum", "total"],
        "count": ["count", "number of", "how many"],
    }
    for agg_type, patterns in agg_patterns.items():
        if any(p in query_lower for p in patterns):
            conditions["aggregation"] = agg_type
            break
    metric_patterns = [
        r"(?:of|in|for)\s+(\w+)",
        r"(\w+)\s+(?:values|measurements)",
        r"(\w+)\s+(?:is|are|was|were)",
    ]
    for pattern in metric_patterns:
        match = re.search(pattern, query_lower)
        if match:
            metric = match.group(1)
            if metric not in ["the", "a", "an"]:
                conditions["metric"] = metric
                break
    comp_patterns = {
        "greater_than": [r"greater than (\d+)", r"more than (\d+)", r"above (\d+)"],
        "less_than": [r"less than (\d+)", r"under (\d+)", r"below (\d+)"],
        "equals": [r"equal to (\d+)", r"exactly (\d+)"],
    }
    for comp_type, pats in comp_patterns.items():
        for pat in pats:
            match = re.search(pat, query_lower)
            if match:
                conditions["comparison"] = {
                    "type": comp_type,
                    "value": float(match.group(1)),
                }
                break
        if "comparison" in conditions:
            break
    return conditions if conditions else None


def extract_pagination_info(query: str) -> dict:
    """
    Extract pagination-related information from a query string.

    This function analyzes natural language queries to identify pagination
    requests, including page numbers, navigation directions, and result
    limits. It supports various ways of expressing pagination in natural
    language.

    Args:
        query (str): User query string containing pagination expressions

    Returns:
        dict: Pagination information containing:
            - type (str): Pagination type
                - 'absolute': Direct page number reference
                - 'relative': Next/previous navigation
                - 'limit': Result count limitation
            - page (int, optional): Page number for absolute pagination
            - direction (str, optional): Navigation direction
                - 'next': Forward navigation
                - 'previous': Backward navigation
            - size (int, optional): Number of results per page
            - offset (int, optional): Number of results to skip

    Supported Patterns:
        Absolute Pagination:
            - "page 5"
            - "go to page 3"
            - "show page 10"

        Relative Navigation:
            - "next page"
            - "previous page"
            - "show next"
            - "go back"

        Result Limiting:
            - "show 10 results"
            - "limit to 20 items"
            - "first 5 entries"

    Examples:
        >>> extract_pagination_info("show page 5")
        {
            'type': 'absolute',
            'page': 5,
            'size': 10  # default page size
        }

        >>> extract_pagination_info("next page")
        {
            'type': 'relative',
            'direction': 'next',
            'size': 10  # default page size
        }

        >>> extract_pagination_info("show 20 results")
        {
            'type': 'limit',
            'size': 20
        }

    Note:
        - Page numbers start from 1
        - Default page size is used when not specified
        - Invalid page numbers or sizes return None
        - Relative navigation requires current page context
    """
    query_lower = query.lower()
    match = re.search(r"page\s+(\d+)", query_lower)
    if match:
        return {"type": "absolute", "page": int(match.group(1))}
    if "next page" in query_lower:
        return {"type": "relative", "direction": "next"}
    if "previous page" in query_lower or "prev page" in query_lower:
        return {"type": "relative", "direction": "previous"}
    match = re.search(r"show\s+(\d+)\s+(?:results|items)", query_lower)
    if match:
        return {"type": "limit", "size": int(match.group(1))}
    return None


def extract_entity_references(query: str) -> dict:
    """
    Extract entity references and filters from a query string.

    This function analyzes natural language queries to identify references
    to entities, their attributes, and relationships. It supports multiple
    formats for specifying entity information and filters.

    Args:
        query (str): User query string containing entity references

    Returns:
        dict: Entity references containing:
            - explicit (dict): Key-value pairs from explicit assignments
                Example: {"name": "john", "role": "admin"}
            - typed (dict): References with type specifications
                Example: {"user": "john", "status": "active"}
            - ids (list): Extracted entity IDs
                Example: ["123", "456"]
            - implicit (dict): Contextually inferred references
                Example: {"category": "books", "owner": "current_user"}

    Supported Patterns:
        Explicit Assignments:
            - "name=john"
            - "status=active"
            - "category=books"

        Typed References:
            - "user:john"
            - "status:active"
            - "type:document"

        ID References:
            - "#123"
            - "id:456"
            - "document #789"

        Entity-Specific:
            - "user john"
            - "status active"
            - "in category books"
            - "by owner alice"

    Examples:
        >>> extract_entity_references("show documents by user:john with status=active")
        {
            'explicit': {'status': 'active'},
            'typed': {'user': 'john'},
            'implicit': {'type': 'document'}
        }

        >>> extract_entity_references("find order #123 and related items")
        {
            'ids': ['123'],
            'implicit': {'type': 'order', 'include': 'items'}
        }

    Note:
        - Entity references are case-insensitive
        - Multiple references to the same entity type are collected in lists
        - Ambiguous references are resolved using context
        - Invalid or malformed references are ignored
        - Special characters in values are properly escaped
    """
    references = {}
    for token in query.split():
        if "=" in token:
            key, value = token.split("=", 1)
            references[key.strip()] = value.strip()
    typed_refs = re.finditer(r"(\w+):(\w+)", query)
    for m in typed_refs:
        references[m.group(1)] = m.group(2)
    id_match = re.search(r"#(\d+)", query)
    if id_match:
        references["id"] = id_match.group(1)
    entity_patterns = {
        "user": [r"user (\w+)", r"by (\w+)"],
        "status": [r"status (\w+)", r"state (\w+)"],
        "category": [r"in (\w+)", r"category (\w+)"],
    }
    for entity_type, pats in entity_patterns.items():
        for pat in pats:
            match = re.search(pat, query.lower())
            if match:
                references[entity_type] = match.group(1)
                break
    return references if references else None


def analyze_query_intent(query: str) -> dict:
    """Analyze natural language query to determine primary and secondary intents."""
    query_lower = query.lower()
    intents = set()

    # Check for temporal patterns first
    temporal_matches = [p for p in patterns["temporal"] if re.search(p, query_lower)]
    if temporal_matches:
        logger.debug(f"Temporal matches: {temporal_matches}")
        intents.add("temporal")

    # Check for metric patterns
    metric_matches = [p for p in patterns["metric"] if re.search(p, query_lower)]
    if metric_matches:
        logger.debug(f"Metric matches: {metric_matches}")
        intents.add("metric")

    # Check for relationship patterns
    relationship_matches = [
        p for p in patterns["relationship"] if re.search(p, query_lower)
    ]
    if relationship_matches:
        logger.debug(f"Relationship matches: {relationship_matches}")
        intents.add("relationship")

    # Check for risk patterns
    risk_matches = [p for p in patterns["risk"] if re.search(p, query_lower)]
    if risk_matches:
        logger.debug(f"Risk matches: {risk_matches}")
        intents.add("risk")

    # Special case: If we have a date/time and any value-related terms, it's temporal_metric
    has_date = bool(
        re.search(
            r"\d{4}-\d{2}-\d{2}|(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}",
            query_lower,
        )
    )
    has_value_terms = bool(
        re.search(r"total|value|amount|sum|cost|price|revenue|sales", query_lower)
    )

    # Also check for month names with just the day
    month_day = bool(
        re.search(
            r"(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?",
            query_lower,
        )
    )

    if (has_date or month_day) and has_value_terms:
        intents.add("temporal")
        intents.add("metric")
        logger.debug(
            "Detected combined temporal-metric query based on date and value terms"
        )

    # Determine primary intent
    primary_intent = determine_primary_intent(intents)
    result = {"primary_intent": primary_intent, "all_intents": list(intents)}
    logger.debug(f"Final intent analysis: {result}")
    return result


def extract_filters_from_query(query: str) -> dict:
    """
    Extract key-value filters and conditions from a query string.

    This function parses natural language queries to identify filtering
    conditions expressed in various formats. It supports both explicit
    and implicit filter specifications.

    Args:
        query (str): Natural language query containing filter conditions

    Returns:
        dict: Extracted filters containing:
            - explicit (dict): Direct key-value specifications
            - implicit (dict): Inferred filter conditions
            - ranges (dict): Numeric and temporal ranges
            - negation (list): Excluded values

    Supported Formats:
        Explicit Filters:
            - "status=active"
            - "priority=high"
            - "type=document"

        Implicit Filters:
            - "active documents"
            - "high priority"
            - "completed tasks"

        Range Filters:
            - "price between 10 and 20"
            - "age > 25"
            - "count <= 100"

        Negation:
            - "not status=deleted"
            - "exclude category=draft"
            - "without tag=temp"

    Examples:
        >>> extract_filters_from_query("show active tasks with priority=high")
        {
            'explicit': {'priority': 'high'},
            'implicit': {'status': 'active', 'type': 'task'}
        }

        >>> extract_filters_from_query("price between 10 and 20 not category=draft")
        {
            'ranges': {'price': {'min': 10, 'max': 20}},
            'negation': {'category': 'draft'}
        }

    Note:
        - Filters are case-insensitive
        - Multiple values for the same key are collected in lists
        - Range values are normalized to consistent units
        - Implicit filters are extracted based on context
        - Negation is preserved in the filter structure
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
    if query_intent["primary_intent"] == "temporal":
        return (
            base_prompt
            + " Focus on temporal relationships and event sequences that are explicitly documented in the data."
        )
    elif query_intent["primary_intent"] == "relationship":
        return (
            base_prompt
            + " Focus on relationships and connections that are directly evidenced in the data structure."
        )
    elif query_intent["primary_intent"] == "metric":
        return (
            base_prompt
            + " Focus on numerical patterns and trends that are explicitly present in the data."
        )
    elif query_intent["primary_intent"] == "risk":
        return (
            base_prompt
            + " Focus on risk indicators and warning signs that are directly present in the data."
        )

    return base_prompt
