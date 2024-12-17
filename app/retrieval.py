from typing import List, Dict, Optional, Tuple
import json
from collections import defaultdict
from datetime import datetime, timedelta
import re
import statistics
import openai

from app.config import MAX_CHUNKS
from app.utils import parse_timestamp
from app.embedding import get_embedding, vector_search_with_filter
from app.parsing import json_to_path_chunks, extract_entities
from app.intent import analyze_query_intent, extract_filters_from_query, get_system_prompt
from app.database import get_files_to_process, upsert_file_metadata, init_db, get_chunk_archetypes, get_chunk_relationships
from app.config import embedding_model
from app.utils import get_json_files
from app.archetype import ArchetypeDetector

# Include your retrieval logic: get_relevant_chunks, hybrid_retrieval, hierarchical_retrieval, etc.
# We'll place get_relevant_chunks and others here:

def get_relevant_chunks(conn, query: str, top_k: int = 5) -> List[Dict]:
    """Get relevant chunks with enriched context including multi-level relationships."""
    cur = conn.cursor()
    
    # Get query archetype
    detector = ArchetypeDetector()
    query_archetype = detector.detect_archetypes({'query': query})
    query_embedding = get_embedding(query, query_archetype)
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    
    # Enhanced SQL query with archetype-aware relationship traversal and scoring
    cur.execute("""
        WITH RECURSIVE relationship_chain AS (
            -- Base case: direct relationships
            SELECT 
                source_chunk_id,
                target_chunk_id,
                relationship_type,
                metadata,
                ARRAY[source_chunk_id]::varchar[] as path,
                1 as depth,
                -- Track archetype path
                ARRAY[(
                    SELECT archetype::varchar 
                    FROM chunk_archetypes 
                    WHERE chunk_id = source_chunk_id 
                    ORDER BY confidence DESC 
                    LIMIT 1
                )]::varchar[] as archetype_path,
                -- Add archetype-based scoring
                CASE 
                    WHEN EXISTS (
                        SELECT 1 FROM chunk_archetypes 
                        WHERE chunk_id = source_chunk_id 
                        AND archetype = %s
                    ) THEN 0.8
                    ELSE 0.5
                END as archetype_score
            FROM chunk_relationships
            
            UNION ALL
            
            -- Recursive case
            SELECT 
                r.source_chunk_id,
                r.target_chunk_id,
                r.relationship_type,
                r.metadata,
                (rc.path || r.target_chunk_id)::varchar[],
                rc.depth + 1,
                (rc.archetype_path || (
                    SELECT archetype::varchar 
                    FROM chunk_archetypes 
                    WHERE chunk_id = r.target_chunk_id 
                    ORDER BY confidence DESC 
                    LIMIT 1
                ))::varchar[],
                rc.archetype_score * 0.9  -- Decay score with depth
            FROM chunk_relationships r
            JOIN relationship_chain rc ON r.source_chunk_id = rc.target_chunk_id
            WHERE rc.depth < CASE 
                WHEN rc.archetype_path[array_length(rc.archetype_path, 1)] = 'entity_definition' THEN 4
                WHEN rc.archetype_path[array_length(rc.archetype_path, 1)] = 'event' THEN 3
                WHEN rc.archetype_path[array_length(rc.archetype_path, 1)] = 'metric' THEN 2
                ELSE 2
            END
            AND NOT r.target_chunk_id = ANY(rc.path)
        )
        SELECT 
            c.id,
            c.chunk_json,
            c.metadata,
            (c.embedding <=> %s::vector) * COALESCE(MIN(rc.archetype_score), 1.0) as distance,
            array_agg(DISTINCT jsonb_build_object(
                'type', rc.relationship_type,
                'target', rc.target_chunk_id,
                'metadata', rc.metadata,
                'archetype_path', rc.archetype_path
            )) FILTER (WHERE rc.relationship_type IS NOT NULL) as relationships
        FROM json_chunks c
        LEFT JOIN relationship_chain rc ON c.id = rc.source_chunk_id
        GROUP BY c.id, c.chunk_json, c.metadata, c.embedding
        ORDER BY distance ASC
        LIMIT %s
    """, (
        query_archetype[0][0] if query_archetype else 'unknown',  # First archetype type
        embedding_str,
        top_k
    ))
    
    results = cur.fetchall()
    
    # Format results with enriched context
    enriched_chunks = []
    for id, chunk_json, metadata, distance, relationships in results:
        enriched_chunk = {
            'id': id,
            'content': chunk_json,
            'relevance_score': 1 - distance,
            'metadata': metadata,
            'relationships': relationships if relationships else []
        }
        enriched_chunks.append(enriched_chunk)
        print(f"\nDEBUG: Retrieved chunk {id} with {len(relationships) if relationships else 0} relationships")
    
    cur.close()
    return enriched_chunks

def build_prompt(query, retrieved_chunks, query_intent):
    """Build a prompt for the LLM using retrieved context and query intent."""
    
    # Format chunks based on content type
    formatted_chunks = []
    for chunk in retrieved_chunks:
        if isinstance(chunk, dict):
            if chunk.get('type') == 'metric_aggregation':
                # Format aggregated metrics
                metrics = chunk.get('content', {})
                formatted_chunks.append("Aggregated Metrics:")
                for metric, stats in metrics.items():
                    formatted_chunks.append(f"- {metric}:")
                    for stat, value in stats.items():
                        formatted_chunks.append(f"  {stat}: {value}")
            
            elif chunk.get('type') == 'relationship_graph':
                # Format relationship data
                relationships = chunk.get('content', {})
                formatted_chunks.append("Entity Relationships:")
                for entity_id, relations in relationships.items():
                    formatted_chunks.append(f"- Entity {entity_id} connections:")
                    for rel in relations:
                        formatted_chunks.append(f"  {rel['type']} -> {rel['related_id']}")
            
            else:
                # Format regular content
                content = chunk.get('content')
                if content:
                    formatted_chunks.append(json.dumps(content, indent=2))
        else:
            formatted_chunks.append(str(chunk))

    context_str = "\n\n".join(formatted_chunks)
    
    # Build intent-specific guidelines
    guidelines = [
        "- Use specific details from the context",
        "- Always use human-readable names in addition to IDs",
        "- If information isn't in the context, say so"
    ]
    
    if 'temporal' in query_intent['all_intents']:
        guidelines.extend([
            "- Preserve chronological order",
            "- Include specific dates and times when available"
        ])
        
    if 'aggregation' in query_intent['all_intents']:
        guidelines.extend([
            "- Include relevant metrics and their values",
            "- Highlight significant patterns or trends"
        ])
        
    if 'entity' in query_intent['all_intents']:
        guidelines.extend([
            "- Show relationships between entities",
            "- Include relevant entity attributes"
        ])

    prompt = f"""Use the provided context to answer the user's query.

Query Intent: {query_intent['primary_intent']}
Additional Intents: {', '.join(query_intent['all_intents'])}

Guidelines:
{chr(10).join(guidelines)}

Context:
{context_str}

Question: {query}

Answer based only on the provided context, using human-readable names."""

    print("\nDEBUG: Prompt sent to OpenAI:")
    print("=" * 80)
    print(prompt)
    print("=" * 80)
    
    return prompt

def summarize_chunks(chunks):
    """
    Summarize chunks while preserving temporal relationships.
    
    Args:
        chunks (list): List of chunk texts to summarize
        
    Returns:
        str: Summarized text
    """
    print("\nDEBUG: Summarizing chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}:")
        print("-" * 40)
        print(chunk)
        print("-" * 40)
    
    prompt = """Summarize these chunks of context, preserving:
1. Chronological order of events
2. Specific dates and times
3. Key temporal relationships

Context:
""" + "\n\n".join(chunks)

    print("\nDEBUG: Sending summarization prompt to OpenAI")
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You summarize text while preserving chronological order and temporal relationships."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=300
    )
    return completion.choices[0].message.content.strip()

def summarize_chunk_group(chunks, parent_context):
    import openai
    try:
        parent_data = json.loads(parent_context)
        context_type = parent_data.get('context', {}).get('type', 'unknown')
        
        prompt = f"""Summarize these related chunks while preserving their structure.
        
Parent context: {json.dumps(parent_data.get('context', {}))}
Content type: {context_type}

Guidelines:
1. Maintain relationships between elements
2. Preserve key identifiers and references
3. Keep critical metadata
4. Highlight important patterns

Content to summarize:
{json.dumps(chunks, indent=2)}
"""
        
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You create structured summaries of JSON content while preserving relationships and context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        return completion.choices[0].message.content
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return summarize_chunks(chunks)

def get_chunk_by_id(conn, chunk_id):
    cur = conn.cursor()
    cur.execute("SELECT chunk_text::text FROM json_chunks WHERE id = %s", (chunk_id,))
    result = cur.fetchone()
    return result[0] if result else None

def group_by_parent(conn, chunks):
    cur = conn.cursor()
    chunk_ids = [json.loads(c)['id'] for c in chunks]
    format_str = ','.join(['%s'] * len(chunk_ids))
    cur.execute(f"""
        SELECT parent_chunk_id, child_chunk_id, c.chunk_text
        FROM chunk_hierarchy h
        JOIN json_chunks c ON c.id = h.child_chunk_id
        WHERE child_chunk_id IN ({format_str})
    """, tuple(chunk_ids))
    
    parent_map = defaultdict(list)
    for parent_id, child_id, chunk_text in cur.fetchall():
        parent_map[parent_id].append(chunk_text)
    
    return parent_map

def hierarchical_retrieval(conn, query, top_k=20):
    initial_chunks = get_relevant_chunks(conn, query, top_k=top_k)
    if len(initial_chunks) <= MAX_CHUNKS:
        return initial_chunks
    parent_map = group_by_parent(conn, initial_chunks)
    summarized = []
    for parent_id, children in parent_map.items():
        if len(children) > MAX_CHUNKS:
            parent_context = get_chunk_by_id(conn, parent_id)
            summary = summarize_chunk_group(children, parent_context)
            summarized.append(summary)
        else:
            summarized.extend(children)
    if len(summarized) > MAX_CHUNKS:
        return [summarize_chunks(summarized)]
    return summarized

def temporal_retrieval(conn, query, top_k=20):
    cur = conn.cursor()
    base_query = """
    WITH base_data AS (
        SELECT DISTINCT 
            c.chunk_text,
            COALESCE(
                (c.chunk_json -> 'temporal_metadata' ->> 'timestamp')::timestamp,
                (c.chunk_json -> 'value' ->> 'value')::timestamp
            ) as event_time
        FROM json_chunks c
        WHERE (
            c.chunk_json -> 'value' ->> 'value' ~ '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' 
            OR c.chunk_json -> 'temporal_metadata' ->> 'timestamp' IS NOT NULL
        )
    )
    SELECT chunk_text::text
    FROM base_data
    WHERE event_time IS NOT NULL
    ORDER BY event_time DESC
    LIMIT %s
    """
    cur.execute(base_query, (top_k,))
    results = cur.fetchall()
    return [r[0] for r in results]

def metric_retrieval(conn, query: str, top_k: int = 5) -> List[str]:
    """Retrieve and format metric data from relevant chunks."""
    
    chunks = get_relevant_chunks(conn, query, top_k)
    context_entries = []
    seen_entries = set()
    
    # Common metric indicators from archetype detector
    metric_indicators = {
        'value', 'amount', 'count', 'total', 'quantity',
        'rate', 'percentage', 'score', 'ratio', 'stock',
        'reserved', 'available', 'balance', 'level'
    }
    
    def extract_metrics(obj, path="", parent_id=None):
        """Extract metrics recursively from any JSON structure."""
        if not isinstance(obj, dict):
            return
            
        # Extract ID and metric values
        current_id = None
        metrics = {}
        
        for key, value in obj.items():
            # Get ID if present
            if key.endswith('_id') and isinstance(value, str):
                current_id = value
            
            # Extract numeric values
            if isinstance(value, (int, float)):
                # Check if key indicates a metric
                if any(indicator in key.lower() for indicator in metric_indicators):
                    metrics[key] = value
            
            # Process nested structures
            elif isinstance(value, dict):
                extract_metrics(value, f"{path}.{key}", current_id or parent_id)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        extract_metrics(item, f"{path}[{i}]", current_id or parent_id)
        
        # Create entry if we found metrics
        if metrics:
            id_str = current_id or parent_id or "Unknown"
            entry = f"Entity {id_str}: "
            metrics_list = []
            for key, value in metrics.items():
                # Clean up key name
                clean_key = key.replace('_', ' ').strip()
                metrics_list.append(f"{clean_key}: {value}")
            
            entry += ", ".join(metrics_list)
            if entry not in seen_entries:
                seen_entries.add(entry)
                context_entries.append(entry)
    
    # Process each chunk
    for chunk in chunks:
        if isinstance(chunk.get('content'), dict):
            extract_metrics(chunk['content'])
    
    print(f"\nDEBUG: Generated {len(context_entries)} metric entries")
    return context_entries

def summarize_by_timewindow(retrieved_texts, window_size=timedelta(hours=1)):
    if not retrieved_texts:
        return []
    from app.parsing import parse_timestamp
    windows = defaultdict(list)
    skipped = []
    for text in retrieved_texts:
        try:
            data = json.loads(text)
            timestamp = None
            if 'temporal_metadata' in data:
                timestamp = data['temporal_metadata'].get('timestamp')
            if not timestamp and 'context' in data:
                timestamp = data['context'].get('timestamp')
            if timestamp:
                parsed_time = parse_timestamp(timestamp)
                if parsed_time:
                    window_key = parsed_time.replace(minute=0, second=0, microsecond=0)
                    windows[window_key].append(text)
                else:
                    skipped.append(text)
            else:
                skipped.append(text)
        except (json.JSONDecodeError, AttributeError) as e:
            skipped.append(text)
    summarized = []
    for window_time, texts in sorted(windows.items()):
        if len(texts) > 1:
            try:
                summary = summarize_chunks(texts)
                if summary:
                    summarized.append(summary)
            except Exception as e:
                summarized.extend(texts)
        else:
            summarized.extend(texts)
    summarized.extend(skipped)
    return summarized

def create_metric_summary(metric_name, texts):
    from app.utils import parse_timestamp
    values = []
    timestamps = []
    contexts = []
    for text in texts:
        try:
            data = json.loads(text)
            value = data.get('value', {}).get('value')
            if value is not None:
                values.append(value)
                if 'temporal_metadata' in data:
                    timestamp = data['temporal_metadata'].get('timestamp')
                    if timestamp:
                        timestamps.append(parse_timestamp(timestamp))
                if 'context' in data:
                    contexts.append(data['context'])
        except:
            continue
    if not values:
        return None
    try:
        mean_val = sum(values)/len(values)
        from app.retrieval import summarize_contexts
        summary = {
            'metric_name': metric_name,
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': mean_val,
            'median': statistics.median(values) if len(values) > 1 else values[0],
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
            'timestamps': [t.isoformat() for t in sorted(timestamps) if t] if timestamps else None,
            'context_summary': summarize_contexts(contexts) if contexts else None
        }
        return json.dumps(summary)
    except Exception as e:
        return None

def summarize_metrics(retrieved_texts):
    """Removed - no longer needed as metric_retrieval handles everything"""
    return retrieved_texts

def summarize_contexts(contexts):
    if not contexts:
        return {}
    summary = defaultdict(list)
    for context in contexts:
        for key, value in context.items():
            if value not in summary[key]:
                summary[key].append(value)
    return {k: v[0] if len(v)==1 else v for k,v in summary.items()}

def detect_state_transitions(states):
    # Move your detect_state_transitions code here if needed
    if not states:
        return []
    transitions = []
    prev_state = None
    sorted_states = sorted(states, key=lambda x: parse_timestamp(x.get('timestamp')) or datetime.max)
    for state in sorted_states:
        current_state = state.get('states')
        if not current_state:
            continue
        if prev_state and current_state != prev_state:
            transition = {
                'from': prev_state,
                'to': current_state,
                'timestamp': state.get('timestamp'),
                'context': state.get('context', {}),
                'duration': calculate_transition_duration(prev_state, current_state, state.get('timestamp'))
            }
            transitions.append(transition)
        prev_state = current_state
    return transitions

def calculate_transition_duration(from_state, to_state, timestamp):
    # From your code
    if not (from_state and to_state and timestamp):
        return None
    t = parse_timestamp(timestamp)
    if not t:
        return None
    return 0.0  # placeholder

def answer_query(conn, query):
    filters = extract_filters_from_query(query)
    query_intent = analyze_query_intent(query)
    
    if filters:
        retrieved_texts = hybrid_retrieval(conn, query, filters, top_k=MAX_CHUNKS)
    elif query_intent['primary_intent'] == 'temporal':
        retrieved_texts = temporal_retrieval(conn, query, top_k=MAX_CHUNKS)
    elif query_intent['primary_intent'] == 'aggregation':
        retrieved_texts = metric_retrieval(conn, query, top_k=MAX_CHUNKS)
    else:
        retrieved_texts = get_relevant_chunks(conn, query, top_k=MAX_CHUNKS)
    
    if len(retrieved_texts) > MAX_CHUNKS:
        if query_intent['primary_intent'] == 'temporal':
            retrieved_texts = summarize_by_timewindow(retrieved_texts)
        elif 'aggregation' in query_intent['all_intents']:
            retrieved_texts = summarize_metrics(retrieved_texts)
        else:
            retrieved_texts = hierarchical_retrieval(conn, query, top_k=20)
    
    prompt = build_prompt(query, retrieved_texts, query_intent)
    
    from app.intent import get_system_prompt
    import openai
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system", 
                "content": get_system_prompt(query_intent)
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        temperature=0.0,
        max_tokens=300
    )
    return completion.choices[0].message.content.strip()

def hybrid_retrieval(conn, query, filters, top_k=20):
    """Combine filter-based and semantic search with archetype-aware scoring."""
    detector = ArchetypeDetector()
    query_archetypes = detector.detect_archetypes({'query': query})
    query_embedding = get_embedding(query, query_archetypes[0] if query_archetypes else None)
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    
    # Build filter conditions
    filter_conditions = []
    filter_values = []
    for key, value in filters.items():
        filter_conditions.append(f"kv.key = %s AND kv.value = %s")
        filter_values.extend([key, str(value)])
    
    hybrid_query = """
    WITH RECURSIVE relationship_chain AS (
        -- Base case with archetype context
        SELECT 
            source_chunk_id,
            target_chunk_id,
            relationship_type,
            metadata,
            ARRAY[source_chunk_id] as path,
            1 as depth,
            ARRAY[(
                SELECT archetype 
                FROM chunk_archetypes 
                WHERE chunk_id = source_chunk_id 
                ORDER BY confidence DESC 
                LIMIT 1
            )] as archetype_path,
            -- Initial relationship score
            CASE 
                WHEN relationship_type = 'reference' THEN 0.8
                WHEN relationship_type IN ('before', 'after', 'during') THEN 0.7
                WHEN relationship_type IN ('aggregates', 'breaks_down') THEN 0.9
                ELSE 0.5
            END as relationship_score
        FROM chunk_relationships
        
        UNION ALL
        
        -- Recursive case with archetype-aware scoring
        SELECT 
            r.source_chunk_id,
            r.target_chunk_id,
            r.relationship_type,
            r.metadata,
            rc.path || r.target_chunk_id,
            rc.depth + 1,
            rc.archetype_path || (
                SELECT archetype 
                FROM chunk_archetypes 
                WHERE chunk_id = r.target_chunk_id 
                ORDER BY confidence DESC 
                LIMIT 1
            ),
            -- Compound relationship score based on archetype rules
            rc.relationship_score * CASE 
                WHEN rc.archetype_path[array_length(rc.archetype_path, 1)] = 'entity_definition' 
                AND r.relationship_type = 'reference' THEN 0.9
                WHEN rc.archetype_path[array_length(rc.archetype_path, 1)] = 'event' 
                AND r.relationship_type IN ('before', 'after', 'during') THEN 0.8
                WHEN rc.archetype_path[array_length(rc.archetype_path, 1)] = 'metric' 
                AND r.relationship_type IN ('aggregates', 'breaks_down') THEN 0.95
                ELSE 0.6
            END
        FROM chunk_relationships r
        JOIN relationship_chain rc ON r.source_chunk_id = rc.target_chunk_id
        WHERE rc.depth < CASE 
            WHEN rc.archetype_path[array_length(rc.archetype_path, 1)] = 'entity_definition' THEN 4
            WHEN rc.archetype_path[array_length(rc.archetype_path, 1)] = 'event' THEN 3
            WHEN rc.archetype_path[array_length(rc.archetype_path, 1)] = 'metric' THEN 2
            ELSE 2
        END
        AND NOT r.target_chunk_id = ANY(rc.path)
    ),
    scored_chunks AS (
        SELECT 
            c.id,
            c.chunk_text,
            c.metadata,
            c.embedding <=> %s::vector as vector_score,
            COALESCE(
                (SELECT MAX(relationship_score) 
                FROM relationship_chain 
                WHERE source_chunk_id = c.id), 
                0.5
            ) as rel_score,
            CASE 
                WHEN c.metadata->'archetypes' @> %s::jsonb THEN 0.8
                ELSE 1.0 
            END as archetype_boost
        FROM json_chunks c
        WHERE EXISTS (
            SELECT 1 FROM chunk_key_values kv 
            WHERE kv.chunk_id = c.id 
            AND ({' OR '.join(filter_conditions)})
        )
    )
    SELECT 
        id,
        chunk_text,
        metadata,
        -- Combined scoring using vector similarity, relationship score, and archetype boost
        (vector_score * 0.4 + (1 - rel_score) * 0.3 + archetype_boost * 0.3) as final_score
    FROM scored_chunks
    ORDER BY final_score ASC
    LIMIT %s
    """
    
    query_params = filter_values + [
        embedding_str,
        json.dumps([{'type': a[0]} for a in query_archetypes]) if query_archetypes else '[]',
        top_k
    ]
    
    cur = conn.cursor()
    cur.execute(hybrid_query.format(' OR '.join(filter_conditions)), query_params)
    results = cur.fetchall()
    cur.close()
    
    return [
        {
            'id': r[0],
            'content': r[1],
            'metadata': r[2],
            'score': 1 - r[3]
        }
        for r in results
    ]

def analyze_related_chunks(conn, query: str, base_chunks: List[Dict]) -> List[Dict]:
    """
    Analyze relationships between chunks and perform cross-reference calculations.
    
    Args:
        conn: Database connection
        query: Original query string
        base_chunks: Initial retrieved chunks
        
    Returns:
        List of enriched chunks with relationship data
    """
    cur = conn.cursor()
    
    # Extract IDs and references from base chunks
    referenced_ids = set()
    for chunk in base_chunks:
        chunk_data = json.loads(chunk)
        for key, value in chunk_data.get('context', {}).items():
            if '_id' in key.lower() and isinstance(value, str):
                referenced_ids.add(value)
    
    # Define common comparison patterns
    comparison_patterns = {
        'lead_times': {
            'query': """
                SELECT 
                    c.id,
                    c.chunk_json ->> 'supplier_id' AS supplier_id,
                    (c.chunk_json -> 'lead_times' ->> 'actual')::float AS actual_lead_time,
                    (c.chunk_json -> 'lead_times' ->> 'contracted')::float AS contracted_lead_time,
                    c.chunk_json as full_data
                FROM json_chunks c
                WHERE c.chunk_json ->> 'type' = 'supplier_data'
                AND c.chunk_json ->> 'supplier_id' = ANY(%s)
            """,
            'analysis': lambda rows: [
                {
                    'supplier_id': row[1],
                    'discrepancy': row[2] - row[3],
                    'percent_difference': ((row[2] - row[3]) / row[3] * 100),
                    'actual': row[2],
                    'contracted': row[3],
                    'full_data': row[4]
                }
                for row in rows
            ]
        },
        'price_comparison': {
            'query': """
                SELECT 
                    c.id,
                    c.chunk_json ->> 'product_id' AS product_id,
                    (c.chunk_json -> 'pricing' ->> 'list_price')::float AS list_price,
                    (c.chunk_json -> 'pricing' ->> 'actual_price')::float AS actual_price,
                    c.chunk_json as full_data
                FROM json_chunks c
                WHERE c.chunk_json ->> 'type' = 'product_pricing'
                AND c.chunk_json ->> 'product_id' = ANY(%s)
            """,
            'analysis': lambda rows: [
                {
                    'product_id': row[1],
                    'discount': row[2] - row[3],
                    'discount_percentage': ((row[2] - row[3]) / row[2] * 100),
                    'list_price': row[2],
                    'actual_price': row[3],
                    'full_data': row[4]
                }
                for row in rows
            ]
        }
    }
    
    # Detect comparison type from query
    comparison_type = None
    if 'lead time' in query.lower() or 'delivery' in query.lower():
        comparison_type = 'lead_times'
    elif 'price' in query.lower() or 'cost' in query.lower():
        comparison_type = 'price_comparison'
    
    if comparison_type and referenced_ids:
        pattern = comparison_patterns[comparison_type]
        cur.execute(pattern['query'], (list(referenced_ids),))
        rows = cur.fetchall()
        
        if rows:
            analysis_results = pattern['analysis'](rows)
            print(f"\nDEBUG: Cross-reference analysis ({comparison_type}):")
            for result in analysis_results:
                print(f"  - {result}")
            
            # Enrich original chunks with analysis
            for chunk in base_chunks:
                chunk_data = json.loads(chunk)
                chunk_data['analysis'] = {
                    'type': comparison_type,
                    'results': analysis_results
                }
                chunk = json.dumps(chunk_data)
    
    cur.close()
    return base_chunks

def post_process_chunks(chunks: List[Dict], query_intent: Dict) -> List[Dict]:
    """Post-process retrieved chunks based on query intent."""
    
    def analyze_risk_metrics(chunk_list):
        """Analyze risk indicators and metrics across chunks."""
        risk_indicators = []
        
        for chunk in chunk_list:
            content = chunk.get('content', {})
            if not isinstance(content, dict):
                continue
                
            # Look for any alerts or warnings in the data
            alerts = []
            def extract_alerts(obj, path=[]):
                if isinstance(obj, dict):
                    if 'type' in obj and any(risk_word in str(obj.get('type', '')).lower() 
                                           for risk_word in ['alert', 'warning', 'risk']):
                        alerts.append({
                            'path': path,
                            'data': obj
                        })
                    for k, v in obj.items():
                        if isinstance(v, (dict, list)):
                            extract_alerts(v, path + [k])
                elif isinstance(obj, list):
                    for i, item in enumerate(v):
                        if isinstance(item, (dict, list)):
                            extract_alerts(item, path + [str(i)])
            
            extract_alerts(content)
            
            if alerts:
                risk_indicators.append({
                    'alerts': alerts,
                    'content': content
                })
        
        return [{'type': 'risk_analysis', 'content': risk_indicators}]
    
    # Apply processing based on intent
    primary_intent = query_intent['primary_intent']
    all_intents = query_intent.get('all_intents', [])
    
    if primary_intent == 'aggregation' or 'aggregation' in all_intents:
        return aggregate_metrics(chunks)
    elif primary_intent == 'temporal' or 'temporal' in all_intents:
        return process_temporal(chunks)
    elif primary_intent == 'entity' or 'relationship' in all_intents:
        return process_relationships(chunks)
    elif 'risk_analysis' in all_intents:
        return analyze_risk_metrics(chunks)
    
    # Default to original chunks if no specific processing needed
    return chunks

def format_response(chunks: List[Dict], query_archetype: str) -> Dict:
    """Format response based on archetype patterns."""
    if not query_archetype:
        return {'results': chunks}
        
    if query_archetype == 'metric_data':
        return {
            'metrics': chunks,
            'summary': f"Found {len(chunks)} metric data points"
        }
    elif query_archetype == 'event_log':
        return {
            'timeline': sorted(chunks, key=lambda x: x.get('metadata', {}).get('timestamp', '')),
            'summary': f"Found {len(chunks)} events"
        }
    elif query_archetype == 'entity_definition':
        return {
            'entity': chunks[0] if chunks else None,
            'related': chunks[1:],
            'summary': f"Found primary entity with {len(chunks)-1} related items"
        }
    
    return {'results': chunks}

