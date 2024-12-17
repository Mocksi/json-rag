import json
from collections import defaultdict
from datetime import datetime, timedelta
import re
import statistics

from app.config import MAX_CHUNKS
from app.utils import parse_timestamp
from app.embedding import vector_search_with_filter
from app.parsing import json_to_path_chunks, extract_entities
from app.intent import analyze_query_intent, extract_filters_from_query, get_system_prompt
from app.database import get_files_to_process, upsert_file_metadata, init_db
from app.config import embedding_model
from app.utils import get_json_files

# Include your retrieval logic: get_relevant_chunks, hybrid_retrieval, hierarchical_retrieval, etc.
# We'll place get_relevant_chunks and others here:

def get_relevant_chunks(conn, query, top_k=5):
    """
    Retrieve most relevant chunks for a query using vector similarity search.
    
    Args:
        conn: PostgreSQL database connection
        query (str): User's query string
        top_k (int): Maximum number of chunks to return (default: 5)
        
    Returns:
        list: Retrieved chunks ordered by relevance score
        
    Note:
        Uses cosine similarity between query embedding and stored chunk embeddings.
        Includes debug output for query processing and chunk scores.
    """
    print(f"\nDEBUG: Processing query: '{query}'")
    cur = conn.cursor()
    
    # Generate and log query embedding
    query_embedding = embedding_model.encode([query])[0]
    print(f"DEBUG: Generated embedding of size {len(query_embedding)}")
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
    
    # Enhanced query with scores and IDs
    search_query = f"""
    SELECT id, chunk_text::text, embedding <-> '{embedding_str}' as score
    FROM json_chunks
    ORDER BY score
    LIMIT {top_k};
    """
    
    cur.execute(search_query)
    results = cur.fetchall()
    
    # Log retrieval results
    print("\nDEBUG: Retrieved chunks:")
    retrieved_texts = []
    for chunk_id, chunk_text, score in results:
        print(f"Chunk {chunk_id}: score = {score:.4f}")
        retrieved_texts.append(chunk_text)
    
    cur.close()
    return retrieved_texts

def build_prompt(query, retrieved_texts, query_intent):
    """
    Build a prompt for the LLM using retrieved context and query intent.
    
    Args:
        query (str): User's query
        retrieved_texts (list): List of relevant text chunks
        query_intent (dict): Query intent analysis results
        
    Returns:
        str: Formatted prompt for the LLM
    """
    context_str = "\n\n".join(retrieved_texts)
    prompt = f"""Use the provided context to answer the user's query.

Context type: {query_intent['primary_intent']}
Additional intents: {', '.join(query_intent['all_intents'])}

Guidelines:
- Focus on {query_intent['primary_intent']} aspects
- Use specific details from the context
- Always use human-readable names in addition to IDs when available
- Show relationships between named entities
- If information isn't in the context, say so
- For temporal queries, preserve chronological order
- For metrics, include specific values and trends

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

def metric_retrieval(conn, query, top_k=20):
    cur = conn.cursor()
    if "peak" in query.lower() and "network" in query.lower():
        base_query = """
        WITH numeric_chunks AS (
            SELECT DISTINCT 
                c.id,
                c.chunk_text,
                (c.chunk_text -> 'value' ->> 'value')::float as numeric_value
            FROM json_chunks c
            JOIN chunk_key_values kv ON c.id = kv.chunk_id
            WHERE kv.key LIKE '%%network%%'
                AND c.chunk_text -> 'value' ->> 'type' = 'primitive'
                AND c.chunk_text -> 'value' ->> 'value' ~ '^[0-9]+\.?[0-9]*$'
        ),
        ranked_chunks AS (
            SELECT *,
                percent_rank() OVER (ORDER BY numeric_value DESC) as value_rank
            FROM numeric_chunks
        )
        SELECT chunk_text::text
        FROM ranked_chunks
        WHERE value_rank <= 0.2
        ORDER BY numeric_value DESC
        LIMIT %s
        """
    else:
        base_query = """
        SELECT DISTINCT c.chunk_text::text
        FROM json_chunks c
        WHERE c.chunk_text -> 'value' ->> 'type' = 'primitive'
            AND c.chunk_text -> 'value' ->> 'value' ~ '^[0-9]+\.?[0-9]*$'
        LIMIT %s
        """
    cur.execute(base_query, (top_k,))
    results = cur.fetchall()
    return [r[0] for r in results]

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
    """
    Summarize metric-related chunks with statistical analysis.
    
    Args:
        retrieved_texts (list): List of chunk texts containing metrics
        
    Returns:
        list: Summarized metric information
    """
    if not retrieved_texts:
        return []
        
    print("\nDEBUG: Summarizing metrics from chunks:")
    metrics = defaultdict(list)
    other_chunks = []
    
    for i, text in enumerate(retrieved_texts):
        try:
            data = json.loads(text)
            print(f"\nChunk {i}:")
            print(f"Path: {data.get('path')}")
            print(f"Value: {data.get('value')}")
            
            value_data = data.get('value', {})
            if value_data.get('type') == 'primitive' and isinstance(value_data.get('value'), (int, float)):
                metric_name = data.get('path', '').split('.')[-1]
                metrics[metric_name].append(text)
                print(f"Added to metric group: {metric_name}")
            else:
                other_chunks.append(text)
                print("Added to other chunks (non-metric)")
        except Exception as e:
            print(f"Error processing chunk {i}: {e}")
            other_chunks.append(text)
    
    print(f"\nDEBUG: Found {len(metrics)} metric groups:")
    for metric_name, texts in metrics.items():
        print(f"- {metric_name}: {len(texts)} values")

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
    filter_conditions = []
    filter_values = []
    for key, value in filters.items():
        filter_conditions.append(f"kv.key = %s AND kv.value = %s")
        filter_values.extend([key, str(value)])
    filter_query = f"""
    SELECT DISTINCT c.chunk_text::text
    FROM json_chunks c
    JOIN chunk_key_values kv ON c.id = kv.chunk_id
    WHERE {' OR '.join(filter_conditions)}
    """
    cur = conn.cursor()
    cur.execute(filter_query, filter_values)
    filter_results = [r[0] for r in cur.fetchall()]
    if len(filter_results) < top_k:
        semantic_results = get_relevant_chunks(conn, query, top_k=top_k - len(filter_results))
        combined_results = filter_results + [r for r in semantic_results if r not in filter_results]
        return combined_results[:top_k]
    return filter_results[:top_k]

