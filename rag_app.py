import os
import json
import psycopg2
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
import openai
import argparse
from datetime import datetime, timedelta
import hashlib
import glob
import sys
from dotenv import load_dotenv
from collections import defaultdict
import statistics
import itertools
import re

# Load environment variables from .env file
load_dotenv()

# =========================
# Configuration
# =========================
DATA_DIR = "data/json_docs"
POSTGRES_CONN_STR = "dbname=myragdb user=drew host=localhost port=5432"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")
openai.api_key = OPENAI_API_KEY

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_name)

class FlexibleModel(BaseModel):
    __root__: dict

seen_keys = set()
MAX_CHUNKS = 4

# =========================
# Utility Functions
# =========================
def compute_file_hash(filepath):
    """
    Compute SHA-256 hash of a file.
    
    Args:
        filepath (str): Path to the file to hash
        
    Returns:
        str: Hexadecimal representation of the file's SHA-256 hash
    """
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_json_files():
    """
    Get list of JSON files from the configured data directory.
    
    Returns:
        list: List of paths to JSON files in the data directory
    """
    return glob.glob(os.path.join(DATA_DIR, "*.json"))

def get_file_info(conn):
    """
    Retrieve metadata about previously processed files from database.
    
    Args:
        conn: PostgreSQL database connection
        
    Returns:
        dict: Mapping of filenames to tuples of (hash, last_modified_time)
    """
    cur = conn.cursor()
    cur.execute("SELECT filename, file_hash, last_modified FROM file_metadata;")
    rows = cur.fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}

def upsert_file_metadata(conn, filename, file_hash, mod_time):
    """
    Update or insert file metadata in the database.
    
    Args:
        conn: PostgreSQL database connection
        filename (str): Name of the file
        file_hash (str): SHA-256 hash of the file
        mod_time (datetime): Last modification time
    """
    cur = conn.cursor()
    query = """
    INSERT INTO file_metadata (filename, file_hash, last_modified)
    VALUES (%s, %s, %s)
    ON CONFLICT (filename) DO UPDATE SET
        file_hash = EXCLUDED.file_hash,
        last_modified = EXCLUDED.last_modified;
    """
    cur.execute(query, (filename, file_hash, mod_time))
    conn.commit()

def get_files_to_process(conn):
    """
    Identify files that need processing (new or modified).
    
    Args:
        conn: PostgreSQL database connection
        
    Returns:
        list: Tuples of (filename, hash, modification_time) for files needing processing
    """
    existing_info = get_file_info(conn)
    files = get_json_files()
    to_process = []
    for f in files:
        f_hash = compute_file_hash(f)
        f_mtime = datetime.utcfromtimestamp(os.path.getmtime(f))
        if f not in existing_info:
            to_process.append((f, f_hash, f_mtime))
        else:
            old_hash, old_mtime = existing_info[f]
            if old_hash != f_hash:
                to_process.append((f, f_hash, f_mtime))
    return to_process

def create_enhanced_chunk(path, value, context=None, entities=None):
    """
    Create an enhanced text chunk with path, type, value, and related context.
    
    Args:
        path (str): JSON path to the current value
        value: The value at the current path
        context (dict, optional): Parent context information
        entities (dict, optional): Known entities and their relationships
        
    Returns:
        str: Formatted string containing the chunk with context
    """
    val_type = type(value).__name__
    
    # Handle different value types appropriately
    if isinstance(value, dict):
        val_str = json.dumps({k: str(v) if not isinstance(v, (dict, list)) else "..." 
                            for k, v in value.items()}, indent=2)
    elif isinstance(value, list):
        val_str = f"Array with {len(value)} items"
    else:
        val_str = str(value)
    
    # Build context information
    context_info = []
    if context:
        context_str = {k: str(v) if not isinstance(v, (dict, list)) else "..."
                      for k, v in context.items()}
        context_info.append(f"Context: {json.dumps(context_str, indent=2)}")
    
    # Add related entities if they exist
    related_entities = []
    if entities:
        for entity_id, info in entities.items():
            entity_path = info['path']
            if path.startswith(entity_path) or entity_path.startswith(path):
                related_entities.append(f"Related Entity: {info['type']}={entity_id}")
                # Add entity context if available
                if info.get('context'):
                    entity_context = {k: str(v) for k, v in info['context'].items() 
                                   if k != info['type']}
                    related_entities.append(f"Entity Context: {json.dumps(entity_context, indent=2)}")
    
    # Break path into hierarchical contexts
    path_contexts = []
    path_parts = path.split('.')
    current_context = "$"
    for part in path_parts[1:]:  # Skip the root $
        current_context = f"{current_context}.{part}"
        if context and current_context in context:
            path_contexts.append(f"Parent Path: {current_context}")
            path_contexts.append(f"Parent Context: {json.dumps(context[current_context], indent=2)}")
    
    # Combine all components
    chunk_parts = [
        f"Path: {path}",
        f"Type: {val_type}",
        f"Value: {val_str}"
    ]
    
    if path_contexts:
        chunk_parts.extend(path_contexts)
    if context_info:
        chunk_parts.extend(context_info)
    if related_entities:
        chunk_parts.extend(related_entities)
    
    return "\n".join(chunk_parts)

def json_to_path_chunks(json_obj, current_path="$", entities=None, parent_context=None, path_contexts=None):
    """
    Convert JSON object to path-aware chunks with context preservation.
    
    Args:
        json_obj: JSON object to process
        current_path (str): Current path in JSON structure
        entities (dict, optional): Known entities and relationships
        parent_context (dict, optional): Context from parent nodes
        path_contexts (dict, optional): Accumulated path contexts
        
    Returns:
        list: List of text chunks with preserved context
    """
    chunks = []
    if path_contexts is None:
        path_contexts = {}
    
    if isinstance(json_obj, dict):
        # Store context for this path
        path_contexts[current_path] = {k: str(v) if not isinstance(v, (dict, list)) else "..."
                                     for k, v in json_obj.items()}
        
        # Create chunk for the entire object with its context
        chunks.append(create_enhanced_chunk(
            current_path, 
            json_obj,
            context=path_contexts,
            entities=entities
        ))
        
        # Process individual fields
        for key, value in json_obj.items():
            seen_keys.add(key)
            new_path = f"{current_path}.{key}"
            chunks.extend(json_to_path_chunks(
                value,
                current_path=new_path,
                entities=entities,
                parent_context=json_obj,
                path_contexts=path_contexts
            ))
            
    elif isinstance(json_obj, list):
        seen_keys.add(current_path + "[]")
        for i, item in enumerate(json_obj):
            new_path = f"{current_path}[{i}]"
            chunks.extend(json_to_path_chunks(
                item,
                current_path=new_path,
                entities=entities,
                parent_context=parent_context,
                path_contexts=path_contexts
            ))
    else:
        chunks.append(create_enhanced_chunk(
            current_path,
            json_obj,
            context=path_contexts,
            entities=entities
        ))
    
    return chunks

def track_entity_relationships(json_obj, current_path="$", parent_context=None):
    """
    Track relationships between entities with enhanced context preservation.
    
    Args:
        json_obj: JSON object to analyze
        current_path (str): Current path in JSON structure
        parent_context (dict, optional): Context from parent nodes
        
    Returns:
        list: List of relationship dictionaries containing entity information
    """
    relationships = []
    
    def process_node(node, path, context):
        if isinstance(node, dict):
            # Track entities with their types and relationships
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
                # Look for relationships in the path
                path_parts = path.split('.')
                for i, part in enumerate(path_parts):
                    if part in ['member', 'owner', 'participant', 'author', 'assignee']:
                        found_entity['role'] = part
                    elif part in ['project', 'team', 'organization', 'department']:
                        found_entity['group_type'] = part
                        if i > 0:
                            found_entity['group_context'] = path_parts[i-1]
                
                # Add role context if available
                if 'actor' in node:
                    actor = node['actor']
                    if isinstance(actor, dict):
                        if 'role' in actor:
                            found_entity['actor_role'] = actor['role']
                        if 'permissions' in actor:
                            found_entity['permissions'] = actor['permissions']
                
                # Add organizational context
                if 'organization' in context:
                    found_entity['org_context'] = context['organization']
                
                # Add project context
                if 'project' in context:
                    found_entity['project_context'] = context['project']
                
                relationships.append(found_entity)
            
            # Recursively process all dictionary values
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
    Extract and track entities with their relationships and context.
    
    Args:
        json_obj: JSON object to process
        current_path (str): Current path in JSON structure
        
    Returns:
        dict: Dictionary of entities with context and relationships
    """
    entities = {}
    relationships = track_entity_relationships(json_obj, current_path)
    
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

def index_chunk_keys(conn, chunk_id, chunk_text):
    """
    Index key-value pairs from chunk text for hybrid retrieval.
    
    Args:
        conn: PostgreSQL database connection
        chunk_id (str): Unique identifier for the chunk
        chunk_text (str): Text content of the chunk
    """
    # Extract key-value pairs from chunk_text if present
    extracted_pairs = extract_key_value_pairs(chunk_text)
    cur = conn.cursor()
    for k, v in extracted_pairs.items():
        cur.execute("""
            INSERT INTO chunk_keys_index (key_name, key_value, chunk_id)
            VALUES (%s, %s, %s)
        """, (k, v, chunk_id))

def hybrid_retrieval(conn, query, top_k=5):
    """
    Perform hybrid retrieval with enhanced numeric comparisons.
    
    Args:
        conn: PostgreSQL database connection
        query (str): User's query
        top_k (int): Number of results to return
        
    Returns:
        list: Retrieved chunks filtered by keywords and numeric conditions
    """
    filters = extract_filters_from_query(query)
    
    filtered_chunk_ids = None
    if filters:
        cur = conn.cursor()
        conditions = []
        params = []
        
        for k, v in filters.items():
            # Check for numeric comparisons
            if '_gt_' in k or '_lt_' in k:
                try:
                    value = float(v)
                    conditions.append(f"(key_name=%s AND key_value::float {'>=' if '_gt_' in k else '<='} %s)")
                    params.extend([k.split('_')[0], value])
                except ValueError:
                    conditions.append("(key_name=%s AND key_value=%s)")
                    params.extend([k, v])
            else:
                conditions.append("(key_name=%s AND key_value=%s)")
                params.extend([k, v])
                
        sql = f"SELECT chunk_id FROM chunk_keys_index WHERE {' AND '.join(conditions)}"
        cur.execute(sql, tuple(params))
        filtered_chunk_ids = [r[0] for r in cur.fetchall()]

    return vector_search_with_filter(conn, query, filtered_chunk_ids, top_k)

def vector_search_with_filter(conn, query, allowed_chunk_ids, top_k):
    """
    Perform vector similarity search with optional ID filtering.
    
    Args:
        conn: PostgreSQL database connection
        query (str): Search query
        allowed_chunk_ids (list): List of chunk IDs to consider, or None for all
        top_k (int): Number of results to return
        
    Returns:
        list: Retrieved chunks ordered by vector similarity
    """
    query_embedding = embedding_model.encode([query])[0]
    embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    cur = conn.cursor()
    if allowed_chunk_ids and len(allowed_chunk_ids) > 0:
        format_ids = ",".join(["%s"] * len(allowed_chunk_ids))
        sql = f"""
        SELECT chunk_text
        FROM json_chunks
        WHERE id IN ({format_ids})
        ORDER BY embedding <-> '{embedding_str}'
        LIMIT {top_k};
        """
        cur.execute(sql, tuple(allowed_chunk_ids))
    else:
        sql = f"""
        SELECT chunk_text
        FROM json_chunks
        ORDER BY embedding <-> '{embedding_str}'
        LIMIT {top_k};
        """
        cur.execute(sql)

    results = cur.fetchall()
    return [r[0] for r in results]

def load_and_embed_new_data(conn):
    """
    Load new or modified files, process them, and store embeddings.
    
    Args:
        conn: PostgreSQL database connection
        
    Processes files, extracts entities, generates chunks, and stores embeddings.
    Updates file metadata and schema evolution tracking.
    """
    to_process = get_files_to_process(conn)
    if not to_process:
        print("No new or changed files to process.")
        return

    print(f"Processing {len(to_process)} new or modified files...")
    
    documents = []
    entities_by_file = {}
    
    for f, f_hash, f_mtime in to_process:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                validated = FlexibleModel.parse_obj(data)
                documents.append((f, f_hash, f_mtime, validated.__root__))
                entities = extract_entities(validated.__root__)
                entities_by_file[f] = entities
                
                # Print detailed entity and relationship information
                if entities:
                    print(f"\nEntities and relationships in {f}:")
                    for entity_id, info in entities.items():
                        print(f"\nEntity: {entity_id}")
                        print(f"Type: {info['type']}")
                        print(f"Path: {info['path']}")
                        if info.get('role'):
                            print(f"Role: {info['role']}")
                        if info.get('group_type'):
                            print(f"Group: {info['group_type']}")
                            if info.get('group_context'):
                                print(f"Group Context: {info['group_context']}")
                        if info.get('context'):
                            print("Context:")
                            for k, v in info['context'].items():
                                print(f"  {k}: {v}")
                
                print(f"Validated document: {f}")
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Skipping invalid JSON document {f}:", e)

    print(f"Successfully validated {len(documents)} documents")
    
    # Print entity summary for each file
    for f, entities in entities_by_file.items():
        if entities:
            print(f"\nEntities found in {f}:")
            for entity_id, info in entities.items():
                print(f"  - {info['type']}={entity_id} at {info['path']}")
                if info.get('context'):
                    context_summary = {k: str(v) for k, v in info['context'].items() 
                                    if k != info['type']}
                    print(f"    Context: {json.dumps(context_summary, indent=2)}")

    all_chunks = []
    seen_keys.clear()
    for (f, f_hash, f_mtime, doc) in documents:
        doc_chunks = json_to_path_chunks(
            doc,
            entities=entities_by_file.get(f),
            parent_context=None
        )
        all_chunks.extend((f, chunk) for chunk in doc_chunks)

    if not all_chunks:
        print("No chunks to embed from new files.")
        for f, f_hash, f_mtime, d in documents:
            upsert_file_metadata(conn, f, f_hash, f_mtime)
        return

    print(f"Generated {len(all_chunks)} chunks")
    print("Generating embeddings...")
    
    chunk_texts = [c[1] for c in all_chunks]
    chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)
    print("Embeddings generated.")

    cur = conn.cursor()
    for i, (f, text) in enumerate(all_chunks):
        embedding_list = chunk_embeddings[i].tolist()
        embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
        chunk_id = f"{f}:{i}"
        cur.execute("""
            INSERT INTO json_chunks (id, chunk_text, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (id) DO NOTHING;
        """, (chunk_id, text, embedding_str))
        # Index keys from this chunk
        index_chunk_keys(conn, chunk_id, text)

    conn.commit()
    print(f"Embedded {len(all_chunks)} new chunks.")

    for f, f_hash, f_mtime, d in documents:
        upsert_file_metadata(conn, f, f_hash, f_mtime)

def get_relevant_chunks(conn, query, top_k=5):
    """
    Retrieve most relevant chunks for a query using vector similarity.
    
    Args:
        conn: PostgreSQL database connection
        query (str): User's query
        top_k (int): Number of chunks to retrieve
        
    Returns:
        list: Most relevant text chunks for the query
    """
    cur = conn.cursor()
    query_embedding = embedding_model.encode([query])[0]
    query_embedding_list = query_embedding.tolist()
    embedding_str = "[" + ",".join(str(x) for x in query_embedding_list) + "]"

    search_query = f"""
    SELECT chunk_text
    FROM json_chunks
    ORDER BY embedding <-> '{embedding_str}'
    LIMIT {top_k};
    """
    cur.execute(search_query)
    results = cur.fetchall()
    retrieved_texts = [r[0] for r in results]
    cur.close()
    return retrieved_texts

def build_prompt(user_query, retrieved_chunks):
    """
    Build a prompt for the language model using retrieved context.
    
    Args:
        user_query (str): User's question
        retrieved_chunks (list): Relevant context chunks
        
    Returns:
        str: Formatted prompt with context and query
    """
    context_str = "\n\n".join(retrieved_chunks)
    prompt = f"""You are a helpful assistant. Use the provided context to answer the user's query.

Guidelines:
- Use specific names and identifiers from the context
- Reference exact paths when relevant
- If multiple similar items exist, distinguish them clearly
- If the answer isn't in the context, say so
- Maintain the relationships between entities as shown in the context

Context:
{context_str}

Question: {user_query}

Only use the provided context to answer."""
    return prompt

def summarize_chunks(chunks):
    """
    Summarize a list of chunks into a shorter context while preserving temporal information.
    
    Args:
        chunks (list): List of text chunks to summarize
        
    Returns:
        str: Condensed summary of the chunks with preserved timeline
    """
    prompt = """Summarize these chunks of context, preserving:
1. Chronological order of events
2. Specific dates and times
3. Key temporal relationships

Context:
""" + "\n\n".join(chunks)

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

def extract_timestamp(chunk_text):
    """
    Extract timestamp information from chunk text.
    
    Args:
        chunk_text (str): Text content of the chunk
        
    Returns:
        datetime or None: Extracted timestamp if found, None otherwise
    """
    try:
        # Look for common date patterns
        lines = chunk_text.split('\n')
        for line in lines:
            if any(key in line.lower() for key in ['date', 'time', 'created', 'modified', 'timestamp']):
                # Try different date formats
                for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%Y-%m-%d %H:%M:%S']:
                    try:
                        date_str = line.split(':', 1)[1].strip()
                        return datetime.strptime(date_str, fmt)
                    except (ValueError, IndexError):
                        continue
    except Exception:
        pass
    return None

def hierarchical_retrieval(conn, query, top_k=20):
    """
    Perform hierarchical retrieval with automatic summarization and timeline preservation.
    
    Args:
        conn: PostgreSQL database connection
        query (str): User's query
        top_k (int): Initial number of chunks to retrieve
        
    Returns:
        list: Either original chunks or summarized versions if too many
    """
    # Start by retrieving top_k chunks
    initial_chunks = get_relevant_chunks(conn, query, top_k=top_k)
    
    # Handle timeline queries
    if "timeline" in query.lower() or any(word in query.lower() for word in ['when', 'date', 'time', 'chronological']):
        # Sort chunks by timestamp if available
        initial_chunks.sort(key=lambda x: extract_timestamp(x) if extract_timestamp(x) else datetime.max)
        print("Retrieved chunks in chronological order")
    
    # If initial_chunks > MAX_CHUNKS, summarize in batches while preserving order
    if len(initial_chunks) > MAX_CHUNKS:
        batch_size = MAX_CHUNKS
        summaries = []
        
        # Group chunks while maintaining chronological order
        for i in range(0, len(initial_chunks), batch_size):
            batch = initial_chunks[i:i+batch_size]
            # Include temporal context in summary prompt
            time_range = [extract_timestamp(c) for c in batch if extract_timestamp(c)]
            if time_range:
                context = f"Events from {min(time_range)} to {max(time_range)}:\n\n"
            else:
                context = ""
            
            summary = summarize_chunks([context + c for c in batch])
            summaries.append(summary)
            
        # Now summarize the summaries if needed
        while len(summaries) > MAX_CHUNKS:
            new_summaries = []
            for i in range(0, len(summaries), batch_size):
                batch = summaries[i:i+batch_size]
                summary = summarize_chunks(batch)
                new_summaries.append(summary)
            summaries = new_summaries
            
        return summaries
    else:
        return initial_chunks

def answer_query_with_hierarchy(conn, query):
    """
    Answer query using hierarchical retrieval and summarization.
    
    Args:
        conn: PostgreSQL database connection
        query (str): User's question
        
    Returns:
        str: Generated answer based on retrieved and potentially summarized context
    """
    # Use hierarchical retrieval if needed
    chunks_or_summaries = hierarchical_retrieval(conn, query)
    
    # If we end up with multiple summaries, just join them
    if len(chunks_or_summaries) > MAX_CHUNKS:
        # Summarize again
        final_summary = summarize_chunks(chunks_or_summaries)
        chunks = [final_summary]
    else:
        chunks = chunks_or_summaries

    prompt = build_prompt(query, chunks)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        max_tokens=300
    )
    return completion.choices[0].message.content.strip()

def chat_loop(conn):
    """
    Run interactive chat loop for user queries.
    
    Args:
        conn: PostgreSQL database connection
    """
    print("Enter your queries below. Type ':quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == ":quit":
            print("Exiting chat.")
            break
        
        load_and_embed_new_data(conn)

        try:
            answer = answer_query_with_hierarchy(conn, user_input)
            print("Assistant:", answer)
        except Exception as e:
            print("Error processing query:", str(e))

def reset_database(conn):
    """
    Reset the database by truncating all tables.
    
    Args:
        conn: PostgreSQL database connection
        
    Requires user confirmation before proceeding.
    """
    print("Warning: This will delete all stored embeddings and metadata.")
    confirmation = input("Are you sure you want to reset the database? (yes/no): ")
    if confirmation.lower() != 'yes':
        print("Database reset cancelled.")
        return
    
    print("Resetting database...")
    cur = conn.cursor()
    cur.execute("""
        BEGIN;
          TRUNCATE TABLE json_chunks CASCADE;
          TRUNCATE TABLE file_metadata CASCADE;
          TRUNCATE TABLE schema_evolution CASCADE;
        COMMIT;
    """)
    print("Database reset complete.")

def extract_filters_from_query(query):
    """
    Extract filter conditions from a user query.
    
    Args:
        query (str): User's query string
        
    Returns:
        dict: Dictionary of extracted filters (key-value pairs)
        
    Example:
        "Show projects in department=Engineering" -> {"department": "Engineering"}
    """
    filters = {}
    tokens = query.split()
    for token in tokens:
        if "=" in token:
            k, v = token.split("=", 1)
            filters[k.strip()] = v.strip()
    return filters

def extract_key_value_pairs(chunk_text):
    """
    Extract key-value pairs from chunk text with enhanced numeric handling.
    
    Args:
        chunk_text (str): Text content of the chunk
        
    Returns:
        dict: Dictionary of extracted key-value pairs including numeric comparisons
        
    Example:
        Input: "value: 42" -> {
            "value": "42",
            "value_gt_41": True,
            "value_lt_43": True,
            "value_exact": 42
        }
    """
    pairs = {}
    lines = chunk_text.split('\n')
    
    # Extract from structured fields
    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            key = key.strip()
            value = value.strip()
            pairs[key] = value
            
            # Try to parse numeric values
            try:
                num_value = float(value)
                # Add numeric comparisons
                pairs[f"{key}_gt_{num_value-1}"] = True
                pairs[f"{key}_lt_{num_value+1}"] = True
                pairs[f"{key}_exact"] = num_value
            except ValueError:
                pass
            
    # Look for JSON-like structures
    try:
        for line in lines:
            if line.startswith('Value: {'):
                json_str = line[7:]  # Remove "Value: "
                data = json.loads(json_str)
                if isinstance(data, dict):
                    for k, v in data.items():
                        pairs[k] = v
                        # Try to parse numeric values in JSON
                        if isinstance(v, (int, float)):
                            pairs[f"{k}_gt_{v-1}"] = True
                            pairs[f"{k}_lt_{v+1}"] = True
                            pairs[f"{k}_exact"] = v
    except json.JSONDecodeError:
        pass
        
    return pairs

def summarize_related_events(events, entity_id):
    """
    Group and summarize events by entity and relationship.
    
    Args:
        events (list): List of events to process
        entity_id (str): Entity ID to focus on
        
    Returns:
        dict: Events grouped by context with summaries
    """
    related_events = defaultdict(list)
    for event in events:
        if entity_id in event.get('actors', []):
            context = event.get('context', 'general')
            related_events[context].append(event)
    return related_events

def aggregate_entity_context(conn, entity_id):
    """
    Aggregate all context related to a specific entity.
    
    Args:
        conn: PostgreSQL database connection
        entity_id (str): Entity ID to gather context for
        
    Returns:
        dict: Aggregated context by category
    """
    context = defaultdict(list)
    cur = conn.cursor()
    
    # Get all chunks mentioning this entity
    cur.execute("""
        SELECT chunk_text
        FROM json_chunks
        WHERE chunk_text LIKE %s
    """, (f"%{entity_id}%",))
    
    chunks = cur.fetchall()
    
    for chunk in chunks:
        chunk_text = chunk[0]
        # Categorize context
        if "role" in chunk_text.lower():
            context['roles'].append(extract_role_info(chunk_text))
        if "project" in chunk_text.lower():
            context['projects'].append(extract_project_info(chunk_text))
        if "organization" in chunk_text.lower():
            context['organizations'].append(extract_org_info(chunk_text))
        if any(word in chunk_text.lower() for word in ['created', 'modified', 'date', 'time']):
            context['timeline'].append(extract_temporal_info(chunk_text))
    
    return context

def extract_role_info(chunk_text):
    """
    Extract role-related information from chunk text.
    
    Args:
        chunk_text (str): Text to analyze
        
    Returns:
        dict: Role information
    """
    role_info = {}
    lines = chunk_text.split('\n')
    for line in lines:
        if 'role' in line.lower():
            role_info['role'] = line.split(':', 1)[1].strip() if ':' in line else line
        if 'permissions' in line.lower():
            role_info['permissions'] = line.split(':', 1)[1].strip() if ':' in line else line
    return role_info

def extract_project_info(chunk_text):
    """
    Extract project-related information from chunk text.
    
    Args:
        chunk_text (str): Text to analyze
        
    Returns:
        dict: Project information
    """
    project_info = {}
    lines = chunk_text.split('\n')
    for line in lines:
        if 'project' in line.lower():
            project_info['name'] = line.split(':', 1)[1].strip() if ':' in line else line
        if 'status' in line.lower():
            project_info['status'] = line.split(':', 1)[1].strip() if ':' in line else line
    return project_info

def extract_org_info(chunk_text):
    """
    Extract organization-related information from chunk text.
    
    Args:
        chunk_text (str): Text to analyze
        
    Returns:
        dict: Organization information
    """
    org_info = {}
    lines = chunk_text.split('\n')
    for line in lines:
        if 'organization' in line.lower():
            org_info['name'] = line.split(':', 1)[1].strip() if ':' in line else line
        if 'department' in line.lower():
            org_info['department'] = line.split(':', 1)[1].strip() if ':' in line else line
    return org_info

def extract_temporal_info(chunk_text):
    """
    Extract temporal information from chunk text.
    
    Args:
        chunk_text (str): Text to analyze
        
    Returns:
        dict: Temporal information
    """
    temporal_info = {}
    timestamp = extract_timestamp(chunk_text)
    if timestamp:
        temporal_info['timestamp'] = timestamp
        temporal_info['type'] = 'event'
        # Extract event type if available
        lines = chunk_text.split('\n')
        for line in lines:
            if any(action in line.lower() for action in ['created', 'modified', 'updated', 'deleted']):
                temporal_info['action'] = line.split(':', 1)[1].strip() if ':' in line else line
    return temporal_info

def get_entity_summary(conn, entity_id):
    """
    Generate a comprehensive summary for an entity.
    
    Args:
        conn: PostgreSQL database connection
        entity_id (str): Entity ID to summarize
        
    Returns:
        str: Formatted summary of entity context
    """
    context = aggregate_entity_context(conn, entity_id)
    
    # Build summary sections
    summary_parts = []
    
    if context['roles']:
        roles_summary = "Roles:\n" + "\n".join(f"- {r['role']}" for r in context['roles'])
        summary_parts.append(roles_summary)
    
    if context['projects']:
        projects_summary = "Projects:\n" + "\n".join(f"- {p['name']}" for p in context['projects'])
        summary_parts.append(projects_summary)
    
    if context['organizations']:
        orgs_summary = "Organizations:\n" + "\n".join(f"- {o['name']}" for o in context['organizations'])
        summary_parts.append(orgs_summary)
    
    if context['timeline']:
        timeline = sorted(context['timeline'], key=lambda x: x.get('timestamp', datetime.max))
        timeline_summary = "Timeline:\n" + "\n".join(
            f"- {t['timestamp']}: {t.get('action', 'event')}" for t in timeline
        )
        summary_parts.append(timeline_summary)
    
    return "\n\n".join(summary_parts)

def track_numeric_values_over_time(json_obj, value_identifiers=['value', 'metric', 'amount']):
    """
    Track any numeric values through time in a JSON structure.
    
    Args:
        json_obj: The JSON object to process
        value_identifiers (list): List of field names that might contain numeric values
        
    Returns:
        list: Time series data sorted by timestamp
    """
    time_series = []
    
    def process_node(node, timestamp=None, path=None):
        if isinstance(node, dict):
            # Look for timestamp first
            if 'timestamp' in node or 'time' in node or 'date' in node:
                timestamp = node.get('timestamp') or node.get('time') or node.get('date')
            
            # Look for numeric values
            for key, value in node.items():
                if (key in value_identifiers and 
                    isinstance(value, (int, float))):
                    time_series.append({
                        'timestamp': timestamp,
                        'path': path,
                        'field': key,
                        'value': value
                    })
                    
            # Recurse through all children
            for key, value in node.items():
                new_path = f"{path}.{key}" if path else key
                process_node(value, timestamp, new_path)
                
        elif isinstance(node, list):
            for i, item in enumerate(node):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                process_node(item, timestamp, new_path)
    
    process_node(json_obj)
    return sorted(time_series, key=lambda x: x['timestamp'] if x['timestamp'] else '')

def analyze_time_series(time_series, group_by='field'):
    """
    Analyze time series data with flexible grouping.
    
    Args:
        time_series (list): List of time series data points
        group_by (str): Field to group by ('field', 'path', or None)
        
    Returns:
        dict: Analysis results grouped by the specified field
    """
    grouped_data = defaultdict(list)
    
    # Group data points
    for point in time_series:
        key = point.get(group_by) if group_by else 'all'
        grouped_data[key].append(point)
    
    results = {}
    for key, points in grouped_data.items():
        values = [p['value'] for p in points]
        results[key] = {
            'count': len(values),
            'mean': statistics.mean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'std_dev': statistics.stdev(values) if len(values) > 1 else None,
            'min': min(values) if values else None,
            'max': max(values) if values else None,
            'first_timestamp': min(p['timestamp'] for p in points if p['timestamp']),
            'last_timestamp': max(p['timestamp'] for p in points if p['timestamp'])
        }
    
    return results

def detect_patterns(time_series, window_size=5):
    """
    Detect patterns and trends in time series data.
    
    Args:
        time_series (list): List of time series data points
        window_size (int): Size of rolling window for pattern detection
        
    Returns:
        dict: Detected patterns and trends
    """
    patterns = defaultdict(list)
    
    # Group by field
    for field, points in itertools.groupby(time_series, key=lambda x: x['field']):
        points = list(points)
        values = [p['value'] for p in points]
        
        # Skip if not enough data points
        if len(values) < window_size:
            continue
            
        # Calculate rolling statistics
        for i in range(len(values) - window_size + 1):
            window = values[i:i+window_size]
            pattern = {
                'field': field,
                'start_idx': i,
                'trend': 'increasing' if window[-1] > window[0] else 'decreasing',
                'volatility': statistics.stdev(window),
                'avg_value': statistics.mean(window)
            }
            patterns[field].append(pattern)
    
    return dict(patterns)

def generate_time_series_report(json_obj, value_identifiers=None):
    """
    Generate a comprehensive time series analysis report.
    
    Args:
        json_obj: The JSON object to process
        value_identifiers (list, optional): List of field names to track
        
    Returns:
        dict: Comprehensive analysis report
    """
    if value_identifiers is None:
        value_identifiers = ['value', 'metric', 'amount', 'count', 'score']
        
    # Track numeric values
    time_series = track_numeric_values_over_time(json_obj, value_identifiers)
    
    # Generate report
    report = {
        'overview': {
            'total_points': len(time_series),
            'unique_fields': len(set(p['field'] for p in time_series)),
            'date_range': {
                'start': min(p['timestamp'] for p in time_series if p['timestamp']),
                'end': max(p['timestamp'] for p in time_series if p['timestamp'])
            }
        },
        'by_field': analyze_time_series(time_series, 'field'),
        'by_path': analyze_time_series(time_series, 'path'),
        'patterns': detect_patterns(time_series),
        'generated_at': datetime.now()
    }
    
    return report

def group_related_changes(time_series_data, time_window_seconds=300):
    """
    Group numeric changes that occur within the same time window.
    
    Args:
        time_series_data (list): List of time series entries
        time_window_seconds (int): Size of time window for grouping
        
    Returns:
        dict: Changes grouped by time window with context
    """
    grouped_changes = defaultdict(list)
    
    # Sort data by timestamp first
    sorted_data = sorted(time_series_data, 
                        key=lambda x: parse_timestamp(x['timestamp']) if x['timestamp'] else datetime.max)
    
    for entry in sorted_data:
        timestamp = parse_timestamp(entry['timestamp']) if entry['timestamp'] else None
        if timestamp:
            # Create window key
            window_key = timestamp.replace(second=0, microsecond=0)
            
            # Add context to the entry
            entry_with_context = {
                **entry,
                'window_start': window_key,
                'window_end': window_key + timedelta(seconds=time_window_seconds),
                'related_paths': [e['path'] for e in grouped_changes[window_key] 
                                if e['path'].startswith(entry['path']) or 
                                   entry['path'].startswith(e['path'])]
            }
            
            grouped_changes[window_key].append(entry_with_context)
    
    return grouped_changes

def detect_related_changes(time_series_data):
    """
    Find numeric values that change together.
    
    Args:
        time_series_data (list): List of time series entries
        
    Returns:
        list: Pairs of related changes with correlation info
    """
    related_changes = []
    
    # Sort by timestamp
    sorted_data = sorted(time_series_data, 
                        key=lambda x: parse_timestamp(x['timestamp']) if x['timestamp'] else datetime.max)
    
    # Group by path prefix
    path_groups = defaultdict(list)
    for entry in sorted_data:
        path_parts = entry['path'].split('.')
        for i in range(len(path_parts)):
            prefix = '.'.join(path_parts[:i+1])
            path_groups[prefix].append(entry)
    
    # Analyze related changes
    for path_prefix, entries in path_groups.items():
        if len(entries) < 2:
            continue
            
        # Look for temporal relationships
        for i in range(len(entries) - 1):
            current = entries[i]
            next_entry = entries[i + 1]
            
            # Check if changes are related
            if current['path'].startswith(next_entry['path']) or \
               next_entry['path'].startswith(current['path']):
                
                # Calculate time difference
                current_time = parse_timestamp(current['timestamp'])
                next_time = parse_timestamp(next_entry['timestamp'])
                if current_time and next_time:
                    time_diff = (next_time - current_time).total_seconds()
                    
                    # Calculate value relationship
                    correlation = {
                        'time_difference': time_diff,
                        'value_ratio': next_entry['value'] / current['value'] if current['value'] != 0 else None,
                        'absolute_change': next_entry['value'] - current['value'],
                        'percentage_change': ((next_entry['value'] - current['value']) / current['value'] * 100) 
                                          if current['value'] != 0 else None
                    }
                    
                    related_changes.append({
                        'first': current,
                        'second': next_entry,
                        'relationship': correlation,
                        'path_prefix': path_prefix
                    })
    
    return related_changes

def analyze_change_patterns(grouped_changes, related_changes):
    """
    Analyze patterns in grouped and related changes.
    
    Args:
        grouped_changes (dict): Changes grouped by time window
        related_changes (list): List of related change pairs
        
    Returns:
        dict: Analysis of change patterns
    """
    patterns = {
        'time_windows': {},
        'relationships': {},
        'correlations': defaultdict(list)
    }
    
    # Analyze time windows
    for window, changes in grouped_changes.items():
        patterns['time_windows'][window] = {
            'count': len(changes),
            'paths_affected': len(set(c['path'] for c in changes)),
            'total_magnitude': sum(abs(c['value']) for c in changes),
            'changes': sorted(changes, key=lambda x: abs(x['value']), reverse=True)
        }
    
    # Analyze relationships
    for rel in related_changes:
        path_pair = (rel['first']['path'], rel['second']['path'])
        patterns['correlations'][path_pair].append(rel['relationship'])
    
    # Summarize correlations
    for path_pair, correlations in patterns['correlations'].items():
        patterns['relationships'][path_pair] = {
            'count': len(correlations),
            'avg_time_diff': statistics.mean(c['time_difference'] for c in correlations),
            'avg_value_ratio': statistics.mean(c['value_ratio'] for c in correlations if c['value_ratio']),
            'correlation_strength': len(correlations) / len(grouped_changes)
        }
    
    return patterns

def parse_timestamp(timestamp_str):
    """
    Parse timestamp string into datetime object.
    
    Args:
        timestamp_str (str): Timestamp string to parse
        
    Returns:
        datetime: Parsed timestamp or None if invalid
    """
    try:
        # Try common formats
        formats = [
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%Y-%m-%d'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
                
        # Try parsing with dateutil as fallback
        from dateutil import parser
        return parser.parse(timestamp_str)
    except:
        return None

def analyze_json_patterns(json_obj):
    """
    Analyze JSON structure to determine its archetype and characteristics.
    
    Args:
        json_obj: JSON object to analyze
        
    Returns:
        dict: Analysis results including archetype and strategies
    """
    def is_api_response(obj):
        """Detect common API response patterns"""
        api_indicators = {
            # Response wrapper patterns
            'data', 'meta', 'errors', 'status', 'code',
            # Pagination patterns
            'pagination', 'page', 'per_page', 'total', 'next', 'previous',
            # Resource patterns
            'items', 'results', 'records', 'list',
            # API metadata
            'version', 'generated_at', 'api_version'
        }
        
        if isinstance(obj, dict):
            keys = {k.lower() for k in obj.keys()}
            # Check for common API patterns
            if (('data' in keys and any(k in keys for k in ['meta', 'pagination'])) or
                ('items' in keys and 'pagination' in keys) or
                ('status' in keys and 'code' in keys)):
                return True
            
            # Look for nested API patterns
            return any(is_api_response(v) for v in obj.values() if isinstance(v, dict))

    def has_timestamps(obj):
        """Check for timestamp patterns in values"""
        timestamp_patterns = [
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}",  # ISO format
            r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}",  # Common datetime
            r"timestamp|date|time|created_at|updated_at"  # Common field names
        ]
        
        if isinstance(obj, dict):
            # Check field names and string values
            for k, v in obj.items():
                if any(pattern in k.lower() for pattern in timestamp_patterns):
                    return True
                if isinstance(v, str) and any(re.search(pattern, v) for pattern in timestamp_patterns):
                    return True
                if has_timestamps(v):
                    return True
        elif isinstance(obj, list):
            return any(has_timestamps(item) for item in obj)
        return False

    def has_state_transitions(obj):
        """Detect state/status fields and transitions"""
        state_indicators = {
            'status', 'state', 'phase', 'stage', 'step',
            'progress', 'condition', 'resolution'
        }
        
        if isinstance(obj, dict):
            keys = {k.lower() for k in obj.keys()}
            if keys & state_indicators:
                return True
            return any(has_state_transitions(v) for v in obj.values())
        elif isinstance(obj, list):
            return any(has_state_transitions(item) for item in obj)
        return False

    def has_metric_series(obj):
        """Detect numeric series or metrics"""
        metric_indicators = {
            'value', 'amount', 'count', 'metric',
            'measurement', 'score', 'quantity'
        }
        
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k.lower() in metric_indicators and isinstance(v, (int, float)):
                    return True
                if has_metric_series(v):
                    return True
        elif isinstance(obj, list):
            numeric_count = sum(1 for item in obj if isinstance(item, (int, float)))
            return numeric_count > 1 or any(has_metric_series(item) for item in obj)
        return False

    def analyze_entity_relationships(obj):
        """Detect entity references and relationships"""
        entity_indicators = {
            'id', 'name', 'type', 'reference',
            'parent', 'child', 'owner', 'user'
        }
        relationships = []
        
        def process_node(node, path="$"):
            if isinstance(node, dict):
                entity_fields = {k: v for k, v in node.items() if k.lower() in entity_indicators}
                if entity_fields:
                    relationships.append({
                        'path': path,
                        'type': 'entity',
                        'fields': entity_fields
                    })
                
                for k, v in node.items():
                    process_node(v, f"{path}.{k}")
            elif isinstance(node, list):
                for i, item in enumerate(node):
                    process_node(item, f"{path}[{i}]")
        
        process_node(obj)
        return relationships

    def calculate_max_depth(obj):
        """Calculate maximum nesting depth"""
        if isinstance(obj, dict):
            if not obj:
                return 1
            return 1 + max(calculate_max_depth(v) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return 1
            return 1 + max(calculate_max_depth(item) for item in obj)
        return 0

    # Analyze the JSON structure
    characteristics = {
        'api_response': is_api_response(json_obj),
        'temporal': has_timestamps(json_obj),
        'state_based': has_state_transitions(json_obj),
        'metric_based': has_metric_series(json_obj),
        'entity_relationships': analyze_entity_relationships(json_obj),
        'nesting_depth': calculate_max_depth(json_obj)
    }

    # Determine primary archetype
    archetypes = {
        'api_response': characteristics['api_response'],
        'event_log': characteristics['temporal'] and characteristics['state_based'],
        'metric_series': characteristics['temporal'] and characteristics['metric_based'],
        'entity_graph': len(characteristics['entity_relationships']) > 3,
        'state_machine': characteristics['state_based'] and not characteristics['temporal'],
        'configuration': characteristics['nesting_depth'] > 3 and not any([
            characteristics['temporal'],
            characteristics['metric_based'],
            characteristics['state_based']
        ])
    }

    # API responses can contain other archetypes - check for secondary patterns
    if archetypes['api_response']:
        secondary_patterns = {k: v for k, v in archetypes.items() 
                            if k != 'api_response' and v}
        if secondary_patterns:
            archetype = f"api_response_with_{max(secondary_patterns.items(), key=lambda x: x[1])[0]}"
        else:
            archetype = 'api_response'
    else:
        archetype = max(archetypes.items(), key=lambda x: x[1])[0]

    return {
        'archetype': archetype,
        'characteristics': characteristics,
        'chunking_strategy': get_chunking_strategy(archetype),
        'indexing_strategy': get_indexing_strategy(archetype)
    }

def get_chunking_strategy(archetype):
    """
    Get appropriate chunking strategy for the detected archetype.
    
    Args:
        archetype (str): Detected JSON archetype
        
    Returns:
        dict: Chunking strategy configuration
    """
    strategies = {
        'api_response': {
            'method': 'resource_based',
            'max_size': 1000,
            'preserve_metadata': True
        },
        'event_log': {
            'method': 'temporal',
            'window_size': '1h',
            'overlap': '5m'
        },
        'metric_series': {
            'method': 'sliding_window',
            'window_size': 100,
            'stride': 50
        },
        'entity_graph': {
            'method': 'entity_centered',
            'context_depth': 2,
            'include_references': True
        },
        'state_machine': {
            'method': 'state_based',
            'include_transitions': True,
            'group_by_entity': True
        },
        'configuration': {
            'method': 'hierarchical',
            'max_depth': 3,
            'preserve_paths': True
        }
    }
    
    return strategies.get(archetype.split('_with_')[0], {
        'method': 'default',
        'max_size': 1000
    })

def get_indexing_strategy(archetype):
    """
    Get appropriate indexing strategy for the detected archetype.
    
    Args:
        archetype (str): Detected JSON archetype
        
    Returns:
        dict: Indexing strategy configuration
    """
    strategies = {
        'api_response': {
            'method': 'resource_based',
            'index_metadata': True,
            'include_status': True
        },
        'event_log': {
            'method': 'temporal',
            'index_timestamps': True,
            'index_states': True
        },
        'metric_series': {
            'method': 'numeric',
            'index_values': True,
            'track_statistics': True
        },
        'entity_graph': {
            'method': 'graph',
            'index_relationships': True,
            'track_references': True
        },
        'state_machine': {
            'method': 'state_based',
            'index_transitions': True,
            'track_sequences': True
        },
        'configuration': {
            'method': 'hierarchical',
            'index_paths': True,
            'track_dependencies': True
        }
    }
    
    return strategies.get(archetype.split('_with_')[0], {
        'method': 'default',
        'index_all': True
    })

def create_chunks(data, chunking_strategy):
    """
    Create chunks based on detected archetype and strategy.
    
    Args:
        data: JSON data to chunk
        chunking_strategy (dict): Strategy configuration for chunking
        
    Returns:
        list: Generated chunks based on strategy
    """
    method = chunking_strategy.get('method', 'default')
    
    if method == 'temporal':
        return create_timeseries_chunks(data, chunking_strategy)
    elif method == 'resource_based':
        return create_api_chunks(data, chunking_strategy)
    elif method == 'entity_centered':
        return create_entity_chunks(data, chunking_strategy)
    elif method == 'state_based':
        return create_state_chunks(data, chunking_strategy)
    else:
        return json_to_path_chunks(data)

def create_timeseries_chunks(data, strategy):
    """
    Create chunks for time series data.
    
    Args:
        data: JSON data containing time series
        strategy (dict): Chunking strategy configuration
        
    Returns:
        list: Time-window based chunks
    """
    chunks = []
    window_size = parse_duration(strategy.get('window_size', '5m'))
    overlap = parse_duration(strategy.get('overlap', '1m'))
    
    def extract_metrics(obj, timestamp=None, path="$"):
        metrics = []
        if isinstance(obj, dict):
            # Look for timestamp first
            new_timestamp = obj.get('timestamp') or obj.get('time') or timestamp
            
            for key, value in obj.items():
                current_path = f"{path}.{key}"
                if isinstance(value, (int, float)) and key not in ['timestamp', 'time']:
                    metrics.append({
                        'name': key,
                        'value': value,
                        'timestamp': new_timestamp,
                        'path': current_path
                    })
                elif isinstance(value, (dict, list)):
                    metrics.extend(extract_metrics(value, new_timestamp, current_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                metrics.extend(extract_metrics(item, timestamp, f"{path}[{i}]"))
        return metrics
    
    # Extract all metrics with timestamps
    all_metrics = extract_metrics(data)
    
    # Group metrics by time window
    windows = defaultdict(list)
    for metric in all_metrics:
        if metric['timestamp']:
            window_start = parse_timestamp(metric['timestamp'])
            window_start = window_start.replace(
                second=window_start.second - (window_start.second % window_size.seconds),
                microsecond=0
            )
            windows[window_start].append(metric)
    
    # Create overlapping windows if needed
    if overlap:
        overlapped_windows = defaultdict(list)
        for window_start, metrics in windows.items():
            overlapped_windows[window_start].extend(metrics)
            if overlap:
                # Add metrics to overlapping windows
                overlap_start = window_start - overlap
                if overlap_start in windows:
                    overlapped_windows[overlap_start].extend(metrics)
        windows = overlapped_windows
    
    # Create chunks for each window
    for window_start, window_metrics in sorted(windows.items()):
        chunks.append({
            'type': 'time_series',
            'window_start': window_start,
            'window_end': window_start + window_size,
            'metrics': window_metrics,
            'statistics': calculate_window_statistics(window_metrics)
        })
    
    return chunks

def create_api_chunks(data, strategy):
    """
    Create chunks for API response data.
    
    Args:
        data: JSON API response data
        strategy (dict): Chunking strategy configuration
        
    Returns:
        list: API resource chunks
    """
    chunks = []
    max_size = strategy.get('max_size', 1000)
    preserve_metadata = strategy.get('preserve_metadata', True)
    
    # Extract pagination info if present
    pagination = None
    metadata = {}
    if isinstance(data, dict):
        if 'pagination' in data:
            pagination = data['pagination']
        elif 'meta' in data and 'pagination' in data['meta']:
            pagination = data['meta']['pagination']
            
        # Collect metadata
        if preserve_metadata:
            metadata = {
                k: v for k, v in data.items() 
                if k in ['meta', 'status', 'code', 'version']
            }
    
    def process_items(items, context=None):
        current_chunk = []
        current_size = 0
        
        for item in items:
            item_size = len(str(item))
            if current_size + item_size > max_size and current_chunk:
                # Create chunk and start new one
                chunks.append({
                    'type': 'api_resource',
                    'content': current_chunk,
                    'pagination_context': pagination,
                    'metadata': metadata,
                    'parent_context': context
                })
                current_chunk = []
                current_size = 0
            
            current_chunk.append(item)
            current_size += item_size
        
        # Add remaining items
        if current_chunk:
            chunks.append({
                'type': 'api_resource',
                'content': current_chunk,
                'pagination_context': pagination,
                'metadata': metadata,
                'parent_context': context
            })
    
    # Find and process item arrays
    if isinstance(data, dict):
        for key, value in data.items():
            if key in ['items', 'data', 'results'] and isinstance(value, list):
                process_items(value, context={'container_key': key})
            elif isinstance(value, dict):
                create_api_chunks(value, strategy)
    
    return chunks

def create_entity_chunks(data, strategy):
    """
    Create entity-centered chunks.
    
    Args:
        data: JSON data containing entities
        strategy (dict): Chunking strategy configuration
        
    Returns:
        list: Entity-based chunks
    """
    chunks = []
    context_depth = strategy.get('context_depth', 2)
    include_references = strategy.get('include_references', True)
    
    # First pass: identify entities and their relationships
    entities = extract_entities(data)
    
    # Second pass: create chunks with context
    for entity_id, entity_info in entities.items():
        # Get related entities within context depth
        related_entities = get_related_entities(
            entities, 
            entity_id, 
            max_depth=context_depth
        )
        
        chunk = {
            'type': 'entity',
            'entity_id': entity_id,
            'entity_type': entity_info['type'],
            'content': entity_info['context'],
            'path': entity_info['path'],
            'related_entities': related_entities
        }
        
        if include_references:
            chunk['references'] = find_entity_references(data, entity_id)
            
        chunks.append(chunk)
    
    return chunks

def create_state_chunks(data, strategy):
    """
    Create state-based chunks.
    
    Args:
        data: JSON data containing state information
        strategy (dict): Chunking strategy configuration
        
    Returns:
        list: State-based chunks
    """
    chunks = []
    include_transitions = strategy.get('include_transitions', True)
    group_by_entity = strategy.get('group_by_entity', True)
    
    def extract_states(obj, path="$", entity_id=None):
        states = []
        if isinstance(obj, dict):
            # Look for state indicators
            state_fields = {k: v for k, v in obj.items() 
                          if k.lower() in ['status', 'state', 'phase']}
            
            if state_fields:
                state = {
                    'path': path,
                    'states': state_fields,
                    'entity_id': entity_id,
                    'timestamp': obj.get('timestamp') or obj.get('time'),
                    'context': {k: v for k, v in obj.items() 
                              if k not in state_fields and k not in ['timestamp', 'time']}
                }
                states.append(state)
            
            # Look for entity ID if not already found
            if not entity_id:
                for key in ['id', 'uuid', 'key']:
                    if key in obj:
                        entity_id = obj[key]
                        break
            
            # Recurse into nested objects
            for key, value in obj.items():
                states.extend(extract_states(value, f"{path}.{key}", entity_id))
                
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                states.extend(extract_states(item, f"{path}[{i}]", entity_id))
                
        return states
    
    # Extract all states
    all_states = extract_states(data)
    
    if group_by_entity:
        # Group states by entity
        entity_states = defaultdict(list)
        for state in all_states:
            if state['entity_id']:
                entity_states[state['entity_id']].append(state)
        
        # Create chunks for each entity
        for entity_id, states in entity_states.items():
            if include_transitions:
                transitions = detect_state_transitions(states)
            else:
                transitions = None
                
            chunks.append({
                'type': 'state_sequence',
                'entity_id': entity_id,
                'states': states,
                'transitions': transitions
            })
    else:
        # Create individual state chunks
        for state in all_states:
            chunks.append({
                'type': 'state',
                'content': state
            })
    
    return chunks

def calculate_window_statistics(metrics):
    """Calculate statistical summaries for a window of metrics."""
    stats = defaultdict(dict)
    for metric in metrics:
        name = metric['name']
        value = metric['value']
        if name not in stats:
            stats[name] = {
                'count': 0,
                'sum': 0,
                'values': []
            }
        stats[name]['count'] += 1
        stats[name]['sum'] += value
        stats[name]['values'].append(value)
    
    # Calculate statistics
    for name, data in stats.items():
        values = data['values']
        data['mean'] = data['sum'] / data['count']
        data['min'] = min(values)
        data['max'] = max(values)
        if len(values) > 1:
            data['std_dev'] = statistics.stdev(values)
        del data['values']  # Remove raw values to save space
    
    return dict(stats)

def parse_duration(duration_str):
    """Parse duration string into timedelta."""
    if not duration_str:
        return timedelta(minutes=5)  # default
    
    unit = duration_str[-1]
    value = int(duration_str[:-1])
    
    if unit == 's':
        return timedelta(seconds=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    else:
        raise ValueError(f"Invalid duration unit: {unit}")

def analyze_query_intent(query):
    """
    Analyze query to determine type of search needed.
    
    Args:
        query (str): User's search query
        
    Returns:
        dict: Query intent analysis results
    """
    temporal_patterns = [
        r'when', r'time', r'during', r'before', r'after',
        r'latest', r'recent', r'first', r'last'
    ]
    
    aggregation_patterns = [
        r'how many', r'count', r'total', r'average',
        r'min', r'max', r'sum', r'mean'
    ]
    
    pagination_patterns = [
        r'all', r'every', r'list', r'show me',
        r'page', r'next', r'previous'
    ]
    
    entity_patterns = [
        r'who', r'user', r'person', r'owner',
        r'assigned to', r'created by'
    ]
    
    state_patterns = [
        r'status', r'state', r'condition',
        r'phase', r'stage', r'step'
    ]
    
    query_lower = query.lower()
    
    intents = {
        'temporal': any(re.search(p, query_lower) for p in temporal_patterns),
        'aggregation': any(re.search(p, query_lower) for p in aggregation_patterns),
        'pagination': any(re.search(p, query_lower) for p in pagination_patterns),
        'entity': any(re.search(p, query_lower) for p in entity_patterns),
        'state': any(re.search(p, query_lower) for p in state_patterns)
    }
    
    # Determine primary intent
    primary_intent = max(intents.items(), key=lambda x: x[1])[0]
    all_intents = [k for k, v in intents.items() if v]
    
    # Extract additional context
    context = {
        'time_range': extract_time_range(query) if intents['temporal'] else None,
        'metric_conditions': extract_metric_conditions(query) if intents['aggregation'] else None,
        'pagination_info': extract_pagination_info(query) if intents['pagination'] else None,
        'entity_references': extract_entity_references(query) if intents['entity'] else None
    }
    
    return {
        'primary_intent': primary_intent,
        'all_intents': all_intents,
        'context': context
    }

def create_archetype_indices(conn, chunks, archetype):
    """
    Create appropriate indices based on archetype.
    
    Args:
        conn: PostgreSQL database connection
        chunks (list): List of chunks to index
        archetype (str): Detected archetype
    """
    cur = conn.cursor()
    
    # Create base tables if they don't exist
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunk_metadata (
            chunk_id TEXT PRIMARY KEY,
            archetype TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    if archetype == 'event_log':
        create_temporal_index(conn, chunks)
        create_actor_index(conn, chunks)
        create_state_index(conn, chunks)
        
    elif archetype.startswith('api_response'):
        create_resource_index(conn, chunks)
        create_id_lookup_index(conn, chunks)
        
    elif archetype == 'metric_series':
        create_numeric_range_index(conn, chunks)
        create_metric_name_index(conn, chunks)
        
    elif archetype == 'entity_graph':
        create_entity_relationship_index(conn, chunks)
        
    elif archetype == 'state_machine':
        create_state_transition_index(conn, chunks)
    
    # Always create vector index as fallback
    create_vector_index(conn, chunks)
    
    conn.commit()

def hybrid_search(conn, query, archetype):
    """
    Perform hybrid search based on query intent and archetype.
    
    Args:
        conn: PostgreSQL database connection
        query (str): User's search query
        archetype (str): Document archetype
        
    Returns:
        list: Search results
    """
    intent = analyze_query_intent(query)
    results = []
    
    # Primary search based on intent and archetype
    if intent['primary_intent'] == 'temporal' and archetype == 'event_log':
        results = temporal_search(conn, query, intent['context']['time_range'])
        
    elif intent['primary_intent'] == 'aggregation' and archetype == 'metric_series':
        results = metric_search(conn, query, intent['context']['metric_conditions'])
        
    elif intent['primary_intent'] == 'pagination' and archetype.startswith('api_response'):
        results = paginated_search(conn, query, intent['context']['pagination_info'])
        
    elif intent['primary_intent'] == 'entity':
        results = entity_search(conn, query, intent['context']['entity_references'])
        
    elif intent['primary_intent'] == 'state':
        results = state_search(conn, query)
    
    # If no results or low confidence, fallback to vector search
    if not results or len(results) < 3:
        vector_results = vector_search(conn, query)
        results = merge_search_results(results, vector_results)
    
    return results

def merge_search_results(primary_results, fallback_results, max_results=10):
    """
    Merge results from different search methods.
    
    Args:
        primary_results (list): Results from primary search method
        fallback_results (list): Results from fallback search method
        max_results (int): Maximum number of results to return
        
    Returns:
        list: Merged and deduplicated results
    """
    seen_ids = set()
    merged = []
    
    # Add primary results first
    for result in primary_results:
        result_id = result.get('id')
        if result_id not in seen_ids:
            seen_ids.add(result_id)
            merged.append(result)
    
    # Add fallback results if needed
    for result in fallback_results:
        if len(merged) >= max_results:
            break
        result_id = result.get('id')
        if result_id not in seen_ids:
            seen_ids.add(result_id)
            merged.append(result)
    
    return merged

def extract_time_range(query):
    """Extract time range from query."""
    # Add implementation
    pass

def extract_metric_conditions(query):
    """Extract metric conditions from query."""
    # Add implementation
    pass

def extract_pagination_info(query):
    """Extract pagination information from query."""
    # Add implementation
    pass

def extract_entity_references(query):
    """Extract entity references from query."""
    # Add implementation
    pass

def summarize_event_sequence(events):
    """
    Enhanced event sequence summarization.
    
    Args:
        events (list): List of events to summarize
        
    Returns:
        dict: Detailed event sequence analysis
    """
    # Group events by causal chains
    chains = defaultdict(list)
    automation_triggers = set()
    
    for event in sorted(events, key=lambda x: x['timestamp']):
        # Track automated actions
        if 'triggered_by' in event:
            automation_triggers.add(event['triggered_by']['type'])
        
        # Group by service/component
        service = event.get('service', 'unknown')
        chains[service].append(event)
    
    summaries = []
    for service, sequence in chains.items():
        # Build service-specific timeline
        timeline = {
            'service': service,
            'start_time': sequence[0]['timestamp'],
            'end_time': sequence[-1]['timestamp'],
            'transitions': [],
            'metrics': defaultdict(list)
        }
        
        # Track state transitions and metrics
        previous_state = None
        for event in sequence:
            # Track metrics if present
            if 'metrics' in event:
                for metric, value in event['metrics'].items():
                    timeline['metrics'][metric].append({
                        'time': event['timestamp'],
                        'value': value
                    })
            
            # Track state changes
            current_state = event['event_type']
            if previous_state != current_state:
                timeline['transitions'].append({
                    'from': previous_state,
                    'to': current_state,
                    'time': event['timestamp'],
                    'automated': bool(event.get('triggered_by'))
                })
                previous_state = current_state
        
        summaries.append(timeline)
    
    return {
        'chains': summaries,
        'automation_types': list(automation_triggers),
        'duration': max(s['end_time'] for s in summaries) - min(s['start_time'] for s in summaries)
    }

def summarize_api_data(response):
    """
    Enhanced API response summarization.
    
    Args:
        response (dict): API response to summarize
        
    Returns:
        dict: Detailed API response analysis
    """
    # Extract pagination context
    pagination = response.get('meta', {})
    is_complete = pagination.get('page') == pagination.get('total_pages')
    
    # Track resource relationships and aggregates
    resources = defaultdict(lambda: {
        'count': 0,
        'relationships': defaultdict(set),
        'aggregates': defaultdict(float),
        'thresholds': defaultdict(list)
    })
    
    def process_resource(data, parent=None):
        for key, value in data.items():
            if isinstance(value, dict):
                if 'id' in value:  # This is a resource
                    resource_type = key
                    resource_id = value['id']
                    resources[resource_type]['count'] += 1
                    
                    # Track relationship to parent
                    if parent:
                        resources[resource_type]['relationships'][parent['type']].add(parent['id'])
                    
                    # Process numeric values for aggregation
                    for k, v in value.items():
                        if isinstance(v, (int, float)):
                            resources[resource_type]['aggregates'][k] += v
                        elif isinstance(v, dict) and 'threshold' in v:
                            resources[resource_type]['thresholds'][k].append(v['threshold'])
                    
                    # Recurse with new parent context
                    process_resource(value, {'type': resource_type, 'id': resource_id})
                else:
                    # Recurse without changing parent
                    process_resource(value, parent)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        process_resource(item, parent)
    
    process_resource(response.get('data', {}))
    
    return {
        'pagination_context': {
            'current_page': pagination.get('page'),
            'total_pages': pagination.get('total_pages'),
            'is_complete': is_complete
        },
        'resources': dict(resources),
        'cross_resource_relationships': {
            r_type: dict(rels) for r_type, rels in 
            ((t, d['relationships']) for t, d in resources.items())
        }
    }

def summarize_metric_progression(metrics):
    """
    Enhanced metric series summarization.
    
    Args:
        metrics (list): List of metrics to analyze
        
    Returns:
        dict: Detailed metric analysis
    """
    def calculate_rate_of_change(values, timestamps):
        if len(values) < 2:
            return None
        time_diff = (timestamps[-1] - timestamps[0]).total_seconds()
        value_diff = values[-1] - values[0]
        return value_diff / time_diff if time_diff > 0 else 0

    # Track each metric's progression
    metric_analysis = defaultdict(lambda: {
        'values': [],
        'timestamps': [],
        'threshold_violations': [],
        'rate_of_change': None,
        'correlations': {}
    })
    
    # First pass: collect values
    for timepoint in metrics:
        timestamp = parse_timestamp(timepoint['timestamp'])
        for metric_name, value in timepoint.items():
            if isinstance(value, (int, float)) and metric_name != 'timestamp':
                metric_analysis[metric_name]['values'].append(value)
                metric_analysis[metric_name]['timestamps'].append(timestamp)
                
                # Check for threshold violations
                if 'alerts' in timepoint:
                    for alert in timepoint['alerts']:
                        if alert['metric'] == metric_name:
                            metric_analysis[metric_name]['threshold_violations'].append({
                                'time': timestamp,
                                'threshold': alert['threshold'],
                                'value': value
                            })
    
    # Second pass: calculate rates and correlations
    metric_names = list(metric_analysis.keys())
    for metric_name in metric_names:
        analysis = metric_analysis[metric_name]
        
        # Calculate rate of change
        analysis['rate_of_change'] = calculate_rate_of_change(
            analysis['values'],
            analysis['timestamps']
        )
        
        # Calculate correlations with other metrics
        for other_metric in metric_names:
            if other_metric != metric_name:
                correlation = calculate_correlation(
                    analysis['values'],
                    metric_analysis[other_metric]['values']
                )
                if abs(correlation) > 0.5:  # Only track strong correlations
                    analysis['correlations'][other_metric] = correlation
    
    return {
        'metrics': dict(metric_analysis),
        'time_range': {
            'start': min(min(m['timestamps']) for m in metric_analysis.values()),
            'end': max(max(m['timestamps']) for m in metric_analysis.values())
        },
        'significant_correlations': [
            (m1, m2, analysis['correlations'][m2])
            for m1, analysis in metric_analysis.items()
            for m2 in analysis['correlations']
            if abs(analysis['correlations'][m2]) > 0.8  # Very strong correlations
        ]
    }

def calculate_correlation(values1, values2):
    """
    Calculate Pearson correlation coefficient between two series.
    
    Args:
        values1 (list): First series of values
        values2 (list): Second series of values
        
    Returns:
        float: Correlation coefficient
    """
    if len(values1) != len(values2):
        return 0
    n = len(values1)
    if n == 0:
        return 0
    
    mean1 = sum(values1) / n
    mean2 = sum(values2) / n
    
    variance1 = sum((x - mean1) ** 2 for x in values1)
    variance2 = sum((x - mean2) ** 2 for x in values2)
    
    covariance = sum((values1[i] - mean1) * (values2[i] - mean2) for i in range(n))
    
    if variance1 == 0 or variance2 == 0:
        return 0
        
    return covariance / (variance1 * variance2) ** 0.5

def main():
    """
    Main entry point for the application.
    
    Handles command line arguments, database connection,
    and orchestrates the overall application flow.
    """
    conn = psycopg2.connect(POSTGRES_CONN_STR)
    try:
        # Initial load/embedding
        load_and_embed_new_data(conn)
        
        if len(sys.argv) > 1:
            parser = argparse.ArgumentParser(description="JSON-based RAG system.")
            parser.add_argument("--new", action="store_true", help="Reset the database (clear all data)")
            args = parser.parse_args()
            
            if args.new:
                reset_database(conn)
                load_and_embed_new_data(conn)  # Reload data after reset

        # Start interactive chat
        chat_loop(conn)
        
    finally:
        conn.close()

if __name__ == "__main__":
    main()
