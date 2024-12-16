import os
import json
import psycopg2
from pydantic import BaseModel, ValidationError
from sentence_transformers import SentenceTransformer
import openai
import argparse
from datetime import datetime
import hashlib
import glob
import sys
from dotenv import load_dotenv

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
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def get_json_files():
    return glob.glob(os.path.join(DATA_DIR, "*.json"))

def get_file_info(conn):
    cur = conn.cursor()
    cur.execute("SELECT filename, file_hash, last_modified FROM file_metadata;")
    rows = cur.fetchall()
    return {r[0]: (r[1], r[2]) for r in rows}

def upsert_file_metadata(conn, filename, file_hash, mod_time):
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
    """Create an enhanced text chunk with path, type, value, and related context."""
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
    """Track relationships between entities with enhanced context"""
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
    """Enhanced entity extraction with relationship tracking"""
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
                'group_context': rel.get('group_context')
            }
    
    return entities

def load_and_embed_new_data(conn):
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

    conn.commit()
    print(f"Embedded {len(all_chunks)} new chunks.")

    for f, f_hash, f_mtime, d in documents:
        upsert_file_metadata(conn, f, f_hash, f_mtime)

def get_relevant_chunks(conn, query, top_k=5):
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

def answer_query(conn, query):
    retrieved_chunks = get_relevant_chunks(conn, query)
    if len(retrieved_chunks) > MAX_CHUNKS:
        retrieved_chunks = retrieved_chunks[:MAX_CHUNKS]

    prompt = build_prompt(query, retrieved_chunks)
    
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
    print("Enter your queries below. Type ':quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == ":quit":
            print("Exiting chat.")
            break
        
        load_and_embed_new_data(conn)

        try:
            answer = answer_query(conn, user_input)
            print("Assistant:", answer)
        except Exception as e:
            print("Error processing query:", str(e))

def reset_database(conn):
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

def main():
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
