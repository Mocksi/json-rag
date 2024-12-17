from app.retrieval import answer_query
from app.database import get_files_to_process, upsert_file_metadata
from app.utils import compute_file_hash, get_json_files
from app.models import FlexibleModel
from app.parsing import extract_entities, json_to_path_chunks, extract_key_value_pairs
from app.config import MAX_CHUNKS

def load_and_embed_new_data(conn):
    """
    Processes new or modified JSON files, extracts entities and relationships,
    generates embeddings, and stores them in the database.
    
    Args:
        conn: PostgreSQL database connection
        
    Returns:
        bool: True if processing was successful, None if no files to process
        
    Flow:
        1. Identifies new/modified JSON files
        2. Validates and extracts entities
        3. Generates chunks with context
        4. Creates embeddings
        5. Stores in database with key-value pairs
    """
    from app.retrieval import get_relevant_chunks
    from app.database import get_files_to_process
    to_process = get_files_to_process(conn, compute_file_hash, get_json_files)
    if not to_process:
        print("No new or changed files to process.")
        return
    print(f"Processing {len(to_process)} new or modified files...")

    documents = []
    entities_by_file = {}

    import json
    from datetime import datetime
    for f, f_hash, f_mtime in to_process:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                validated = FlexibleModel.parse_obj(data)
                documents.append((f, f_hash, f_mtime, validated.__root__))
                entities = extract_entities(validated.__root__)
                entities_by_file[f] = entities
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

    for f, entities in entities_by_file.items():
        if entities:
            print(f"\nEntities found in {f}:")
            for entity_id, info in entities.items():
                print(f"  - {info['type']}={entity_id} at {info['path']}")
                if info.get('context'):
                    context_summary = {k: str(v) for k, v in info['context'].items() if k != info['type']}
                    print(f"    Context: {json.dumps(context_summary, indent=2)}")

    all_chunks = []
    from app.parsing import json_to_path_chunks, extract_key_value_pairs
    from app.database import upsert_file_metadata
    from app.config import embedding_model
    from app.embedding import vector_search_with_filter

    seen_keys = set()
    for (f, f_hash, f_mtime, doc) in documents:
        doc_chunks = json_to_path_chunks(doc, entities=entities_by_file.get(f))
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
    for i, (f, chunk) in enumerate(all_chunks):
        embedding_list = chunk_embeddings[i].tolist()
        embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
        chunk_id = f"{f}:{i}"
        
        # Serialize chunk to both text and JSON
        chunk_text = json.dumps(chunk)
        
        cur.execute("""
            INSERT INTO json_chunks (id, chunk_text, chunk_json, embedding, metadata)
            VALUES (%s, %s, %s::jsonb, %s, '{}'::jsonb)
            ON CONFLICT (id) DO NOTHING;
        """, (chunk_id, chunk_text, chunk_text, embedding_str))
        
        try:
            # Extract key-value pairs from the chunk dictionary directly
            pairs = {}
            
            # Add path information
            path = chunk.get('path', '')
            pairs['path'] = path
            path_parts = path.split('.')
            if len(path_parts) > 1:
                pairs['path_root'] = path_parts[0]
                pairs['path_leaf'] = path_parts[-1]
            
            # Add value information
            value_data = chunk.get('value', {})
            if isinstance(value_data, dict):
                pairs['value_type'] = value_data.get('type', 'unknown')
                if value_data.get('type') == 'primitive':
                    val = value_data.get('value')
                    if isinstance(val, (int, float, str)):
                        pairs['value'] = str(val)
                        pairs['python_type'] = value_data.get('python_type', '')
            
            # Add context information
            for key, value in chunk.get('context', {}).items():
                if isinstance(value, (str, int, float, bool)):
                    pairs[f"context_{key}"] = str(value)
            
            # Add display names
            for key, value in chunk.get('display_names', {}).items():
                pairs[f"display_{key}"] = str(value)
            
            # Add entity information
            for entity_id, entity_data in chunk.get('entities', {}).items():
                pairs[f"entity_{entity_id}"] = entity_data.get('type', 'unknown')
                if 'name' in entity_data:
                    pairs[f"entity_{entity_id}_name"] = str(entity_data['name'])
            
            # Insert key-value pairs
            for k, v in pairs.items():
                cur.execute("""
                    INSERT INTO chunk_key_values (chunk_id, key, value)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (chunk_id, key) DO UPDATE SET value = EXCLUDED.value;
                """, (chunk_id, k, str(v)))
                
        except Exception as e:
            print(f"Error processing chunk {chunk_id}: {e}")
            continue

    conn.commit()
    cur.close()
    
    print("Successfully embedded and indexed new chunks")
    return True

def initialize_embeddings(conn):
    """
    Initializes embeddings for all JSON files in the data directory.
    Called after database reset or when changes are detected.
    
    Args:
        conn: PostgreSQL database connection
    """
    print("Initializing embeddings for all JSON files...")
    load_and_embed_new_data(conn)

def chat_loop(conn):
    """
    Runs the main interactive chat loop, processing user queries and returning answers.
    Automatically checks for and processes new/modified files before each query.
    
    Args:
        conn: PostgreSQL database connection
        
    Commands:
        :quit - Exits the chat loop
    """
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
