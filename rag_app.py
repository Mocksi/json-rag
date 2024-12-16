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

def json_to_path_chunks(json_obj, current_path="$"):
    chunks = []
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            seen_keys.add(key)
            new_path = f"{current_path}.{key}"
            chunks.extend(json_to_path_chunks(value, current_path=new_path))
    elif isinstance(json_obj, list):
        seen_keys.add(current_path + "[]")
        for i, item in enumerate(json_obj):
            new_path = f"{current_path}[{i}]"
            chunks.extend(json_to_path_chunks(item, current_path=new_path))
    else:
        val_type = type(json_obj).__name__
        val_str = str(json_obj)
        text_representation = f"Path: {current_path}\nType: {val_type}\nValue: {val_str}"
        chunks.append(text_representation)
    return chunks

def load_and_embed_new_data(conn):
    to_process = get_files_to_process(conn)
    if not to_process:
        print("No new or changed files to process.")
        return

    print(f"Processing {len(to_process)} new or modified files...")
    
    documents = []
    for f, f_hash, f_mtime in to_process:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                validated = FlexibleModel.parse_obj(data)
                documents.append((f, f_hash, f_mtime, validated.__root__))
                print(f"Validated document: {f}")
        except (ValidationError, json.JSONDecodeError) as e:
            print(f"Skipping invalid JSON document {f}:", e)

    print(f"Successfully validated {len(documents)} documents")

    all_chunks = []
    for (f, f_hash, f_mtime, doc) in documents:
        doc_chunks = json_to_path_chunks(doc)
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
