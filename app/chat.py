from typing import List, Dict, Optional, Tuple
import json
from app.parsing import extract_key_value_pairs, json_to_path_chunks
from app.embedding import get_embedding
from app.archetype import ArchetypeDetector
from app.relationships import detect_relationships
from app.database import (
    get_files_to_process,
    upsert_file_metadata,
    save_chunk_archetypes
)
from app.utils import get_json_files, compute_file_hash
from pydantic import ValidationError
from app.retrieval import answer_query
from app.database import upsert_file_metadata, save_chunk_archetypes
from app.utils import get_json_files
from app.models import FlexibleModel
from app.config import MAX_CHUNKS, embedding_model
from app.archetype import ArchetypeDetector
from app.relationships import detect_relationships
from typing import List, Dict
import json

def load_and_embed_new_data(conn):
    """Load and embed new or modified JSON files."""
    print("\nChecking for new data to embed...")
    
    # Get list of files that need processing
    documents = get_files_to_process(conn, compute_file_hash, get_json_files)
    
    if not documents:
        print("No new or modified files to process.")
        return True
        
    print(f"Found {len(documents)} files to process")
    
    # Process each file
    all_chunks = []
    detector = ArchetypeDetector()
    
    for f, f_hash, f_mtime in documents:
        print(f"\nProcessing {f}")
        
        try:
            with open(f, 'r') as file:
                data = json.load(file)
        except Exception as e:
            print(f"Error loading file {f}: {e}")
            continue
        
        # Generate chunks with metadata
        doc_chunks = json_to_path_chunks(data, f)
        all_chunks.extend((f, chunk) for chunk in doc_chunks)
    
    if not all_chunks:
        print("No chunks to embed from new files.")
        return True

    print(f"Generated {len(all_chunks)} chunks")
    print("Generating embeddings...")

    # Generate embeddings in batches
    chunk_texts = [json.dumps(c[1]) for c in all_chunks]  # Serialize chunk data
    chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)
    
    # Store chunks and embeddings
    cur = conn.cursor()
    for i, (f, chunk) in enumerate(all_chunks):
        chunk_id = f"{f}:{i}"
        embedding_list = chunk_embeddings[i].tolist()
        embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
        
        # Detect archetypes and relationships
        chunk_archetypes = detector.detect_archetypes(chunk)
        print(f"\nDEBUG: Chunk type before relationships: {type(chunk)}")
        print(f"DEBUG: Chunk content: {chunk}")

        # Get archetype for embedding
        archetype = None
        if chunk_archetypes:
            archetype = {
                'type': chunk_archetypes[0][0],  # First archetype type
                'confidence': chunk_archetypes[0][1]  # First archetype confidence
            }

        # Generate embedding with archetype context
        embedding_list = get_embedding(json.dumps(chunk), archetype).tolist()
        embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"

        relationships = detect_relationships([chunk], conn)
        
        chunk_metadata = {
            'archetypes': [
                {'type': a[0], 'confidence': a[1]} 
                for a in chunk_archetypes
            ] if chunk_archetypes else [],
            'source_file': f,
            'chunk_index': i,
            'relationships': relationships.get('direct', {})
        }
        
        # Store the chunk with its embedding
        cur.execute("""
            INSERT INTO json_chunks (id, chunk_text, chunk_json, embedding, metadata)
            VALUES (%s, %s, %s::jsonb, %s::vector, %s::jsonb)
            ON CONFLICT (id) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                chunk_json = EXCLUDED.chunk_json,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata
        """, (
            chunk_id,
            json.dumps(chunk),  # Store full chunk as text
            json.dumps(chunk),  # Store as JSONB for querying
            embedding_str,
            json.dumps(chunk_metadata)
        ))
        
        # Store relationships in dedicated table
        if relationships.get('direct'):
            for rel_type, rels in relationships['direct'].items():
                for rel in rels:
                    # Create synthetic chunk IDs for entity references
                    source_id = f"{f}:{chunk_id}"
                    target_id = f"entity:{rel['target']}"
                    
                    cur.execute("""
                        INSERT INTO chunk_relationships 
                        (source_chunk_id, target_chunk_id, relationship_type, metadata)
                        VALUES (%s, %s, %s, %s::jsonb)
                        ON CONFLICT (source_chunk_id, target_chunk_id, relationship_type) 
                        DO UPDATE SET metadata = EXCLUDED.metadata
                    """, (
                        source_id,
                        target_id, 
                        rel['type'],
                        json.dumps(rel['context'])
                    ))
    
    # Update file metadata
    for f, f_hash, f_mtime in documents:
        upsert_file_metadata(conn, f, f_hash, f_mtime)
    
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

def build_prompt(query: str, context: List[str], query_intent: Dict) -> str:
    """Build prompt with enhanced context formatting."""
    prompt = "Use the provided context to answer the user's query.\n\n"
    
    # Add intent-specific guidelines
    prompt += f"Context type: {query_intent['primary_intent']}\n"
    prompt += f"Additional intents: {', '.join(query_intent['all_intents'])}\n\n"
    
    # Add analysis guidelines
    prompt += "Guidelines:\n"
    prompt += "- Focus on aggregation aspects\n"
    prompt += "- Use specific details from the context\n"
    prompt += "- Always use human-readable names in addition to IDs when available\n"
    prompt += "- Show relationships between named entities\n"
    prompt += "- If information isn't in the context, say so\n"
    prompt += "- For temporal queries, preserve chronological order\n"
    prompt += "- For metrics, include specific values and trends\n\n"
    
    # Format context entries
    prompt += "Context:\n"
    if context:
        prompt += "\n".join(f"- {entry}" for entry in context)
    prompt += "\n\n"
    
    # Add query
    prompt += f"Question: {query}\n\n"
    prompt += "Answer based only on the provided context, using human-readable names."
    
    return prompt
