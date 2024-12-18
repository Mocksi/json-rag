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
from app.json_parser import generate_chunk_id, normalize_json_path
from typing import List, Dict
import json

def serialize_for_debug(obj):
    """Helper function to serialize objects for debug output."""
    if hasattr(obj, 'tolist'):  # Handle numpy arrays
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: serialize_for_debug(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_debug(x) for x in obj]
    return obj

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
    chunk_cache = {}  # Cache to store chunks by ID
    
    # First pass: Process all files and collect chunks
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
    
    # Second pass: Store all chunks and build cache
    cur = conn.cursor()
    print("\nStoring chunks and building cache...")
    for i, (f, chunk) in enumerate(all_chunks):
        # Generate proper chunk ID using our new system
        chunk_id = generate_chunk_id(f, chunk.get('path', f'chunk_{i}'))
        chunk_cache[chunk_id] = chunk  # Cache the chunk
        
        # Detect archetypes
        chunk_archetypes = detector.detect_archetypes(chunk)
        print(f"\nDEBUG: Processing chunk {chunk_id}")
        print(f"DEBUG: Chunk path: {chunk.get('path')}")

        # Get archetype for embedding
        archetype = None
        if chunk_archetypes:
            archetype = {
                'type': chunk_archetypes[0][0],
                'confidence': chunk_archetypes[0][1]
            }

        # Generate embedding with archetype context
        embedding_list = get_embedding(json.dumps(chunk), archetype).tolist()
        embedding_str = "[" + ",".join(str(x) for x in embedding_list) + "]"
        
        chunk_metadata = {
            'archetypes': [
                {'type': a[0], 'confidence': a[1]} 
                for a in chunk_archetypes
            ] if chunk_archetypes else [],
            'source_file': f,
            'chunk_index': i
        }
        
        # Store the chunk with its embedding
        cur.execute("""
            INSERT INTO json_chunks (
                id, chunk_text, chunk_json, embedding, metadata,
                path, depth, parent_id, source_file
            )
            VALUES (%s, %s, %s::jsonb, %s::vector, %s::jsonb, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                chunk_text = EXCLUDED.chunk_text,
                chunk_json = EXCLUDED.chunk_json,
                embedding = EXCLUDED.embedding,
                metadata = EXCLUDED.metadata,
                path = EXCLUDED.path,
                depth = EXCLUDED.depth,
                parent_id = EXCLUDED.parent_id,
                source_file = EXCLUDED.source_file
        """, (
            chunk_id,
            json.dumps(chunk),
            json.dumps(chunk),
            embedding_str,
            json.dumps(chunk_metadata),
            chunk.get('path', ''),  # Add path
            chunk.get('depth', 0),  # Add depth
            chunk.get('parent_id'),  # Add parent_id
            f  # Add source_file
        ))
    
    # Third pass: Create relationships now that all chunks are cached
    print("\nCreating relationships...")
    
    # First, build a map of all IDs to their chunks
    id_to_chunk = {}
    for i, (f, chunk) in enumerate(all_chunks):
        chunk_id = generate_chunk_id(f, chunk.get('path', f'chunk_{i}'))
        
        # If this chunk has an ID value, map it
        if chunk.get('value') and isinstance(chunk.get('value'), str):
            path_parts = chunk.get('path', '').split('.')
            if any(part.endswith('id') or part == 'id' for part in path_parts):
                id_to_chunk[chunk.get('value')] = {
                    'chunk_id': chunk_id,
                    'path': chunk.get('path'),
                    'chunk': chunk
                }
    
    # Now process relationships using the ID map
    for i, (f, chunk) in enumerate(all_chunks):
        chunk_id = generate_chunk_id(f, chunk.get('path', f'chunk_{i}'))
        print(f"\nDEBUG: Processing relationships for chunk: {chunk_id}")
        print(f"DEBUG: Chunk path: {chunk.get('path')}")
        
        # Extract relationships from chunk value and context
        value = chunk.get('value')
        context = chunk.get('context', {})
        
        # Helper function to extract all ID fields from a dict
        def extract_ids(obj, prefix=''):
            ids = {}
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, str) and (k.endswith('_id') or k == 'id'):
                        ids[prefix + k if prefix else k] = v
                    elif isinstance(v, (dict, list)):
                        ids.update(extract_ids(v, prefix + k + '_' if prefix else k + '_'))
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, dict):
                        ids.update(extract_ids(item, prefix))
            return ids
        
        # Extract all IDs from context
        all_ids = extract_ids(context)
        print(f"DEBUG: Found IDs in context: {all_ids}")
        
        # Create relationships for each ID
        for key, val in all_ids.items():
            if val in id_to_chunk:
                target_info = id_to_chunk[val]
                print(f"DEBUG: Found target chunk for ID {val}: {target_info['path']}")
                
                # Determine relationship type based on key
                rel_type = 'reference'
                if key.startswith('shipments_'):
                    rel_type = 'shipment_reference'
                elif key.startswith('warehouses_'):
                    rel_type = 'warehouse_reference'
                elif key.startswith('products_'):
                    rel_type = 'product_reference'
                elif key.startswith('suppliers_'):
                    rel_type = 'supplier_reference'
                
                cur.execute("""
                    INSERT INTO chunk_relationships 
                    (source_chunk_id, target_chunk_id, relationship_type, metadata)
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (source_chunk_id, target_chunk_id, relationship_type) 
                    DO UPDATE SET metadata = EXCLUDED.metadata
                """, (
                    chunk_id,
                    target_info['chunk_id'],
                    rel_type,
                    json.dumps({
                        'field': key,
                        'source_path': chunk.get('path'),
                        'target_path': target_info['path'],
                        'target_value': val
                    })
                ))
            else:
                print(f"DEBUG: No chunk found with ID value: {val}")
    
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
