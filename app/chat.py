"""
JSON RAG Chat Module

This module provides the core chat functionality for the JSON RAG system,
handling data ingestion, embedding generation, and interactive querying.
It manages the processing of JSON files, creation of embeddings, and
maintains relationships between different chunks of data.

Key Components:
    - Data Ingestion: Processes and chunks JSON files
    - Embedding Generation: Creates vector embeddings for chunks
    - Relationship Detection: Maps connections between data chunks
    - Interactive Chat: Handles user queries and generates responses
    - Cache Management: Maintains efficient data access patterns

The module integrates with various components including:
    - Archetype detection for data pattern recognition
    - Vector embeddings for semantic search
    - PostgreSQL for persistent storage
    - Relationship mapping for data connections
"""

from typing import List, Dict, Optional, Tuple
import json
from pydantic import ValidationError
import argparse
from pathlib import Path
import logging

from app.processing.parsing import extract_key_value_pairs
from app.processing.json_parser import json_to_path_chunks, process_json_files
from app.retrieval.embedding import get_embedding
from app.analysis.archetype import ArchetypeDetector
from app.analysis.relationships import process_relationships
from app.storage.database import (
    get_files_to_process,
    upsert_file_metadata,
    save_chunk_archetypes,
    get_chunk_by_id,
    init_db
)
from app.utils.utils import get_json_files, compute_file_hash
from app.retrieval.retrieval import answer_query, QueryPipeline
from app.core.models import FlexibleModel
from app.core.config import MAX_CHUNKS, embedding_model
from app.processing.json_parser import generate_chunk_id, normalize_json_path
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

def serialize_for_debug(obj):
    """
    Serialize complex objects for debug output, handling special cases.
    
    This function provides a safe way to serialize objects that might contain
    non-JSON serializable types (like numpy arrays) into a format suitable
    for debug logging.
    
    Args:
        obj: Object to serialize, can be of any type
        
    Returns:
        Serialized version of the object with special types handled:
            - numpy arrays -> Python lists
            - dictionaries -> Recursively serialized dictionaries
            - lists -> Recursively serialized lists
            - Other types -> As is
            
    Example:
        >>> data = {'array': np.array([1, 2, 3]), 'nested': {'value': 42}}
        >>> serialized = serialize_for_debug(data)
        >>> print(serialized)
        {'array': [1, 2, 3], 'nested': {'value': 42}}
    """
    if hasattr(obj, 'tolist'):  # Handle numpy arrays
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: serialize_for_debug(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_debug(x) for x in obj]
    return obj

def load_and_embed_new_data(conn):
    """
    Load and embed new or modified JSON files into the vector database.
    
    This function performs a multi-pass process to load, chunk, embed,
    and index JSON files that have been added or modified since the last run.
    It handles the complete pipeline from file processing to relationship
    detection.
    
    Process Flow:
        1. File Detection:
            - Identifies new/modified JSON files
            - Computes file hashes for change detection
            
        2. Chunking (First Pass):
            - Processes each JSON file into chunks
            - Generates metadata for each chunk
            - Collects all chunks for batch processing
            
        3. Embedding Generation:
            - Creates vector embeddings for all chunks
            - Uses batch processing for efficiency
            
        4. Storage (Second Pass):
            - Stores chunks in database
            - Caches chunks for relationship detection
            - Detects and stores archetypes
            
        5. Relationship Detection (Third Pass):
            - Maps relationships between chunks
            - Creates bidirectional references
            - Stores relationship metadata
            
    Args:
        conn: PostgreSQL database connection
        
    Returns:
        bool: True if processing completed successfully
        
    Example:
        >>> with psycopg2.connect(POSTGRES_CONN_STR) as conn:
        ...     success = load_and_embed_new_data(conn)
        >>> print("Processing complete" if success else "Processing failed")
        
    Note:
        - Uses transaction to ensure data consistency
        - Handles errors gracefully for individual files
        - Maintains file metadata for incremental updates
    """
    print("\nChecking for new data to embed...")
    
    # Get list of files that need processing
    documents = get_files_to_process(conn, compute_file_hash, get_json_files)
    
    if not documents:
        logger.info("No new or modified files to process.")
        return True
        
    logger.info(f"Found {len(documents)} files to process")
    
    # Process each file
    all_chunks = []
    detector = ArchetypeDetector()
    chunk_cache = {}  # Cache to store chunks by ID
    
    # First pass: Process all files and collect chunks
    for f, f_hash, f_mtime in documents:
        logger.info(f"Processing {f}")
        
        try:
            with open(f, 'r') as file:
                data = json.load(file)
        except Exception as e:
            logger.error(f"Error loading file {f}: {e}")
            continue
        
        # Generate chunks with metadata
        doc_chunks = json_to_path_chunks(data, f)
        all_chunks.extend((f, chunk) for chunk in doc_chunks)
    
    if not all_chunks:
        print("No chunks to embed from new files.")
        return True

    logger.info(f"Generated {len(all_chunks)} chunks")
    logger.info("Generating embeddings...")

    # Generate embeddings in batches
    chunk_texts = [json.dumps(c[1]) for c in all_chunks]  # Serialize chunk data
    chunk_embeddings = embedding_model.encode(chunk_texts, show_progress_bar=True)
    
    # Second pass: Store all chunks and build cache
    cur = conn.cursor()
    logger.info("Storing chunks and building cache...")
    for i, (f, chunk) in enumerate(all_chunks):
        # Generate proper chunk ID using our new system
        chunk_id = generate_chunk_id(f, chunk.get('path', f'chunk_{i}'))
        chunk_cache[chunk_id] = chunk  # Cache the chunk
        
        # Detect archetypes
        chunk_archetypes = detector.detect_archetypes(chunk)
        logger.debug(f"Processing chunk {chunk_id}")
        logger.debug(f"Chunk path: {chunk.get('path')}")

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
    logger.info("Creating relationships...")
    
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
        logger.debug(f"Processing relationships for chunk: {chunk_id}")
        logger.debug(f"Chunk path: {chunk.get('path')}")
        
        # Extract relationships from chunk value and context
        value = chunk.get('value')
        context = chunk.get('context', {})
        
        # Helper function to extract all ID fields from a dict
        def extract_ids(obj, prefix=''):
            """
            Recursively extract ID fields from nested dictionaries and lists.
            
            This helper function traverses complex data structures to find and
            collect ID fields, maintaining the full path context through prefixing.
            It handles both direct ID fields and nested structures.
            
            Args:
                obj: Object to extract IDs from (dict or list)
                prefix: String prefix for nested field names (default: '')
                
            Returns:
                dict: Mapping of ID field paths to their values, where:
                    - Keys are the full path to the ID field
                    - Values are the ID values found
                    
            Example:
                >>> data = {
                ...     'user': {
                ...         'id': '123',
                ...         'team': {'team_id': 'T456'}
                ...     }
                ... }
                >>> extract_ids(data)
                {'user_id': '123', 'user_team_team_id': 'T456'}
                
            Note:
                - Handles both 'id' and '*_id' field patterns
                - Maintains path context through nesting
                - Processes both objects and arrays
            """
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
        logger.debug(f"Found IDs in context: {all_ids}")
        
        # Create relationships for each ID
        for key, val in all_ids.items():
            if val in id_to_chunk:
                target_info = id_to_chunk[val]
                logger.debug(f"Found target chunk for ID {val}: {target_info['path']}")
                
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
                logger.debug(f"No chunk found with ID value: {val}")
    
    # Update file metadata
    for f, f_hash, f_mtime in documents:
        upsert_file_metadata(conn, f, f_hash, f_mtime)
    
    conn.commit()
    cur.close()
    
    logger.info("Successfully embedded and indexed new chunks")
    return True

def initialize_embeddings(conn):
    """
    Initialize vector embeddings for all JSON files in the data directory.
    
    This function serves as the primary initialization point for the system's
    vector embeddings. It's called either after a database reset or when
    changes are detected in the JSON files.
    
    Process:
        1. Scans the data directory for all JSON files
        2. Processes each file through the embedding pipeline
        3. Stores embeddings and metadata in the database
        
    Args:
        conn: PostgreSQL database connection object
        
    Example:
        >>> with psycopg2.connect(POSTGRES_CONN_STR) as conn:
        ...     initialize_embeddings(conn)
        Initializing embeddings for all JSON files...
        
    Note:
        - This is a potentially long-running operation for large datasets
        - Progress is logged to provide visibility into the process
        - Existing embeddings are preserved and only new/modified files are processed
    """
    print("Initializing embeddings for all JSON files...")
    load_and_embed_new_data(conn)

def chat_loop(conn):
    """
    Run the main interactive chat loop for processing user queries.
    
    This function implements the primary interaction point with users,
    processing their queries and returning relevant answers based on
    the embedded JSON data. It automatically checks for and processes
    new or modified files before each query.
    
    Features:
        - Interactive prompt for user queries
        - Automatic data refresh before each query
        - Error handling for query processing
        - Special command support (:quit)
        
    Args:
        conn: PostgreSQL database connection object
        
    Commands:
        :quit - Exits the chat loop
        
    Example:
        >>> with psycopg2.connect(POSTGRES_CONN_STR) as conn:
        ...     chat_loop(conn)
        Enter your queries below. Type ':quit' to exit.
        You: Tell me about recent shipments
        Assistant: Found 3 recent shipments...
        
    Note:
        - The loop continues until explicitly terminated
        - Each query triggers a check for new data
        - Errors during query processing are caught and reported
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
    """
    Build a structured prompt for the language model with enhanced context formatting.
    
    This function constructs a detailed prompt that combines the user's query,
    relevant context, and query intent information. The prompt is designed to
    guide the language model toward providing accurate and contextual responses.
    
    Prompt Structure:
        1. Base Instructions
        2. Intent Information
        3. Analysis Guidelines
        4. Context Entries
        5. User Query
        6. Response Format Guidelines
        
    Args:
        query: User's input query string
        context: List of relevant context strings
        query_intent: Dictionary containing:
            - primary_intent: Main intent of the query
            - all_intents: List of all detected intents
            
    Returns:
        str: Formatted prompt string ready for the language model
        
    Example:
        >>> context = ["Product ID: 123, Name: Widget", "Stock: 50 units"]
        >>> intent = {
        ...     "primary_intent": "inventory_query",
        ...     "all_intents": ["inventory_query", "product_info"]
        ... }
        >>> prompt = build_prompt("How many widgets do we have?", context, intent)
        
    Note:
        - Guidelines are dynamically adjusted based on query intent
        - Context is formatted for clear separation
        - Human-readable names are emphasized
    """
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

def assemble_context(chunks: List[Dict], max_tokens: int = 3000) -> str:
    """
    Assemble context for the LLM, organizing chunks by their relationships.
    
    Args:
        chunks: List of chunks with relationships
        max_tokens: Maximum tokens to include
        
    Returns:
        Formatted context string
    """
    def format_chunk(chunk: Dict, section: str) -> str:
        content = chunk['content']
        relationships = chunk.get('relationships', [])
        
        # Format the chunk content
        chunk_text = f"\n### {section}\n"
        chunk_text += json.dumps(content, indent=2)
        
        # Add relationship context if available
        if relationships:
            chunk_text += "\nRelationships:"
            for rel in relationships:
                rel_type = rel['type']
                confidence = rel.get('confidence', 0.0)
                if confidence >= 0.6:  # Only include high-confidence relationships
                    chunk_text += f"\n- {rel_type.upper()}: {rel.get('metadata', {}).get('value', 'N/A')}"
        
        return chunk_text
    
    # Group chunks by their role
    primary_chunks = []
    supporting_chunks = []
    context_chunks = []
    
    for chunk in chunks:
        score = chunk.get('score', 0.0)
        if score >= 0.8:
            primary_chunks.append(chunk)
        elif score >= 0.6:
            supporting_chunks.append(chunk)
        else:
            context_chunks.append(chunk)
    
    # Assemble the context parts
    context_parts = []
    token_count = 0
    
    # Add primary information (60% of token budget)
    primary_token_limit = int(max_tokens * 0.6)
    for chunk in primary_chunks:
        if token_count >= primary_token_limit:
            break
        context_parts.append(format_chunk(chunk, "Primary Information"))
        token_count += len(context_parts[-1].split())
    
    # Add supporting information (30% of token budget)
    supporting_token_limit = int(max_tokens * 0.3)
    for chunk in supporting_chunks:
        if token_count >= primary_token_limit + supporting_token_limit:
            break
        context_parts.append(format_chunk(chunk, "Supporting Information"))
        token_count += len(context_parts[-1].split())
    
    # Add contextual information (10% of token budget)
    for chunk in context_chunks:
        if token_count >= max_tokens:
            break
        context_parts.append(format_chunk(chunk, "Additional Context"))
        token_count += len(context_parts[-1].split())
    
    # Combine all parts
    full_context = "\n\n".join(context_parts)
    
    # Add relationship summary at the end
    relationship_summary = "\n### Relationship Summary\n"
    relationship_types = {
        'explicit': "Direct references between entities",
        'semantic': "Contextually related information",
        'temporal': "Time-based relationships"
    }
    
    for rel_type, description in relationship_types.items():
        rel_count = sum(
            1 for chunk in chunks
            for rel in chunk.get('relationships', [])
            if rel['type'] == rel_type and rel.get('confidence', 0) >= 0.6
        )
        if rel_count > 0:
            relationship_summary += f"- Found {rel_count} {rel_type} relationships ({description})\n"
    
    full_context += relationship_summary
    
    return full_context

def run_test_queries(conn) -> None:
    """Run all test queries from the test file."""
    try:
        with open('data/json_docs/test_queries', 'r') as f:
            content = f.read()
            
        # Skip the directions line and parse queries
        queries = []
        for line in content.split('\n'):
            if line.startswith('"') and line.endswith('"'):
                # Remove quotes and any trailing punctuation
                query = line.strip('"\n.,')
                queries.append(query)
        
        logger.info(f"Running {len(queries)} test queries...")
        
        for i, query in enumerate(queries, 1):
            print(f"\nYou: {query}")
            response = answer_query(conn, query)
            print(f"\nAssistant: {response}")
            print("-" * 80)
            
    except FileNotFoundError:
        logger.error("Test queries file not found at data/json_docs/test_queries")
    except Exception as e:
        logger.error(f"Error running test queries: {e}")

def main():
    """Main entry point for the chat application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='JSON RAG System')
    parser.add_argument('--test', action='store_true', help='Run test queries')
    args = parser.parse_args()
    
    try:
        # Initialize database connection
        conn = init_db()
        
        if args.test:
            # Run test queries mode
            run_test_queries(conn)
            return
            
        # Normal interactive mode
        logger.info("Checking for new data to embed...")
        
        # Process any new JSON files
        load_and_embed_new_data(conn)
            
        # Interactive query loop
        while True:
            try:
                query = input("\nYou: ")
                if query.lower() in ['exit', 'quit', 'q']:
                    break
                    
                response = answer_query(conn, query)
                print(f"\nAssistant: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
                print("\nAssistant: I encountered an error. Please try again.")
                
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    main()
