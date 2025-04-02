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

from typing import List, Dict
import json
import argparse
from tqdm import tqdm
import logging
import os

from app.processing.json_parser import json_to_path_chunks, generate_chunk_id
from app.retrieval.embedding import get_embedding
from app.analysis.archetype import ArchetypeDetector
from app.storage.database import (
    get_files_to_process,
    upsert_file_metadata,
    init_db,
)
from app.utils.utils import get_json_files, compute_file_hash
from app.retrieval.retrieval import answer_query
from app.core.config import embedding_model
from app.utils.logging_config import get_logger
from app.processing.json_processor import process_json_files
from app.query_intent import analyze_query_intent
from app.llm_integration import get_llm_response
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
POSTGRES_CONN_STR = os.getenv("POSTGRES_CONN_STR")
JSON_DATA_DIR = os.getenv("JSON_DATA_DIR", "data/json_docs")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    if hasattr(obj, "tolist"):  # Handle numpy arrays
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: serialize_for_debug(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_debug(x) for x in obj]
    return obj


def load_and_embed_new_data(store):
    """
    Load, chunk, embed, and store data from JSON files in the JSON_DATA_DIR.

    This function uses `process_json_files` to handle the processing of all
    JSON files found in the specified directory. It leverages streaming for
    large files and handles embedding and storage (PostgreSQL or ChromaDB).
    For PostgreSQL, it subsequently performs relationship detection based on
    the processed chunks.

    Args:
        store: Database connection (PostgreSQL) or collection (ChromaDB)

    Returns:
        bool: True if processing completed successfully

    Note:
        - This function processes *all* JSON files in the directory on each run.
        - Relationship detection is currently only implemented for PostgreSQL.
    """
    print("\nProcessing JSON files for embedding...")
    logger.info("Starting data loading and embedding process.")

    # Determine if we're using PostgreSQL or ChromaDB
    is_postgres = hasattr(store, 'cursor')

    # Use process_json_files to handle scanning, chunking, embedding, and storage
    json_data_dir = os.getenv("JSON_DATA_DIR", "data/json_docs")
    try:
        # process_json_files returns processed chunks needed for relationship detection
        processed_chunks = process_json_files(json_data_dir, store)
    except Exception as e:
        logger.error(f"Error during process_json_files: {e}", exc_info=True)
        return False

    if not processed_chunks:
        logger.info("No chunks were processed or returned.")
        # Depending on whether process_json_files returns empty on success or failure,
        # we might want to return True here if it means no *new* data was found.
        # Assuming for now it means nothing was done or an error occurred upstream.
        return True # Let's assume empty list means no new data processed successfully.

    logger.info(f"Successfully processed {len(processed_chunks)} chunks via process_json_files.")

    # Third pass (Relationship Detection - PostgreSQL only)
    # Adapt this section to use the structure of `processed_chunks`
    if is_postgres:
        logger.info("Starting Relationship Detection (PostgreSQL)...")
        cur = None
        try:
            cur = store.cursor()

            # Build a map of all potential ID values to their chunk details
            # Assumes processed_chunks is a list of dicts like:
            # {'id': chunk_id, 'content': original_chunk_dict, 'metadata': {'path': '...', 'source_file': '...'}}
            id_to_chunk_map = {}
            for chunk_data in processed_chunks:
                chunk_content = chunk_data.get('content', {})
                chunk_metadata = chunk_data.get('metadata', {})
                chunk_id = chunk_data.get('id')
                chunk_path = chunk_metadata.get('path')

                # Check if the chunk's value looks like an ID field
                # This logic might need refinement based on how 'value' is structured in chunks
                value = chunk_content.get('value')
                if isinstance(value, str) and value: # Check if value is a non-empty string
                    # Check if the path suggests this is an ID field
                    path_parts = chunk_path.split('.') if chunk_path else []
                    if any(part.endswith("id") or part == "id" for part in path_parts):
                         # Map the ID value (value) to the chunk's details
                        id_to_chunk_map[value] = {
                            "chunk_id": chunk_id,
                            "path": chunk_path,
                            # Keep the original chunk content if needed, but minimize memory?
                            # "chunk": chunk_content
                        }

            logger.info(f"Built ID map with {len(id_to_chunk_map)} potential reference targets.")

            # Now process relationships using the ID map
            rels_created = 0
            for chunk_data in tqdm(processed_chunks, desc="Detecting Relationships"):
                source_chunk_id = chunk_data.get('id')
                chunk_content = chunk_data.get('content', {})
                chunk_metadata = chunk_data.get('metadata', {})
                source_chunk_path = chunk_metadata.get('path')

                # Extract potential relationship IDs from chunk context
                context = chunk_content.get("context", {})

                # --- Re-integrate extract_ids helper function ---
                def extract_ids(obj, prefix=""):
                    ids = {}
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            # Check if the value is a string and the key looks like an ID
                            if isinstance(v, str) and v and (k.endswith("_id") or k == "id"):
                                ids[prefix + k if prefix else k] = v
                            # Recurse into nested dicts or lists
                            elif isinstance(v, (dict, list)):
                                ids.update(
                                    extract_ids(v, prefix + k + "_" if prefix else k + "_")
                                )
                    elif isinstance(obj, list):
                        for i, item in enumerate(obj):
                             # Only recurse into dicts within lists for structure
                            if isinstance(item, dict):
                                # Pass prefix as is, list index isn't usually part of the semantic key path
                                ids.update(extract_ids(item, prefix))
                            # Could potentially handle lists of simple IDs here if needed
                    return ids
                # --- End of extract_ids ---

                # Extract all potential IDs from the chunk's context
                potential_target_ids = extract_ids(context)
                # Also consider the chunk's own value if it represents an ID link itself
                # Example: a simple string chunk that is a foreign key
                value = chunk_content.get('value')
                path = chunk_metadata.get('path', '')
                if isinstance(value, str) and value and (path.endswith("_id") or path.endswith(".id")):
                     potential_target_ids[path.split('.')[-1]] = value # Use field name as key

                # logger.debug(f"Chunk {source_chunk_id}: Found potential target IDs in context/value: {potential_target_ids}")

                # Create relationships for each found ID that maps to a known chunk
                for field_key, target_id_value in potential_target_ids.items():
                    if target_id_value in id_to_chunk_map:
                        target_info = id_to_chunk_map[target_id_value]
                        target_chunk_id = target_info["chunk_id"]
                        target_chunk_path = target_info["path"]

                        # Avoid self-references unless meaningful
                        if source_chunk_id == target_chunk_id:
                            continue

                        # logger.debug(f"Found target chunk for ID {target_id_value}: {target_chunk_path} (Chunk ID: {target_chunk_id})")

                        # Determine relationship type based on key (simple example)
                        # This can be made more sophisticated
                        rel_type = "reference"
                        if field_key.startswith("shipments"): rel_type = "shipment_reference"
                        elif field_key.startswith("warehouses"): rel_type = "warehouse_reference"
                        elif field_key.startswith("products"): rel_type = "product_reference"
                        elif field_key.startswith("suppliers"): rel_type = "supplier_reference"
                        elif field_key.endswith("_id"): rel_type = f"{field_key[:-3]}_reference"
                        elif field_key == "id": rel_type = "identity_reference" # Or primary key reference


                        rel_metadata = {
                            "field": field_key,          # The key/field in the source chunk holding the ID
                            "source_path": source_chunk_path,
                            "target_path": target_chunk_path,
                            "target_value": target_id_value # The actual ID value that linked them
                        }

                        # Insert relationship into the database
                        cur.execute(
                            """
                            INSERT INTO chunk_relationships
                            (source_chunk, target_chunk, relationship_type, metadata)
                            VALUES (%s, %s, %s, %s::jsonb)
                            ON CONFLICT (source_chunk, target_chunk, relationship_type)
                            DO UPDATE SET metadata = EXCLUDED.metadata
                        """,
                            (
                                source_chunk_id,
                                target_chunk_id,
                                rel_type,
                                json.dumps(rel_metadata),
                            ),
                        )
                        rels_created += 1
                    # else:
                        # logger.debug(f"No chunk found with ID value: {target_id_value} (from field {field_key})")

            logger.info(f"Finished Relationship Detection. Created/Updated {rels_created} relationships.")
            store.commit()

        except (Exception, psycopg2.DatabaseError) as error:
            logger.error(f"Error during PostgreSQL relationship processing: {error}", exc_info=True)
            if store:
                store.rollback() # Rollback on error
            return False # Indicate failure
        finally:
            if cur:
                cur.close()

    # Removed file metadata update logic - process_json_files might handle this internally or it's deferred.

    logger.info("Successfully finished loading and embedding process.")
    return True


def initialize_embeddings(store):
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
        store: Database backend (PostgreSQL connection or ChromaDB collection)

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
    load_and_embed_new_data(store)


def chat_loop(store):
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
        store: Database backend (PostgreSQL connection or ChromaDB collection)

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

        load_and_embed_new_data(store)

        try:
            answer = answer_query(store, user_input)
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


def assemble_context(chunks: List[Dict], max_tokens: int = 8000) -> str:
    """
    Assemble context for the LLM, organizing chunks by their relationships.

    Args:
        chunks: List of chunks with relationships
        max_tokens: Maximum tokens to include (default reduced to 8000 to leave room for system prompt and query)

    Returns:
        Formatted context string
    """

    def format_chunk(chunk: Dict, section: str) -> str:
        content = chunk["content"]
        relationships = chunk.get("relationships", [])

        # Format the chunk content
        chunk_text = f"\n### {section}\n"

        # Limit the size of content to prevent token overflow
        if isinstance(content, dict):
            # Keep only first-level key-value pairs for large objects
            if len(json.dumps(content)) > 500:  # Reduced from 1000
                simplified = {}
                for k, v in content.items():
                    if isinstance(v, (dict, list)) and len(json.dumps(v)) > 200:  # Reduced from 300
                        if isinstance(v, dict):
                            simplified[k] = {key: "..." for key in list(v.keys())[:3]}  # Reduced from 5
                            if len(v) > 3:
                                simplified[k]["..."] = f"({len(v) - 3} more items)"
                        else:  # list
                            simplified[k] = ["..."] if len(v) > 0 else []
                            simplified[k].append(f"({len(v)} items total)")
                    else:
                        simplified[k] = v
                content = simplified

        chunk_text += json.dumps(content, indent=2)

        # Add relationship context if available
        if relationships:
            chunk_text += "\nRelationships:"
            for rel in relationships:
                rel_type = rel["type"]
                confidence = rel.get("confidence", 0.0)
                if confidence >= 0.7:  # Increased from 0.6 to be more selective
                    chunk_text += f"\n- {rel_type.upper()}: {rel.get('metadata', {}).get('value', 'N/A')}"

        return chunk_text

    # Group chunks by their role
    primary_chunks = []
    supporting_chunks = []
    context_chunks = []

    for chunk in chunks:
        score = chunk.get("score", 0.0)
        if score >= 0.8:
            primary_chunks.append(chunk)
        elif score >= 0.6:
            supporting_chunks.append(chunk)
        else:
            context_chunks.append(chunk)

    # Assemble the context parts
    context_parts = []
    token_count = 0

    # Add primary information (70% of token budget)
    primary_token_limit = int(max_tokens * 0.7)
    for chunk in primary_chunks:
        if token_count >= primary_token_limit:
            break
        context_parts.append(format_chunk(chunk, "Primary Information"))
        token_count += len(context_parts[-1].split())

    # Add supporting information (20% of token budget)
    supporting_token_limit = int(max_tokens * 0.2)
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
        "explicit": "Direct references between entities",
        "semantic": "Contextually related information",
        "temporal": "Time-based relationships",
    }

    for rel_type, description in relationship_types.items():
        rel_count = sum(
            1
            for chunk in chunks
            for rel in chunk.get("relationships", [])
            if rel["type"] == rel_type and rel.get("confidence", 0) >= 0.7  # Increased from 0.6
        )
        if rel_count > 0:
            relationship_summary += (
                f"- Found {rel_count} {rel_type} relationships ({description})\n"
            )

    full_context += relationship_summary

    return full_context


def run_test_queries(conn) -> None:
    """Run all test queries from the test file."""
    try:
        with open("data/json_docs/test_queries", "r") as f:
            content = f.read()

        # Skip the directions line and parse queries
        queries = []
        for line in content.split("\n"):
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


def get_relevant_chunks(store, query, query_intent=None, max_chunks=10):
    """
    Get the most relevant chunks for a query, using advanced retrieval when appropriate.
    
    Args:
        store: Database backend (PostgreSQL or ChromaDB)
        query: User query string
        query_intent: Optional intent analysis results
        max_chunks: Maximum number of chunks to return
        
    Returns:
        List of relevant chunks with metadata
    """
    # Check if the query appears to be about advertising metrics
    use_advanced_retrieval = False
    
    # Check query for ad metrics related terms
    ad_metrics_terms = ['ad', 'ads', 'advertising', 'campaign', 'impression', 'click', 'conversion', 
                         'ctr', 'cpm', 'cpc', 'adMetrics', 'metrics']
    
    query_lower = query.lower()
    if any(term in query_lower for term in ad_metrics_terms):
        use_advanced_retrieval = True
        logger.info("Query appears to be about ad metrics, using advanced retrieval")
    
    # Check intent if available
    if query_intent and isinstance(query_intent, dict):
        entities = query_intent.get('entities', [])
        topics = query_intent.get('topics', [])
        
        # Check for ad-related entities or topics
        for item in entities + topics:
            if isinstance(item, str) and any(term in item.lower() for term in ad_metrics_terms):
                use_advanced_retrieval = True
                break
    
    # Use advanced retrieval if available and appropriate
    if use_advanced_retrieval:
        try:
            from app.retrieval.advanced_retrieval import retrieve_from_hierarchical_chunks
            chunks = retrieve_from_hierarchical_chunks(store, query, top_k=max_chunks)
            
            if chunks:
                logger.info(f"Retrieved {len(chunks)} chunks using advanced retrieval")
                return chunks
            else:
                logger.info("Advanced retrieval returned no results, falling back to standard")
                # Fall through to standard retrieval
        except ImportError:
            logger.info("Advanced retrieval module not available, using standard retrieval")
            # Fall through to standard retrieval
    
    # Use standard retrieval
    is_postgres = hasattr(store, 'cursor')
    
    if is_postgres:
        # PostgreSQL retrieval
        from app.retrieval.postgres import get_relevant_chunks as postgres_retrieval
        return postgres_retrieval(store, query, max_chunks)
    else:
        # ChromaDB retrieval
        from app.retrieval.chroma import search_chunks
        return search_chunks(store, query, max_chunks)


def answer_query(store, query: str) -> str:
    """
    Process a user query and return an answer based on the JSON embeddings.
    
    This function:
    1. Analyzes the query intent
    2. Retrieves relevant chunks using vector search
    3. Builds a prompt with context for the LLM
    4. Gets response from LLM and returns it
    
    Args:
        store: Database connection
        query: User's question
        
    Returns:
        str: Response to the query based on relevant JSON data
    """
    # First, analyze intent to better understand what the user is asking
    intent = analyze_query_intent(query)
    logger.debug(f"Query intent: {intent}")
    
    # Retrieve relevant chunks, using our function that selects
    # between standard and advanced retrieval as appropriate
    chunks = get_relevant_chunks(store, query, intent)
    
    if not chunks:
        return "I couldn't find any relevant information in the database. Could you rephrase your question?"
    
    # Build prompt with context from chunks
    context = assemble_context(chunks)
    prompt = build_prompt(query, context, intent)
    
    # Get response from LLM
    answer = get_llm_response(prompt)
    
    return answer


def main():
    """Main entry point for the chat application."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="JSON RAG System")
    parser.add_argument("--test", action="store_true", help="Run test queries")
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
                if query.lower() in ["exit", "quit", "q"]:
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
        if "conn" in locals():
            conn.close()


if __name__ == "__main__":
    main()
