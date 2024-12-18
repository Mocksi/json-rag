"""
JSON Document Processing Module

This module provides functionality for processing JSON documents into chunks while
preserving hierarchical relationships and semantic context. It handles the complete
processing pipeline from file ingestion to relationship detection and storage.

Key Features:
    - Hierarchical Chunking: Preserves document structure and relationships
    - Context-Aware Processing: Maintains parent-child relationships
    - Relationship Detection: Identifies and stores entity relationships
    - Embedding Generation: Creates context-aware vector embeddings
    - Batch Processing: Handles multiple files and directories
    - Caching: Optimizes processing with chunk caching
    - Error Handling: Graceful error recovery and logging

The module is designed to work with complex JSON documents containing:
    - Nested structures
    - Cross-references
    - Entity relationships
    - Metadata
    - Array collections

Usage:
    >>> from app.processor import JSONProcessor
    >>> processor = JSONProcessor(db_connection)
    >>> chunk_ids = processor.process_file("data/document.json")
    >>> print(f"Processed into {len(chunk_ids)} chunks")
"""

from typing import List, Dict, Optional
import json
from pathlib import Path
from app.storage.database import store_chunk, get_chunk_with_context
from app.processing.json_parser import process_json_file, combine_chunk_with_context
from app.retrieval.embedding import get_embedding

class JSONProcessor:
    """
    Process JSON files with hierarchical relationship preservation.
    
    This class handles the complete processing pipeline for JSON documents,
    including chunking, relationship detection, embedding generation, and
    storage. It maintains a cache of processed chunks to optimize relationship
    processing and context generation.
    
    Key Responsibilities:
        - File Processing: Convert JSON files into processable chunks
        - Relationship Detection: Identify and store entity relationships
        - Context Generation: Build hierarchical context for chunks
        - Embedding Creation: Generate context-aware embeddings
        - Cache Management: Optimize processing with chunk caching
        
    Attributes:
        conn: Database connection for storing chunks and relationships
        chunk_cache (dict): Cache of processed chunks for optimization
        
    Example:
        >>> processor = JSONProcessor(db_connection)
        >>> # Process a single file
        >>> chunk_ids = processor.process_file("data/users.json")
        >>> # Process an entire directory
        >>> results = processor.process_directory("data/documents")
        
    Note:
        - The chunk cache is cleared between directory processing
        - Relationships are processed after all chunks are stored
        - Errors during processing are logged and handled gracefully
    """
    
    def __init__(self, conn):
        self.conn = conn
        self.chunk_cache = {}  # Cache of processed chunks
        
    def process_file(self, file_path: str) -> List[str]:
        """
        Process a JSON file and store chunks with hierarchical information.
        
        This method handles the complete processing of a single JSON file,
        including chunking, embedding generation, and relationship detection.
        It uses a two-pass approach to ensure all relationships are properly
        captured.
        
        Process Flow:
            1. First Pass:
                - Generate hierarchical chunks
                - Cache chunks for relationship processing
                - Generate context-aware embeddings
                - Store chunks in database
                
            2. Second Pass:
                - Process relationships between chunks
                - Store relationship information
                - Update chunk metadata
                
        Args:
            file_path (str): Path to the JSON file to process
            
        Returns:
            List[str]: List of stored chunk IDs in processing order
            
        Example:
            >>> processor = JSONProcessor(conn)
            >>> chunk_ids = processor.process_file("data/users.json")
            >>> print(f"Generated {len(chunk_ids)} chunks")
            
        Note:
            - Chunks are cached during processing
            - Relationships are processed after all chunks are stored
            - Parent-child relationships are preserved
            - Embeddings include hierarchical context
        """
        # Process file into hierarchical chunks
        chunks = process_json_file(file_path)
        stored_ids = []
        
        # First pass: Store all chunks
        for chunk in chunks:
            # Cache the chunk data for relationship processing
            self.chunk_cache[chunk['id']] = chunk
            
            # Get parent and child chunks for context
            parent_chunks = []
            if chunk['parent_id'] and chunk['parent_id'] in self.chunk_cache:
                parent_context = self.chunk_cache[chunk['parent_id']]
                if parent_context:
                    parent_chunks = [parent_context]
            
            child_chunks = []  # We'll process relationships in second pass
            
            # Generate embedding with hierarchical context
            context_text = combine_chunk_with_context(chunk, parent_chunks, child_chunks)
            embedding = get_embedding(context_text)
            
            # Store chunk with all information
            chunk_id = store_chunk(self.conn, chunk, embedding)
            stored_ids.append(chunk_id)
        
        # Second pass: Process relationships between existing chunks
        for chunk_id in stored_ids:
            chunk = self.chunk_cache[chunk_id]
            self._process_chunk_relationships(chunk)
            
        return stored_ids
    
    def _process_chunk_relationships(self, chunk: Dict) -> None:
        """
        Process and store relationships for a chunk.
        
        This internal method analyzes a chunk's content to identify and store
        various types of relationships with other chunks. It handles both
        direct references and array-based relationships.
        
        Relationship Types:
            - Parent-Child: Hierarchical document structure
            - Entity References: Direct ID references
            - Array References: Lists of related entities
            - Product Relations: Special handling for related products
            
        Detection Process:
            1. Parent Relationships:
                - Check for parent_id
                - Validate parent exists in cache
                
            2. Entity References:
                - Scan for ID fields
                - Generate target paths
                - Validate targets exist
                
            3. Array References:
                - Process lists of IDs
                - Handle plural field names
                - Create multiple relationships
                
            4. Special Cases:
                - Related products
                - Custom reference types
                
        Args:
            chunk (Dict): Chunk data dictionary containing:
                - id: Unique chunk identifier
                - content: Chunk content with potential references
                - parent_id: Optional parent chunk identifier
                - path: JSON path in the document
                - source_file: Original file path
                
        Note:
            - All relationships are stored in the database
            - Invalid references are logged but not stored
            - Relationship metadata includes reference context
            - Debug logging is enabled for relationship processing
        """
        print(f"\nDEBUG: Processing relationships for chunk: {chunk['id']}")
        print(f"DEBUG: Chunk path: {chunk['path']}")
        print(f"DEBUG: Chunk content: {json.dumps(chunk['content'], indent=2)}")
        
        # Process parent-child relationships
        if chunk['parent_id'] and chunk['parent_id'] in self.chunk_cache:
            print(f"DEBUG: Found parent relationship: {chunk['id']} -> {chunk['parent_id']}")
            self._store_relationship(
                source_id=chunk['id'],
                target_id=chunk['parent_id'],
                rel_type='child_of'
            )
        
        # Process entity relationships from the content
        content = chunk.get('content', {})
        if isinstance(content, dict):
            # Handle references in the content
            for key, value in content.items():
                print(f"DEBUG: Processing field: {key} = {value}")
                
                if isinstance(value, str) and (
                    key.endswith('_id') or 
                    key in {'manufacturer_id', 'supplier_id', 'warranty_policy_id', 'parent_id'}
                ):
                    # Generate target chunk ID using the referenced entity path
                    base_type = key.split('_id')[0]
                    target_path = normalize_json_path(f"data.{base_type}s.{value}")
                    target_id = generate_chunk_id(chunk['source_file'], target_path)
                    
                    print(f"DEBUG: Generated target path for {key}: {target_path}")
                    print(f"DEBUG: Generated target ID: {target_id}")
                    
                    # Store the relationship if the target exists
                    if target_id in self.chunk_cache:
                        print(f"DEBUG: Found target in cache, creating relationship")
                        self._store_relationship(
                            source_id=chunk['id'],
                            target_id=target_id,
                            rel_type=f"references_{key}",
                            metadata={'referenced_id': value}
                        )
                    else:
                        print(f"DEBUG: Target not found in cache: {target_id}")
                
                # Handle array of references
                elif isinstance(value, list) and key.endswith('_ids'):
                    base_type = key[:-4]  # Remove '_ids' suffix
                    print(f"DEBUG: Processing array of references for {base_type}")
                    for ref_id in value:
                        if isinstance(ref_id, str):
                            target_path = normalize_json_path(f"data.{base_type}s.{ref_id}")
                            target_id = generate_chunk_id(chunk['source_file'], target_path)
                            
                            print(f"DEBUG: Generated target path for {ref_id}: {target_path}")
                            print(f"DEBUG: Generated target ID: {target_id}")
                            
                            if target_id in self.chunk_cache:
                                print(f"DEBUG: Found target in cache, creating relationship")
                                self._store_relationship(
                                    source_id=chunk['id'],
                                    target_id=target_id,
                                    rel_type=f"references_{base_type}",
                                    metadata={'referenced_id': ref_id}
                                )
                            else:
                                print(f"DEBUG: Target not found in cache: {target_id}")
                                
                # Handle related products
                elif key == 'related_products' and isinstance(value, list):
                    print(f"DEBUG: Processing related products: {value}")
                    for related_id in value:
                        if isinstance(related_id, str):
                            target_path = normalize_json_path(f"data.products.{related_id}")
                            target_id = generate_chunk_id(chunk['source_file'], target_path)
                            
                            print(f"DEBUG: Generated target path for related product {related_id}: {target_path}")
                            print(f"DEBUG: Generated target ID: {target_id}")
                            
                            if target_id in self.chunk_cache:
                                print(f"DEBUG: Found target in cache, creating relationship")
                                self._store_relationship(
                                    source_id=chunk['id'],
                                    target_id=target_id,
                                    rel_type='related_product',
                                    metadata={'referenced_id': related_id}
                                )
                            else:
                                print(f"DEBUG: Target not found in cache: {target_id}")
    
    def _store_relationship(self, source_id: str, target_id: str, rel_type: str, metadata: Dict = None) -> None:
        """
        Store a relationship between chunks in the database.
        
        This internal method handles the persistent storage of chunk relationships,
        including metadata and timestamps. It uses UPSERT logic to handle
        duplicate relationships gracefully.
        
        Storage Process:
            1. Prepare relationship data and metadata
            2. Attempt database insertion
            3. Handle conflicts with existing relationships
            4. Update timestamps and metadata if needed
            
        Args:
            source_id (str): ID of the source chunk
            target_id (str): ID of the target chunk
            rel_type (str): Type of relationship (e.g., 'child_of', 'references')
            metadata (Dict, optional): Additional relationship metadata. Defaults to None.
                Can include:
                - referenced_id: Original reference ID
                - context: Additional relationship context
                - confidence: Relationship confidence score
                
        Note:
            - Uses database transaction for atomicity
            - Updates existing relationships on conflict
            - Logs errors but continues processing
            - Updates detection timestamp on each store
        """
        print(f"DEBUG: Storing relationship: {source_id} -> {target_id} ({rel_type})")
        if metadata is None:
            metadata = {}
            
        cur = self.conn.cursor()
        try:
            cur.execute("""
                INSERT INTO chunk_relationships 
                    (source_chunk_id, target_chunk_id, relationship_type, metadata)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_chunk_id, target_chunk_id, relationship_type) 
                DO UPDATE SET
                    metadata = EXCLUDED.metadata,
                    detected_at = CURRENT_TIMESTAMP
            """, (source_id, target_id, rel_type, json.dumps(metadata)))
            self.conn.commit()
            print(f"DEBUG: Successfully stored relationship")
        except Exception as e:
            print(f"ERROR storing relationship: {e}")
            self.conn.rollback()
        finally:
            cur.close()
    
    def process_directory(self, directory: str, pattern: str = "*.json") -> Dict[str, List[str]]:
        """
        Process all JSON files in a directory matching a pattern.
        
        This method handles batch processing of multiple JSON files in a directory,
        maintaining a clean processing state for each file. It clears the chunk
        cache before processing to ensure clean state.
        
        Process Flow:
            1. Clear chunk cache for clean state
            2. Discover JSON files matching pattern
            3. Process each file individually
            4. Collect results and handle errors
            
        Args:
            directory (str): Directory path containing JSON files
            pattern (str, optional): Glob pattern for matching files.
                Defaults to "*.json".
                
        Returns:
            Dict[str, List[str]]: Mapping of filenames to their chunk IDs, where:
                - Keys are file paths relative to the directory
                - Values are lists of generated chunk IDs
                
        Example:
            >>> processor = JSONProcessor(conn)
            >>> results = processor.process_directory("data/")
            >>> for file, chunks in results.items():
            ...     print(f"{file}: {len(chunks)} chunks")
            
        Note:
            - Chunk cache is cleared before processing
            - Failed files are logged but don't stop processing
            - Returns partial results if some files fail
            - File paths in results are relative to directory
        """
        results = {}
        directory = Path(directory)
        
        # Clear chunk cache before processing directory
        self.chunk_cache = {}
        
        for json_file in directory.glob(pattern):
            try:
                chunk_ids = self.process_file(str(json_file))
                results[str(json_file)] = chunk_ids
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
                
        return results
    
    def reprocess_chunks(self, chunk_ids: List[str]) -> None:
        """
        Reprocess existing chunks to update their embeddings with full context.
        
        This method allows updating chunk embeddings with complete context after
        all relationships have been established. It's useful for improving
        embedding quality with full hierarchical context.
        
        Process Flow:
            1. Clear chunk cache for clean state
            2. For each chunk:
                - Retrieve full context from database
                - Generate new context-aware embedding
                - Update chunk with new embedding
                
        Args:
            chunk_ids (List[str]): List of chunk IDs to reprocess
            
        Example:
            >>> processor = JSONProcessor(conn)
            >>> # After processing files and establishing relationships
            >>> processor.reprocess_chunks(['chunk1', 'chunk2'])
            
        Note:
            - Chunk cache is cleared before reprocessing
            - Skips chunks that can't be retrieved
            - Updates embeddings with full context
            - Maintains existing relationships
            - Useful for improving embedding quality
        """
        # Clear chunk cache before reprocessing
        self.chunk_cache = {}
        
        for chunk_id in chunk_ids:
            # Get full context for the chunk
            chunk_data = get_chunk_with_context(self.conn, chunk_id)
            if not chunk_data:
                continue
                
            # Cache the chunk data
            self.chunk_cache[chunk_id] = chunk_data
            
            # Generate new embedding with full context
            context_text = combine_chunk_with_context(
                chunk_data,
                chunk_data.get('parents', []),
                chunk_data.get('children', [])
            )
            embedding = get_embedding(context_text)
            
            # Update the chunk with new embedding
            chunk_data['embedding'] = embedding
            store_chunk(self.conn, chunk_data, embedding) 