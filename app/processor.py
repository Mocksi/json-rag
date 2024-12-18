from typing import List, Dict, Optional
import json
from pathlib import Path
from .database import store_chunk, get_chunk_with_context
from .json_parser import process_json_file, combine_chunk_with_context
from .embeddings import get_embedding

class JSONProcessor:
    """
    Process JSON files with hierarchical relationship preservation.
    """
    
    def __init__(self, conn):
        self.conn = conn
        self.chunk_cache = {}  # Cache of processed chunks
        
    def process_file(self, file_path: str) -> List[str]:
        """
        Process a JSON file and store chunks with hierarchical information.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            list: List of stored chunk IDs
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
        
        Args:
            chunk: Chunk data dictionary
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
        Store a relationship between chunks.
        
        Args:
            source_id: Source chunk ID
            target_id: Target chunk ID
            rel_type: Type of relationship
            metadata: Optional relationship metadata
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
        Process all JSON files in a directory.
        
        Args:
            directory: Directory path to process
            pattern: Glob pattern for JSON files
            
        Returns:
            dict: Mapping of filenames to their chunk IDs
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
        
        Args:
            chunk_ids: List of chunk IDs to reprocess
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