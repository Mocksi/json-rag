"""
JSON Document Processing Module

This module provides functionality for processing JSON documents into chunks while
preserving hierarchical relationships and semantic context.
"""

from typing import List, Dict, Optional
import json
from pathlib import Path
from app.storage.database import store_chunk, get_chunk_with_context
from app.processing.json_parser import process_json_file, combine_chunk_with_context
from app.retrieval.embedding import get_embedding
from app.analysis.relationships import process_relationships
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

class JSONProcessor:
    def __init__(self, conn):
        self.conn = conn
        self.chunk_cache = {}
    
    def process_file(self, file_path: str) -> List[str]:
        """
        Process a JSON file into chunks with relationships.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            List of stored chunk IDs
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
            
            # Generate embedding with hierarchical context
            context_text = combine_chunk_with_context(chunk, parent_chunks, [])
            embedding = get_embedding(context_text)
            
            # Store chunk with all information
            chunk_id = store_chunk(self.conn, chunk, embedding)
            stored_ids.append(chunk_id)
        
        # Second pass: Process relationships
        logger.info("Processing relationships...")
        relationships = process_relationships(list(self.chunk_cache.values()))
        
        # Store relationships in database
        cur = self.conn.cursor()
        for rel in relationships:
            cur.execute("""
                INSERT INTO chunk_relationships (
                    source_chunk_id,
                    target_chunk_id,
                    relationship_type,
                    confidence,
                    metadata
                ) VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (source_chunk_id, target_chunk_id, relationship_type) 
                DO UPDATE SET
                    confidence = EXCLUDED.confidence,
                    metadata = EXCLUDED.metadata
            """, (
                rel['source_chunk_id'],
                rel['target_chunk_id'],
                rel['relationship_type'],
                rel['confidence'],
                json.dumps(rel['metadata'])
            ))
        
        self.conn.commit()
        cur.close()
        
        logger.info(f"Processed {len(chunks)} chunks with {len(relationships)} relationships")
        return stored_ids
    
    def reprocess_chunks(self, chunk_ids: List[str]) -> None:
        """
        Reprocess a set of chunks to update embeddings and relationships.
        
        Args:
            chunk_ids: List of chunk IDs to reprocess
        """
        # Clear chunk cache before reprocessing
        self.chunk_cache = {}
        
        # First pass: Get all chunks and update embeddings
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
        
        # Second pass: Reprocess relationships
        if self.chunk_cache:
            relationships = process_relationships(list(self.chunk_cache.values()))
            
            # Update relationships in database
            cur = self.conn.cursor()
            for rel in relationships:
                cur.execute("""
                    INSERT INTO chunk_relationships (
                        source_chunk_id,
                        target_chunk_id,
                        relationship_type,
                        confidence,
                        metadata
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (source_chunk_id, target_chunk_id, relationship_type) 
                    DO UPDATE SET
                        confidence = EXCLUDED.confidence,
                        metadata = EXCLUDED.metadata
                """, (
                    rel['source_chunk_id'],
                    rel['target_chunk_id'],
                    rel['relationship_type'],
                    rel['confidence'],
                    json.dumps(rel['metadata'])
                ))
            
            self.conn.commit()
            cur.close()
            
            logger.info(f"Reprocessed {len(chunk_ids)} chunks with {len(relationships)} relationships")