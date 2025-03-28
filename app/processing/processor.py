"""
JSON Document Processing Module

This module provides functionality for processing JSON documents into chunks while
preserving hierarchical relationships and semantic context.
"""

from typing import List, Dict
import json
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
            self.chunk_cache[chunk["id"]] = chunk

            # Get parent and child chunks for context
            parent_chunks = []
            if chunk["parent_id"] and chunk["parent_id"] in self.chunk_cache:
                parent_context = self.chunk_cache[chunk["parent_id"]]
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
            cur.execute(
                """
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
            """,
                (
                    rel["source_chunk_id"],
                    rel["target_chunk_id"],
                    rel["relationship_type"],
                    rel["confidence"],
                    json.dumps(rel["metadata"]),
                ),
            )

        self.conn.commit()
        cur.close()

        logger.info(
            f"Processed {len(chunks)} chunks with {len(relationships)} relationships"
        )
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
                chunk_data.get("parents", []),
                chunk_data.get("children", []),
            )
            embedding = get_embedding(context_text)

            # Update the chunk with new embedding
            chunk_data["embedding"] = embedding
            store_chunk(self.conn, chunk_data, embedding)

        # Second pass: Reprocess relationships
        if self.chunk_cache:
            relationships = process_relationships(list(self.chunk_cache.values()))

            # Update relationships in database
            cur = self.conn.cursor()
            for rel in relationships:
                cur.execute(
                    """
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
                """,
                    (
                        rel["source_chunk_id"],
                        rel["target_chunk_id"],
                        rel["relationship_type"],
                        rel["confidence"],
                        json.dumps(rel["metadata"]),
                    ),
                )

            self.conn.commit()
            cur.close()

            logger.info(
                f"Reprocessed {len(chunk_ids)} chunks with {len(relationships)} relationships"
            )


def process_json_files(conn, file_paths: List[str]):
    """Process JSON files with grouping."""
    for path in file_paths:
        with open(path) as f:
            json.load(f)

        # Process chunks with group info
        chunks = process_json_file(path)
        for chunk in chunks:
            # Extract timestamp if available
            timestamp = None
            if "timestamp" in chunk:
                timestamp = chunk["timestamp"]
            elif "date" in chunk:
                timestamp = chunk["date"]

            # Determine group type and key
            group_type = determine_group_type(chunk)
            group_key = extract_group_key(chunk, group_type)

            # Store chunk with group info
            chunk_id = store_chunk(
                conn,
                chunk,
                timestamp=timestamp,
                group_key=group_key,
                group_type=group_type,
            )

            # Update summaries
            update_group_summary(conn, chunk, chunk_id, group_key, group_type)


def determine_group_type(chunk: Dict) -> str:
    """Determine appropriate grouping for chunk."""
    if "timestamp" in chunk or "date" in chunk:
        return "date"
    if "type" in chunk:
        return "type"
    if "category" in chunk:
        return "category"
    return "default"


def extract_group_key(chunk: Dict, group_type: str) -> str:
    """Extract group key based on type."""
    if group_type == "date":
        timestamp = chunk.get("timestamp") or chunk.get("date")
        if timestamp:
            return timestamp.split("T")[0]  # Get date part
    return chunk.get(group_type, "unknown")


def update_group_summary(
    conn, chunk: Dict, chunk_id: str, group_key: str, group_type: str
):
    """Update summary for a group."""
    with conn.cursor() as cur:
        # Get existing summary
        cur.execute(
            """
        SELECT id, count, total_value, chunk_ids
        FROM json_summaries
        WHERE group_key = %s AND group_type = %s
        """,
            (group_key, group_type),
        )

        result = cur.fetchone()

        if result:
            # Update existing summary
            summary_id, count, total, chunk_ids = result
            cur.execute(
                """
            UPDATE json_summaries
            SET count = count + 1,
                total_value = total_value + %s,
                chunk_ids = array_append(chunk_ids, %s),
                last_updated = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
                (chunk.get("value", 0), chunk_id, summary_id),
            )
        else:
            # Create new summary
            cur.execute(
                """
            INSERT INTO json_summaries 
                (group_key, group_type, count, total_value, chunk_ids)
            VALUES (%s, %s, 1, %s, ARRAY[%s])
            """,
                (group_key, group_type, chunk.get("value", 0), chunk_id),
            )

    conn.commit()
