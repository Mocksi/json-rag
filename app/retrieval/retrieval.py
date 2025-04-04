from typing import List, Dict, Optional
import json
from openai import OpenAI

from app.utils.logging_config import get_logger
from app.retrieval.embedding import get_base_chunks
from app.analysis.archetype import ArchetypeDetector

logger = get_logger(__name__)

# Initialize OpenAI client
client = OpenAI()


class QueryPipeline:
    """Unified pipeline for consistent query processing."""

    def __init__(self, store, llm_client):
        self.store = store
        self.llm = llm_client
        self.logger = get_logger(__name__)
        self.retriever = TwoTierRetrieval(store)

    def execute(self, query: str) -> Dict:
        """Execute query through pipeline."""
        try:
            # Step 1: Analysis
            analysis = self.analyze_query(query)

            # Step 2: Retrieval
            results = self.retriever.get_data(query, analysis)

            # Step 3: Generate Response
            response = self.generate_response(query, results, analysis)

            return {"status": "success", "response": response}

        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            return {"status": "error", "error": str(e)}

    def analyze_query(self, query: str) -> Dict:
        """Analyze query to determine needs."""
        messages = [
            {
                "role": "system",
                "content": "You analyze queries into structured execution plans.",
            },
            {
                "role": "user",
                "content": f"""Analyze this query and output a structured plan.
            Query: {query}
            
            Output JSON with:
            {{
                "data_needs": [],     # Required data types (e.g. "documents", "records")
                "operations": [],     # Required operations (e.g. "aggregate", "filter", "list")
                "filters": {{         # Any filters to apply
                    "type": null      # Type filter if needed
                }},
                "temporal_context": {{  # Temporal understanding (for LLM reasoning)
                    "description": null,  # Human description of time context (e.g. "morning of March 18, 2024")
                    "relative_to": null,  # Reference point if needed (e.g. "order_timestamp")
                    "comparison": null    # Type of comparison needed (e.g. "before", "after", "between")
                }},
                "relationships": {{   # Required relationship types
                    "temporal": false,  # Whether temporal order matters
                    "entity": false,    # Whether entity connections matter
                    "sequential": false  # Whether event sequences matter
                }},
                "output_format": "text"
            }}
            """,
            },
        ]

        completion = self.llm.chat.completions.create(
            model="o3-mini-2025-01-31", messages=messages
        )

        return json.loads(completion.choices[0].message.content)

    def generate_response(self, query: str, results: List[Dict], analysis: Dict) -> str:
        """Generate final response from retrieved data."""
        # Format results based on type
        formatted_data = []
        for result in results:
            if result.get("type") == "summary":
                formatted_data.extend(
                    [
                        {
                            "date": group["date"],
                            "count": group["count"],
                            "data": group["chunks"],
                        }
                        for group in result.get("groups", [])
                    ]
                )
            else:
                formatted_data.append(result)

        # Build system prompt with temporal reasoning guidance
        system_prompt = """You are a precise data analyst assistant that:
1. Only uses information explicitly present in the provided data
2. Shows calculations when performing aggregations
3. Responds with "no data available" if information is missing
4. Never makes assumptions or inferences beyond the data
5. Maintains chronological order when relevant
6. Understands and interprets temporal expressions in context
7. Considers timezone and locale when comparing timestamps"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"""Generate a response for this query using the provided data.
            
Query: {query}

Data:
{json.dumps(formatted_data, indent=2)}

Analysis:
{json.dumps(analysis, indent=2)}

Guidelines:
- Use only information from the provided data
- Format according to {analysis.get("output_format", "text")}
- Be clear and concise
- If aggregating values, show the calculation
- For temporal comparisons, use the temporal context from the analysis
- Consider timezone and locale when comparing timestamps
""",
            },
        ]

        completion = self.llm.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.0
        )

        return completion.choices[0].message.content.strip()


class TwoTierRetrieval:
    """Two-tier retrieval system for large JSON datasets."""

    def __init__(self, store):
        self.store = store
        self.logger = get_logger(__name__)
        self.max_chunks = 3  # Limit number of chunks
        self.max_relationships = 3  # Limit number of relationships per chunk
        self.is_postgres = hasattr(store, 'cursor')

    def get_data(self, query: str, analysis: Dict) -> List[Dict]:
        """Multi-step retrieval process with focused context building."""
        try:
            self.logger.debug(
                f"Starting retrieval with analysis: {json.dumps(analysis, indent=2)}"
            )

            # Step 1: Get initial chunks with high relevance
            results = get_base_chunks(
                self.store,
                query,
                filters=analysis.get("filters", {}),
                limit=self.max_chunks,  # Use max_chunks limit
            )

            if not results:
                return []

            # Step 2: Analyze initial results to determine what additional context we need
            needed_context = self._analyze_context_needs(results, analysis)

            # Step 3: Build focused context for each result
            enriched_results = []
            
            if self.is_postgres:
                # PostgreSQL approach with cursor
                with self.store.cursor() as cur:
                    for chunk in results:
                        context = self._build_focused_context(cur, chunk, needed_context)
                        if context:
                            enriched_results.append(context)
            else:
                # ChromaDB approach - simplified context building
                for chunk in results:
                    # For ChromaDB, we don't have the same relationship data structure
                    # So just use the chunk directly with minimal additional context
                    enriched_results.append({
                        "id": chunk["id"],
                        "content": chunk["content"],
                        "metadata": chunk["metadata"],
                        "type": "document",
                        "relationships": [] # No relationships in ChromaDB mode for now
                    })

            return enriched_results

        except Exception as e:
            self.logger.error(f"Retrieval error: {e}", exc_info=True)
            return []

    def _analyze_context_needs(
        self, initial_results: List[Dict], analysis: Dict
    ) -> Dict:
        """Determine what additional context we need based on initial results."""
        needs = {
            "entity_ids": set(),
            "time_range": {"start": None, "end": None},
            "relationship_types": set(),
        }

        for chunk in initial_results:
            # Track unique entities
            entity_id = chunk["metadata"].get("entity_id")
            if entity_id:
                needs["entity_ids"].add(entity_id)

            # Track time range if temporal relationships matter
            if analysis.get("relationships", {}).get("temporal"):
                timestamp = chunk["metadata"].get("timestamp")
                if timestamp:
                    if (
                        not needs["time_range"]["start"]
                        or timestamp < needs["time_range"]["start"]
                    ):
                        needs["time_range"]["start"] = timestamp
                    if (
                        not needs["time_range"]["end"]
                        or timestamp > needs["time_range"]["end"]
                    ):
                        needs["time_range"]["end"] = timestamp

            # Track relationship types
            for rel in chunk.get("relationships", []):
                needs["relationship_types"].add(rel["type"])

        return needs

    def _build_focused_context(
        self, cur, chunk: Dict, needed_context: Dict
    ) -> Optional[Dict]:
        """Build focused context for a chunk based on analysis of what's needed."""
        try:
            # Get archetypes for the chunk
            archetypes = ArchetypeDetector().detect_archetypes(chunk["chunk_json"])

            # Get relationships for the chunk
            relationships = []
            cur.execute(
                """
                SELECT relationship_type, target_chunk, metadata
                FROM chunk_relationships
                WHERE source_chunk = %s
                ORDER BY metadata->>'confidence' DESC
                LIMIT %s
                """,
                (chunk["id"], self.max_relationships),
            )
            for rel in cur.fetchall():
                relationships.append({
                    "type": rel[0],
                    "target": rel[1],
                    "metadata": rel[2],
                })

            # Build where clause for related chunks
            where_clause = "1=1"
            params = []
            if needed_context.get("temporal"):
                where_clause += " AND metadata->>'type' = 'temporal'"
            if needed_context.get("entity"):
                where_clause += " AND metadata->>'type' = 'entity'"

            # Get focused set of related chunks
            cur.execute(
                f"""
                WITH RECURSIVE chunk_tree AS (
                    SELECT 
                        c.id,
                        c.chunk_json,
                        c.metadata,
                        r.relationship_type,
                        1 as tree_depth,
                        ARRAY[c.id] as path
                    FROM json_chunks c
                    LEFT JOIN chunk_relationships r ON r.target_chunk = c.id
                    WHERE {where_clause}
                    AND c.id != %s  -- Exclude self
                    
                    UNION
                    
                    SELECT 
                        c.id,
                        c.chunk_json,
                        c.metadata,
                        r.relationship_type,
                        t.tree_depth + 1,
                        t.path || c.id
                    FROM chunk_tree t
                    JOIN chunk_relationships r ON r.source_chunk = t.id
                    JOIN json_chunks c ON c.id = r.target_chunk
                    WHERE t.tree_depth < 2  -- Reduce depth
                    AND NOT c.id = ANY(t.path)
                )
                SELECT DISTINCT ON (id)
                    chunk_json,
                    metadata->>'type' as chunk_type,
                    relationship_type,
                    tree_depth,
                    metadata
                FROM chunk_tree
                ORDER BY id, tree_depth
                LIMIT %s  -- Limit related chunks
            """,
                params + [chunk["id"], self.max_chunks],
            )

            related = cur.fetchall()

            return {
                "main_chunk": {
                    "content": chunk["chunk_json"],
                    "type": chunk["metadata"].get("type"),
                    "archetypes": archetypes,
                    "relationships": relationships,
                    "metadata": chunk["metadata"],
                },
                "related_chunks": [
                    {
                        "content": r[0],
                        "type": r[1],
                        "relationship": r[2],
                        "depth": r[3],
                        "metadata": r[4],
                    }
                    for r in related
                ],
                "relevance": 1.0 - chunk["distance"],
                "source": chunk["metadata"].get("file_path", ""),
            }

        except Exception as e:
            self.logger.error(f"Error building context: {e}")
            return None


def answer_query(store, query: str) -> str:
    """Answer query using the unified pipeline.
    
    Args:
        store: Database backend (PostgreSQL connection or ChromaDB collection)
        query: The user's search query
        
    Returns:
        Response text from the AI assistant
    """
    pipeline = QueryPipeline(store, client)
    result = pipeline.execute(query)

    if result["status"] == "error":
        logger.error(f"Pipeline error: {result['error']}")
        return "I encountered an error while processing your query. Please try again."

    return result["response"]
