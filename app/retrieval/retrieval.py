from typing import List, Dict, Optional, Tuple, Union
import json
from collections import defaultdict
import statistics
from datetime import timedelta
from openai import OpenAI
from time import sleep
from datetime import datetime

from app.core.config import embedding_model
from app.utils.logging_config import get_logger
from app.retrieval.embedding import get_embedding, get_base_chunks
from app.storage.database import get_chunk_relationships, get_chunk_archetypes
from app.analysis.archetype import ArchetypeDetector

logger = get_logger(__name__)

# Initialize OpenAI client
client = OpenAI()

class QueryPipeline:
    """Unified pipeline for consistent query processing."""
    
    def __init__(self, conn, llm_client):
        self.conn = conn
        self.llm = llm_client
        self.logger = get_logger(__name__)
        self.retriever = TwoTierRetrieval(conn)

    def execute(self, query: str) -> Dict:
        """Execute query through pipeline."""
        try:
            # Step 1: Analysis
            analysis = self.analyze_query(query)
            
            # Step 2: Retrieval
            results = self.retriever.get_data(query, analysis)
            
            # Step 3: Generate Response
            response = self.generate_response(query, results, analysis)
            
            return {
                "status": "success",
                "response": response
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def analyze_query(self, query: str) -> Dict:
        """Analyze query to determine needs."""
        messages = [
            {"role": "system", "content": "You analyze queries into structured execution plans."},
            {"role": "user", "content": f"""Analyze this query and output a structured plan.
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
            """}
        ]
        
        completion = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0
        )
        
        return json.loads(completion.choices[0].message.content)

    def generate_response(self, query: str, results: List[Dict], analysis: Dict) -> str:
        """Generate final response from retrieved data."""
        # Format results based on type
        formatted_data = []
        for result in results:
            if result.get("type") == "summary":
                formatted_data.extend([{
                    "date": group["date"],
                    "count": group["count"],
                    "data": group["chunks"]
                } for group in result.get("groups", [])])
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
            {"role": "user", "content": f"""Generate a response for this query using the provided data.
            
Query: {query}

Data:
{json.dumps(formatted_data, indent=2)}

Analysis:
{json.dumps(analysis, indent=2)}

Guidelines:
- Use only information from the provided data
- Format according to {analysis.get('output_format', 'text')}
- Be clear and concise
- If aggregating values, show the calculation
- For temporal comparisons, use the temporal context from the analysis
- Consider timezone and locale when comparing timestamps
"""}
        ]
        
        completion = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.0
        )
        
        return completion.choices[0].message.content.strip()

class TwoTierRetrieval:
    """Two-tier retrieval system for large JSON datasets."""
    
    def __init__(self, conn):
        self.conn = conn
        self.logger = get_logger(__name__)

    def get_data(self, query: str, analysis: Dict) -> List[Dict]:
        """Multi-step retrieval process with focused context building."""
        try:
            self.logger.debug(f"Starting retrieval with analysis: {json.dumps(analysis, indent=2)}")
            
            # Step 1: Get initial chunks with high relevance
            results = get_base_chunks(
                self.conn,
                query,
                filters=analysis.get("filters", {}),
                limit=5  # Start with fewer chunks
            )
            
            if not results:
                return []

            # Step 2: Analyze initial results to determine what additional context we need
            needed_context = self._analyze_context_needs(results, analysis)
            
            # Step 3: Build focused context for each result
            enriched_results = []
            with self.conn.cursor() as cur:
                for chunk in results:
                    context = self._build_focused_context(cur, chunk, needed_context)
                    if context:
                        enriched_results.append(context)
            
            return enriched_results
            
        except Exception as e:
            self.logger.error(f"Retrieval error: {e}", exc_info=True)
            return []

    def _analyze_context_needs(self, initial_results: List[Dict], analysis: Dict) -> Dict:
        """Determine what additional context we need based on initial results."""
        needs = {
            "entity_ids": set(),
            "time_range": {"start": None, "end": None},
            "relationship_types": set()
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
                    if not needs["time_range"]["start"] or timestamp < needs["time_range"]["start"]:
                        needs["time_range"]["start"] = timestamp
                    if not needs["time_range"]["end"] or timestamp > needs["time_range"]["end"]:
                        needs["time_range"]["end"] = timestamp
            
            # Track relationship types
            for rel in chunk.get("relationships", []):
                needs["relationship_types"].add(rel["type"])
        
        return needs

    def _build_focused_context(self, cur, chunk: Dict, needed_context: Dict) -> Optional[Dict]:
        """Build focused context for a chunk based on analysis of what's needed."""
        try:
            # Initialize archetype detector
            detector = ArchetypeDetector()
            
            # Get archetypes and relationships
            archetypes = detector.detect_archetypes(chunk["chunk_json"])
            relationships = detector.detect_relationships(chunk["chunk_json"])
            
            # Build focused query conditions
            conditions = []
            params = []
            
            # Only get related chunks that share relevant entities
            if chunk["metadata"].get("entity_id") in needed_context["entity_ids"]:
                conditions.append("c.metadata->>'entity_id' = %s")
                params.append(chunk["metadata"]["entity_id"])
            
            # Only get chunks within relevant time range if needed
            if needed_context["time_range"]["start"]:
                conditions.append("c.metadata->>'timestamp' >= %s")
                params.append(needed_context["time_range"]["start"])
            if needed_context["time_range"]["end"]:
                conditions.append("c.metadata->>'timestamp' <= %s")
                params.append(needed_context["time_range"]["end"])
            
            # Build WHERE clause
            where_clause = " AND ".join(conditions) if conditions else "TRUE"
            
            # Get focused set of related chunks
            cur.execute(f"""
                WITH RECURSIVE chunk_tree AS (
                    SELECT 
                        c.id,
                        c.chunk_json,
                        c.metadata,
                        r.relationship_type,
                        1 as tree_depth,
                        ARRAY[c.id] as path
                    FROM json_chunks c
                    LEFT JOIN chunk_relationships r ON r.target_chunk_id = c.id
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
                    JOIN chunk_relationships r ON r.source_chunk_id = t.id
                    JOIN json_chunks c ON c.id = r.target_chunk_id
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
                LIMIT 5  -- Reduce related chunks
            """, params + [chunk["id"]])
            
            related = cur.fetchall()
            
            return {
                "main_chunk": {
                    "content": chunk["chunk_json"],
                    "type": chunk["metadata"].get("type"),
                    "archetypes": archetypes,
                    "relationships": relationships,
                    "metadata": chunk["metadata"]
                },
                "related_chunks": [
                    {
                        "content": r[0],
                        "type": r[1],
                        "relationship": r[2],
                        "depth": r[3],
                        "metadata": r[4]
                    }
                    for r in related
                ],
                "relevance": 1.0 - chunk["distance"],
                "source": chunk["source_file"]
            }
            
        except Exception as e:
            self.logger.error(f"Error building context: {e}")
            return None

def answer_query(conn, query: str) -> str:
    """Answer query using the unified pipeline."""
    pipeline = QueryPipeline(conn, client)
    result = pipeline.execute(query)
    
    if result["status"] == "error":
        logger.error(f"Pipeline error: {result['error']}")
        return "I encountered an error while processing your query. Please try again."
        
    return result["response"]
