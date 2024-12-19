from typing import List, Dict, Optional, Tuple, Union
import json
from collections import defaultdict
import statistics
from datetime import timedelta
from openai import OpenAI

from app.core.config import embedding_model
from app.utils.logging_config import get_logger
from app.retrieval.embedding import get_embedding, get_base_chunks
from app.storage.database import get_chunk_relationships, get_chunk_archetypes

logger = get_logger(__name__)

# Initialize OpenAI client
client = OpenAI()

class QueryPipeline:
    """Unified pipeline for consistent query processing."""
    
    def __init__(self, conn, llm_client):
        self.conn = conn
        self.llm = llm_client
        self.logger = get_logger(__name__)

    def execute(self, query: str) -> Dict:
        """Execute query through standardized pipeline."""
        self.logger.debug(f"Processing query: {query}")
        self.query = query
        
        try:
            # Step 1: Analysis
            analysis = self.analyze_query(query)
            self.logger.debug(f"Query analysis: {json.dumps(analysis, indent=2)}")
            
            # Step 2: Planning
            plan = self.create_plan(analysis)
            self.logger.debug(f"Execution plan: {json.dumps(plan, indent=2)}")
            
            # Step 3: Retrieval
            retrieved_data = self.execute_retrieval(plan)
            self.logger.debug(f"Retrieved {len(retrieved_data)} results")
            
            # Step 4: Processing
            processed_data = self.execute_processing(retrieved_data, plan)
            self.logger.debug(f"Processed {len(processed_data)} results")
            
            # Step 5: Response Generation
            response = self.generate_response(query, processed_data, analysis)
            
            return {
                "status": "success",
                "response": response,
                "metadata": {
                    "result_count": len(processed_data)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }

    def analyze_query(self, query: str) -> Dict:
        """Analyze query to determine needs and operations."""
        prompt = f"""Analyze this query and output a structured plan.
        Query: {query}
        
        Output JSON with:
        {{
            "data_needs": [],     # Required data types (e.g. "orders", "products")
            "operations": [],     # Required operations (e.g. "filter", "aggregate")
            "filters": {{}},      # Any filters to apply (e.g. {{"date": "03/18/2024"}})
            "relationships": [],  # Required relationships (e.g. ["orders.products"])
            "output_format": "text"   # How to format the response
        }}
        """
        
        try:
            completion = self.llm.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You analyze queries into structured execution plans."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return json.loads(completion.choices[0].message.content)
            
        except Exception as e:
            self.logger.error(f"Analysis error: {e}")
            return {
                "data_needs": [],
                "operations": [],
                "filters": {},
                "relationships": [],
                "output_format": "text"
            }

    def create_plan(self, analysis: Dict) -> Dict:
        """Create execution plan from analysis."""
        return {
            "retrieval": {
                "chunks": self._plan_chunk_retrieval(analysis),
                "relationships": self._plan_relationship_retrieval(analysis)
            },
            "processing": {
                "operations": self._plan_processing_operations(analysis),
                "formatting": analysis.get("output_format", "text")
            }
        }

    def execute_retrieval(self, plan: Dict) -> List[Dict]:
        """Execute retrieval steps from plan."""
        try:
            chunk_params = plan["retrieval"]["chunks"]
            query = chunk_params.get("query", self.query)
            filters = chunk_params.get("filters", {})
            limit = chunk_params.get("limit", 20)
            
            base_chunks = get_base_chunks(self.conn, query, filters, limit)
            if not base_chunks:
                self.logger.warning("No base chunks found")
                return []
            
            self.logger.debug(f"Retrieved {len(base_chunks)} chunks")
            
            if plan["retrieval"].get("relationships"):
                for chunk in base_chunks:
                    chunk_id = chunk.get("id")
                    if chunk_id:
                        relationships = get_chunk_relationships(self.conn, chunk_id)
                        if relationships:
                            chunk["relationships"] = relationships
                            
            return base_chunks
            
        except Exception as e:
            self.logger.error(f"Error in retrieval: {e}")
            return []

    def execute_processing(self, data: List[Dict], plan: Dict) -> List[Dict]:
        """Execute processing steps from plan."""
        logger.debug(f"Processing {len(data)} chunks")
        
        results = data
        for operation in plan["processing"]["operations"]:
            op_type = operation.get("type")
            params = operation.get("params", {})
            
            if op_type == "aggregate":
                results = self._aggregate_data(results, params)
                if results:
                    logger.debug(f"Aggregation produced {len(results)} results with total: {results[0].get('sum', 0)}")
                
        return results

    def generate_response(self, query: str, data: List[Dict], analysis: Dict) -> str:
        """Generate final response."""
        prompt = self._build_response_prompt(query, data, analysis)
        
        completion = self.llm.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You generate clear responses from structured data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        
        return completion.choices[0].message.content.strip()

    # Helper methods
    def _plan_chunk_retrieval(self, analysis: Dict) -> Dict:
        """Plan chunk retrieval based on analysis."""
        return {
            "query": self.query,  # Store the original query
            "filters": analysis.get("filters", {}),
            "limit": 20
        }

    def _plan_relationship_retrieval(self, analysis: Dict) -> List[str]:
        """Plan relationship retrieval based on analysis."""
        return analysis.get("relationships", [])

    def _plan_processing_operations(self, analysis: Dict) -> List[Dict]:
        """Plan processing operations based on analysis."""
        operations = []
        for op in analysis.get("operations", []):
            if "aggregate" in op.lower():
                operations.append({
                    "type": "aggregate",
                    "params": {"field": "value"}
                })
            elif "group" in op.lower():
                operations.append({
                    "type": "group",
                    "params": {"by": "type"}
                })
        return operations

    def _build_response_prompt(self, query: str, data: List[Dict], analysis: Dict) -> str:
        """Build prompt for response generation."""
        return f"""Generate a response for this query using the provided data.
        
Query: {query}

Data:
{json.dumps(data, indent=2)}

Analysis:
{json.dumps(analysis, indent=2)}

Guidelines:
- Use only information from the provided data
- Format according to {analysis.get('output_format', 'text')}
- Be clear and concise
"""

    def _aggregate_data(self, results: List[Dict], params: Dict) -> List[Dict]:
        """Aggregate data based on parameters."""
        field = params.get("field", "value")
        aggregated = defaultdict(list)
        seen_values = set()  # Track what we've seen
        
        for result in results:
            content = result.get("content", {})
            if isinstance(content, dict):
                # Look for numeric values in the content
                context = content.get("context", {})
                self._find_numeric_values(context, field, aggregated, seen_values)
                            
        return [{
            "field": key,
            "count": len(values),
            "sum": sum(values),
            "mean": statistics.mean(values) if values else 0,
            "median": statistics.median(values) if values else 0
        } for key, values in aggregated.items()]

    def _find_numeric_values(self, obj: Union[Dict, List], field: str, aggregated: defaultdict, seen: set):
        """Recursively find numeric values in nested structure."""
        if isinstance(obj, dict):
            # Look for numeric values in the context
            for k, v in obj.items():
                if k == "total_amount" or k == "price":
                    if isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "").isdigit()):
                        value = float(v)
                        product_id = obj.get("product_id")
                        if product_id and value not in seen:
                            aggregated[product_id].append(value)
                            seen.add(value)
                elif isinstance(v, (dict, list)):
                    self._find_numeric_values(v, field, aggregated, seen)
                
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    self._find_numeric_values(item, field, aggregated, seen)

    def _group_data(self, results: List[Dict], params: Dict) -> List[Dict]:
        """Group data based on parameters."""
        group_by = params.get("by")
        if not group_by:
            return results
            
        grouped = defaultdict(list)
        for result in results:
            key = result.get("content", {}).get(group_by)
            if key:
                grouped[key].append(result)
                
        return [
            {
                "group": key,
                "items": items,
                "count": len(items)
            }
            for key, items in grouped.items()
        ]

    def _filter_data(self, results: List[Dict], params: Dict) -> List[Dict]:
        """Filter data based on parameters."""
        conditions = params.get("conditions", {})
        if not conditions:
            return results
            
        filtered = []
        for result in results:
            content = result.get("content", {})
            matches = True
            
            for field, value in conditions.items():
                if content.get(field) != value:
                    matches = False
                    break
                    
            if matches:
                filtered.append(result)
                
        return filtered

def answer_query(conn, query: str) -> str:
    """Answer query using the unified pipeline."""
    pipeline = QueryPipeline(conn, client)
    result = pipeline.execute(query)
    
    if result["status"] == "error":
        logger.error(f"Pipeline error: {result['error']}")
        return "I encountered an error while processing your query. Please try again."
        
    return result["response"]

