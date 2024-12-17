# JSON RAG Integration

A tool for efficiently loading and integrating nested JSON data structures into RAG (Retrieval-Augmented Generation) systems, with enhanced entity tracking and context preservation.

## Key Features

* **Advanced Query Understanding**:
  - Temporal patterns (exact dates, relative ranges, named periods)
  - Metric aggregations (average, maximum, minimum, sum, count)
  - Entity relationships (direct and semantic connections)
  - State transitions and system conditions
  - Hybrid search combining vector similarity and filters

* **Smart Data Processing**:
  - Automatic entity detection and relationship mapping
  - Context-aware chunking with hierarchy preservation
  - Key-value pair extraction for filtered searches
  - Embedded metadata tracking
  - Batch processing with change detection

* **Enhanced Retrieval**:
  - Vector similarity search using PGVector
  - Hierarchical context assembly
  - Entity-aware result filtering
  - Relationship-based context expansion
  - Confidence scoring and ranking

* **Debug Capabilities**:
  - Query intent analysis with pattern matching
  - Entity relationship detection logs
  - Embedding generation details
  - Vector search scores and filtering
  - Chunk processing and context assembly
  - Database operation logging

## Quick Start

1. Clone and install:
```bash
git clone https://github.com/yourusername/json_rag.git
cd json_rag
python -m venv rag_env
source rag_env/bin/activate  # Windows: rag_env\Scripts\activate
pip install -r requirements.txt
```

2. Configure database:
```bash
# Update POSTGRES_CONN_STR in app/config.py:
POSTGRES_CONN_STR = "dbname=myragdb user=your_user host=localhost port=5432"
```

3. Set up environment:
```bash
# Create .env file with:
OPENAI_API_KEY=your-key-here
```

4. Initialize system:
```bash
python -m app.main --new  # Resets database and initializes embeddings
```

5. Start interactive chat:
```bash
python -m app.main  # Checks for changes and updates embeddings if needed
```

## Usage Examples

### Temporal Queries
```bash
# Exact date range
> Show events between 2024-01-01 and 2024-02-01

# Relative time range
> What happened in the last 7 days
> Show changes from the last 2 weeks

# Named periods
> Show this week's activity
> Get all events from this month
```

### Metric Queries
```bash
# Aggregations with conditions
> Show average CPU usage above 80%
> What was the peak network usage last month
> Count all errors where status=failed

# Trend analysis
> Show the progression of memory usage
> Track response times over the last hour
```

### Entity Queries
```bash
# Direct relationships
> Show all suppliers connected to warehouse WH-EAST
> What items are related to order #123

# Filtered searches
> Find all transactions by user john
> List orders with status=pending
```

### State Queries
```bash
# Transition tracking
> Track status changes where category=shipping
> Show all state transitions in the last week
> List items that changed from pending to active
```

## Architecture

```
project_root/
    app/
        __init__.py
        config.py          # Configuration and environment settings
        database.py        # PostgreSQL interaction and schema management
        models.py         # Data validation with Pydantic
        parsing.py        # JSON parsing and chunk generation
        embedding.py      # Vector embeddings and similarity search
        retrieval.py      # Chunk retrieval and context assembly
        intent.py         # Query intent analysis and prompt generation
        relationships.py  # Entity relationship tracking
        chat.py          # Interactive chat interface
        main.py          # Application entry point
        utils.py         # Utility functions
    data/json_docs/      # JSON document storage
    requirements.txt
    .env
```

## Debug Mode

Enable detailed debugging output:
```bash
export DEBUG_LEVEL=verbose
python -m app.main
```

Debug output includes:
- Query intent analysis with pattern matches
- Entity detection and relationship mapping
- Embedding generation metrics
- Vector search scores and filtering
- Chunk processing and context assembly
- Database operations and performance metrics

## License

MIT License - see LICENSE file for details.

## Roadmap

- [ ] Streaming support for large JSON files
- [ ] Additional embedding model options
- [ ] Enhanced relationship visualization
- [ ] Custom chunking strategies
- [ ] Advanced caching mechanisms
- [ ] Performance optimization for large datasets
- [ ] API interface for non-interactive usage