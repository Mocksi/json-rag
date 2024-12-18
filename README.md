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
  - Key-value pair extraction for filtered searches
  - Embedded metadata tracking
  - Batch processing with change detection

* **Archetype-Aware Processing**:
  - Pattern detection (entities, events, metrics)
  - Archetype-based scoring and ranking
  - Relationship validation by type
  - Context-aware embedding generation
  - Hierarchical traversal limits

* **Hierarchical Data Management**:
  - Full JSON structure preservation
  - Parent-child relationship tracking
  - Depth-aware chunk generation
  - Contextual embedding with ancestry
  - Path-based chunk identification

* **Enhanced Retrieval**:
  - Vector similarity search using PGVector
  - Hierarchical context assembly
  - Entity-aware result filtering
  - Relationship-based context expansion
  - Confidence scoring and ranking


## Quick Start

1. Clone and install:
```bash
git clone https://github.com/Mocksi/json-rag.git
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

4. Initialize and run:
```bash
python -m app.main --new  # Truncates all tables and starts fresh
python -m app.main        # Normal operation
```

## Architecture
```
app/
    config.py          # Configuration settings
    database.py        # PostgreSQL interaction
    models.py          # Data validation
    parsing.py         # JSON parsing
    embedding.py       # Vector embeddings
    retrieval.py       # Chunk retrieval
    relationships.py   # Entity tracking
    archetype.py       # Pattern detection
    main.py           # Entry point
```

## License

MIT License - see LICENSE file for details.
