# JSON RAG Integration

A tool for efficiently loading and integrating nested JSON data structures into RAG (Retrieval-Augmented Generation) systems, with enhanced entity tracking and context preservation.

## Key Features

* **Smart Archetype Detection**: Automatically identifies data patterns (event logs, API responses, metrics, etc.)
* **Context-Aware Chunking**: Preserves relationships and structure based on data type
* **Intelligent Summarization**: Generates summaries tailored to data patterns:
  - Event sequences with causal chains
  - API responses with resource relationships
  - Metric series with trend analysis
  - Temporal data with time-based grouping
  - State transitions with context preservation
* **Advanced Intent Detection**: Automatically identifies query types:
  - Temporal queries (time-based relationships)
  - Metric queries (numerical analysis)
  - Entity queries (relationship mapping)
  - State queries (transition tracking)
* **Debug Capabilities**:
  - Query intent analysis logging
  - Vector similarity search debugging
  - Chunk retrieval scoring
  - Prompt construction tracking
  - Summarization process monitoring

## Quick Start

1. Clone and install:
```bash
git clone https://github.com/yourusername/json_rag.git
cd json_rag
python -m venv rag_env
source rag_env/bin/activate  # Windows: rag_env\Scripts\activate
pip install -r requirements.txt
```

2. Set up environment:
```bash
# Create .env file with:
OPENAI_API_KEY=your-key-here
```

3. Initialize system:
```bash
python rag_app.py --new  # Resets database and initializes embeddings
```

4. Start interactive chat:
```bash
python rag_app.py  # Checks for changes and updates embeddings if needed
```

## Usage Examples

Query metrics:
```
> Show peak network usage periods
> What were the highest CPU metrics today?
> Show me the progression of all metrics during peak usage
```

Query temporal data:
```
> Show events from the last hour
> What happened this week?
> What's the trend of errors over time?
```

Query state changes:
```
> Show all state transitions
> What was the system state yesterday?
> Track status changes during the outage
```

## Debug Mode

To enable detailed debugging output:
```bash
export DEBUG_LEVEL=verbose  # Shows detailed processing information
python rag_app.py
```

Debug output includes:
- Query intent analysis
- Vector search scores
- Retrieved chunk details
- Prompt construction
- Summarization process

## Architecture

The system is organized into modules:
- `intent.py`: Query intent analysis
- `retrieval.py`: Vector search and chunk retrieval
- `embedding.py`: Document embedding
- `parsing.py`: JSON structure analysis
- `prompts.py`: LLM prompt management
- `database.py`: PostgreSQL integration

## License

MIT License - see LICENSE file for details.

## Roadmap

- [ ] Support for streaming large JSON files
- [ ] Additional RAG system integrations
- [ ] Enhanced relationship mapping
- [ ] Custom document processors
- [ ] Advanced metadata handling
- [ ] Improved debugging tools
- [ ] Performance optimization