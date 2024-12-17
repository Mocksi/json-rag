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
python rag_app.py --new
```

4. Start interactive chat:
```bash
python rag_app.py
```

## Usage Examples

Query metrics:
```
> Show peak network usage periods
> What were the highest CPU metrics today?
```

Query temporal data:
```
> Show events from the last hour
> What happened this week?
```

Query state changes:
```
> Show all state transitions
> What was the system state yesterday?
```

## Data Structure Support

- Event Logs
- API Responses
- Time Series Data
- Entity Relationships
- State Machines
- Configuration Data
- Metric Series

## License

MIT License - see LICENSE file for details.

## Roadmap

- [ ] Support for streaming large JSON files
- [ ] Additional RAG system integrations
- [ ] Enhanced relationship mapping
- [ ] Custom document processors
- [ ] Advanced metadata handling