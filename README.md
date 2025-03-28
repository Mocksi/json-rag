# JSON RAG Integration

A tool for efficiently loading and integrating nested JSON data structures into RAG (Retrieval-Augmented Generation) systems, with enhanced entity tracking, relationship detection, and context preservation.

## Key Features

* **Advanced Query Understanding**:
  - Temporal patterns (exact dates, relative ranges, named periods)
  - Metric aggregations (average, maximum, minimum, sum, count)
  - Entity relationships (direct, semantic, and cross-file connections)
  - State transitions and system conditions
  - Hybrid search combining vector similarity, relationships, and filters

* **Smart Data Processing**:
  - Automatic entity detection and relationship mapping
  - Cross-file relationship detection and validation
  - Key-value pair extraction for filtered searches
  - Embedded metadata tracking
  - Batch processing with change detection

* **Archetype-Aware Processing**:
  - Pattern detection (entities, events, metrics, collections)
  - Archetype-based scoring and ranking
  - Relationship validation by archetype
  - Context-aware embedding generation
  - Archetype-specific traversal strategies

* **Hierarchical Data Management**:
  - Full JSON structure preservation
  - Parent-child relationship tracking
  - Cross-file relationship mapping
  - Contextual embedding with ancestry
  - Path-based chunk identification

* **Enhanced Retrieval**:
  - Vector similarity search using PGVector
  - Relationship-aware context assembly
  - Entity-aware result filtering
  - Cross-file context expansion
  - Confidence-based scoring and ranking


## Quick Start

1. Clone and install:
```bash
git clone https://github.com/Mocksi/json-rag.git
cd json_rag
uv venv rag_env
source rag_env/bin/activate  # Windows: .\rag_env\Scripts\activate
uv pip install -r requirements.txt
```

2. Set up environment:
```bash
# Create .env file with:
OPENAI_API_KEY=your-key-here
POSTGRES_DB=crowllector
POSTGRES_USER=crowllector
POSTGRES_PASSWORD=yourpassword
POSTGRES_HOST=localhost
POSTGRES_DB_PORT=5432
```

3. Initialize and run:
```bash
python -m app.main --new  # Truncates all tables and starts fresh
python -m app.main        # Normal operation
```

## Architecture
```
app/
├── analysis/           # Analysis and pattern detection
│   ├── archetype.py   # Pattern and archetype detection
│   └── relationships.py# Cross-file relationship analysis
├── core/              # Core system components
│   ├── config.py      # Configuration settings
│   └── models.py      # Data models
├── processing/        # Data processing modules
│   ├── json_parser.py # JSON structure parsing
│   ├── parsing.py     # Document parsing and chunking
│   └── processor.py   # Data processing pipeline
├── retrieval/         # Query processing and retrieval
│   ├── embedding.py   # Vector embedding generation
│   └── retrieval.py   # Query pipeline and execution
├── storage/           # Data persistence
│   └── database.py    # PostgreSQL and vector storage
├── utils/             # Utility modules
│   └── logging_config.py # Logging configuration
├── __init__.py        # Package initialization
├── chat.py           # Chat interface and interactions
└── main.py           # Application entry point
```

The codebase is organized into logical modules:

- **analysis/**: Modules for analyzing data patterns, cross-file relationships, and user intent
- **core/**: Core system configuration and shared components
- **processing/**: Data processing and relationship detection modules
- **retrieval/**: Relationship-aware search and context assembly
- **storage/**: Database interaction and relationship persistence
- **utils/**: Shared utility functions and helpers

Each module is designed to be independent with clear responsibilities, while working together through well-defined interfaces.

## Installation Requirements

- Python 3.8 or higher
- PostgreSQL 12 or higher with PGVector extension
- OpenAI API key
- Required Python packages (see requirements.txt)

## Documentation

The codebase features comprehensive inline documentation:
- Detailed module-level docstrings explaining key concepts
- Function and class documentation with examples
- Type hints and parameter descriptions
- Usage examples and implementation notes

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on:
- Setting up your development environment
- Code style guidelines
- Pull request process
- Development workflow

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior.

## License

MIT License - see LICENSE file for details.

## Roadmap

- [x] Cross-file relationship detection
- [x] Archetype-aware retrieval
- [x] Relationship-based context expansion
- [x] Confidence scoring algorithm refinement
- [ ] State transition handling improvements
- [ ] Batch processing optimization
- [ ] Metric aggregation capabilities
- [ ] Entity filtering rules improvement
- [ ] Context assembly performance optimization
- [ ] Advanced archetype pattern detection

## Query Pipeline

The system implements a structured reasoning pipeline:

1. **Query Analysis**: 
   - Determines required data types
   - Identifies needed operations (filtering, aggregation)
   - Detects relationships and constraints

2. **Plan Creation**:
   - Builds retrieval strategy
   - Plans processing operations
   - Determines result formatting

3. **Execution**:
   - Retrieves relevant chunks
   - Processes according to plan
   - Assembles coherent response

This systematic approach ensures consistent and reliable query handling while preserving context and relationships.

