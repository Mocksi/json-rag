# JSON RAG Integration

A tool for efficiently loading and integrating nested JSON data structures into RAG (Retrieval-Augmented Generation) systems, with enhanced entity tracking and context preservation.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/downloads/)

## Overview

This project provides utilities for processing hierarchical JSON data and preparing it for use in RAG applications. It features advanced entity tracking, relationship mapping, hybrid retrieval, and hierarchical summarization.

## Features

* Advanced entity tracking and relationship mapping
* Hierarchical context preservation in chunks
* Path-aware chunking strategy
* Hybrid retrieval (vector + keyword filtering)
* Hierarchical summarization for large contexts
* Schema evolution tracking
* Secure API key management
* PostgreSQL vector storage

## Installation

1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv rag_env
source rag_env/bin/activate  # On Windows: rag_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your-key-here
```

## Usage

Start the interactive chat:
```bash
python rag_app.py
```

Reset database and start fresh:
```bash
python rag_app.py --new
```

Type `:quit` to exit the chat.

## Advanced Features

### Hybrid Retrieval
Combine vector similarity search with keyword filtering:
```
You: department=Engineering Show all projects
```

### Hierarchical Summarization
Automatically handles large context windows by:
1. Retrieving relevant chunks
2. Summarizing in batches if needed
3. Maintaining entity relationships in summaries

## Data Processing

The system processes JSON files with:
* Entity detection and relationship mapping
* Context-aware chunking
* Path preservation
* Hierarchical relationship tracking

Example of processed structure:
```json
{
  "organization": {
    "name": "Acme Corp",
    "projects": [
      {
        "name": "Project Alpha",
        "team": {
          "lead": {
            "name": "Alice Smith",
            "role": "Project Manager"
          }
        }
      }
    ]
  }
}
```

## Database Schema

The system uses PostgreSQL with the following tables:
- `json_chunks`: Stores document chunks and their embeddings
- `file_metadata`: Tracks processed files and their hashes
- `schema_evolution`: Monitors JSON schema changes over time
- `chunk_keys_index`: Indexes key-value pairs for hybrid retrieval

## Features in Detail

### Entity Tracking
- Identifies and tracks named entities
- Maps relationships between entities
- Preserves hierarchical context
- Tracks roles and associations

### Context Preservation
- Maintains path-based context
- Preserves parent-child relationships
- Tracks organizational hierarchy
- Maintains reference integrity

### Intelligent Chunking
- Context-aware chunk creation
- Entity relationship preservation
- Hierarchical path tracking
- Smart boundary detection

### Hybrid Search
- Vector similarity search
- Keyword-based filtering
- Combined ranking system
- Path-aware querying

### Hierarchical Summarization
- Multi-level summarization
- Context preservation in summaries
- Entity relationship maintenance
- Automatic chunk management

## License

Distributed under the MIT License. See `LICENSE` for more information.


## Roadmap

- [ ] Support for streaming large JSON files
- [ ] Additional RAG system integrations
- [ ] Enhanced relationship mapping
- [ ] Custom document processors
- [ ] Advanced metadata handling