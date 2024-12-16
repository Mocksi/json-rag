# JSON RAG Integration

A tool for efficiently loading and integrating nested JSON data structures into RAG (Retrieval-Augmented Generation) systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/downloads/)

## Overview

This project provides utilities for processing hierarchical JSON data and preparing it for use in RAG applications. It handles nested JSON structures while preserving relationships between different data levels, making it ideal for complex organizational data.

## Features

* Flattens nested JSON structures while maintaining referential integrity
* Preserves metadata and tracks schema evolution
* Supports recursive traversal of deeply nested objects
* Handles arrays and nested object arrays
* Maintains parent-child relationships in the processed output

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

## Data Structure
The system processes JSON files from the `data/json_docs/` directory. Example structure:

```json
{
  "metadata": {
    "version": "1.2",
    "source": "internal"
  },
  "offices": [
    {
      "city": "Chicago",
      "employees": [
        {
          "name": "Alice",
          "role": "Manager"
        }
      ]
    }
  ]
}
```

## Database Schema

The system uses PostgreSQL with the following tables:
- `json_chunks`: Stores document chunks and their embeddings
- `file_metadata`: Tracks processed files and their hashes
- `schema_evolution`: Monitors JSON schema changes over time

## License

Distributed under the MIT License. See `LICENSE` for more information.


## Roadmap

- [ ] Support for streaming large JSON files
- [ ] Additional RAG system integrations
- [ ] Enhanced relationship mapping
- [ ] Custom document processors
- [ ] Advanced metadata handling