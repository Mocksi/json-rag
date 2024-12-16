# JSON RAG Integration

A tool for efficiently loading and integrating nested JSON data structures into RAG (Retrieval-Augmented Generation) systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10-blue)](https://www.python.org/downloads/)

## Overview

This project provides utilities for processing hierarchical JSON data and preparing it for use in RAG applications. It handles nested JSON structures while preserving relationships between different data levels, making it ideal for complex organizational data.

## Features

* Flattens nested JSON structures while maintaining referential integrity
* Preserves metadata at each level of the hierarchy
* Supports recursive traversal of deeply nested objects
* Handles arrays and nested object arrays
* Maintains parent-child relationships in the processed output

## Installation

```bash
pip install json-rag-integration
```

## Quick Start

```python
from json_rag import JSONProcessor

# Load your JSON data
processor = JSONProcessor('data.json')

# Process and prepare for RAG
processed_data = processor.prepare_for_rag()

# Use with your favorite RAG system
documents = processed_data.to_documents()
```

## Sample Data Structure
The system is designed to handle complex nested JSON structures:

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
          "role": "Manager",
          "details": {
            "years_at_company": 5,
            "projects": [
              {"name": "ProjectX", "status": "ongoing"},
              {"name": "ProjectY", "status": "completed"}
            ]
          }
        }
      ]
    }
  ]
}
```

## Key Components
### JSONProcessor
The main class that handles JSON processing and preparation for RAG systems.

```python
# Initialize with custom configuration
config = {
    "flatten_arrays": True,
    "preserve_paths": True,
    "include_metadata": True
}
processor = JSONProcessor('data.json', config=config)
```

### Document Generation

The system generates the following document types:

* Root level metadata document
* Office level documents with location context
* Employee documents with hierarchical relationships
* Project documents linked to employees

## Best Practices

1. **Path Preservation**
  * Keep track of JSON paths for better context retrieval
  * Use consistent path formatting

2. **Metadata Handling**
  * Include relevant metadata in each generated document
  * Maintain versioning information

3. **Relationship Mapping**
  * Maintain explicit references between related documents
  * Use consistent reference formatting

4. **Array Processing**
  * Configure array handling based on your use case
  * Consider pagination for large arrays

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.


## Roadmap

- [ ] Support for streaming large JSON files
- [ ] Additional RAG system integrations
- [ ] Enhanced relationship mapping
- [ ] Custom document processors
- [ ] Advanced metadata handling