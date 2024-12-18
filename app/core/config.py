"""
Configuration Module for JSON RAG System

This module manages configuration settings for the JSON RAG (Retrieval-Augmented Generation)
system. It handles environment variables, default settings, and runtime configuration
for various system components.

Configuration Categories:
    - Database Settings: Connection strings, pool sizes
    - Embedding Model: Model selection and parameters
    - Processing Limits: Chunk sizes, batch sizes
    - Search Settings: Similarity thresholds, result limits
    - Logging: Log levels and output configuration

Usage:
    >>> from app.core.config import POSTGRES_CONN_STR, embedding_model
    >>> print(f"Using database: {POSTGRES_CONN_STR}")
    >>> print(f"Embedding model: {embedding_model}")
"""

import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai

# Load environment variables from .env file
load_dotenv()

# Data storage configuration
DATA_DIR = "data/json_docs"  # Directory for JSON document storage

# Database configuration
POSTGRES_CONN_STR = "dbname=myragdb user=drew host=localhost port=5432"

# OpenAI API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

openai.api_key = OPENAI_API_KEY

# Embedding model configuration
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Efficient, general-purpose embedding model
embedding_model = SentenceTransformer(model_name)

# Retrieval system parameters
MAX_CHUNKS = 4  # Maximum chunks to return in search results

# Relationship detection configuration
SIMILARITY_THRESHOLD = 0.85  # Minimum cosine similarity for semantic relationships
MIN_RELATIONSHIP_CONFIDENCE = 0.7  # Minimum confidence score for relationship detection
MAX_SEMANTIC_MATCHES = 5  # Maximum semantic matches per entity search
