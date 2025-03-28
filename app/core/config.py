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

# Database configuration - load from environment variables with fallbacks
DB_NAME = os.environ.get("POSTGRES_DB", "crowllector")
DB_USER = os.environ.get("POSTGRES_USER", "crowllector")
DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "yourpassword")
DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
DB_PORT = os.environ.get("POSTGRES_DB_PORT", "5432")

POSTGRES_CONN_STR = f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} host={DB_HOST} port={DB_PORT}"

# OpenAI API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("WARNING: OPENAI_API_KEY environment variable is not set.")
    print("To use OpenAI features, please create a .env file with your API key.")
    print("Example: OPENAI_API_KEY=your_api_key_here")
    OPENAI_API_KEY = "dummy_key_for_testing"

openai.api_key = OPENAI_API_KEY

# Embedding model configuration
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Efficient, general-purpose embedding model
embedding_model = SentenceTransformer(model_name)

# Retrieval system parameters
MAX_CHUNKS = 2  # Maximum chunks to return in search results (reduced from 4 to reduce token usage)

# Relationship detection configuration
SIMILARITY_THRESHOLD = 0.85  # Minimum cosine similarity for semantic relationships
MIN_RELATIONSHIP_CONFIDENCE = 0.7  # Minimum confidence score for relationship detection
MAX_SEMANTIC_MATCHES = 5  # Maximum semantic matches per entity search
