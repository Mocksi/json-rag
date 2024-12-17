import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai

"""
Configuration module for the JSON RAG system.
Contains environment variables, model settings, and system constants.

Environment Variables Required:
    OPENAI_API_KEY: API key for OpenAI services
"""

load_dotenv()

# Directory containing JSON documents to process
DATA_DIR = "data/json_docs"

# PostgreSQL connection string
POSTGRES_CONN_STR = "dbname=myragdb user=drew host=localhost port=5432"

# OpenAI API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

openai.api_key = OPENAI_API_KEY

# Embedding model configuration
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_name)

# System constants
MAX_CHUNKS = 4  # Maximum number of chunks to return in search results

# Relationship detection settings
SIMILARITY_THRESHOLD = 0.85  # Minimum similarity score for semantic relationships
MIN_RELATIONSHIP_CONFIDENCE = 0.7  # Minimum confidence for relationship detection
MAX_SEMANTIC_MATCHES = 5  # Maximum number of semantic matches per entity
