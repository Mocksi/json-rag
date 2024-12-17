import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import openai

load_dotenv()

DATA_DIR = "data/json_docs"
POSTGRES_CONN_STR = "dbname=myragdb user=drew host=localhost port=5432"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

openai.api_key = OPENAI_API_KEY

model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(model_name)

MAX_CHUNKS = 4  # also move this constant here as it's a config-like constant

SIMILARITY_THRESHOLD = 0.85  # Minimum similarity score for semantic relationships
MIN_RELATIONSHIP_CONFIDENCE = 0.7  # Minimum confidence for relationship detection
MAX_SEMANTIC_MATCHES = 5  # Maximum number of semantic matches per entity
