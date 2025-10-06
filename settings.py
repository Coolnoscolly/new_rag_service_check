from dotenv import load_dotenv
import os

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = os.getenv("QDRANT_PORT", "6333")
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = os.getenv("API_PORT", "8000")

EMBEDDING_OLLAMA_HOST = os.getenv("EMBEDDING_OLLAMA_HOST", "localhost")
GENERATION_OLLAMA_HOST = os.getenv("GENERATION_OLLAMA_HOST", "localhost")