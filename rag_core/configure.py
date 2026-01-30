import os

TEXT_COLUMNS = ["title", "summary", "authors", "primary_category"]

#  Paths 
DATA_PATH = os.getenv("DATA_PATH", "data/arxiv_ai.csv")
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", "artifacts")

# Models
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "tiiuae/falcon-rw-1b"
)

# Retrieval 
TOP_K = int(os.getenv("TOP_K", 3))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", 0.5))
FAISS_WEIGHT = float(os.getenv("FAISS_WEIGHT", 0.5))

#  Chunking 
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 300))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Runtime 
DEVICE = os.getenv("DEVICE", "cpu")
