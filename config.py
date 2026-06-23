import chromadb
from sentence_transformers import CrossEncoder

# model names
EMBED_MODEL = "nomic-embed-text"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# category mapping for metadata filtering
CATEGORY_MAP = {
    "refunds.txt": "billing",
    "plans.txt": "billing",
    "shipping.txt": "orders",
    "accounts.txt": "account",
    "security.txt": "account",
    "support.txt": "general",
}

# ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="support_docs")

# cross-encoder
reranker = CrossEncoder(RERANKER_MODEL)