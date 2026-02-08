"""
FAISS retrieval logic. Loads index from project root (faiss_index).
Uses Azure OpenAI embeddings (same config as ingest).
"""
from pathlib import Path

from langchain_community.vectorstores import FAISS

from app.embeddings import get_embeddings

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "faiss_index"

embeddings = get_embeddings()
_db = None


def _get_db():
    global _db
    if _db is None:
        _db = FAISS.load_local(
            str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True
        )
    return _db


def retrieve_context(query: str, k: int = 4) -> str:
    db = _get_db()
    docs = db.similarity_search(query, k=k)
    return "\n\n".join(d.page_content for d in docs)
