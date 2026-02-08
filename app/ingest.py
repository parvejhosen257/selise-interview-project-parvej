"""
Semantic chunking + Azure OpenAI embeddings + FAISS indexing.
Loads PDF and text files from data/docs. Run from project root: python -m app.ingest
"""
import os
from pathlib import Path

from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader

from app.embeddings import get_embeddings

embeddings = get_embeddings()

# Resolve paths relative to project root (parent of app/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_PATH = PROJECT_ROOT / "data" / "docs"
INDEX_PATH = PROJECT_ROOT / "faiss_index"

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md", ".rst")


def ingest_docs(path: str | Path | None = None) -> None:
    path = path or DOCS_PATH
    path = Path(path)
    docs = []

    for file in os.listdir(path):
        if file.startswith("."):
            continue
        file_path = path / file
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        try:
            if ext == ".pdf":
                docs.extend(PyPDFLoader(str(file_path)).load())
            else:
                docs.extend(TextLoader(str(file_path)).load())
        except Exception as e:
            print(f"Skip {file}: {e}")

    if not docs:
        raise ValueError(f"No .pdf/.txt/.md/.rst documents found in {path}")

    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
    )
    chunks = chunker.split_documents(docs)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(INDEX_PATH))
    print(f"Indexed {len(chunks)} chunks into {INDEX_PATH}")


if __name__ == "__main__":
    ingest_docs()
