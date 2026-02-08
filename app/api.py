"""
FastAPI interface for the agentic RAG system.
Run: uvicorn app.api:app --reload (from project root)
"""
from fastapi import FastAPI

from app.crew import run_crew
from app.retriever import retrieve_chunks

app = FastAPI(title="Mini Agentic RAG (CrewAI)")


@app.get("/ask")
def ask(q: str):
    answer = str(run_crew(q))
    chunks = retrieve_chunks(q)
    return {"answer": answer, "retrieved_chunks": chunks}


@app.get("/health")
def health():
    return {"status": "ok"}
