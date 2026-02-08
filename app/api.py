"""
FastAPI interface for the agentic RAG system.
Run: uvicorn app.api:app --reload (from project root)
"""
from fastapi import FastAPI

from app.crew import run_crew

app = FastAPI(title="Mini Agentic RAG (CrewAI)")


@app.get("/ask")
def ask(q: str):
    return {"answer": str(run_crew(q))}


@app.get("/health")
def health():
    return {"status": "ok"}
