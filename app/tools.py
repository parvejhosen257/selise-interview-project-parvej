"""
CrewAI tool: Document Retriever (satisfies explicit tool calling).
"""
from crewai.tools import tool

from app.retriever import retrieve_context


@tool("Document Retriever")
def document_retriever(query: str) -> str:
    """Retrieves relevant document context for a given question."""
    return retrieve_context(query)
