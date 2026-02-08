"""
Single task: retrieve → evaluate sufficiency → answer (one agent, one pass).
"""
from crewai import Task

from app.agents import rag_agent
from app.tools import document_retriever


def create_tasks(question: str):
    task = Task(
        description=f"""In one pass:
1. Use the Document Retriever tool to get relevant context for the question.
2. Evaluate whether the retrieved documents are sufficient to answer the question.
3. If sufficient: give a clearly formatted answer with inline citations (e.g. [1], [2] or section names). At the end, add a "Sources" section listing each source used (document/section and brief reference).
4. If insufficient: say that the retrieved docs are not sufficient to answer.

Use plain text only: no markdown (no **, ##, -, *, or other markdown). Short paragraphs and simple line breaks for readability. Use citations in the body and list all sources at the end.

Question: {question}""",
        expected_output=(
            "Plain text only (no markdown). Answer with inline citations and a Sources section at the end. "
            "If docs are insufficient, a clear statement to that effect."
        ),
        agent=rag_agent,
        tools=[document_retriever],
    )
    return [task]
