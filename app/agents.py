"""
CrewAI: single RAG agent — retrieve, evaluate sufficiency, then answer (one pass).
Uses Azure OpenAI for chat (gpt-4o-mini deployment).
"""
import os

from crewai import Agent
from crewai.llm import LLM

from app.config import (
    AZURE_CHAT_BASE_URL,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_CHAT_DEPLOYMENT,
)

os.environ["OPENAI_API_KEY"] = AZURE_OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = AZURE_CHAT_BASE_URL

llm = LLM(
    model=AZURE_OPENAI_CHAT_DEPLOYMENT,
    base_url=AZURE_CHAT_BASE_URL,
    api_key=AZURE_OPENAI_API_KEY,
    default_query={"api-version": AZURE_OPENAI_API_VERSION},
    temperature=0,
)

# Single agent: retrieve → evaluate if docs suffice → answer (one pass)
rag_agent = Agent(
    role="RAG Assistant",
    goal=(
        "For each question:"
        "(1) Retrieve relevant context using the Document Retriever."
        "(2) Provide a clear, concise, and accurate answer based solely on the retrieved documents."
        "(3) Cite the supporting sections, clauses, or laws where applicable."
        "(4) Do not invent or assume any information."
        "(5) If the answer cannot be determined from the documents, respond: 'I don't have sufficient information to answer this question.'"
         ),
    backstory=(
        "You are a document answering assistant. You retrieve, judge relevance, then answer or decline. "
      
    ),
    llm=llm,
    verbose=False,
)
