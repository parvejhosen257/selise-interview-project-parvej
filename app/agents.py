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
        "For each question: (1) Use the Document Retriever to get relevant context. "
        "(2) Answer clearly and cite the docs; "
        "Do not invent information. Be concise."
    ),
    backstory=(
        "You are a legal/document assistant. You retrieve, judge relevance, then answer or decline. "
      
    ),
    llm=llm,
    verbose=False,
)
