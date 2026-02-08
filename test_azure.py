#!/usr/bin/env python3
"""
Quick test that Azure OpenAI embeddings and chat are working.
Run from project root: python test_azure.py
"""
import os

# Load .env before importing app (which uses config)
from dotenv import load_dotenv
load_dotenv()

def test_embeddings():
    """Test Azure OpenAI embeddings (text-embedding-ada-002)."""
    from langchain_openai import AzureOpenAIEmbeddings
    from app.config import (
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_API_VERSION,
        AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
        AZURE_OPENAI_ENDPOINT,
    )
    if not AZURE_OPENAI_API_KEY:
        print("SKIP embeddings: AZURE_OPENAI_API_KEY not set")
        return False
    print("Testing embeddings...", end=" ")
    emb = AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    )
    vec = emb.embed_query("test sentence for embedding")
    assert isinstance(vec, list) and len(vec) > 0, "expected a vector"
    print(f"OK (dim={len(vec)})")
    return True

def test_chat():
    """Test Azure OpenAI chat (gpt-4o-mini) via LangChain."""
    from langchain_openai import AzureChatOpenAI
    from app.config import (
        AZURE_OPENAI_API_KEY,
        AZURE_OPENAI_API_VERSION,
        AZURE_OPENAI_CHAT_DEPLOYMENT,
        AZURE_OPENAI_ENDPOINT,
    )
    if not AZURE_OPENAI_API_KEY:
        print("SKIP chat: AZURE_OPENAI_API_KEY not set")
        return False
    print("Testing chat...", end=" ")
    llm = AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_CHAT_DEPLOYMENT,
        temperature=0,
    )
    reply = llm.invoke("Reply with exactly: OK")
    content = (reply.content if hasattr(reply, "content") else str(reply)).strip()
    assert content, "empty reply"
    print(f"OK (reply: {content[:50]!r})")
    return True

if __name__ == "__main__":
    print("Azure OpenAI connectivity check\n")
    ok_e = test_embeddings()
    ok_c = test_chat()
    print()
    if ok_e and ok_c:
        print("All checks passed.")
    else:
        print("Some checks failed or were skipped.")
        exit(1)
