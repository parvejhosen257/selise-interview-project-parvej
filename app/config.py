"""
Azure OpenAI configuration from environment.
Set in .env: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, deployment names.
"""
import os

from dotenv import load_dotenv

load_dotenv()

# Azure OpenAI (cognitiveservices.azure.com)
AZURE_OPENAI_ENDPOINT = os.getenv(
    "AZURE_OPENAI_ENDPOINT",
    "https://assessment-6-temp-resource.cognitiveservices.azure.com",
).rstrip("/")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
AZURE_OPENAI_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"
)

# Base URL for chat (OpenAI-compatible client)
AZURE_CHAT_BASE_URL = (
    f"{AZURE_OPENAI_ENDPOINT}/openai/deployments/{AZURE_OPENAI_CHAT_DEPLOYMENT}"
)
