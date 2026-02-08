# Mini Agentic RAG (CrewAI)

A minimal RAG system with semantic chunking, Azure OpenAI embeddings, FAISS, and a self-critic CrewAI agent that retrieves documents, evaluates whether they suffice to answer, then answers with inline citations and a Sources section at the end.

---

## Features

- Semantic chunking (LangChain) for better retrieval
- Azure OpenAI for embeddings and chat (gpt-4o-mini)
- FAISS vector store for similarity search
- Self-critic RAG agent with tool calling (Document Retriever)
- Retrieve → evaluate sufficiency → answer (with self-critique of relevance)
- Answers with citations and a Sources section at the end
- FastAPI and CLI interfaces

---

## Architecture

### System overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OFFLINE: INGEST                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  data/docs/ (.pdf, .txt, .md, .rst)                                         │
│         │                                                                   │
│         ▼                                                                   │
│  Load documents  →  Semantic chunking  →  Azure OpenAI embeddings           │
│         │                                                                   │
│         ▼                                                                   │
│  FAISS index (faiss_index/)                                                 │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           ONLINE: QUERY                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│  User question  →  API / CLI  →  Crew (self-critic agent)                   │
│                              →  RAG Agent (retrieve, critique, answer)     │
│                                    │                                        │
│                                    ├─► Document Retriever tool              │
│                                    │        │                               │
│                                    │        ▼                               │
│                                    │   FAISS similarity_search              │
│                                    │                                        │
│                                    ▼                                        │
│                             Evaluate sufficiency                            │
│                                    │                                        │
│                                    ▼                                        │
│                             Answer (or “insufficient”)                      │
│                             + citations + Sources section                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Self-critic agent and steps

The RAG agent acts as a self-critic: it retrieves, evaluates whether the retrieved content is sufficient, then answers or declines.

```mermaid
flowchart LR
    subgraph Input
        Q[User Question]
    end

    subgraph Agent["Self-critic RAG Agent"]
        S1[1. Retrieve]
        S2[2. Evaluate / Self-critique]
        S3[3. Answer]
        S1 --> S2 --> S3
    end

    subgraph Tool
        T[Document Retriever]
        DB[(FAISS Index)]
        T --> DB
    end

    Q --> Agent
    S1 --> T
    T --> S2
    S3 --> Out[Answer + Citations + Sources]
```

| Step | Description |
|------|-------------|
| 1. Retrieve | Agent calls the **Document Retriever** tool with the question. The tool runs a FAISS similarity search (Azure embeddings) and returns the top-k chunks. |
| 2. Evaluate / Self-critique | Agent evaluates whether the retrieved chunks are **sufficient** to answer the question (self-critique of relevance and coverage). |
| 3. Answer | If sufficient: answer with inline citations (e.g. [1], [2]) and a **Sources** section at the end. If not: state that the retrieved docs are insufficient. |

---

## Project structure

```
├── data/
│   └── docs/              # Put .pdf, .txt, .md, .rst files here
├── faiss_index/           # Created by ingest (do not edit)
├── app/
│   ├── config.py          # Azure OpenAI settings from .env
│   ├── embeddings.py     # Azure embeddings (rate-limited)
│   ├── ingest.py         # Semantic chunking + FAISS index build
│   ├── retriever.py      # FAISS retrieval (retrieve_context)
│   ├── tools.py          # CrewAI tool: Document Retriever
│   ├── agents.py         # Self-critic RAG agent (CrewAI)
│   ├── tasks.py          # Task: retrieve → evaluate → answer
│   ├── crew.py           # Crew orchestration
│   ├── api.py            # FastAPI server
│   └── cli.py            # Interactive CLI
├── .env.example
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- Azure OpenAI resource (embeddings + chat deployments)

---

## Setup

From the project root:

```bash
python -m venv env
source env/bin/activate   # Windows: env\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your Azure OpenAI values:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.cognitiveservices.azure.com
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

Optional (if you hit rate limits):

```env
EMBEDDING_RATE_LIMIT_DELAY_SECONDS=0.5
```

---

## Usage

### 1. Add documents

Place files in `data/docs/` (e.g. PDFs, .txt, .md, .rst).

### 2. Ingest (build FAISS index)

Run once (or after adding/updating docs):

```bash
python -m app.ingest
```

### 3. Ask questions

CLI (interactive):

```bash
python -m app.cli
```

API:

```bash
uvicorn app.api:app --reload
```

Then:

```text
GET /ask?q=Your+question
GET /health
```

---

## Configuration summary

| Variable | Purpose |
|----------|---------|
| AZURE_OPENAI_ENDPOINT | Azure OpenAI base URL |
| AZURE_OPENAI_API_KEY | API key |
| AZURE_OPENAI_CHAT_DEPLOYMENT | Chat model (e.g. gpt-4o-mini) |
| AZURE_OPENAI_EMBEDDING_DEPLOYMENT | Embedding model (e.g. text-embedding-ada-002) |
| AZURE_OPENAI_API_VERSION | API version |
| EMBEDDING_RATE_LIMIT_DELAY_SECONDS | Optional delay between embedding calls |

---

## Output format

The self-critic agent returns:

- Inline citations (e.g. [1], [2] or section names).
- A **Sources** section at the end listing each document/section used.
- If the retrieved docs are not sufficient, a clear statement that they are insufficient.

---

## License

Use as needed for the assessment or project.
