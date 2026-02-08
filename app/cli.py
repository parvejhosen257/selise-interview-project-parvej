"""
CLI interface for the agentic RAG system.
Run: python -m app.cli (from project root)
"""
from app.crew import run_crew

if __name__ == "__main__":
    print("Mini Agentic RAG — type your question (or 'exit' to quit).\n")
    while True:
        try:
            q = input("Ask a question (or exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q.lower() == "exit":
            break
        print(run_crew(q))
        print()
