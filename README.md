# Chat with Your Codebase

A Retrieval-Augmented Generation (RAG) tool that enables natural language conversations with any public GitHub repository. Point it at a repo, ask questions about the code, and get intelligent answers powered by semantic search and language models.

## Features

- **Repository Ingestion**: Clone and analyze GitHub repositories with syntax-aware chunking
- **Semantic Search**: BAAI/bge-base-en-v1.5 embeddings with FAISS vector indexing
- **RAG Pipeline**: Google Gemma-2-2b-it model for intelligent code discussions
- **CLI Interface**: Simple command-line tool built with Typer

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Ingest a repository**:
   ```bash
   python -m codechat ingest --repo-url https://github.com/pallets/flask --out-store stores/flask
   ```

3. **Chat with the codebase**:
   ```bash
   python -m codechat chat --store stores/flask
   ```

## Development

Install development dependencies and run tests:
```bash
pip install -r requirements-dev.txt
pytest --cov=codechat tests/
```

## License

MIT License
