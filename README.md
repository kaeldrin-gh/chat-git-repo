# Chat with Your Codebase

A Retrieval-Augmented Generation (RAG) tool that enables natural language conversations with any public GitHub repository. Point it at a repo, ask questions about the code, and get intelligent answers powered by semantic search and language models.

## Features

- **Repository Ingestion**: Clone and analyze GitHub repositories with syntax-aware chunking
- **Semantic Search**: BAAI/bge-base-en-v1.5 embeddings with FAISS vector indexing
- **RAG Pipeline**: Google Gemma-2-2b-it model for intelligent code discussions
- **CLI Interface**: Simple command-line tool built with Typer

## Requirements

- Python 3.8+
- HuggingFace account with access token (for Gemma model)
- Git installed on your system

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kaeldrin-gh/chat-git-repo.git
   cd chat-git-repo
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up HuggingFace token**:
   ```bash
   export HUGGINGFACE_TOKEN=your_huggingface_token_here
   ```

## Usage

### 1. Ingest a Repository

First, process a GitHub repository to create a vector store:

```bash
python -m codechat ingest --repo-url https://github.com/owner/repo --out-store stores/repo_name
```

### 2. Chat with the Code

Start an interactive chat session:

```bash
python -m codechat chat --store stores/repo_name
```

### 3. Get Store Information

View statistics about a processed repository:

```bash
python -m codechat info --store stores/repo_name
```

## Example Chat Session

Here's an example of chatting with an F1 Dashboard repository:

```
💬 Ready to chat! (Type 'quit' to exit)

🤔 Your question: What is the main purpose of this F1 dashboard application?

💡 Answer:
Based on the code context, the F1 dashboard application aims to display real-time data about Formula One races, including driver positions, weather conditions, pit stops, and race control events. It also provides tools for visualizing and analyzing this data.

🤔 Your question: What API endpoints are available in this application?

💡 Answer:
The application uses the following API endpoints:
* `/api/v1/drivers`
* `/api/v1/positions`
* `/api/v1/intervals`
* `/api/v1/carData`
* `/api/v1/locationData`
* `/api/v1/weatherData`
* `/api/v1/raceControl`

🤔 Your question: How does the application fetch F1 race data?

💡 Answer:
The application fetches F1 race data from an external API. It uses a WebSocket connection to receive real-time updates.
```

## Project Structure

```
codechat/
├── ingestion/          # Repository cloning and code chunking
├── embeddings/         # Text embedding generation
├── vectordb/           # FAISS vector store management
├── rag/                # RAG chain and chat functionality
├── interface/          # CLI interface
└── config.py           # Configuration settings
```

## Development

Install development dependencies and run tests:

```bash
pip install -r requirements-dev.txt
pytest --cov=codechat tests/
```

## License

MIT License
