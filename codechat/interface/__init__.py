"""Command-line interface for the codechat system."""

import logging
import sys
from pathlib import Path
from typing import List

import typer
from tqdm import tqdm

from ..config import DEFAULT_STORAGE_DIR
from ..embeddings.embedder import CodeEmbedder
from ..ingestion.chunker import SyntaxAwareChunker
from ..ingestion.git_loader import GitLoader
from ..rag.chain import CodeChat
from ..vectordb.faiss_store import FaissStore

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(help="Chat with your codebase using RAG")


@app.command()
def ingest(
    repo_url: str = typer.Option(..., "--repo-url", "-r", help="GitHub repository URL to ingest"),
    out_store: Path = typer.Option(
        DEFAULT_STORAGE_DIR / "default",
        "--out-store",
        "-o",
        help="Output directory for the vector store",
    ),
    embedding_model: str = typer.Option(
        "BAAI/bge-base-en-v1.5",
        "--embedding-model",
        "-e",
        help="Embedding model to use",
    ),
    batch_size: int = typer.Option(
        32,
        "--batch-size",
        "-b",
        help="Batch size for embedding generation",
    ),
) -> None:
    """Ingest a GitHub repository into a vector store."""
    typer.echo(f"🔄 Starting ingestion of repository: {repo_url}")
    
    try:
        # Initialize components
        git_loader = GitLoader()
        chunker = SyntaxAwareChunker()
        embedder = CodeEmbedder(model_name=embedding_model)
        store = FaissStore(dimension=embedder.embedding_dimension)
        
        # Clone repository
        typer.echo("📥 Cloning repository...")
        repo_path = git_loader.clone_repo(repo_url)
        typer.echo(f"✅ Repository cloned to: {repo_path}")
        
        # Process files and create chunks
        typer.echo("🔍 Processing code files...")
        all_chunks = []
        file_count = 0
        
        for file_path, content in git_loader.iter_code_files(repo_path):
            chunks = chunker.chunk_file(file_path, content)
            all_chunks.extend(chunks)
            file_count += 1
            
            if file_count % 10 == 0:
                typer.echo(f"Processed {file_count} files, {len(all_chunks)} chunks so far...")
        
        typer.echo(f"✅ Processed {file_count} files, created {len(all_chunks)} chunks")
        
        if not all_chunks:
            typer.echo("❌ No code chunks found. Please check the repository URL.")
            sys.exit(1)
        
        # Generate embeddings in batches
        typer.echo("🧠 Generating embeddings...")
        chunk_texts = [str(chunk) for chunk in all_chunks]
        
        all_embeddings = []
        for i in tqdm(range(0, len(chunk_texts), batch_size), desc="Embedding batches"):
            batch_texts = chunk_texts[i:i + batch_size]
            batch_embeddings = embedder.embed_texts(batch_texts)
            all_embeddings.append(batch_embeddings)
        
        # Combine all embeddings
        import numpy as np
        embeddings = np.vstack(all_embeddings)
        
        # Add to store
        typer.echo("💾 Building vector index...")
        store.add_chunks(all_chunks, embeddings)
        
        # Save store
        typer.echo(f"💾 Saving vector store to: {out_store}")
        store.save(out_store)
        
        # Display statistics
        stats = store.get_stats()
        typer.echo("\n📊 Ingestion Statistics:")
        typer.echo(f"  Total chunks: {stats['num_chunks']}")
        typer.echo(f"  Files processed: {stats['num_files']}")
        typer.echo(f"  Languages found: {list(stats['languages'].keys())}")
        typer.echo(f"  Embedding dimension: {stats['dimension']}")
        
        # Cleanup
        git_loader.cleanup()
        
        typer.echo("✅ Ingestion completed successfully!")
        
    except Exception as e:
        typer.echo(f"❌ Error during ingestion: {e}", err=True)
        sys.exit(1)


@app.command()
def chat(
    store: Path = typer.Option(..., "--store", "-s", help="Path to the vector store"),
    model: str = typer.Option(
        "google/gemma-3n-E4B-it",
        "--model",
        "-m",
        help="Language model to use for generation",
    ),
    interactive: bool = typer.Option(
        True,
        "--interactive/--no-interactive",
        "-i/-ni",
        help="Run in interactive mode",
    ),
) -> None:
    """Chat with a codebase using the vector store."""
    typer.echo(f"💬 Loading chat system from store: {store}")
    
    try:
        # Load vector store
        if not store.exists():
            typer.echo(f"❌ Vector store not found: {store}", err=True)
            sys.exit(1)
        
        vector_store = FaissStore.load(store)
        stats = vector_store.get_stats()
        
        typer.echo(f"✅ Loaded vector store with {stats['num_chunks']} chunks")
        typer.echo(f"   Files: {stats['num_files']}")
        typer.echo(f"   Languages: {list(stats['languages'].keys())}")
        
        # Initialize chat system
        typer.echo(f"🤖 Loading language model: {model}")
        code_chat = CodeChat(vector_store, model_name=model)
        
        # Get model info
        model_info = code_chat.get_model_info()
        typer.echo(f"✅ Model loaded on device: {model_info['device']}")
        
        if interactive:
            # Interactive chat loop
            typer.echo("\n💬 Ready to chat! (Type 'quit' to exit)")
            typer.echo("=" * 50)
            
            while True:
                try:
                    question = typer.prompt("\n🤔 Your question")
                    
                    if question.lower() in ["quit", "exit", "q"]:
                        break
                    
                    if not question.strip():
                        continue
                    
                    typer.echo("\n🤖 Thinking...")
                    answer = code_chat.chat(question)
                    
                    typer.echo(f"\n💡 Answer:\n{answer}")
                    typer.echo("-" * 50)
                    
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    typer.echo(f"❌ Error generating answer: {e}", err=True)
            
            typer.echo("\n👋 Goodbye!")
        else:
            # Single question mode (for scripting)
            question = sys.stdin.read().strip()
            if question:
                answer = code_chat.chat(question)
                typer.echo(answer)
            else:
                typer.echo("❌ No question provided via stdin", err=True)
                sys.exit(1)
    
    except Exception as e:
        typer.echo(f"❌ Error in chat system: {e}", err=True)
        sys.exit(1)


@app.command()
def info(
    store: Path = typer.Option(..., "--store", "-s", help="Path to the vector store"),
) -> None:
    """Display information about a vector store."""
    try:
        if not store.exists():
            typer.echo(f"❌ Vector store not found: {store}", err=True)
            sys.exit(1)
        
        vector_store = FaissStore.load(store)
        stats = vector_store.get_stats()
        
        typer.echo(f"\n📊 Vector Store Information: {store}")
        typer.echo("=" * 50)
        typer.echo(f"Total chunks: {stats['num_chunks']}")
        typer.echo(f"Files processed: {stats['num_files']}")
        typer.echo(f"Embedding dimension: {stats['dimension']}")
        
        typer.echo(f"\nLanguage distribution:")
        for lang, count in stats['languages'].items():
            typer.echo(f"  {lang}: {count} chunks")
        
        typer.echo(f"\nChunk type distribution:")
        for chunk_type, count in stats['chunk_types'].items():
            typer.echo(f"  {chunk_type}: {count} chunks")
        
        typer.echo(f"\nTop files by chunk count:")
        for file_path, count in stats['top_files'][:5]:
            typer.echo(f"  {file_path}: {count} chunks")
        
    except Exception as e:
        typer.echo(f"❌ Error loading store info: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    app()
