"""Configuration settings for the codechat system."""

import os
from pathlib import Path
from typing import Dict, List

# Paths
DEFAULT_STORAGE_DIR: Path = Path("./stores")
TEMP_CLONE_DIR: Path = Path("./temp_repos")

# Random seeds for reproducibility
RANDOM_SEED: int = 42
NUMPY_SEED: int = 42

# Model settings
DEFAULT_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
DEFAULT_LLM_MODEL: str = "google/gemma-2-2b-it"

# Chunking settings
MAX_CHUNK_SIZE: int = 2048
CHUNK_OVERLAP: int = 200

# File extensions to process
SUPPORTED_EXTENSIONS: List[str] = [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h"]

# FAISS settings
FAISS_INDEX_TYPE: str = "IndexFlatIP"  # Inner product for cosine similarity
EMBEDDING_DIMENSION: int = 768  # BGE base model dimension

# Generation settings
MAX_CONTEXT_LENGTH: int = 4096
MAX_GENERATION_LENGTH: int = 512
TEMPERATURE: float = 0.1
TOP_K: int = 5  # Number of chunks to retrieve

# Default prompt template
DEFAULT_PROMPT_TEMPLATE: str = """You are a helpful coding assistant. Answer the user's question using only the provided code context. If the context doesn't contain enough information to answer the question, respond with "I don't know based on the provided code context."

Code Context:
{context}

Question: {question}

Answer:"""

# Environment variables
def get_hf_token() -> str | None:
    """Get HuggingFace token from environment."""
    return os.getenv("HUGGINGFACE_TOKEN")


# GPU settings
USE_GPU: bool = True  # Will fallback to CPU if not available
DEVICE_MAP: str = "auto"
TORCH_DTYPE: str = "float16"  # Use float16 for memory efficiency
