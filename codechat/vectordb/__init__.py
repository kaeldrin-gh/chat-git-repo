"""FAISS-based vector store for code embeddings."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np

from ..config import EMBEDDING_DIMENSION, FAISS_INDEX_TYPE
from ..ingestion.chunker import CodeChunk


class FaissStore:
    """FAISS-based vector store for storing and querying code embeddings."""

    def __init__(self, dimension: int = EMBEDDING_DIMENSION) -> None:
        """Initialize the FAISS store.
        
        Args:
            dimension: Embedding dimension for the index.
        """
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.chunks: List[CodeChunk] = []
        self.metadata: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[CodeChunk], embeddings: np.ndarray) -> None:
        """Add code chunks and their embeddings to the store.
        
        Args:
            chunks: List of CodeChunk objects.
            embeddings: NumPy array of embeddings with shape (n_chunks, dimension).
        """
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match number of embeddings")
        
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Ensure embeddings are normalized for cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Add to FAISS index
        self.index.add(embeddings.astype(np.float32))
        
        # Store chunks and metadata
        self.chunks.extend(chunks)
        for chunk in chunks:
            self.metadata.append({
                "file_path": str(chunk.file_path),
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "language": chunk.language,
                "chunk_type": chunk.chunk_type,
            })

    def query(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[CodeChunk, float]]:
        """Query the store for similar code chunks.
        
        Args:
            query_embedding: Query embedding vector.
            k: Number of results to return.
            
        Returns:
            List of tuples (CodeChunk, similarity_score) sorted by relevance.
        """
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Query embedding dimension {query_embedding.shape[1]} doesn't match index dimension {self.dimension}")
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding.astype(np.float32), min(k, self.index.ntotal))
        
        # Return results
        results: List[Tuple[CodeChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1:  # Valid result
                results.append((self.chunks[idx], float(score)))
        
        return results

    def save(self, store_path: Path) -> None:
        """Save the store to disk.
        
        Args:
            store_path: Path to save the store.
        """
        store_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(store_path / "index.faiss"))
        
        # Save chunks
        with open(store_path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        
        # Save metadata
        with open(store_path / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        
        # Save store info
        store_info = {
            "dimension": self.dimension,
            "index_type": FAISS_INDEX_TYPE,
            "num_chunks": len(self.chunks),
        }
        with open(store_path / "info.json", "w") as f:
            json.dump(store_info, f, indent=2)

    @classmethod
    def load(cls, store_path: Path) -> "FaissStore":
        """Load a store from disk.
        
        Args:
            store_path: Path to the saved store.
            
        Returns:
            Loaded FaissStore instance.
        """
        # Load store info
        with open(store_path / "info.json", "r") as f:
            store_info = json.load(f)
        
        # Create store instance
        store = cls(dimension=store_info["dimension"])
        
        # Load FAISS index
        store.index = faiss.read_index(str(store_path / "index.faiss"))
        
        # Load chunks
        with open(store_path / "chunks.pkl", "rb") as f:
            store.chunks = pickle.load(f)
        
        # Load metadata
        with open(store_path / "metadata.json", "r") as f:
            store.metadata = json.load(f)
        
        return store

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the store.
        
        Returns:
            Dictionary with store statistics.
        """
        if not self.chunks:
            return {"num_chunks": 0, "dimension": self.dimension}
        
        # Language distribution
        languages = {}
        chunk_types = {}
        file_counts = {}
        
        for chunk in self.chunks:
            languages[chunk.language] = languages.get(chunk.language, 0) + 1
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            file_path = str(chunk.file_path)
            file_counts[file_path] = file_counts.get(file_path, 0) + 1
        
        return {
            "num_chunks": len(self.chunks),
            "dimension": self.dimension,
            "languages": languages,
            "chunk_types": chunk_types,
            "num_files": len(file_counts),
            "top_files": sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10],
        }
