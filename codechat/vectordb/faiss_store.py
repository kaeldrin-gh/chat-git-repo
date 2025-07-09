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
            dimension: Dimension of the embedding vectors.
        """
        self.dimension = dimension
        
        # Create FAISS index
        if FAISS_INDEX_TYPE == "IndexFlatIP":
            self.index = faiss.IndexFlatIP(dimension)
        elif FAISS_INDEX_TYPE == "IndexFlatL2":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            # Default to inner product (cosine similarity with normalized vectors)
            self.index = faiss.IndexFlatIP(dimension)
        
        # Storage for chunks and metadata
        self.chunks: List[CodeChunk] = []
        self.metadata: List[Dict[str, Any]] = []

    def add_chunks(self, chunks: List[CodeChunk], embeddings: np.ndarray) -> None:
        """Add code chunks with their embeddings to the store.
        
        Args:
            chunks: List of code chunks to add.
            embeddings: Corresponding embeddings for the chunks.
            
        Raises:
            ValueError: If embedding dimensions don't match or counts don't match.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"Number of chunks ({len(chunks)}) must match number of embeddings ({len(embeddings)})")
        
        if len(embeddings) > 0 and embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension ({embeddings.shape[1]}) must match store dimension ({self.dimension})")
        
        # Normalize embeddings for cosine similarity
        if len(embeddings) > 0:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            embeddings = embeddings / norms
        
        # Add to FAISS index
        if len(embeddings) > 0:
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
            List of (chunk, similarity_score) tuples, sorted by similarity.
            
        Raises:
            ValueError: If query embedding dimension doesn't match store dimension.
        """
        if query_embedding.shape[0] != self.dimension:
            raise ValueError(f"Query embedding dimension ({query_embedding.shape[0]}) must match store dimension ({self.dimension})")
        
        # Return empty results if no chunks
        if len(self.chunks) == 0:
            return []
        
        # Normalize query embedding
        query_norm = np.linalg.norm(query_embedding)
        if query_norm > 0:
            query_embedding = query_embedding / query_norm
        
        # Search in FAISS index
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        
        # Build results
        results: List[Tuple[CodeChunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append((self.chunks[idx], float(score)))
        
        return results

    def save(self, store_path: Path | str) -> None:
        """Save the store to disk.
        
        Args:
            store_path: Path to save the store.
        """
        store_path = Path(store_path)
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
            "num_chunks": len(self.chunks),
            "index_type": FAISS_INDEX_TYPE,
        }
        with open(store_path / "info.json", "w") as f:
            json.dump(store_info, f, indent=2)

    @classmethod
    def load(cls, store_path: Path | str) -> "FaissStore":
        """Load a FaissStore from disk.
        
        Args:
            store_path: Path to the saved store.
            
        Returns:
            Loaded FaissStore instance.
        """
        store_path = Path(store_path)
        
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
        stats = {
            "num_chunks": len(self.chunks),
            "dimension": self.dimension,
        }
        
        if len(self.chunks) > 0:
            # File statistics
            files = set(chunk.file_path for chunk in self.chunks)
            stats["num_files"] = len(files)
            
            # Language statistics
            languages = {}
            for chunk in self.chunks:
                languages[chunk.language] = languages.get(chunk.language, 0) + 1
            stats["languages"] = languages
            
            # Chunk type statistics
            chunk_types = {}
            for chunk in self.chunks:
                chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
            stats["chunk_types"] = chunk_types
        
        return stats
