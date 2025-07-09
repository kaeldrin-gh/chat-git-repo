"""Tests for the FAISS store module."""

import numpy as np
import pytest
from pathlib import Path

from codechat.ingestion.chunker import CodeChunk
from codechat.vectordb.faiss_store import FaissStore


class TestFaissStore:
    """Test cases for FaissStore class."""

    def test_init(self):
        """Test FaissStore initialization."""
        store = FaissStore(dimension=384)
        
        assert store.dimension == 384
        assert store.index is not None
        assert len(store.chunks) == 0
        assert len(store.metadata) == 0

    def test_add_chunks(self):
        """Test adding chunks to the store."""
        store = FaissStore(dimension=3)
        
        # Create sample chunks
        chunks = [
            CodeChunk("def func1(): pass", Path("test1.py"), 1, 1, "python", "function"),
            CodeChunk("def func2(): pass", Path("test2.py"), 1, 1, "python", "function"),
        ]
        
        # Create sample embeddings
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        
        store.add_chunks(chunks, embeddings)
        
        assert len(store.chunks) == 2
        assert len(store.metadata) == 2
        assert store.index.ntotal == 2

    def test_add_chunks_dimension_mismatch(self):
        """Test error when embedding dimension doesn't match."""
        store = FaissStore(dimension=3)
        
        chunks = [CodeChunk("def func(): pass", Path("test.py"), 1, 1, "python", "function")]
        embeddings = np.array([[1.0, 0.0]], dtype=np.float32)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Embedding dimension"):
            store.add_chunks(chunks, embeddings)

    def test_add_chunks_count_mismatch(self):
        """Test error when chunk count doesn't match embedding count."""
        store = FaissStore(dimension=3)
        
        chunks = [CodeChunk("def func(): pass", Path("test.py"), 1, 1, "python", "function")]
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)  # Too many embeddings
        
        with pytest.raises(ValueError, match="Number of chunks .* must match"):
            store.add_chunks(chunks, embeddings)

    def test_query(self):
        """Test querying the store."""
        store = FaissStore(dimension=3)
        
        # Add some chunks
        chunks = [
            CodeChunk("def add(a, b): return a + b", Path("math.py"), 1, 1, "python", "function"),
            CodeChunk("def multiply(x, y): return x * y", Path("math.py"), 3, 3, "python", "function"),
            CodeChunk("class Calculator: pass", Path("calc.py"), 1, 1, "python", "class"),
        ]
        
        embeddings = np.array([
            [1.0, 0.0, 0.0],  # add function
            [0.8, 0.6, 0.0],  # multiply function (similar to add)
            [0.0, 0.0, 1.0],  # class (different)
        ], dtype=np.float32)
        
        store.add_chunks(chunks, embeddings)
        
        # Query with vector similar to add function
        query_embedding = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        results = store.query(query_embedding, k=2)
        
        assert len(results) == 2
        assert all(isinstance(result, tuple) for result in results)
        assert all(len(result) == 2 for result in results)  # (chunk, score) pairs
        
        # First result should be most similar (add function)
        best_chunk, best_score = results[0]
        assert "add" in best_chunk.content
        assert best_score > 0.8  # High similarity

    def test_query_dimension_mismatch(self):
        """Test error when query dimension doesn't match."""
        store = FaissStore(dimension=3)
        
        # Add a chunk
        chunks = [CodeChunk("def func(): pass", Path("test.py"), 1, 1, "python", "function")]
        embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        store.add_chunks(chunks, embeddings)
        
        # Query with wrong dimension
        query_embedding = np.array([1.0, 0.0], dtype=np.float32)  # Wrong dimension
        
        with pytest.raises(ValueError, match="Query embedding dimension"):
            store.query(query_embedding, k=1)

    def test_query_empty_store(self):
        """Test querying an empty store."""
        store = FaissStore(dimension=3)
        
        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query_embedding, k=5)
        
        assert len(results) == 0

    def test_save_and_load(self, temp_dir):
        """Test saving and loading the store."""
        # Create and populate store
        store = FaissStore(dimension=3)
        chunks = [
            CodeChunk("def func1(): pass", Path("test1.py"), 1, 1, "python", "function"),
            CodeChunk("def func2(): pass", Path("test2.py"), 1, 1, "python", "function"),
        ]
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        store.add_chunks(chunks, embeddings)
        
        # Save store
        store_path = temp_dir / "test_store"
        store.save(store_path)
        
        # Check files exist
        assert (store_path / "index.faiss").exists()
        assert (store_path / "chunks.pkl").exists()
        assert (store_path / "metadata.json").exists()
        assert (store_path / "info.json").exists()
        
        # Load store
        loaded_store = FaissStore.load(store_path)
        
        assert loaded_store.dimension == store.dimension
        assert len(loaded_store.chunks) == len(store.chunks)
        assert len(loaded_store.metadata) == len(store.metadata)
        assert loaded_store.index.ntotal == store.index.ntotal
        
        # Test that loaded store works
        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = loaded_store.query(query_embedding, k=1)
        assert len(results) == 1

    def test_get_stats_empty(self):
        """Test statistics for empty store."""
        store = FaissStore(dimension=3)
        stats = store.get_stats()
        
        assert stats["num_chunks"] == 0
        assert stats["dimension"] == 3

    def test_get_stats_populated(self):
        """Test statistics for populated store."""
        store = FaissStore(dimension=3)
        
        chunks = [
            CodeChunk("def func1(): pass", Path("test1.py"), 1, 1, "python", "function"),
            CodeChunk("def func2(): pass", Path("test1.py"), 3, 3, "python", "function"),
            CodeChunk("class Test: pass", Path("test2.js"), 1, 1, "javascript", "class"),
        ]
        embeddings = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        store.add_chunks(chunks, embeddings)
        
        stats = store.get_stats()
        
        assert stats["num_chunks"] == 3
        assert stats["dimension"] == 3
        assert stats["num_files"] == 2
        assert "python" in stats["languages"]
        assert "javascript" in stats["languages"]
        assert stats["languages"]["python"] == 2
        assert stats["languages"]["javascript"] == 1
        assert "function" in stats["chunk_types"]
        assert "class" in stats["chunk_types"]

    def test_normalization(self):
        """Test that embeddings are properly normalized."""
        store = FaissStore(dimension=3)
        
        # Create unnormalized embeddings
        chunks = [CodeChunk("def func(): pass", Path("test.py"), 1, 1, "python", "function")]
        embeddings = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)  # Length = 5
        
        store.add_chunks(chunks, embeddings)
        
        # Query should work with normalized vectors
        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query_embedding, k=1)
        
        assert len(results) == 1
        chunk, score = results[0]
        # Score should be reasonable (not affected by original magnitude)
        assert 0.0 <= score <= 1.0

    def test_query_k_parameter(self):
        """Test k parameter in query."""
        store = FaissStore(dimension=3)
        
        # Add multiple chunks
        chunks = [
            CodeChunk(f"def func{i}(): pass", Path("test.py"), i, i, "python", "function")
            for i in range(5)
        ]
        embeddings = np.random.rand(5, 3).astype(np.float32)
        store.add_chunks(chunks, embeddings)
        
        query_embedding = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # Test different k values
        results_1 = store.query(query_embedding, k=1)
        results_3 = store.query(query_embedding, k=3)
        results_10 = store.query(query_embedding, k=10)  # More than available
        
        assert len(results_1) == 1
        assert len(results_3) == 3
        assert len(results_10) == 5  # Should return all available
