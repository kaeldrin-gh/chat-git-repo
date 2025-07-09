"""Tests for the embedder module."""

import numpy as np
import pytest

from codechat.embeddings.embedder import CodeEmbedder


class TestCodeEmbedder:
    """Test cases for CodeEmbedder class."""

    @pytest.fixture
    def embedder(self):
        """Create a CodeEmbedder instance for testing."""
        # Use a well-known public model for testing  
        return CodeEmbedder(model_name="all-MiniLM-L6-v2")

    def test_init(self, embedder):
        """Test CodeEmbedder initialization."""
        assert embedder.model_name == "all-MiniLM-L6-v2"
        assert embedder.device in ["cuda", "cpu"]
        assert embedder.model is not None

    def test_embedding_dimension(self, embedder):
        """Test embedding dimension property."""
        dim = embedder.embedding_dimension
        assert isinstance(dim, int)
        assert dim > 0
        # all-MiniLM-L6-v2 has 384 dimensions
        assert dim == 384

    def test_embed_single(self, embedder):
        """Test embedding a single text."""
        text = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
        
        embedding = embedder.embed_single(text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.embedding_dimension,)
        assert not np.isnan(embedding).any()
        assert not np.isinf(embedding).any()
        
        # Check if normalized (should be close to 1.0 for normalized vectors)
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-6

    def test_embed_texts(self, embedder):
        """Test embedding multiple texts."""
        texts = [
            "def hello(): return 'world'",
            "class Calculator: pass",
            "import os",
            "function add(a, b) { return a + b; }"
        ]
        
        embeddings = embedder.embed_texts(texts)
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (len(texts), embedder.embedding_dimension)
        assert not np.isnan(embeddings).any()
        assert not np.isinf(embeddings).any()
        
        # Check if normalized
        norms = np.linalg.norm(embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-6)

    def test_embed_empty_list(self, embedder):
        """Test embedding empty list."""
        embeddings = embedder.embed_texts([])
        
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (0,)

    def test_embed_single_empty_text(self, embedder):
        """Test embedding empty text."""
        embedding = embedder.embed_single("")
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.embedding_dimension,)

    def test_embedding_consistency(self, embedder):
        """Test that same text produces same embedding."""
        text = "def test_function(): pass"
        
        embedding1 = embedder.embed_single(text)
        embedding2 = embedder.embed_single(text)
        
        np.testing.assert_array_almost_equal(embedding1, embedding2, decimal=6)

    def test_embedding_similarity(self, embedder):
        """Test that similar texts have similar embeddings."""
        text1 = "def calculate_sum(a, b): return a + b"
        text2 = "def add_numbers(x, y): return x + y"
        text3 = "class DatabaseConnection: pass"
        
        emb1 = embedder.embed_single(text1)
        emb2 = embedder.embed_single(text2)
        emb3 = embedder.embed_single(text3)
        
        # Similar functions should be more similar than function vs class
        similarity_12 = np.dot(emb1, emb2)
        similarity_13 = np.dot(emb1, emb3)
        
        assert similarity_12 > similarity_13

    def test_get_model_info(self, embedder):
        """Test model info retrieval."""
        info = embedder.get_model_info()
        
        assert isinstance(info, dict)
        assert "model_name" in info
        assert "device" in info
        assert "embedding_dimension" in info
        assert "max_sequence_length" in info
        
        assert info["model_name"] == embedder.model_name
        assert info["embedding_dimension"] == embedder.embedding_dimension

    def test_batch_processing(self, embedder):
        """Test that batch processing gives same results as individual processing."""
        texts = [
            "def func1(): pass",
            "def func2(): return True",
            "class MyClass: pass"
        ]
        
        # Process individually
        individual_embeddings = np.array([embedder.embed_single(text) for text in texts])
        
        # Process as batch
        batch_embeddings = embedder.embed_texts(texts)
        
        np.testing.assert_array_almost_equal(individual_embeddings, batch_embeddings, decimal=6)

    def test_long_text_handling(self, embedder):
        """Test handling of very long texts."""
        # Create a very long text
        long_text = " ".join([f"def function_{i}(): pass" for i in range(1000)])
        
        # Should not crash, even if truncated
        embedding = embedder.embed_single(long_text)
        
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape == (embedder.embedding_dimension,)
        assert not np.isnan(embedding).any()
