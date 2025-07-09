"""Integration tests for the full RAG pipeline."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from codechat.embeddings.embedder import CodeEmbedder
from codechat.ingestion.chunker import CodeChunk, SyntaxAwareChunker
from codechat.ingestion.git_loader import GitLoader
from codechat.vectordb.faiss_store import FaissStore


@pytest.mark.integration
class TestRAGPipeline:
    """Integration tests for the complete RAG pipeline."""

    def test_end_to_end_pipeline(self, temp_dir, sample_python_code):
        """Test the complete pipeline from code to embeddings."""
        # Create a mock repository
        repo_dir = temp_dir / "test_repo"
        repo_dir.mkdir()
        
        # Create test files
        (repo_dir / "main.py").write_text(sample_python_code)
        (repo_dir / "utils.py").write_text("""
def helper_function():
    return "helper"

class Utility:
    def process(self, data):
        return data.upper()
""")
        
        # Initialize components
        loader = GitLoader(temp_dir=temp_dir)
        chunker = SyntaxAwareChunker()
        embedder = CodeEmbedder(model_name="all-MiniLM-L6-v2")  # Smaller model for testing
        store = FaissStore(dimension=embedder.embedding_dimension)
        
        # Process repository
        all_chunks = []
        for file_path, content in loader.iter_code_files(repo_dir):
            chunks = chunker.chunk_file(file_path, content)
            all_chunks.extend(chunks)
        
        assert len(all_chunks) > 0
        
        # Generate embeddings
        chunk_texts = [str(chunk) for chunk in all_chunks]
        embeddings = embedder.embed_texts(chunk_texts)
        
        # Add to store
        store.add_chunks(all_chunks, embeddings)
        
        # Test querying
        query_embedding = embedder.embed_single("fibonacci function")
        results = store.query(query_embedding, k=3)
        
        assert len(results) > 0
        
        # Should find relevant code
        found_fibonacci = any("fibonacci" in chunk.content.lower() for chunk, _ in results)
        assert found_fibonacci
        
        # Test save/load
        store_path = temp_dir / "test_store"
        store.save(store_path)
        
        loaded_store = FaissStore.load(store_path)
        loaded_results = loaded_store.query(query_embedding, k=3)
        
        assert len(loaded_results) == len(results)

    def test_chunking_strategies(self, temp_dir):
        """Test different chunking strategies on various code types."""
        test_codes = {
            "python_functions.py": """
def function_one():
    '''First function'''
    return 1

def function_two():
    '''Second function'''
    return 2

class TestClass:
    def method_one(self):
        return "method1"
    
    def method_two(self):
        return "method2"
""",
            "javascript_module.js": """
function moduleFunction() {
    return "module";
}

class ModuleClass {
    constructor() {
        this.value = 0;
    }
    
    getValue() {
        return this.value;
    }
}

export { moduleFunction, ModuleClass };
""",
            "large_file.py": "\n".join([f"def function_{i}():\n    return {i}" for i in range(100)])
        }
        
        chunker = SyntaxAwareChunker(max_chunk_size=500, chunk_overlap=50)
        
        for filename, code in test_codes.items():
            file_path = Path(filename)
            chunks = chunker.chunk_file(file_path, code)
            
            assert len(chunks) > 0
            
            # Check that chunks contain expected content
            all_content = "\n".join(chunk.content for chunk in chunks)
            assert len(all_content) >= len(code) * 0.8  # Most content should be preserved
            
            # For large files, should create multiple chunks
            if filename == "large_file.py":
                assert len(chunks) > 1

    def test_embedding_quality(self):
        """Test embedding quality and similarity."""
        embedder = CodeEmbedder(model_name="all-MiniLM-L6-v2")
        
        # Related code snippets
        similar_codes = [
            "def add(a, b): return a + b",
            "def sum_numbers(x, y): return x + y",
            "def calculate_sum(first, second): return first + second"
        ]
        
        # Unrelated code
        different_code = "class DatabaseConnection: pass"
        
        # Get embeddings
        similar_embeddings = embedder.embed_texts(similar_codes)
        different_embedding = embedder.embed_single(different_code)
        
        # Calculate similarities
        similarities_within = []
        for i in range(len(similar_embeddings)):
            for j in range(i + 1, len(similar_embeddings)):
                sim = np.dot(similar_embeddings[i], similar_embeddings[j])
                similarities_within.append(sim)
        
        similarities_cross = []
        for emb in similar_embeddings:
            sim = np.dot(emb, different_embedding)
            similarities_cross.append(sim)
        
        # Similar codes should be more similar to each other than to different code
        avg_within = np.mean(similarities_within)
        avg_cross = np.mean(similarities_cross)
        
        assert avg_within > avg_cross
        assert avg_within > 0.65  # Should be quite similar
        assert avg_cross < 0.6   # Should be less similar

    def test_store_scalability(self, temp_dir):
        """Test store performance with larger number of chunks."""
        embedder = CodeEmbedder(model_name="all-MiniLM-L6-v2")
        store = FaissStore(dimension=embedder.embedding_dimension)
        
        # Create many dummy chunks
        chunks = []
        texts = []
        for i in range(100):
            chunk_content = f"def function_{i}(): return {i}"
            chunk = CodeChunk(
                content=chunk_content,
                file_path=Path(f"file_{i % 10}.py"),
                start_line=i,
                end_line=i,
                language="python",
                chunk_type="function"
            )
            chunks.append(chunk)
            texts.append(str(chunk))
        
        # Generate embeddings
        embeddings = embedder.embed_texts(texts)
        
        # Add to store
        store.add_chunks(chunks, embeddings)
        
        # Test querying
        query_embedding = embedder.embed_single("def function_50(): return 50")
        results = store.query(query_embedding, k=10)
        
        assert len(results) == 10
        
        # First result should be function_50 or very similar
        best_chunk, best_score = results[0]
        assert "function_" in best_chunk.content
        assert best_score > 0.65

    def test_multilingual_support(self):
        """Test support for multiple programming languages."""
        chunker = SyntaxAwareChunker()
        embedder = CodeEmbedder(model_name="all-MiniLM-L6-v2")
        
        code_samples = {
            "test.py": "def hello(): return 'Python'",
            "test.js": "function hello() { return 'JavaScript'; }",
            "test.java": "public String hello() { return \"Java\"; }",
            "test.go": "func hello() string { return \"Go\" }",
        }
        
        all_chunks = []
        for filename, code in code_samples.items():
            file_path = Path(filename)
            chunks = chunker.chunk_file(file_path, code)
            all_chunks.extend(chunks)
        
        # Should detect different languages
        languages = {chunk.language for chunk in all_chunks}
        assert len(languages) == 4
        assert "python" in languages
        assert "javascript" in languages
        assert "java" in languages
        assert "go" in languages
        
        # Should be able to embed all chunks
        texts = [str(chunk) for chunk in all_chunks]
        embeddings = embedder.embed_texts(texts)
        
        assert embeddings.shape[0] == len(all_chunks)
        assert not np.isnan(embeddings).any()
