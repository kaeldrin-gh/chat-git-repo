"""Tests for the config module."""

import os
from pathlib import Path

import pytest

from codechat import config


class TestConfig:
    """Test cases for configuration settings."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        assert config.DEFAULT_STORAGE_DIR == Path("./stores")
        assert config.TEMP_CLONE_DIR == Path("./temp_repos")
        assert config.RANDOM_SEED == 42
        assert config.NUMPY_SEED == 42
        assert config.DEFAULT_EMBEDDING_MODEL == "BAAI/bge-base-en-v1.5"
        assert config.DEFAULT_LLM_MODEL == "google/gemma-2-2b-it"

    def test_chunk_settings(self):
        """Test chunk-related configuration."""
        assert config.MAX_CHUNK_SIZE == 2048
        assert config.CHUNK_OVERLAP == 200
        assert isinstance(config.MAX_CHUNK_SIZE, int)
        assert isinstance(config.CHUNK_OVERLAP, int)
        assert config.CHUNK_OVERLAP < config.MAX_CHUNK_SIZE

    def test_supported_extensions(self):
        """Test supported file extensions."""
        expected_extensions = [".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h"]
        assert config.SUPPORTED_EXTENSIONS == expected_extensions
        assert all(ext.startswith(".") for ext in config.SUPPORTED_EXTENSIONS)

    def test_faiss_settings(self):
        """Test FAISS-related configuration."""
        assert config.FAISS_INDEX_TYPE == "IndexFlatIP"
        assert config.EMBEDDING_DIMENSION == 768
        assert isinstance(config.EMBEDDING_DIMENSION, int)
        assert config.EMBEDDING_DIMENSION > 0

    def test_generation_settings(self):
        """Test generation-related settings."""
        assert config.MAX_CONTEXT_LENGTH == 4096
        assert config.MAX_GENERATION_LENGTH == 512
        assert config.TEMPERATURE == 0.1
        assert config.TOP_K == 5
        
        assert 0.0 <= config.TEMPERATURE <= 1.0
        assert config.TOP_K > 0
        assert config.MAX_GENERATION_LENGTH > 0
        assert config.MAX_CONTEXT_LENGTH > config.MAX_GENERATION_LENGTH

    def test_prompt_template(self):
        """Test default prompt template."""
        template = config.DEFAULT_PROMPT_TEMPLATE
        assert isinstance(template, str)
        assert "{context}" in template
        assert "{question}" in template
        assert "code context" in template.lower()

    def test_get_hf_token(self, monkeypatch):
        """Test HuggingFace token retrieval."""
        # Test when token is not set
        monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)
        assert config.get_hf_token() is None
        
        # Test when token is set
        test_token = "test_token_123"
        monkeypatch.setenv("HUGGINGFACE_TOKEN", test_token)
        assert config.get_hf_token() == test_token

    def test_gpu_settings(self):
        """Test GPU-related settings."""
        assert isinstance(config.USE_GPU, bool)
        assert config.DEVICE_MAP == "auto"
        assert config.TORCH_DTYPE == "float16"

    def test_paths_are_pathlib(self):
        """Test that path configurations are Path objects."""
        assert isinstance(config.DEFAULT_STORAGE_DIR, Path)
        assert isinstance(config.TEMP_CLONE_DIR, Path)
