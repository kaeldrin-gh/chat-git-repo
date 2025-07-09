"""Tests for the configuration module."""

import os
from pathlib import Path

import pytest

from codechat.config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    DEFAULT_PROMPT_TEMPLATE,
    DEFAULT_STORAGE_DIR,
    MAX_CHUNK_SIZE,
    SUPPORTED_EXTENSIONS,
    get_hf_token,
)


class TestConfig:
    """Test cases for configuration module."""

    def test_default_values(self):
        """Test that default configuration values are set correctly."""
        assert DEFAULT_EMBEDDING_MODEL == "BAAI/bge-base-en-v1.5"
        assert DEFAULT_LLM_MODEL == "google/gemma-2-2b-it"
        assert MAX_CHUNK_SIZE == 2048
        assert isinstance(SUPPORTED_EXTENSIONS, list)
        assert len(SUPPORTED_EXTENSIONS) > 0

    def test_supported_extensions(self):
        """Test that supported extensions include common programming languages."""
        assert ".py" in SUPPORTED_EXTENSIONS
        assert ".js" in SUPPORTED_EXTENSIONS
        assert ".ts" in SUPPORTED_EXTENSIONS
        assert ".java" in SUPPORTED_EXTENSIONS
        assert ".go" in SUPPORTED_EXTENSIONS

    def test_default_storage_dir(self):
        """Test default storage directory configuration."""
        assert isinstance(DEFAULT_STORAGE_DIR, Path)
        assert DEFAULT_STORAGE_DIR.name == "stores"

    def test_default_prompt_template(self):
        """Test default prompt template."""
        assert isinstance(DEFAULT_PROMPT_TEMPLATE, str)
        assert "{context}" in DEFAULT_PROMPT_TEMPLATE
        assert "{question}" in DEFAULT_PROMPT_TEMPLATE

    def test_get_hf_token_none(self):
        """Test get_hf_token returns None when env var not set."""
        # Ensure env var is not set
        if "HUGGINGFACE_TOKEN" in os.environ:
            del os.environ["HUGGINGFACE_TOKEN"]
        
        token = get_hf_token()
        assert token is None

    def test_get_hf_token_set(self):
        """Test get_hf_token returns token when env var is set."""
        test_token = "test_token_123"
        os.environ["HUGGINGFACE_TOKEN"] = test_token
        
        try:
            token = get_hf_token()
            assert token == test_token
        finally:
            # Clean up
            del os.environ["HUGGINGFACE_TOKEN"]

    def test_chunk_size_positive(self):
        """Test that chunk size is positive."""
        assert MAX_CHUNK_SIZE > 0

    def test_config_types(self):
        """Test that configuration values have correct types."""
        assert isinstance(DEFAULT_EMBEDDING_MODEL, str)
        assert isinstance(DEFAULT_LLM_MODEL, str)
        assert isinstance(MAX_CHUNK_SIZE, int)
        assert isinstance(SUPPORTED_EXTENSIONS, list)
        assert isinstance(DEFAULT_PROMPT_TEMPLATE, str)
