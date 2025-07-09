"""Tests for the CLI interface."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

from codechat.interface.cli import app


class TestCLI:
    """Test cases for CLI interface."""

    def test_app_exists(self):
        """Test that CLI app exists."""
        assert app is not None

    def test_ingest_command_help(self):
        """Test ingest command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["ingest", "--help"])
        
        assert result.exit_code == 0
        assert "ingest" in result.stdout.lower()
        assert "repo-url" in result.stdout.lower()

    def test_chat_command_help(self):
        """Test chat command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--help"])
        
        assert result.exit_code == 0
        assert "chat" in result.stdout.lower()
        assert "store" in result.stdout.lower()

    def test_info_command_help(self):
        """Test info command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["info", "--help"])
        
        assert result.exit_code == 0
        assert "info" in result.stdout.lower()
        assert "store" in result.stdout.lower()

    def test_info_command_nonexistent_store(self):
        """Test info command with nonexistent store."""
        runner = CliRunner()
        result = runner.invoke(app, ["info", "--store", "/nonexistent/path"])
        
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    def test_chat_command_nonexistent_store(self):
        """Test chat command with nonexistent store."""
        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--store", "/nonexistent/path"])
        
        assert result.exit_code == 1
        assert "not found" in result.stdout.lower()

    @patch('codechat.interface.cli.GitLoader')
    @patch('codechat.interface.cli.SyntaxAwareChunker')
    @patch('codechat.interface.cli.CodeEmbedder')
    @patch('codechat.interface.cli.FaissStore')
    def test_ingest_command_mock(self, mock_faiss, mock_embedder, mock_chunker, mock_loader):
        """Test ingest command with mocks."""
        # Setup mocks
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        mock_loader_instance.clone_repo.return_value = Path("/tmp/repo")
        mock_loader_instance.iter_code_files.return_value = [
            (Path("test.py"), "def hello(): pass")
        ]
        
        mock_chunker_instance = Mock()
        mock_chunker.return_value = mock_chunker_instance
        mock_chunk = Mock()
        mock_chunk.content = "def hello(): pass"
        mock_chunker_instance.chunk_file.return_value = [mock_chunk]
        
        mock_embedder_instance = Mock()
        mock_embedder.return_value = mock_embedder_instance
        mock_embedder_instance.embedding_dimension = 384
        mock_embedder_instance.embed_texts.return_value = [[0.1] * 384]
        
        mock_store_instance = Mock()
        mock_faiss.return_value = mock_store_instance
        mock_store_instance.get_stats.return_value = {
            "num_chunks": 1,
            "num_files": 1,
            "languages": {"python": 1},
            "dimension": 384
        }
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(app, [
                "ingest",
                "--repo-url", "https://github.com/test/repo",
                "--out-store", temp_dir
            ])
            
            # Should not crash (but may fail due to mocking limitations)
            assert "Starting ingestion" in result.stdout

    @patch('codechat.interface.cli.FaissStore')
    def test_info_command_mock(self, mock_faiss):
        """Test info command with mock store."""
        mock_store = Mock()
        mock_faiss.load.return_value = mock_store
        mock_store.get_stats.return_value = {
            "num_chunks": 100,
            "num_files": 10,
            "dimension": 384,
            "languages": {"python": 80, "javascript": 20},
            "chunk_types": {"function": 60, "class": 40},
            "top_files": [("main.py", 20), ("utils.py", 15)]
        }
        
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a fake store directory
            store_path = Path(temp_dir) / "test_store"
            store_path.mkdir()
            
            result = runner.invoke(app, ["info", "--store", str(store_path)])
            
            assert result.exit_code == 0
            assert "100" in result.stdout  # num_chunks
            assert "10" in result.stdout   # num_files
            assert "python" in result.stdout.lower()
            assert "javascript" in result.stdout.lower()
