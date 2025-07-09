"""Tests for the CLI interface."""

import subprocess
import tempfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from codechat.interface.cli import app


class TestCLI:
    """Test cases for the CLI interface."""

    def test_help_command(self):
        """Test the help command."""
        runner = CliRunner()
        result = runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "Chat with your codebase using RAG" in result.stdout

    def test_ingest_help(self):
        """Test the ingest command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["ingest", "--help"])
        
        assert result.exit_code == 0
        assert "Ingest a GitHub repository" in result.stdout
        assert "--repo-url" in result.stdout

    def test_chat_help(self):
        """Test the chat command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--help"])
        
        assert result.exit_code == 0
        assert "Chat with a codebase" in result.stdout
        assert "--store" in result.stdout

    def test_info_help(self):
        """Test the info command help."""
        runner = CliRunner()
        result = runner.invoke(app, ["info", "--help"])
        
        assert result.exit_code == 0
        assert "Display information" in result.stdout

    def test_info_nonexistent_store(self):
        """Test info command with non-existent store."""
        runner = CliRunner()
        result = runner.invoke(app, ["info", "--store", "/nonexistent/path"])
        
        assert result.exit_code == 1
        assert "Vector store not found" in result.stdout

    def test_chat_nonexistent_store(self):
        """Test chat command with non-existent store."""
        runner = CliRunner()
        result = runner.invoke(app, ["chat", "--store", "/nonexistent/path"])
        
        assert result.exit_code == 1
        assert "Vector store not found" in result.stdout

    @pytest.mark.integration
    def test_module_execution(self):
        """Test that the module can be executed with python -m codechat."""
        # Test help command
        result = subprocess.run(
            ["python", "-m", "codechat", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent
        )
        
        assert result.returncode == 0
        assert "Chat with your codebase using RAG" in result.stdout

    def test_invalid_command(self):
        """Test invalid command handling."""
        runner = CliRunner()
        result = runner.invoke(app, ["invalid_command"])
        
        assert result.exit_code != 0
