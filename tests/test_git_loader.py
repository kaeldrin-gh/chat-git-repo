"""Tests for the git loader module."""

import tempfile
from pathlib import Path

import pytest

from codechat.ingestion.git_loader import GitLoader


class TestGitLoader:
    """Test cases for GitLoader class."""

    def test_init(self, temp_dir):
        """Test GitLoader initialization."""
        loader = GitLoader(temp_dir=temp_dir)
        assert loader.temp_dir == temp_dir
        assert loader.temp_dir.exists()

    def test_init_default_temp_dir(self):
        """Test GitLoader initialization with default temp dir."""
        loader = GitLoader()
        assert loader.temp_dir.exists()
        loader.cleanup()

    def test_is_code_file(self, temp_dir):
        """Test _is_code_file method."""
        loader = GitLoader(temp_dir=temp_dir)
        
        # Create test files
        python_file = temp_dir / "test.py"
        python_file.write_text("print('hello')")
        
        js_file = temp_dir / "test.js"
        js_file.write_text("console.log('hello');")
        
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("hello world")
        
        # Test supported extensions
        assert loader._is_code_file(python_file)
        assert loader._is_code_file(js_file)
        assert not loader._is_code_file(txt_file)
        
        # Test directory
        test_dir = temp_dir / "subdir"
        test_dir.mkdir()
        assert not loader._is_code_file(test_dir)

    def test_is_code_file_exclude_dirs(self, temp_dir):
        """Test _is_code_file excludes certain directories."""
        loader = GitLoader(temp_dir=temp_dir)
        
        # Create files in excluded directories
        git_dir = temp_dir / ".git"
        git_dir.mkdir()
        git_file = git_dir / "config.py"
        git_file.write_text("# git config")
        
        pycache_dir = temp_dir / "__pycache__"
        pycache_dir.mkdir()
        cache_file = pycache_dir / "module.py"
        cache_file.write_text("# cache")
        
        assert not loader._is_code_file(git_file)
        assert not loader._is_code_file(cache_file)

    def test_is_code_file_large_files(self, temp_dir):
        """Test _is_code_file excludes large files."""
        loader = GitLoader(temp_dir=temp_dir)
        
        # Create a large file (>1MB)
        large_file = temp_dir / "large.py"
        large_content = "# " + "x" * (1024 * 1024 + 1)  # Just over 1MB
        large_file.write_text(large_content)
        
        assert not loader._is_code_file(large_file)

    def test_iter_code_files(self, temp_dir):
        """Test iter_code_files method."""
        loader = GitLoader(temp_dir=temp_dir)
        
        # Create test files
        python_file = temp_dir / "test.py"
        python_file.write_text("def hello(): return 'world'")
        
        js_file = temp_dir / "test.js"
        js_file.write_text("function hello() { return 'world'; }")
        
        # Create subdirectory with file
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        sub_python_file = subdir / "sub.py"
        sub_python_file.write_text("import os")
        
        # Create non-code file
        txt_file = temp_dir / "readme.txt"
        txt_file.write_text("This is a readme")
        
        # Collect files
        files = list(loader.iter_code_files(temp_dir))
        
        # Should find 3 code files
        assert len(files) == 3
        
        # Check file paths and contents
        file_paths = {file_path for file_path, _ in files}
        assert python_file in file_paths
        assert js_file in file_paths
        assert sub_python_file in file_paths
        assert txt_file not in file_paths

    def test_cleanup(self, temp_dir):
        """Test cleanup method."""
        loader = GitLoader(temp_dir=temp_dir)
        
        # Create a test file
        test_file = temp_dir / "test.py"
        test_file.write_text("print('test')")
        
        assert temp_dir.exists()
        assert test_file.exists()
        
        loader.cleanup()
        
        assert not temp_dir.exists()

    @pytest.mark.integration
    def test_clone_repo_integration(self, temp_dir):
        """Integration test for cloning a real repository."""
        loader = GitLoader(temp_dir=temp_dir)
        
        # Use a small, reliable test repository
        repo_url = "https://github.com/octocat/Hello-World.git"
        
        try:
            repo_path = loader.clone_repo(repo_url)
            assert repo_path.exists()
            assert repo_path.is_dir()
            
            # Should contain at least one file
            files = list(repo_path.rglob("*"))
            assert len(files) > 0
            
        except Exception as e:
            pytest.skip(f"Network or Git error: {e}")
        finally:
            loader.cleanup()
