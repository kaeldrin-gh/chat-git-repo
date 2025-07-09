"""Git repository loader for code ingestion."""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, List, Tuple

import git
from git import Repo

logger = logging.getLogger(__name__)

from ..config import SUPPORTED_EXTENSIONS, TEMP_CLONE_DIR


class GitLoader:
    """Handles cloning and iterating through code files in Git repositories."""

    def __init__(self, temp_dir: Path | None = None) -> None:
        """Initialize the GitLoader.
        
        Args:
            temp_dir: Directory for temporary repo clones. Uses default if None.
        """
        self.temp_dir = temp_dir or TEMP_CLONE_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def clone_repo(self, repo_url: str, target_dir: Path | None = None) -> Path:
        """Clone a Git repository to a local directory.
        
        Args:
            repo_url: URL of the Git repository to clone.
            target_dir: Directory to clone into. Uses temp dir if None.
            
        Returns:
            Path to the cloned repository.
            
        Raises:
            git.GitError: If cloning fails.
        """
        if target_dir is None:
            # Create a unique temp directory for this repo
            repo_name = repo_url.split("/")[-1].replace(".git", "")
            target_dir = self.temp_dir / repo_name
            
        # Remove existing directory if it exists
        if target_dir.exists():
            shutil.rmtree(target_dir)
            
        try:
            repo = Repo.clone_from(repo_url, target_dir, depth=1)
            return Path(repo.working_dir)
        except git.GitError as e:
            raise git.GitError(f"Failed to clone repository {repo_url}: {e}")

    def iter_code_files(self, repo_path: Path) -> Iterator[Tuple[Path, str]]:
        """Iterate through code files in a repository.
        
        Args:
            repo_path: Path to the Git repository.
            
        Yields:
            Tuples of (file_path, file_content) for each code file.
        """
        for file_path in repo_path.rglob("*"):
            if self._is_code_file(file_path):
                try:
                    content = file_path.read_text(encoding="utf-8", errors="ignore")
                    yield file_path, content
                except (UnicodeDecodeError, PermissionError):
                    # Skip files that can't be read
                    continue

    def _is_code_file(self, file_path: Path) -> bool:
        """Check if a file is a supported code file.
        
        Args:
            file_path: Path to the file to check.
            
        Returns:
            True if the file is a supported code file.
        """
        if not file_path.is_file():
            return False
            
        # Check file extension
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            return False
            
        # Skip common non-code directories
        exclude_dirs = {".git", "__pycache__", "node_modules", ".pytest_cache", ".mypy_cache"}
        if any(part in exclude_dirs for part in file_path.parts):
            return False
            
        # Skip files that are too large (>1MB)
        try:
            if file_path.stat().st_size > 1024 * 1024:
                return False
        except OSError:
            return False
            
        return True

    def cleanup(self) -> None:
        """Clean up temporary directories."""
        if self.temp_dir.exists():
            self._force_remove_readonly(self.temp_dir)
    
    def _force_remove_readonly(self, path: Path) -> None:
        """Force remove readonly files on Windows."""
        import stat
        
        def handle_remove_readonly(func, path, exc):
            """Handle readonly files during removal."""
            if exc[0] == PermissionError:
                # Change the file to be writable and try again
                os.chmod(path, stat.S_IWRITE)
                func(path)
            else:
                raise exc[1]
        
        try:
            shutil.rmtree(path, onerror=handle_remove_readonly)
        except Exception:
            # If still fails, try to change permissions recursively
            try:
                for root, dirs, files in os.walk(path):
                    for dir in dirs:
                        os.chmod(os.path.join(root, dir), stat.S_IWRITE)
                    for file in files:
                        os.chmod(os.path.join(root, file), stat.S_IWRITE)
                shutil.rmtree(path)
            except Exception:
                # Final fallback - just log the error
                logger.warning(f"Failed to clean up temporary directory: {path}")
