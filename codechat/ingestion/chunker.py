"""Syntax-aware code chunking for better semantic understanding."""

import re
from pathlib import Path
from typing import Dict, List, Tuple

from ..config import CHUNK_OVERLAP, MAX_CHUNK_SIZE


class CodeChunk:
    """Represents a chunk of code with metadata."""

    def __init__(
        self,
        content: str,
        file_path: Path,
        start_line: int,
        end_line: int,
        language: str,
        chunk_type: str = "code",
    ) -> None:
        """Initialize a code chunk.
        
        Args:
            content: The code content.
            file_path: Path to the source file.
            start_line: Starting line number.
            end_line: Ending line number.
            language: Programming language.
            chunk_type: Type of chunk (code, function, class, etc.).
        """
        self.content = content
        self.file_path = file_path
        self.start_line = start_line
        self.end_line = end_line
        self.language = language
        self.chunk_type = chunk_type

    def __str__(self) -> str:
        """String representation for indexing."""
        metadata = f"File: {self.file_path}\nLines: {self.start_line}-{self.end_line}\nLanguage: {self.language}\n\n"
        return metadata + self.content


class SyntaxAwareChunker:
    """Chunks code files with syntax awareness for better semantic understanding."""

    def __init__(self, max_chunk_size: int = MAX_CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> None:
        """Initialize the chunker.
        
        Args:
            max_chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks in characters.
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Language-specific patterns for semantic boundaries
        self.language_patterns = {
            "python": {
                "function": r"^(def\s+\w+\s*\([^)]*\)\s*(?:->\s*[^:]+)?\s*:)",
                "class": r"^(class\s+\w+(?:\([^)]*\))?\s*:)",
                "import": r"^(import\s+\w+|from\s+\w+\s+import)",
                "comment": r"^(\s*#.*|^\s*\"\"\".*?\"\"\")",
            },
            "javascript": {
                "function": r"^(function\s+\w+\s*\([^)]*\)\s*\{|const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*\{)",
                "class": r"^(class\s+\w+(?:\s+extends\s+\w+)?\s*\{)",
                "import": r"^(import\s+.*from\s+['\"].*['\"]|const\s+.*=\s+require\(['\"].*['\"]\))",
                "comment": r"^(\s*//.*|^\s*/\*.*?\*/)",
            },
            "typescript": {
                "function": r"^(function\s+\w+\s*\([^)]*\)\s*:?[^{]*\{|const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*:?[^=]*=>\s*\{)",
                "class": r"^(class\s+\w+(?:\s+extends\s+\w+)?(?:\s+implements\s+\w+)?\s*\{)",
                "interface": r"^(interface\s+\w+\s*\{|type\s+\w+\s*=)",
                "import": r"^(import\s+.*from\s+['\"].*['\"])",
            },
            "java": {
                "method": r"^(\s*(?:public|private|protected)?\s*(?:static\s+)?[^{]*\s+\w+\s*\([^)]*\)\s*(?:throws\s+[^{]*)?\{)",
                "class": r"^(\s*(?:public|private|protected)?\s*(?:abstract\s+)?class\s+\w+(?:\s+extends\s+\w+)?(?:\s+implements\s+[^{]*)?\s*\{)",
                "import": r"^(import\s+[^;]+;|package\s+[^;]+;)",
            },
            "go": {
                "function": r"^(func\s+(?:\([^)]*\)\s+)?\w+\s*\([^)]*\)(?:\s*[^{]*)?\s*\{)",
                "struct": r"^(type\s+\w+\s+struct\s*\{)",
                "import": r"^(import\s*\(|import\s+['\"].*['\"])",
            },
        }

    def chunk_file(self, file_path: Path, content: str) -> List[CodeChunk]:
        """Chunk a code file with syntax awareness.
        
        Args:
            file_path: Path to the source file.
            content: File content to chunk.
            
        Returns:
            List of CodeChunk objects.
        """
        language = self._detect_language(file_path)
        lines = content.split("\n")
        
        # First, try to extract semantic blocks (functions, classes, etc.)
        semantic_blocks = self._extract_semantic_blocks(lines, language)
        
        # Then chunk each block if it's too large
        chunks: List[CodeChunk] = []
        for block in semantic_blocks:
            if len(block["content"]) <= self.max_chunk_size:
                chunks.append(
                    CodeChunk(
                        content=block["content"],
                        file_path=file_path,
                        start_line=block["start_line"],
                        end_line=block["end_line"],
                        language=language,
                        chunk_type=block["type"],
                    )
                )
            else:
                # Split large blocks with overlap
                sub_chunks = self._split_with_overlap(
                    block["content"], block["start_line"], file_path, language
                )
                chunks.extend(sub_chunks)
        
        return chunks

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension.
        
        Args:
            file_path: Path to the file.
            
        Returns:
            Programming language identifier.
        """
        extension_map = {
            ".py": "python",
            ".js": "javascript", 
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
        }
        return extension_map.get(file_path.suffix.lower(), "text")

    def _extract_semantic_blocks(self, lines: List[str], language: str) -> List[Dict]:
        """Extract semantic blocks (functions, classes, etc.) from code.
        
        Args:
            lines: Lines of code.
            language: Programming language.
            
        Returns:
            List of semantic blocks with metadata.
        """
        blocks: List[Dict] = []
        patterns = self.language_patterns.get(language, {})
        
        current_block: Dict = {
            "content": "",
            "start_line": 1,
            "end_line": 1,
            "type": "code",
            "indent_level": 0,
        }
        
        for i, line in enumerate(lines, 1):
            # Check if this line starts a new semantic block
            block_type = self._identify_block_type(line, patterns)
            
            if block_type and current_block["content"].strip():
                # Finish current block
                current_block["end_line"] = i - 1
                blocks.append(current_block)
                
                # Start new block
                current_block = {
                    "content": line + "\n",
                    "start_line": i,
                    "end_line": i,
                    "type": block_type,
                    "indent_level": self._get_indent_level(line),
                }
            else:
                # Add line to current block
                current_block["content"] += line + "\n"
                current_block["end_line"] = i
        
        # Add the last block
        if current_block["content"].strip():
            blocks.append(current_block)
        
        return blocks

    def _identify_block_type(self, line: str, patterns: Dict[str, str]) -> str | None:
        """Identify the type of semantic block a line starts.
        
        Args:
            line: Line of code to analyze.
            patterns: Language-specific regex patterns.
            
        Returns:
            Block type or None if no pattern matches.
        """
        for block_type, pattern in patterns.items():
            if re.match(pattern, line.strip(), re.MULTILINE | re.DOTALL):
                return block_type
        return None

    def _get_indent_level(self, line: str) -> int:
        """Get the indentation level of a line.
        
        Args:
            line: Line of code.
            
        Returns:
            Indentation level (number of leading spaces).
        """
        return len(line) - len(line.lstrip())

    def _split_with_overlap(
        self, content: str, start_line: int, file_path: Path, language: str
    ) -> List[CodeChunk]:
        """Split large content into overlapping chunks.
        
        Args:
            content: Content to split.
            start_line: Starting line number.
            file_path: Source file path.
            language: Programming language.
            
        Returns:
            List of CodeChunk objects.
        """
        chunks: List[CodeChunk] = []
        lines = content.split("\n")
        
        i = 0
        while i < len(lines):
            # Calculate chunk end
            chunk_lines: List[str] = []
            char_count = 0
            
            for j in range(i, len(lines)):
                line_with_newline = lines[j] + "\n"
                if char_count + len(line_with_newline) > self.max_chunk_size and chunk_lines:
                    break
                chunk_lines.append(lines[j])
                char_count += len(line_with_newline)
            
            if chunk_lines:
                chunk_content = "\n".join(chunk_lines)
                chunks.append(
                    CodeChunk(
                        content=chunk_content,
                        file_path=file_path,
                        start_line=start_line + i,
                        end_line=start_line + i + len(chunk_lines) - 1,
                        language=language,
                        chunk_type="code_fragment",
                    )
                )
            
            # Calculate overlap for next chunk
            if i + len(chunk_lines) >= len(lines):
                break
                
            # Move forward with overlap
            overlap_chars = 0
            overlap_lines = 0
            for k in range(len(chunk_lines) - 1, -1, -1):
                line_chars = len(chunk_lines[k]) + 1  # +1 for newline
                if overlap_chars + line_chars <= self.chunk_overlap:
                    overlap_chars += line_chars
                    overlap_lines += 1
                else:
                    break
            
            i += max(1, len(chunk_lines) - overlap_lines)
        
        return chunks
