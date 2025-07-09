"""Tests for the chunker module."""

from pathlib import Path

import pytest

from codechat.ingestion.chunker import CodeChunk, SyntaxAwareChunker


class TestCodeChunk:
    """Test cases for CodeChunk class."""

    def test_init(self):
        """Test CodeChunk initialization."""
        file_path = Path("test.py")
        content = "def hello(): return 'world'"
        
        chunk = CodeChunk(
            content=content,
            file_path=file_path,
            start_line=1,
            end_line=1,
            language="python",
            chunk_type="function"
        )
        
        assert chunk.content == content
        assert chunk.file_path == file_path
        assert chunk.start_line == 1
        assert chunk.end_line == 1
        assert chunk.language == "python"
        assert chunk.chunk_type == "function"

    def test_str(self):
        """Test CodeChunk string representation."""
        chunk = CodeChunk(
            content="def hello(): return 'world'",
            file_path=Path("test.py"),
            start_line=1,
            end_line=1,
            language="python",
            chunk_type="function"
        )
        
        str_repr = str(chunk)
        assert "File: test.py" in str_repr
        assert "Lines: 1-1" in str_repr
        assert "Language: python" in str_repr
        assert "def hello(): return 'world'" in str_repr


class TestSyntaxAwareChunker:
    """Test cases for SyntaxAwareChunker class."""

    def test_init(self):
        """Test SyntaxAwareChunker initialization."""
        chunker = SyntaxAwareChunker(max_chunk_size=1000, chunk_overlap=100)
        assert chunker.max_chunk_size == 1000
        assert chunker.chunk_overlap == 100

    def test_detect_language(self):
        """Test language detection from file extensions."""
        chunker = SyntaxAwareChunker()
        
        assert chunker._detect_language(Path("test.py")) == "python"
        assert chunker._detect_language(Path("test.js")) == "javascript"
        assert chunker._detect_language(Path("test.ts")) == "typescript"
        assert chunker._detect_language(Path("test.java")) == "java"
        assert chunker._detect_language(Path("test.go")) == "go"
        assert chunker._detect_language(Path("test.txt")) == "text"

    def test_get_indent_level(self):
        """Test indentation level calculation."""
        chunker = SyntaxAwareChunker()
        
        assert chunker._get_indent_level("def func():") == 0
        assert chunker._get_indent_level("    return True") == 4
        assert chunker._get_indent_level("        value = 1") == 8
        assert chunker._get_indent_level("\t\tif True:") == 2  # tabs count as 1 each

    def test_identify_block_type_python(self):
        """Test block type identification for Python."""
        chunker = SyntaxAwareChunker()
        patterns = chunker.language_patterns["python"]
        
        assert chunker._identify_block_type("def my_function():", patterns) == "function"
        assert chunker._identify_block_type("class MyClass:", patterns) == "class"
        assert chunker._identify_block_type("import os", patterns) == "import"
        assert chunker._identify_block_type("from typing import List", patterns) == "import"
        assert chunker._identify_block_type("regular_line = 1", patterns) is None

    def test_identify_block_type_javascript(self):
        """Test block type identification for JavaScript."""
        chunker = SyntaxAwareChunker()
        patterns = chunker.language_patterns["javascript"]
        
        assert chunker._identify_block_type("function myFunc() {", patterns) == "function"
        assert chunker._identify_block_type("const myFunc = () => {", patterns) == "function"
        assert chunker._identify_block_type("class MyClass {", patterns) == "class"
        assert chunker._identify_block_type("import React from 'react'", patterns) == "import"

    def test_chunk_file_python(self, sample_python_code):
        """Test chunking a Python file."""
        chunker = SyntaxAwareChunker(max_chunk_size=2048)
        file_path = Path("test.py")
        
        chunks = chunker.chunk_file(file_path, sample_python_code)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, CodeChunk) for chunk in chunks)
        assert all(chunk.language == "python" for chunk in chunks)
        assert all(chunk.file_path == file_path for chunk in chunks)
        
        # Check that chunks contain expected content
        all_content = "\n".join(chunk.content for chunk in chunks)
        assert "fibonacci" in all_content
        assert "Calculator" in all_content

    def test_chunk_file_javascript(self, sample_javascript_code):
        """Test chunking a JavaScript file."""
        chunker = SyntaxAwareChunker(max_chunk_size=2048)
        file_path = Path("test.js")
        
        chunks = chunker.chunk_file(file_path, sample_javascript_code)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, CodeChunk) for chunk in chunks)
        assert all(chunk.language == "javascript" for chunk in chunks)
        assert all(chunk.file_path == file_path for chunk in chunks)

    def test_chunk_file_large_content(self, sample_python_code):
        """Test chunking with small chunk size to force splitting."""
        chunker = SyntaxAwareChunker(max_chunk_size=200, chunk_overlap=50)
        file_path = Path("test.py")
        
        chunks = chunker.chunk_file(file_path, sample_python_code)
        
        # Should create multiple chunks due to size limit
        assert len(chunks) > 1
        
        # Each chunk should be reasonably sized
        for chunk in chunks:
            assert len(chunk.content) <= chunker.max_chunk_size * 1.1  # Allow some tolerance

    def test_chunk_file_empty_content(self):
        """Test chunking empty content."""
        chunker = SyntaxAwareChunker()
        file_path = Path("empty.py")
        
        chunks = chunker.chunk_file(file_path, "")
        
        # Should handle empty content gracefully
        assert len(chunks) == 0 or (len(chunks) == 1 and not chunks[0].content.strip())

    def test_chunk_file_single_line(self):
        """Test chunking single line content."""
        chunker = SyntaxAwareChunker()
        file_path = Path("single.py")
        content = "print('hello world')"
        
        chunks = chunker.chunk_file(file_path, content)
        
        assert len(chunks) == 1
        assert chunks[0].content.strip() == content
        assert chunks[0].start_line == 1
        assert chunks[0].end_line == 1

    def test_extract_semantic_blocks_python(self, sample_python_code):
        """Test semantic block extraction for Python."""
        chunker = SyntaxAwareChunker()
        lines = sample_python_code.split("\n")
        
        blocks = chunker._extract_semantic_blocks(lines, "python")
        
        assert len(blocks) > 0
        
        # Should have separate blocks for function and class
        block_types = [block["type"] for block in blocks]
        assert "function" in block_types or "code" in block_types
        assert "class" in block_types or "code" in block_types

    def test_split_with_overlap(self):
        """Test splitting large content with overlap."""
        chunker = SyntaxAwareChunker(max_chunk_size=100, chunk_overlap=20)
        content = "\n".join([f"line {i}" for i in range(20)])  # 20 lines
        file_path = Path("test.py")
        
        chunks = chunker._split_with_overlap(content, 1, file_path, "python")
        
        assert len(chunks) > 1
        
        # Check overlap between consecutive chunks
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            
            # Next chunk should start before current chunk ends (overlap)
            assert next_chunk.start_line <= current_chunk.end_line
