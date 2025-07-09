#!/usr/bin/env python3
"""Setup script to help users configure the environment."""

import os
import sys
from pathlib import Path

def main():
    """Main setup function."""
    print("🚀 Chat with Your Codebase - Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ is required")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")
    
    # Check if HuggingFace token is set
    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("\n⚠️  HuggingFace token not found!")
        print("   You need to set your HuggingFace token to use the Gemma model.")
        print("   1. Get a token from: https://huggingface.co/settings/tokens")
        print("   2. Set it as environment variable:")
        print("      export HUGGINGFACE_TOKEN=your_token_here")
        print("\n   Or set it now for this session:")
        token = input("   Enter your HuggingFace token (or press Enter to skip): ").strip()
        if token:
            os.environ["HUGGINGFACE_TOKEN"] = token
            print("   ✅ Token set for this session!")
        else:
            print("   ⚠️  Skipping token setup - you'll need to set it later")
    else:
        print(f"✅ HuggingFace token found: {hf_token[:8]}...")
    
    # Check if requirements are installed
    print("\n📦 Checking dependencies...")
    try:
        import torch
        import transformers
        import faiss
        import sentence_transformers
        print("✅ All main dependencies are installed!")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Please run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality...")
    try:
        from codechat.config import DEFAULT_LLM_MODEL, DEFAULT_EMBEDDING_MODEL
        print(f"✅ Default LLM model: {DEFAULT_LLM_MODEL}")
        print(f"✅ Default embedding model: {DEFAULT_EMBEDDING_MODEL}")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        sys.exit(1)
    
    print("\n🎉 Setup complete!")
    print("\n📖 Quick start:")
    print("   1. Ingest a repository:")
    print("      python -m codechat ingest --repo-url https://github.com/owner/repo --out-store stores/repo_name")
    print("   2. Chat with the code:")
    print("      python -m codechat chat --store stores/repo_name")
    print("\n   For more information, see the README.md")

if __name__ == "__main__":
    main()
