#!/usr/bin/env bash
set -e

echo "🚀 Starting smoke test for codechat..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Create a temporary directory for the test
TEST_STORE="/tmp/codechat_smoke_test"
rm -rf "$TEST_STORE"

echo "🔄 Testing ingestion..."
# Test ingestion with a small, reliable repository
python -m codechat ingest --repo-url https://github.com/psf/requests --out-store "$TEST_STORE"

echo "🔍 Testing store info..."
python -m codechat info --store "$TEST_STORE"

echo "💬 Testing chat functionality..."
# Test chat with a predefined question (non-interactive mode)
echo "Where is authentication handled?" | python -m codechat chat --store "$TEST_STORE" --no-interactive

echo "🧹 Cleaning up..."
rm -rf "$TEST_STORE"

echo "✅ ALL CHECKS PASSED"
