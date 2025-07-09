#!/usr/bin/env python3
"""Interactive test of the CLI chat functionality."""

import subprocess
import sys
import time
from pathlib import Path

def test_interactive_chat():
    """Test the CLI chat functionality with predefined questions."""
    
    store_path = Path("stores/f1-dashboard")
    
    if not store_path.exists():
        print("❌ Store not found. Please run ingestion first.")
        return False
    
    # Questions to test
    test_questions = [
        "What is the main purpose of this F1 dashboard application?",
        "How does the application fetch F1 race data?", 
        "What API endpoints are available in this application?",
        "quit"
    ]
    
    print("🚀 Testing CLI Chat Interface...")
    print("=" * 50)
    
    # Start the CLI chat process
    process = subprocess.Popen(
        [sys.executable, "-m", "codechat", "chat", "--store", str(store_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    try:
        # Send questions
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            
            # Send the question
            process.stdin.write(question + "\n")
            process.stdin.flush()
            
            if question == "quit":
                break
                
            # Wait a bit for response
            time.sleep(2)
            
            # Try to read some output
            try:
                # Read available output
                output = ""
                while True:
                    line = process.stdout.readline()
                    if not line:
                        break
                    output += line
                    if "Chat with your codebase" in line or ">" in line:
                        break
                
                print(f"💬 Response: {output.strip()}")
                
            except Exception as e:
                print(f"⚠️ Error reading output: {e}")
        
        # Wait for process to finish
        process.wait(timeout=10)
        
        print("\n✅ CLI chat test completed!")
        return True
        
    except Exception as e:
        print(f"❌ CLI chat test failed: {e}")
        return False
        
    finally:
        # Clean up process
        if process.poll() is None:
            process.terminate()
            process.wait()

if __name__ == "__main__":
    success = test_interactive_chat()
    if not success:
        sys.exit(1)
