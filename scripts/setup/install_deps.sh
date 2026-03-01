#!/bin/bash
# pip install + Ollama model pull
set -e
pip install -r requirements.txt
pip install -r requirements-rag.txt 2>/dev/null || true
echo "Pull Ollama model (optional): ollama pull llama3.1:8b"
