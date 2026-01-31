# Video Semantic Search

Search inside videos using natural language queries. Powered by SigLIP 2 + ChromaDB.

## How It Works

1. **Index**: Extract frames from video → Generate embeddings → Store in vector DB
2. **Search**: Text query → Generate embedding → Find matching frames

## Requirements

- Python 3.10+
- NVIDIA GPU (recommended)

## Setup

# Install uv (if not installed)
pip install uv

# Install dependencies
uv sync
