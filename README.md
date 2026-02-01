# Multimodal Semantic Search Engine

A semantic search engine that lets you search across **videos**, **images**, and **documents** using natural language queries.

## Features

- **Video Search**: Find specific moments in videos by describing what you're looking for
- **Image Search**: Search through image collections using natural language
- **Document Search**: Search PDFs, Word docs, and text files by meaning, not just keywords
- **Unified Interface**: Google-style tabbed results (All | Videos | Images | Documents)
- **GPU Accelerated**: Fast inference with CUDA support

## How It Works

```
VISUAL CONTENT (SigLIP 2)
┌─────────────────────────────────────────────────────────────────┐
│  Video ──► Frames ──► SigLIP 2 ──► visual_content collection   │
│  Image ──────────────► SigLIP 2 ──► visual_content collection   │
└─────────────────────────────────────────────────────────────────┘

DOCUMENTS (E5-small)
┌─────────────────────────────────────────────────────────────────┐
│  Document ──► Text Extraction ──► Chunking ──► E5-small ──► documents collection │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| Visual Embeddings | [SigLIP 2](https://huggingface.co/google/siglip2-base-patch16-224) | 768-dim image-text embeddings |
| Document Embeddings | [multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | 384-dim text embeddings, 100+ languages |
| Vector Database | [ChromaDB](https://www.trychroma.com/) | Persistent vector storage with HNSW indexing |
| Frame Extraction | OpenCV | Video processing at configurable FPS |
| Frontend | Streamlit | Interactive web interface |

## Requirements

- Python 3.10+
- NVIDIA GPU (recommended for faster inference)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/video-semantic-search.git
cd video-semantic-search

# Install uv (if not installed)
pip install uv

# Install dependencies
uv sync
```

## Usage

### Streamlit App

```bash
uv run streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Jupyter Notebook

For experimentation and development:

```bash
uv run jupyter notebook main.ipynb
```

## Supported Formats

| Type | Formats |
|------|---------|
| Video | `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm` |
| Image | `.jpg`, `.jpeg`, `.png`, `.webp`, `.gif` |
| Document | `.pdf`, `.docx`, `.txt`, `.md` |

## Project Structure

```
video-semantic-search/
├── app.py                 # Streamlit web application
├── main.ipynb             # Jupyter notebook for experimentation
├── src/
│   ├── video_search.py    # Video indexing and search (SigLIP 2)
│   ├── image_search.py    # Image indexing (SigLIP 2)
│   └── document_search.py # Document indexing and search (E5-small)
├── data/
│   ├── videos/            # Uploaded videos
│   ├── images/            # Uploaded images
│   ├── documents/         # Uploaded documents
│   └── chroma_db/         # Vector database storage
└── pyproject.toml         # Dependencies
```

## Configuration

Key parameters in `src/video_search.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SAMPLE_RATE` | 1.0 | Frames per second to extract from videos |
| `BATCH_SIZE` | 8 | Batch size for embedding generation |
| `TOP_K` | 5 | Default number of search results |

Key parameters in `src/document_search.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CHUNK_SIZE` | 400 | Words per document chunk |
| `CHUNK_OVERLAP` | 50 | Overlapping words between chunks |

## License

MIT
