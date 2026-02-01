"""
Document Semantic Search Module

This module provides document search functionality:
- Text extraction from PDF, DOCX, TXT, and MD files
- Text chunking with overlap
- Embedding generation using multilingual-e5-small
- Vector storage and search using ChromaDB
"""

from pathlib import Path
from typing import Optional, Callable
import chromadb
from sentence_transformers import SentenceTransformer

# =============================================================================
# CONFIGURATION
# =============================================================================

# multilingual-e5-small: 470MB, 384-dim, 100+ languages
# Lightweight alternative to BGE-M3 (2.3GB) with excellent multilingual support
# Note: E5 models use "query: " and "passage: " prefixes for optimal performance
DOCUMENT_MODEL_NAME = "intfloat/multilingual-e5-small"
DOCUMENT_EMBEDDING_DIM = 384
DEFAULT_CHUNK_SIZE = 400  # words
DEFAULT_CHUNK_OVERLAP = 50  # words
DEFAULT_TOP_K = 10

DATA_DIR = Path("./data")
DOCUMENTS_DIR = DATA_DIR / "documents"
DB_DIR = DATA_DIR / "chroma_db"

# Ensure directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}


# =============================================================================
# MODEL LOADING
# =============================================================================

_document_model = None


def load_document_model():
    """
    Load multilingual-e5-small model for document embedding.
    
    Returns:
        SentenceTransformer: The loaded model
    """
    global _document_model
    if _document_model is None:
        _document_model = SentenceTransformer(DOCUMENT_MODEL_NAME)
    return _document_model


# =============================================================================
# CHROMADB
# =============================================================================

_chroma_client = None
_document_collection = None


def get_chroma_client():
    """Get or create ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    return _chroma_client


def get_document_collection():
    """Get or create the documents collection."""
    global _document_collection
    if _document_collection is None:
        client = get_chroma_client()
        _document_collection = client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
    return _document_collection


def get_document_stats():
    """
    Get statistics about the indexed documents.
    
    Returns:
        dict: Statistics including total chunks and document list
    """
    collection = get_document_collection()
    total_chunks = collection.count()
    
    stats = {
        "total_chunks": total_chunks,
        "documents": [],
        "document_count": 0
    }
    
    if total_chunks > 0:
        all_data = collection.get(include=["metadatas"])
        doc_ids = set(m.get("doc_id", "unknown") for m in all_data["metadatas"])
        stats["documents"] = list(doc_ids)
        stats["document_count"] = len(doc_ids)
    
    return stats


def clear_document_database():
    """Clear all data from the documents collection."""
    global _document_collection
    client = get_chroma_client()
    try:
        client.delete_collection("documents")
    except Exception:
        pass
    _document_collection = client.create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"}
    )
    return True


# =============================================================================
# TEXT EXTRACTION
# =============================================================================

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain text or markdown file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text_from_pdf(file_path: str) -> tuple[str, list[dict]]:
    """
    Extract text from a PDF file.
    
    Returns:
        tuple: (full_text, page_info_list)
    """
    import fitz  # PyMuPDF
    
    doc = fitz.open(file_path)
    full_text = ""
    page_info = []
    
    for page_num, page in enumerate(doc):
        text = page.get_text()
        page_info.append({
            "page": page_num + 1,
            "start_char": len(full_text),
            "text": text
        })
        full_text += text + "\n\n"
    
    doc.close()
    return full_text, page_info


def extract_text_from_docx(file_path: str) -> str:
    """Extract text from a DOCX file."""
    from docx import Document
    
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_text(file_path: str) -> tuple[str, str, Optional[list]]:
    """
    Extract text from a document file.
    
    Args:
        file_path: Path to the document
        
    Returns:
        tuple: (text, file_type, page_info or None)
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")
    
    if ext in {".txt", ".md"}:
        return extract_text_from_txt(file_path), ext[1:], None
    elif ext == ".pdf":
        text, page_info = extract_text_from_pdf(file_path)
        return text, "pdf", page_info
    elif ext == ".docx":
        return extract_text_from_docx(file_path), "docx", None
    
    raise ValueError(f"Unsupported file type: {ext}")


# =============================================================================
# TEXT CHUNKING
# =============================================================================

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[dict]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: The text to chunk
        chunk_size: Number of words per chunk
        overlap: Number of overlapping words between chunks
        
    Returns:
        list: List of dicts with chunk_idx and text
    """
    words = text.split()
    chunks = []
    
    if len(words) == 0:
        return chunks
    
    step = max(1, chunk_size - overlap)
    
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < overlap and chunks:
            # Skip very small final chunks
            break
        
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "chunk_idx": len(chunks),
            "text": chunk_text,
            "word_start": i,
            "word_end": i + len(chunk_words)
        })
    
    return chunks


# =============================================================================
# EMBEDDING
# =============================================================================

def embed_texts(
    texts: list[str], 
    model: SentenceTransformer,
    is_query: bool = False
) -> list[list[float]]:
    """
    Generate embeddings for a list of texts.
    
    E5 models work best with prefixes:
    - "query: " for search queries
    - "passage: " for document chunks being indexed
    
    Args:
        texts: List of text strings
        model: multilingual-e5-small model
        is_query: If True, use "query: " prefix; otherwise use "passage: "
        
    Returns:
        list: List of embedding vectors
    """
    # Add appropriate prefix for E5 model
    prefix = "query: " if is_query else "passage: "
    prefixed_texts = [prefix + text for text in texts]
    
    embeddings = model.encode(
        prefixed_texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )
    return embeddings.tolist()


# =============================================================================
# INDEXING & SEARCH
# =============================================================================

def index_document(
    file_path: str,
    doc_id: str,
    model: SentenceTransformer,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    progress_callback: Optional[Callable] = None
) -> int:
    """
    Index a document into ChromaDB.
    
    Args:
        file_path: Path to document file
        doc_id: Unique identifier for this document
        model: multilingual-e5-small model
        chunk_size: Words per chunk
        chunk_overlap: Overlap between chunks
        progress_callback: Optional callback(stage, current, total, message)
        
    Returns:
        int: Number of chunks indexed
    """
    collection = get_document_collection()
    
    # Stage 1: Extract text
    if progress_callback:
        progress_callback("extract", 0, 100, "Extracting text...")
    
    text, file_type, page_info = extract_text(file_path)
    
    if not text.strip():
        raise ValueError("No text extracted from document")
    
    if progress_callback:
        progress_callback("extract", 100, 100, "Text extracted")
    
    # Stage 2: Chunk text
    if progress_callback:
        progress_callback("chunk", 0, 100, "Chunking text...")
    
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    
    if len(chunks) == 0:
        raise ValueError("No chunks created from document")
    
    if progress_callback:
        progress_callback("chunk", 100, 100, f"Created {len(chunks)} chunks")
    
    # Stage 3: Generate embeddings
    if progress_callback:
        progress_callback("embed", 0, 100, "Generating embeddings...")
    
    chunk_texts = [c["text"] for c in chunks]
    embeddings = embed_texts(chunk_texts, model)
    
    if progress_callback:
        progress_callback("embed", 100, 100, "Embeddings generated")
    
    # Stage 4: Store in ChromaDB
    if progress_callback:
        progress_callback("store", 0, 1, "Storing in database...")
    
    ids = [f"{doc_id}_chunk_{c['chunk_idx']}" for c in chunks]
    
    metadatas = []
    for chunk in chunks:
        meta = {
            "type": "document",
            "doc_id": doc_id,
            "doc_path": str(file_path),
            "file_type": file_type,
            "chunk_idx": chunk["chunk_idx"],
            "text": chunk["text"][:500],  # Store first 500 chars for display
        }
        
        # Add page info for PDFs
        if page_info:
            for pi in page_info:
                if chunk["word_start"] * 6 >= pi["start_char"]:  # Rough estimate
                    meta["page"] = pi["page"]
        
        metadatas.append(meta)
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )
    
    if progress_callback:
        progress_callback("done", 1, 1, f"Indexed {len(chunks)} chunks")
    
    return len(chunks)


def search_documents(
    query: str,
    model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K
) -> list[dict]:
    """
    Search for document chunks matching the text query.
    
    Args:
        query: Natural language search query
        model: multilingual-e5-small model
        top_k: Number of results to return
        
    Returns:
        list: List of result dicts
    """
    collection = get_document_collection()
    
    if collection.count() == 0:
        return []
    
    # Embed query with "query: " prefix for E5 model
    query_embedding = embed_texts([query], model, is_query=True)
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    
    search_results = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        
        score = 1 - distance  # Convert distance to similarity
        
        search_results.append({
            "rank": i + 1,
            "doc_id": meta["doc_id"],
            "doc_path": meta.get("doc_path", ""),
            "file_type": meta.get("file_type", ""),
            "chunk_idx": meta["chunk_idx"],
            "text": meta.get("text", ""),
            "page": meta.get("page"),
            "score": score,
        })
    
    return search_results
