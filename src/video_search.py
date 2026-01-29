"""
Video Semantic Search Core Module

This module provides the core functionality for video semantic search:
- Frame extraction from videos using OpenCV
- Embedding generation using SigLIP 2
- Vector storage and search using ChromaDB
"""

import cv2
import torch
import numpy as np
from pathlib import Path
from datetime import timedelta
from transformers import AutoProcessor, AutoModel
import chromadb

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = "google/siglip2-base-patch16-224"
EMBEDDING_DIM = 768
DEFAULT_SAMPLE_RATE = 1.0
DEFAULT_BATCH_SIZE = 16
DEFAULT_TOP_K = 10

DATA_DIR = Path("./data")
VIDEOS_DIR = DATA_DIR / "videos"
DB_DIR = DATA_DIR / "chroma_db"

# Ensure directories exist
VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MODEL LOADING
# =============================================================================

def get_device():
    """Get the best available device (CUDA or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """
    Load SigLIP 2 model and processor.
    
    Returns:
        tuple: (model, processor, device)
    """
    device = get_device()
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    model = AutoModel.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        attn_implementation="sdpa" if device == "cuda" else None,
        device_map="auto" if device == "cuda" else None
    ).eval()
    
    if device == "cpu":
        model = model.to(device)
    
    return model, processor, device


# =============================================================================
# CHROMADB
# =============================================================================

_chroma_client = None
_collection = None


def get_chroma_client():
    """Get or create ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    return _chroma_client


def get_collection():
    """Get or create the video frames collection."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name="video_frames",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def get_database_stats():
    """
    Get statistics about the indexed database.
    
    Returns:
        dict: Statistics including total frames and video list
    """
    collection = get_collection()
    total_frames = collection.count()
    
    stats = {
        "total_frames": total_frames,
        "videos": [],
        "video_count": 0
    }
    
    if total_frames > 0:
        all_data = collection.get(include=["metadatas"])
        video_ids = set(m.get("video_id", "unknown") for m in all_data["metadatas"])
        stats["videos"] = list(video_ids)
        stats["video_count"] = len(video_ids)
    
    return stats


def clear_database():
    """Clear all data from the collection."""
    global _collection
    client = get_chroma_client()
    try:
        client.delete_collection("video_frames")
    except Exception:
        pass
    _collection = client.create_collection(
        name="video_frames",
        metadata={"hnsw:space": "cosine"}
    )
    return True


# =============================================================================
# FRAME EXTRACTION
# =============================================================================

def extract_frames(video_path: str, sample_rate: float = DEFAULT_SAMPLE_RATE, progress_callback=None):
    """
    Extract frames from video at specified sample rate.
    
    Args:
        video_path: Path to video file
        sample_rate: Frames per second to extract (1.0 = 1 frame/sec)
        progress_callback: Optional callback function(current, total) for progress updates
    
    Returns:
        tuple: (frames, metadata, video_info)
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    video_info = {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "duration": duration
    }
    
    # Calculate sampling interval
    frame_interval = max(1, int(fps / sample_rate))
    
    frames = []
    metadata = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            timestamp = frame_idx / fps
            
            metadata.append({
                "frame_idx": frame_idx,
                "timestamp": timestamp,
                "timestamp_str": str(timedelta(seconds=int(timestamp))),
            })
        
        frame_idx += 1
        
        if progress_callback:
            progress_callback(frame_idx, total_frames)
    
    cap.release()
    return frames, metadata, video_info


def get_frame_at_index(video_path: str, frame_idx: int):
    """Retrieve a single frame from video at specific index."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def get_frame_at_timestamp(video_path: str, timestamp: float):
    """Retrieve a single frame from video at specific timestamp (seconds)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_idx = int(timestamp * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


# =============================================================================
# EMBEDDING
# =============================================================================

def embed_images(frames: list, model, processor, device, batch_size: int = DEFAULT_BATCH_SIZE, progress_callback=None):
    """
    Generate embeddings for frames using SigLIP 2.
    
    Args:
        frames: List of RGB numpy arrays
        model: SigLIP model
        processor: SigLIP processor
        device: Device string
        batch_size: Batch size for processing
        progress_callback: Optional callback function(current, total) for progress
    
    Returns:
        numpy array of embeddings
    """
    all_embeddings = []
    total_batches = (len(frames) + batch_size - 1) // batch_size
    
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i + batch_size]
        
        inputs = processor(images=batch, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.get_image_features(**inputs)
            # Handle both tensor and BaseModelOutputWithPooling return types
            if hasattr(output, 'pooler_output'):
                embeddings = output.pooler_output
            elif hasattr(output, 'last_hidden_state'):
                embeddings = output.last_hidden_state[:, 0, :]  # CLS token
            else:
                embeddings = output  # Already a tensor
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        all_embeddings.append(embeddings.cpu())
        
        if progress_callback:
            batch_num = i // batch_size + 1
            progress_callback(batch_num, total_batches)
    
    return torch.cat(all_embeddings, dim=0).numpy()


def embed_text(query: str, model, processor, device):
    """
    Generate embedding for text query using SigLIP 2.
    
    Args:
        query: Text query string
        model: SigLIP model
        processor: SigLIP processor
        device: Device string
    
    Returns:
        numpy array of shape (1, EMBEDDING_DIM)
    """
    inputs = processor(
        text=[query],
        padding="max_length",
        return_tensors="pt"
    ).to(model.device)
    
    with torch.no_grad():
        output = model.get_text_features(**inputs)
        # Handle both tensor and BaseModelOutputWithPooling return types
        if hasattr(output, 'pooler_output'):
            embedding = output.pooler_output
        elif hasattr(output, 'last_hidden_state'):
            embedding = output.last_hidden_state[:, 0, :]  # CLS token
        else:
            embedding = output  # Already a tensor
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.cpu().numpy()


# =============================================================================
# INDEXING & SEARCH
# =============================================================================

def index_video(
    video_path: str,
    video_id: str,
    model,
    processor,
    device,
    sample_rate: float = DEFAULT_SAMPLE_RATE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback=None
):
    """
    Index a video into ChromaDB.
    
    Args:
        video_path: Path to video file
        video_id: Unique identifier for this video
        model: SigLIP model
        processor: SigLIP processor
        device: Device string
        sample_rate: Frames per second to extract
        batch_size: Batch size for embedding
        progress_callback: Optional callback(stage, current, total, message)
    
    Returns:
        int: Number of frames indexed
    """
    collection = get_collection()
    
    # Stage 1: Extract frames
    if progress_callback:
        progress_callback("extract", 0, 100, "Extracting frames...")
    
    def extract_progress(current, total):
        if progress_callback:
            progress_callback("extract", current, total, f"Extracting frames: {current}/{total}")
    
    frames, metadata, video_info = extract_frames(video_path, sample_rate, extract_progress)
    
    if len(frames) == 0:
        raise ValueError("No frames extracted from video")
    
    # Stage 2: Generate embeddings
    if progress_callback:
        progress_callback("embed", 0, 100, "Generating embeddings...")
    
    def embed_progress(current, total):
        if progress_callback:
            progress_callback("embed", current, total, f"Embedding batch {current}/{total}")
    
    embeddings = embed_images(frames, model, processor, device, batch_size, embed_progress)
    
    # Stage 3: Store in ChromaDB
    if progress_callback:
        progress_callback("store", 0, 1, "Storing in database...")
    
    ids = [f"{video_id}_frame_{m['frame_idx']}" for m in metadata]
    
    for m in metadata:
        m["video_id"] = video_id
        m["video_path"] = str(video_path)
    
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        metadatas=metadata
    )
    
    if progress_callback:
        progress_callback("done", 1, 1, f"Indexed {len(frames)} frames")
    
    return len(frames)


def search(query: str, model, processor, device, top_k: int = DEFAULT_TOP_K):
    """
    Search for frames matching the text query.
    
    Args:
        query: Natural language search query
        model: SigLIP model
        processor: SigLIP processor
        device: Device string
        top_k: Number of results to return
    
    Returns:
        list: List of result dicts with video_id, timestamp, score, etc.
    """
    collection = get_collection()
    
    if collection.count() == 0:
        return []
    
    query_embedding = embed_text(query, model, processor, device)
    
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
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
            "video_id": meta["video_id"],
            "video_path": meta.get("video_path", ""),
            "timestamp": meta["timestamp"],
            "timestamp_str": meta["timestamp_str"],
            "frame_idx": meta["frame_idx"],
            "score": score,
        })
    
    return search_results
