"""
Image Semantic Search Module

This module provides image search functionality:
- Image loading and embedding using SigLIP 2
- Vector storage and search using ChromaDB (shared with video)
"""

import torch
from pathlib import Path
from typing import Optional, Callable
from PIL import Image
import chromadb

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = Path("./data")
IMAGES_DIR = DATA_DIR / "images"
DB_DIR = DATA_DIR / "chroma_db"

# Ensure directories exist
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


# =============================================================================
# CHROMADB (Shared with video_search)
# =============================================================================

_chroma_client = None
_visual_collection = None


def get_chroma_client():
    """Get or create ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(DB_DIR))
    return _chroma_client


def get_visual_collection():
    """Get or create the visual content collection (shared with video)."""
    global _visual_collection
    if _visual_collection is None:
        client = get_chroma_client()
        _visual_collection = client.get_or_create_collection(
            name="visual_content",
            metadata={"hnsw:space": "cosine"}
        )
    return _visual_collection


def get_image_stats():
    """
    Get statistics about indexed images only.
    
    Returns:
        dict: Statistics including total images and image list
    """
    collection = get_visual_collection()
    total_items = collection.count()
    
    stats = {
        "total_images": 0,
        "images": [],
    }
    
    if total_items > 0:
        # Query for images only
        all_data = collection.get(
            where={"type": "image"},
            include=["metadatas"]
        )
        
        if all_data["metadatas"]:
            image_ids = set(m.get("source_id", "unknown") for m in all_data["metadatas"])
            stats["images"] = list(image_ids)
            stats["total_images"] = len(image_ids)
    
    return stats


# =============================================================================
# IMAGE LOADING
# =============================================================================

def load_image(image_path: str) -> Image.Image:
    """Load an image file and convert to RGB."""
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def get_image_info(image_path: str) -> dict:
    """Get image metadata."""
    img = Image.open(image_path)
    return {
        "width": img.width,
        "height": img.height,
        "format": img.format,
        "mode": img.mode
    }


# =============================================================================
# INDEXING
# =============================================================================

def index_image(
    image_path: str,
    image_id: str,
    model,
    processor,
    device: str,
    progress_callback: Optional[Callable] = None
) -> int:
    """
    Index a single image into ChromaDB.
    
    Args:
        image_path: Path to image file
        image_id: Unique identifier for this image
        model: SigLIP model
        processor: SigLIP processor
        device: Device string
        progress_callback: Optional callback(stage, current, total, message)
        
    Returns:
        int: 1 if successful
    """
    collection = get_visual_collection()
    
    # Stage 1: Load image
    if progress_callback:
        progress_callback("load", 0, 100, "Loading image...")
    
    path = Path(image_path)
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    
    img = load_image(image_path)
    img_info = get_image_info(image_path)
    
    if progress_callback:
        progress_callback("load", 100, 100, "Image loaded")
    
    # Stage 2: Generate embedding
    if progress_callback:
        progress_callback("embed", 0, 100, "Generating embedding...")
    
    inputs = processor(images=[img], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        output = model.get_image_features(**inputs)
        # Handle both tensor and BaseModelOutputWithPooling return types
        if hasattr(output, 'pooler_output'):
            embedding = output.pooler_output
        elif hasattr(output, 'last_hidden_state'):
            embedding = output.last_hidden_state[:, 0, :]
        else:
            embedding = output
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    if progress_callback:
        progress_callback("embed", 100, 100, "Embedding generated")
    
    # Stage 3: Store in ChromaDB
    if progress_callback:
        progress_callback("store", 0, 1, "Storing in database...")
    
    metadata = {
        "type": "image",
        "source_id": image_id,
        "source_path": str(image_path),
        "width": img_info["width"],
        "height": img_info["height"],
    }
    
    collection.add(
        ids=[f"image_{image_id}"],
        embeddings=embedding.cpu().numpy().tolist(),
        metadatas=[metadata]
    )
    
    if progress_callback:
        progress_callback("done", 1, 1, "Image indexed!")
    
    return 1


def index_images_batch(
    image_paths: list[str],
    model,
    processor,
    device: str,
    batch_size: int = 16,
    progress_callback: Optional[Callable] = None
) -> int:
    """
    Index multiple images in batches.
    
    Args:
        image_paths: List of image paths
        model: SigLIP model
        processor: SigLIP processor
        device: Device string
        batch_size: Number of images per batch
        progress_callback: Optional callback(current, total, message)
        
    Returns:
        int: Number of images indexed
    """
    collection = get_visual_collection()
    total = len(image_paths)
    indexed = 0
    
    for i in range(0, total, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load images
        images = []
        valid_paths = []
        for path in batch_paths:
            try:
                img = load_image(path)
                images.append(img)
                valid_paths.append(path)
            except Exception:
                continue
        
        if not images:
            continue
        
        # Generate embeddings
        inputs = processor(images=images, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.get_image_features(**inputs)
            if hasattr(output, 'pooler_output'):
                embeddings = output.pooler_output
            elif hasattr(output, 'last_hidden_state'):
                embeddings = output.last_hidden_state[:, 0, :]
            else:
                embeddings = output
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        
        # Store in ChromaDB
        ids = []
        metadatas = []
        for j, path in enumerate(valid_paths):
            p = Path(path)
            image_id = p.stem
            img_info = get_image_info(path)
            
            ids.append(f"image_{image_id}_{i+j}")
            metadatas.append({
                "type": "image",
                "source_id": image_id,
                "source_path": str(path),
                "width": img_info["width"],
                "height": img_info["height"],
            })
        
        collection.add(
            ids=ids,
            embeddings=embeddings.cpu().numpy().tolist(),
            metadatas=metadatas
        )
        
        indexed += len(valid_paths)
        
        if progress_callback:
            progress_callback(indexed, total, f"Indexed {indexed}/{total} images")
    
    return indexed
