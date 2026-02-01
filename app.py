"""
Multimodal Semantic Search - Streamlit App

Search inside videos, images, and documents using natural language queries.
Powered by SigLIP 2 (visual) + E5-small (documents) + ChromaDB
"""

import streamlit as st
from pathlib import Path
from PIL import Image

from src.video_search import (
    load_model,
    get_visual_stats,
    clear_visual_database,
    index_video,
    search_visual,
    get_frame_at_index,
    VIDEOS_DIR,
    IMAGES_DIR,
    DOCUMENTS_DIR,
)
from src.image_search import (
    index_image,
    get_image_stats,
    load_image,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.document_search import (
    load_document_model,
    get_document_stats,
    clear_document_database,
    index_document,
    search_documents,
    SUPPORTED_EXTENSIONS as SUPPORTED_DOC_EXTENSIONS,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Multimodal Semantic Search",
    page_icon="🔍",
    layout="wide",
)

# =============================================================================
# MODEL LOADING (CACHED)
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_siglip_model():
    """Load and cache the SigLIP 2 model for visual content."""
    with st.spinner("Loading SigLIP 2 model... (this may take a minute on first run)"):
        model, processor, device = load_model()
    return model, processor, device


@st.cache_resource(show_spinner=False)
def get_document_model_cached():
    """Load and cache the multilingual-e5-small model for documents."""
    with st.spinner("Loading document model (E5-small)..."):
        model = load_document_model()
    return model


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with upload and settings."""
    
    st.sidebar.title("🔍 Semantic Search")
    st.sidebar.markdown("---")
    
    # Number of results slider
    st.sidebar.subheader("Search Settings")
    st.sidebar.slider(
        "Number of results",
        min_value=3,
        max_value=20,
        value=5,
        key="top_k",
        help="How many results to show per category"
    )
    
    st.sidebar.markdown("---")
    
    # Upload sections with tabs
    st.sidebar.subheader("Upload Content")
    upload_tab = st.sidebar.radio(
        "Content type",
        ["Video", "Image", "Document"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if upload_tab == "Video":
        render_video_upload()
    elif upload_tab == "Image":
        render_image_upload()
    else:
        render_document_upload()
    
    st.sidebar.markdown("---")
    
    # Database stats
    render_database_stats()
    
    st.sidebar.markdown("---")
    
    # Clear database buttons
    st.sidebar.subheader("Manage Data")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Clear Visual", width="stretch"):
            clear_visual_database()
            st.sidebar.success("Visual data cleared!")
            st.rerun()
    
    with col2:
        if st.button("Clear Docs", width="stretch"):
            clear_document_database()
            st.sidebar.success("Documents cleared!")
            st.rerun()


def render_video_upload():
    """Render video upload section."""
    uploaded_file = st.sidebar.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video to index for searching",
        key="video_uploader"
    )
    
    video_id = st.sidebar.text_input(
        "Video ID (optional)",
        placeholder="my_video",
        help="A unique name for this video",
        key="video_id_input"
    )
    
    sample_rate = st.sidebar.slider(
        "Sample Rate (fps)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Frames per second to extract"
    )
    
    if st.sidebar.button("Index Video", type="primary", width="stretch"):
        if uploaded_file is not None:
            index_uploaded_video(uploaded_file, video_id, sample_rate)
        else:
            st.sidebar.error("Please upload a video first")


def render_image_upload():
    """Render image upload section."""
    uploaded_files = st.sidebar.file_uploader(
        "Choose image files",
        type=["jpg", "jpeg", "png", "webp", "gif"],
        help="Upload images to index for searching",
        accept_multiple_files=True,
        key="image_uploader"
    )
    
    if st.sidebar.button("Index Images", type="primary", width="stretch"):
        if uploaded_files:
            index_uploaded_images(uploaded_files)
        else:
            st.sidebar.error("Please upload images first")


def render_document_upload():
    """Render document upload section."""
    uploaded_files = st.sidebar.file_uploader(
        "Choose document files",
        type=["txt", "md", "pdf", "docx"],
        help="Upload documents to index for searching",
        accept_multiple_files=True,
        key="doc_uploader"
    )
    
    if st.sidebar.button("Index Documents", type="primary", width="stretch"):
        if uploaded_files:
            index_uploaded_documents(uploaded_files)
        else:
            st.sidebar.error("Please upload documents first")


def render_database_stats():
    """Render database statistics."""
    st.sidebar.subheader("Database Status")
    
    visual_stats = get_visual_stats()
    doc_stats = get_document_stats()
    
    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("Videos", visual_stats["video_count"])
    col2.metric("Images", visual_stats.get("image_count", 0))
    col3.metric("Docs", doc_stats["document_count"])
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Frames", visual_stats["total_frames"])
    col2.metric("Chunks", doc_stats["total_chunks"])
    
    # Expandable details
    if visual_stats["videos"]:
        with st.sidebar.expander("Indexed Videos"):
            for v in visual_stats["videos"]:
                st.write(f"• {v}")
    
    if visual_stats.get("images"):
        with st.sidebar.expander("Indexed Images"):
            for img in visual_stats["images"][:10]:
                st.write(f"• {img}")
            if len(visual_stats["images"]) > 10:
                st.write(f"... and {len(visual_stats['images']) - 10} more")
    
    if doc_stats["documents"]:
        with st.sidebar.expander("Indexed Documents"):
            for doc in doc_stats["documents"]:
                st.write(f"• {doc}")


# =============================================================================
# INDEXING FUNCTIONS
# =============================================================================

def index_uploaded_video(uploaded_file, video_id: str, sample_rate: float):
    """Handle video upload and indexing."""
    if not video_id:
        video_id = Path(uploaded_file.name).stem
    
    video_path = VIDEOS_DIR / uploaded_file.name
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    model, processor, device = get_siglip_model()
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    def progress_callback(stage, current, total, message):
        if stage == "extract":
            progress = current / total * 0.4
        elif stage == "embed":
            progress = 0.4 + (current / total * 0.5)
        elif stage == "store":
            progress = 0.9 + (current / total * 0.1)
        else:
            progress = 1.0
        
        progress_bar.progress(min(progress, 1.0))
        status_text.text(message)
    
    try:
        num_frames = index_video(
            video_path=str(video_path),
            video_id=video_id,
            model=model,
            processor=processor,
            device=device,
            sample_rate=sample_rate,
            progress_callback=progress_callback
        )
        
        progress_bar.progress(1.0)
        status_text.text(f"Indexed {num_frames} frames!")
        st.rerun()
        
    except Exception as e:
        st.sidebar.error(f"Error indexing video: {e}")


def index_uploaded_images(uploaded_files):
    """Handle image upload and indexing."""
    model, processor, device = get_siglip_model()
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    total = len(uploaded_files)
    indexed = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            image_id = Path(uploaded_file.name).stem
            image_path = IMAGES_DIR / uploaded_file.name
            
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            index_image(
                image_path=str(image_path),
                image_id=image_id,
                model=model,
                processor=processor,
                device=device
            )
            
            indexed += 1
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Indexed {indexed}/{total} images")
            
        except Exception as e:
            st.sidebar.warning(f"Error with {uploaded_file.name}: {e}")
    
    progress_bar.progress(1.0)
    status_text.text(f"Indexed {indexed} images!")
    st.rerun()


def index_uploaded_documents(uploaded_files):
    """Handle document upload and indexing."""
    model = get_document_model_cached()
    
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    total = len(uploaded_files)
    indexed = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            doc_id = Path(uploaded_file.name).stem
            doc_path = DOCUMENTS_DIR / uploaded_file.name
            
            with open(doc_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            index_document(
                file_path=str(doc_path),
                doc_id=doc_id,
                model=model
            )
            
            indexed += 1
            progress_bar.progress((i + 1) / total)
            status_text.text(f"Indexed {indexed}/{total} documents")
            
        except Exception as e:
            st.sidebar.warning(f"Error with {uploaded_file.name}: {e}")
    
    progress_bar.progress(1.0)
    status_text.text(f"Indexed {indexed} documents!")
    st.rerun()


# =============================================================================
# MAIN AREA
# =============================================================================

def render_main_area():
    """Render the main search and results area."""
    
    st.title("🔍 Multimodal Semantic Search")
    st.markdown("Search across videos, images, and documents using natural language")
    
    # Preload models (cached, so only loads once)
    siglip_model, siglip_processor, device = get_siglip_model()
    doc_model = get_document_model_cached()
    
    # Device info
    device_info = f"Using: {device.upper()}"
    if device == "cuda":
        import torch
        device_info += f" ({torch.cuda.get_device_name(0)})"
    st.caption(device_info)
    
    st.markdown("---")
    
    # Search input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "What are you looking for?",
            placeholder="e.g., a person talking, outdoor scene, machine learning...",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button("Search", type="primary", width="stretch")
    
    top_k = st.session_state.get("top_k", 5)
    
    # Perform search
    if search_button and query:
        perform_search(query, siglip_model, siglip_processor, device, top_k)
    elif query:
        perform_search(query, siglip_model, siglip_processor, device, top_k)


def perform_search(query: str, siglip_model, siglip_processor, device, top_k: int):
    """Perform search across all content types and display results with tabs."""
    
    visual_stats = get_visual_stats()
    doc_stats = get_document_stats()
    
    has_visual = visual_stats["total_items"] > 0
    has_docs = doc_stats["total_chunks"] > 0
    
    if not has_visual and not has_docs:
        st.warning("No content indexed yet. Upload videos, images, or documents in the sidebar to get started.")
        return
    
    st.markdown(f"### Results for: *\"{query}\"*")
    
    # Perform all searches upfront
    visual_results = []
    doc_results = []
    
    if has_visual:
        with st.spinner("Searching visual content..."):
            visual_results = search_visual(query, siglip_model, siglip_processor, device, top_k)
    
    if has_docs:
        doc_model = get_document_model_cached()
        with st.spinner("Searching documents..."):
            doc_results = search_documents(query, doc_model, top_k)
    
    # Separate video and image results
    video_results = [r for r in visual_results if r.get("type") == "video"]
    image_results = [r for r in visual_results if r.get("type") == "image"]
    
    # Check if we have any results
    if not visual_results and not doc_results:
        st.info("No results found. Try a different query.")
        return
    
    # Count results for tab labels
    video_count = len(video_results)
    image_count = len(image_results)
    doc_count = len(doc_results)
    total_count = video_count + image_count + doc_count
    
    # Create tabs with result counts
    tab_all, tab_videos, tab_images, tab_docs = st.tabs([
        f"All ({total_count})",
        f"Videos ({video_count})",
        f"Images ({image_count})",
        f"Documents ({doc_count})"
    ])
    
    # All tab - combined view
    with tab_all:
        if video_results:
            st.subheader(f"🎬 Videos ({video_count})")
            display_video_results(video_results)
        if image_results:
            st.subheader(f"🖼️ Images ({image_count})")
            display_image_results(image_results)
        if doc_results:
            st.subheader(f"📄 Documents ({doc_count})")
            display_document_results(doc_results)
    
    # Videos tab
    with tab_videos:
        if video_results:
            display_video_results(video_results)
        else:
            st.info("No video results found.")
    
    # Images tab
    with tab_images:
        if image_results:
            display_image_results(image_results)
        else:
            st.info("No image results found.")
    
    # Documents tab
    with tab_docs:
        if doc_results:
            display_document_results(doc_results)
        else:
            st.info("No document results found.")


def display_video_results(results):
    """Display video search results in a grid."""
    cols_per_row = 5
    
    for row_start in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for i, col in enumerate(cols):
            result_idx = row_start + i
            if result_idx >= len(results):
                break
            
            result = results[result_idx]
            
            with col:
                frame = get_frame_at_index(result["source_path"], result.get("frame_idx", 0))
                if frame is not None:
                    st.image(frame, width="stretch")
                
                st.markdown(f"**#{result['rank']}** | {result.get('timestamp_str', '')}")
                st.caption(f"Score: {result['score']:.1%}")
                st.caption(f"Video: {result.get('video_id', result['source_id'])}")
                
                if st.button("▶ Play", key=f"play_video_{result_idx}", width="stretch"):
                    st.session_state.selected_result = result
    
    # Video player for selected result
    if "selected_result" in st.session_state and st.session_state.selected_result:
        result = st.session_state.selected_result
        if result.get("type") == "video":
            st.markdown("---")
            st.markdown(f"### Playing: {result.get('video_id', '')} @ {result.get('timestamp_str', '')}")
            
            video_path = result.get("video_path") or result.get("source_path", "")
            if Path(video_path).exists():
                st.video(video_path, start_time=int(result.get("timestamp", 0)))
            else:
                st.error(f"Video file not found: {video_path}")


def display_image_results(results):
    """Display image search results in a grid."""
    cols_per_row = 5
    
    for row_start in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for i, col in enumerate(cols):
            result_idx = row_start + i
            if result_idx >= len(results):
                break
            
            result = results[result_idx]
            
            with col:
                try:
                    img = Image.open(result["source_path"])
                    st.image(img, width="stretch")
                except Exception:
                    st.write("Image not found")
                
                st.markdown(f"**#{result['rank']}**")
                st.caption(f"Score: {result['score']:.1%}")
                st.caption(f"{result['source_id']}")


def display_document_results(results):
    """Display document search results as cards."""
    for result in results:
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Document header
                doc_name = result["doc_id"]
                file_type = result.get("file_type", "").upper()
                page_info = f" (Page {result['page']})" if result.get("page") else ""
                chunk_info = f"Chunk {result['chunk_idx'] + 1}"
                
                st.markdown(f"**{doc_name}** `{file_type}` {page_info} - {chunk_info}")
                
                # Text preview
                text_preview = result.get("text", "")[:300]
                if len(result.get("text", "")) > 300:
                    text_preview += "..."
                st.text(text_preview)
            
            with col2:
                st.metric("Score", f"{result['score']:.1%}")
            
            st.markdown("---")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main app entry point."""
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
