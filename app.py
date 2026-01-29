"""
Video Semantic Search - Streamlit App

Search inside videos using natural language queries powered by SigLIP 2 + ChromaDB
"""

import streamlit as st
from pathlib import Path
import tempfile
import os

from src.video_search import (
    load_model,
    get_database_stats,
    clear_database,
    index_video,
    search,
    get_frame_at_index,
    VIDEOS_DIR,
)

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Video Semantic Search",
    page_icon="🎬",
    layout="wide",
)

# =============================================================================
# MODEL LOADING (CACHED)
# =============================================================================

@st.cache_resource(show_spinner=False)
def get_model():
    """Load and cache the SigLIP model."""
    with st.spinner("Loading SigLIP 2 model... (this may take a minute on first run)"):
        model, processor, device = load_model()
    return model, processor, device


# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with upload and settings."""
    
    st.sidebar.title("🎬 Video Search")
    st.sidebar.markdown("---")
    
    # Number of results slider
    st.sidebar.subheader("🔢 Search Settings")
    st.sidebar.slider(
        "Number of results",
        min_value=3,
        max_value=20,
        value=3,
        key="top_k",
        help="How many results to show"
    )
    
    st.sidebar.markdown("---")
    
    # Video Upload
    st.sidebar.subheader("📁 Upload Video")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a video to index for searching"
    )
    
    # Video ID input
    video_id = st.sidebar.text_input(
        "Video ID (optional)",
        placeholder="my_video",
        help="A unique name for this video"
    )
    
    # Sample rate slider
    sample_rate = st.sidebar.slider(
        "Sample Rate (fps)",
        min_value=0.5,
        max_value=5.0,
        value=1.0,
        step=0.5,
        help="Frames per second to extract. Higher = more frames, slower indexing"
    )
    
    # Index button
    if st.sidebar.button("🔄 Index Video", type="primary", width="stretch"):
        if uploaded_file is not None:
            index_uploaded_video(uploaded_file, video_id, sample_rate)
        else:
            st.sidebar.error("Please upload a video first")
    
    st.sidebar.markdown("---")
    
    # Database stats
    st.sidebar.subheader("📊 Database Status")
    stats = get_database_stats()
    
    col1, col2 = st.sidebar.columns(2)
    col1.metric("Videos", stats["video_count"])
    col2.metric("Frames", stats["total_frames"])
    
    if stats["videos"]:
        with st.sidebar.expander("Indexed Videos"):
            for v in stats["videos"]:
                st.write(f"• {v}")
    
    st.sidebar.markdown("---")
    
    # Clear database button
    if st.sidebar.button("🗑️ Clear Database", width="stretch"):
        clear_database()
        st.sidebar.success("Database cleared!")
        st.rerun()


def index_uploaded_video(uploaded_file, video_id: str, sample_rate: float):
    """Handle video upload and indexing."""
    
    # Generate video ID if not provided
    if not video_id:
        video_id = Path(uploaded_file.name).stem
    
    # Save uploaded file to disk (required for OpenCV)
    video_path = VIDEOS_DIR / uploaded_file.name
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Get model
    model, processor, device = get_model()
    
    # Create progress elements
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    def progress_callback(stage, current, total, message):
        if stage == "extract":
            progress = current / total * 0.4  # 0-40%
        elif stage == "embed":
            progress = 0.4 + (current / total * 0.5)  # 40-90%
        elif stage == "store":
            progress = 0.9 + (current / total * 0.1)  # 90-100%
        else:
            progress = 1.0
        
        progress_bar.progress(progress)
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
        status_text.text(f"✅ Indexed {num_frames} frames!")
        st.rerun()
        
    except Exception as e:
        st.sidebar.error(f"Error indexing video: {e}")


# =============================================================================
# MAIN AREA
# =============================================================================

def render_main_area():
    """Render the main search and results area."""
    
    st.title("🔍 Video Semantic Search")
    st.markdown("Search inside your videos using natural language")
    
    # Check if model is loaded
    model, processor, device = get_model()
    
    # Device info
    device_info = f"🖥️ Using: {device.upper()}"
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
            placeholder="e.g., a person talking, outdoor scene, text on screen...",
            label_visibility="collapsed"
        )
    with col2:
        search_button = st.button("🔍 Search", type="primary", width="stretch")
    
    # Get number of results from session state (set in sidebar)
    top_k = st.session_state.get("top_k", 3)
    
    # Perform search
    if search_button and query:
        perform_search(query, model, processor, device, top_k)
    elif query:
        # Also search on Enter (text_input submits)
        perform_search(query, model, processor, device, top_k)


def perform_search(query: str, model, processor, device, top_k: int):
    """Perform search and display results."""
    
    stats = get_database_stats()
    if stats["total_frames"] == 0:
        st.warning("📭 No videos indexed yet. Upload a video in the sidebar to get started.")
        return
    
    with st.spinner("Searching..."):
        results = search(query, model, processor, device, top_k)
    
    if not results:
        st.info("No results found. Try a different query.")
        return
    
    st.markdown(f"### Results for: *\"{query}\"*")
    st.markdown(f"Found {len(results)} matching frames")
    
    # Display results in a grid
    cols_per_row = 5
    
    for row_start in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for i, col in enumerate(cols):
            result_idx = row_start + i
            if result_idx >= len(results):
                break
            
            result = results[result_idx]
            
            with col:
                # Load thumbnail
                frame = get_frame_at_index(result["video_path"], result["frame_idx"])
                
                if frame is not None:
                    st.image(frame, width="stretch")
                
                # Result info
                st.markdown(f"**#{result['rank']}** | {result['timestamp_str']}")
                st.caption(f"Score: {result['score']:.1%}")
                st.caption(f"📹 {result['video_id']}")
                
                # Play button
                if st.button(f"▶️ Play", key=f"play_{result_idx}", width="stretch"):
                    st.session_state.selected_result = result
    
    # Video player for selected result
    if "selected_result" in st.session_state and st.session_state.selected_result:
        result = st.session_state.selected_result
        st.markdown("---")
        st.markdown(f"### 🎬 Playing: {result['video_id']} @ {result['timestamp_str']}")
        
        video_path = result["video_path"]
        if Path(video_path).exists():
            st.video(video_path, start_time=int(result["timestamp"]))
        else:
            st.error(f"Video file not found: {video_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main app entry point."""
    render_sidebar()
    render_main_area()


if __name__ == "__main__":
    main()
