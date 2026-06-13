# app.py - Streamlit UI for Hugging Face Spaces deployment
import streamlit as st
import os
import tempfile
import json
import datetime
import time
from pathlib import Path
from rag_app import ModernRAGSystem

# ============================================================================
# Page Configuration
# ============================================================================

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide"
)

# ============================================================================
# Custom CSS for Professional Styling
# ============================================================================

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #94a3b8;
        margin: 0.5rem 0 0 0;
    }
    .custom-card {
        background: #f8fafc;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid #e2e8f0;
    }
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        color: #94a3b8;
        font-size: 0.7rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# Custom Header
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1>📚 Research Paper Q&amp;A System</h1>
    <p>Ask questions about your research papers - powered by RAG, FAISS, and Microsoft Phi-2</p>
    <p style="font-size: 0.85rem; margin-top: 0.75rem;">Built by Rupali Shetye | LIU Brooklyn, Master's in Artificial Intelligence</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# Initialize Session State
# ============================================================================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "pdf_count" not in st.session_state:
    st.session_state.pdf_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "show_success" not in st.session_state:
    st.session_state.show_success = False

# ============================================================================
# Sidebar for configuration
# ============================================================================

with st.sidebar:
    st.header("⚙️ Configuration")
    
    uploaded_files = st.file_uploader(
        "📄 Upload PDF Files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload your research papers or documents"
    )
    
    with st.expander("ℹ️ How to Use"):
        st.markdown("""
        1. Upload PDF files (research papers, documents)
        2. Click 'Initialize RAG System'
        3. Wait 2-5 minutes for first-time setup
        4. Ask questions in the chat box
        
        **Note:** First request after inactivity takes 30-60 seconds (cold start).
        """)
    
    # Show success message if just initialized
    if st.session_state.show_success:
        st.success("✅ RAG system initialized successfully!")
    
    if st.button("🚀 Initialize RAG System", type="primary"):
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("📁 Saving uploaded files...")
                progress_bar.progress(20)
                
                pdf_dir = Path("./uploaded_pdfs")
                pdf_dir.mkdir(exist_ok=True)
                
                saved_files = []
                for uploaded_file in uploaded_files:
                    file_path = pdf_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    saved_files.append(uploaded_file.name)
                
                st.session_state.pdf_count = len(saved_files)
                
                status_text.text("🤖 Loading AI model and processing PDFs...")
                progress_bar.progress(40)
                
                st.session_state.rag_system = ModernRAGSystem(
                    pdf_directory=str(pdf_dir),
                    chunk_size=1000,
                    chunk_overlap=200,
                    top_k_results=3
                )
                
                status_text.text("🔨 Building vector database...")
                progress_bar.progress(60)
                
                if st.session_state.rag_system.initialize():
                    progress_bar.progress(100)
                    st.session_state.initialized = True
                    st.session_state.show_success = True
                    
                    try:
                        if hasattr(st.session_state.rag_system, 'vector_store') and st.session_state.rag_system.vector_store:
                            if hasattr(st.session_state.rag_system.vector_store, 'index'):
                                st.session_state.chunk_count = st.session_state.rag_system.vector_store.index.ntotal
                    except Exception:
                        st.session_state.chunk_count = len(saved_files) * 50
                    
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()
                else:
                    st.error("❌ Failed to initialize. Check your PDFs.")
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please upload at least one PDF file.")
    
    st.divider()
    
    # System status display
    if st.session_state.initialized:
        st.markdown("### 📊 System Status")
        st.markdown(f"""
        <div class="custom-card">
            <strong>📄 Documents:</strong> {st.session_state.pdf_count}<br>
            <strong>🔍 Chunks:</strong> {st.session_state.chunk_count}<br>
            <strong>🤖 LLM:</strong> microsoft/phi-2 (2.7B)
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    
    with col2:
        if st.button("💾 Download Chat", use_container_width=True):
            if st.session_state.messages:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                export_data = {
                    "timestamp": timestamp,
                    "messages": st.session_state.messages
                }
                json_str = json.dumps(export_data, indent=2)
                st.download_button(
                    label="📥 Save",
                    data=json_str,
                    file_name=f"chat_history_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
            else:
                st.info("No chat history")

# ============================================================================
# Main chat interface
# ============================================================================

st.markdown("### 💬 Conversation")

if not st.session_state.messages:
    st.markdown("""
    <div class="info-box">
        <strong>💡 Welcome to RAG Research Assistant!</strong><br>
        Upload PDF files, initialize the system, then ask questions about your documents.
    </div>
    """, unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.caption(f"Source {i+1}: {source[:200]}...")

# ============================================================================
# Quick Demo Questions
# ============================================================================

if st.session_state.initialized:
    st.markdown("---")
    st.markdown("### 🔍 Quick Demo Questions")
    st.caption("Click any question to try it out")
    
    col1, col2 = st.columns(2)
    
    demo_questions = [
        "What factors influenced yellow taxi ridership decline?",
        "How does congestion pricing affect taxi demand?",
        "What methods were used to forecast taxi ridership?",
        "What is the relationship between COVID-19 and taxi usage?"
    ]
    
    for idx, q in enumerate(demo_questions):
        target_col = col1 if idx % 2 == 0 else col2
        with target_col:
            if st.button(f"📌 {q}", key=f"demo_{idx}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                with st.chat_message("user"):
                    st.markdown(q)
                
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Generating answer..."):
                        try:
                            answer, sources = st.session_state.rag_system.ask_question(q)
                            st.markdown(answer)
                            if sources:
                                with st.expander("📚 Retrieved Sources"):
                                    for i, doc in enumerate(sources):
                                        st.caption(f"**Chunk {i+1}:** {doc.page_content[:300]}...")
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": [doc.page_content for doc in sources] if sources else []
                            })
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                st.rerun()

# ============================================================================
# Chat input
# ============================================================================

if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.initialized:
        st.warning("⚠️ Please upload PDFs and initialize the RAG system first.")
        st.stop()
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching documents and generating answer..."):
            try:
                answer, sources = st.session_state.rag_system.ask_question(prompt)
                st.markdown(answer)
                
                if sources:
                    with st.expander("📚 Retrieved Sources"):
                        for i, doc in enumerate(sources):
                            st.caption(f"**Chunk {i+1}:** {doc.page_content[:300]}...")
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": [doc.page_content for doc in sources] if sources else []
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")

# ============================================================================
# Footer
# ============================================================================

st.markdown("""
<div class="footer">
    Built with LangChain, FAISS, and Microsoft Phi-2 | Deployed on Hugging Face Spaces
</div>
""", unsafe_allow_html=True)
