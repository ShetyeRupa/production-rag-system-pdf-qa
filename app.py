# app.py - Streamlit UI for Hugging Face Spaces deployment
import streamlit as st
import os
import tempfile
from pathlib import Path
from rag_app import ModernRAGSystem

st.set_page_config(
    page_title="RAG Research Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Research Paper Q&A System")
st.caption("Ask questions about your research papers - powered by RAG, FAISS, and Microsoft Phi-2")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_system" not in st.session_state:
    st.session_state.rag_system = None
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload your research papers or documents"
    )
    
    if st.button("Initialize RAG System"):
        if uploaded_files:
            with st.spinner("Processing PDFs and building vector database... This may take 3-5 minutes on first run."):
                # Save uploaded files temporarily
                pdf_dir = Path("./uploaded_pdfs")
                pdf_dir.mkdir(exist_ok=True)
                
                for uploaded_file in uploaded_files:
                    file_path = pdf_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Initialize RAG system with uploaded PDFs
                st.session_state.rag_system = ModernRAGSystem(
                    pdf_directory=str(pdf_dir),
                    chunk_size=1000,
                    chunk_overlap=200,
                    top_k_results=3
                )
                
                if st.session_state.rag_system.initialize():
                    st.session_state.initialized = True
                    st.success("RAG system initialized successfully!")
                else:
                    st.error("Failed to initialize. Check your PDFs.")
        else:
            st.warning("Please upload at least one PDF file.")
    
    st.divider()
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("View Sources"):
                for i, source in enumerate(message["sources"]):
                    st.caption(f"Source {i+1}: {source[:200]}...")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.initialized:
        st.warning("Please upload PDFs and initialize the RAG system first.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            try:
                answer, sources = st.session_state.rag_system.ask_question(prompt)
                st.markdown(answer)
                
                # Show sources
                if sources:
                    with st.expander("📚 Retrieved Sources"):
                        for i, doc in enumerate(sources):
                            st.caption(f"**Chunk {i+1}:** {doc.page_content[:300]}...")
                
                # Store in session
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": [doc.page_content for doc in sources]
                })
            except Exception as e:
                st.error(f"Error: {str(e)}")