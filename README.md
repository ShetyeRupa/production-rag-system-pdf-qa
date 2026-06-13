# Production RAG System for Research Paper Q&A

[![GitHub Repository](https://img.shields.io/badge/GitHub-View_on_GitHub-blue?logo=github)](https://github.com/ShetyeRupa/production-rag-system-pdf-qa)

A complete, production-ready Retrieval-Augmented Generation (RAG) system that enables natural language question answering over your own PDF documents. Built with LangChain, FAISS, and local LLMs - no API keys required, no cloud costs.

## Overview

This system implements a full RAG pipeline that:
1. Loads and processes PDF documents from a local directory
2. Splits documents into intelligent chunks for optimal retrieval
3. Creates vector embeddings using sentence transformers
4. Stores embeddings in a FAISS vector database for fast similarity search
5. Retrieves relevant context based on user questions
6. Generates grounded answers using a local LLM (microsoft/phi-2)

Unlike cloud-based solutions (OpenAI, Pinecone), this system runs entirely on your local machine with zero recurring costs and complete data privacy.

## Features

| Feature | Description |
|---------|-------------|
| **Local Execution** | Runs entirely on your machine - no API calls, no cloud dependencies |
| **Zero Cost** | No subscription fees, no usage-based pricing |
| **Privacy Preserving** | Your documents never leave your computer |
| **Production Logging** | Comprehensive logging for monitoring and debugging |
| **Error Handling** | Robust error handling throughout the pipeline |
| **Source Tracking** | Returns source documents with every answer for verification |
| **Configurable** | Adjust chunk size, overlap, retrieval count, and models |
| **Multi-PDF Support** | Process and query across multiple PDF documents simultaneously |

## Architecture

```
PDF Documents → Text Chunking → Embeddings → FAISS Vector Store
                                                    ↓
User Question → Embedding → Similarity Search → Retrieved Context
                                                    ↓
                                    LLM Generation → Grounded Answer
```

### Components

| Component | Technology | Purpose |
|-----------|------------|---------|
| Document Loading | PyPDFLoader | Extract text from PDF files |
| Text Splitting | RecursiveCharacterTextSplitter | Chunk documents for context window limits |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Convert text to vector representations |
| Vector Store | FAISS | Efficient similarity search |
| LLM | microsoft/phi-2 (2.7B parameters) | Generate answers from retrieved context |
| Orchestration | LangChain | Pipeline coordination |

## Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|----------------|-------|
| Python | 3.9+ | Recommended: 3.10 or 3.11 |
| RAM | 8 GB | 16 GB recommended for larger document sets |
| Storage | 10 GB free | For model caching and vector storage |
| Internet | Initial setup only | For downloading models (first run only) |

## Installation

### Step 1: Clone or Create Project

```bash
git clone https://github.com/ShetyeRupa/production-rag-system-pdf-qa.git
cd production-rag-system-pdf-qa
```

### Step 2: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Create `requirements.txt`:

```txt
langchain-core>=0.3.0
langchain-text-splitters>=0.3.0
langchain-community>=0.3.0
pypdf>=3.0.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.0
transformers>=4.36.0
torch>=2.0.0
huggingface-hub>=0.20.0
```

Then install:

```bash
pip install -r requirements.txt
```

### Step 4: Download the Code

Save the provided `rag_app.py` in your project directory.

## Configuration

The system can be configured by modifying parameters in the `main()` function:

```python
system = ProductionRAGSystem(
    pdf_directory="./pdfs",           # Directory containing PDF files
    chunk_size=1000,                  # Size of each text chunk (characters)
    chunk_overlap=200,                # Overlap between consecutive chunks
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    llm_model="microsoft/phi-2",      # Local LLM to use
    top_k_results=3                   # Number of chunks to retrieve per query
)
```

### Parameter Guide

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `chunk_size` | Characters per chunk | 500-1500 (smaller = more precise) |
| `chunk_overlap` | Overlap between chunks | 10-20% of chunk_size |
| `top_k_results` | Chunks retrieved per query | 3-5 |

## Usage

### Step 1: Add Your PDFs

Create a `pdfs` folder in your project directory and add your PDF files:

```bash
mkdir pdfs
# Copy your PDF files into ./pdfs/
```

### Step 2: Run the System

```bash
python rag_app.py
```

### Step 3: Ask Questions

Once initialized, type your questions at the `QUESTION:` prompt:

```
QUESTION: What factors influenced yellow taxi ridership decline?

[PROCESSING] Generating answer...

ANSWER: The decline in yellow taxi ridership can be attributed to...
SOURCES: Retrieved 3 relevant document chunks
```

### Interactive Commands

| Command | Action |
|---------|--------|
| `quit` or `exit` | End the session |
| `status` | Display system configuration |
| `Ctrl+C` | Force exit |

## Example Output

```
============================================================
PRODUCTION RAG SYSTEM - INITIALIZATION
============================================================

[1/4] Loading PDF documents...
[INFO] Found 6 PDF file(s)
[INFO] Loading: research_paper_1.pdf
[INFO] Loading: research_paper_2.pdf
[INFO] Loaded 62 document pages total

[2/4] Splitting documents into chunks...
[INFO] Created 321 text chunks from documents

[3/4] Creating vector database...
[INFO] Generating embeddings...
[INFO] Vector store created successfully

[4/4] Loading language model...
[INFO] Loading LLM model: microsoft/phi-2
[INFO] LLM pipeline ready

============================================================
SYSTEM READY
============================================================
PDF Directory: /path/to/project/pdfs
Total Documents: 62 pages
Total Chunks: 321
Retrieval: Top 3 chunks per query
LLM Model: microsoft/phi-2
============================================================

QUESTION: What is the main finding of the research?
ANSWER: The research found that congestion surcharges significantly reduced...
```

## System Demonstration

Below are screenshots showing the RAG system in action with real research papers about NYC taxi services.

### System Initialization

The system successfully loads 6 research papers (62 pages), creates 321 searchable chunks, and initializes the Microsoft Phi-2 LLM.

![System Initialization](screenshots/Start.png)

### Question 1: Factors Influencing Taxi Ridership Decline

The system retrieves relevant chunks about surcharges, competition from Uber/Lyft, and consumer preference changes.

**Question:** What factors influenced yellow taxi ridership decline?

![Question 1](screenshots/Q1.png)
![Answer 1 Part 1](screenshots/Q1_A1.png)

### Question 2: Congestion Pricing Effects

The system correctly identifies when information is not available in the documents and summarizes key findings from the research.

**Question:** How does congestion pricing affect taxi demand?

![Question 2](screenshots/Q2.png)
![Answer 2 Part 1](screenshots/Q2_A2-1.png)
![Answer 2 Part 2](screenshots/Q2_A2-2.png)

### Question 3: Forecasting Methods

The system accurately extracts specific methodology information including ARIMA models and AR(1) with RMSE values.

**Question:** What methods were used to forecast taxi ridership?

![Question 3](screenshots/Q3.png)
![Answer 3 Part 1](screenshots/Q3_A3-1.png)
![Answer 3 Part 2](screenshots/Q3_A3-2.png)

### Question 4: COVID-19 Impact on Taxis

The system retrieves and synthesizes information from academic papers about pandemic-related mobility changes.

**Question:** What is the relationship between COVID-19 and taxi usage?

![Question 4](screenshots/Q4.png)
![Answer 4 Part 1](screenshots/Q4_A4-1.png)
![Answer 4 Part 2](screenshots/Q4_A4-2.png)

## Project Structure

```
my-rag-system/
├── rag_app.py              # Main application code
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── pdfs/                  # Add your PDF files here
│   ├── paper1.pdf
│   └── paper2.pdf
└── venv/                  # Virtual environment (not tracked)
```

## Troubleshooting

### Issue: ModuleNotFoundError

**Solution:** Reinstall dependencies with the correct versions:

```bash
pip uninstall langchain langchain-community -y
pip install -r requirements.txt
```

### Issue: Slow First Run

**Cause:** The system downloads embedding and LLM models on first run (several GB).

**Solution:** This is normal. Subsequent runs will use cached models and start quickly.

### Issue: Out of Memory

**Solution:** Reduce document batch size or use a smaller LLM:

```python
llm_model="microsoft/phi-1_5"  # Smaller, faster model
```

### Issue: No PDFs Found

**Solution:** Ensure your `pdfs` folder exists and contains `.pdf` files.

## Performance Metrics

| Metric | Value |
|--------|-------|
| Document Processing | ~2 seconds per 10 pages |
| Embedding Generation | ~0.5 seconds per 100 chunks |
| Query Response Time | 3-8 seconds (depends on hardware) |
| Model Size (phi-2) | 5.6 GB |
| Embedding Model Size | 90 MB |

## Future Enhancements

| Enhancement | Description | Difficulty |
|-------------|-------------|------------|
| Web Interface | Add Streamlit or Gradio UI | Easy |
| Multiple LLM Support | Switch between models dynamically | Easy |
| Document Upload API | Add REST endpoint for file uploads | Medium |
| Persistent Storage | Save vector store to disk between runs | Easy |
| Batch Querying | Process multiple questions from CSV | Easy |
| Docker Deployment | Containerize for cloud deployment | Medium |

## License

MIT License - Free for personal and commercial use.

## Author

Rupali Ravindra Shetye
Master's in Artificial Intelligence, LIU Brooklyn

## Acknowledgments

- LangChain for orchestration framework
- Hugging Face for transformers and embeddings
- FAISS for efficient vector search
- Microsoft for phi-2 LLM

## Contact

**Rupali Ravindra Shetye**  
Master's in Artificial Intelligence, LIU Brooklyn

**GitHub Repository:** [https://github.com/ShetyeRupa/production-rag-system-pdf-qa](https://github.com/ShetyeRupa/production-rag-system-pdf-qa)

For questions, feedback, or collaboration opportunities, please open an issue on GitHub.