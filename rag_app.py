# rag_app.py - Production Ready RAG System for LangChain 2026
# No deprecated imports. Uses only modern, supported LangChain components.
# Complete Retrieval-Augmented Generation system for PDF documents
# Uses local models only - no API keys required

import os
import logging
from pathlib import Path
from typing import List, Tuple

# Modern LangChain imports (no langchain.chains or langchain-community)
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Standalone integration packages
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline

# Transformers for local LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModernRAGSystem:
    """
    Modern RAG System using LangChain 2026 architecture.
    No deprecated chains or community packages.
    Uses a manual RAG loop for maximum compatibility and control.
    """
    
    def __init__(
        self,
        pdf_directory: str = "./pdfs",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        llm_model: str = "microsoft/phi-2",
        top_k_results: int = 3
    ):
        self.pdf_directory = Path(pdf_directory)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        self.llm_model_name = llm_model
        self.top_k_results = top_k_results
        
        self.vector_store = None
        self.llm_pipeline = None
        self.is_ready = False
    
    def create_pdf_directory(self) -> None:
        """Create the PDF directory if it doesn't exist."""
        if not self.pdf_directory.exists():
            self.pdf_directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created PDF directory at {self.pdf_directory}")
            print(f"\n[INFO] Created PDF directory: {self.pdf_directory}")
            print("[INFO] Please add your PDF files to this directory and run the program again.")
    
    def load_documents(self) -> List[Document]:
        """
        Load all PDF documents from the configured directory.
        
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Scanning for PDF files in {self.pdf_directory}")
        pdf_files = list(self.pdf_directory.glob("*.pdf"))
        
        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {self.pdf_directory}. "
                f"Please add at least one PDF file to this directory."
            )
        
        logger.info(f"Found {len(pdf_files)} PDF file(s)")
        print(f"\n[INFO] Found {len(pdf_files)} PDF file(s)")
        
        documents = []
        for pdf_file in pdf_files:
            logger.info(f"Loading: {pdf_file.name}")
            print(f"[INFO] Loading: {pdf_file.name}")
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
            
        logger.info(f"Loaded {len(documents)} document pages total")
        return documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of document chunks
        """
        logger.info(f"Splitting documents into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})")
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} document chunks")
        print(f"[INFO] Created {len(chunks)} text chunks from documents")
        
        return chunks
    
    def create_vector_store(self, chunks: List[Document]) -> FAISS:
        """
        Create a FAISS vector store from document chunks.
        """
        logger.info(f"Creating embeddings using model: {self.embedding_model_name}")
        print(f"[INFO] Generating embeddings (this may take a moment)...")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        vector_store = FAISS.from_documents(chunks, embeddings)
        logger.info("Vector store created successfully")
        print("[INFO] Vector store created successfully")
        
        return vector_store
    
    def create_llm_pipeline(self) -> HuggingFacePipeline:
        """
        Create a local LLM pipeline for text generation.
        """
        logger.info(f"Loading LLM model: {self.llm_model_name}")
        print(f"\n[INFO] Loading LLM model: {self.llm_model_name}")
        print("[INFO] This may take 2-5 minutes on first run (model caching enabled)")
        
        tokenizer = AutoTokenizer.from_pretrained(
            self.llm_model_name,
            trust_remote_code=True
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_model_name,
            trust_remote_code=True
        )
        
        logger.info("LLM model loaded, creating pipeline")
        print("[INFO] LLM model loaded, initializing pipeline...")
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            do_sample=True,
            top_p=0.95
        )
        
        logger.info("LLM pipeline ready")
        print("[INFO] LLM pipeline ready")
        
        return HuggingFacePipeline(pipeline=pipe)
    
    def format_docs(self, docs: List[Document]) -> str:
        """Format retrieved documents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def ask_question(self, question: str) -> Tuple[str, List[Document]]:
        """
        Ask a question using manual RAG loop.
        No deprecated chains - uses direct retrieval and generation.
        
        Returns:
            Tuple containing (answer_text, source_documents)
        """
        if not self.is_ready:
            raise RuntimeError("RAG system not initialized. Call initialize() first.")
        
        if not question or not question.strip():
            return "Please provide a valid question.", []
        
        logger.info(f"Processing question: {question[:100]}...")
        
        try:
            # Step 1: Retrieve relevant documents
            retriever = self.vector_store.as_retriever(
                search_kwargs={"k": self.top_k_results}
            )
            retrieved_docs = retriever.invoke(question)
            
            # Step 2: Format context
            context = self.format_docs(retrieved_docs)
            
            # Step 3: Create prompt
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that answers questions based ONLY on the provided context.
If the context does not contain the answer, say "I don't have enough information to answer that based on the provided documents."
Do not use any prior knowledge.

Context:
{context}"""),
                ("human", "{question}")
            ])
            
            # Step 4: Create manual RAG chain
            chain = (
                {"context": lambda x: context, "question": lambda x: x}
                | prompt_template
                | self.llm_pipeline
                | StrOutputParser()
            )
            
            # Step 5: Generate answer
            answer = chain.invoke(question)
            
            logger.info(f"Answer generated using {len(retrieved_docs)} source chunks")
            return answer, retrieved_docs
            
        except Exception as e:
            logger.error(f"Question processing failed: {str(e)}")
            return f"Error processing question: {str(e)}", []
    
    def initialize(self) -> bool:
        """
        Initialize the complete RAG system.
        """
        print("\n" + "="*60)
        print("MODERN RAG SYSTEM - INITIALIZATION")
        print("="*60)
        
        try:
            self.create_pdf_directory()
            pdf_files = list(self.pdf_directory.glob("*.pdf"))
            if not pdf_files:
                print("\n[ERROR] No PDF files found.")
                print(f"[ACTION] Please add PDF files to: {self.pdf_directory.absolute()}")
                return False
            
            print("\n[1/4] Loading PDF documents...")
            documents = self.load_documents()
            
            print("\n[2/4] Splitting documents into chunks...")
            chunks = self.split_documents(documents)
            
            print("\n[3/4] Creating vector database...")
            self.vector_store = self.create_vector_store(chunks)
            
            print("\n[4/4] Loading language model...")
            self.llm_pipeline = self.create_llm_pipeline()
            
            self.is_ready = True
            
            print("\n" + "="*60)
            print("SYSTEM READY")
            print("="*60)
            print(f"PDF Directory: {self.pdf_directory.absolute()}")
            print(f"Total Documents: {len(documents)} pages")
            print(f"Total Chunks: {len(chunks)}")
            print(f"Retrieval: Top {self.top_k_results} chunks per query")
            print(f"LLM Model: {self.llm_model_name}")
            print("="*60 + "\n")
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")
            print(f"\n[ERROR] Failed to initialize RAG system: {str(e)}")
            return False
    
    def run_interactive_session(self) -> None:
        """Run an interactive command-line session."""
        if not self.is_ready:
            print("\n[ERROR] System not initialized.")
            return
        
        print("\n" + "="*60)
        print("INTERACTIVE SESSION")
        print("="*60)
        print("Commands:")
        print("  - Type your question and press Enter")
        print("  - Type 'quit' or 'exit' to end the session")
        print("  - Type 'status' to see system information")
        print("="*60 + "\n")
        
        while True:
            try:
                question = input("QUESTION: ").strip()
                
                if not question:
                    continue
                    
                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n[INFO] Session ended. Goodbye!")
                    break
                    
                if question.lower() == 'status':
                    self.print_status()
                    continue
                
                print("\n[PROCESSING] Generating answer...")
                answer, sources = self.ask_question(question)
                
                print("\n" + "-"*60)
                print(f"ANSWER: {answer}")
                print("-"*60)
                print(f"SOURCES: Retrieved {len(sources)} relevant document chunks")
                print("-"*60 + "\n")
                
            except KeyboardInterrupt:
                print("\n\n[INFO] Session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n[ERROR] {str(e)}\n")
    
    def print_status(self) -> None:
        """Print current system status."""
        print("\n" + "="*50)
        print("SYSTEM STATUS")
        print("="*50)
        print(f"Ready: {self.is_ready}")
        print(f"PDF Directory: {self.pdf_directory.absolute()}")
        print(f"Embedding Model: {self.embedding_model_name}")
        print(f"LLM Model: {self.llm_model_name}")
        print(f"Chunk Size: {self.chunk_size}")
        print(f"Chunk Overlap: {self.chunk_overlap}")
        print(f"Top K Results: {self.top_k_results}")
        print("="*50 + "\n")


def main():
    """Main entry point."""
    system = ModernRAGSystem(
        pdf_directory="./pdfs",
        chunk_size=1000,
        chunk_overlap=200,
        top_k_results=3
    )
    
    if system.initialize():
        system.run_interactive_session()
    else:
        print("\n[ERROR] System initialization failed.")
        print("[ACTION] Please ensure:")
        print("  1. You have installed the required packages")
        print("  2. You have added PDF files to the './pdfs' folder")
        print("  3. You have sufficient disk space for model caching")


if __name__ == "__main__":
    main()