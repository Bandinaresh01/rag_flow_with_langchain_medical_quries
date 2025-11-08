"""
Medical Knowledge Base Preprocessor
---------------------------------
This script processes medical knowledge documents for a chatbot:
1. Loads PDF files containing medical information
2. Splits content into manageable chunks
3. Creates embeddings using OpenAI
4. Stores vectors in a FAISS index for fast retrieval

Requirements:
- OpenAI API key set in environment: $env:OPENAI_API_KEY = 'your-key'
- PDF file(s) containing medical knowledge
- Python packages: langchain-community, langchain-text-splitters,
                  langchain-openai, faiss-cpu, python-dotenv

Usage:
    python preprocess.py [pdf_path]

    pdf_path: Optional path to PDF file (default: health_knowledge_base.pdf)

Example:
    python preprocess.py medical_data.pdf

The script will create a 'medical_index' directory containing the FAISS index.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

def check_environment() -> bool:
    """Verify that all required environment variables are set."""
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ Error: OPENAI_API_KEY environment variable is not set")
        print("Please set it with: $env:OPENAI_API_KEY = 'your-key-here'")
        return False
    return True

def load_pdf(pdf_path: Path) -> Optional[List]:
    """Load and parse the PDF file."""
    try:
        if not pdf_path.exists():
            print(f"❌ Error: PDF file not found at {pdf_path}")
            return None
        
        print(f"📚 Loading PDF: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()
        print(f"✅ Successfully loaded {len(documents)} pages")
        return documents
    
    except Exception as e:
        print(f"❌ Error loading PDF: {str(e)}")
        return None

def split_documents(documents: List, chunk_size: int = 1000, 
                   chunk_overlap: int = 50) -> Optional[List]:
    """Split documents into smaller chunks for processing."""
    try:
        print("\n📄 Splitting documents into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        
        chunks = splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Preview first chunk
        if chunks:
            print("\n🔹 Example of first chunk:")
            print("-" * 60)
            print(chunks[0].page_content[:500])
            print("-" * 60)
        
        return chunks
    
    except Exception as e:
        print(f"❌ Error splitting documents: {str(e)}")
        return None

def create_vector_store(chunks: List, store_path: str = "medical_index") -> Optional[FAISS]:
    """Create and save a FAISS vector store from document chunks."""
    try:
        print("\n🔤 Creating embeddings and vector store...")
        embeddings = OpenAIEmbeddings()
        
        db = FAISS.from_documents(chunks, embeddings)
        
        # Save the vector store
        db.save_local(store_path)
        print(f"✅ Saved vector store to {store_path}/")
        return db
    
    except Exception as e:
        print(f"❌ Error creating vector store: {str(e)}")
        return None

def test_vector_store(db: FAISS) -> None:
    """Test the vector store with a sample query."""
    try:
        print("\n🔍 Testing vector store with sample query...")
        query = "What should I do if I have a sore throat?"
        results = db.similarity_search(query, k=2)
        
        print("\nSample Results:")
        for i, doc in enumerate(results, 1):
            print(f"\nResult {i}:")
            print("-" * 60)
            print(doc.page_content[:500])
            print("-" * 60)
    
    except Exception as e:
        print(f"❌ Error testing vector store: {str(e)}")

def main():
    """Main execution flow."""
    # Load environment variables
    load_dotenv()

    # Check environment setup
    if not check_environment():
        sys.exit(1)
    
    # Get PDF path from command line or use default
    pdf_path = Path(sys.argv[1] if len(sys.argv) > 1 else "health_knowledge_base.pdf")
    
    # Process the PDF
    documents = load_pdf(pdf_path)
    if not documents:
        sys.exit(1)
    
    # Split into chunks
    chunks = split_documents(documents)
    if not chunks:
        sys.exit(1)
    
    # Create and save vector store
    db = create_vector_store(chunks)
    if not db:
        sys.exit(1)
    
    # Test the vector store
    test_vector_store(db)
    
    print("\n✨ Processing complete! Your medical knowledge base is ready for use.")

if __name__ == "__main__":
    main()
