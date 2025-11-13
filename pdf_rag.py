"""
PDF RAG Module using Mistral AI
Handles PDF document processing, embedding, and retrieval
"""

import os
from typing import List, Optional
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document


class PDFRAGSystem:
    """PDF RAG system powered by Mistral AI"""
    
    def __init__(self, api_key: str, model: str = "mistral-medium-latest"):
        """
        Initialize PDF RAG system
        
        Args:
            api_key: Mistral API key
            model: Mistral model name
        """
        self.api_key = api_key
        self.model = model
        
        # Initialize Mistral LLM
        self.llm = ChatMistralAI(
            model=self.model,
            mistral_api_key=self.api_key,
            temperature=0.7
        )
        
        # Initialize Mistral embeddings
        self.embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=self.api_key
        )
        
        self.vectorstore = None
        self.retrieval_chain = None
        
        # Define prompt template
        self.prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            
            <context>
            {context}
            </context>
            
            Question: {input}
            
            Answer:
            """
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load PDF document
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document objects
        """
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        return documents
    
    def split_documents(
        self, 
        documents: List[Document], 
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of split documents
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        splits = text_splitter.split_documents(documents)
        return splits
    
    def create_vectorstore(self, documents: List[Document]) -> FAISS:
        """
        Create FAISS vectorstore from documents
        
        Args:
            documents: List of documents to embed
            
        Returns:
            FAISS vectorstore
        """
        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        return self.vectorstore
    
    def setup_retrieval_chain(self, k: int = 5):
        """
        Setup retrieval chain for QA
        
        Args:
            k: Number of documents to retrieve
        """
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized. Call create_vectorstore first.")
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k}
        )
        
        # Create document chain
        document_chain = create_stuff_documents_chain(
            llm=self.llm,
            prompt=self.prompt
        )
        
        # Create retrieval chain
        self.retrieval_chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=document_chain
        )
    
    def process_pdf(
        self, 
        pdf_path: str, 
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        k: int = 5
    ):
        """
        Complete PDF processing pipeline
        
        Args:
            pdf_path: Path to PDF file
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            k: Number of documents to retrieve
        """
        # Load PDF
        documents = self.load_pdf(pdf_path)
        
        # Split documents
        splits = self.split_documents(
            documents, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Create vectorstore
        self.create_vectorstore(splits)
        
        # Setup retrieval chain
        self.setup_retrieval_chain(k=k)
    
    def query(self, question: str) -> dict:
        """
        Query the RAG system
        
        Args:
            question: User question
            
        Returns:
            Dictionary with answer and context
        """
        if self.retrieval_chain is None:
            raise ValueError("Retrieval chain not initialized. Call process_pdf first.")
        
        response = self.retrieval_chain.invoke({"input": question})
        return response
    
    def save_vectorstore(self, path: str):
        """Save vectorstore to disk"""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not initialized.")
        self.vectorstore.save_local(path)
    
    def load_vectorstore(self, path: str):
        """Load vectorstore from disk"""
        self.vectorstore = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True
        )


def create_pdf_rag_system(api_key: str, model: str = "mistral-large-latest") -> PDFRAGSystem:
    """
    Factory function to create PDF RAG system
    
    Args:
        api_key: Mistral API key
        model: Mistral model name
        
    Returns:
        PDFRAGSystem instance
    """
    return PDFRAGSystem(api_key=api_key, model=model)


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()
    
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    
    # Initialize system
    pdf_rag = create_pdf_rag_system(MISTRAL_API_KEY)
    
    # Process PDF
    pdf_rag.process_pdf("sample.pdf")
    
    # Query
    response = pdf_rag.query("What is this document about?")
    print(response['answer'])
