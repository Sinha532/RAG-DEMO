""" 
Healthcare RAG Chatbot with Multi-Source Search
Main Streamlit application with SQL, PDF, and Web Search using Groq AI 
"""

import streamlit as st
import os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import tempfile

from database import (
    create_patient_database,
    query_database,
    get_patient_info,
    get_database_schema
)
from pdf_rag import create_pdf_rag_system
from web_search import create_web_search_system

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Healthcare RAG Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'db_initialized' not in st.session_state:
    st.session_state.db_initialized = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'show_schema' not in st.session_state:
    st.session_state.show_schema = False
if 'show_samples' not in st.session_state:
    st.session_state.show_samples = False
if 'db_schema_cache' not in st.session_state:
    st.session_state.db_schema_cache = None
if 'sample_patients_cache' not in st.session_state:
    st.session_state.sample_patients_cache = None
if 'pdf_rag_initialized' not in st.session_state:
    st.session_state.pdf_rag_initialized = False
if 'pdf_rag_system' not in st.session_state:
    st.session_state.pdf_rag_system = None
if 'web_search_system' not in st.session_state:
    st.session_state.web_search_system = None

# Load API keys
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')  # Optional

# Validate API keys
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please set it in your .env file.")
    st.stop()

if not MISTRAL_API_KEY:
    st.error("❌ MISTRAL_API_KEY not found. Please set it in your .env file.")
    st.stop()

# Initialize Groq LLM for SQL queries
@st.cache_resource
def initialize_groq_llm():
    """Initialize Groq LLM"""
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=2048,
        timeout=None,
        max_retries=2
    )

groq_llm = initialize_groq_llm()

# Initialize Web Search System
@st.cache_resource
def initialize_web_search():
    """Initialize web search system with Groq"""
    search_provider = "tavily" if TAVILY_API_KEY else "duckduckgo"
    return create_web_search_system(
        groq_api_key=GROQ_API_KEY,
        tavily_api_key=TAVILY_API_KEY,
        search_provider=search_provider,
        model="llama-3.3-70b-versatile"
    )

# SQL Query Prompt Template
sql_prompt = ChatPromptTemplate.from_template(
    """
    You are a SQL expert working with a patient healthcare SQLite database.
    
    Given the following database schema:
    {schema}
    
    And the user's question:
    {question}
    
    Generate a valid SQL query to answer the question.
    Return ONLY the SQL query, nothing else.
    
    Important rules:
    - Use proper SQLite syntax
    - Do NOT use markdown formatting or code blocks
    - Return only the raw SQL query
    - Use appropriate JOINs when needed
    - Handle date formatting properly
    
    SQL Query:
    """
)

# Natural Language Response Prompt
response_prompt = ChatPromptTemplate.from_template(
    """
    Based on the following database query results, provide a clear and concise answer to the user's question.
    
    User Question: {question}
    
    Query Results:
    {results}
    
    Provide a natural language response that directly answers the question.
    If the results are empty, inform the user that no matching records were found.
    
    Answer:
    """
)

def generate_sql_query(question: str, schema: str) -> str:
    """Generate SQL query from natural language question using Groq"""
    chain = sql_prompt | groq_llm
    response = chain.invoke({
        "schema": schema,
        "question": question
    })
    
    query = response.content.strip()
    # Clean up query
    query = query.replace("``````", "").strip()
    return query

def generate_natural_response(question: str, results: pd.DataFrame) -> str:
    """Generate natural language response from query results using Groq"""
    chain = response_prompt | groq_llm
    
    # Convert DataFrame to string
    if results.empty:
        results_str = "No results found"
    else:
        results_str = results.to_string()
    
    response = chain.invoke({
        "question": question,
        "results": results_str
    })
    
    return response.content

def handle_sql_query(question: str):
    """Handle SQL database queries"""
    try:
        # Get schema
        if st.session_state.db_schema_cache is None:
            st.session_state.db_schema_cache = get_database_schema()
        
        schema = st.session_state.db_schema_cache
        
        # Generate SQL query
        with st.spinner("🔍 Generating SQL query with Groq..."):
            sql_query = generate_sql_query(question, schema)
            st.code(sql_query, language="sql")
        
        # Execute query
        with st.spinner("📊 Executing query..."):
            results = query_database(sql_query)
        
        # Display results
        if isinstance(results, pd.DataFrame) and not results.empty:
            st.dataframe(results, use_container_width=True)
            
            # Generate natural language response
            with st.spinner("✍️ Generating response with Groq..."):
                nl_response = generate_natural_response(question, results)
            
            return nl_response
        elif isinstance(results, pd.DataFrame) and results.empty:
            return "No matching records found in the database."
        else:
            return f"Query execution error: {results}"
            
    except Exception as e:
        return f"Error processing SQL query: {str(e)}"

def handle_pdf_query(question: str):
    """Handle PDF document queries"""
    try:
        if not st.session_state.pdf_rag_initialized:
            return "⚠️ Please upload and process a PDF document first."
        
        with st.spinner("🔍 Searching PDF documents..."):
            response = st.session_state.pdf_rag_system.query(question)
            answer = response.get('answer', 'No answer found.')
        
        # Display source documents if available
        if 'context' in response:
            with st.expander("📄 Source Documents"):
                for i, doc in enumerate(response['context'], 1):
                    st.markdown(f"**Document {i}:**")
                    st.text(doc.page_content[:500] + "...")
                    st.divider()
        
        return answer
        
    except Exception as e:
        return f"Error processing PDF query: {str(e)}"

def handle_web_search(question: str, medical_only: bool = False):
    """Handle web search queries"""
    try:
        # Initialize web search if not already done
        if st.session_state.web_search_system is None:
            st.session_state.web_search_system = initialize_web_search()
        
        with st.spinner("🌐 Searching the web with Groq..."):
            if medical_only:
                # Use medical-specific search with trusted domains
                search_results = st.session_state.web_search_system.medical_search(question)
                
                # Generate answer
                chain = response_prompt | groq_llm
                response = chain.invoke({
                    "question": question,
                    "results": search_results
                })
                answer = response.content
            else:
                # Use general search with answer generation
                result = st.session_state.web_search_system.search_and_answer(question)
                answer = result['answer']
                search_results = result['search_results']
        
        # Display search results in expander
        with st.expander("🔍 Search Results"):
            st.text(search_results)
        
        return answer
        
    except Exception as e:
        return f"Error processing web search: {str(e)}"

# Streamlit UI
st.title("🏥 Healthcare RAG Chatbot")
st.markdown("**Multi-Source Intelligence: SQL Database + PDF Documents + Web Search**")
st.markdown("*Powered by LangChain, Groq (Llama 3.3), Mistral AI & DuckDuckGo/Tavily*")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model info
    st.info("🚀 Using **Groq** for ultra-fast inference with Llama 3.3 70B")
    
    # Database initialization
    st.subheader("📊 Database Setup")
    
    if st.button("🔄 Initialize/Reset Database", use_container_width=True):
        with st.spinner("Creating database..."):
            result = create_patient_database()
            st.success(result)
            st.session_state.db_initialized = True
            st.session_state.db_schema_cache = get_database_schema()
            st.session_state.sample_patients_cache = get_patient_info()
    
    # Show database info
    if st.session_state.db_initialized:
        if st.button("📋 Show Database Schema", use_container_width=True):
            st.session_state.show_schema = not st.session_state.show_schema
        
        if st.button("👥 Show Sample Patients", use_container_width=True):
            st.session_state.show_samples = not st.session_state.show_samples
    
    st.divider()
    
    # PDF Upload
    st.subheader("📄 PDF Document Upload")
    uploaded_file = st.file_uploader("Upload PDF for RAG", type=['pdf'])
    
    if uploaded_file is not None:
        if st.button("🚀 Process PDF", use_container_width=True):
            with st.spinner("Processing PDF with Mistral AI..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    # Initialize PDF RAG system
                    if st.session_state.pdf_rag_system is None:
                        st.session_state.pdf_rag_system = create_pdf_rag_system(MISTRAL_API_KEY)
                    
                    # Process PDF
                    st.session_state.pdf_rag_system.process_pdf(tmp_path)
                    st.session_state.pdf_rag_initialized = True
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                    st.success("✅ PDF processed successfully!")
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
    
    st.divider()
    
    # Query mode selection
    st.subheader("🎯 Query Mode")
    query_mode = st.radio(
        "Select query type:",
        ["SQL Database", "PDF Documents", "Web Search", "Medical Web Search"],
        help="Choose your information source"
    )
    
    # Web search provider info
    if query_mode in ["Web Search", "Medical Web Search"]:
        search_provider = "Tavily" if TAVILY_API_KEY else "DuckDuckGo"
        st.info(f"🔍 Using: **{search_provider}** + **Groq Llama 3.3**")
        
        if query_mode == "Medical Web Search":
            st.caption("🏥 Searches trusted medical sources: PubMed, Mayo Clinic, WHO, CDC, NIH")
    
    st.divider()
    
    # Clear chat
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Display schema if toggled
if st.session_state.show_schema:
    with st.expander("📋 Database Schema", expanded=True):
        if st.session_state.db_schema_cache is None:
            st.session_state.db_schema_cache = get_database_schema()
        st.code(st.session_state.db_schema_cache)

# Display sample patients if toggled
if st.session_state.show_samples:
    with st.expander("👥 Sample Patients", expanded=True):
        if st.session_state.sample_patients_cache is None:
            st.session_state.sample_patients_cache = get_patient_info()
        st.dataframe(st.session_state.sample_patients_cache, use_container_width=True)

# Display chat history
st.subheader("💬 Chat History")
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display dataframe if present
        if "dataframe" in message:
            st.dataframe(message["dataframe"], use_container_width=True)

# Chat input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response based on mode
    with st.chat_message("assistant"):
        if query_mode == "SQL Database":
            response = handle_sql_query(prompt)
        elif query_mode == "PDF Documents":
            response = handle_pdf_query(prompt)
        elif query_mode == "Web Search":
            response = handle_web_search(prompt, medical_only=False)
        else:  # Medical Web Search
            response = handle_web_search(prompt, medical_only=True)
        
        st.markdown(response)
    
    # Add assistant response to chat
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        <p>Healthcare RAG Chatbot | Multi-Source Intelligence System</p>
        <p style='font-size: 12px;'>Powered by Groq (Llama 3.3) • SQL Database • PDF Documents • Web Search</p>
    </div>
    """,
    unsafe_allow_html=True
)
