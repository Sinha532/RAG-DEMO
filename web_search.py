"""
Web Search Module using Tavily and DuckDuckGo
Handles internet searches for medical information
Text generation powered by Groq for ultra-fast responses
"""

import os
from typing import List, Dict, Any, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

class WebSearchSystem:
    """Web search system with multiple search providers using Groq LLM"""
    
    def __init__(self, groq_api_key: str, tavily_api_key: Optional[str] = None, 
                 search_provider: str = "duckduckgo", model: str = "llama-3.3-70b-versatile"):
        """
        Initialize web search system with Groq
        
        Args:
            groq_api_key: Groq API key for LLM
            tavily_api_key: Tavily API key (optional)
            search_provider: "tavily" or "duckduckgo" (default)
            model: Groq model name (default: llama-3.3-70b-versatile)
        """
        self.groq_api_key = groq_api_key
        self.tavily_api_key = tavily_api_key
        self.search_provider = search_provider
        
        # Initialize Groq LLM for response generation
        self.llm = ChatGroq(
            model=model,
            groq_api_key=groq_api_key,
            temperature=0.3,
            max_tokens=2048,
            timeout=None,
            max_retries=2
        )
        
        # Initialize search tool based on provider
        if search_provider == "tavily" and tavily_api_key:
            self.search_tool = TavilySearchResults(
                api_key=tavily_api_key,
                max_results=5,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False
            )
        else:
            self.search_tool = DuckDuckGoSearchRun()
        
        # Response generation prompt
        self.response_prompt = ChatPromptTemplate.from_template(
            """
            You are a knowledgeable healthcare assistant with access to internet search results.
            
            User Question: {question}
            
            Search Results:
            {search_results}
            
            Based on the search results above, provide a comprehensive, accurate, and well-structured answer.
            
            Important guidelines:
            - Synthesize information from multiple sources
            - Cite specific facts when available
            - Mention if information is from recent studies or guidelines
            - Be clear about medical disclaimers when appropriate
            - If search results are insufficient, acknowledge limitations
            - Use bullet points for clarity when listing multiple items
            - Keep the response professional and informative
            
            Answer:
            """
        )
    
    def search(self, query: str, max_results: int = 5) -> str:
        """
        Perform web search and return formatted results
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Formatted search results
        """
        try:
            if self.search_provider == "tavily":
                # Tavily returns structured results
                results = self.search_tool.invoke({"query": query})
                
                formatted_results = ""
                for idx, result in enumerate(results[:max_results], 1):
                    formatted_results += f"\n{idx}. {result.get('title', 'No title')}\n"
                    formatted_results += f"   URL: {result.get('url', 'No URL')}\n"
                    formatted_results += f"   Content: {result.get('content', 'No content')}\n"
                
                return formatted_results
            else:
                # DuckDuckGo returns plain text
                results = self.search_tool.invoke(query)
                return results
                
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def search_and_answer(self, question: str) -> Dict[str, Any]:
        """
        Search the web and generate a comprehensive answer using Groq
        
        Args:
            question: User's question
            
        Returns:
            Dictionary with answer and search results
        """
        try:
            # Perform search
            search_results = self.search(question)
            
            # Generate answer using Groq LLM
            chain = self.response_prompt | self.llm
            response = chain.invoke({
                "question": question,
                "search_results": search_results
            })
            
            return {
                "answer": response.content,
                "search_results": search_results,
                "provider": self.search_provider,
                "model": self.llm.model_name
            }
            
        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "search_results": "",
                "provider": self.search_provider,
                "model": "error"
            }
    
    def medical_search(self, query: str, domains: List[str] = None) -> str:
        """
        Perform medical-specific search with domain filtering
        
        Args:
            query: Medical search query
            domains: List of trusted medical domains
            
        Returns:
            Search results from trusted medical sources
        """
        if domains is None:
            domains = [
                "pubmed.ncbi.nlm.nih.gov",
                "mayoclinic.org",
                "who.int",
                "cdc.gov",
                "nih.gov",
                "webmd.com",
                "medicalnewstoday.com"
            ]
        
        # Add domain filtering to query
        if self.search_provider == "tavily" and self.tavily_api_key:
            try:
                results = self.search_tool.invoke({
                    "query": query,
                    "include_domains": domains
                })
                
                formatted_results = ""
                for idx, result in enumerate(results, 1):
                    formatted_results += f"\n{idx}. {result.get('title', 'No title')}\n"
                    formatted_results += f"   URL: {result.get('url', 'No URL')}\n"
                    formatted_results += f"   Content: {result.get('content', 'No content')}\n"
                
                return formatted_results
            except Exception as e:
                return f"Medical search error: {str(e)}"
        else:
            # For DuckDuckGo, append site: operators
            modified_query = f"{query} " + " OR ".join([f"site:{domain}" for domain in domains])
            return self.search(modified_query)


def create_web_search_system(groq_api_key: str, tavily_api_key: Optional[str] = None,
                             search_provider: str = "duckduckgo",
                             model: str = "llama-3.3-70b-versatile") -> WebSearchSystem:
    """
    Factory function to create web search system with Groq
    
    Args:
        groq_api_key: Groq API key
        tavily_api_key: Tavily API key (optional)
        search_provider: Search provider to use
        model: Groq model to use
        
    Returns:
        Configured WebSearchSystem instance
    """
    return WebSearchSystem(groq_api_key, tavily_api_key, search_provider, model)
