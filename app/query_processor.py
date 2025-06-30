"""
Query Processor module.
Handles query validation, classification, and preprocessing using Gemini.
"""

import hashlib
import logging
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

from .config import config
from .models import QueryClassification, QueryStatus

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class QueryProcessor:
    """Handles query validation and classification using Gemini."""
    
    def __init__(self):
        """Initialize the query processor with Gemini models."""
        try:
            # Initialize Gemini chat model for classification
            self.llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                temperature=config.GEMINI_TEMPERATURE,
                google_api_key=config.GOOGLE_API_KEY
            )
            
            # Initialize embeddings model for similarity checking
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=config.GOOGLE_API_KEY
            )
            
            logger.info("Query processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize query processor: {e}")
            raise
    
    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a query as valid or invalid using Gemini.
        
        Args:
            query: The user query to classify
            
        Returns:
            QueryClassification object with classification result
        """
        try:
            # System prompt for query classification
            system_prompt = """You are a query classifier for a web search agent. Your job is to determine if a user query is valid for web searching.

VALID queries are:
- Questions seeking information (e.g., "What is machine learning?")
- Research requests (e.g., "Best programming languages for beginners")
- How-to queries (e.g., "How to install Python")
- Comparison requests (e.g., "Python vs Java")
- Current events/news (e.g., "Latest AI developments")
- Product/service searches (e.g., "Best laptops under $1000")

INVALID queries are:
- Personal commands (e.g., "walk my pet", "add apples to grocery")
- Requests for actions outside web search (e.g., "send email", "book appointment")
- Nonsensical text or gibberish
- Empty or extremely short queries (less than 3 characters)
- Harmful or inappropriate content requests

You MUST respond with ONLY a valid JSON object. Do not include any explanation or text outside the JSON.

The JSON format is:
{
    "status": "valid",
    "confidence": 0.9,
    "reason": "This is a travel query seeking information about tourist destinations",
    "suggested_improvements": null
}

OR

{
    "status": "invalid",
    "confidence": 0.8,
    "reason": "This appears to be a personal command rather than an information query",
    "suggested_improvements": ["Try asking a question that can be answered by web search", "Rephrase as an information-seeking question"]
}"""

            user_prompt = f"Classify this query: '{query}'"
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Get response from Gemini
            response = self.llm.invoke(messages)
            
            # Parse the JSON response
            import json
            import re
            try:
                response_text = response.content.strip()
                
                # Sometimes Gemini adds extra text, try to extract JSON
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    json_text = response_text
                
                result_data = json.loads(json_text)
                
                # Create QueryClassification object
                classification = QueryClassification(
                    query=query,
                    status=QueryStatus(result_data["status"]),
                    confidence=result_data["confidence"],
                    reason=result_data.get("reason"),
                    suggested_improvements=result_data.get("suggested_improvements")
                )
                
                logger.info(f"Query classified as {classification.status} with confidence {classification.confidence}")
                return classification
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response: {e}")
                # Fallback classification
                return self._fallback_classification(query)
                
        except Exception as e:
            logger.error(f"Error classifying query '{query}': {e}")
            return self._fallback_classification(query)
    
    def _fallback_classification(self, query: str) -> QueryClassification:
        """
        Fallback classification when Gemini fails.
        Uses simple heuristics to classify the query.
        """
        # Simple heuristic-based classification
        query_lower = query.lower().strip()
        
        # Check for obviously invalid patterns
        invalid_patterns = [
            "walk my", "add to", "send email", "book appointment",
            "call", "text", "schedule", "remind me"
        ]
        
        is_invalid = any(pattern in query_lower for pattern in invalid_patterns)
        is_too_short = len(query.strip()) < 3
        
        if is_invalid or is_too_short:
            return QueryClassification(
                query=query,
                status=QueryStatus.INVALID,
                confidence=0.7,
                reason="Query appears to be a personal command or too short",
                suggested_improvements=["Try asking a question that can be answered by web search"]
            )
        else:
            return QueryClassification(
                query=query,
                status=QueryStatus.VALID,
                confidence=0.6,
                reason="Basic validation passed (fallback classification)",
                suggested_improvements=None
            )
    
    def generate_query_hash(self, query: str) -> str:
        """
        Generate a unique hash for a query for caching purposes.
        
        Args:
            query: The query to hash
            
        Returns:
            SHA-256 hash of the normalized query
        """
        # Normalize the query (lowercase, strip whitespace)
        normalized_query = query.lower().strip()
        
        # Generate hash
        return hashlib.sha256(normalized_query.encode()).hexdigest()
    
    def generate_embedding(self, query: str) -> list[float]:
        """
        Generate embedding vector for a query for similarity comparison.
        
        Args:
            query: The query to embed
            
        Returns:
            List of floats representing the query embedding
        """
        try:
            # Generate embedding using Gemini
            embedding = self.embeddings.embed_query(query)
            logger.debug(f"Generated embedding for query '{query}' (dimension: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding for query '{query}': {e}")
            # Return empty embedding as fallback
            return []
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize a query for consistent processing.
        
        Args:
            query: The raw query string
            
        Returns:
            Normalized query string
        """
        # Basic normalization
        normalized = query.strip()
        
        # Remove multiple spaces
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized