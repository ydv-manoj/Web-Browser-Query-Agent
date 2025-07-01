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
                google_api_key=config.GEMINI_API_KEY,  # Fixed: Use GEMINI_API_KEY
                timeout=30
            )
            
            # Initialize embeddings model for similarity checking with updated model
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=config.GEMINI_EMBEDDING_MODEL,  # Use config setting
                google_api_key=config.GEMINI_API_KEY,  # Fixed: Use GEMINI_API_KEY
                timeout=30
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
            # Simplified system prompt for better reliability
            system_prompt = """You are a query classifier. Determine if a user query is valid for web searching.

VALID queries seek information that can be found on the web:
- Questions: "What is machine learning?"
- Research: "Best programming languages 2024"
- How-to: "How to install Python"
- Comparisons: "Python vs Java"
- News/current events: "Latest AI developments"
- Product searches: "Best laptops under $1000"

INVALID queries are:
- Personal commands: "walk my pet", "send email", "book appointment"
- Nonsensical text or gibberish
- Too short (less than 3 characters)
- Harmful content requests

Respond ONLY with valid JSON:

{
    "status": "valid",
    "confidence": 0.9,
    "reason": "Clear information-seeking query about travel destinations"
}

OR

{
    "status": "invalid", 
    "confidence": 0.8,
    "reason": "Personal command, not information query"
}"""

            user_prompt = f"Classify: '{query}'"
            
            # Create messages
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            # Get response from Gemini with timeout
            try:
                response = self.llm.invoke(messages)
            except Exception as api_error:
                logger.error(f"Gemini API error: {api_error}")
                return self._fallback_classification(query)
            
            # Parse the JSON response
            import json
            import re
            try:
                response_text = response.content.strip()
                
                # Extract JSON from response
                json_match = re.search(r'\{[^}]*\}', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(0)
                else:
                    json_text = response_text
                
                result_data = json.loads(json_text)
                
                # Validate required fields
                if "status" not in result_data or "confidence" not in result_data:
                    logger.warning("Invalid response format from Gemini, using fallback")
                    return self._fallback_classification(query)
                
                # Create QueryClassification object
                classification = QueryClassification(
                    query=query,
                    status=QueryStatus(result_data["status"]),
                    confidence=float(result_data["confidence"]),
                    reason=result_data.get("reason", "Classified by Gemini"),
                    suggested_improvements=result_data.get("suggested_improvements")
                )
                
                logger.info(f"Query classified as {classification.status} with confidence {classification.confidence}")
                return classification
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Failed to parse Gemini response: {e}")
                logger.debug(f"Raw response: {response.content}")
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
            "call", "text", "schedule", "remind me", "set alarm",
            "play music", "turn on", "turn off"
        ]
        
        # Check for valid question patterns
        valid_patterns = [
            "what", "how", "why", "when", "where", "who", "which",
            "best", "top", "compare", "vs", "versus", "difference",
            "explain", "guide", "tutorial", "learn", "help"
        ]
        
        is_invalid = any(pattern in query_lower for pattern in invalid_patterns)
        is_too_short = len(query.strip()) < 3
        has_valid_pattern = any(pattern in query_lower for pattern in valid_patterns)
        
        if is_invalid or is_too_short:
            return QueryClassification(
                query=query,
                status=QueryStatus.INVALID,
                confidence=0.7,
                reason="Query appears to be a personal command or too short",
                suggested_improvements=["Try asking a question that can be answered by web search"]
            )
        elif has_valid_pattern:
            return QueryClassification(
                query=query,
                status=QueryStatus.VALID,
                confidence=0.8,
                reason="Contains question words or information-seeking patterns",
                suggested_improvements=None
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
            # Generate embedding using Gemini with timeout and error handling
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