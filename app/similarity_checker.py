import logging
import json
import re
from typing import List, Optional, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import config
from .models import SimilarQuery

logger = logging.getLogger(__name__)


class SimilarityChecker:
    """
    Enhanced similarity checker that combines embedding-based similarity 
    with LLM-based semantic validation using Gemini AI.
    """

    def __init__(self) -> None:
        self.similarity_threshold: float = config.SIMILARITY_THRESHOLD
        self.semantic_similarity_threshold: float = 0.8  # New threshold for LLM validation
        
        # Initialize Gemini for semantic similarity checking
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                temperature=0.1,  # Low temperature for consistent similarity assessment
                google_api_key=config.GEMINI_API_KEY,
                timeout=30
            )
            logger.info("Gemini LLM initialized for semantic similarity checking")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini LLM: {e}")
            self.llm = None
        
        logger.info(
            "SimilarityChecker initialised (embedding_threshold=%.3f, semantic_threshold=%.3f)",
            self.similarity_threshold, self.semantic_similarity_threshold
        )

    def _check_semantic_similarity(self, query1: str, query2: str) -> Dict[str, Any]:
        """
        Use Gemini AI to check if two queries are semantically similar.
        
        Args:
            query1: First query
            query2: Second query
            
        Returns:
            Dict with 'is_similar' (bool), 'confidence' (float), 'reason' (str)
        """
        if not self.llm:
            logger.warning("Gemini LLM not available, falling back to embedding-only similarity")
            return {"is_similar": True, "confidence": 1.0, "reason": "LLM not available"}
        
        try:
            system_prompt = """You are a semantic similarity expert. Compare two user queries and determine if they are asking for similar information or have similar intent.

Consider these factors:
- Topic similarity (e.g., both about programming, both about cooking)
- Intent similarity (e.g., both asking for resources, both asking for tutorials)
- Scope similarity (e.g., both general questions, both specific to a technology)
- Context similarity (e.g., both for beginners, both for 2025)

Examples of SIMILAR queries:
- "Python tutorials for beginners" vs "Learn Python programming basics"
- "Best restaurants in NYC" vs "Top places to eat in New York City"
- "JavaScript frameworks 2025" vs "Modern JS frameworks to learn"

Examples of NOT SIMILAR queries:
- "Python tutorials" vs "Java tutorials" (different languages)
- "Docker resources" vs "HTML resources" (completely different topics)
- "How to cook pasta" vs "Best pasta restaurants" (different intents)

Respond ONLY with valid JSON:

{
    "is_similar": true,
    "confidence": 0.85,
    "reason": "Both queries are asking for learning resources about programming languages"
}

OR

{
    "is_similar": false,
    "confidence": 0.95,
    "reason": "Queries are about different technologies - Docker vs HTML"
}"""

            user_prompt = f"""Compare these two queries for semantic similarity:

Query 1: "{query1}"
Query 2: "{query2}"

Are these queries semantically similar?"""

            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]

            response = self.llm.invoke(messages)
            
            # Parse the JSON response
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
                if "is_similar" not in result_data or "confidence" not in result_data:
                    logger.warning("Invalid response format from Gemini, using fallback")
                    return {"is_similar": True, "confidence": 0.5, "reason": "Invalid LLM response"}
                
                return {
                    "is_similar": bool(result_data["is_similar"]),
                    "confidence": float(result_data["confidence"]),
                    "reason": result_data.get("reason", "Assessed by Gemini")
                }
                
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.error(f"Failed to parse Gemini similarity response: {e}")
                logger.debug(f"Raw response: {response.content}")
                return {"is_similar": True, "confidence": 0.5, "reason": "Parse error"}
                
        except Exception as e:
            logger.error(f"Error checking semantic similarity: {e}")
            return {"is_similar": True, "confidence": 0.5, "reason": f"LLM error: {str(e)}"}

    def find_similar_query(
        self,
        query_embedding: List[float],
        vector_store,
        current_query: str = None
    ) -> Optional[SimilarQuery]:
        """
        Find similar query with both embedding and semantic validation.
        
        Args:
            query_embedding: Embedding vector of the current query
            vector_store: Vector store instance
            current_query: The actual query text for semantic comparison
            
        Returns:
            SimilarQuery if a semantically similar query is found, None otherwise
        """
        try:
            # Step 1: Get embedding-based matches
            matches = vector_store.find_similar_queries(
                embedding=query_embedding,
                threshold=self.similarity_threshold,
                limit=5  # Get top 5 to have options for semantic validation
            )
            
            if not matches:
                logger.info("No similar queries above embedding threshold %.3f",
                            self.similarity_threshold)
                return None

            logger.info(f"Found {len(matches)} embedding-based matches, validating with Gemini...")

            # Step 2: Validate each match with Gemini (if current_query is provided)
            if current_query and self.llm:
                for hit in matches:
                    cached_query = hit["original_query"]
                    embedding_similarity = hit["similarity_score"]
                    
                    logger.info("Checking semantic similarity: '%s' vs '%s' (embedding: %.3f)",
                              current_query, cached_query, embedding_similarity)
                    
                    # Get semantic similarity assessment
                    semantic_result = self._check_semantic_similarity(current_query, cached_query)
                    
                    logger.info("Semantic assessment: similar=%s, confidence=%.3f, reason='%s'",
                              semantic_result["is_similar"], 
                              semantic_result["confidence"],
                              semantic_result["reason"])
                    
                    # Check if semantically similar with sufficient confidence
                    if (semantic_result["is_similar"] and 
                        semantic_result["confidence"] >= self.semantic_similarity_threshold):
                        
                        logger.info("✅ Found semantically similar query: '%s' (embedding: %.3f, semantic: %.3f)",
                                  cached_query, embedding_similarity, semantic_result["confidence"])
                        
                        return SimilarQuery(
                            original_query=cached_query,
                            similarity_score=embedding_similarity,
                            cached_result=hit["cached_result"],
                            semantic_similarity=semantic_result["confidence"],
                            semantic_reason=semantic_result["reason"]
                        )
                    else:
                        logger.info("❌ Query not semantically similar: '%s' - %s",
                                  cached_query, semantic_result["reason"])
                
                logger.info("No semantically similar queries found after LLM validation")
                return None
            
            else:
                # Fallback to embedding-only if no current_query or LLM unavailable
                hit = matches[0]
                logger.info("Using embedding-only similarity: '%s' (%.3f)",
                          hit["original_query"], hit["similarity_score"])
                
                return SimilarQuery(
                    original_query=hit["original_query"],
                    similarity_score=hit["similarity_score"],
                    cached_result=hit["cached_result"]
                )
                
        except Exception as exc:
            logger.error("Similarity lookup failed: %s", exc)
            return None

    def find_multiple_similar_queries(
        self,
        query_embedding: List[float],
        vector_store,
        current_query: str = None,
        limit: int = 5
    ) -> List[SimilarQuery]:
        """
        Find multiple similar queries with semantic validation.
        """
        try:
            # Get embedding-based matches
            matches = vector_store.find_similar_queries(
                embedding=query_embedding,
                threshold=self.similarity_threshold,
                limit=limit * 2  # Get more candidates for semantic filtering
            )
            
            if not matches:
                return []
            
            similar_queries = []
            
            # Validate with Gemini if current_query provided
            if current_query and self.llm:
                for hit in matches:
                    if len(similar_queries) >= limit:
                        break
                        
                    cached_query = hit["original_query"]
                    embedding_similarity = hit["similarity_score"]
                    
                    # Get semantic similarity assessment
                    semantic_result = self._check_semantic_similarity(current_query, cached_query)
                    
                    # Check if semantically similar
                    if (semantic_result["is_similar"] and 
                        semantic_result["confidence"] >= self.semantic_similarity_threshold):
                        
                        similar_queries.append(SimilarQuery(
                            original_query=cached_query,
                            similarity_score=embedding_similarity,
                            cached_result=hit["cached_result"],
                            semantic_similarity=semantic_result["confidence"],
                            semantic_reason=semantic_result["reason"]
                        ))
            else:
                # Fallback to embedding-only
                similar_queries = [
                    SimilarQuery(
                        original_query=m["original_query"],
                        similarity_score=m["similarity_score"],
                        cached_result=m["cached_result"]
                    )
                    for m in matches[:limit]
                ]
            
            return similar_queries
            
        except Exception as exc:
            logger.error("Similarity batch lookup failed: %s", exc)
            return []

    def get_similarity_stats(self, vector_store) -> Dict[str, Any]:
        """Get similarity statistics including semantic validation stats."""
        try:
            stats = vector_store.get_cache_stats()
            total = stats.cache_hits + stats.cache_misses
            hit_rate = (stats.cache_hits / total) * 100 if total else 0.0
            
            return {
                "embedding_threshold": self.similarity_threshold,
                "semantic_threshold": self.semantic_similarity_threshold,
                "llm_available": self.llm is not None,
                "total_queries_processed": total,
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "similarity_hit_rate_percent": hit_rate,
                "total_stored_queries": getattr(stats, 'total_entries', 0)
            }
        except Exception as exc:
            logger.error("Stats generation failed: %s", exc)
            return {
                "embedding_threshold": self.similarity_threshold,
                "semantic_threshold": self.semantic_similarity_threshold,
                "llm_available": self.llm is not None,
                "total_queries_processed": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "similarity_hit_rate_percent": 0.0,
                "total_stored_queries": 0
            }