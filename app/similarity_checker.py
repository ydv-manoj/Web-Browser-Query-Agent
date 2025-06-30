"""
Similarity Checker module.
Handles query similarity detection using embeddings and cosine similarity.
"""

import logging
import numpy as np
from typing import List, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity

from .config import config
from .models import SimilarQuery, QueryResult, CacheEntry

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class SimilarityChecker:
    """Handles query similarity detection using embedding vectors."""
    
    def __init__(self):
        """Initialize the similarity checker."""
        logger.info("Similarity checker initialized")
    
    def find_similar_query(
        self, 
        query_embedding: List[float], 
        cached_entries: List[CacheEntry]
    ) -> Optional[SimilarQuery]:
        """
        Find the most similar cached query if similarity exceeds threshold.
        
        Args:
            query_embedding: Embedding vector of the new query
            cached_entries: List of cached query entries with embeddings
            
        Returns:
            SimilarQuery object if a similar query is found, None otherwise
        """
        try:
            if not query_embedding or not cached_entries:
                logger.debug("No embedding or cached entries available for similarity check")
                return None
            
            # Find the most similar query
            best_match = self._find_best_match(query_embedding, cached_entries)
            
            if best_match and best_match[1] >= config.SIMILARITY_THRESHOLD:
                logger.info(f"Found similar query with similarity score: {best_match[1]:.3f}")
                
                return SimilarQuery(
                    original_query=best_match[0].query,
                    similarity_score=best_match[1],
                    cached_result=best_match[0].result
                )
            else:
                similarity_score = best_match[1] if best_match else 0.0
                logger.debug(f"No similar query found. Best similarity: {similarity_score:.3f}")
                return None
                
        except Exception as e:
            logger.error(f"Error finding similar query: {e}")
            return None
    
    def _find_best_match(
        self, 
        query_embedding: List[float], 
        cached_entries: List[CacheEntry]
    ) -> Optional[Tuple[CacheEntry, float]]:
        """
        Find the cached entry with highest similarity to the query.
        
        Args:
            query_embedding: Embedding vector of the new query
            cached_entries: List of cached query entries
            
        Returns:
            Tuple of (best_matching_entry, similarity_score) or None
        """
        try:
            # Filter entries that have valid embeddings
            valid_entries = [
                entry for entry in cached_entries 
                if entry.embedding and len(entry.embedding) == len(query_embedding)
            ]
            
            if not valid_entries:
                logger.debug("No valid cached entries with embeddings found")
                return None
            
            # Convert to numpy arrays for efficient computation
            query_vector = np.array(query_embedding).reshape(1, -1)
            cached_vectors = np.array([entry.embedding for entry in valid_entries])
            
            # Compute cosine similarities
            similarities = cosine_similarity(query_vector, cached_vectors)[0]
            
            # Find the best match
            best_index = np.argmax(similarities)
            best_similarity = similarities[best_index]
            best_entry = valid_entries[best_index]
            
            logger.debug(f"Best similarity found: {best_similarity:.3f} with query: '{best_entry.query}'")
            
            return (best_entry, float(best_similarity))
            
        except Exception as e:
            logger.error(f"Error computing similarities: {e}")
            return None
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        try:
            if not embedding1 or not embedding2:
                return 0.0
            
            if len(embedding1) != len(embedding2):
                logger.warning("Embeddings have different dimensions")
                return 0.0
            
            # Convert to numpy arrays
            vec1 = np.array(embedding1).reshape(1, -1)
            vec2 = np.array(embedding2).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(vec1, vec2)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def rank_similar_queries(
        self, 
        query_embedding: List[float], 
        cached_entries: List[CacheEntry],
        top_k: int = 5
    ) -> List[Tuple[CacheEntry, float]]:
        """
        Rank cached queries by similarity to the input query.
        
        Args:
            query_embedding: Embedding vector of the new query
            cached_entries: List of cached query entries
            top_k: Number of top similar queries to return
            
        Returns:
            List of tuples (CacheEntry, similarity_score) sorted by similarity
        """
        try:
            if not query_embedding or not cached_entries:
                return []
            
            # Calculate similarities for all valid entries
            similarities = []
            
            for entry in cached_entries:
                if entry.embedding and len(entry.embedding) == len(query_embedding):
                    similarity = self.calculate_similarity(query_embedding, entry.embedding)
                    similarities.append((entry, similarity))
            
            # Sort by similarity score (descending)
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k results
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Error ranking similar queries: {e}")
            return []
    
    def is_duplicate_query(
        self, 
        query_embedding: List[float], 
        cached_entries: List[CacheEntry],
        duplicate_threshold: float = 0.95
    ) -> Optional[CacheEntry]:
        """
        Check if a query is essentially a duplicate of a cached query.
        
        Args:
            query_embedding: Embedding vector of the new query
            cached_entries: List of cached query entries
            duplicate_threshold: Threshold for considering queries as duplicates
            
        Returns:
            CacheEntry if duplicate found, None otherwise
        """
        try:
            best_match = self._find_best_match(query_embedding, cached_entries)
            
            if best_match and best_match[1] >= duplicate_threshold:
                logger.info(f"Duplicate query detected with similarity: {best_match[1]:.3f}")
                return best_match[0]
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking for duplicate query: {e}")
            return None
    
    def get_similarity_statistics(
        self, 
        query_embedding: List[float], 
        cached_entries: List[CacheEntry]
    ) -> dict:
        """
        Get statistical information about query similarity.
        
        Args:
            query_embedding: Embedding vector of the new query
            cached_entries: List of cached query entries
            
        Returns:
            Dictionary with similarity statistics
        """
        try:
            if not query_embedding or not cached_entries:
                return {
                    "total_cached_queries": 0,
                    "valid_embeddings": 0,
                    "max_similarity": 0.0,
                    "avg_similarity": 0.0,
                    "min_similarity": 0.0
                }
            
            similarities = []
            valid_count = 0
            
            for entry in cached_entries:
                if entry.embedding and len(entry.embedding) == len(query_embedding):
                    similarity = self.calculate_similarity(query_embedding, entry.embedding)
                    similarities.append(similarity)
                    valid_count += 1
            
            if similarities:
                max_sim = max(similarities)
                avg_sim = sum(similarities) / len(similarities)
                min_sim = min(similarities)
            else:
                max_sim = avg_sim = min_sim = 0.0
            
            return {
                "total_cached_queries": len(cached_entries),
                "valid_embeddings": valid_count,
                "max_similarity": max_sim,
                "avg_similarity": avg_sim,
                "min_similarity": min_sim,
                "similarities": similarities
            }
            
        except Exception as e:
            logger.error(f"Error computing similarity statistics: {e}")
            return {
                "total_cached_queries": 0,
                "valid_embeddings": 0,
                "max_similarity": 0.0,
                "avg_similarity": 0.0,
                "min_similarity": 0.0,
                "error": str(e)
            }