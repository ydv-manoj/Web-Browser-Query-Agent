import logging
from typing import List, Optional, Dict, Any
from .config import config
from .models import SimilarQuery

logger = logging.getLogger(__name__)


class SimilarityChecker:
    """
    Provides fast nearest-neighbour look-ups using a ChromaDB cosine index.
    Retains the public interface expected by API callers.
    """

    def __init__(self) -> None:
        self.similarity_threshold: float = config.SIMILARITY_THRESHOLD
        logger.info(
            "SimilarityChecker initialised (threshold=%.3f)",
            self.similarity_threshold
        )

    # ---------- single result -------------------------------------------------
    def find_similar_query(
        self,
        query_embedding: List[float],
        vector_store
    ) -> Optional[SimilarQuery]:
        try:
            matches = vector_store.find_similar_queries(
                embedding=query_embedding,
                threshold=self.similarity_threshold,
                limit=1
            )
            if not matches:
                logger.info("No similar queries above threshold %.3f",
                            self.similarity_threshold)
                return None

            hit = matches[0]
            logger.info("Nearest neighbour: '%s' (%.3f)",
                        hit["original_query"], hit["similarity_score"])

            return SimilarQuery(
                original_query=hit["original_query"],
                similarity_score=hit["similarity_score"],
                cached_result=hit["cached_result"]
            )
        except Exception as exc:
            logger.error("Similarity lookup failed: %s", exc)
            return None

    # ---------- top-k results --------------------------------------------------
    def find_multiple_similar_queries(
        self,
        query_embedding: List[float],
        vector_store,
        limit: int = 5
    ) -> List[SimilarQuery]:
        try:
            matches = vector_store.find_similar_queries(
                embedding=query_embedding,
                threshold=self.similarity_threshold,
                limit=limit
            )
            return [
                SimilarQuery(
                    original_query=m["original_query"],
                    similarity_score=m["similarity_score"],
                    cached_result=m["cached_result"]
                )
                for m in matches
            ]
        except Exception as exc:
            logger.error("Similarity batch lookup failed: %s", exc)
            return []

    # ---------- metrics -------------------------------------------------------
    def get_similarity_stats(self, vector_store) -> Dict[str, Any]:
        try:
            stats = vector_store.get_cache_stats()
            total = stats.cache_hits + stats.cache_misses
            hit_rate = (stats.cache_hits / total) * 100 if total else 0.0
            return {
                "similarity_threshold": self.similarity_threshold,
                "total_queries_processed": total,
                "cache_hits": stats.cache_hits,
                "cache_misses": stats.cache_misses,
                "similarity_hit_rate_percent": hit_rate,
                "total_stored_queries": stats.total_entries
            }
        except Exception as exc:
            logger.error("Stats generation failed: %s", exc)
            return {
                "similarity_threshold": self.similarity_threshold,
                "total_queries_processed": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "similarity_hit_rate_percent": 0.0,
                "total_stored_queries": 0
            }
