"""
Vector Store Manager using ChromaDB
Fixed version with robust initialization and error handling
"""

import logging
import json
import os
import shutil
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings

from .config import config
from .models import QueryResult, CacheStats

# Set up logging
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages query results using ChromaDB vector database."""
    
    def __init__(self):
        """Initialize the vector store manager with robust error handling."""
        self.db_path = Path(config.CACHE_DIR) / "chroma_db"
        self.collection_name = "query_results"
        self.collection = None
        self.client = None
        
        # Statistics tracking
        self.stats_file = Path(config.CACHE_DIR) / "vector_stats.json"
        
        # Initialize ChromaDB
        self._initialize_chromadb()
        self._ensure_stats_file()
        
        logger.info("Vector store manager initialized successfully")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB with robust error handling."""
        try:
            # Create the database directory if it doesn't exist
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client with settings to disable telemetry
            self.client = chromadb.PersistentClient(
                path=str(self.db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                    is_persistent=True
                )
            )
            
            # Try to get existing collection first
            try:
                existing_collections = self.client.list_collections()
                collection_exists = any(col.name == self.collection_name for col in existing_collections)
                
                if collection_exists:
                    self.collection = self.client.get_collection(name=self.collection_name)
                    logger.info(f"Loaded existing collection: {self.collection_name}")
                else:
                    # Collection doesn't exist, create it
                    self.collection = self.client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={
                            "description": "Web query results with embeddings",
                            "hnsw:space": "cosine"          # << key line
                        }
                    )
                    logger.info(f"Created new collection: {self.collection_name}")
                    
            except Exception as e:
                logger.warning(f"Error accessing collection: {e}")
                # Try to create the collection
                try:
                    self.collection = self.client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={
                            "description": "Web query results with embeddings",
                            "hnsw:space": "cosine"          # << key line
                        }
                    )
                    logger.info(f"Created new collection after error: {self.collection_name}")
                except Exception as create_error:
                    logger.error(f"Failed to create collection: {create_error}")
                    # Try to delete and recreate
                    try:
                        self.client.delete_collection(name=self.collection_name)
                        logger.info(f"Deleted existing problematic collection: {self.collection_name}")
                    except:
                        pass
                    
                    self.collection = self.client.create_collection(
                        name=self.collection_name,
                        metadata={
                            "description": "Web query results with embeddings",
                            "hnsw:space": "cosine"          # << key line
                        }
                    )
                    logger.info(f"Recreated collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            # Fallback: try to reset the database
            try:
                if self.db_path.exists():
                    shutil.rmtree(self.db_path)
                    logger.info("Removed corrupted database directory")
                
                self.db_path.mkdir(parents=True, exist_ok=True)
                
                self.client = chromadb.PersistentClient(
                    path=str(self.db_path),
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )
                
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={
                        "description": "Web query results with embeddings",
                        "hnsw:space": "cosine"          # << key line
                    }
                )
                logger.info(f"Created fresh collection after reset: {self.collection_name}")
                
            except Exception as fallback_error:
                logger.error(f"Complete initialization failed: {fallback_error}")
                raise RuntimeError(f"Unable to initialize ChromaDB: {fallback_error}")
    
    def _ensure_stats_file(self):
        """Ensure stats file exists with default values."""
        if not self.stats_file.exists():
            default_stats = {
                "total_queries": 0,
                "valid_queries": 0,
                "invalid_queries": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_pages_scraped": 0,
                "total_execution_time": 0.0,
                "last_updated": datetime.now().isoformat()
            }
            self._save_stats(default_stats)
    
    def _load_stats(self) -> Dict[str, Any]:
        """Load statistics from file."""
        try:
            with open(self.stats_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading stats: {e}")
            return {
                "total_queries": 0,
                "valid_queries": 0,
                "invalid_queries": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_pages_scraped": 0,
                "total_execution_time": 0.0,
                "last_updated": datetime.now().isoformat()
            }
    
    def _save_stats(self, stats: Dict[str, Any]):
        """Save statistics to file."""
        try:
            stats["last_updated"] = datetime.now().isoformat()
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving stats: {e}")
    
    def _update_stats(self, **kwargs):
        """Update specific statistics."""
        stats = self._load_stats()
        for key, value in kwargs.items():
            if key in stats:
                if isinstance(stats[key], (int, float)) and isinstance(value, (int, float)):
                    stats[key] += value
                else:
                    stats[key] = value
        self._save_stats(stats)
    
    def save_query_result(self, query_result: QueryResult, embedding: List[float]) -> bool:
        """
        Save a query result with its embedding to the vector database.
        
        Args:
            query_result: The query result object
            embedding: The query embedding vector
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return False
            
            # Prepare metadata
            metadata = {
                "query": query_result.query,
                "query_hash": query_result.query_hash,
                # "status": query_result.classification.status.value,
                # "confidence": query_result.classification.confidence,
                "created_at": query_result.created_at.isoformat(),
                "execution_time": query_result.execution_time or 0.0,
                "pages_scraped": len([c for c in query_result.scraped_content if c.success]),
                "total_content_length": sum(c.content_length for c in query_result.scraped_content if c.success)
            }
            
            # Prepare document (summary for search)
            document = f"{query_result.query}"
            if query_result.summary and query_result.summary.summary:
                document += f" | {query_result.summary.summary}"
            
            # Convert query_result to JSON for storage
            result_json = query_result.dict()
            # Convert datetime objects to strings for JSON serialization
            if 'created_at' in result_json:
                result_json['created_at'] = query_result.created_at.isoformat()
            
            # Add to collection
            self.collection.add(
                ids=[query_result.query_hash],
                embeddings=[embedding],
                documents=[document],
                metadatas=[{
                    **metadata,
                    "full_result": json.dumps(result_json, default=str)
                }]
            )
            
            # Update statistics
            self._update_stats(
                total_queries=1,
                valid_queries=1 if query_result.classification.status.value == 'valid' else 0,
                invalid_queries=1 if query_result.classification.status.value == 'invalid' else 0,
                cache_misses=1,
                total_pages_scraped=len([c for c in query_result.scraped_content if c.success]),
                total_execution_time=query_result.execution_time or 0.0
            )
            
            logger.info(f"Saved query result for: '{query_result.query}' with hash: {query_result.query_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving query result: {e}")
            return False
    
    def get_cached_result(self, query_hash: str) -> Optional[QueryResult]:
        """
        Get a cached result by query hash.
        
        Args:
            query_hash: The query hash to search for
            
        Returns:
            QueryResult if found, None otherwise
        """
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return None
            
            # Search by exact query hash
            results = self.collection.get(
                ids=[query_hash],
                include=["metadatas"]
            )
            
            if results['ids']:
                metadata = results['metadatas'][0]
                
                # Reconstruct QueryResult from stored JSON
                result_json = json.loads(metadata['full_result'])
                
                # Convert ISO string back to datetime
                if 'created_at' in result_json:
                    result_json['created_at'] = datetime.fromisoformat(result_json['created_at'])
                
                query_result = QueryResult(**result_json)
                query_result.cache_hit = True
                
                # Update cache hit stats
                self._update_stats(cache_hits=1)
                
                logger.info(f"Found cached result for hash: {query_hash}")
                return query_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None
        
    def _distance_to_cosine(self, distance: float) -> float:
        """
        Convert ChromaDB HNSW distance (cosine) to similarity score in [0,1].

        For unit-length vectors:  cosine_sim = 1 - (distance ** 2) / 2  [22]
        """
        return max(0.0, 1.0 - (distance * distance) / 2.0)

    def find_similar_queries(
        self,
        embedding: List[float],
        threshold: float = 0.85,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Use ChromaDB’s native cosine search to fetch similar queries.
        """
        if not self.collection:
            logger.error("Collection not initialised")
            return []

        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit * 2,
            include=["metadatas", "distances", "documents"]
        )

        matches = []
        for doc_id, dist, meta, _ in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
            results["documents"][0]
        ):
            score = self._distance_to_cosine(dist)
            if score >= threshold:
                qr_json = json.loads(meta["full_result"])
                qr_json["created_at"] = datetime.fromisoformat(qr_json["created_at"])
                cached = QueryResult(**qr_json)
                cached.cache_hit = True

                matches.append({
                    "original_query": meta["query"],
                    "similarity_score": score,
                    "cached_result": cached,
                    "distance": dist
                })

        matches.sort(key=lambda x: x["similarity_score"], reverse=True)
        return matches[:limit]

    
    def get_all_queries(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get all stored queries with metadata.
        
        Args:
            limit: Optional limit on number of results
            
        Returns:
            List of query metadata
        """
        try:
            if not self.collection:
                logger.error("Collection not initialized")
                return []
            
            # Get all items from collection
            results = self.collection.get(
                include=["metadatas", "documents"],
                limit=limit
            )
            
            queries = []
            if results['ids']:
                for doc_id, metadata, document in zip(
                    results['ids'],
                    results['metadatas'],
                    results['documents']
                ):
                    queries.append({
                        'id': doc_id,
                        'query': metadata['query'],
                        'status': metadata['status'],
                        'confidence': metadata['confidence'],
                        'created_at': metadata['created_at'],
                        'execution_time': metadata['execution_time'],
                        'pages_scraped': metadata['pages_scraped'],
                        'document': document
                    })
            
            # Sort by creation time (newest first)
            queries.sort(key=lambda x: x['created_at'], reverse=True)
            
            return queries
            
        except Exception as e:
            logger.error(f"Error getting all queries: {e}")
            return []
    
    def get_cache_stats(self) -> CacheStats:
        """
        Get cache statistics.
        
        Returns:
            CacheStats object with current statistics
        """
        try:
            stats = self._load_stats()
            
            # Get collection count
            collection_count = 0
            if self.collection:
                try:
                    collection_count = self.collection.count()
                except:
                    collection_count = 0
            
            # Calculate average execution time
            avg_execution_time = 0.0
            if stats["total_queries"] > 0:
                avg_execution_time = stats["total_execution_time"] / stats["total_queries"]
            
            return CacheStats(
                total_queries=stats["total_queries"],
                valid_queries=stats["valid_queries"],
                invalid_queries=stats["invalid_queries"],
                cache_hits=stats["cache_hits"],
                cache_misses=stats["cache_misses"],
                total_pages_scraped=stats["total_pages_scraped"],
                average_execution_time=avg_execution_time,
                last_updated=datetime.fromisoformat(stats["last_updated"]),
                total_entries=collection_count
            )
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return CacheStats()
    
    def get_cache_size(self) -> Dict[str, Any]:
        """
        Get cache size information.
        
        Returns:
            Dictionary with cache size details
        """
        try:
            # Get collection count
            total_entries = 0
            if self.collection:
                try:
                    total_entries = self.collection.count()
                except:
                    total_entries = 0
            
            # Calculate database size
            db_size_bytes = 0
            if self.db_path.exists():
                for file_path in self.db_path.rglob('*'):
                    if file_path.is_file():
                        try:
                            db_size_bytes += file_path.stat().st_size
                        except:
                            pass
            
            return {
                "total_entries": total_entries,
                "total_cache_size_bytes": db_size_bytes,
                "database_path": str(self.db_path),
                "collection_name": self.collection_name
            }
            
        except Exception as e:
            logger.error(f"Error getting cache size: {e}")
            return {
                "total_entries": 0,
                "total_cache_size_bytes": 0,
                "database_path": str(self.db_path),
                "collection_name": self.collection_name
            }
    
    def clear_cache(self) -> bool:
        """
        Clear all cached data.
        
        Returns:
            bool: True if cleared successfully, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Delete the collection
            try:
                self.client.delete_collection(name=self.collection_name)
            except:
                pass  # Collection might not exist
            
            # Recreate the collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Web query results with embeddings"}
            )
            
            # Reset statistics
            self._save_stats({
                "total_queries": 0,
                "valid_queries": 0,
                "invalid_queries": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "total_pages_scraped": 0,
                "total_execution_time": 0.0,
                "last_updated": datetime.now().isoformat()
            })
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def delete_query(self, query_hash: str) -> bool:
        """
        Delete a specific query from the cache.
        
        Args:
            query_hash: The query hash to delete
            
        Returns:
            bool: True if deleted successfully, False otherwise
        """
        try:
            if not self.collection:
                return False
            
            self.collection.delete(ids=[query_hash])
            logger.info(f"Deleted query with hash: {query_hash}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting query {query_hash}: {e}")
            return False
    
    def search_queries(self, search_term: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search queries by text content.
        
        Args:
            search_term: Term to search for
            limit: Maximum number of results
            
        Returns:
            List of matching queries
        """
        try:
            # Get all queries and filter by search term
            all_queries = self.get_all_queries()
            
            matching_queries = []
            search_lower = search_term.lower()
            
            for query_data in all_queries:
                query_text = query_data['query'].lower()
                document_text = query_data['document'].lower()
                
                if (search_lower in query_text or 
                    search_lower in document_text):
                    matching_queries.append(query_data)
                
                if len(matching_queries) >= limit:
                    break
            
            return matching_queries
            
        except Exception as e:
            logger.error(f"Error searching queries: {e}")
            return []