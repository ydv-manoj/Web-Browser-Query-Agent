"""
Cache Manager module.
Handles storage and retrieval of query results using JSON files.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .config import config
from .models import QueryResult, CacheEntry, AppStats

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class CacheManager:
    """Manages caching of query results and embeddings."""
    
    def __init__(self):
        """Initialize the cache manager."""
        self.cache_dir = config.CACHE_DIR
        self.queries_file = config.QUERIES_CACHE_FILE
        self.embeddings_file = config.EMBEDDINGS_CACHE_FILE
        self.stats_file = self.cache_dir / "stats.json"
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize cache files if they don't exist
        self._initialize_cache_files()
        
        logger.info("Cache manager initialized")
    
    def _initialize_cache_files(self):
        """Initialize cache files with empty data if they don't exist."""
        try:
            # Initialize queries cache
            if not self.queries_file.exists():
                self._write_json_file(self.queries_file, [])
                logger.info("Created new queries cache file")
            
            # Initialize embeddings cache
            if not self.embeddings_file.exists():
                self._write_json_file(self.embeddings_file, {})
                logger.info("Created new embeddings cache file")
            
            # Initialize stats file
            if not self.stats_file.exists():
                initial_stats = AppStats()
                self._write_json_file(self.stats_file, initial_stats.dict())
                logger.info("Created new stats file")
                
        except Exception as e:
            logger.error(f"Error initializing cache files: {e}")
            raise
    
    def save_query_result(self, query_result: QueryResult, embedding: List[float]) -> bool:
        """
        Save a query result and its embedding to cache.
        
        Args:
            query_result: The QueryResult object to cache
            embedding: The query embedding vector
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Create cache entry
            cache_entry = CacheEntry(
                query_hash=query_result.query_hash,
                query=query_result.query,
                result=query_result,
                embedding=embedding,
                created_at=datetime.now()
            )
            
            # Load existing cache
            cached_entries = self.load_cached_entries()
            
            # Check if entry already exists (update if so)
            existing_index = None
            for i, entry in enumerate(cached_entries):
                if entry.query_hash == query_result.query_hash:
                    existing_index = i
                    break
            
            if existing_index is not None:
                # Update existing entry
                cached_entries[existing_index] = cache_entry
                logger.info(f"Updated existing cache entry for query: {query_result.query}")
            else:
                # Add new entry
                cached_entries.append(cache_entry)
                logger.info(f"Added new cache entry for query: {query_result.query}")
            
            # Save updated cache
            cache_data = [entry.dict() for entry in cached_entries]
            self._write_json_file(self.queries_file, cache_data)
            
            # Update statistics
            self._update_stats(cache_miss=True)
            
            # Clean up old entries
            self._cleanup_old_entries()
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving query result to cache: {e}")
            return False
    
    def load_cached_entries(self) -> List[CacheEntry]:
        """
        Load all cached entries from the cache file.
        
        Returns:
            List of CacheEntry objects
        """
        try:
            cache_data = self._read_json_file(self.queries_file)
            
            if not isinstance(cache_data, list):
                logger.warning("Invalid cache data format, returning empty list")
                return []
            
            # Convert dictionaries to CacheEntry objects
            cached_entries = []
            for entry_data in cache_data:
                try:
                    # Handle datetime parsing
                    if isinstance(entry_data.get('created_at'), str):
                        entry_data['created_at'] = datetime.fromisoformat(entry_data['created_at'])
                    if isinstance(entry_data.get('last_accessed'), str):
                        entry_data['last_accessed'] = datetime.fromisoformat(entry_data['last_accessed'])
                    
                    # Handle nested result parsing
                    if 'result' in entry_data and isinstance(entry_data['result'], dict):
                        result_data = entry_data['result']
                        # Convert datetime fields in result
                        if isinstance(result_data.get('created_at'), str):
                            result_data['created_at'] = datetime.fromisoformat(result_data['created_at'])
                        
                        # Convert scraped_content items
                        if 'scraped_content' in result_data:
                            for content in result_data['scraped_content']:
                                if isinstance(content.get('scraped_at'), str):
                                    content['scraped_at'] = datetime.fromisoformat(content['scraped_at'])
                        
                        entry_data['result'] = QueryResult(**result_data)
                    
                    cache_entry = CacheEntry(**entry_data)
                    cached_entries.append(cache_entry)
                    
                except Exception as e:
                    logger.warning(f"Error parsing cache entry: {e}")
                    continue
            
            logger.debug(f"Loaded {len(cached_entries)} cached entries")
            return cached_entries
            
        except Exception as e:
            logger.error(f"Error loading cached entries: {e}")
            return []
    
    def get_cached_result(self, query_hash: str) -> Optional[QueryResult]:
        """
        Get a cached result by query hash.
        
        Args:
            query_hash: The hash of the query to retrieve
            
        Returns:
            QueryResult if found, None otherwise
        """
        try:
            cached_entries = self.load_cached_entries()
            
            for entry in cached_entries:
                if entry.query_hash == query_hash:
                    # Update access information
                    entry.update_access()
                    self._update_access_info(entry)
                    
                    # Update statistics
                    self._update_stats(cache_hit=True)
                    
                    logger.info(f"Cache hit for query hash: {query_hash}")
                    return entry.result
            
            logger.debug(f"Cache miss for query hash: {query_hash}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting cached result: {e}")
            return None
    
    def _update_access_info(self, cache_entry: CacheEntry):
        """Update access information for a cache entry."""
        try:
            cached_entries = self.load_cached_entries()
            
            for i, entry in enumerate(cached_entries):
                if entry.query_hash == cache_entry.query_hash:
                    cached_entries[i] = cache_entry
                    break
            
            # Save updated cache
            cache_data = [entry.dict() for entry in cached_entries]
            self._write_json_file(self.queries_file, cache_data)
            
        except Exception as e:
            logger.error(f"Error updating access info: {e}")
    
    def _cleanup_old_entries(self):
        """Remove old cache entries based on expiry settings."""
        try:
            cached_entries = self.load_cached_entries()
            current_time = datetime.now()
            expiry_threshold = current_time - timedelta(days=config.CACHE_EXPIRY_DAYS)
            
            # Filter out expired entries
            valid_entries = [
                entry for entry in cached_entries 
                if entry.created_at > expiry_threshold
            ]
            
            if len(valid_entries) < len(cached_entries):
                removed_count = len(cached_entries) - len(valid_entries)
                logger.info(f"Removed {removed_count} expired cache entries")
                
                # Save cleaned cache
                cache_data = [entry.dict() for entry in valid_entries]
                self._write_json_file(self.queries_file, cache_data)
                
        except Exception as e:
            logger.error(f"Error cleaning up old entries: {e}")
    
    def get_cache_stats(self) -> AppStats:
        """
        Get cache and application statistics.
        
        Returns:
            AppStats object with current statistics
        """
        try:
            stats_data = self._read_json_file(self.stats_file)
            
            if isinstance(stats_data, dict):
                # Handle datetime parsing
                if 'last_updated' in stats_data and isinstance(stats_data['last_updated'], str):
                    stats_data['last_updated'] = datetime.fromisoformat(stats_data['last_updated'])
                
                return AppStats(**stats_data)
            else:
                logger.warning("Invalid stats data format, returning default stats")
                return AppStats()
                
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return AppStats()
    
    def _update_stats(self, cache_hit: bool = False, cache_miss: bool = False, 
                      query_valid: bool = False, query_invalid: bool = False,
                      execution_time: float = 0.0, pages_scraped: int = 0):
        """Update application statistics."""
        try:
            stats = self.get_cache_stats()
            
            if cache_hit:
                stats.cache_hits += 1
            if cache_miss:
                stats.cache_misses += 1
            if query_valid:
                stats.valid_queries += 1
                stats.total_queries += 1
            if query_invalid:
                stats.invalid_queries += 1
                stats.total_queries += 1
            
            if execution_time > 0:
                # Update average execution time
                total_queries = stats.total_queries or 1
                current_total = stats.average_execution_time * (total_queries - 1)
                stats.average_execution_time = (current_total + execution_time) / total_queries
            
            if pages_scraped > 0:
                stats.total_pages_scraped += pages_scraped
            
            stats.last_updated = datetime.now()
            
            # Save updated stats
            self._write_json_file(self.stats_file, stats.dict())
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    def clear_cache(self) -> bool:
        """
        Clear all cached data.
        
        Returns:
            True if cleared successfully, False otherwise
        """
        try:
            # Clear cache files
            self._write_json_file(self.queries_file, [])
            self._write_json_file(self.embeddings_file, {})
            
            # Reset stats
            initial_stats = AppStats()
            self._write_json_file(self.stats_file, initial_stats.dict())
            
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_size(self) -> dict:
        """
        Get information about cache size and storage.
        
        Returns:
            Dictionary with cache size information
        """
        try:
            queries_size = self.queries_file.stat().st_size if self.queries_file.exists() else 0
            embeddings_size = self.embeddings_file.stat().st_size if self.embeddings_file.exists() else 0
            stats_size = self.stats_file.stat().st_size if self.stats_file.exists() else 0
            
            cached_entries = self.load_cached_entries()
            
            return {
                "total_entries": len(cached_entries),
                "queries_file_size_bytes": queries_size,
                "embeddings_file_size_bytes": embeddings_size,
                "stats_file_size_bytes": stats_size,
                "total_cache_size_bytes": queries_size + embeddings_size + stats_size,
                "cache_directory": str(self.cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache size: {e}")
            return {
                "total_entries": 0,
                "queries_file_size_bytes": 0,
                "embeddings_file_size_bytes": 0,
                "stats_file_size_bytes": 0,
                "total_cache_size_bytes": 0,
                "cache_directory": str(self.cache_dir),
                "error": str(e)
            }
    
    def _read_json_file(self, file_path: Path) -> any:
        """Read and parse a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON file {file_path}: {e}")
            return None
    
    def _write_json_file(self, file_path: Path, data: any):
        """Write data to a JSON file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            logger.error(f"Error writing JSON file {file_path}: {e}")
            raise