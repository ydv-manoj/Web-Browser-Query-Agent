"""
Pydantic models for data validation and serialization.
Defines the structure of data used throughout the application.
Updated to include vector database and API support while preserving existing models.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class QueryStatus(str, Enum):
    """Enum for query validation status."""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"
    PROCESSING = "processing"  # Added for API support
    COMPLETED = "completed"    # Added for API support
    FAILED = "failed"          # Added for API support

class ScrapingStatus(str, Enum):
    """Enumeration for scraping status - NEW for enhanced scraping."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    ERROR = "error"

class QueryClassification(BaseModel):
    """Model for query classification result."""
    query: str
    status: QueryStatus
    confidence: float = Field(ge=0.0, le=1.0)
    reason: Optional[str] = None
    suggested_improvements: Optional[List[str]] = None

class ScrapedContent(BaseModel):
    """Model for content scraped from a single webpage."""
    url: str
    title: Optional[str] = None
    content: str
    content_length: int
    scraped_at: datetime = Field(default_factory=datetime.now)
    success: bool = True
    error_message: Optional[str] = None
    status: ScrapingStatus = Field(default=ScrapingStatus.SUCCESS)  # Added for enhanced status tracking
    
    @validator('content_length', pre=True, always=True)
    def set_content_length(cls, v, values):
        """Automatically calculate content length."""
        if 'content' in values:
            return len(values['content'])
        return 0
    
    class Config:
        use_enum_values = True

class SummaryResult(BaseModel):
    """Model for summarized content."""
    summary: str
    key_points: List[str]
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    word_count: int
    language: str = Field(default="en")  # Added for internationalization
    generated_at: datetime = Field(default_factory=datetime.now)  # Added for tracking
    
    @validator('word_count', pre=True, always=True)
    def set_word_count(cls, v, values):
        """Automatically calculate word count."""
        if 'summary' in values:
            return len(values['summary'].split())
        return 0

# Alias for backward compatibility and API consistency
ContentSummary = SummaryResult

class QueryResult(BaseModel):
    """Model for complete query result."""
    query: str
    query_hash: str
    classification: QueryClassification
    scraped_content: List[ScrapedContent]
    summary: Optional[SummaryResult] = None
    similar_queries: List[str] = []
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    execution_time: Optional[float] = None
    cache_hit: bool = False
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        arbitrary_types_allowed = True
        validate_assignment = True
    
    @property
    def successful_scrapes(self) -> List[ScrapedContent]:
        """Get only successful scrapes."""
        return [content for content in self.scraped_content if content.success]
    
    @property
    def total_content_length(self) -> int:
        """Get total length of all scraped content."""
        return sum(content.content_length for content in self.successful_scrapes)
    
    @property
    def success_rate(self) -> float:
        """Get scraping success rate."""
        if not self.scraped_content:
            return 0.0
        return len(self.successful_scrapes) / len(self.scraped_content)

class SimilarQuery(BaseModel):
    """Model for similar query results with semantic validation."""
    original_query: str
    similarity_score: float
    cached_result: "QueryResult"  # Forward reference
    semantic_similarity: Optional[float] = None  # New field for LLM confidence
    semantic_reason: Optional[str] = None  # New field for LLM reasoning
    
    class Config:
        arbitrary_types_allowed = True

class AgentResponse(BaseModel):
    """Model for final agent response to user."""
    success: bool
    message: str
    query: str
    result: Optional[QueryResult] = None
    similar_query_used: Optional[SimilarQuery] = None
    error: Optional[str] = None
    execution_time: float
    timestamp: datetime = Field(default_factory=datetime.now)  # Added for API responses
    
    class Config:
        arbitrary_types_allowed = True

class WebSearchResult(BaseModel):
    """Model for web search result before scraping."""
    title: str
    url: str
    snippet: Optional[str] = None
    position: int
    source_engine: Optional[str] = Field(default=None, description="Search engine used")  # Added for tracking
    
    class Config:
        validate_assignment = True

class CacheEntry(BaseModel):
    """Model for cache entry storage."""
    query_hash: str
    query: str
    result: QueryResult
    embedding: List[float]
    created_at: datetime
    last_accessed: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    
    def update_access(self):
        """Update access information."""
        self.last_accessed = datetime.now()
        self.access_count += 1

class AppStats(BaseModel):
    """Model for application statistics."""
    total_queries: int = 0
    valid_queries: int = 0
    invalid_queries: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    average_execution_time: float = 0.0
    total_pages_scraped: int = 0
    last_updated: datetime = Field(default_factory=datetime.now)

# Enhanced statistics model for vector database
class CacheStats(BaseModel):
    """Enhanced cache and application statistics for vector database."""
    total_queries: int = Field(ge=0, default=0, description="Total queries processed")
    valid_queries: int = Field(ge=0, default=0, description="Number of valid queries")
    invalid_queries: int = Field(ge=0, default=0, description="Number of invalid queries")
    cache_hits: int = Field(ge=0, default=0, description="Number of cache hits")
    cache_misses: int = Field(ge=0, default=0, description="Number of cache misses")
    total_pages_scraped: int = Field(ge=0, default=0, description="Total pages scraped")
    average_execution_time: float = Field(ge=0.0, default=0.0, description="Average execution time in seconds")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last update timestamp")
    total_entries: int = Field(ge=0, default=0, description="Total entries in vector database")
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate query success rate."""
        return (self.valid_queries / self.total_queries * 100) if self.total_queries > 0 else 0.0

# ============================================================================
# API REQUEST/RESPONSE MODELS - NEW for FastAPI support
# ============================================================================

class SearchRequest(BaseModel):
    """Request model for search API."""
    query: str = Field(min_length=1, max_length=500, description="Search query")
    use_cache: bool = Field(default=True, description="Whether to use cached results")
    max_results: Optional[int] = Field(default=None, ge=1, le=20, description="Maximum number of results")
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class SearchResponse(BaseModel):
    """Response model for search API."""
    success: bool = Field(description="Whether the search was successful")
    message: str = Field(description="Response message")
    query: str = Field(description="The processed query")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Search results")
    similar_query_used: Optional[Dict[str, Any]] = Field(default=None, description="Similar query information")
    execution_time: float = Field(ge=0.0, description="Execution time in seconds")
    error: Optional[str] = Field(default=None, description="Error message if failed")
    cache_hit: bool = Field(default=False, description="Whether result came from cache")

class ValidateRequest(BaseModel):
    """Request model for query validation API."""
    query: str = Field(min_length=1, max_length=500, description="Query to validate")
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

class ValidateResponse(BaseModel):
    """Response model for query validation API."""
    query: str = Field(description="The validated query")
    status: str = Field(description="Validation status")
    confidence: float = Field(ge=0.0, le=1.0, description="Validation confidence")
    reason: str = Field(description="Validation reason")
    suggested_improvements: List[str] = Field(default_factory=list, description="Improvement suggestions")

class StatsResponse(BaseModel):
    """Response model for statistics API."""
    total_queries: int = Field(ge=0, description="Total queries processed")
    valid_queries: int = Field(ge=0, description="Valid queries")
    invalid_queries: int = Field(ge=0, description="Invalid queries")
    cache_hits: int = Field(ge=0, description="Cache hits")
    cache_misses: int = Field(ge=0, description="Cache misses")
    cache_hit_rate: float = Field(ge=0.0, le=100.0, description="Cache hit rate percentage")
    average_execution_time: float = Field(ge=0.0, description="Average execution time")
    total_pages_scraped: int = Field(ge=0, description="Total pages scraped")
    total_entries: int = Field(ge=0, description="Total vector database entries")
    cache_size_mb: float = Field(ge=0.0, description="Cache size in MB")
    last_updated: str = Field(description="Last updated timestamp")

class QueryHistoryItem(BaseModel):
    """Individual query history item."""
    id: str = Field(description="Query ID/hash")
    query: str = Field(description="The query text")
    status: str = Field(description="Query status")
    confidence: float = Field(ge=0.0, le=1.0, description="Classification confidence")
    created_at: str = Field(description="Creation timestamp")
    execution_time: float = Field(ge=0.0, description="Execution time")
    pages_scraped: int = Field(ge=0, description="Number of pages scraped")
    total_content_length: int = Field(ge=0, description="Total content length")

class QueryHistoryResponse(BaseModel):
    """Response model for query history API."""
    queries: List[QueryHistoryItem] = Field(default_factory=list, description="List of query history items")
    total_count: int = Field(ge=0, description="Total number of queries")
    limit: int = Field(ge=0, description="Applied limit")
    search_term: Optional[str] = Field(default=None, description="Search term used for filtering")

class HealthResponse(BaseModel):
    """Response model for health check API."""
    status: str = Field(description="Health status")
    timestamp: str = Field(description="Check timestamp")
    components: Dict[str, str] = Field(description="Component health status")
    version: Optional[str] = Field(default=None, description="Application version")

class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str = Field(description="Error detail message")
    error_type: Optional[str] = Field(default=None, description="Type of error")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat(), description="Error timestamp")

# ============================================================================
# VECTOR DATABASE MODELS - NEW for ChromaDB support
# ============================================================================

class VectorSearchResult(BaseModel):
    """Result from vector similarity search."""
    id: str = Field(description="Document ID")
    score: float = Field(ge=0.0, le=1.0, description="Similarity score")
    metadata: Dict[str, Any] = Field(description="Document metadata")
    content: Optional[str] = Field(default=None, description="Document content")

class VectorStoreStats(BaseModel):
    """Vector store statistics."""
    total_documents: int = Field(ge=0, description="Total documents in store")
    collection_name: str = Field(description="Collection name")
    database_size_bytes: int = Field(ge=0, description="Database size in bytes")
    last_updated: datetime = Field(description="Last update timestamp")

# ============================================================================
# FORWARD REFERENCES AND MODEL REBUILDING
# ============================================================================

# Forward references for circular dependencies
SimilarQuery.model_rebuild()
QueryResult.model_rebuild()

# ============================================================================
# EXPORTS
# ============================================================================

# Export all models for easy importing
__all__ = [
    # Original models (preserved)
    'QueryStatus', 'QueryClassification', 'ScrapedContent', 'SummaryResult',
    'QueryResult', 'SimilarQuery', 'AgentResponse', 'WebSearchResult',
    'CacheEntry', 'AppStats',
    
    # New enums
    'ScrapingStatus',
    
    # Enhanced models
    'CacheStats', 'ContentSummary',
    
    # API models
    'SearchRequest', 'SearchResponse', 'ValidateRequest', 'ValidateResponse',
    'StatsResponse', 'QueryHistoryItem', 'QueryHistoryResponse',
    'HealthResponse', 'ErrorResponse',
    
    # Vector database models
    'VectorSearchResult', 'VectorStoreStats'
]