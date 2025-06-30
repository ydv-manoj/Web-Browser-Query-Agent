"""
Pydantic models for data validation and serialization.
Defines the structure of data used throughout the application.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class QueryStatus(str, Enum):
    """Enum for query validation status."""
    VALID = "valid"
    INVALID = "invalid"
    PENDING = "pending"

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
    
    @validator('content_length', pre=True, always=True)
    def set_content_length(cls, v, values):
        """Automatically calculate content length."""
        if 'content' in values:
            return len(values['content'])
        return 0

class SummaryResult(BaseModel):
    """Model for summarized content."""
    summary: str
    key_points: List[str]
    sources: List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    word_count: int
    
    @validator('word_count', pre=True, always=True)
    def set_word_count(cls, v, values):
        """Automatically calculate word count."""
        if 'summary' in values:
            return len(values['summary'].split())
        return 0

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

class SimilarQuery(BaseModel):
    """Model for similar query match."""
    original_query: str
    similarity_score: float
    cached_result: QueryResult

class AgentResponse(BaseModel):
    """Model for final agent response to user."""
    success: bool
    message: str
    query: str
    result: Optional[QueryResult] = None
    similar_query_used: Optional[SimilarQuery] = None
    error: Optional[str] = None
    execution_time: float
    
class WebSearchResult(BaseModel):
    """Model for web search result before scraping."""
    title: str
    url: str
    snippet: Optional[str] = None
    position: int

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