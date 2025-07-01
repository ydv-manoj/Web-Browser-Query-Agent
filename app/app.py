"""
FastAPI Routes and Endpoints - Fixed version with CORS and proper error handling
"""

import asyncio
import logging
import time
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ValidationError

from .config import config
from .models import QueryResult, AgentResponse, QueryStatus, QueryClassification
from .query_processor import QueryProcessor
from .undetected_web_scraper import EnhancedUndetectedWebScraper
from .summarizer import ContentSummarizer
from .similarity_checker import SimilarityChecker
from .cache_manager import VectorStoreManager

# Set up logging
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Web Query Agent API",
    description="AI-powered web search and summarization tool",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates - with error handling
try:
    # Create directories if they don't exist
    Path("static").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)
    
    app.mount("/static", StaticFiles(directory="static"), name="static")
    templates = Jinja2Templates(directory="templates")
    logger.info("Static files and templates mounted successfully")
except Exception as e:
    logger.error(f"Error mounting static files: {e}")
    templates = None

# Initialize components with error handling
query_processor = None
similarity_checker = None
summarizer = None
vector_store = None

try:
    query_processor = QueryProcessor()
    logger.info("✅ Query processor initialized")
except Exception as e:
    logger.error(f"❌ Query processor failed: {e}")

try:
    similarity_checker = SimilarityChecker()
    logger.info("✅ Similarity checker initialized")
except Exception as e:
    logger.error(f"❌ Similarity checker failed: {e}")

try:
    summarizer = ContentSummarizer()
    logger.info("✅ Content summarizer initialized")
except Exception as e:
    logger.error(f"❌ Content summarizer failed: {e}")

try:
    vector_store = VectorStoreManager()
    logger.info("✅ Vector store initialized")
except Exception as e:
    logger.error(f"❌ Vector store failed: {e}")

# Pydantic models for API
class SearchRequest(BaseModel):
    query: str
    use_cache: bool = True

class SearchResponse(BaseModel):
    success: bool
    message: str
    query: str
    result: Optional[Dict[str, Any]] = None
    similar_query_used: Optional[Dict[str, Any]] = None
    execution_time: float
    error: Optional[str] = None

class ValidateRequest(BaseModel):
    query: str

class ValidateResponse(BaseModel):
    query: str
    status: str
    confidence: float
    reason: str
    suggested_improvements: List[str] = []

class StatsResponse(BaseModel):
    total_queries: int
    valid_queries: int
    invalid_queries: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    average_execution_time: float
    total_pages_scraped: int
    total_entries: int
    cache_size_mb: float
    last_updated: str

class QueryHistoryResponse(BaseModel):
    queries: List[Dict[str, Any]]
    total_count: int

# Web Query Agent class
class WebQueryAgent:
    """Main Web Query Agent class for API usage."""
    
    def __init__(self):
        self.query_processor = query_processor
        self.similarity_checker = similarity_checker
        self.summarizer = summarizer
        self.vector_store = vector_store
    
    async def process_query(self, query: str, use_cache: bool = True) -> Dict[str, Any]:
            """Process a user query through the complete pipeline."""
            start_time = time.time()
            
            try:
                logger.info(f"🔍 Processing API query: '{query}'")
                
                # Check if components are initialized
                if not self.query_processor:
                    raise Exception("Query processor not initialized")
                
                # Step 1: Normalize and validate
                normalized_query = self.query_processor.normalize_query(query)
                query_hash = self.query_processor.generate_query_hash(normalized_query)
                
                # Step 2: Classify the query
                classification = self.query_processor.classify_query(normalized_query)
                
                if classification.status == QueryStatus.INVALID:
                    return {
                        "success": False,
                        "message": f"Invalid query: {classification.reason}",
                        "query": normalized_query,
                        "error": classification.reason,
                        "execution_time": time.time() - start_time
                    }
                
                if use_cache and self.vector_store:
                    # Step 3: Check for cached results
                    cached_result = self.vector_store.get_cached_result(query_hash)
                    if cached_result:
                        return {
                            "success": True,
                            "message": "Found cached result for your query",
                            "query": normalized_query,
                            "result": cached_result.dict(),
                            "execution_time": time.time() - start_time,
                            "cache_hit": True
                        }
                    
                    # Step 4: Check for similar queries with semantic validation
                    query_embedding = self.query_processor.generate_embedding(normalized_query)
                    if self.similarity_checker:
                        # Pass the normalized query text for semantic comparison
                        similar_query = self.similarity_checker.find_similar_query(
                            query_embedding, 
                            self.vector_store,
                            current_query=normalized_query  # Pass the query text
                        )
                        
                        if similar_query:
                            # Include semantic similarity info in response
                            similar_query_info = {
                                "original_query": similar_query.original_query,
                                "embedding_similarity": similar_query.similarity_score,
                                "semantic_similarity": similar_query.semantic_similarity,
                                "semantic_reason": similar_query.semantic_reason
                            }
                            
                            return {
                                "success": True,
                                "message": f"Found semantically similar query: '{similar_query.original_query}'",
                                "query": normalized_query,
                                "result": similar_query.cached_result.dict(),
                                "similar_query_used": similar_query_info,
                                "execution_time": time.time() - start_time,
                                "cache_hit": True
                            }
                else:
                    query_embedding = self.query_processor.generate_embedding(normalized_query)
                
                # Step 5: Perform web scraping
                logger.info("🌐 Starting web scraping...")
                scraped_content = await self._scrape_web_content(normalized_query)
                
                if not scraped_content:
                    return {
                        "success": False,
                        "message": "Failed to scrape any web content for your query",
                        "query": normalized_query,
                        "error": "Web scraping failed",
                        "execution_time": time.time() - start_time
                    }
                
                # Step 6: Summarize content
                summary = None
                if self.summarizer:
                    summary = self.summarizer.summarize_content(normalized_query, scraped_content)
                
                # Step 7: Create query result
                query_result = QueryResult(
                    query=normalized_query,
                    query_hash=query_hash,
                    classification=classification,
                    scraped_content=scraped_content,
                    summary=summary,
                    embedding=query_embedding,
                    created_at=datetime.now(),
                    execution_time=time.time() - start_time
                )
                
                # Step 8: Save to vector store
                if use_cache and self.vector_store:
                    self.vector_store.save_query_result(query_result, query_embedding)
                
                return {
                    "success": True,
                    "message": "Successfully processed your query",
                    "query": normalized_query,
                    "result": query_result.dict(),
                    "execution_time": time.time() - start_time,
                    "cache_hit": False
                }
                
            except Exception as e:
                logger.error(f"❌ Error processing query '{query}': {e}")
                logger.error(f"❌ Traceback: {traceback.format_exc()}")
                return {
                    "success": False,
                    "message": "An error occurred while processing your query",
                    "query": query,
                    "error": str(e),
                    "execution_time": time.time() - start_time
                }
                
    async def _scrape_web_content(self, query: str):
        """Scrape web content using EnhancedUndetectedWebScraper."""
        try:
            async with EnhancedUndetectedWebScraper() as scraper:
                return await scraper.search_and_scrape(query)
        except Exception as e:
            logger.error(f"❌ Scraping failed: {e}")
            return []

# Initialize agent
agent = WebQueryAgent()

# Routes

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main web interface."""
    try:
        if templates:
            return templates.TemplateResponse("index.html", {"request": request})
        else:
            return HTMLResponse("""
                <html>
                    <body>
                        <h1>Web Query Agent API</h1>
                        <p>Templates not found. Please ensure templates/index.html exists.</p>
                        <p>API Documentation: <a href="/docs">/docs</a></p>
                    </body>
                </html>
            """)
    except Exception as e:
        logger.error(f"Error serving home page: {e}")
        return HTMLResponse(f"<h1>Error: {str(e)}</h1>")

@app.post("/api/search")
async def search_query(request: SearchRequest):
    """Search endpoint for processing queries."""
    try:
        logger.info(f"Received search request: {request.query}")
        
        result = await agent.process_query(request.query, request.use_cache)
        
        # Serialize datetime fields in the result (if any)
        def serialize_datetimes(obj):
            if isinstance(obj, dict):
                return {k: serialize_datetimes(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_datetimes(i) for i in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj

        response_data = {
            "success": result["success"],
            "message": result["message"],
            "query": result["query"],
            "result": serialize_datetimes(result.get("result")),
            "similar_query_used": serialize_datetimes(result.get("similar_query_used")),
            # "execution_time": result["execution_time"],
            "error": result.get("error")
        }
        
        return JSONResponse(content=response_data)
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return JSONResponse(
            status_code=422,
            content={"detail": "Invalid request data", "errors": str(e)}
        )
    except Exception as e:
        logger.error(f"Search API error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Internal server error: {str(e)}"}
        )

@app.post("/api/validate")
async def validate_query(request: ValidateRequest):
    """Validate a query without performing search."""
    try:
        if not query_processor:
            raise HTTPException(status_code=503, detail="Query processor not available")
        
        classification = query_processor.classify_query(request.query)
        
        response_data = {
            "query": request.query,
            "status": classification.status.value,
            "confidence": classification.confidence,
            "reason": classification.reason or "",
            "suggested_improvements": classification.suggested_improvements or []
        }
        
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Validate API error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Validation error: {str(e)}"}
        )

@app.get("/api/stats")
async def get_stats():
    """Get application statistics."""
    try:
        if not vector_store:
            # Return default stats if vector store not available
            return JSONResponse(content={
                "total_queries": 0,
                "valid_queries": 0,
                "invalid_queries": 0,
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0,
                "average_execution_time": 0.0,
                "total_pages_scraped": 0,
                "total_entries": 0,
                "cache_size_mb": 0.0,
                "last_updated": datetime.now().isoformat()
            })
        
        stats = vector_store.get_cache_stats()
        cache_size = vector_store.get_cache_size()
        
        cache_hit_rate = 0.0
        total_queries = stats.cache_hits + stats.cache_misses
        if total_queries > 0:
            cache_hit_rate = (stats.cache_hits / total_queries) * 100
        
        response_data = {
            "total_queries": stats.total_queries,
            "valid_queries": stats.valid_queries,
            "invalid_queries": stats.invalid_queries,
            "cache_hits": stats.cache_hits,
            "cache_misses": stats.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "average_execution_time": stats.average_execution_time,
            "total_pages_scraped": stats.total_pages_scraped,
            "total_entries": cache_size["total_entries"],
            "cache_size_mb": cache_size["total_cache_size_bytes"] / (1024 * 1024),
            "last_updated": stats.last_updated.isoformat()
        }
        
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Stats API error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Stats error: {str(e)}"}
        )

@app.delete("/api/cache")
async def clear_cache():
    """Clear all cached data."""
    try:
        if not vector_store:
            return JSONResponse(
                status_code=503,
                content={"detail": "Vector store not available"}
            )
        
        success = vector_store.clear_cache()
        if success:
            return JSONResponse(content={"success": True, "message": "Cache cleared successfully"})
        else:
            return JSONResponse(
                status_code=500,
                content={"detail": "Failed to clear cache"}
            )
    except Exception as e:
        logger.error(f"Clear cache API error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Clear cache error: {str(e)}"}
        )

@app.get("/api/history")
async def get_query_history(limit: int = 20, search: Optional[str] = None):
    """Get query history with optional search."""
    try:
        if not vector_store:
            return JSONResponse(content={
                "queries": [],
                "total_count": 0
            })
        
        if search:
            queries = vector_store.search_queries(search, limit)
        else:
            queries = vector_store.get_all_queries(limit)
        
        response_data = {
            "queries": queries,
            "total_count": len(queries)
        }
        
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"History API error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"History error: {str(e)}"}
        )

@app.delete("/api/history/{query_hash}")
async def delete_query(query_hash: str):
    """Delete a specific query from history."""
    try:
        if not vector_store:
            return JSONResponse(
                status_code=503,
                content={"detail": "Vector store not available"}
            )
        
        success = vector_store.delete_query(query_hash)
        if success:
            return JSONResponse(content={"success": True, "message": "Query deleted successfully"})
        else:
            return JSONResponse(
                status_code=404,
                content={"detail": "Query not found"}
            )
    except Exception as e:
        logger.error(f"Delete query API error: {e}")
        return JSONResponse(
            status_code=500,
            content={"detail": f"Delete error: {str(e)}"}
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    try:
        components = {
            "vector_store": "operational" if vector_store else "unavailable",
            "query_processor": "operational" if query_processor else "unavailable",
            "summarizer": "operational" if summarizer else "unavailable",
            "similarity_checker": "operational" if similarity_checker else "unavailable"
        }
        
        # Test vector store if available
        if vector_store:
            try:
                vector_store.get_cache_stats()
            except Exception as e:
                components["vector_store"] = f"error: {str(e)}"
        
        status = "healthy" if all(comp != "unavailable" for comp in components.values()) else "degraded"
        
        response_data = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "components": components
        }
        
        return JSONResponse(content=response_data)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("🚀 Starting Web Query Agent API...")
    logger.info("✅ FastAPI application started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("🛑 Shutting down Web Query Agent API...")
    logger.info("✅ Application shutdown complete")