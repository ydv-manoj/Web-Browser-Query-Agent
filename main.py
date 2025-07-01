#!/usr/bin/env python3
"""
Web Browser Query Agent - CLI and Web Interface
A tool that validates queries, searches the web, and provides AI-powered summaries.
Supports both CLI and web interface via FastAPI.
"""

import asyncio
import click
import logging
import time
import uvicorn
from datetime import datetime
from typing import Optional

# Import application modules
from app.config import config
from app.models import QueryResult, AgentResponse, QueryStatus
from app.query_processor import QueryProcessor
from app.undetected_web_scraper import EnhancedUndetectedWebScraper
from app.summarizer import ContentSummarizer
from app.similarity_checker import SimilarityChecker
from app.cache_manager import VectorStoreManager

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,  # Default to INFO for CLI
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
    ]
)
logger = logging.getLogger(__name__)

class WebQueryAgent:
    """Main Web Query Agent class that orchestrates all components."""
    
    def __init__(self):
        """Initialize the web query agent."""
        logger.info("Initializing Web Query Agent...")
        
        self.query_processor = QueryProcessor()
        logger.info("✅ Query processor initialized")
        
        self.similarity_checker = SimilarityChecker()
        logger.info("✅ Similarity checker initialized")
        
        self.summarizer = ContentSummarizer()
        logger.info("✅ Content summarizer initialized")
        
        self.vector_store = VectorStoreManager()
        logger.info("✅ Vector store initialized")
        
        logger.info("🚀 Web Query Agent initialized successfully")
    
    async def process_query(self, query: str, use_cache: bool = True) -> AgentResponse:
        """
        Process a user query through the complete pipeline.
        
        Args:
            query: The user's search query
            use_cache: Whether to use cached/similar results
            
        Returns:
            AgentResponse with results or error information
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Starting to process query: '{query}'")
            
            # Step 1: Normalize and validate the query
            logger.info("📝 Step 1: Normalizing and validating query...")
            normalized_query = self.query_processor.normalize_query(query)
            query_hash = self.query_processor.generate_query_hash(normalized_query)
            
            logger.info(f"✅ Normalized query: '{normalized_query}'")
            logger.info(f"🔑 Query hash: {query_hash}")
            
            # Step 2: Classify the query
            logger.info("🔍 Step 2: Classifying query...")
            classification = self.query_processor.classify_query(normalized_query)
            logger.info(f"📋 Classification: {classification.status.value} (confidence: {classification.confidence:.2f})")
            
            if classification.status == QueryStatus.INVALID:
                execution_time = time.time() - start_time
                logger.warning(f"❌ Query invalid: {classification.reason}")
                return AgentResponse(
                    success=False,
                    message=f"Invalid query: {classification.reason}",
                    query=normalized_query,
                    error=classification.reason,
                    execution_time=execution_time
                )
            
            if use_cache:
                # Step 3: Check for cached results
                logger.info("💾 Step 3: Checking for cached results...")
                cached_result = self.vector_store.get_cached_result(query_hash)
                if cached_result:
                    execution_time = time.time() - start_time
                    cached_result.cache_hit = True
                    logger.info("✅ Found cached result!")
                    
                    return AgentResponse(
                        success=True,
                        message="Found cached result for your query",
                        query=normalized_query,
                        result=cached_result,
                        execution_time=execution_time
                    )
                else:
                    logger.info("🔍 No cached result found")
                
                # Step 4: Check for similar queries using vector search
                logger.info("🔄 Step 4: Checking for similar queries using vector search...")
                query_embedding = self.query_processor.generate_embedding(normalized_query)
                similar_query = self.similarity_checker.find_similar_query(query_embedding, self.vector_store)
                
                if similar_query:
                    execution_time = time.time() - start_time
                    similar_query.cached_result.cache_hit = True
                    logger.info(f"✅ Found similar query: '{similar_query.original_query}' (similarity: {similar_query.similarity_score:.2f})")
                    
                    return AgentResponse(
                        success=True,
                        message=f"Found similar query: '{similar_query.original_query}' (similarity: {similar_query.similarity_score:.2f})",
                        query=normalized_query,
                        result=similar_query.cached_result,
                        similar_query_used=similar_query,
                        execution_time=execution_time
                    )
                else:
                    logger.info("🔍 No similar queries found")
            else:
                # Generate embedding even if not using cache
                query_embedding = self.query_processor.generate_embedding(normalized_query)
            
            # Step 5: Perform web scraping with EnhancedUndetectedWebScraper
            logger.info("🌐 Step 5: Starting web scraping...")
            logger.info("🚀 Using EnhancedUndetectedWebScraper...")
            
            scraped_content = await self._scrape_web_content(normalized_query)
            
            if not scraped_content:
                execution_time = time.time() - start_time
                logger.error("❌ Failed to scrape any web content")
                return AgentResponse(
                    success=False,
                    message="Failed to scrape any web content for your query",
                    query=normalized_query,
                    error="Web scraping failed",
                    execution_time=execution_time
                )
            
            successful_content = [c for c in scraped_content if c.success]
            logger.info(f"✅ Web scraping completed: {len(successful_content)}/{len(scraped_content)} pages scraped successfully")
            
            # Step 6: Summarize content
            logger.info("📄 Step 6: Summarizing content...")
            summary = self.summarizer.summarize_content(normalized_query, scraped_content)
            logger.info("✅ Content summarization completed")
            
            # Step 7: Create query result
            logger.info("💾 Step 7: Creating query result...")
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
            if use_cache:
                logger.info("💾 Step 8: Saving to vector store...")
                self.vector_store.save_query_result(query_result, query_embedding)
                logger.info("✅ Result saved to vector store successfully")
            
            execution_time = time.time() - start_time
            logger.info(f"🎉 Query processing completed successfully in {execution_time:.2f}s")
            
            return AgentResponse(
                success=True,
                message="Successfully processed your query",
                query=normalized_query,
                result=query_result,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Error processing query '{query}': {e}")
            logger.exception("Full error details:")
            
            return AgentResponse(
                success=False,
                message="An error occurred while processing your query",
                query=query,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _scrape_web_content(self, query: str):
        """Scrape web content using EnhancedUndetectedWebScraper only."""
        scraped_content = []
        
        try:
            logger.info("🕵️ Initializing EnhancedUndetectedWebScraper...")
            
            async with EnhancedUndetectedWebScraper() as scraper:
                logger.info("✅ EnhancedUndetectedWebScraper initialized successfully")
                logger.info(f"🔧 Browser config: {scraper.config_choice['name']}")
                logger.info(f"👤 User-Agent: {scraper.config_choice['user_agent']}")
                logger.info(f"📋 Headers count: {len(scraper.config_choice['headers'])}")
                
                logger.info(f"🚀 Starting search and scrape for: '{query}'")
                scraped_content = await scraper.search_and_scrape(query)
                
                if scraped_content:
                    successful_scrapes = [c for c in scraped_content if c.success]
                    failed_scrapes = [c for c in scraped_content if not c.success]
                    
                    logger.info(f"📊 Scraping results:")
                    logger.info(f"   ✅ Successful: {len(successful_scrapes)}")
                    logger.info(f"   ❌ Failed: {len(failed_scrapes)}")
                    logger.info(f"   📝 Total content: {sum(c.content_length for c in successful_scrapes)} characters")
                    
                    if successful_scrapes:
                        logger.info("✅ EnhancedUndetectedWebScraper succeeded!")
                        for i, content in enumerate(successful_scrapes, 1):
                            logger.info(f"   {i}. {content.title} ({content.content_length} chars)")
                    else:
                        logger.warning("⚠️ No successful scrapes found")
                        
                    if failed_scrapes:
                        logger.warning("❌ Some scrapes failed:")
                        for i, content in enumerate(failed_scrapes, 1):
                            logger.warning(f"   {i}. {content.url} - {content.error_message}")
                else:
                    logger.warning("⚠️ No content scraped")
                
        except Exception as e:
            logger.error(f"❌ EnhancedUndetectedWebScraper failed: {e}")
            logger.exception("Full scraping error details:")
            scraped_content = []
        
        return scraped_content


# CLI Interface
@click.group()
@click.version_option(version="1.0.0")
def cli():
    """Web Browser Query Agent - AI-powered web search and summarization tool."""
    pass

@cli.command()
@click.argument('query', required=True)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--json', 'output_json', is_flag=True, help='Output results in JSON format')
@click.option('--no-cache', is_flag=True, help='Skip cache and force new search')
def search(query: str, verbose: bool, output_json: bool, no_cache: bool):
    """Search the web for a query and get an AI-powered summary."""
    
    # Set logging level based on verbose flag
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔧 Verbose mode enabled - showing all logs")
    else:
        logging.getLogger().setLevel(logging.INFO)
    
    # Initialize agent
    logger.info("🚀 Initializing Web Query Agent...")
    agent = WebQueryAgent()
    
    # Process query
    try:
        click.echo(f"🔍 Processing query: '{query}'")
        if verbose:
            click.echo("⏳ This may take a moment...")
            click.echo("📝 Check the logs below for detailed progress...")
        
        if no_cache:
            click.echo("🚫 Cache disabled - forcing new search")
        
        # Run async process
        logger.info("🎯 Starting async query processing...")
        response = asyncio.run(agent.process_query(query, use_cache=not no_cache))
        
        if output_json:
            import json
            click.echo(json.dumps(response.dict(), indent=2, default=str))
        else:
            _display_response(response, verbose)
            
    except KeyboardInterrupt:
        click.echo("\n❌ Search cancelled by user")
        logger.info("🛑 Search cancelled by user")
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        logger.error(f"❌ CLI Error: {e}")
        if verbose:
            import traceback
            click.echo(traceback.format_exc())
            logger.exception("Full CLI error details:")

@cli.command()
def stats():
    """Display cache and application statistics."""
    try:
        agent = WebQueryAgent()
        stats = agent.vector_store.get_cache_stats()
        cache_size = agent.vector_store.get_cache_size()
        
        click.echo("\n📊 Web Query Agent Statistics")
        click.echo("=" * 40)
        click.echo(f"Total queries processed: {stats.total_queries}")
        click.echo(f"Valid queries: {stats.valid_queries}")
        click.echo(f"Invalid queries: {stats.invalid_queries}")
        click.echo(f"Cache hits: {stats.cache_hits}")
        click.echo(f"Cache misses: {stats.cache_misses}")
        click.echo(f"Cache hit rate: {(stats.cache_hits / max(1, stats.cache_hits + stats.cache_misses)) * 100:.1f}%")
        click.echo(f"Average execution time: {stats.average_execution_time:.2f}s")
        click.echo(f"Total pages scraped: {stats.total_pages_scraped}")
        click.echo(f"Vector DB entries: {cache_size['total_entries']}")
        click.echo(f"Database size: {cache_size['total_cache_size_bytes'] / (1024*1024):.1f} MB")
        click.echo(f"Last updated: {stats.last_updated}")
        
    except Exception as e:
        click.echo(f"❌ Error getting statistics: {e}")

@cli.command()
@click.confirmation_option(prompt='Are you sure you want to clear all cached data?')
def clear_cache():
    """Clear all cached query results from vector database."""
    try:
        agent = WebQueryAgent()
        if agent.vector_store.clear_cache():
            click.echo("✅ Vector database cleared successfully")
        else:
            click.echo("❌ Failed to clear vector database")
    except Exception as e:
        click.echo(f"❌ Error clearing cache: {e}")

@cli.command()
@click.argument('query', required=True)
def validate(query: str):
    """Validate a query without performing web search."""
    try:
        agent = WebQueryAgent()
        classification = agent.query_processor.classify_query(query)
        
        click.echo(f"\n🔍 Query Validation Results")
        click.echo("=" * 30)
        click.echo(f"Query: '{query}'")
        click.echo(f"Status: {classification.status.value.upper()}")
        click.echo(f"Confidence: {classification.confidence:.2f}")
        click.echo(f"Reason: {classification.reason}")
        
        if classification.suggested_improvements:
            click.echo("\n💡 Suggestions:")
            for suggestion in classification.suggested_improvements:
                click.echo(f"  • {suggestion}")
                
    except Exception as e:
        click.echo(f"❌ Error validating query: {e}")

@cli.command()
@click.option('--limit', '-l', default=10, help='Number of queries to show')
@click.option('--search', '-s', help='Search term to filter queries')
def history(limit: int, search: Optional[str]):
    """Show query history from vector database."""
    try:
        agent = WebQueryAgent()
        
        if search:
            queries = agent.vector_store.search_queries(search, limit)
            click.echo(f"\n🔍 Search results for '{search}' (showing {len(queries)} results)")
        else:
            queries = agent.vector_store.get_all_queries(limit)
            click.echo(f"\n📚 Query History (showing last {len(queries)} queries)")
        
        click.echo("=" * 60)
        
        for i, query_data in enumerate(queries, 1):
            click.echo(f"{i}. Query: '{query_data['query']}'")
            click.echo(f"   Status: {query_data['status']} | Confidence: {query_data['confidence']:.2f}")
            click.echo(f"   Created: {query_data['created_at']} | Pages: {query_data['pages_scraped']}")
            click.echo(f"   Execution time: {query_data['execution_time']:.2f}s")
            click.echo()
        
        if not queries:
            click.echo("No queries found.")
            
    except Exception as e:
        click.echo(f"❌ Error getting history: {e}")

@cli.command()
@click.option('--host', default='127.0.0.1', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--reload', is_flag=True, help='Enable auto-reload for development')
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI web server."""
    click.echo("🌐 Starting Web Query Agent server...")
    click.echo(f"📍 Server will be available at: http://{host}:{port}")
    click.echo("🔧 Press Ctrl+C to stop the server")
    
    try:
        # Import the FastAPI app
        from app.app import app
        
        # Configure uvicorn
        uvicorn.run(
            "app.app:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except KeyboardInterrupt:
        click.echo("\n🛑 Server stopped by user")
    except Exception as e:
        click.echo(f"❌ Error starting server: {e}")

def _display_response(response: AgentResponse, verbose: bool):
    """Display the agent response in a user-friendly format."""
    
    if not response.success:
        click.echo(f"\n❌ {response.message}")
        if response.error and verbose:
            click.echo(f"Error details: {response.error}")
        return
    
    click.echo(f"\n✅ {response.message}")
    click.echo(f"⏱️  Execution time: {response.execution_time:.2f}s")
    
    if response.similar_query_used:
        click.echo(f"🔄 Used similar query: '{response.similar_query_used.original_query}' (similarity: {response.similar_query_used.similarity_score:.2f})")
    
    if response.result:
        result = response.result
        
        # Display classification
        if verbose:
            click.echo(f"\n📋 Query Classification:")
            click.echo(f"   Status: {result.classification.status.value}")
            click.echo(f"   Confidence: {result.classification.confidence:.2f}")
        
        # Display summary
        if result.summary:
            click.echo(f"\n📄 Summary:")
            click.echo(f"{result.summary.summary}")
            
            click.echo(f"\n🔑 Key Points:")
            for i, point in enumerate(result.summary.key_points, 1):
                click.echo(f"   {i}. {point}")
            
            if verbose and result.summary.sources:
                click.echo(f"\n🔗 Sources ({len(result.summary.sources)}):")
                for i, source in enumerate(result.summary.sources, 1):
                    click.echo(f"   {i}. {source}")
        
        # Display scraping info
        if verbose and result.scraped_content:
            successful_scrapes = [c for c in result.scraped_content if c.success]
            click.echo(f"\n🌐 Web Scraping Results:")
            click.echo(f"   Pages scraped: {len(successful_scrapes)}/{len(result.scraped_content)}")
            click.echo(f"   Total content: {sum(c.content_length for c in successful_scrapes)} characters")

if __name__ == "__main__":
    cli()