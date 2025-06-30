"""
Content Summarizer module.
Uses Gemini to create comprehensive summaries from multiple scraped sources.
"""

import json
import logging
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from .config import config
from .models import ScrapedContent, SummaryResult

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class ContentSummarizer:
    """Summarizes content from multiple sources using Gemini."""
    
    def __init__(self):
        """Initialize the content summarizer."""
        try:
            # Initialize Gemini model for summarization
            self.llm = ChatGoogleGenerativeAI(
                model=config.GEMINI_MODEL,
                temperature=config.GEMINI_TEMPERATURE,
                google_api_key=config.GOOGLE_API_KEY
            )
            
            logger.info("Content summarizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize content summarizer: {e}")
            raise
    
    def summarize_content(self, query: str, scraped_content: List[ScrapedContent]) -> SummaryResult:
        """
        Create a comprehensive summary from multiple scraped sources.
        
        Args:
            query: The original user query
            scraped_content: List of scraped content from different sources
            
        Returns:
            SummaryResult object with summary and metadata
        """
        try:
            # Filter successful scrapes with meaningful content
            valid_content = [
                content for content in scraped_content 
                if content.success and len(content.content) >= config.MIN_CONTENT_LENGTH
            ]
            
            if not valid_content:
                logger.warning("No valid content available for summarization")
                return self._create_empty_summary(query, scraped_content)
            
            # Create system prompt for summarization
            system_prompt = self._create_system_prompt()
            
            # Create user prompt with content
            user_prompt = self._create_user_prompt(query, valid_content)
            
            # Get summary from Gemini
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse the JSON response
            summary_data = self._parse_summary_response(response.content)
            
            # Create SummaryResult object
            summary_result = SummaryResult(
                summary=summary_data["summary"],
                key_points=summary_data["key_points"],
                sources=[content.url for content in valid_content],
                confidence=summary_data["confidence"],
                word_count=len(summary_data["summary"].split())
            )
            
            logger.info(f"Successfully created summary for query '{query}' using {len(valid_content)} sources")
            return summary_result
            
        except Exception as e:
            logger.error(f"Error summarizing content for query '{query}': {e}")
            return self._create_error_summary(query, scraped_content, str(e))
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for content summarization."""
        return """You are an expert content summarizer that creates comprehensive, accurate summaries from multiple web sources.

Your task is to:
1. Synthesize information from multiple sources into a coherent, well-structured summary
2. Extract the most important key points and insights
3. Maintain accuracy and avoid hallucination
4. Provide balanced coverage of different perspectives if they exist
5. Focus on answering the user's specific query

Guidelines:
- Create a comprehensive summary that directly addresses the user's query
- Identify and highlight the most important key points (3-7 points)
- Maintain factual accuracy and cite when information comes from specific sources
- Use clear, concise language appropriate for a general audience
- Avoid redundancy between sources
- If sources contradict each other, mention both perspectives
- Focus on recent and authoritative information
- Indicate your confidence level in the summary

Respond ONLY with a JSON object in this exact format:
{
    "summary": "Comprehensive summary that directly answers the user's query...",
    "key_points": [
        "Key point 1 with specific details",
        "Key point 2 with specific details",
        "Key point 3 with specific details"
    ],
    "confidence": 0.0-1.0
}

Ensure the summary is substantial (at least 150 words) but concise (maximum 500 words)."""
    
    def _create_user_prompt(self, query: str, content_list: List[ScrapedContent]) -> str:
        """
        Create the user prompt with query and content.
        
        Args:
            query: The original user query
            content_list: List of valid scraped content
            
        Returns:
            Formatted user prompt string
        """
        prompt_parts = [
            f"User Query: {query}",
            "",
            "Content from multiple web sources:"
        ]
        
        for i, content in enumerate(content_list, 1):
            # Truncate very long content to stay within token limits
            truncated_content = content.content[:2000] + "..." if len(content.content) > 2000 else content.content
            
            prompt_parts.extend([
                f"\n--- Source {i}: {content.title} ({content.url}) ---",
                truncated_content,
                ""
            ])
        
        prompt_parts.extend([
            f"\nPlease create a comprehensive summary that directly answers the query: '{query}'",
            "Base your response on the provided content and follow the JSON format specified in the system prompt."
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_summary_response(self, response_content: str) -> dict:
        """
        Parse the JSON response from Gemini.
        
        Args:
            response_content: Raw response from Gemini
            
        Returns:
            Parsed summary data dictionary
        """
        try:
            # Try to parse as JSON
            summary_data = json.loads(response_content)
            
            # Validate required fields
            required_fields = ["summary", "key_points", "confidence"]
            for field in required_fields:
                if field not in summary_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate types
            if not isinstance(summary_data["summary"], str):
                raise ValueError("Summary must be a string")
            
            if not isinstance(summary_data["key_points"], list):
                raise ValueError("Key points must be a list")
            
            if not isinstance(summary_data["confidence"], (int, float)):
                raise ValueError("Confidence must be a number")
            
            # Ensure confidence is in valid range
            summary_data["confidence"] = max(0.0, min(1.0, float(summary_data["confidence"])))
            
            return summary_data
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            # Try to extract content if JSON parsing fails
            return self._extract_fallback_summary(response_content)
        
        except ValueError as e:
            logger.error(f"Invalid summary data structure: {e}")
            return self._extract_fallback_summary(response_content)
    
    def _extract_fallback_summary(self, response_content: str) -> dict:
        """
        Extract summary information when JSON parsing fails.
        
        Args:
            response_content: Raw response content
            
        Returns:
            Fallback summary data dictionary
        """
        # Use the raw content as summary
        summary = response_content.strip()
        
        # Try to extract key points if they exist in the text
        key_points = []
        lines = summary.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith(('•', '-', '*', '1.', '2.', '3.')):
                key_points.append(line.lstrip('•-*123456789. '))
        
        # If no key points found, create generic ones
        if not key_points:
            sentences = summary.split('. ')
            key_points = [sentence.strip() + '.' for sentence in sentences[:3] if len(sentence.strip()) > 20]
        
        return {
            "summary": summary,
            "key_points": key_points[:7],  # Limit to 7 key points
            "confidence": 0.5  # Lower confidence for fallback
        }
    
    def _create_empty_summary(self, query: str, scraped_content: List[ScrapedContent]) -> SummaryResult:
        """Create an empty summary when no valid content is available."""
        failed_urls = [content.url for content in scraped_content if not content.success]
        
        return SummaryResult(
            summary=f"I was unable to find sufficient information to answer your query: '{query}'. "
                   f"The web scraping encountered issues with the available sources.",
            key_points=[
                "No sufficient content was found for this query",
                "Web scraping may have encountered technical issues",
                "Consider rephrasing your query or trying again later"
            ],
            sources=failed_urls,
            confidence=0.0,
            word_count=0
        )
    
    def _create_error_summary(self, query: str, scraped_content: List[ScrapedContent], error_message: str) -> SummaryResult:
        """Create an error summary when summarization fails."""
        return SummaryResult(
            summary=f"An error occurred while processing your query: '{query}'. "
                   f"Error: {error_message}",
            key_points=[
                "Content summarization encountered an error",
                "The query processing system may be temporarily unavailable",
                "Please try again later or contact support"
            ],
            sources=[content.url for content in scraped_content],
            confidence=0.0,
            word_count=0
        )