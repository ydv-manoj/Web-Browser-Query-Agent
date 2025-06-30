"""
Configuration module for the Web Query Agent.
Handles environment variables, API keys, and application settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class Config:
    """Application configuration class."""
    
    # API Keys
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    
    # Gemini Model Configuration
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    GEMINI_MAX_TOKENS: Optional[int] = None
    
    # Scraping Configuration
    MAX_SEARCH_RESULTS: int = int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    SCRAPE_TIMEOUT: int = int(os.getenv("SCRAPE_TIMEOUT", "30"))
    HEADLESS_BROWSER: bool = os.getenv("HEADLESS_BROWSER", "true").lower() == "true"
    
    # Similarity Thresholds
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.85"))
    
    # Cache Configuration
    CACHE_DIR: Path = Path("cache")
    QUERIES_CACHE_FILE: Path = CACHE_DIR / "queries.json"
    EMBEDDINGS_CACHE_FILE: Path = CACHE_DIR / "embeddings.json"
    CACHE_EXPIRY_DAYS: int = int(os.getenv("CACHE_EXPIRY_DAYS", "7"))
    
    # Search Engine Configuration
    SEARCH_ENGINE: str = os.getenv("SEARCH_ENGINE", "duckduckgo")  # duckduckgo or google
    USER_AGENT: str = os.getenv("USER_AGENT", 
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")
    
    # Validation Configuration
    MIN_CONTENT_LENGTH: int = int(os.getenv("MIN_CONTENT_LENGTH", "100"))
    MAX_CONTENT_LENGTH: int = int(os.getenv("MAX_CONTENT_LENGTH", "10000"))
    
    # Logging Configuration
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present."""
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        # Create cache directory if it doesn't exist
        cls.CACHE_DIR.mkdir(exist_ok=True)
        
        return True

# Global config instance
config = Config()

# Validate configuration on import
config.validate_config()