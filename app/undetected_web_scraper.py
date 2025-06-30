"""
Debug Enhanced Undetected Web Scraper with detailed search logging.
This version includes comprehensive debugging for search functionality.
"""

import asyncio
import logging
import re
import urllib.parse
import random
from typing import List, Optional
from datetime import datetime

try:
    from undetected_playwright import async_playwright
    UNDETECTED_AVAILABLE = True
except ImportError:
    # Fallback to regular playwright
    from playwright.async_api import async_playwright
    UNDETECTED_AVAILABLE = False

from bs4 import BeautifulSoup

from .config import config
from .models import ScrapedContent, WebSearchResult

# Set up logging
logging.basicConfig(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

class EnhancedUndetectedWebScraper:
    """Enhanced Undetected Playwright scraper with comprehensive browser headers and detailed debugging."""
    
    def __init__(self):
        """Initialize the enhanced undetected web scraper."""
        self.browser = None
        self.playwright = None
        
        if UNDETECTED_AVAILABLE:
            logger.info("Using undetected-playwright-python for maximum stealth")
        else:
            logger.warning("undetected-playwright-python not available, using regular playwright")
        
        # Comprehensive browser configurations with realistic headers
        self.browser_configs = [
            {
                "name": "Chrome Windows",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
                "viewport": {"width": 1920, "height": 1080},
                "platform": "Win32",
                "headers": {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br, zstd',
                    'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"',
                    'sec-ch-ua-form-factors': '"Desktop"',
                    'sec-ch-ua-full-version-list': '"Not A(Brand";v="8.0.0.0", "Chromium";v="132.0.6834.110", "Google Chrome";v="132.0.6834.110"',
                    'sec-ch-ua-arch': '"x86"',
                    'sec-ch-ua-bitness': '"64"',
                    'sec-ch-ua-wow64': '?0',
                    'sec-fetch-dest': 'document',
                    'sec-fetch-mode': 'navigate',
                    'sec-fetch-site': 'none',
                    'sec-fetch-user': '?1',
                    'upgrade-insecure-requests': '1',
                    'cache-control': 'max-age=0',
                    'dnt': '1',
                    'sec-gpc': '1'
                }
            },
            {
                "name": "Chrome MacOS",
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
                "viewport": {"width": 1440, "height": 900},
                "platform": "MacIntel",
                "headers": {
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept-Encoding': 'gzip, deflate, br, zstd',
                    'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"macOS"',
                    'sec-ch-ua-form-factors': '"Desktop"',
                    'sec-fetch-dest': 'document',
                    'sec-fetch-mode': 'navigate',
                    'sec-fetch-site': 'none',
                    'sec-fetch-user': '?1',
                    'upgrade-insecure-requests': '1',
                    'cache-control': 'max-age=0'
                }
            }
        ]
        
        logger.info("Enhanced undetected web scraper initialized")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()
    
    async def initialize(self):
        """Initialize the undetected browser."""
        try:
            self.playwright = await async_playwright().start()
            
            # Choose random browser configuration
            self.config_choice = random.choice(self.browser_configs)
            logger.info(f"Using browser config: {self.config_choice['name']}")
            
            # Launch browser with undetected settings
            launch_options = {
                "headless": config.HEADLESS_BROWSER,
                "args": [
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-blink-features=AutomationControlled",
                    "--disable-extensions",
                    "--no-first-run",
                    "--disable-default-apps",
                    f"--user-agent={self.config_choice['user_agent']}"
                ]
            }
            
            self.browser = await self.playwright.chromium.launch(**launch_options)
            
            logger.info("Enhanced undetected browser initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced undetected browser: {e}")
            raise
    
    async def cleanup(self):
        """Clean up browser resources."""
        try:
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
                logger.info("Enhanced undetected browser closed successfully")
        except Exception as e:
            logger.error(f"Error closing browser: {e}")
    
    async def search_and_scrape(self, query: str) -> List[ScrapedContent]:
        """Search for a query and scrape the top results using enhanced undetected techniques."""
        try:
            # Get search results
            search_results = await self._get_search_results_enhanced(query)
            
            if not search_results:
                logger.warning(f"No search results found for query: {query}")
                return []
            
            # Scrape content from each result
            scraped_content = []
            for i, result in enumerate(search_results[:config.MAX_SEARCH_RESULTS]):
                logger.info(f"Scraping result {i+1}/{len(search_results)}: {result.url}")
                
                content = await self._scrape_url_enhanced(result.url, result.title)
                if content:
                    scraped_content.append(content)
                
                # Add realistic delay
                await self._human_delay()
            
            logger.info(f"Successfully scraped {len(scraped_content)} pages for query: {query}")
            return scraped_content
            
        except Exception as e:
            logger.error(f"Error during enhanced search and scrape for query '{query}': {e}")
            return []
    
    async def _get_search_results_enhanced(self, query: str) -> List[WebSearchResult]:
        """Get search results using enhanced undetected techniques with detailed debugging."""
        try:
            logger.info(f"🔍 Starting search for query: '{query}'")
            
            # Create enhanced context
            context = await self._create_enhanced_context()
            page = await context.new_page()
            
            # Apply enhanced scripts
            await self._apply_enhanced_scripts(page)
            
            # Try multiple search engines
            results = []
            
            # Try DuckDuckGo first (more reliable)
            logger.info("🦆 Trying DuckDuckGo search...")
            try:
                ddg_results = await self._search_duckduckgo_enhanced(page, query)
                results.extend(ddg_results)
                logger.info(f"🦆 DuckDuckGo found {len(ddg_results)} results")
            except Exception as e:
                logger.error(f"❌ DuckDuckGo search failed: {e}")
                logger.exception("DuckDuckGo search error details:")
            
            # Try Google if needed
            if len(results) < config.MAX_SEARCH_RESULTS:
                logger.info("🔍 Trying Google search...")
                try:
                    google_results = await self._search_google_enhanced(page, query)
                    results.extend(google_results)
                    logger.info(f"🔍 Google found {len(google_results)} results")
                except Exception as e:
                    logger.error(f"❌ Google search failed: {e}")
                    logger.exception("Google search error details:")
            
            await context.close()
            
            logger.info(f"📊 Total search results found: {len(results)}")
            return results[:config.MAX_SEARCH_RESULTS]
            
        except Exception as e:
            logger.error(f"Error getting enhanced search results: {e}")
            logger.exception("Search results error details:")
            return []
    
    async def _create_enhanced_context(self):
        """Create a browser context with enhanced configurations."""
        context = await self.browser.new_context(
            viewport=self.config_choice["viewport"],
            user_agent=self.config_choice["user_agent"],
            locale='en-US',
            timezone_id='America/New_York',
            permissions=['geolocation'],
            geolocation={'latitude': 40.7589, 'longitude': -73.9851},
            extra_http_headers=self.config_choice["headers"],
            java_script_enabled=True
        )
        
        return context
    
    async def _apply_enhanced_scripts(self, page):
        """Apply enhanced scripts to hide automation traces."""
        # Remove webdriver traces
        await page.add_init_script("""
            delete Object.getPrototypeOf(navigator).webdriver;
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined,
            });
        """)
        
        # Enhanced navigator properties
        await page.add_init_script("""
            Object.defineProperty(navigator, 'plugins', {
                get: () => [
                    {name: "Chrome PDF Plugin", description: "Portable Document Format"},
                    {name: "Chrome PDF Viewer", description: "Portable Document Format"},
                    {name: "Native Client", description: "Native Client"}
                ]
            });
            
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en'],
            });
        """)
    
    async def _search_duckduckgo_enhanced(self, page, query: str) -> List[WebSearchResult]:
        """Search DuckDuckGo with enhanced techniques and detailed debugging."""
        try:
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://html.duckduckgo.com/html/?q={encoded_query}"
            
            logger.info(f"🦆 Navigating to DuckDuckGo: {url}")
            
            # Navigate with human-like behavior
            response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            logger.info(f"🦆 DuckDuckGo response status: {response.status}")
            
            if response.status >= 400:
                logger.error(f"❌ DuckDuckGo returned HTTP {response.status}")
                return []
            
            await self._simulate_realistic_behavior(page)
            
            # Take screenshot for debugging
            await page.screenshot(path="debug_duckduckgo.png")
            logger.info("📸 DuckDuckGo screenshot saved as debug_duckduckgo.png")
            
            # Get page title for verification
            title = await page.title()
            logger.info(f"🦆 DuckDuckGo page title: {title}")
            
            # Extract results
            content = await page.content()
            logger.info(f"🦆 DuckDuckGo page content length: {len(content)} characters")
            
            # Debug: Save HTML content
            with open("debug_duckduckgo.html", "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("💾 DuckDuckGo HTML saved as debug_duckduckgo.html")
            
            soup = BeautifulSoup(content, 'html.parser')
            
            results = []
            
            # Try multiple result selectors
            result_selectors = [
                'div.result',
                '.result',
                'div[class*="result"]',
                '.web-result',
                '.result__body'
            ]
            
            result_containers = []
            for selector in result_selectors:
                containers = soup.select(selector)
                if containers:
                    logger.info(f"🦆 Found {len(containers)} results with selector: {selector}")
                    result_containers = containers
                    break
                else:
                    logger.info(f"🦆 No results found with selector: {selector}")
            
            if not result_containers:
                logger.warning("🦆 No result containers found with any selector")
                # Debug: Print first 1000 chars of HTML
                logger.info(f"🦆 HTML preview: {content[:1000]}...")
                return []
            
            for i, container in enumerate(result_containers[:10]):  # Limit to first 10
                try:
                    # Try multiple title selectors
                    title_selectors = [
                        'a.result__a',
                        '.result__a',
                        'a[class*="result"]',
                        'h2 a',
                        'h3 a',
                        'a'
                    ]
                    
                    title_link = None
                    for title_selector in title_selectors:
                        title_link = container.select_one(title_selector)
                        if title_link:
                            break
                    
                    if not title_link:
                        logger.debug(f"🦆 No title link found in result {i+1}")
                        continue
                    
                    title = title_link.get_text(strip=True)
                    url = title_link.get('href', '')
                    
                    if not title or not url:
                        logger.debug(f"🦆 Missing title or URL in result {i+1}")
                        continue
                    
                    # Get snippet
                    snippet_selectors = [
                        'a.result__snippet',
                        '.result__snippet',
                        '.snippet'
                    ]
                    
                    snippet = None
                    for snippet_selector in snippet_selectors:
                        snippet_elem = container.select_one(snippet_selector)
                        if snippet_elem:
                            snippet = snippet_elem.get_text(strip=True)
                            break
                    
                    if self._is_valid_url(url):
                        results.append(WebSearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                            position=len(results) + 1
                        ))
                        logger.info(f"🦆 Added result {len(results)}: {title}")
                    else:
                        logger.debug(f"🦆 Invalid URL in result {i+1}: {url}")
                        
                except Exception as e:
                    logger.debug(f"🦆 Error extracting DuckDuckGo result {i+1}: {e}")
                    continue
            
            logger.info(f"🦆 DuckDuckGo enhanced search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in DuckDuckGo enhanced search: {e}")
            logger.exception("DuckDuckGo search error details:")
            return []
    
    async def _search_google_enhanced(self, page, query: str) -> List[WebSearchResult]:
        """Search Google with maximum enhanced techniques and detailed debugging."""
        try:
            encoded_query = urllib.parse.quote_plus(query)
            url = f"https://www.google.com/search?q={encoded_query}&hl=en"
            
            logger.info(f"🔍 Navigating to Google: {url}")
            
            # Navigate with enhanced behavior
            response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            logger.info(f"🔍 Google response status: {response.status}")
            
            if response.status >= 400:
                logger.error(f"❌ Google returned HTTP {response.status}")
                return []
            
            # Handle consent
            consent_handled = await self._handle_google_consent(page)
            if consent_handled:
                logger.info("🔍 Google consent handled")
                await self._human_delay(2, 4)
            
            await self._simulate_realistic_behavior(page)
            
            # Take screenshot for debugging
            await page.screenshot(path="debug_google.png")
            logger.info("📸 Google screenshot saved as debug_google.png")
            
            # Get page title
            title = await page.title()
            logger.info(f"🔍 Google page title: {title}")
            
            # Extract results
            content = await page.content()
            logger.info(f"🔍 Google page content length: {len(content)} characters")
            
            # Debug: Save HTML content
            with open("debug_google.html", "w", encoding="utf-8") as f:
                f.write(content)
            logger.info("💾 Google HTML saved as debug_google.html")
            
            results = []
            
            # Try to find h3 elements (Google result titles)
            h3_elements = await page.locator('h3').all()
            logger.info(f"🔍 Found {len(h3_elements)} h3 elements")
            
            if len(h3_elements) == 0:
                # Try alternative selectors
                alt_selectors = [
                    '[data-header-feature] h3',
                    '.g h3',
                    '.yuRUbf h3',
                    'h3',
                    '.LC20lb'
                ]
                
                for selector in alt_selectors:
                    elements = await page.locator(selector).all()
                    if elements:
                        logger.info(f"🔍 Found {len(elements)} elements with selector: {selector}")
                        h3_elements = elements
                        break
            
            for i, h3 in enumerate(h3_elements[:10]):  # Limit to first 10
                try:
                    title = await h3.inner_text()
                    if not title or len(title.strip()) < 3:
                        continue
                    
                    # Find parent link
                    link_locator = h3.locator('xpath=ancestor::a[1]')
                    if await link_locator.count() == 0:
                        # Try alternative: find nearby link
                        link_locator = h3.locator('xpath=../a | xpath=../../a | xpath=../../../a')
                    
                    if await link_locator.count() == 0:
                        logger.debug(f"🔍 No link found for result {i+1}")
                        continue
                    
                    url = await link_locator.first.get_attribute('href')
                    if not url:
                        continue
                    
                    # Clean Google redirect URLs
                    if url.startswith('/url?q='):
                        url = url.replace('/url?q=', '').split('&')[0]
                        url = urllib.parse.unquote(url)
                    
                    if self._is_valid_url(url):
                        results.append(WebSearchResult(
                            title=title.strip(),
                            url=url,
                            snippet=None,
                            position=len(results) + 1
                        ))
                        logger.info(f"🔍 Added Google result {len(results)}: {title}")
                        
                        if len(results) >= config.MAX_SEARCH_RESULTS:
                            break
                    else:
                        logger.debug(f"🔍 Invalid URL in result {i+1}: {url}")
                        
                except Exception as e:
                    logger.debug(f"🔍 Error extracting Google result {i+1}: {e}")
                    continue
            
            logger.info(f"🔍 Google enhanced search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in Google enhanced search: {e}")
            logger.exception("Google search error details:")
            return []
    
    async def _handle_google_consent(self, page) -> bool:
        """Handle Google consent with multiple strategies."""
        try:
            consent_selectors = [
                'button:has-text("Accept all")',
                'button:has-text("I agree")', 
                'button:has-text("Accept")',
                'button:has-text("Reject all")',
                '#L2AGLb',
                '[aria-label="Accept all"]'
            ]
            
            for selector in consent_selectors:
                try:
                    element = page.locator(selector)
                    if await element.count() > 0:
                        await element.first.click()
                        logger.info(f"🔍 Clicked consent button: {selector}")
                        return True
                except:
                    continue
                    
            return False
        except Exception as e:
            logger.debug(f"Error handling consent: {e}")
            return False
    
    async def _simulate_realistic_behavior(self, page):
        """Simulate realistic human behavior on the page."""
        try:
            # Random mouse movement
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            await page.mouse.move(x, y)
            
            # Random scroll
            scroll_distance = random.randint(200, 500)
            await page.evaluate(f"window.scrollBy(0, {scroll_distance})")
            await self._human_delay(1, 2)
            
        except Exception as e:
            logger.debug(f"Error simulating behavior: {e}")
    
    async def _scrape_url_enhanced(self, url: str, title: Optional[str] = None) -> Optional[ScrapedContent]:
        """Scrape content from a URL using enhanced techniques."""
        try:
            context = await self._create_enhanced_context()
            page = await context.new_page()
            
            await self._apply_enhanced_scripts(page)
            
            # Navigate to URL
            response = await page.goto(url, timeout=config.SCRAPE_TIMEOUT * 1000)
            
            if not response or response.status >= 400:
                await context.close()
                return ScrapedContent(
                    url=url,
                    title=title,
                    content="",
                    content_length=0,
                    success=False,
                    error_message=f"HTTP {response.status if response else 'No response'}"
                )
            
            await page.wait_for_load_state("domcontentloaded")
            
            # Extract title if not provided
            if not title:
                try:
                    title_elem = await page.query_selector('title')
                    title = await title_elem.inner_text() if title_elem else url
                except:
                    title = url
            
            # Extract content
            content = await self._extract_page_content(page)
            
            await context.close()
            
            return ScrapedContent(
                url=url,
                title=title,
                content=content,
                content_length=len(content),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error scraping URL {url} with enhanced method: {e}")
            return ScrapedContent(
                url=url,
                title=title,
                content="",
                content_length=0,
                success=False,
                error_message=str(e)
            )
    
    async def _extract_page_content(self, page):
        """Extract clean text content from a page."""
        try:
            content = await page.content()
            soup = BeautifulSoup(content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer", "aside", "iframe"]):
                element.decompose()
            
            # Find main content
            main_content = (
                soup.find('main') or
                soup.find('article') or
                soup.find('div', class_=re.compile(r'content|main|article', re.I)) or
                soup.find('body')
            )
            
            if main_content:
                text = main_content.get_text(separator=' ', strip=True)
            else:
                text = soup.get_text(separator=' ', strip=True)
            
            # Clean up text
            text = self._clean_text(text)
            
            if len(text) > config.MAX_CONTENT_LENGTH:
                text = text[:config.MAX_CONTENT_LENGTH] + "..."
            
            return text
            
        except Exception as e:
            logger.error(f"Error extracting page content: {e}")
            return ""
    
    async def _human_delay(self, min_seconds: float = 1.0, max_seconds: float = 3.0):
        """Add human-like delay."""
        delay = random.uniform(min_seconds, max_seconds)
        await asyncio.sleep(delay)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]+', ' ', text)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) > 10]
        return ' '.join(cleaned_lines).strip()
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if a URL is valid for scraping."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            if not parsed.scheme or not parsed.netloc:
                return False
            
            skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx']
            if any(url.lower().endswith(ext) for ext in skip_extensions):
                return False
            
            skip_domains = ['facebook.com', 'twitter.com', 'instagram.com', 'linkedin.com', 'youtube.com']
            if any(domain in parsed.netloc.lower() for domain in skip_domains):
                return False
            
            return True
            
        except Exception:
            return False