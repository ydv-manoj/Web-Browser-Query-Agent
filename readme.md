# Web Browser Query Agent - Architecture & Implementation Plan

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│                 │    │                  │    │                 │
│   CLI/FastAPI   │───▶│  Query Processor │───▶│   Web Scraper   │
│   Interface     │    │     (Gemini)     │    │  (Playwright)   │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│                 │    │                  │    │                 │
│  Results Cache  │◀───│ Similarity Check │◀───│   Content       │
│   (JSON/DB)     │    │  (Embeddings)    │    │  Summarizer     │
│                 │    │                  │    │   (Gemini)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔄 Data Flow

1. **User Input** → CLI receives query
2. **Query Classification** → Gemini validates if query is valid
3. **Similarity Check** → Check cached queries using embeddings
4. **Web Scraping** → Playwright scrapes top 5 Google results
5. **Content Processing** → Gemini summarizes scraped content
6. **Storage** → Save results with query embeddings for future use
7. **Response** → Return formatted results to user

## 🛠️ Technology Stack

### Core Framework
- **Backend**: FastAPI (for later web interface)
- **CLI**: Python argparse/click for MVP
- **AI Model**: Google Gemini (via langchain-google-genai)

### Web Scraping
- **LangChain**: Framework orchestration
- **Playwright**: Headless browser automation
- **Tools**: NavigateTool, ExtractTextTool from LangChain

### Data Processing
- **Embeddings**: Google Gemini embeddings for similarity
- **Storage**: JSON files for MVP (PostgreSQL for production)
- **Caching**: In-memory + file-based cache

### Dependencies
```bash
pip install fastapi uvicorn
pip install langchain langchain-community
pip install langchain-google-genai
pip install playwright
pip install python-dotenv
pip install click
```

## 📁 Project Structure

```
web_query_agent/
├── main.py              # CLI entry point
├── app/
│   ├── __init__.py
│   ├── config.py        # Configuration and environment
│   ├── models.py        # Pydantic models
│   ├── query_processor.py   # Query validation and classification
│   ├── web_scraper.py       # Playwright web scraping logic
│   ├── similarity_checker.py # Embedding-based similarity
│   ├── summarizer.py        # Content summarization with Gemini
│   └── cache_manager.py     # Results caching system
├── cache/
│   ├── queries.json     # Cached query results
│   └── embeddings.json  # Query embeddings
├── requirements.txt
└── .env                 # API keys
```

## 🔧 Core Components

### 1. Query Processor
- **Input Validation**: Uses Gemini to classify valid/invalid queries
- **Query Normalization**: Standardizes query format
- **Intent Detection**: Understands what user is looking for

### 2. Similarity Checker
- **Embedding Generation**: Creates vector representations of queries
- **Cosine Similarity**: Compares new queries with cached ones
- **Threshold Matching**: Configurable similarity threshold (0.85+)

### 3. Web Scraper
- **Search Integration**: Automated Google/DuckDuckGo search
- **Content Extraction**: Scrapes top 5 search results
- **Data Cleaning**: Removes ads, navigation, irrelevant content

### 4. Content Summarizer
- **Multi-source Synthesis**: Combines information from multiple pages
- **Structured Output**: Organized, coherent summaries
- **Source Attribution**: Maintains source links and credibility

### 5. Cache Manager
- **Query Storage**: Stores queries, embeddings, and results
- **Retrieval Logic**: Fast lookup for similar queries
- **Data Persistence**: JSON files for MVP, DB for production

## 🚀 Implementation Phases

### Phase 1: CLI MVP (Current Focus)
- [x] Basic query input handling
- [ ] Gemini integration for query validation
- [ ] Playwright web scraping (top 5 results)
- [ ] Basic content summarization
- [ ] Simple JSON caching

### Phase 2: Advanced Features
- [ ] Similarity checking with embeddings
- [ ] Improved query classification
- [ ] Better content extraction
- [ ] Enhanced summarization

### Phase 3: Web Interface
- [ ] FastAPI backend
- [ ] React/HTML frontend
- [ ] API endpoints
- [ ] WebSocket for real-time updates

### Phase 4: Production Ready
- [ ] Database integration (PostgreSQL)
- [ ] Caching optimization (Redis)
- [ ] Rate limiting and security
- [ ] Containerization (Docker)

## 🎯 Success Metrics

1. **Query Classification Accuracy**: >95% valid/invalid detection
2. **Similarity Detection**: >90% accuracy for similar queries
3. **Scraping Success Rate**: >95% successful page extractions
4. **Response Time**: <30 seconds for new queries, <2 seconds for cached
5. **User Satisfaction**: Clear, comprehensive, and accurate results

## 🔒 Security Considerations

- **API Key Management**: Secure storage of Gemini API keys
- **Rate Limiting**: Prevent abuse of scraping capabilities
- **Content Filtering**: Avoid scraping malicious or inappropriate content
- **Data Privacy**: No storage of sensitive user information

## 📊 Testing Strategy

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Response time and throughput
- **User Acceptance Tests**: Real-world query scenarios

This architecture provides a scalable foundation that can evolve from a simple CLI tool to a full-featured web application while maintaining clean separation of concerns and extensibility.