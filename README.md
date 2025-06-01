# 🇹🇷 Turkish Airlines Baggage Policy RAG Assistant

> **AI-Powered Customer Service System for Turkish Airlines Baggage Policies**



# ✈️ Multi-Airline RAG Policy Assistant

## Frontend Interface

![Multi-Airline RAG Policy Assistant Interface](streamlit_interface1.png)

The intuitive chat interface allows customers to ask questions in natural language and receive instant, accurate responses about Turkish Airlines baggage policies.

> **AI-powered airline policy query system with intelligent document retrieval and natural language responses**

A production-ready RAG (Retrieval Augmented Generation) system that automatically scrapes, processes, and enables intelligent querying of airline baggage policies through natural language interactions.

## 🎯 Problem & Solution

**Problem:** Airline policy information is scattered across multiple websites, difficult to navigate, and constantly changing. Customers struggle to find accurate, up-to-date policy information quickly.

**Solution:** Automated multi-airline policy scraping with AI-powered semantic search and natural language question answering through OpenAI integration.

## 🚀 Key Features

- **🤖 AI-Powered Query System**: Natural language questions with contextual answers
- **✈️ Multi-Airline Support**: Turkish Airlines & Pegasus Airlines (extensible)
- **🔍 Semantic Search**: Vector-based similarity search with embeddings
- **📡 Automated Scraping**: Real-time policy data extraction and updates
- **🌐 Multi-Language**: Turkish & English policy support
- **📊 Interactive Dashboard**: User-friendly Streamlit interface
- **🔄 Real-time Updates**: Automatic policy synchronization
- **📈 Analytics**: Query patterns and system performance monitoring

## 🛠️ Tech Stack

### **Backend**
- **FastAPI**: High-performance async API framework
- **PostgreSQL + pgvector**: Vector database for embeddings
- **OpenAI GPT**: Language model for response generation
- **Sentence Transformers**: Multilingual embedding generation
- **BeautifulSoup**: Intelligent web scraping
- **Pydantic**: Data validation and settings management

### **Frontend**
- **Streamlit**: Interactive web application
- **Plotly**: Data visualization and analytics

### **Infrastructure**
- **Docker Compose**: Containerized deployment
- **AsyncPG**: High-performance database connections
- **Multi-container Architecture**: Scalable microservices design

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │────│    FastAPI       │────│   PostgreSQL    │
│   Frontend      │    │    Backend       │    │   + pgvector    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                           │
                              │                           │
                    ┌──────────────────┐         ┌─────────────────┐
                    │   Web Scraper    │         │   Embeddings    │
                    │   Multi-Airline  │────────│   Vector Store   │
                    └──────────────────┘         └─────────────────┘
                              │
                    ┌──────────────────┐
                    │   OpenAI GPT     │
                    │   RAG Engine     │
                    └──────────────────┘
```

### **Data Flow**
1. **Scraper** → Extracts policies from airline websites
2. **Processor** → Generates embeddings and stores in vector database
3. **API** → Handles user queries with semantic search
4. **RAG Engine** → Retrieves relevant docs and generates responses
5. **Frontend** → Displays results in user-friendly interface

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- 8GB+ RAM recommended
- OpenAI API key (optional, for enhanced responses)

### 1. Clone Repository
```bash
git clone <repository-url>
cd multi-airline-rag-system
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional)
nano .env
```

### 3. Start System
```bash
# Build and start all services
docker-compose up -d

# Wait for services to initialize (2-3 minutes)
docker-compose logs -f
```

### 4. Initialize Data
```bash
# Run data scraping (required for first setup)
docker-compose run scraper python scraper_only.py

# Verify data collection
curl http://localhost:8000/stats
```

### 5. Access Applications

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:8501 | Main user interface |
| **API** | http://localhost:8000 | REST API & documentation |
| **API Docs** | http://localhost:8000/docs | Interactive API documentation |

## 📖 Usage Examples

### Web Interface
1. Open http://localhost:8501
2. Select airline focus (All/Turkish Airlines/Pegasus)
3. Ask questions in natural language:
   - *"What are the baggage weight limits for international flights?"*
   - *"Can I travel with my pet in the cabin?"*
   - *"Compare excess baggage fees between airlines"*

### API Usage
```bash
# Health check
curl http://localhost:8000/health

# Search policies
curl "http://localhost:8000/search?q=baggage%20weight%20limit"

# Vector similarity search
curl "http://localhost:8000/vector/similarity-search?q=pet%20travel"

# AI-powered chat
curl -X POST "http://localhost:8000/chat/openai?question=What%20items%20are%20prohibited?"
```

## ⚙️ Configuration

### Environment Variables
```bash
# Database
DB_HOST=db
DB_DATABASE=global_gate
DB_USER=postgres
DB_PASSWORD=qeqe

# OpenAI (optional)
OPENAI_API_KEY=your_openai_api_key

# Scraping configuration
SCRAPE_AIRLINES=all  # all|thy_only|pegasus_only
```

### Supported Airlines
- **Turkish Airlines**: Baggage, pets, sports equipment, restrictions
- **Pegasus Airlines**: General rules, pricing, pet travel
- **Extensible**: Add new airlines via `airline_configs.py`

## 🔧 Troubleshooting

### Common Issues

#### ❌ "Database has 0 policies"
**Cause**: Scraper didn't run during initial startup (timing issue)


# Verify data
curl http://localhost:8000/stats
```

#### ❌ "Vector operations timeout"
**Cause**: Embeddings not generated yet

**Solution**:
```bash
# Generate embeddings
curl -X POST http://localhost:8000/vector/embed-policies

# Check status
curl http://localhost:8000/vector/stats
```

#### ❌ "API connection failed"
**Cause**: Services still starting up

**Solution**:
```bash
# Check service status
docker-compose ps

# Restart if needed
docker-compose restart api

# Wait and retry (services need 2-3 minutes)
```

### System Health Check
```bash
# Complete system test
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "database": "connected",
  "rag_features": {
    "vector_operations": "available",
    "embedding_service": "available",
    "openai_service": "available"
  }
}
```

## 📊 Performance & Monitoring

### System Metrics
- **Response Time**: <2s for typical queries
- **Embedding Generation**: ~32 docs/batch
- **Vector Search**: <500ms similarity queries
- **Concurrent Users**: 10+ supported

### Monitoring Endpoints
```bash
# System statistics
curl http://localhost:8000/stats

# Vector/embedding status  
curl http://localhost:8000/vector/stats

# Available data sources
curl http://localhost:8000/sources
```

## 🔄 Development & Extensibility

### Adding New Airlines
1. Configure in `airline_configs.py`
2. Define scraping strategies
3. Update database schema if needed
4. Test with `scrape_specific_airline('new_airline')`

### Custom Scrapers
```python
# Example airline configuration
NEW_AIRLINE_CONFIG = {
    'airline_id': 'new_airline',
    'airline_name': 'New Airline',
    'base_url': 'https://www.newairline.com',
    'pages': {
        'baggage': {
            'url': 'https://www.newairline.com/baggage',
            'parsing_strategy': 'custom_strategy'
        }
    }
}
```

## 📝 API Documentation

Full API documentation available at: http://localhost:8000/docs

### Key Endpoints
- `GET /health` - System health check
- `GET /search` - Text-based policy search  
- `GET /vector/similarity-search` - Semantic vector search
- `POST /chat/openai` - AI-powered responses
- `GET /stats` - System statistics
- `GET /sources` - Available data sources

## 📄 License

MIT License - See LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

**Built with ❤️ for travelers worldwide**