# AI-Powered Multi-Airline Policy Assistant

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL%20+%20pgvector-336791?logo=postgresql)
![React](https://img.shields.io/badge/React-61DAFB?logo=react)
![Docker](https://img.shields.io/badge/Docker-2496ED?logo=docker)
![Coverage](https://img.shields.io/badge/Test_Coverage-90%25-success)

## Problem

Airline policy information is scattered, inconsistent, and hard to access when you actually need it. [Phone support averages 2-12 hours](https://www.mightytravels.com/2024/10/how-major-airlines-customer-service-response-times-compare-analysis-of-7-leading-carriers-in-2024/). [80% of travelers](https://www.cxtoday.com/speech-analytics/chatbots-are-still-frustrating-customers-here-is-why/) say standard chatbots can't answer simple policy questions. And when you're rushing between gates or driving to the airport, typing through FAQ pages isn't practical.

## Solution

A voice-enabled assistant that understands natural language and compares policies across multiple airlines. Semantic search interprets intent—not just keywords. Response time under 3 seconds. Voice support in Turkish and English for hands-free use.

| Interface | Result |
|:---------:|:------:|
| ![Main Interface](assets/interface1.png) | ![Response](assets/interface2.png) |
| ![Backend](assets/interface4.png) | ![Voice Input](assets/interface3.png) |

## System Architecture

Speech services run as separate streams—TTS failure doesn't block query processing. Dual-LLM setup enables quality comparison and prevents vendor lock-in.

```mermaid
graph TD
    A[User] <--> B[Frontend React]
    B --> C[FastAPI Backend]
    
    B --> I[Audio Input]
    I --> J[AssemblyAI STT]
    J --> C
    
    C <--> D[PostgreSQL + pgvector]
    E[Web Scraper] --> D
    C <--> F[OpenAI/Claude APIs]
    
    C --> K[AWS Polly TTS]
    K --> L[Audio Output]
    L --> B
    
    G[Prometheus] --> H[Grafana]
    C --> G
    
    classDef userPath fill:#e1f5fe
    classDef audioPath fill:#f3e5f5
    classDef backend fill:#e8f5e8
    classDef monitoring fill:#fff3e0
    
    class A,B userPath
    class I,J,K,L audioPath
    class C,D,E,F backend
    class G,H monitoring
```

## Technology Decisions

| Technology | Purpose | Trade-off Reasoning |
|------------|---------|---------------------|
| **pgvector** | Vector Search | Reduced overhead vs Pinecone/Weaviate. Sufficient for policy-scale datasets. |
| **gte-multilingual-base** | Embeddings | Open-source, no per-query cost. Native Turkish/English support. |
| **FastAPI** | Backend | Async handles concurrent LLM + STT + TTS calls. |
| **BeautifulSoup** | Scraping | Sufficient for general-purpose scraping needs. |
| **OpenAI + Claude** | LLM | Dual-provider prevents lock-in. Enables quality comparison. |
| **AssemblyAI** | STT | Strong Turkish accuracy. Handles background noise well. |

## Core Capabilities

**RAG Pipeline**: Semantic search with source attribution. Reduces hallucination risk.

**Voice Interface**: Real-time STT/TTS in Turkish and English.

**Production Monitoring**: Prometheus + Grafana for latency, errors, and API costs.

**Multi-LLM Support**: Switch providers without code changes.

**Intelligent Caching**: Multi-layer LRU cache for repeated queries.

## Quick Start

```bash
git clone <repository-url>
cd multi-airline-rag-system
cp .env.example .env

docker-compose up -d
docker-compose run scraper python scraper_only.py

curl http://localhost:8000/health
```

| Service | URL |
|---------|-----|
| Frontend | http://localhost:8501 |
| API | http://localhost:8000 |
| Grafana | http://localhost:3000 |
