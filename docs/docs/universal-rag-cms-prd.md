# Universal RAG CMS System: Product Requirements Document

## Executive Summary

This Product Requirements Document outlines the architecture and implementation strategy for building a universal RAG (Retrieval-Augmented Generation) CMS system from scratch using modern LangChain best practices and **Supabase as the core infrastructure**. The system addresses the critical need to replace a problematic 3,826-line monolithic architecture that suffers from competing AI layers and performance degradation. The new architecture emphasizes clean, maintainable code through the **FTI (Feature/Training/Inference) pipeline pattern**, comprehensive API integrations, and modular design principles that prevent the "enhancement trap."

The system leverages Supabase's PostgreSQL with pgvector extension for unified data and vector storage, Supabase Auth for authentication, Supabase Storage for media files, and Edge Functions for serverless compute. It will handle diverse content types beyond casino reviews, integrate DataForSEO image search capabilities, generate authoritative hyperlinks, publish to WordPress via REST API, and extract metadata automatically. Performance optimization strategies focus on **contextual retrieval** (achieving 49% failure rate reduction), multi-level caching, and async processing patterns that enable sub-second response times at scale.

## Quick Start Guide

### Prerequisites

- Supabase account (Pro/Team plan recommended for production)
- Python 3.11+
- Node.js 18+ (for Edge Functions)
- Docker & Docker Compose
- Supabase CLI
- Poetry for Python dependency management

### Initial Setup (5 minutes)

```bash
# Clone the repository
git clone https://github.com/your-org/universal-rag-cms.git
cd universal-rag-cms

# Install Supabase CLI
npm install -g supabase

# Initialize Supabase project
supabase init

# Link to your Supabase project
supabase link --project-ref your-project-ref

# Push database schema
supabase db push

# Install Python dependencies
poetry install

# Copy environment variables
cp .env.example .env
# Edit .env with your Supabase URL, keys, and API keys

# Start local development
supabase start
poetry run python -m src.api
```

### First RAG Query (2 minutes)

```python
# Quick test script
from src.integrations.supabase import SupabaseRAGStore
from src.chains.rag_chain import ModularRAGChain

# Initialize
rag_store = SupabaseRAGStore(SUPABASE_URL, SUPABASE_KEY)
rag_chain = ModularRAGChain(rag_store)

# Store content
await rag_chain.ingest_content(
    "Your first content",
    content_type="article"
)

# Query
response = await rag_chain.query("What is in my content?")
print(response.content)
```

## Solving Your Monolithic System Problems

### Direct Solutions to Your Current Issues

**Problem: 7+ Competing AI Layers**

```python
# OLD: Multiple AI enhancement layers
content = generate_adaptive_article()
content = expand_content_with_comprehensive_research(content)
content = ensure_affiliate_compliance(content)
content = enhance_content_with_native_multimodal(content)
content = enhance_content_with_eeat_signals(content)
# ... and more

# NEW: Single, clean LCEL chain
rag_chain = (
    {"context": retriever, "query": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)
result = await rag_chain.ainvoke(query)
```

**Problem: 3,826 Lines of Unmaintainable Code**

- **Solution**: ~500 lines of clean, modular code
- Each component is independent and testable
- Clear separation of concerns with FTI architecture
- No more "enhancement trap" - single pass generation

**Problem: Disabled Validation Due to Performance**

- **Solution**: Validation built into the chain using structured outputs
- No post-processing needed - get it right the first time
- Supabase RLS for data validation at the database level

**Problem: 47.5% Content Quality Score**

- **Solution**: Quality through simplicity
- Single LLM call with comprehensive context
- Better prompts instead of multiple AI layers
- Structured output validation ensures consistency

### Core Architectural Principles

**FTI Pipeline Architecture**: The foundation follows the Feature/Training/Inference separation pattern, decomposing the RAG system into three independent, scalable components connected through shared storage layers:

- **Feature Pipeline**: Processes raw content into embeddings and structured metadata, storing results in vector databases and feature stores
- **Training Pipeline**: Handles model fine-tuning, prompt optimization, and performance evaluation using stored features
- **Inference Pipeline**: Generates responses using trained models and retrieved features through LangChain LCEL chains

**Microservices Design**: Each major component operates as an independent service with well-defined APIs:

- RAG Business Layer Service
- Content Processing Service
- API Integration Service
- Publishing Service
- Monitoring and Analytics Service

### Technology Stack

**Core Framework**: LangChain with LCEL (LangChain Expression Language) for declarative chain composition
**Orchestration**: LangGraph for complex multi-agent workflows and state management
**Database & Vector Store**: Supabase (PostgreSQL with pgvector extension) for unified data and vector storage
**Storage**: Supabase Storage for media files and documents
**Authentication**: Supabase Auth for secure access control
**API Framework**: FastAPI with Supabase client integration
**Edge Functions**: Supabase Edge Functions for serverless compute
**Caching**: Supabase with Redis for hot data caching
**Development Environment**: Cursor IDE with AI-assisted development workflows
**Deployment**: Supabase Cloud with Docker containers for additional services

## Functional Requirements

### Content Management Capabilities

**Universal Content Type Support**: The system must process and generate content for diverse domains including but not limited to:

- News articles and editorial content
- Product reviews and comparisons
- Technical documentation
- Marketing copy and landing pages
- Educational materials and tutorials

**Content Processing Pipeline**:

- Automatic content type detection and routing
- Metadata-rich indexing with structured data extraction
- Progressive enhancement based on content characteristics
- Adaptive chunking strategies optimized for different content types

### API Integration Requirements

**DataForSEO Integration**:

- Image search capabilities with metadata extraction (alt text, dimensions, source attribution)
- Rate limiting compliance (2,000 requests/minute, max 30 simultaneous)
- Batch processing support (up to 100 tasks per request)
- Exponential backoff retry mechanisms for failed requests
- Cost optimization through intelligent caching and request batching

**WordPress REST API Publishing**:

- Multi-authentication support (Application Passwords, JWT, OAuth 2.0)
- Automated content creation with rich media handling
- Custom field management and metadata synchronization
- Two-step media upload process with automatic attachment
- Bulk publishing capabilities with error recovery

**Authoritative Hyperlink Generation**:

- Contextual internal linking based on semantic similarity
- SEO-optimized anchor text generation
- Link quality scoring and validation
- Canonical URL management
- Authority-based link distribution algorithms

### RAG System Capabilities

**Advanced Retrieval Methods**:

- **Contextual Retrieval**: Prepend context to chunks before embedding (49% failure rate reduction)
- **Hybrid Search**: Combine dense embeddings with BM25 for lexical matching
- **Multi-Query Retrieval**: Generate multiple query variations for improved recall
- **Self-Query Retrieval**: Metadata filtering for precise content targeting
- **Maximal Marginal Relevance**: Diverse result selection to reduce redundancy

**Generation Pipeline**:

- Streaming response generation for improved user experience
- Context window optimization with intelligent truncation
- Response quality validation and hallucination detection
- Multi-model routing based on query complexity and cost optimization

## Supabase Integration Architecture

### Database Schema Design

**Core Tables Structure**:

```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Content items table
CREATE TABLE content_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    content_type TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    author_id UUID REFERENCES auth.users(id),
    status TEXT DEFAULT 'draft',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    published_at TIMESTAMPTZ,
    wordpress_post_id INTEGER,
    FULLTEXT content_search (title, content)
);

-- Vector embeddings table
CREATE TABLE content_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector(1536), -- OpenAI embeddings dimension
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(content_id, chunk_index)
);

-- Create vector similarity search index
CREATE INDEX ON content_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Media assets table
CREATE TABLE media_assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID REFERENCES content_items(id),
    storage_path TEXT NOT NULL,
    file_name TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    alt_text TEXT,
    caption TEXT,
    wordpress_media_id INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- RAG query cache table
CREATE TABLE rag_query_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash TEXT NOT NULL UNIQUE,
    query_text TEXT NOT NULL,
    query_embedding vector(1536),
    response TEXT NOT NULL,
    sources JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

-- Create RLS policies
ALTER TABLE content_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE media_assets ENABLE ROW LEVEL SECURITY;
```

## Performance and Monitoring Specifications

### Key Performance Indicators

**Response Time Targets**:

- Simple queries: < 500ms end-to-end
- Complex queries: < 2 seconds
- Batch processing: 100 queries/minute sustained throughput

**Quality Metrics**:

- Retrieval precision@5: > 0.8
- Response relevance score: > 0.85
- Hallucination detection: < 5% false positive rate
- User satisfaction rating: > 4.2/5.0

**Resource Utilization**:

- Memory usage: < 4GB per instance
- CPU utilization: < 70% average
- API cost optimization: < $0.02 per query

## Security and Compliance

### Security Requirements

**Data Protection**:

- Encryption at rest for all stored embeddings and content
- TLS 1.3 for all API communications
- API key rotation every 90 days
- Input sanitization for all user-generated content

**Access Control**:

- Role-based access control (RBAC) for administrative functions
- API rate limiting per user/organization
- Audit logging for all content modifications
- Multi-factor authentication for administrative access

**Content Safety**:

- Automated content moderation using OpenAI Moderation API
- Bias detection and mitigation in generated content
- Source attribution validation for copyright compliance
- GDPR compliance for personal data handling

## Success Metrics and KPIs

### Technical Success Criteria

**System Reliability**:

- 99.9% uptime SLA
- < 0.1% error rate across all endpoints
- Successful failover within 30 seconds
- Zero data loss incidents

**Performance Benchmarks**:

- 67% reduction in retrieval failures (baseline vs. contextual retrieval)
- 90% cache hit rate for frequently requested content
- Sub-second response times for 95% of queries
- 50% reduction in LLM API costs through optimization

### Business Success Metrics

**Content Quality**:

- Automated content generation accuracy > 90%
- Editorial review time reduced by 60%
- SEO performance improvement of 40% average ranking increase
- User engagement metrics showing 25% longer session duration

**Operational Efficiency**:

- Development velocity increased by 40% through clean architecture
- Time to implement new content types reduced from weeks to days
- Maintenance overhead reduced by 70% compared to monolithic system
- Zero critical security vulnerabilities in production

This comprehensive PRD provides the foundation for building a production-ready, scalable, and maintainable universal RAG CMS system that avoids the pitfalls of monolithic AI architectures while delivering exceptional performance and reliability. 