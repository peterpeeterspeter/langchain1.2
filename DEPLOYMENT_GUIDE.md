# Universal RAG CMS - Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Universal RAG CMS system to production environments. The system includes advanced RAG capabilities, contextual retrieval, real-time monitoring, and enterprise-grade configuration management.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   API Gateway   │    │   Monitoring    │
│   (nginx/ALB)   │────│   (FastAPI)     │────│   (Prometheus)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   RAG Engine    │
                       │   (Contextual)  │
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │   Supabase      │
                       │   (PostgreSQL)  │
                       └─────────────────┘
```

## Prerequisites

### System Requirements

**Minimum Requirements:**
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: 1Gbps

**Recommended for Production:**
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- Network: 10Gbps

### Software Dependencies

- Python 3.9+
- Docker & Docker Compose
- nginx (for load balancing)
- PostgreSQL 14+ (via Supabase)

### API Keys Required

- **Supabase**: Project URL and Service Role Key
- **OpenAI**: API Key (for embeddings and LLM)
- **Anthropic**: API Key (optional, for Claude models)

## Environment Setup

### 1. Environment Variables

Create a `.env` file with the following configuration:

```bash
# Environment
RAG_CMS_ENVIRONMENT=production
RAG_CMS_DEBUG=false

# API Configuration
RAG_CMS_API_HOST=0.0.0.0
RAG_CMS_API_PORT=8000
RAG_CMS_API_WORKERS=4
RAG_CMS_SECRET_KEY=your-super-secret-production-key

# Database (Supabase)
RAG_CMS_SUPABASE_URL=https://your-project.supabase.co
RAG_CMS_SUPABASE_KEY=your-service-role-key

# AI Models
RAG_CMS_OPENAI_API_KEY=sk-your-openai-key
RAG_CMS_ANTHROPIC_API_KEY=sk-ant-your-anthropic-key

# Performance
RAG_CMS_MAX_CONCURRENT_REQUESTS=100
RAG_CMS_REQUEST_RATE_LIMIT=1000
RAG_CMS_CACHE_TTL_HOURS=24
RAG_CMS_DATABASE_POOL_SIZE=20

# Security
RAG_CMS_ALLOWED_HOSTS=["your-domain.com"]
RAG_CMS_CORS_ORIGINS=["https://your-frontend.com"]

# Monitoring
RAG_CMS_ENABLE_METRICS=true
RAG_CMS_METRICS_PORT=9090
RAG_CMS_LOG_LEVEL=INFO

# Features
RAG_CMS_ENABLE_CONTEXTUAL_RETRIEVAL=true
RAG_CMS_ENABLE_MULTI_QUERY=true
RAG_CMS_ENABLE_SELF_QUERY=true
RAG_CMS_ENABLE_PERFORMANCE_OPTIMIZATION=true
```

### 2. Supabase Setup

#### Database Migrations

Run the database migrations to set up the required schema:

```bash
# Apply Task 3.6 migrations
python -m src.database.migrations.apply_migrations
```

#### Required Tables

The system requires these tables:
- `documents` - Document storage
- `contextual_chunks` - Contextual embeddings
- `hybrid_search_config` - Search configuration
- `contextual_cache` - Intelligent caching
- `retrieval_metrics` - Performance metrics
- `query_variations` - Multi-query storage

#### RPC Functions

Ensure these RPC functions are deployed:
- `hybrid_search_documents()`
- `contextual_search_with_mmr()`
- `get_retrieval_analytics()`
- `optimize_retrieval_parameters()`
- `cleanup_expired_cache()`

### 3. Docker Deployment

#### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY .env .

# Create non-root user
RUN useradd -m -u 1000 raguser && chown -R raguser:raguser /app
USER raguser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Docker Compose

```yaml
version: '3.8'

services:
  rag-cms-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - RAG_CMS_ENVIRONMENT=production
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./temp:/tmp/rag_cms
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - rag-cms-api
    restart: unless-stopped

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

volumes:
  prometheus_data:
```

### 4. nginx Configuration

```nginx
upstream rag_cms_backend {
    server rag-cms-api:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    # Proxy settings
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    # API endpoints
    location /api/ {
        proxy_pass http://rag_cms_backend;
        proxy_timeout 300s;
        proxy_read_timeout 300s;
    }

    # WebSocket support
    location /api/v1/config/ws/ {
        proxy_pass http://rag_cms_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }

    # Health check
    location /health {
        proxy_pass http://rag_cms_backend;
        access_log off;
    }

    # Static files (if any)
    location /static/ {
        alias /app/static/;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## Deployment Steps

### 1. Prepare Environment

```bash
# Clone repository
git clone https://github.com/peterpeeterspeter/langchain1.2.git
cd langchain1.2

# Create environment file
cp .env.example .env
# Edit .env with your configuration

# Create required directories
mkdir -p logs temp ssl
```

### 2. Build and Deploy

```bash
# Build Docker image
docker-compose build

# Start services
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs -f rag-cms-api
```

### 3. Verify Deployment

```bash
# Health check
curl https://your-domain.com/health

# API documentation
curl https://your-domain.com/docs

# Test contextual query
curl -X POST https://your-domain.com/api/v1/contextual/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "max_results": 5}'
```

### 4. Database Initialization

```bash
# Run migrations
docker-compose exec rag-cms-api python -m src.database.migrations.apply_migrations

# Verify tables
docker-compose exec rag-cms-api python -c "
from src.api.contextual_retrieval_api import get_supabase_client
import asyncio
async def check():
    client = await get_supabase_client()
    result = client.table('contextual_chunks').select('*').limit(1).execute()
    print('Database connection successful')
asyncio.run(check())
"
```

## Monitoring and Observability

### 1. Prometheus Metrics

Configure Prometheus to scrape metrics:

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'rag-cms'
    static_configs:
      - targets: ['rag-cms-api:9090']
    scrape_interval: 5s
    metrics_path: /metrics
```

### 2. Key Metrics to Monitor

- **API Performance**: Response times, error rates, throughput
- **Retrieval Performance**: Query latency, cache hit rates, confidence scores
- **System Resources**: CPU, memory, disk usage
- **Database**: Connection pool, query performance
- **AI Models**: Token usage, API latency

### 3. Alerting Rules

```yaml
# alerts.yml
groups:
  - name: rag-cms
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: High error rate detected

      - alert: SlowQueries
        expr: histogram_quantile(0.95, rate(query_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: Slow query performance detected
```

### 4. Log Management

Configure structured logging:

```python
# In production settings
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "json",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "/app/logs/rag_cms.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "formatter": "json",
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}
```

## Performance Optimization

### 1. Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_contextual_chunks_embedding 
ON contextual_chunks USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

CREATE INDEX CONCURRENTLY idx_contextual_chunks_metadata 
ON contextual_chunks USING gin (metadata);

CREATE INDEX CONCURRENTLY idx_retrieval_metrics_timestamp 
ON retrieval_metrics (created_at);

-- Analyze tables
ANALYZE contextual_chunks;
ANALYZE retrieval_metrics;
```

### 2. Caching Strategy

- **Redis**: For session and query caching
- **Application Cache**: For configuration and metadata
- **CDN**: For static assets and API responses

### 3. Connection Pooling

```python
# Database connection pool settings
DATABASE_CONFIG = {
    "pool_size": 20,
    "max_overflow": 30,
    "pool_timeout": 30,
    "pool_recycle": 3600,
    "pool_pre_ping": True,
}
```

## Security Considerations

### 1. API Security

- **Rate Limiting**: Implement per-IP and per-user limits
- **Authentication**: Use JWT tokens or API keys
- **Input Validation**: Validate all inputs with Pydantic
- **CORS**: Configure appropriate origins

### 2. Data Security

- **Encryption**: Encrypt sensitive data at rest
- **Access Control**: Implement RLS policies in Supabase
- **Audit Logging**: Log all data access and modifications
- **Backup**: Regular encrypted backups

### 3. Infrastructure Security

- **TLS**: Use TLS 1.3 for all communications
- **Firewall**: Restrict access to necessary ports
- **Updates**: Regular security updates
- **Secrets**: Use environment variables or secret managers

## Scaling Considerations

### 1. Horizontal Scaling

```yaml
# docker-compose.scale.yml
services:
  rag-cms-api:
    deploy:
      replicas: 4
      update_config:
        parallelism: 2
        delay: 10s
      restart_policy:
        condition: on-failure
```

### 2. Load Balancing

- **nginx**: For HTTP load balancing
- **HAProxy**: For advanced load balancing
- **Cloud Load Balancers**: AWS ALB, GCP Load Balancer

### 3. Database Scaling

- **Read Replicas**: For read-heavy workloads
- **Connection Pooling**: PgBouncer for connection management
- **Partitioning**: For large tables

## Backup and Recovery

### 1. Database Backup

```bash
# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Supabase backup (via API)
curl -X POST "https://api.supabase.com/v1/projects/${PROJECT_ID}/database/backups" \
  -H "Authorization: Bearer ${SUPABASE_ACCESS_TOKEN}"

# Local backup
pg_dump "${DATABASE_URL}" > "${BACKUP_DIR}/backup_${DATE}.sql"

# Compress and encrypt
gzip "${BACKUP_DIR}/backup_${DATE}.sql"
gpg --encrypt --recipient backup@yourcompany.com "${BACKUP_DIR}/backup_${DATE}.sql.gz"
```

### 2. Application Backup

```bash
# Backup configuration and logs
tar -czf "app_backup_$(date +%Y%m%d).tar.gz" \
  .env \
  logs/ \
  temp/ \
  docker-compose.yml
```

### 3. Recovery Procedures

1. **Database Recovery**: Restore from Supabase backup or SQL dump
2. **Application Recovery**: Redeploy from Git repository
3. **Configuration Recovery**: Restore from backup
4. **Verification**: Run health checks and integration tests

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Check embedding cache size
   - Monitor vector operations
   - Adjust batch sizes

2. **Slow Queries**
   - Check database indexes
   - Analyze query plans
   - Optimize retrieval parameters

3. **API Timeouts**
   - Increase timeout settings
   - Check AI model latency
   - Optimize concurrent requests

### Debug Commands

```bash
# Check system resources
docker stats

# View logs
docker-compose logs -f rag-cms-api

# Database connections
docker-compose exec rag-cms-api python -c "
from src.api.contextual_retrieval_api import get_supabase_client
import asyncio
asyncio.run(get_supabase_client())
"

# Test API endpoints
curl -X GET https://your-domain.com/health
curl -X GET https://your-domain.com/api/v1/contextual/health
```

## Maintenance

### Regular Tasks

1. **Daily**
   - Monitor system metrics
   - Check error logs
   - Verify backup completion

2. **Weekly**
   - Review performance metrics
   - Update dependencies
   - Clean up old logs

3. **Monthly**
   - Security updates
   - Performance optimization
   - Capacity planning

### Update Procedures

```bash
# Update application
git pull origin main
docker-compose build
docker-compose up -d --no-deps rag-cms-api

# Update dependencies
pip-compile requirements.in
docker-compose build --no-cache

# Database migrations
docker-compose exec rag-cms-api python -m src.database.migrations.apply_migrations
```

## Support and Documentation

- **API Documentation**: https://your-domain.com/docs
- **Monitoring Dashboard**: https://your-domain.com:9090
- **Health Status**: https://your-domain.com/health
- **GitHub Repository**: https://github.com/peterpeeterspeter/langchain1.2

For additional support, refer to the comprehensive documentation in the `docs/` directory or contact the development team. 