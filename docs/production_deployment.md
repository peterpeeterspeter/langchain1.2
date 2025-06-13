# Production Deployment Guide

This guide covers deploying the Enhanced Confidence Scoring System in production environments.

## Prerequisites

### System Requirements
- **Python**: 3.8+
- **Memory**: 8GB RAM minimum (16GB+ recommended)
- **CPU**: 4 cores minimum (8 cores recommended)
- **Storage**: 50GB SSD minimum

### Dependencies
```bash
pip install langchain langchain-core pydantic
pip install openai anthropic redis
```

## Configuration

### Production Config
```python
# config/production.py
PRODUCTION_CONFIG = {
    "model_name": "gpt-4",
    "enable_enhanced_confidence": True,
    "enable_caching": True,
    "confidence_config": {
        'quality_threshold': 0.75,
        'cache_strategy': 'ADAPTIVE',
        'max_regeneration_attempts': 1,
    },
    "cache": {
        "strategy": "ADAPTIVE",
        "max_size": 10000,
        "redis_url": "redis://localhost:6379"
    }
}
```

### Environment Variables
```bash
# .env
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
REDIS_URL=redis://localhost:6379
APP_ENV=production
```

## Deployment Options

### Docker Deployment
```yaml
# docker-compose.yml
version: '3.8'
services:
  rag-app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - APP_ENV=production
    depends_on:
      - redis
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - rag-app
```

### Kubernetes Deployment
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: enhanced-rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: enhanced-rag-system
  template:
    metadata:
      labels:
        app: enhanced-rag-system
    spec:
      containers:
      - name: rag-app
        image: enhanced-rag-system:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## Security

### API Security
```python
# Rate limiting
from fastapi import FastAPI
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response
```

### NGINX Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://rag-app:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

### Health Checks
```python
# health.py
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/ready")
async def readiness_check():
    # Check dependencies
    return {"status": "ready"}
```

### Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter('requests_total', 'Total requests')
CONFIDENCE_SCORE = Histogram('confidence_score', 'Confidence scores')
```

## Scaling

### Auto-scaling
```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: enhanced-rag-system
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

## Performance Optimization

### Caching Strategy
```python
# Use Redis for production caching
cache_config = {
    'strategy': 'ADAPTIVE',
    'redis_url': 'redis://redis-cluster:6379',
    'max_size': 10000,
    'default_ttl': 3600
}
```

### Connection Pooling
```python
# Database connections
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30
)
```

## Backup and Recovery

### Backup Script
```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup Redis
redis-cli --rdb $BACKUP_DIR/dump.rdb

# Backup config
cp -r /app/config $BACKUP_DIR/
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   ```bash
   # Check memory usage
   docker stats
   
   # Adjust limits
   docker-compose up -d --memory=4g
   ```

2. **Slow Response Times**
   ```python
   # Enable caching
   chain = create_universal_rag_chain(
       enable_caching=True,
       cache_strategy='AGGRESSIVE'
   )
   ```

3. **Cache Issues**
   ```bash
   # Check Redis status
   redis-cli ping
   
   # Clear cache if needed
   redis-cli flushall
   ```

## Maintenance

### Regular Tasks
```python
# Daily maintenance
async def daily_maintenance():
    # Clean old cache entries
    await clean_expired_cache()
    
    # Update performance metrics
    await update_metrics()
    
    # Health check
    await system_health_check()
```

### Rolling Updates
```bash
# Zero-downtime deployment
docker-compose up -d --scale rag-app=6
sleep 30
docker-compose up -d --scale rag-app=3
```

## Best Practices

1. **Use ADAPTIVE cache strategy** for optimal performance
2. **Set quality threshold to 0.75+** for production
3. **Enable monitoring and alerting**
4. **Regular backups and health checks**
5. **Load testing before deployment**

For more information, see:
- [System Overview](./enhanced_confidence_scoring_system.md)
- [API Reference](./api_reference.md)
- [Quick Start Guide](./quick_start.md) 