# Universal RAG CMS Configuration Management API

A comprehensive REST API for managing configurations, monitoring, A/B testing, and contextual retrieval in the Universal RAG CMS system.

## Features

### 🔧 Configuration Management
- **Prompt Optimization**: Manage prompt configurations with validation and versioning
- **Configuration History**: Track changes and rollback to previous versions
- **Validation**: Validate configurations before deployment

### 🎯 Contextual Retrieval Configuration
- **Retrieval Settings**: Manage contextual retrieval system configuration
- **Performance Profiles**: Apply optimized settings for different use cases
- **Query-Type Optimization**: Optimize configuration for specific query types
- **Environment Management**: Separate configurations for dev/staging/production

### 📊 Real-Time Monitoring & Analytics
- **Real-Time Metrics**: Live performance monitoring via REST and WebSocket
- **Alert Management**: Create, acknowledge, and manage system alerts
- **Performance Reports**: Generate comprehensive performance analytics

### 🎛️ Performance Profiling
- **System Snapshots**: Capture current system performance state
- **Optimization Reports**: Get recommendations for system improvements
- **Resource Monitoring**: Track CPU, memory, and response times

### 🚀 Feature Flags & A/B Testing
- **Feature Flags**: Gradual rollout and canary deployments
- **A/B Experiments**: Statistical analysis of feature variants
- **User Segmentation**: Hash-based and random segmentation strategies

## Quick Start

### Installation

```bash
pip install -r src/api/requirements.txt
```

### Environment Setup

Set up your environment variables:

```bash
export SUPABASE_URL="your_supabase_url"
export SUPABASE_KEY="your_supabase_key"
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
```

### Running the API

```bash
# Development mode
python -m src.api.main

# Or with uvicorn directly
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Access Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

### Configuration Management

#### Get Current Configuration
```http
GET /api/v1/config/prompt-optimization
```

#### Update Configuration
```http
PUT /api/v1/config/prompt-optimization?updated_by=user123
Content-Type: application/json

{
  "config_data": {
    "temperature": 0.7,
    "max_tokens": 1024,
    "system_prompt": "You are a helpful assistant"
  },
  "change_notes": "Updated temperature for better creativity"
}
```

#### Validate Configuration
```http
POST /api/v1/config/prompt-optimization/validate
Content-Type: application/json

{
  "config_data": {
    "temperature": 0.7,
    "max_tokens": 1024
  }
}
```

#### Get Configuration History
```http
GET /api/v1/config/prompt-optimization/history?limit=10
```

#### Rollback Configuration
```http
POST /api/v1/config/prompt-optimization/rollback/{version_id}?updated_by=user123
```

### Contextual Retrieval Configuration

#### Get Retrieval Configuration
```http
GET /retrieval/api/v1/config/?environment=production&include_sensitive=false
```

#### Update Retrieval Configuration
```http
PUT /retrieval/api/v1/config/
Content-Type: application/json

{
  "config_data": {
    "retrieval_strategy": "hybrid",
    "dense_weight": 0.7,
    "sparse_weight": 0.3,
    "mmr_lambda": 0.7,
    "max_results": 10
  },
  "validate_only": false,
  "force_update": false
}
```

#### Validate Retrieval Configuration
```http
POST /retrieval/api/v1/config/validate
Content-Type: application/json

{
  "retrieval_strategy": "contextual",
  "context_window_size": 512,
  "embedding_model": "text-embedding-3-large"
}
```

#### Export Configuration
```http
POST /retrieval/api/v1/config/export
Content-Type: application/json

{
  "environment": "production",
  "include_sensitive": false,
  "format": "json"
}
```

#### Update Performance Profile
```http
PUT /retrieval/api/v1/config/performance-profile
Content-Type: application/json

{
  "profile": "LATENCY_FOCUSED",
  "custom_overrides": {
    "cache_ttl_hours": 1,
    "max_concurrent_requests": 50
  }
}
```

#### Get Performance Profiles
```http
GET /retrieval/api/v1/config/performance-profiles
```

#### Optimize for Query Type
```http
POST /retrieval/api/v1/config/optimize-for-query-type
Content-Type: application/json

{
  "query_type": "casino_review",
  "sample_queries": [
    "What are the best online casinos for slots?",
    "Compare Betway vs 888 Casino bonuses"
  ]
}
```

#### Get Supported Query Types
```http
GET /retrieval/api/v1/config/query-types
```

#### Reload Configuration
```http
POST /retrieval/api/v1/config/reload
```

#### Configuration Health Check
```http
GET /retrieval/api/v1/config/health
```

### Monitoring & Analytics

#### Get Real-Time Metrics
```http
GET /api/v1/config/analytics/real-time?window_minutes=5
```

#### Get Active Alerts
```http
GET /api/v1/config/analytics/alerts
```

#### Acknowledge Alert
```http
POST /api/v1/config/analytics/alerts/acknowledge
Content-Type: application/json

{
  "alert_id": "alert_123",
  "acknowledged_by": "user123"
}
```

#### Generate Performance Report
```http
POST /api/v1/config/analytics/report?start_date=2024-01-01T00:00:00&end_date=2024-01-31T23:59:59
```

### Performance Profiling

#### Get Performance Snapshot
```http
GET /api/v1/config/profiling/snapshot
```

#### Get Optimization Report
```http
GET /api/v1/config/profiling/optimization-report?hours=24
```

### Feature Flags

#### List Feature Flags
```http
GET /api/v1/config/feature-flags
```

#### Get Specific Feature Flag
```http
GET /api/v1/config/feature-flags/{flag_name}
```

#### Create Feature Flag
```http
POST /api/v1/config/feature-flags
Content-Type: application/json

{
  "name": "enhanced_rag_search",
  "description": "Enhanced RAG search algorithm",
  "status": "gradual_rollout",
  "rollout_percentage": 25.0,
  "variants": [
    {
      "name": "control",
      "weight": 50.0,
      "config_overrides": {}
    },
    {
      "name": "enhanced",
      "weight": 50.0,
      "config_overrides": {
        "search_algorithm": "enhanced_v2"
      }
    }
  ]
}
```

#### Update Feature Flag
```http
PUT /api/v1/config/feature-flags/{flag_name}
Content-Type: application/json

{
  "rollout_percentage": 50.0,
  "status": "enabled"
}
```

### A/B Testing

#### Create Experiment
```http
POST /api/v1/config/experiments
Content-Type: application/json

{
  "feature_flag_name": "enhanced_rag_search",
  "experiment_name": "Enhanced Search Performance Test",
  "hypothesis": "Enhanced algorithm improves search relevance by 15%",
  "success_metrics": ["relevance_score", "user_satisfaction"],
  "duration_days": 14
}
```

#### Track Experiment Metric
```http
POST /api/v1/config/experiments/track-metric
Content-Type: application/json

{
  "experiment_id": "exp_123",
  "variant_name": "enhanced",
  "metric_type": "conversion",
  "metric_value": 1.0,
  "user_context": {
    "user_id": "user_456",
    "session_id": "session_789"
  }
}
```

#### Get Experiment Results
```http
GET /api/v1/config/experiments/{experiment_id}/results
```

### WebSocket Real-Time Monitoring

Connect to real-time metrics via WebSocket:

```javascript
const ws = new WebSocket('ws://localhost:8000/api/v1/config/ws/metrics');

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Real-time metrics:', data);
};
```

### Health Check

```http
GET /api/v1/config/health
```

## Response Format

All API responses follow a consistent format:

```json
{
  "status": "success|error",
  "data": {},
  "message": "Optional message",
  "timestamp": "2024-01-19T12:00:00Z"
}
```

## Error Handling

The API uses standard HTTP status codes:

- `200` - Success
- `400` - Bad Request (validation errors)
- `404` - Not Found
- `500` - Internal Server Error

Error responses include detailed information:

```json
{
  "status": "error",
  "detail": "Error description",
  "timestamp": "2024-01-19T12:00:00Z"
}
```

## Authentication

Currently, the API uses basic authentication via query parameters. For production, implement proper JWT or OAuth2 authentication.

## Rate Limiting

Consider implementing rate limiting for production deployments using middleware like `slowapi`.

## Security Considerations

1. **Environment Variables**: Never commit credentials to version control
2. **CORS**: Configure `allow_origins` appropriately for production
3. **HTTPS**: Use HTTPS in production environments
4. **Input Validation**: All input is validated using Pydantic models
5. **SQL Injection**: Supabase client handles parameterized queries

## Integration Examples

### Python Client Example

```python
import httpx
import asyncio

async def get_current_config():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8000/api/v1/config/prompt-optimization")
        return response.json()

# Run example
config = asyncio.run(get_current_config())
print(config)
```

### JavaScript Client Example

```javascript
// Get current configuration
fetch('/api/v1/config/prompt-optimization')
  .then(response => response.json())
  .then(data => console.log(data));

// Update configuration
fetch('/api/v1/config/prompt-optimization?updated_by=user123', {
  method: 'PUT',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    config_data: {
      temperature: 0.8,
      max_tokens: 2048
    },
    change_notes: 'Increased creativity'
  })
})
.then(response => response.json())
.then(data => console.log(data));
```

## Monitoring and Observability

The API includes built-in monitoring capabilities:

1. **Health Checks**: Basic and detailed health endpoints
2. **Logging**: Structured logging throughout the application
3. **Metrics**: Real-time performance metrics
4. **Error Tracking**: Global exception handling

## Development

### Project Structure

```
src/api/
├── __init__.py
├── main.py              # FastAPI application
├── config_management.py # Main API router
├── requirements.txt     # Dependencies
└── README.md           # This file
```

### Testing

Create tests for your API endpoints:

```python
import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/config/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
```

### Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Write comprehensive docstrings
4. Include unit tests for new endpoints
5. Update this documentation for API changes

## License

This project is part of the Universal RAG CMS system. See the main project LICENSE for details. 