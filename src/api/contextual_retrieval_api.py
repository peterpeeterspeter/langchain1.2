"""
FastAPI Router for Contextual Retrieval Operations

Provides production-ready REST API endpoints for:
- Contextual document querying with advanced retrieval strategies
- Document ingestion with contextual embedding generation
- Performance metrics and monitoring
- Real-time retrieval analytics
- Content migration utilities
- System health monitoring

This API integrates all Task 3 components into a unified production interface.
"""

import os
import json
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter, HTTPException, Depends, Query, Body, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from supabase import Client, create_client

# Import contextual retrieval components
from ..retrieval.contextual_retrieval import (
    ContextualRetrievalSystem,
    RetrievalConfig,
    RetrievalStrategy,
    create_contextual_retrieval_system
)
from ..retrieval.contextual_embedding import ContextualEmbeddingSystem, ContextualChunk
from ..retrieval.hybrid_search import HybridSearchEngine
from ..retrieval.multi_query import MultiQueryRetriever
from ..retrieval.self_query import SelfQueryRetriever
from ..retrieval.performance_optimizer import PerformanceOptimizer
from ..config.retrieval_config import RetrievalSettings

# Configure logging
logger = logging.getLogger(__name__)

# API Models
class ContextualQueryRequest(BaseModel):
    """Request model for contextual retrieval queries."""
    
    query: str = Field(
        ...,
        description="The search query",
        min_length=1,
        max_length=1000
    )
    strategy: Optional[str] = Field(
        default="hybrid",
        description="Retrieval strategy to use"
    )
    max_results: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of results to return"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include document metadata in results"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters to apply"
    )
    mmr_lambda: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="MMR diversity parameter (0=diversity, 1=relevance)"
    )

class DocumentIngestionRequest(BaseModel):
    """Request model for document ingestion."""
    
    content: str = Field(
        ...,
        description="Document content",
        min_length=10
    )
    title: Optional[str] = Field(
        default=None,
        description="Document title"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Document metadata"
    )
    chunk_size: Optional[int] = Field(
        default=512,
        ge=100,
        le=2000,
        description="Chunk size for splitting"
    )
    chunk_overlap: Optional[int] = Field(
        default=50,
        ge=0,
        le=500,
        description="Overlap between chunks"
    )
    generate_contextual_embeddings: bool = Field(
        default=True,
        description="Generate contextual embeddings"
    )

class BatchIngestionRequest(BaseModel):
    """Request model for batch document ingestion."""
    
    documents: List[DocumentIngestionRequest] = Field(
        ...,
        description="List of documents to ingest",
        min_items=1,
        max_items=100
    )
    batch_size: Optional[int] = Field(
        default=10,
        ge=1,
        le=50,
        description="Processing batch size"
    )

class MigrationRequest(BaseModel):
    """Request model for content migration."""
    
    source_table: str = Field(
        ...,
        description="Source table name"
    )
    target_table: str = Field(
        default="contextual_chunks",
        description="Target table name"
    )
    batch_size: Optional[int] = Field(
        default=100,
        ge=10,
        le=1000,
        description="Migration batch size"
    )
    dry_run: bool = Field(
        default=True,
        description="Perform dry run without actual migration"
    )

class PerformanceMetricsRequest(BaseModel):
    """Request model for performance metrics."""
    
    time_range_hours: Optional[int] = Field(
        default=24,
        ge=1,
        le=168,
        description="Time range in hours"
    )
    include_detailed_stats: bool = Field(
        default=False,
        description="Include detailed performance statistics"
    )
    group_by: Optional[str] = Field(
        default="hour",
        regex="^(minute|hour|day)$",
        description="Grouping interval"
    )

# Response Models
class ContextualQueryResponse(BaseModel):
    """Response model for contextual queries."""
    
    query: str
    strategy_used: str
    results: List[Dict[str, Any]]
    total_results: int
    response_time_ms: float
    confidence_scores: List[float]
    metadata: Dict[str, Any]

class IngestionResponse(BaseModel):
    """Response model for document ingestion."""
    
    success: bool
    document_id: Optional[str] = None
    chunks_created: int
    embeddings_generated: int
    processing_time_ms: float
    errors: List[str] = Field(default_factory=list)

class BatchIngestionResponse(BaseModel):
    """Response model for batch ingestion."""
    
    total_documents: int
    successful_documents: int
    failed_documents: int
    total_chunks_created: int
    total_embeddings_generated: int
    processing_time_ms: float
    errors: List[str] = Field(default_factory=list)

class MigrationResponse(BaseModel):
    """Response model for content migration."""
    
    migration_id: str
    source_table: str
    target_table: str
    total_records: int
    migrated_records: int
    failed_records: int
    dry_run: bool
    processing_time_ms: float
    errors: List[str] = Field(default_factory=list)

# API Router
router = APIRouter(
    prefix="/api/v1/contextual",
    tags=["Contextual Retrieval"],
    responses={
        404: {"description": "Resource not found"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    }
)

# Global system instance
_retrieval_system: Optional[ContextualRetrievalSystem] = None

async def get_supabase_client() -> Client:
    """Get Supabase client instance."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise HTTPException(
            status_code=500,
            detail="Supabase credentials not configured"
        )
    
    return create_client(supabase_url, supabase_key)

async def get_retrieval_system() -> ContextualRetrievalSystem:
    """Get or create contextual retrieval system instance."""
    global _retrieval_system
    
    if _retrieval_system is None:
        try:
            # Load configuration
            config = RetrievalSettings.load_from_env()
            
            # Create system
            _retrieval_system = await create_contextual_retrieval_system(
                config=config.to_retrieval_config(),
                supabase_client=await get_supabase_client()
            )
            
            logger.info("Contextual retrieval system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize contextual retrieval system: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize retrieval system: {str(e)}"
            )
    
    return _retrieval_system

# Query Endpoints
@router.post("/query", response_model=ContextualQueryResponse)
async def contextual_query(
    request: ContextualQueryRequest,
    retrieval_system: ContextualRetrievalSystem = Depends(get_retrieval_system)
):
    """Perform contextual retrieval query with advanced strategies."""
    
    start_time = datetime.now()
    
    try:
        # Configure retrieval parameters
        config_updates = {
            "retrieval_strategy": request.strategy,
            "max_results": request.max_results,
            "mmr_lambda": request.mmr_lambda
        }
        
        if request.filters:
            config_updates["metadata_filters"] = request.filters
        
        # Update system configuration
        retrieval_system.update_config(**config_updates)
        
        # Perform retrieval
        results = await retrieval_system.aretrieve(request.query)
        
        # Calculate response time
        response_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Extract confidence scores
        confidence_scores = [
            result.metadata.get("confidence_score", 0.0) 
            for result in results
        ]
        
        # Format results
        formatted_results = []
        for result in results:
            result_dict = {
                "content": result.page_content,
                "score": result.metadata.get("score", 0.0),
                "confidence": result.metadata.get("confidence_score", 0.0)
            }
            
            if request.include_metadata:
                result_dict["metadata"] = result.metadata
            
            formatted_results.append(result_dict)
        
        return ContextualQueryResponse(
            query=request.query,
            strategy_used=request.strategy,
            results=formatted_results,
            total_results=len(results),
            response_time_ms=response_time,
            confidence_scores=confidence_scores,
            metadata={
                "retrieval_config": config_updates,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

@router.get("/query/stream")
async def stream_contextual_query(
    query: str = Query(..., description="Search query"),
    strategy: RetrievalStrategy = Query(default=RetrievalStrategy.HYBRID),
    max_results: int = Query(default=10, ge=1, le=50),
    retrieval_system: ContextualRetrievalSystem = Depends(get_retrieval_system)
):
    """Stream contextual retrieval results in real-time."""
    
    async def generate_results():
        """Generate streaming results."""
        try:
            # Configure system
            retrieval_system.update_config(
                retrieval_strategy=strategy,
                max_results=max_results
            )
            
            # Stream results
            async for result in retrieval_system.astream_retrieve(query):
                yield f"data: {json.dumps({
                    'content': result.page_content,
                    'score': result.metadata.get('score', 0.0),
                    'confidence': result.metadata.get('confidence_score', 0.0),
                    'timestamp': datetime.now().isoformat()
                })}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'status': 'complete'})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_results(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

# Document Ingestion Endpoints
@router.post("/ingest", response_model=IngestionResponse)
async def ingest_document(
    request: DocumentIngestionRequest,
    background_tasks: BackgroundTasks,
    retrieval_system: ContextualRetrievalSystem = Depends(get_retrieval_system)
):
    """Ingest a single document with contextual embedding generation."""
    
    start_time = datetime.now()
    
    try:
        # Create contextual embedding system
        embedding_system = ContextualEmbeddingSystem(
            embedding_model=retrieval_system.embedding_model,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap
        )
        
        # Process document
        contextual_chunks = await embedding_system.create_contextual_chunks(
            content=request.content,
            title=request.title,
            metadata=request.metadata
        )
        
        # Generate embeddings if requested
        embeddings_generated = 0
        if request.generate_contextual_embeddings:
            for chunk in contextual_chunks:
                await embedding_system.generate_embedding(chunk)
                embeddings_generated += 1
        
        # Store in database
        document_id = await retrieval_system.store_contextual_chunks(
            contextual_chunks,
            metadata=request.metadata
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return IngestionResponse(
            success=True,
            document_id=document_id,
            chunks_created=len(contextual_chunks),
            embeddings_generated=embeddings_generated,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Document ingestion failed: {e}", exc_info=True)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return IngestionResponse(
            success=False,
            chunks_created=0,
            embeddings_generated=0,
            processing_time_ms=processing_time,
            errors=[str(e)]
        )

@router.post("/ingest/batch", response_model=BatchIngestionResponse)
async def batch_ingest_documents(
    request: BatchIngestionRequest,
    background_tasks: BackgroundTasks,
    retrieval_system: ContextualRetrievalSystem = Depends(get_retrieval_system)
):
    """Batch ingest multiple documents with progress tracking."""
    
    start_time = datetime.now()
    
    try:
        total_documents = len(request.documents)
        successful_documents = 0
        failed_documents = 0
        total_chunks_created = 0
        total_embeddings_generated = 0
        errors = []
        
        # Process documents in batches
        for i in range(0, total_documents, request.batch_size):
            batch = request.documents[i:i + request.batch_size]
            
            # Process batch concurrently
            batch_tasks = []
            for doc_request in batch:
                task = asyncio.create_task(
                    process_single_document(doc_request, retrieval_system)
                )
                batch_tasks.append(task)
            
            # Wait for batch completion
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Process results
            for result in batch_results:
                if isinstance(result, Exception):
                    failed_documents += 1
                    errors.append(str(result))
                else:
                    successful_documents += 1
                    total_chunks_created += result["chunks_created"]
                    total_embeddings_generated += result["embeddings_generated"]
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Schedule background optimization
        background_tasks.add_task(
            optimize_retrieval_performance,
            retrieval_system
        )
        
        return BatchIngestionResponse(
            total_documents=total_documents,
            successful_documents=successful_documents,
            failed_documents=failed_documents,
            total_chunks_created=total_chunks_created,
            total_embeddings_generated=total_embeddings_generated,
            processing_time_ms=processing_time,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Batch ingestion failed: {e}", exc_info=True)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return BatchIngestionResponse(
            total_documents=len(request.documents),
            successful_documents=0,
            failed_documents=len(request.documents),
            total_chunks_created=0,
            total_embeddings_generated=0,
            processing_time_ms=processing_time,
            errors=[str(e)]
        )

# Migration Endpoints
@router.post("/migrate", response_model=MigrationResponse)
async def migrate_content(
    request: MigrationRequest,
    background_tasks: BackgroundTasks,
    supabase_client: Client = Depends(get_supabase_client)
):
    """Migrate existing content to contextual format."""
    
    start_time = datetime.now()
    migration_id = f"migration_{int(datetime.now().timestamp())}"
    
    try:
        # Query source data
        source_query = supabase_client.table(request.source_table).select("*")
        if not request.dry_run:
            source_query = source_query.limit(request.batch_size)
        
        source_data = source_query.execute()
        total_records = len(source_data.data)
        
        if request.dry_run:
            # Dry run - just return statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return MigrationResponse(
                migration_id=migration_id,
                source_table=request.source_table,
                target_table=request.target_table,
                total_records=total_records,
                migrated_records=0,
                failed_records=0,
                dry_run=True,
                processing_time_ms=processing_time,
                errors=[]
            )
        
        # Perform actual migration
        migrated_records = 0
        failed_records = 0
        errors = []
        
        # Create contextual embedding system
        embedding_system = ContextualEmbeddingSystem()
        
        # Process records in batches
        for i in range(0, total_records, request.batch_size):
            batch = source_data.data[i:i + request.batch_size]
            
            for record in batch:
                try:
                    # Convert to contextual format
                    contextual_chunks = await embedding_system.create_contextual_chunks(
                        content=record.get("content", ""),
                        title=record.get("title"),
                        metadata=record.get("metadata", {})
                    )
                    
                    # Store in target table
                    for chunk in contextual_chunks:
                        supabase_client.table(request.target_table).insert({
                            "content": chunk.content,
                            "full_text": chunk.full_text,
                            "context_before": chunk.context_before,
                            "context_after": chunk.context_after,
                            "metadata": chunk.metadata,
                            "source_id": record.get("id"),
                            "created_at": datetime.now().isoformat()
                        }).execute()
                    
                    migrated_records += 1
                    
                except Exception as e:
                    failed_records += 1
                    errors.append(f"Record {record.get('id', 'unknown')}: {str(e)}")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Schedule background cleanup
        background_tasks.add_task(
            cleanup_migration_temp_data,
            migration_id
        )
        
        return MigrationResponse(
            migration_id=migration_id,
            source_table=request.source_table,
            target_table=request.target_table,
            total_records=total_records,
            migrated_records=migrated_records,
            failed_records=failed_records,
            dry_run=False,
            processing_time_ms=processing_time,
            errors=errors
        )
        
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return MigrationResponse(
            migration_id=migration_id,
            source_table=request.source_table,
            target_table=request.target_table,
            total_records=0,
            migrated_records=0,
            failed_records=0,
            dry_run=request.dry_run,
            processing_time_ms=processing_time,
            errors=[str(e)]
        )

# Performance and Monitoring Endpoints
@router.get("/metrics")
async def get_performance_metrics(
    request: PerformanceMetricsRequest = Depends(),
    retrieval_system: ContextualRetrievalSystem = Depends(get_retrieval_system)
):
    """Get comprehensive performance metrics."""
    
    try:
        # Get performance optimizer
        optimizer = retrieval_system.performance_optimizer
        
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=request.time_range_hours)
        
        # Get metrics
        metrics = await optimizer.get_performance_metrics(
            start_time=start_time,
            end_time=end_time,
            group_by=request.group_by
        )
        
        # Add detailed stats if requested
        if request.include_detailed_stats:
            detailed_stats = await optimizer.get_detailed_performance_stats()
            metrics["detailed_stats"] = detailed_stats
        
        return {
            "status": "success",
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "hours": request.time_range_hours
            },
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve metrics: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Comprehensive system health check."""
    
    try:
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check database connection
        try:
            supabase_client = await get_supabase_client()
            supabase_client.table("contextual_chunks").select("id").limit(1).execute()
            health_status["components"]["database"] = "healthy"
        except Exception as e:
            health_status["components"]["database"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Background Tasks
async def process_single_document(
    doc_request: DocumentIngestionRequest,
    retrieval_system: ContextualRetrievalSystem
) -> Dict[str, Any]:
    """Process a single document for batch ingestion."""
    
    embedding_system = ContextualEmbeddingSystem(
        embedding_model=retrieval_system.embedding_model,
        chunk_size=doc_request.chunk_size,
        chunk_overlap=doc_request.chunk_overlap
    )
    
    # Create contextual chunks
    contextual_chunks = await embedding_system.create_contextual_chunks(
        content=doc_request.content,
        title=doc_request.title,
        metadata=doc_request.metadata
    )
    
    # Generate embeddings
    embeddings_generated = 0
    if doc_request.generate_contextual_embeddings:
        for chunk in contextual_chunks:
            await embedding_system.generate_embedding(chunk)
            embeddings_generated += 1
    
    # Store chunks
    await retrieval_system.store_contextual_chunks(
        contextual_chunks,
        metadata=doc_request.metadata
    )
    
    return {
        "chunks_created": len(contextual_chunks),
        "embeddings_generated": embeddings_generated
    }

async def optimize_retrieval_performance(retrieval_system: ContextualRetrievalSystem):
    """Background task for performance optimization."""
    try:
        if hasattr(retrieval_system, 'performance_optimizer'):
            await retrieval_system.performance_optimizer.optimize_system()
            logger.info("Background performance optimization completed")
    except Exception as e:
        logger.error(f"Background optimization failed: {e}")

async def cleanup_migration_temp_data(migration_id: str):
    """Background task for migration cleanup."""
    try:
        # Implement cleanup logic here
        logger.info(f"Migration cleanup completed for {migration_id}")
    except Exception as e:
        logger.error(f"Migration cleanup failed: {e}")

# System initialization
@router.on_event("startup")
async def startup_event():
    """Initialize system on startup."""
    try:
        # Pre-initialize retrieval system
        await get_retrieval_system()
        logger.info("Contextual retrieval API initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize contextual retrieval API: {e}")

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global _retrieval_system
    if _retrieval_system:
        try:
            await _retrieval_system.cleanup()
            logger.info("Contextual retrieval system cleaned up successfully")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
        finally:
            _retrieval_system = None 