"""
Database Utilities for Security Operations
Secure database connection and operation utilities for Universal RAG CMS

Features:
- Secure connection management
- Connection pooling
- Query safety and parameterization
- Transaction management
- Error handling and logging
"""

import asyncio
import logging
import os
from typing import Optional, Any, Dict, List
from contextlib import asynccontextmanager
import asyncpg
from asyncpg import Pool

logger = logging.getLogger(__name__)

# Global connection pool
_connection_pool: Optional[Pool] = None


class DatabaseManager:
    """Database manager for security operations"""
    
    def __init__(self):
        self.pool = None
    
    async def initialize(self) -> bool:
        """Initialize the database connection pool."""
        try:
            # Get database URL from environment
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                # Construct from Supabase credentials
                supabase_url = os.getenv('SUPABASE_URL')
                supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
                
                if not supabase_url or not supabase_key:
                    raise ValueError("Database credentials not found in environment")
                
                # Extract database URL from Supabase URL
                project_id = supabase_url.split('//')[1].split('.')[0]
                database_url = f"postgresql://postgres:{supabase_key}@db.{project_id}.supabase.co:5432/postgres"
            
            # Create connection pool
            self.pool = await asyncpg.create_pool(
                database_url,
                min_size=5,
                max_size=20,
                command_timeout=60,
                server_settings={
                    'application_name': 'universal_rag_cms_security',
                    'search_path': 'public'
                }
            )
            
            logger.info("Database connection pool initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Database pool initialization failed: {e}")
            return False
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self.pool:
            await self.initialize()
        
        if not self.pool:
            raise Exception("Database connection pool not available")
        
        async with self.pool.acquire() as connection:
            yield connection
    
    async def execute_query(
        self,
        query: str, 
        params: Optional[tuple] = None,
        fetch_one: bool = False,
        fetch_all: bool = False
    ) -> Any:
        """Execute a database query safely."""
        try:
            async with self.get_connection() as conn:
                if fetch_one:
                    return await conn.fetchrow(query, *(params or ()))
                elif fetch_all:
                    return await conn.fetch(query, *(params or ()))
                else:
                    return await conn.execute(query, *(params or ()))
        
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise
    
    async def close(self):
        """Close the database connection pool."""
        if self.pool:
            await self.pool.close()
            self.pool = None


async def initialize_database_pool() -> bool:
    """Initialize the global database connection pool."""
    
    global _connection_pool
    
    try:
        # Get database URL from environment
        database_url = os.getenv('DATABASE_URL')
        if not database_url:
            # Construct from Supabase credentials
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_SERVICE_KEY')
            
            if not supabase_url or not supabase_key:
                raise ValueError("Database credentials not found in environment")
            
            # Extract database URL from Supabase URL
            # Format: https://project.supabase.co -> postgres://postgres:password@db.project.supabase.co:5432/postgres
            project_id = supabase_url.split('//')[1].split('.')[0]
            database_url = f"postgresql://postgres:{supabase_key}@db.{project_id}.supabase.co:5432/postgres"
        
        # Create connection pool
        _connection_pool = await asyncpg.create_pool(
            database_url,
            min_size=5,
            max_size=20,
            command_timeout=60,
            server_settings={
                'application_name': 'universal_rag_cms_security',
                'search_path': 'public'
            }
        )
        
        logger.info("Database connection pool initialized successfully")
        return True
    
    except Exception as e:
        logger.error(f"Database pool initialization failed: {e}")
        return False


@asynccontextmanager
async def get_database_connection():
    """Get a database connection from the pool."""
    
    global _connection_pool
    
    if not _connection_pool:
        await initialize_database_pool()
    
    if not _connection_pool:
        raise Exception("Database connection pool not available")
    
    async with _connection_pool.acquire() as connection:
        yield connection


async def execute_query(
    query: str, 
    params: Optional[tuple] = None,
    fetch_one: bool = False,
    fetch_all: bool = False
) -> Any:
    """
    Execute a database query safely.
    
    Args:
        query: SQL query with parameter placeholders
        params: Query parameters
        fetch_one: Whether to fetch one result
        fetch_all: Whether to fetch all results
        
    Returns:
        Query results or None
    """
    
    try:
        async with get_database_connection() as conn:
            if fetch_one:
                return await conn.fetchrow(query, *(params or ()))
            elif fetch_all:
                return await conn.fetch(query, *(params or ()))
            else:
                return await conn.execute(query, *(params or ()))
    
    except Exception as e:
        logger.error(f"Database query failed: {e}")
        raise


async def execute_transaction(operations: list) -> bool:
    """
    Execute multiple operations in a transaction.
    
    Args:
        operations: List of (query, params) tuples
        
    Returns:
        True if successful, False otherwise
    """
    
    try:
        async with get_database_connection() as conn:
            async with conn.transaction():
                for query, params in operations:
                    await conn.execute(query, *(params or ()))
        
        return True
    
    except Exception as e:
        logger.error(f"Transaction failed: {e}")
        return False


async def check_table_exists(table_name: str) -> bool:
    """Check if a table exists in the database."""
    
    try:
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = $1
            )
        """
        
        result = await execute_query(query, (table_name,), fetch_one=True)
        return result[0] if result else False
    
    except Exception as e:
        logger.error(f"Error checking table existence: {e}")
        return False


async def get_table_schema(table_name: str) -> Dict[str, Any]:
    """Get table schema information."""
    
    try:
        query = """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns 
            WHERE table_name = $1
            ORDER BY ordinal_position
        """
        
        results = await execute_query(query, (table_name,), fetch_all=True)
        
        schema = {
            'table_name': table_name,
            'columns': []
        }
        
        for row in results:
            schema['columns'].append({
                'name': row[0],
                'type': row[1],
                'nullable': row[2] == 'YES',
                'default': row[3]
            })
        
        return schema
    
    except Exception as e:
        logger.error(f"Error getting table schema: {e}")
        return {}


async def close_database_pool():
    """Close the database connection pool."""
    
    global _connection_pool
    
    if _connection_pool:
        await _connection_pool.close()
        _connection_pool = None
        logger.info("Database connection pool closed") 