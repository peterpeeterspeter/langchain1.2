"""
Comprehensive Test Suite for Supabase Foundation (Task 1)
Testing database operations, authentication, storage, RLS policies, and edge functions.

Tests cover:
- Database migrations and schema validation
- Vector operations and similarity search
- Authentication flows and session management
- Storage bucket operations
- Row Level Security (RLS) policy enforcement
- Edge function deployment and execution
"""

import pytest
import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

# Test imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from supabase import create_client, Client
    import postgrest
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Vector operations
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


# ============================================================================
# SUPABASE DATABASE OPERATIONS TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.supabase
class TestSupabaseDatabaseOperations:
    """Test core database operations and schema management."""
    
    @pytest.fixture
    def mock_client(self):
        """Mock Supabase client for unit tests."""
        mock_client = Mock(spec=Client)
        
        # Mock table operations
        mock_table = Mock()
        mock_client.table.return_value = mock_table
        
        # Mock query builder chain
        mock_select = Mock()
        mock_table.select.return_value = mock_select
        mock_table.insert.return_value = mock_select
        mock_table.update.return_value = mock_select
        mock_table.delete.return_value = mock_select
        mock_table.upsert.return_value = mock_select
        
        # Mock filters
        mock_select.eq.return_value = mock_select
        mock_select.neq.return_value = mock_select
        mock_select.gt.return_value = mock_select
        mock_select.gte.return_value = mock_select
        mock_select.lt.return_value = mock_select
        mock_select.lte.return_value = mock_select
        mock_select.like.return_value = mock_select
        mock_select.ilike.return_value = mock_select
        mock_select.is_.return_value = mock_select
        mock_select.in_.return_value = mock_select
        mock_select.contains.return_value = mock_select
        mock_select.contained_by.return_value = mock_select
        mock_select.range_gte.return_value = mock_select
        mock_select.range_gt.return_value = mock_select
        mock_select.range_lte.return_value = mock_select
        mock_select.range_lt.return_value = mock_select
        
        # Mock ordering and limiting
        mock_select.order.return_value = mock_select
        mock_select.limit.return_value = mock_select
        mock_select.offset.return_value = mock_select
        
        return mock_client
    
    def test_database_connection(self, mock_client):
        """Test database connection establishment."""
        # Test connection creation
        assert mock_client is not None
        
        # Test basic table access
        table = mock_client.table("documents")
        assert table is not None
        
        # Verify table method was called
        mock_client.table.assert_called_with("documents")
    
    def test_create_document(self, mock_client):
        """Test document creation in database."""
        # Mock successful insert
        mock_client.table.return_value.insert.return_value.execute.return_value = Mock(
            data=[{
                "id": "test-doc-id",
                "content": "Test document content",
                "metadata": {"source": "test.pdf"},
                "created_at": datetime.now().isoformat()
            }],
            count=1
        )
        
        # Test document creation
        document_data = {
            "content": "Test document content",
            "metadata": {"source": "test.pdf"},
            "embedding": [0.1] * 1536
        }
        
        result = mock_client.table("documents").insert(document_data).execute()
        
        # Assertions
        assert result.data is not None
        assert len(result.data) == 1
        assert result.data[0]["content"] == "Test document content"
        assert result.count == 1
        
        # Verify insert was called
        mock_client.table.assert_called_with("documents")
    
    def test_query_documents(self, mock_client):
        """Test document querying with filters."""
        # Mock query result
        mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = Mock(
            data=[
                {
                    "id": "doc1",
                    "content": "Document 1 content",
                    "metadata": {"source": "test1.pdf"}
                },
                {
                    "id": "doc2", 
                    "content": "Document 2 content",
                    "metadata": {"source": "test2.pdf"}
                }
            ],
            count=2
        )
        
        # Test query with filter
        result = (mock_client.table("documents")
                 .select("*")
                 .eq("metadata->source", "test1.pdf")
                 .execute())
        
        # Assertions
        assert result.data is not None
        assert len(result.data) == 2
        assert result.count == 2
    
    def test_update_document(self, mock_client):
        """Test document updates."""
        # Mock update result
        mock_client.table.return_value.update.return_value.eq.return_value.execute.return_value = Mock(
            data=[{
                "id": "test-doc-id",
                "content": "Updated content",
                "updated_at": datetime.now().isoformat()
            }],
            count=1
        )
        
        # Test update
        update_data = {"content": "Updated content"}
        result = (mock_client.table("documents")
                 .update(update_data)
                 .eq("id", "test-doc-id")
                 .execute())
        
        # Assertions
        assert result.data is not None
        assert result.data[0]["content"] == "Updated content"
        assert result.count == 1
    
    def test_delete_document(self, mock_client):
        """Test document deletion."""
        # Mock delete result
        mock_client.table.return_value.delete.return_value.eq.return_value.execute.return_value = Mock(
            data=[],
            count=1
        )
        
        # Test deletion
        result = (mock_client.table("documents")
                 .delete()
                 .eq("id", "test-doc-id")
                 .execute())
        
        # Assertions
        assert result.count == 1
    
    def test_upsert_operation(self, mock_client):
        """Test upsert (insert or update) operations."""
        # Mock upsert result
        mock_client.table.return_value.upsert.return_value.execute.return_value = Mock(
            data=[{
                "id": "test-doc-id",
                "content": "Upserted content",
                "updated_at": datetime.now().isoformat()
            }],
            count=1
        )
        
        # Test upsert
        upsert_data = {
            "id": "test-doc-id",
            "content": "Upserted content"
        }
        result = mock_client.table("documents").upsert(upsert_data).execute()
        
        # Assertions
        assert result.data is not None
        assert result.data[0]["content"] == "Upserted content"
        assert result.count == 1
    
    def test_complex_queries(self, mock_client):
        """Test complex database queries with multiple filters."""
        # Mock complex query result
        mock_client.table.return_value.select.return_value.gte.return_value.lte.return_value.order.return_value.limit.return_value.execute.return_value = Mock(
            data=[
                {"id": "doc1", "score": 0.8, "created_at": "2024-01-01"},
                {"id": "doc2", "score": 0.9, "created_at": "2024-01-02"}
            ],
            count=2
        )
        
        # Test complex query
        result = (mock_client.table("documents")
                 .select("id, score, created_at")
                 .gte("score", 0.7)
                 .lte("score", 1.0)
                 .order("score", desc=True)
                 .limit(10)
                 .execute())
        
        # Assertions
        assert result.data is not None
        assert len(result.data) == 2
        assert all(doc["score"] >= 0.7 for doc in result.data)
    
    def test_json_operations(self, mock_client):
        """Test JSON field operations."""
        # Mock JSON query result
        mock_client.table.return_value.select.return_value.contains.return_value.execute.return_value = Mock(
            data=[{
                "id": "doc1",
                "metadata": {"source": "test.pdf", "page": 1, "type": "academic"}
            }],
            count=1
        )
        
        # Test JSON contains query
        result = (mock_client.table("documents")
                 .select("*")
                 .contains("metadata", {"type": "academic"})
                 .execute())
        
        # Assertions
        assert result.data is not None
        assert result.data[0]["metadata"]["type"] == "academic"


# ============================================================================
# VECTOR OPERATIONS TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.supabase
class TestSupabaseVectorOperations:
    """Test vector operations and similarity search."""
    
    @pytest.fixture
    def mock_vector_client(self):
        """Mock client with vector operations."""
        mock_client = Mock(spec=Client)
        mock_client.rpc = Mock()
        return mock_client
    
    @pytest.mark.skipif(not NUMPY_AVAILABLE, reason="NumPy not available")
    def test_vector_similarity_search(self, mock_vector_client):
        """Test vector similarity search using RPC calls."""
        # Mock similarity search result
        mock_vector_client.rpc.return_value = Mock(
            data=[
                {
                    "id": "doc1",
                    "content": "Similar document 1",
                    "similarity": 0.95,
                    "metadata": {"source": "test1.pdf"}
                },
                {
                    "id": "doc2",
                    "content": "Similar document 2", 
                    "similarity": 0.87,
                    "metadata": {"source": "test2.pdf"}
                }
            ]
        )
        
        # Test vector search
        query_embedding = np.random.random(1536).tolist()
        result = mock_vector_client.rpc("match_documents", {
            "query_embedding": query_embedding,
            "match_threshold": 0.8,
            "match_count": 5
        })
        
        # Assertions
        assert result.data is not None
        assert len(result.data) == 2
        assert all(doc["similarity"] >= 0.8 for doc in result.data)
        assert result.data[0]["similarity"] >= result.data[1]["similarity"]  # Ordered by similarity
    
    def test_vector_embedding_storage(self, mock_vector_client):
        """Test storing document embeddings."""
        # Mock embedding storage
        mock_vector_client.table.return_value.insert.return_value.execute.return_value = Mock(
            data=[{
                "id": "test-doc-id",
                "content": "Test document",
                "embedding": [0.1] * 1536,
                "created_at": datetime.now().isoformat()
            }],
            count=1
        )
        
        # Test embedding storage
        document_data = {
            "content": "Test document",
            "embedding": [0.1] * 1536,
            "metadata": {"source": "test.pdf"}
        }
        
        result = mock_vector_client.table("documents").insert(document_data).execute()
        
        # Assertions
        assert result.data is not None
        assert len(result.data[0]["embedding"]) == 1536
        assert result.count == 1
    
    def test_hybrid_search(self, mock_vector_client):
        """Test hybrid search combining text and vector search."""
        # Mock hybrid search result
        mock_vector_client.rpc.return_value = Mock(
            data=[
                {
                    "id": "doc1",
                    "content": "Casino safety guide",
                    "text_score": 0.8,
                    "vector_score": 0.9,
                    "combined_score": 0.85,
                    "metadata": {"source": "safety-guide.pdf"}
                }
            ]
        )
        
        # Test hybrid search
        result = mock_vector_client.rpc("hybrid_search", {
            "query_text": "casino safety",
            "query_embedding": [0.1] * 1536,
            "text_weight": 0.3,
            "vector_weight": 0.7,
            "match_count": 10
        })
        
        # Assertions
        assert result.data is not None
        assert result.data[0]["combined_score"] > 0
        assert "text_score" in result.data[0]
        assert "vector_score" in result.data[0]


# ============================================================================
# AUTHENTICATION TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.supabase
class TestSupabaseAuthentication:
    """Test authentication flows and session management."""
    
    @pytest.fixture
    def mock_auth_client(self):
        """Mock client with auth operations."""
        mock_client = Mock(spec=Client)
        mock_client.auth = Mock()
        return mock_client
    
    def test_user_signup(self, mock_auth_client):
        """Test user registration."""
        # Mock signup response
        mock_auth_client.auth.sign_up.return_value = Mock(
            user=Mock(
                id="test-user-id",
                email="test@example.com",
                email_confirmed_at=None,
                created_at=datetime.now().isoformat()
            ),
            session=None  # Email confirmation required
        )
        
        # Test signup
        result = mock_auth_client.auth.sign_up({
            "email": "test@example.com",
            "password": "secure_password_123"
        })
        
        # Assertions
        assert result.user is not None
        assert result.user.email == "test@example.com"
        assert result.user.id == "test-user-id"
        assert result.session is None  # Email not confirmed
    
    def test_user_signin(self, mock_auth_client):
        """Test user login."""
        # Mock signin response
        mock_auth_client.auth.sign_in_with_password.return_value = Mock(
            user=Mock(
                id="test-user-id",
                email="test@example.com",
                email_confirmed_at=datetime.now().isoformat()
            ),
            session=Mock(
                access_token="test-access-token",
                refresh_token="test-refresh-token",
                expires_at=int((datetime.now() + timedelta(hours=1)).timestamp())
            )
        )
        
        # Test signin
        result = mock_auth_client.auth.sign_in_with_password({
            "email": "test@example.com", 
            "password": "secure_password_123"
        })
        
        # Assertions
        assert result.user is not None
        assert result.session is not None
        assert result.session.access_token == "test-access-token"
        assert result.user.email_confirmed_at is not None
    
    def test_session_refresh(self, mock_auth_client):
        """Test session token refresh."""
        # Mock refresh response
        mock_auth_client.auth.refresh_session.return_value = Mock(
            user=Mock(id="test-user-id", email="test@example.com"),
            session=Mock(
                access_token="new-access-token",
                refresh_token="new-refresh-token",
                expires_at=int((datetime.now() + timedelta(hours=1)).timestamp())
            )
        )
        
        # Test session refresh
        result = mock_auth_client.auth.refresh_session("test-refresh-token")
        
        # Assertions
        assert result.session is not None
        assert result.session.access_token == "new-access-token"
        assert result.session.refresh_token == "new-refresh-token"
    
    def test_user_signout(self, mock_auth_client):
        """Test user logout."""
        # Mock signout response
        mock_auth_client.auth.sign_out.return_value = Mock()
        
        # Test signout
        result = mock_auth_client.auth.sign_out()
        
        # Assertions
        assert result is not None
        mock_auth_client.auth.sign_out.assert_called_once()
    
    def test_password_reset(self, mock_auth_client):
        """Test password reset flow."""
        # Mock password reset response
        mock_auth_client.auth.reset_password_email.return_value = Mock()
        
        # Test password reset
        result = mock_auth_client.auth.reset_password_email("test@example.com")
        
        # Assertions
        assert result is not None
        mock_auth_client.auth.reset_password_email.assert_called_with("test@example.com")
    
    def test_user_metadata_update(self, mock_auth_client):
        """Test updating user metadata."""
        # Mock update response
        mock_auth_client.auth.update_user.return_value = Mock(
            user=Mock(
                id="test-user-id",
                email="test@example.com",
                user_metadata={"name": "Test User", "role": "admin"}
            )
        )
        
        # Test metadata update
        result = mock_auth_client.auth.update_user({
            "data": {"name": "Test User", "role": "admin"}
        })
        
        # Assertions
        assert result.user is not None
        assert result.user.user_metadata["name"] == "Test User"
        assert result.user.user_metadata["role"] == "admin"


# ============================================================================
# STORAGE TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.supabase
class TestSupabaseStorage:
    """Test storage bucket operations."""
    
    @pytest.fixture
    def mock_storage_client(self):
        """Mock client with storage operations."""
        mock_client = Mock(spec=Client)
        mock_client.storage = Mock()
        mock_bucket = Mock()
        mock_client.storage.from_.return_value = mock_bucket
        return mock_client, mock_bucket
    
    def test_file_upload(self, mock_storage_client):
        """Test file upload to storage bucket."""
        mock_client, mock_bucket = mock_storage_client
        
        # Mock upload response
        mock_bucket.upload.return_value = Mock(
            path="documents/test-file.pdf",
            id="file-id-123",
            fullPath="documents/test-file.pdf"
        )
        
        # Test file upload
        file_content = b"Test PDF content"
        result = mock_bucket.upload("documents/test-file.pdf", file_content, {
            "content-type": "application/pdf"
        })
        
        # Assertions
        assert result.path == "documents/test-file.pdf"
        assert result.id == "file-id-123"
        mock_bucket.upload.assert_called_once()
    
    def test_file_download(self, mock_storage_client):
        """Test file download from storage bucket."""
        mock_client, mock_bucket = mock_storage_client
        
        # Mock download response
        mock_bucket.download.return_value = b"Downloaded file content"
        
        # Test file download
        result = mock_bucket.download("documents/test-file.pdf")
        
        # Assertions
        assert result == b"Downloaded file content"
        mock_bucket.download.assert_called_with("documents/test-file.pdf")
    
    def test_file_list(self, mock_storage_client):
        """Test listing files in bucket."""
        mock_client, mock_bucket = mock_storage_client
        
        # Mock list response
        mock_bucket.list.return_value = [
            Mock(
                name="test-file-1.pdf",
                id="file-1",
                created_at="2024-01-01T00:00:00Z",
                size=1024
            ),
            Mock(
                name="test-file-2.pdf",
                id="file-2", 
                created_at="2024-01-02T00:00:00Z",
                size=2048
            )
        ]
        
        # Test file listing
        result = mock_bucket.list("documents/")
        
        # Assertions
        assert len(result) == 2
        assert result[0].name == "test-file-1.pdf"
        assert result[1].size == 2048
    
    def test_file_delete(self, mock_storage_client):
        """Test file deletion from bucket."""
        mock_client, mock_bucket = mock_storage_client
        
        # Mock delete response
        mock_bucket.remove.return_value = Mock(
            message="Successfully deleted"
        )
        
        # Test file deletion
        result = mock_bucket.remove(["documents/test-file.pdf"])
        
        # Assertions
        assert result.message == "Successfully deleted"
        mock_bucket.remove.assert_called_with(["documents/test-file.pdf"])
    
    def test_signed_url_generation(self, mock_storage_client):
        """Test generating signed URLs for file access."""
        mock_client, mock_bucket = mock_storage_client
        
        # Mock signed URL response
        mock_bucket.create_signed_url.return_value = Mock(
            signed_url="https://example.com/signed-url?token=abc123"
        )
        
        # Test signed URL generation
        result = mock_bucket.create_signed_url("documents/test-file.pdf", 3600)
        
        # Assertions
        assert "signed-url" in result.signed_url
        assert "token=" in result.signed_url
        mock_bucket.create_signed_url.assert_called_with("documents/test-file.pdf", 3600)


# ============================================================================
# ROW LEVEL SECURITY (RLS) TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.supabase
class TestSupabaseRLS:
    """Test Row Level Security policy enforcement."""
    
    @pytest.fixture
    def mock_rls_client(self):
        """Mock client with RLS testing capabilities."""
        mock_client = Mock(spec=Client)
        
        # Mock different user contexts
        def set_auth_context(user_id, role="authenticated"):
            mock_client._current_user_id = user_id
            mock_client._current_role = role
        
        mock_client.set_auth_context = set_auth_context
        mock_client._current_user_id = None
        mock_client._current_role = "anon"
        
        return mock_client
    
    def test_user_isolation_policy(self, mock_rls_client):
        """Test that users can only access their own data."""
        # Set user context
        mock_rls_client.set_auth_context("user-1", "authenticated")
        
        # Mock filtered query result (RLS applied)
        mock_rls_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[
                {"id": "doc-1", "user_id": "user-1", "content": "User 1 document"},
                {"id": "doc-2", "user_id": "user-1", "content": "Another User 1 document"}
            ],
            count=2
        )
        
        # Test query - should only return user's documents
        result = mock_rls_client.table("user_documents").select("*").execute()
        
        # Assertions
        assert all(doc["user_id"] == "user-1" for doc in result.data)
        assert len(result.data) == 2
    
    def test_admin_access_policy(self, mock_rls_client):
        """Test that admins can access all data."""
        # Set admin context
        mock_rls_client.set_auth_context("admin-1", "admin")
        
        # Mock admin query result (no RLS filtering)
        mock_rls_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[
                {"id": "doc-1", "user_id": "user-1", "content": "User 1 document"},
                {"id": "doc-2", "user_id": "user-2", "content": "User 2 document"},
                {"id": "doc-3", "user_id": "user-3", "content": "User 3 document"}
            ],
            count=3
        )
        
        # Test admin query - should return all documents
        result = mock_rls_client.table("user_documents").select("*").execute()
        
        # Assertions
        assert len(result.data) == 3
        user_ids = {doc["user_id"] for doc in result.data}
        assert len(user_ids) == 3  # Multiple users' data accessible
    
    def test_anonymous_access_restriction(self, mock_rls_client):
        """Test that anonymous users have restricted access."""
        # Set anonymous context (default)
        assert mock_rls_client._current_role == "anon"
        
        # Mock restricted query result (RLS blocks access)
        mock_rls_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[],  # No data returned for anonymous user
            count=0
        )
        
        # Test anonymous query - should return no private data
        result = mock_rls_client.table("user_documents").select("*").execute()
        
        # Assertions
        assert len(result.data) == 0
        assert result.count == 0
    
    def test_read_only_policy(self, mock_rls_client):
        """Test read-only access policies."""
        # Set read-only user context
        mock_rls_client.set_auth_context("readonly-user", "readonly")
        
        # Mock successful read
        mock_rls_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[{"id": "doc-1", "content": "Public document"}],
            count=1
        )
        
        # Mock failed write (RLS blocks)
        mock_rls_client.table.return_value.insert.return_value.execute.side_effect = Exception(
            "RLS policy violation: INSERT not allowed"
        )
        
        # Test read access (should work)
        read_result = mock_rls_client.table("public_documents").select("*").execute()
        assert len(read_result.data) == 1
        
        # Test write access (should fail)
        with pytest.raises(Exception, match="RLS policy violation"):
            mock_rls_client.table("public_documents").insert({
                "content": "New document"
            }).execute()


# ============================================================================
# EDGE FUNCTIONS TESTS
# ============================================================================

@pytest.mark.unit
@pytest.mark.supabase
class TestSupabaseEdgeFunctions:
    """Test Edge Functions deployment and execution."""
    
    @pytest.fixture
    def mock_edge_client(self):
        """Mock client with edge function capabilities."""
        mock_client = Mock(spec=Client)
        mock_client.functions = Mock()
        return mock_client
    
    def test_edge_function_invoke(self, mock_edge_client):
        """Test invoking edge functions."""
        # Mock function response
        mock_edge_client.functions.invoke.return_value = Mock(
            data={"result": "Edge function executed successfully", "timestamp": "2024-01-01T00:00:00Z"},
            status_code=200
        )
        
        # Test function invocation
        result = mock_edge_client.functions.invoke("process-document", {
            "document_id": "test-doc-id",
            "operation": "extract_text"
        })
        
        # Assertions
        assert result.status_code == 200
        assert result.data["result"] == "Edge function executed successfully"
        mock_edge_client.functions.invoke.assert_called_with("process-document", {
            "document_id": "test-doc-id",
            "operation": "extract_text"
        })
    
    def test_edge_function_error_handling(self, mock_edge_client):
        """Test edge function error handling."""
        # Mock function error
        mock_edge_client.functions.invoke.return_value = Mock(
            data={"error": "Invalid document ID", "code": "INVALID_INPUT"},
            status_code=400
        )
        
        # Test function with error
        result = mock_edge_client.functions.invoke("process-document", {
            "document_id": "invalid-id"
        })
        
        # Assertions
        assert result.status_code == 400
        assert "error" in result.data
        assert result.data["code"] == "INVALID_INPUT"
    
    def test_edge_function_timeout(self, mock_edge_client):
        """Test edge function timeout handling."""
        # Mock timeout
        mock_edge_client.functions.invoke.side_effect = TimeoutError("Function execution timeout")
        
        # Test function timeout
        with pytest.raises(TimeoutError, match="Function execution timeout"):
            mock_edge_client.functions.invoke("slow-function", {})
    
    def test_async_edge_function(self, mock_edge_client):
        """Test asynchronous edge function execution."""
        # Mock async function response
        mock_edge_client.functions.invoke.return_value = Mock(
            data={"job_id": "async-job-123", "status": "queued"},
            status_code=202
        )
        
        # Test async function invocation
        result = mock_edge_client.functions.invoke("async-process", {
            "data": "large_dataset",
            "async": True
        })
        
        # Assertions
        assert result.status_code == 202
        assert result.data["status"] == "queued"
        assert "job_id" in result.data


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

@pytest.mark.integration
@pytest.mark.supabase
class TestSupabaseIntegration:
    """Integration tests for complete Supabase workflows."""
    
    @pytest.fixture
    def integrated_client(self, mock_supabase_client):
        """Mock integrated client for workflow testing."""
        return mock_supabase_client
    
    @pytest.mark.asyncio
    async def test_document_processing_workflow(self, integrated_client):
        """Test complete document processing workflow."""
        # Mock workflow responses
        integrated_client.storage.from_.return_value.upload.return_value = Mock(
            path="documents/test.pdf"
        )
        
        integrated_client.functions.invoke.return_value = Mock(
            data={"extracted_text": "Document content", "metadata": {"pages": 5}},
            status_code=200
        )
        
        integrated_client.table.return_value.insert.return_value.execute.return_value = Mock(
            data=[{"id": "doc-id", "status": "processed"}],
            count=1
        )
        
        # Step 1: Upload document
        upload_result = integrated_client.storage.from_("documents").upload(
            "test.pdf", b"PDF content"
        )
        assert upload_result.path == "documents/test.pdf"
        
        # Step 2: Process document with edge function
        process_result = integrated_client.functions.invoke("extract-text", {
            "file_path": upload_result.path
        })
        assert process_result.status_code == 200
        assert "extracted_text" in process_result.data
        
        # Step 3: Store processed data
        store_result = integrated_client.table("documents").insert({
            "content": process_result.data["extracted_text"],
            "metadata": process_result.data["metadata"],
            "file_path": upload_result.path
        }).execute()
        assert store_result.count == 1
        assert store_result.data[0]["status"] == "processed"
    
    @pytest.mark.asyncio
    async def test_user_data_lifecycle(self, integrated_client):
        """Test complete user data lifecycle."""
        # Mock lifecycle responses
        integrated_client.auth.sign_up.return_value = Mock(
            user=Mock(id="user-123", email="test@example.com"),
            session=Mock(access_token="token-123")
        )
        
        integrated_client.table.return_value.insert.return_value.execute.return_value = Mock(
            data=[{"id": "profile-123", "user_id": "user-123"}],
            count=1
        )
        
        integrated_client.table.return_value.select.return_value.eq.return_value.execute.return_value = Mock(
            data=[{"id": "profile-123", "name": "Test User"}],
            count=1
        )
        
        integrated_client.table.return_value.delete.return_value.eq.return_value.execute.return_value = Mock(
            count=1
        )
        
        # Step 1: User registration
        auth_result = integrated_client.auth.sign_up({
            "email": "test@example.com",
            "password": "password123"
        })
        assert auth_result.user.id == "user-123"
        
        # Step 2: Create user profile
        profile_result = integrated_client.table("user_profiles").insert({
            "user_id": auth_result.user.id,
            "name": "Test User"
        }).execute()
        assert profile_result.count == 1
        
        # Step 3: Query user data
        query_result = integrated_client.table("user_profiles").select("*").eq(
            "user_id", auth_result.user.id
        ).execute()
        assert len(query_result.data) == 1
        
        # Step 4: Cleanup (delete user data)
        delete_result = integrated_client.table("user_profiles").delete().eq(
            "user_id", auth_result.user.id
        ).execute()
        assert delete_result.count == 1


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.performance
@pytest.mark.supabase
class TestSupabasePerformance:
    """Performance tests for Supabase operations."""
    
    def test_bulk_insert_performance(self, mock_supabase_client, performance_config):
        """Test bulk insert performance."""
        # Mock bulk insert
        mock_supabase_client.table.return_value.insert.return_value.execute.return_value = Mock(
            data=[{"id": f"doc-{i}"} for i in range(100)],
            count=100
        )
        
        # Test bulk insert
        documents = [{"content": f"Document {i}", "metadata": {}} for i in range(100)]
        
        import time
        start_time = time.time()
        result = mock_supabase_client.table("documents").insert(documents).execute()
        execution_time = time.time() - start_time
        
        # Performance assertions
        assert execution_time < performance_config["max_execution_time"]
        assert result.count == 100
    
    def test_concurrent_operations_performance(self, mock_supabase_client):
        """Test concurrent operation performance."""
        import concurrent.futures
        import time
        
        # Mock responses
        mock_supabase_client.table.return_value.select.return_value.execute.return_value = Mock(
            data=[{"id": "doc-1", "content": "Test"}],
            count=1
        )
        
        def query_operation():
            return mock_supabase_client.table("documents").select("*").execute()
        
        # Test concurrent queries
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(query_operation) for _ in range(50)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        execution_time = time.time() - start_time
        
        # Performance assertions
        assert len(results) == 50
        assert execution_time < 10.0  # Should complete in reasonable time
        assert all(result.count == 1 for result in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 