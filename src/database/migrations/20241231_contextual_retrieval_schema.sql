-- ============================================================================
-- Task 3.6: Database Schema Migrations for Contextual Retrieval System
-- Universal RAG CMS - Contextual Retrieval Database Schema
-- ============================================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS btree_gin;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- ============================================================================
-- CONTEXTUAL EMBEDDINGS TABLES (Task 3.1)
-- ============================================================================

-- Contextual chunks with enhanced metadata
CREATE TABLE IF NOT EXISTS contextual_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    context TEXT,
    full_text TEXT GENERATED ALWAYS AS (
        CASE 
            WHEN context IS NOT NULL THEN context || ' ' || text
            ELSE text
        END
    ) STORED,
    metadata JSONB DEFAULT '{}',
    
    -- Contextual embedding vectors
    contextual_embedding vector(1536),
    original_embedding vector(1536),
    
    -- Quality and relevance scores
    context_quality_score FLOAT DEFAULT 0.0,
    relevance_score FLOAT DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT contextual_chunks_chunk_index_check CHECK (chunk_index >= 0),
    CONSTRAINT contextual_chunks_quality_score_check CHECK (context_quality_score >= 0.0 AND context_quality_score <= 1.0),
    CONSTRAINT contextual_chunks_relevance_score_check CHECK (relevance_score >= 0.0 AND relevance_score <= 1.0)
);

-- Indexes for contextual chunks
CREATE INDEX IF NOT EXISTS idx_contextual_chunks_document_id ON contextual_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_contextual_chunks_chunk_index ON contextual_chunks(chunk_index);
CREATE INDEX IF NOT EXISTS idx_contextual_chunks_contextual_embedding ON contextual_chunks USING ivfflat (contextual_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_contextual_chunks_original_embedding ON contextual_chunks USING ivfflat (original_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_contextual_chunks_quality_score ON contextual_chunks(context_quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_contextual_chunks_metadata_gin ON contextual_chunks USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_contextual_chunks_full_text_search ON contextual_chunks USING gin(to_tsvector('english', full_text));

-- ============================================================================
-- HYBRID SEARCH INFRASTRUCTURE (Task 3.2)
-- ============================================================================

-- BM25 search metadata and statistics
CREATE TABLE IF NOT EXISTS bm25_statistics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL,
    term_frequency JSONB NOT NULL, -- {term: frequency}
    document_length INTEGER NOT NULL,
    unique_terms INTEGER NOT NULL,
    
    -- BM25 parameters
    k1 FLOAT DEFAULT 1.2,
    b FLOAT DEFAULT 0.75,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT bm25_statistics_document_length_check CHECK (document_length > 0),
    CONSTRAINT bm25_statistics_unique_terms_check CHECK (unique_terms > 0)
);

-- Indexes for BM25 statistics
CREATE INDEX IF NOT EXISTS idx_bm25_statistics_document_id ON bm25_statistics(document_id);
CREATE INDEX IF NOT EXISTS idx_bm25_statistics_term_frequency ON bm25_statistics USING gin(term_frequency);

-- Global corpus statistics for BM25
CREATE TABLE IF NOT EXISTS corpus_statistics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    total_documents INTEGER NOT NULL DEFAULT 0,
    average_document_length FLOAT NOT NULL DEFAULT 0.0,
    total_terms BIGINT NOT NULL DEFAULT 0,
    unique_terms INTEGER NOT NULL DEFAULT 0,
    
    -- Term document frequencies for IDF calculation
    term_document_frequencies JSONB DEFAULT '{}', -- {term: document_count}
    
    -- Update tracking
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    version INTEGER DEFAULT 1,
    
    CONSTRAINT corpus_statistics_total_documents_check CHECK (total_documents >= 0),
    CONSTRAINT corpus_statistics_average_document_length_check CHECK (average_document_length >= 0.0)
);

-- Ensure single row for corpus statistics
CREATE UNIQUE INDEX IF NOT EXISTS idx_corpus_statistics_singleton ON corpus_statistics((1));

-- Hybrid search results cache
CREATE TABLE IF NOT EXISTS hybrid_search_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash TEXT NOT NULL,
    query_text TEXT NOT NULL,
    
    -- Search parameters
    dense_weight FLOAT NOT NULL,
    sparse_weight FLOAT NOT NULL,
    k INTEGER NOT NULL,
    
    -- Results
    results JSONB NOT NULL, -- Array of {document_id, score, rank}
    dense_results JSONB, -- Dense search results
    sparse_results JSONB, -- Sparse search results
    
    -- Performance metrics
    total_time_ms FLOAT,
    dense_time_ms FLOAT,
    sparse_time_ms FLOAT,
    fusion_time_ms FLOAT,
    
    -- Cache metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT hybrid_search_cache_weights_check CHECK (dense_weight + sparse_weight = 1.0),
    CONSTRAINT hybrid_search_cache_k_check CHECK (k > 0)
);

-- Indexes for hybrid search cache
CREATE INDEX IF NOT EXISTS idx_hybrid_search_cache_query_hash ON hybrid_search_cache(query_hash);
CREATE INDEX IF NOT EXISTS idx_hybrid_search_cache_expires_at ON hybrid_search_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_hybrid_search_cache_access_count ON hybrid_search_cache(access_count DESC);

-- ============================================================================
-- MULTI-QUERY RETRIEVAL SYSTEM (Task 3.3)
-- ============================================================================

-- Query variations and expansion cache
CREATE TABLE IF NOT EXISTS query_variations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_query TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    
    -- Generated variations
    variations JSONB NOT NULL, -- Array of query strings
    variation_count INTEGER NOT NULL,
    
    -- Generation metadata
    llm_model TEXT,
    generation_time_ms FLOAT,
    generation_temperature FLOAT DEFAULT 0.7,
    max_variations INTEGER DEFAULT 3,
    
    -- Quality metrics
    variation_quality_scores JSONB, -- Array of quality scores
    average_quality_score FLOAT,
    
    -- Cache metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    access_count INTEGER DEFAULT 0,
    last_accessed TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT query_variations_variation_count_check CHECK (variation_count > 0),
    CONSTRAINT query_variations_max_variations_check CHECK (max_variations > 0)
);

-- Indexes for query variations
CREATE INDEX IF NOT EXISTS idx_query_variations_query_hash ON query_variations(query_hash);
CREATE INDEX IF NOT EXISTS idx_query_variations_original_query ON query_variations USING gin(to_tsvector('english', original_query));
CREATE INDEX IF NOT EXISTS idx_query_variations_expires_at ON query_variations(expires_at);

-- Multi-query retrieval results
CREATE TABLE IF NOT EXISTS multi_query_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL,
    original_query TEXT NOT NULL,
    query_variations JSONB NOT NULL,
    
    -- Retrieval results per variation
    variation_results JSONB NOT NULL, -- {variation: [results]}
    combined_results JSONB NOT NULL, -- Deduplicated and ranked results
    
    -- Performance metrics
    total_time_ms FLOAT,
    parallel_execution BOOLEAN DEFAULT TRUE,
    max_workers INTEGER DEFAULT 4,
    
    -- Quality metrics
    result_diversity_score FLOAT,
    coverage_improvement FLOAT, -- vs single query
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT multi_query_results_max_workers_check CHECK (max_workers > 0)
);

-- Indexes for multi-query results
CREATE INDEX IF NOT EXISTS idx_multi_query_results_session_id ON multi_query_results(session_id);
CREATE INDEX IF NOT EXISTS idx_multi_query_results_original_query ON multi_query_results USING gin(to_tsvector('english', original_query));
CREATE INDEX IF NOT EXISTS idx_multi_query_results_created_at ON multi_query_results(created_at DESC);

-- ============================================================================
-- SELF-QUERY METADATA FILTERING (Task 3.4)
-- ============================================================================

-- Extracted query filters and constraints
CREATE TABLE IF NOT EXISTS query_filters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    original_query TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    
    -- Extracted filters
    extracted_filters JSONB NOT NULL, -- Structured filter conditions
    filter_confidence FLOAT DEFAULT 0.0,
    
    -- Cleaned query (filters removed)
    cleaned_query TEXT NOT NULL,
    
    -- Filter extraction metadata
    extraction_method TEXT, -- 'pattern', 'llm', 'hybrid'
    extraction_time_ms FLOAT,
    llm_model TEXT,
    
    -- Filter application results
    documents_before_filter INTEGER,
    documents_after_filter INTEGER,
    filter_selectivity FLOAT, -- documents_after / documents_before
    
    -- Cache metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    access_count INTEGER DEFAULT 0,
    
    CONSTRAINT query_filters_confidence_check CHECK (filter_confidence >= 0.0 AND filter_confidence <= 1.0),
    CONSTRAINT query_filters_selectivity_check CHECK (filter_selectivity >= 0.0 AND filter_selectivity <= 1.0)
);

-- Indexes for query filters
CREATE INDEX IF NOT EXISTS idx_query_filters_query_hash ON query_filters(query_hash);
CREATE INDEX IF NOT EXISTS idx_query_filters_extracted_filters ON query_filters USING gin(extracted_filters);
CREATE INDEX IF NOT EXISTS idx_query_filters_filter_confidence ON query_filters(filter_confidence DESC);
CREATE INDEX IF NOT EXISTS idx_query_filters_expires_at ON query_filters(expires_at);

-- Metadata field statistics for filter optimization
CREATE TABLE IF NOT EXISTS metadata_field_stats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    field_name TEXT NOT NULL,
    field_type TEXT NOT NULL, -- 'string', 'number', 'date', 'boolean', 'array'
    
    -- Value distribution
    unique_values INTEGER NOT NULL DEFAULT 0,
    total_documents INTEGER NOT NULL DEFAULT 0,
    null_count INTEGER NOT NULL DEFAULT 0,
    
    -- Common values for optimization
    top_values JSONB DEFAULT '[]', -- [{value, count, percentage}]
    value_ranges JSONB, -- For numeric/date fields
    
    -- Filter performance
    avg_filter_time_ms FLOAT,
    filter_usage_count INTEGER DEFAULT 0,
    
    -- Update tracking
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT metadata_field_stats_unique_values_check CHECK (unique_values >= 0),
    CONSTRAINT metadata_field_stats_total_documents_check CHECK (total_documents >= 0),
    CONSTRAINT metadata_field_stats_null_count_check CHECK (null_count >= 0)
);

-- Indexes for metadata field stats
CREATE UNIQUE INDEX IF NOT EXISTS idx_metadata_field_stats_field_name ON metadata_field_stats(field_name);
CREATE INDEX IF NOT EXISTS idx_metadata_field_stats_field_type ON metadata_field_stats(field_type);
CREATE INDEX IF NOT EXISTS idx_metadata_field_stats_filter_usage ON metadata_field_stats(filter_usage_count DESC);

-- ============================================================================
-- MAXIMAL MARGINAL RELEVANCE (Task 3.5)
-- ============================================================================

-- MMR computation cache and results
CREATE TABLE IF NOT EXISTS mmr_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_hash TEXT NOT NULL,
    query_embedding vector(1536) NOT NULL,
    
    -- MMR parameters
    lambda_param FLOAT NOT NULL,
    k INTEGER NOT NULL,
    
    -- Input documents
    candidate_documents JSONB NOT NULL, -- Array of document IDs
    candidate_count INTEGER NOT NULL,
    
    -- MMR results
    selected_documents JSONB NOT NULL, -- Ordered array of selected document IDs
    mmr_scores JSONB NOT NULL, -- {document_id: mmr_score}
    diversity_scores JSONB NOT NULL, -- {document_id: diversity_score}
    
    -- Performance metrics
    computation_time_ms FLOAT,
    similarity_calculations INTEGER,
    
    -- Quality metrics
    final_diversity_score FLOAT,
    relevance_coverage FLOAT,
    
    -- Cache metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    access_count INTEGER DEFAULT 0,
    
    CONSTRAINT mmr_results_lambda_check CHECK (lambda_param >= 0.0 AND lambda_param <= 1.0),
    CONSTRAINT mmr_results_k_check CHECK (k > 0),
    CONSTRAINT mmr_results_candidate_count_check CHECK (candidate_count > 0)
);

-- Indexes for MMR results
CREATE INDEX IF NOT EXISTS idx_mmr_results_query_hash ON mmr_results(query_hash);
CREATE INDEX IF NOT EXISTS idx_mmr_results_query_embedding ON mmr_results USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 100);
CREATE INDEX IF NOT EXISTS idx_mmr_results_lambda_k ON mmr_results(lambda_param, k);
CREATE INDEX IF NOT EXISTS idx_mmr_results_expires_at ON mmr_results(expires_at);

-- Document similarity cache for MMR optimization
CREATE TABLE IF NOT EXISTS document_similarity_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id_1 UUID NOT NULL,
    document_id_2 UUID NOT NULL,
    similarity_score FLOAT NOT NULL,
    
    -- Similarity computation metadata
    similarity_method TEXT DEFAULT 'cosine', -- 'cosine', 'euclidean', 'dot_product'
    embedding_model TEXT,
    computation_time_ms FLOAT,
    
    -- Cache metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    access_count INTEGER DEFAULT 0,
    
    CONSTRAINT document_similarity_cache_similarity_check CHECK (similarity_score >= -1.0 AND similarity_score <= 1.0),
    CONSTRAINT document_similarity_cache_document_order CHECK (document_id_1 < document_id_2)
);

-- Indexes for document similarity cache
CREATE UNIQUE INDEX IF NOT EXISTS idx_document_similarity_cache_documents ON document_similarity_cache(document_id_1, document_id_2);
CREATE INDEX IF NOT EXISTS idx_document_similarity_cache_similarity ON document_similarity_cache(similarity_score DESC);
CREATE INDEX IF NOT EXISTS idx_document_similarity_cache_expires_at ON document_similarity_cache(expires_at);

-- ============================================================================
-- PERFORMANCE MONITORING AND ANALYTICS
-- ============================================================================

-- Retrieval performance metrics
CREATE TABLE IF NOT EXISTS retrieval_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID,
    query_text TEXT NOT NULL,
    query_type TEXT, -- 'factual', 'comparison', 'tutorial', etc.
    
    -- Retrieval strategy used
    strategy TEXT NOT NULL, -- 'dense', 'sparse', 'hybrid', 'contextual', 'multi_query'
    
    -- Performance timing (in milliseconds)
    total_time_ms FLOAT NOT NULL,
    contextual_embedding_time_ms FLOAT DEFAULT 0,
    hybrid_search_time_ms FLOAT DEFAULT 0,
    multi_query_time_ms FLOAT DEFAULT 0,
    self_query_time_ms FLOAT DEFAULT 0,
    mmr_computation_time_ms FLOAT DEFAULT 0,
    
    -- Quality metrics
    relevance_score FLOAT,
    diversity_score FLOAT,
    coverage_score FLOAT,
    confidence_score FLOAT,
    
    -- Resource utilization
    memory_usage_mb FLOAT,
    cpu_usage_percent FLOAT,
    api_calls_count INTEGER DEFAULT 0,
    tokens_used INTEGER DEFAULT 0,
    
    -- Result metrics
    documents_retrieved INTEGER NOT NULL,
    documents_after_filtering INTEGER,
    cache_hit BOOLEAN DEFAULT FALSE,
    
    -- Error tracking
    errors_count INTEGER DEFAULT 0,
    warnings_count INTEGER DEFAULT 0,
    error_details JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT retrieval_metrics_total_time_check CHECK (total_time_ms >= 0),
    CONSTRAINT retrieval_metrics_documents_retrieved_check CHECK (documents_retrieved >= 0)
);

-- Indexes for retrieval metrics
CREATE INDEX IF NOT EXISTS idx_retrieval_metrics_session_id ON retrieval_metrics(session_id);
CREATE INDEX IF NOT EXISTS idx_retrieval_metrics_query_type ON retrieval_metrics(query_type);
CREATE INDEX IF NOT EXISTS idx_retrieval_metrics_strategy ON retrieval_metrics(strategy);
CREATE INDEX IF NOT EXISTS idx_retrieval_metrics_total_time ON retrieval_metrics(total_time_ms);
CREATE INDEX IF NOT EXISTS idx_retrieval_metrics_created_at ON retrieval_metrics(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_retrieval_metrics_cache_hit ON retrieval_metrics(cache_hit);

-- System performance aggregates
CREATE TABLE IF NOT EXISTS performance_aggregates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time_bucket TIMESTAMP WITH TIME ZONE NOT NULL, -- Hourly buckets
    
    -- Query volume
    total_queries INTEGER NOT NULL DEFAULT 0,
    unique_queries INTEGER NOT NULL DEFAULT 0,
    
    -- Performance averages
    avg_response_time_ms FLOAT,
    p95_response_time_ms FLOAT,
    p99_response_time_ms FLOAT,
    
    -- Cache performance
    cache_hit_rate FLOAT,
    cache_size_mb FLOAT,
    
    -- Quality metrics
    avg_relevance_score FLOAT,
    avg_diversity_score FLOAT,
    avg_confidence_score FLOAT,
    
    -- Resource utilization
    avg_memory_usage_mb FLOAT,
    avg_cpu_usage_percent FLOAT,
    total_api_calls INTEGER DEFAULT 0,
    total_tokens_used INTEGER DEFAULT 0,
    
    -- Error rates
    error_rate FLOAT DEFAULT 0.0,
    warning_rate FLOAT DEFAULT 0.0,
    
    -- Strategy distribution
    strategy_distribution JSONB DEFAULT '{}', -- {strategy: count}
    
    -- Update tracking
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT performance_aggregates_cache_hit_rate_check CHECK (cache_hit_rate >= 0.0 AND cache_hit_rate <= 1.0),
    CONSTRAINT performance_aggregates_error_rate_check CHECK (error_rate >= 0.0 AND error_rate <= 1.0)
);

-- Indexes for performance aggregates
CREATE UNIQUE INDEX IF NOT EXISTS idx_performance_aggregates_time_bucket ON performance_aggregates(time_bucket);
CREATE INDEX IF NOT EXISTS idx_performance_aggregates_avg_response_time ON performance_aggregates(avg_response_time_ms);
CREATE INDEX IF NOT EXISTS idx_performance_aggregates_cache_hit_rate ON performance_aggregates(cache_hit_rate DESC);

-- ============================================================================
-- POSTGRES RPC FUNCTIONS
-- ============================================================================

-- Function: Hybrid search combining dense and sparse results
CREATE OR REPLACE FUNCTION search_hybrid(
    query_text TEXT,
    query_embedding vector(1536),
    dense_weight FLOAT DEFAULT 0.7,
    sparse_weight FLOAT DEFAULT 0.3,
    result_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    document_id UUID,
    chunk_id UUID,
    content TEXT,
    dense_score FLOAT,
    sparse_score FLOAT,
    combined_score FLOAT,
    rank INTEGER
) 
LANGUAGE plpgsql
AS $$
DECLARE
    dense_results RECORD;
    sparse_results RECORD;
    combined_results RECORD;
BEGIN
    -- Validate weights
    IF dense_weight + sparse_weight != 1.0 THEN
        RAISE EXCEPTION 'Dense weight and sparse weight must sum to 1.0';
    END IF;
    
    -- Create temporary table for results
    CREATE TEMP TABLE IF NOT EXISTS temp_hybrid_results (
        document_id UUID,
        chunk_id UUID,
        content TEXT,
        dense_score FLOAT DEFAULT 0.0,
        sparse_score FLOAT DEFAULT 0.0,
        combined_score FLOAT DEFAULT 0.0
    );
    
    -- Clear previous results
    DELETE FROM temp_hybrid_results;
    
    -- Dense vector search
    INSERT INTO temp_hybrid_results (document_id, chunk_id, content, dense_score)
    SELECT 
        cc.document_id,
        cc.id,
        cc.text,
        1 - (cc.contextual_embedding <=> query_embedding) as similarity
    FROM contextual_chunks cc
    WHERE cc.contextual_embedding IS NOT NULL
    ORDER BY cc.contextual_embedding <=> query_embedding
    LIMIT result_limit * 2; -- Get more candidates for fusion
    
    -- Sparse BM25 search (simplified implementation)
    WITH query_terms AS (
        SELECT unnest(string_to_array(lower(query_text), ' ')) as term
    ),
    term_scores AS (
        SELECT 
            cc.id as chunk_id,
            cc.document_id,
            cc.text as content,
            SUM(
                CASE 
                    WHEN cc.full_text ILIKE '%' || qt.term || '%' THEN
                        -- Simplified BM25 scoring
                        log(1.0 + (1.0 / GREATEST(1, length(cc.full_text) - length(replace(lower(cc.full_text), qt.term, '')) + 1)))
                    ELSE 0.0
                END
            ) as bm25_score
        FROM contextual_chunks cc
        CROSS JOIN query_terms qt
        WHERE cc.full_text IS NOT NULL
        GROUP BY cc.id, cc.document_id, cc.text
        HAVING SUM(
            CASE 
                WHEN cc.full_text ILIKE '%' || qt.term || '%' THEN 1
                ELSE 0
            END
        ) > 0
    )
    INSERT INTO temp_hybrid_results (document_id, chunk_id, content, sparse_score)
    SELECT 
        ts.document_id,
        ts.chunk_id,
        ts.content,
        ts.bm25_score
    FROM term_scores ts
    ORDER BY ts.bm25_score DESC
    LIMIT result_limit * 2
    ON CONFLICT (chunk_id) DO UPDATE SET
        sparse_score = EXCLUDED.sparse_score;
    
    -- Normalize scores and combine
    WITH score_stats AS (
        SELECT 
            MAX(dense_score) as max_dense,
            MIN(dense_score) as min_dense,
            MAX(sparse_score) as max_sparse,
            MIN(sparse_score) as min_sparse
        FROM temp_hybrid_results
        WHERE dense_score > 0 OR sparse_score > 0
    )
    UPDATE temp_hybrid_results 
    SET combined_score = (
        CASE 
            WHEN ss.max_dense > ss.min_dense THEN
                (dense_score - ss.min_dense) / (ss.max_dense - ss.min_dense) * dense_weight
            ELSE dense_score * dense_weight
        END +
        CASE 
            WHEN ss.max_sparse > ss.min_sparse THEN
                (sparse_score - ss.min_sparse) / (ss.max_sparse - ss.min_sparse) * sparse_weight
            ELSE sparse_score * sparse_weight
        END
    )
    FROM score_stats ss
    WHERE dense_score > 0 OR sparse_score > 0;
    
    -- Return ranked results
    RETURN QUERY
    SELECT 
        thr.document_id,
        thr.chunk_id,
        thr.content,
        thr.dense_score,
        thr.sparse_score,
        thr.combined_score,
        ROW_NUMBER() OVER (ORDER BY thr.combined_score DESC)::INTEGER as rank
    FROM temp_hybrid_results thr
    WHERE thr.combined_score > 0
    ORDER BY thr.combined_score DESC
    LIMIT result_limit;
    
    -- Cleanup
    DROP TABLE IF EXISTS temp_hybrid_results;
END;
$$;

-- Function: Search contextual embeddings with metadata filtering
CREATE OR REPLACE FUNCTION search_contextual_embeddings(
    query_embedding vector(1536),
    metadata_filters JSONB DEFAULT '{}',
    similarity_threshold FLOAT DEFAULT 0.0,
    result_limit INTEGER DEFAULT 10
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    text TEXT,
    context TEXT,
    similarity_score FLOAT,
    metadata JSONB,
    context_quality_score FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT 
        cc.id,
        cc.document_id,
        cc.text,
        cc.context,
        1 - (cc.contextual_embedding <=> query_embedding) as similarity,
        cc.metadata,
        cc.context_quality_score
    FROM contextual_chunks cc
    WHERE 
        cc.contextual_embedding IS NOT NULL
        AND (1 - (cc.contextual_embedding <=> query_embedding)) >= similarity_threshold
        AND (
            metadata_filters = '{}'::jsonb 
            OR cc.metadata @> metadata_filters
        )
    ORDER BY cc.contextual_embedding <=> query_embedding
    LIMIT result_limit;
END;
$$;

-- Function: Update corpus statistics for BM25
CREATE OR REPLACE FUNCTION update_corpus_statistics()
RETURNS VOID
LANGUAGE plpgsql
AS $$
DECLARE
    total_docs INTEGER;
    avg_length FLOAT;
    total_terms_count BIGINT;
    unique_terms_count INTEGER;
BEGIN
    -- Calculate corpus statistics
    SELECT 
        COUNT(*),
        AVG(LENGTH(full_text)),
        SUM(LENGTH(full_text))
    INTO total_docs, avg_length, total_terms_count
    FROM contextual_chunks
    WHERE full_text IS NOT NULL;
    
    -- Estimate unique terms (simplified)
    SELECT COUNT(DISTINCT word)
    INTO unique_terms_count
    FROM (
        SELECT unnest(string_to_array(lower(full_text), ' ')) as word
        FROM contextual_chunks
        WHERE full_text IS NOT NULL
        LIMIT 10000 -- Sample for performance
    ) words;
    
    -- Update or insert corpus statistics
    INSERT INTO corpus_statistics (
        total_documents,
        average_document_length,
        total_terms,
        unique_terms,
        last_updated,
        version
    ) VALUES (
        total_docs,
        avg_length,
        total_terms_count,
        unique_terms_count,
        NOW(),
        1
    )
    ON CONFLICT ((1)) DO UPDATE SET
        total_documents = EXCLUDED.total_documents,
        average_document_length = EXCLUDED.average_document_length,
        total_terms = EXCLUDED.total_terms,
        unique_terms = EXCLUDED.unique_terms,
        last_updated = EXCLUDED.last_updated,
        version = corpus_statistics.version + 1;
END;
$$;

-- Function: Clean expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS INTEGER
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count INTEGER := 0;
BEGIN
    -- Clean hybrid search cache
    DELETE FROM hybrid_search_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Clean query variations cache
    DELETE FROM query_variations WHERE expires_at < NOW();
    
    -- Clean query filters cache
    DELETE FROM query_filters WHERE expires_at < NOW();
    
    -- Clean MMR results cache
    DELETE FROM mmr_results WHERE expires_at < NOW();
    
    -- Clean document similarity cache
    DELETE FROM document_similarity_cache WHERE expires_at < NOW();
    
    RETURN deleted_count;
END;
$$;

-- Function: Get retrieval performance summary
CREATE OR REPLACE FUNCTION get_performance_summary(
    time_range_hours INTEGER DEFAULT 24
)
RETURNS TABLE (
    total_queries INTEGER,
    avg_response_time_ms FLOAT,
    cache_hit_rate FLOAT,
    avg_relevance_score FLOAT,
    error_rate FLOAT,
    top_strategies JSONB
)
LANGUAGE plpgsql
AS $$
DECLARE
    start_time TIMESTAMP WITH TIME ZONE;
BEGIN
    start_time := NOW() - (time_range_hours || ' hours')::INTERVAL;
    
    RETURN QUERY
    SELECT 
        COUNT(*)::INTEGER as total_queries,
        AVG(rm.total_time_ms) as avg_response_time_ms,
        (COUNT(*) FILTER (WHERE rm.cache_hit = true))::FLOAT / GREATEST(COUNT(*), 1) as cache_hit_rate,
        AVG(rm.relevance_score) as avg_relevance_score,
        (COUNT(*) FILTER (WHERE rm.errors_count > 0))::FLOAT / GREATEST(COUNT(*), 1) as error_rate,
        jsonb_object_agg(
            rm.strategy, 
            COUNT(*)
        ) as top_strategies
    FROM retrieval_metrics rm
    WHERE rm.created_at >= start_time;
END;
$$;

-- ============================================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- ============================================================================

-- Enable RLS on all tables
ALTER TABLE contextual_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE bm25_statistics ENABLE ROW LEVEL SECURITY;
ALTER TABLE corpus_statistics ENABLE ROW LEVEL SECURITY;
ALTER TABLE hybrid_search_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_variations ENABLE ROW LEVEL SECURITY;
ALTER TABLE multi_query_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_filters ENABLE ROW LEVEL SECURITY;
ALTER TABLE metadata_field_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE mmr_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_similarity_cache ENABLE ROW LEVEL SECURITY;
ALTER TABLE retrieval_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_aggregates ENABLE ROW LEVEL SECURITY;

-- Basic RLS policies (adjust based on your authentication system)
-- Allow authenticated users to read all data
CREATE POLICY "Allow authenticated read access" ON contextual_chunks FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON bm25_statistics FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON corpus_statistics FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON hybrid_search_cache FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON query_variations FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON multi_query_results FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON query_filters FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON metadata_field_stats FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON mmr_results FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON document_similarity_cache FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON retrieval_metrics FOR SELECT TO authenticated USING (true);
CREATE POLICY "Allow authenticated read access" ON performance_aggregates FOR SELECT TO authenticated USING (true);

-- Allow service role full access
CREATE POLICY "Allow service role full access" ON contextual_chunks FOR ALL TO service_role USING (true);
CREATE POLICY "Allow service role full access" ON bm25_statistics FOR ALL TO service_role USING (true);
CREATE POLICY "Allow service role full access" ON corpus_statistics FOR ALL TO service_role USING (true);
CREATE POLICY "Allow service role full access" ON hybrid_search_cache FOR ALL TO service_role USING (true);
CREATE POLICY "Allow service role full access" ON query_variations FOR ALL TO service_role USING (true);
CREATE POLICY "Allow service role full access" ON multi_query_results FOR ALL TO service_role USING (true);
CREATE POLICY "Allow service role full access" ON query_filters FOR ALL TO service_role USING (true);
CREATE POLICY "Allow service role full access" ON metadata_field_stats FOR ALL TO service_role USING (true);
CREATE POLICY "Allow service role full access" ON mmr_results FOR ALL TO service_role USING (true);
CREATE POLICY "Allow service role full access" ON document_similarity_cache FOR ALL TO service_role USING (true);
CREATE POLICY "Allow service role full access" ON retrieval_metrics FOR ALL TO service_role USING (true);
CREATE POLICY "Allow service role full access" ON performance_aggregates FOR ALL TO service_role USING (true);

-- ============================================================================
-- TRIGGERS FOR AUTOMATIC MAINTENANCE
-- ============================================================================

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update triggers
CREATE TRIGGER update_contextual_chunks_updated_at BEFORE UPDATE ON contextual_chunks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_bm25_statistics_updated_at BEFORE UPDATE ON bm25_statistics FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Trigger to update corpus statistics when chunks are modified
CREATE OR REPLACE FUNCTION trigger_corpus_stats_update()
RETURNS TRIGGER AS $$
BEGIN
    -- Schedule corpus statistics update (async)
    PERFORM pg_notify('corpus_stats_update', '');
    RETURN COALESCE(NEW, OLD);
END;
$$ language 'plpgsql';

CREATE TRIGGER contextual_chunks_corpus_stats_trigger
    AFTER INSERT OR UPDATE OR DELETE ON contextual_chunks
    FOR EACH STATEMENT
    EXECUTE FUNCTION trigger_corpus_stats_update();

-- ============================================================================
-- PERFORMANCE OPTIMIZATION VIEWS
-- ============================================================================

-- View: Recent performance metrics
CREATE OR REPLACE VIEW recent_performance AS
SELECT 
    strategy,
    COUNT(*) as query_count,
    AVG(total_time_ms) as avg_response_time,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY total_time_ms) as p95_response_time,
    AVG(relevance_score) as avg_relevance,
    AVG(CASE WHEN cache_hit THEN 1.0 ELSE 0.0 END) as cache_hit_rate,
    AVG(documents_retrieved) as avg_documents_retrieved
FROM retrieval_metrics 
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY strategy
ORDER BY query_count DESC;

-- View: Cache performance summary
CREATE OR REPLACE VIEW cache_performance AS
SELECT 
    'hybrid_search' as cache_type,
    COUNT(*) as total_entries,
    AVG(access_count) as avg_access_count,
    COUNT(*) FILTER (WHERE expires_at > NOW()) as active_entries,
    AVG(total_time_ms) as avg_computation_time
FROM hybrid_search_cache
UNION ALL
SELECT 
    'query_variations' as cache_type,
    COUNT(*) as total_entries,
    AVG(access_count) as avg_access_count,
    COUNT(*) FILTER (WHERE expires_at > NOW()) as active_entries,
    AVG(generation_time_ms) as avg_computation_time
FROM query_variations
UNION ALL
SELECT 
    'mmr_results' as cache_type,
    COUNT(*) as total_entries,
    AVG(access_count) as avg_access_count,
    COUNT(*) FILTER (WHERE expires_at > NOW()) as active_entries,
    AVG(computation_time_ms) as avg_computation_time
FROM mmr_results;

-- ============================================================================
-- INITIAL DATA AND SETUP
-- ============================================================================

-- Initialize corpus statistics
INSERT INTO corpus_statistics (
    total_documents,
    average_document_length,
    total_terms,
    unique_terms,
    last_updated,
    version
) VALUES (0, 0.0, 0, 0, NOW(), 1)
ON CONFLICT ((1)) DO NOTHING;

-- Create initial performance aggregate bucket
INSERT INTO performance_aggregates (
    time_bucket,
    total_queries,
    unique_queries,
    last_updated
) VALUES (
    date_trunc('hour', NOW()),
    0,
    0,
    NOW()
) ON CONFLICT (time_bucket) DO NOTHING;

-- ============================================================================
-- COMMENTS AND DOCUMENTATION
-- ============================================================================

COMMENT ON TABLE contextual_chunks IS 'Enhanced chunks with contextual information for improved embedding quality';
COMMENT ON TABLE bm25_statistics IS 'BM25 term frequency and document statistics for sparse search';
COMMENT ON TABLE corpus_statistics IS 'Global corpus statistics for BM25 IDF calculations';
COMMENT ON TABLE hybrid_search_cache IS 'Cache for hybrid search results combining dense and sparse retrieval';
COMMENT ON TABLE query_variations IS 'LLM-generated query variations for multi-query retrieval';
COMMENT ON TABLE multi_query_results IS 'Results from parallel multi-query retrieval execution';
COMMENT ON TABLE query_filters IS 'Extracted metadata filters from natural language queries';
COMMENT ON TABLE metadata_field_stats IS 'Statistics about metadata fields for filter optimization';
COMMENT ON TABLE mmr_results IS 'Maximal Marginal Relevance computation results and cache';
COMMENT ON TABLE document_similarity_cache IS 'Cached document similarity scores for MMR optimization';
COMMENT ON TABLE retrieval_metrics IS 'Comprehensive performance and quality metrics for retrieval operations';
COMMENT ON TABLE performance_aggregates IS 'Hourly aggregated performance metrics for monitoring and analytics';

COMMENT ON FUNCTION search_hybrid IS 'Hybrid search combining dense vector similarity and sparse BM25 keyword matching';
COMMENT ON FUNCTION search_contextual_embeddings IS 'Search contextual embeddings with optional metadata filtering';
COMMENT ON FUNCTION update_corpus_statistics IS 'Update global corpus statistics for BM25 calculations';
COMMENT ON FUNCTION cleanup_expired_cache IS 'Remove expired cache entries from all cache tables';
COMMENT ON FUNCTION get_performance_summary IS 'Get performance summary for specified time range';

-- ============================================================================
-- MIGRATION COMPLETION
-- ============================================================================

-- Log migration completion
DO $$
BEGIN
    RAISE NOTICE 'Contextual Retrieval System database migration completed successfully';
    RAISE NOTICE 'Created % tables with comprehensive indexing and RLS policies', 
        (SELECT COUNT(*) FROM information_schema.tables 
         WHERE table_schema = 'public' 
         AND table_name IN (
             'contextual_chunks', 'bm25_statistics', 'corpus_statistics',
             'hybrid_search_cache', 'query_variations', 'multi_query_results',
             'query_filters', 'metadata_field_stats', 'mmr_results',
             'document_similarity_cache', 'retrieval_metrics', 'performance_aggregates'
         ));
    RAISE NOTICE 'Created % RPC functions for optimized retrieval operations', 5;
    RAISE NOTICE 'Migration timestamp: %', NOW();
END $$; 