-- =====================================================
-- FTI Pipeline Database Schema Migration - Task 4.7
-- Enhanced Feature-Training-Inference Pipeline Tables
-- =====================================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- =====================================================
-- 1. CONTENT PROCESSING TABLES
-- =====================================================

-- Content items with FTI processing metadata
CREATE TABLE IF NOT EXISTS content_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    source_url TEXT,
    author TEXT,
    published_at TIMESTAMP WITH TIME ZONE,
    language VARCHAR(10) DEFAULT 'en',
    
    -- FTI Processing Status
    processing_status VARCHAR(20) DEFAULT 'pending',
    feature_extraction_completed BOOLEAN DEFAULT FALSE,
    training_data_generated BOOLEAN DEFAULT FALSE,
    inference_ready BOOLEAN DEFAULT FALSE,
    
    -- Content Quality Metrics
    quality_score FLOAT DEFAULT 0.0,
    readability_score FLOAT DEFAULT 0.0,
    complexity_score FLOAT DEFAULT 0.0,
    authority_score FLOAT DEFAULT 0.0,
    
    -- Processing Metadata
    chunk_count INTEGER DEFAULT 0,
    embedding_count INTEGER DEFAULT 0,
    metadata_extracted JSONB DEFAULT '{}',
    processing_errors JSONB DEFAULT '[]',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Content chunks with contextual information
CREATE TABLE IF NOT EXISTS content_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_item_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_type VARCHAR(50) DEFAULT 'paragraph',
    
    -- Contextual Information
    context_before TEXT,
    context_after TEXT,
    section_title TEXT,
    document_position FLOAT,
    
    -- Chunk Metadata
    word_count INTEGER,
    character_count INTEGER,
    sentence_count INTEGER,
    chunk_quality_score FLOAT DEFAULT 0.0,
    
    -- Processing Status
    embedding_generated BOOLEAN DEFAULT FALSE,
    metadata_extracted BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Content embeddings with enhanced metadata
CREATE TABLE IF NOT EXISTS content_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_item_id UUID REFERENCES content_items(id) ON DELETE CASCADE,
    chunk_id UUID REFERENCES content_chunks(id) ON DELETE CASCADE,
    
    -- Embedding Data
    embedding vector(1536) NOT NULL,
    embedding_model VARCHAR(100) DEFAULT 'text-embedding-3-small',
    embedding_version VARCHAR(20) DEFAULT '1.0',
    
    -- Contextual Embedding Metadata
    context_window_size INTEGER DEFAULT 0,
    context_overlap INTEGER DEFAULT 0,
    semantic_density FLOAT DEFAULT 0.0,
    
    -- Quality Metrics
    embedding_quality_score FLOAT DEFAULT 0.0,
    retrieval_performance FLOAT DEFAULT 0.0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- 2. TRAINING PIPELINE TABLES
-- =====================================================

-- Training configurations and experiments
CREATE TABLE IF NOT EXISTS training_experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_name VARCHAR(255) NOT NULL,
    experiment_type VARCHAR(50) NOT NULL, -- 'prompt_optimization', 'parameter_tuning', 'model_training'
    
    -- Configuration
    base_configuration JSONB NOT NULL DEFAULT '{}',
    experiment_parameters JSONB NOT NULL DEFAULT '{}',
    training_data_filter JSONB DEFAULT '{}',
    
    -- Status and Results
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'running', 'completed', 'failed'
    progress_percentage FLOAT DEFAULT 0.0,
    results JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    
    -- Resource Usage
    training_duration_seconds INTEGER,
    compute_cost FLOAT DEFAULT 0.0,
    data_points_processed INTEGER DEFAULT 0,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- Optimized model configurations
CREATE TABLE IF NOT EXISTS model_configurations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    configuration_name VARCHAR(255) NOT NULL,
    configuration_type VARCHAR(50) NOT NULL, -- 'prompt', 'retrieval', 'generation'
    
    -- Configuration Data
    configuration_data JSONB NOT NULL DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    validation_results JSONB DEFAULT '{}',
    
    -- Optimization History
    parent_configuration_id UUID REFERENCES model_configurations(id),
    optimization_method VARCHAR(100),
    improvement_percentage FLOAT DEFAULT 0.0,
    
    -- Status
    is_active BOOLEAN DEFAULT FALSE,
    is_production_ready BOOLEAN DEFAULT FALSE,
    validation_status VARCHAR(20) DEFAULT 'pending',
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    deployed_at TIMESTAMP WITH TIME ZONE
);

-- =====================================================
-- 3. INFERENCE PIPELINE TABLES
-- =====================================================

-- Query processing and results
CREATE TABLE IF NOT EXISTS query_processing_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    query_hash VARCHAR(64) NOT NULL,
    query_type VARCHAR(50),
    
    -- Processing Pipeline
    processing_steps JSONB DEFAULT '[]',
    retrieval_results JSONB DEFAULT '{}',
    generation_results JSONB DEFAULT '{}',
    enhancement_results JSONB DEFAULT '{}',
    
    -- Performance Metrics
    total_processing_time_ms INTEGER,
    retrieval_time_ms INTEGER,
    generation_time_ms INTEGER,
    enhancement_time_ms INTEGER,
    
    -- Quality Metrics
    confidence_score FLOAT DEFAULT 0.0,
    relevance_score FLOAT DEFAULT 0.0,
    quality_score FLOAT DEFAULT 0.0,
    user_satisfaction FLOAT,
    
    -- Cache Information
    cache_hit BOOLEAN DEFAULT FALSE,
    cache_key VARCHAR(255),
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    response_delivered_at TIMESTAMP WITH TIME ZONE
);

-- Performance monitoring and analytics
CREATE TABLE IF NOT EXISTS fti_performance_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type VARCHAR(50) NOT NULL, -- 'feature', 'training', 'inference'
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    
    -- Context
    component_name VARCHAR(100),
    experiment_id UUID REFERENCES training_experiments(id),
    configuration_id UUID REFERENCES model_configurations(id),
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    tags JSONB DEFAULT '[]',
    
    -- Timestamps
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =====================================================
-- 4. INDEXES FOR PERFORMANCE
-- =====================================================

-- Content items indexes
CREATE INDEX IF NOT EXISTS idx_content_items_type ON content_items(content_type);
CREATE INDEX IF NOT EXISTS idx_content_items_status ON content_items(processing_status);
CREATE INDEX IF NOT EXISTS idx_content_items_quality ON content_items(quality_score DESC);
CREATE INDEX IF NOT EXISTS idx_content_items_created ON content_items(created_at DESC);

-- Content chunks indexes
CREATE INDEX IF NOT EXISTS idx_content_chunks_item ON content_chunks(content_item_id);
CREATE INDEX IF NOT EXISTS idx_content_chunks_type ON content_chunks(chunk_type);
CREATE INDEX IF NOT EXISTS idx_content_chunks_quality ON content_chunks(chunk_quality_score DESC);

-- Content embeddings indexes
CREATE INDEX IF NOT EXISTS idx_content_embeddings_item ON content_embeddings(content_item_id);
CREATE INDEX IF NOT EXISTS idx_content_embeddings_chunk ON content_embeddings(chunk_id);
CREATE INDEX IF NOT EXISTS idx_content_embeddings_vector ON content_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Training experiments indexes
CREATE INDEX IF NOT EXISTS idx_training_experiments_type ON training_experiments(experiment_type);
CREATE INDEX IF NOT EXISTS idx_training_experiments_status ON training_experiments(status);
CREATE INDEX IF NOT EXISTS idx_training_experiments_created ON training_experiments(created_at DESC);

-- Model configurations indexes
CREATE INDEX IF NOT EXISTS idx_model_configurations_type ON model_configurations(configuration_type);
CREATE INDEX IF NOT EXISTS idx_model_configurations_active ON model_configurations(is_active) WHERE is_active = TRUE;
CREATE INDEX IF NOT EXISTS idx_model_configurations_production ON model_configurations(is_production_ready) WHERE is_production_ready = TRUE;

-- Query processing logs indexes
CREATE INDEX IF NOT EXISTS idx_query_logs_hash ON query_processing_logs(query_hash);
CREATE INDEX IF NOT EXISTS idx_query_logs_type ON query_processing_logs(query_type);
CREATE INDEX IF NOT EXISTS idx_query_logs_created ON query_processing_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_query_logs_confidence ON query_processing_logs(confidence_score DESC);

-- Performance metrics indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_type ON fti_performance_metrics(metric_type, metric_name);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_recorded ON fti_performance_metrics(recorded_at DESC);

-- =====================================================
-- 5. RPC FUNCTIONS FOR FTI OPERATIONS
-- =====================================================

-- Function to process content through FTI pipeline
CREATE OR REPLACE FUNCTION process_content_fti(
    p_title TEXT,
    p_content TEXT,
    p_content_type VARCHAR(50),
    p_source_url TEXT DEFAULT NULL,
    p_author TEXT DEFAULT NULL
) RETURNS UUID AS $$
DECLARE
    content_id UUID;
BEGIN
    -- Insert content item
    INSERT INTO content_items (title, content, content_type, source_url, author)
    VALUES (p_title, p_content, p_content_type, p_source_url, p_author)
    RETURNING id INTO content_id;
    
    -- Log the processing start
    INSERT INTO fti_performance_metrics (metric_type, metric_name, metric_value, metadata)
    VALUES ('feature', 'content_processing_started', 1, jsonb_build_object('content_id', content_id));
    
    RETURN content_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to search content with FTI enhancements
CREATE OR REPLACE FUNCTION search_content_fti(
    p_query_text TEXT,
    p_embedding vector(1536),
    p_limit INTEGER DEFAULT 10,
    p_similarity_threshold FLOAT DEFAULT 0.7
) RETURNS TABLE (
    content_id UUID,
    chunk_id UUID,
    chunk_text TEXT,
    similarity_score FLOAT,
    quality_score FLOAT,
    metadata JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ce.content_item_id,
        ce.chunk_id,
        cc.chunk_text,
        (1 - (ce.embedding <=> p_embedding)) as similarity_score,
        COALESCE(cc.chunk_quality_score, 0.0) as quality_score,
        jsonb_build_object(
            'content_type', ci.content_type,
            'title', ci.title,
            'author', ci.author,
            'chunk_type', cc.chunk_type,
            'section_title', cc.section_title
        ) as metadata
    FROM content_embeddings ce
    JOIN content_chunks cc ON ce.chunk_id = cc.id
    JOIN content_items ci ON ce.content_item_id = ci.id
    WHERE (1 - (ce.embedding <=> p_embedding)) >= p_similarity_threshold
        AND ci.inference_ready = TRUE
    ORDER BY (1 - (ce.embedding <=> p_embedding)) DESC
    LIMIT p_limit;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Function to get FTI pipeline analytics
CREATE OR REPLACE FUNCTION get_fti_analytics(
    p_start_date TIMESTAMP WITH TIME ZONE DEFAULT NOW() - INTERVAL '7 days',
    p_end_date TIMESTAMP WITH TIME ZONE DEFAULT NOW()
) RETURNS JSONB AS $$
DECLARE
    result JSONB;
BEGIN
    SELECT jsonb_build_object(
        'content_processing', jsonb_build_object(
            'total_items', COUNT(*),
            'completed_items', COUNT(*) FILTER (WHERE processing_status = 'completed'),
            'average_quality_score', AVG(quality_score),
            'total_chunks', SUM(chunk_count),
            'total_embeddings', SUM(embedding_count)
        ),
        'training_experiments', (
            SELECT jsonb_build_object(
                'total_experiments', COUNT(*),
                'completed_experiments', COUNT(*) FILTER (WHERE status = 'completed'),
                'average_duration_minutes', AVG(training_duration_seconds) / 60.0,
                'total_compute_cost', SUM(compute_cost)
            )
            FROM training_experiments
            WHERE created_at BETWEEN p_start_date AND p_end_date
        ),
        'inference_performance', (
            SELECT jsonb_build_object(
                'total_queries', COUNT(*),
                'average_processing_time_ms', AVG(total_processing_time_ms),
                'average_confidence_score', AVG(confidence_score),
                'cache_hit_rate', AVG(CASE WHEN cache_hit THEN 1.0 ELSE 0.0 END)
            )
            FROM query_processing_logs
            WHERE created_at BETWEEN p_start_date AND p_end_date
        )
    ) INTO result
    FROM content_items
    WHERE created_at BETWEEN p_start_date AND p_end_date;
    
    RETURN result;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- =====================================================
-- 6. TRIGGERS FOR AUTOMATIC UPDATES
-- =====================================================

-- Update timestamps trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply triggers to all tables
CREATE TRIGGER update_content_items_updated_at BEFORE UPDATE ON content_items FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_content_chunks_updated_at BEFORE UPDATE ON content_chunks FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_content_embeddings_updated_at BEFORE UPDATE ON content_embeddings FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_training_experiments_updated_at BEFORE UPDATE ON training_experiments FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_model_configurations_updated_at BEFORE UPDATE ON model_configurations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- 7. ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE content_items ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE content_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_experiments ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_configurations ENABLE ROW LEVEL SECURITY;
ALTER TABLE query_processing_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE fti_performance_metrics ENABLE ROW LEVEL SECURITY;

-- Service role policies (full access)
CREATE POLICY "Service role can manage content_items" ON content_items FOR ALL TO service_role USING (true);
CREATE POLICY "Service role can manage content_chunks" ON content_chunks FOR ALL TO service_role USING (true);
CREATE POLICY "Service role can manage content_embeddings" ON content_embeddings FOR ALL TO service_role USING (true);
CREATE POLICY "Service role can manage training_experiments" ON training_experiments FOR ALL TO service_role USING (true);
CREATE POLICY "Service role can manage model_configurations" ON model_configurations FOR ALL TO service_role USING (true);
CREATE POLICY "Service role can manage query_processing_logs" ON query_processing_logs FOR ALL TO service_role USING (true);
CREATE POLICY "Service role can manage fti_performance_metrics" ON fti_performance_metrics FOR ALL TO service_role USING (true);

-- Anonymous role policies (read-only for public content)
CREATE POLICY "Anonymous can read public content_items" ON content_items FOR SELECT TO anon USING (true);
CREATE POLICY "Anonymous can read public content_chunks" ON content_chunks FOR SELECT TO anon USING (true);
CREATE POLICY "Anonymous can read public content_embeddings" ON content_embeddings FOR SELECT TO anon USING (true);

-- =====================================================
-- 8. INITIAL DATA AND CONFIGURATION
-- =====================================================

-- Insert default model configurations
INSERT INTO model_configurations (configuration_name, configuration_type, configuration_data, is_active, is_production_ready)
VALUES 
    ('default_prompt_config', 'prompt', '{"system_prompt": "You are a helpful AI assistant.", "temperature": 0.7, "max_tokens": 1000}', true, true),
    ('default_retrieval_config', 'retrieval', '{"top_k": 10, "similarity_threshold": 0.7, "mmr_lambda": 0.7}', true, true),
    ('default_generation_config', 'generation', '{"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000}', true, true)
ON CONFLICT DO NOTHING;

-- Create initial performance baseline
INSERT INTO fti_performance_metrics (metric_type, metric_name, metric_value, metadata)
VALUES 
    ('system', 'fti_pipeline_initialized', 1, '{"version": "1.0", "migration": "005_fti_pipeline_schema"}'),
    ('system', 'baseline_performance_target', 500, '{"unit": "milliseconds", "metric": "average_response_time"}')
ON CONFLICT DO NOTHING;

COMMIT; 