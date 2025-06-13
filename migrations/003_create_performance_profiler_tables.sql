-- Performance Profiler Tables Migration
-- Creates tables for storing performance profiles and bottleneck statistics

-- Create performance_profiles table
CREATE TABLE IF NOT EXISTS performance_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_id TEXT NOT NULL,
    profile_data JSONB NOT NULL,
    total_duration_ms FLOAT NOT NULL,
    bottleneck_operations JSONB DEFAULT '[]',
    optimization_suggestions JSONB DEFAULT '[]',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance_profiles
CREATE INDEX IF NOT EXISTS idx_profiles_query ON performance_profiles(query_id);
CREATE INDEX IF NOT EXISTS idx_profiles_duration ON performance_profiles(total_duration_ms DESC);
CREATE INDEX IF NOT EXISTS idx_profiles_time ON performance_profiles(created_at DESC);

-- Create performance_bottlenecks table
CREATE TABLE IF NOT EXISTS performance_bottlenecks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    operation_type TEXT NOT NULL,
    avg_duration_ms FLOAT NOT NULL,
    p95_duration_ms FLOAT NOT NULL,
    p99_duration_ms FLOAT NOT NULL,
    occurrence_count INTEGER NOT NULL,
    impact_score FLOAT NOT NULL, -- 0-100 scale
    suggested_optimizations JSONB DEFAULT '[]',
    detected_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for performance_bottlenecks
CREATE INDEX IF NOT EXISTS idx_bottlenecks_impact ON performance_bottlenecks(impact_score DESC);
CREATE INDEX IF NOT EXISTS idx_bottlenecks_type ON performance_bottlenecks(operation_type);
CREATE INDEX IF NOT EXISTS idx_bottlenecks_time ON performance_bottlenecks(detected_at DESC);

-- Add comments for documentation
COMMENT ON TABLE performance_profiles IS 'Stores detailed performance profiles for individual operations with timing data and optimization suggestions';
COMMENT ON TABLE performance_bottlenecks IS 'Stores aggregated bottleneck statistics and optimization recommendations for operation types';

COMMENT ON COLUMN performance_profiles.query_id IS 'Unique identifier for the query or operation being profiled';
COMMENT ON COLUMN performance_profiles.profile_data IS 'Complete timing hierarchy with nested operations and metadata';
COMMENT ON COLUMN performance_profiles.total_duration_ms IS 'Total operation duration in milliseconds';
COMMENT ON COLUMN performance_profiles.bottleneck_operations IS 'Array of identified bottleneck operations with timing details';
COMMENT ON COLUMN performance_profiles.optimization_suggestions IS 'Array of specific optimization recommendations';

COMMENT ON COLUMN performance_bottlenecks.operation_type IS 'Type of operation (e.g., retrieval, embedding_generation, llm_generation)';
COMMENT ON COLUMN performance_bottlenecks.avg_duration_ms IS 'Average duration in milliseconds across all occurrences';
COMMENT ON COLUMN performance_bottlenecks.p95_duration_ms IS '95th percentile duration in milliseconds';
COMMENT ON COLUMN performance_bottlenecks.p99_duration_ms IS '99th percentile duration in milliseconds';
COMMENT ON COLUMN performance_bottlenecks.occurrence_count IS 'Number of times this operation type has been profiled';
COMMENT ON COLUMN performance_bottlenecks.impact_score IS 'Performance impact score (0-100) based on frequency, duration, and variance';
COMMENT ON COLUMN performance_bottlenecks.suggested_optimizations IS 'Array of operation-specific optimization recommendations'; 