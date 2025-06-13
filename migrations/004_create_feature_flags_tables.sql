-- Feature Flags and A/B Testing Infrastructure Migration
-- Creates tables for advanced feature flags, A/B testing, and experiment analytics

-- Create feature_flags table
CREATE TABLE IF NOT EXISTS feature_flags (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    description TEXT,
    status TEXT NOT NULL CHECK (status IN ('disabled', 'enabled', 'gradual_rollout', 'ab_test', 'canary')),
    rollout_percentage FLOAT DEFAULT 0.0 CHECK (rollout_percentage >= 0.0 AND rollout_percentage <= 100.0),
    segments JSONB DEFAULT '{}',
    variants JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

-- Create indexes for feature_flags
CREATE INDEX IF NOT EXISTS idx_feature_flags_name ON feature_flags(name);
CREATE INDEX IF NOT EXISTS idx_feature_flags_status ON feature_flags(status);
CREATE INDEX IF NOT EXISTS idx_feature_flags_expires ON feature_flags(expires_at) WHERE expires_at IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_feature_flags_updated ON feature_flags(updated_at DESC);

-- Create ab_test_experiments table
CREATE TABLE IF NOT EXISTS ab_test_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    feature_flag_id UUID REFERENCES feature_flags(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    hypothesis TEXT,
    success_metrics JSONB DEFAULT '[]',
    start_date TIMESTAMPTZ DEFAULT NOW(),
    end_date TIMESTAMPTZ,
    status TEXT DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'cancelled')),
    results JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for ab_test_experiments
CREATE INDEX IF NOT EXISTS idx_experiments_flag ON ab_test_experiments(feature_flag_id);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON ab_test_experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiments_dates ON ab_test_experiments(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_experiments_created ON ab_test_experiments(created_at DESC);

-- Create ab_test_metrics table
CREATE TABLE IF NOT EXISTS ab_test_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    experiment_id UUID REFERENCES ab_test_experiments(id) ON DELETE CASCADE,
    variant_name TEXT NOT NULL,
    user_id TEXT,
    session_id TEXT,
    metric_type TEXT NOT NULL,
    metric_value FLOAT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for ab_test_metrics
CREATE INDEX IF NOT EXISTS idx_metrics_experiment ON ab_test_metrics(experiment_id, variant_name);
CREATE INDEX IF NOT EXISTS idx_metrics_type ON ab_test_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_metrics_time ON ab_test_metrics(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_metrics_user ON ab_test_metrics(user_id) WHERE user_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_metrics_session ON ab_test_metrics(session_id) WHERE session_id IS NOT NULL;

-- Add trigger to update updated_at on feature_flags
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_feature_flags_updated_at 
    BEFORE UPDATE ON feature_flags 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Add comments for documentation
COMMENT ON TABLE feature_flags IS 'Advanced feature flags with segmentation, A/B testing, and gradual rollout capabilities';
COMMENT ON TABLE ab_test_experiments IS 'A/B test experiment configurations and metadata';
COMMENT ON TABLE ab_test_metrics IS 'Metrics tracking for A/B test experiments';

COMMENT ON COLUMN feature_flags.name IS 'Unique feature flag identifier';
COMMENT ON COLUMN feature_flags.description IS 'Human-readable description of the feature';
COMMENT ON COLUMN feature_flags.status IS 'Feature flag status: disabled, enabled, gradual_rollout, ab_test, canary';
COMMENT ON COLUMN feature_flags.rollout_percentage IS 'Percentage of users to include (0-100)';
COMMENT ON COLUMN feature_flags.segments IS 'User segmentation configuration (strategy type, criteria, etc.)';
COMMENT ON COLUMN feature_flags.variants IS 'A/B test variants with weights and config overrides';
COMMENT ON COLUMN feature_flags.metadata IS 'Additional metadata for the feature flag';
COMMENT ON COLUMN feature_flags.expires_at IS 'Optional expiration timestamp for temporary flags';

COMMENT ON COLUMN ab_test_experiments.feature_flag_id IS 'Reference to the feature flag being tested';
COMMENT ON COLUMN ab_test_experiments.name IS 'Experiment name for identification';
COMMENT ON COLUMN ab_test_experiments.hypothesis IS 'Hypothesis being tested';
COMMENT ON COLUMN ab_test_experiments.success_metrics IS 'Array of metrics to track for success';
COMMENT ON COLUMN ab_test_experiments.status IS 'Experiment status: active, paused, completed, cancelled';
COMMENT ON COLUMN ab_test_experiments.results IS 'Final experiment results and statistical analysis';

COMMENT ON COLUMN ab_test_metrics.experiment_id IS 'Reference to the experiment';
COMMENT ON COLUMN ab_test_metrics.variant_name IS 'Name of the variant (control, treatment, etc.)';
COMMENT ON COLUMN ab_test_metrics.user_id IS 'User identifier (optional)';
COMMENT ON COLUMN ab_test_metrics.session_id IS 'Session identifier (optional)';
COMMENT ON COLUMN ab_test_metrics.metric_type IS 'Type of metric (impression, conversion, error, response_time_ms, quality_score, etc.)';
COMMENT ON COLUMN ab_test_metrics.metric_value IS 'Numeric value of the metric';
COMMENT ON COLUMN ab_test_metrics.metadata IS 'Additional context for the metric'; 