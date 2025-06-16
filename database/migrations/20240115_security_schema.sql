-- ==============================================
-- Universal RAG CMS Security Schema Migration
-- Task 11: Security and Compliance Tables
-- ==============================================

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ==============================================
-- SECURITY AUDIT LOG TABLE
-- ==============================================

CREATE TABLE IF NOT EXISTS security_audit_log (
    id TEXT PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    session_id TEXT,
    action_type VARCHAR(50) NOT NULL,
    resource_type VARCHAR(50),
    resource_id TEXT,
    ip_address INET,
    user_agent TEXT,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_audit_user_id ON security_audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_action_type ON security_audit_log(action_type);
CREATE INDEX IF NOT EXISTS idx_audit_created_at ON security_audit_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_user_time ON security_audit_log(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_action_time ON security_audit_log(action_type, created_at DESC);

-- ==============================================
-- API KEY MANAGEMENT
-- ==============================================

CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    service_name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    permissions TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_rotated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_used TIMESTAMP WITH TIME ZONE,
    usage_count INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    deactivated_at TIMESTAMP WITH TIME ZONE,
    revocation_reason TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_user_id ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_active ON api_keys(is_active, expires_at);

-- API Key rotation tracking
CREATE TABLE IF NOT EXISTS api_key_rotations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    old_key_id UUID REFERENCES api_keys(id),
    new_key_id UUID REFERENCES api_keys(id),
    rotated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==============================================
-- USER ROLES AND PERMISSIONS
-- ==============================================

CREATE TABLE IF NOT EXISTS user_roles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    role_name VARCHAR(50) NOT NULL,
    granted_by UUID REFERENCES auth.users(id),
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(user_id, role_name)
);

CREATE TABLE IF NOT EXISTS user_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    permission VARCHAR(100) NOT NULL,
    granted_by UUID REFERENCES auth.users(id),
    granted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, permission)
);

CREATE TABLE IF NOT EXISTS role_permissions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    role_name VARCHAR(50) NOT NULL,
    permission VARCHAR(100) NOT NULL,
    UNIQUE(role_name, permission)
);

-- ==============================================
-- SECURITY EVENTS
-- ==============================================

CREATE TABLE IF NOT EXISTS security_events (
    id TEXT PRIMARY KEY,
    violation_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    user_id UUID REFERENCES auth.users(id),
    ip_address INET,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_security_events_type ON security_events(violation_type);
CREATE INDEX IF NOT EXISTS idx_security_events_severity ON security_events(severity);
CREATE INDEX IF NOT EXISTS idx_security_events_created ON security_events(created_at DESC);

-- ==============================================
-- GDPR COMPLIANCE
-- ==============================================

CREATE TABLE IF NOT EXISTS gdpr_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    request_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    requested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    data_export_path TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS gdpr_rectifications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    request_id UUID REFERENCES gdpr_requests(id),
    user_id UUID REFERENCES auth.users(id),
    corrections JSONB NOT NULL,
    applied_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_consents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    consent_type VARCHAR(100) NOT NULL,
    granted BOOLEAN NOT NULL,
    details JSONB,
    ip_address INET,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- ==============================================
-- CONTENT MODERATION
-- ==============================================

CREATE TABLE IF NOT EXISTS content_moderation_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_type VARCHAR(50) NOT NULL,
    content_hash VARCHAR(255) NOT NULL,
    moderation_result JSONB NOT NULL,
    action_taken VARCHAR(50),
    reviewed_by UUID REFERENCES auth.users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for duplicate content detection
CREATE INDEX IF NOT EXISTS idx_content_hash ON content_moderation_log(content_hash);

-- ==============================================
-- SESSION MANAGEMENT
-- ==============================================

CREATE TABLE IF NOT EXISTS user_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id),
    session_token VARCHAR(255) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_sessions_token ON user_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id, is_active);

-- ==============================================
-- SECURITY PREFERENCES
-- ==============================================

CREATE TABLE IF NOT EXISTS user_security_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES auth.users(id) UNIQUE,
    mfa_enabled BOOLEAN DEFAULT FALSE,
    mfa_secret TEXT,
    backup_codes TEXT[],
    login_notifications BOOLEAN DEFAULT TRUE,
    security_alerts BOOLEAN DEFAULT TRUE,
    session_timeout_minutes INTEGER DEFAULT 30,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ==============================================
-- ROW LEVEL SECURITY POLICIES
-- ==============================================

-- Enable RLS
ALTER TABLE security_audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_roles ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_permissions ENABLE ROW LEVEL SECURITY;
ALTER TABLE security_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE gdpr_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_consents ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_security_preferences ENABLE ROW LEVEL SECURITY;

-- Audit log policies
CREATE POLICY "Users view own audit logs" ON security_audit_log
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Admins view all audit logs" ON security_audit_log
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM user_roles 
            WHERE user_id = auth.uid() 
            AND role_name IN ('super_admin', 'admin')
        )
    );

CREATE POLICY "Service role full access to audit logs" ON security_audit_log
    FOR ALL USING (auth.role() = 'service_role');

-- API key policies
CREATE POLICY "Users manage own API keys" ON api_keys
    FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Service role full access to API keys" ON api_keys
    FOR ALL USING (auth.role() = 'service_role');

-- Role management policies
CREATE POLICY "Users view own roles" ON user_roles
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Admins manage all roles" ON user_roles
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM user_roles 
            WHERE user_id = auth.uid() 
            AND role_name IN ('super_admin', 'admin')
        )
    );

-- Permission policies
CREATE POLICY "Users view own permissions" ON user_permissions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Admins manage all permissions" ON user_permissions
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM user_roles 
            WHERE user_id = auth.uid() 
            AND role_name IN ('super_admin', 'admin')
        )
    );

-- Security events policies
CREATE POLICY "Admins view security events" ON security_events
    FOR SELECT USING (
        EXISTS (
            SELECT 1 FROM user_roles 
            WHERE user_id = auth.uid() 
            AND role_name IN ('super_admin', 'admin', 'moderator')
        )
    );

CREATE POLICY "Service role full access to security events" ON security_events
    FOR ALL USING (auth.role() = 'service_role');

-- GDPR request policies
CREATE POLICY "Users view own GDPR requests" ON gdpr_requests
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users create own GDPR requests" ON gdpr_requests
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Admins manage all GDPR requests" ON gdpr_requests
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM user_roles 
            WHERE user_id = auth.uid() 
            AND role_name IN ('super_admin', 'admin')
        )
    );

-- Session policies
CREATE POLICY "Users manage own sessions" ON user_sessions
    FOR ALL USING (auth.uid() = user_id);

-- Security preferences policies
CREATE POLICY "Users manage own security preferences" ON user_security_preferences
    FOR ALL USING (auth.uid() = user_id);

-- ==============================================
-- INSERT DEFAULT ROLE PERMISSIONS
-- ==============================================

INSERT INTO role_permissions (role_name, permission) VALUES
-- Super Admin (all permissions)
('super_admin', 'content:create'),
('super_admin', 'content:read'),
('super_admin', 'content:update'),
('super_admin', 'content:delete'),
('super_admin', 'user:create'),
('super_admin', 'user:read'),
('super_admin', 'user:update'),
('super_admin', 'user:delete'),
('super_admin', 'system:config'),
('super_admin', 'system:monitor'),
('super_admin', 'system:audit'),
('super_admin', 'api:full'),
('super_admin', 'api:no_limit'),

-- Admin
('admin', 'content:create'),
('admin', 'content:read'),
('admin', 'content:update'),
('admin', 'content:delete'),
('admin', 'user:create'),
('admin', 'user:read'),
('admin', 'user:update'),
('admin', 'system:monitor'),
('admin', 'api:full'),

-- Moderator
('moderator', 'content:read'),
('moderator', 'content:update'),
('moderator', 'user:read'),
('moderator', 'system:monitor'),

-- Editor
('editor', 'content:create'),
('editor', 'content:read'),
('editor', 'content:update'),

-- Viewer
('viewer', 'content:read'),

-- API User
('api_user', 'content:read'),
('api_user', 'api:read')

ON CONFLICT (role_name, permission) DO NOTHING;

-- ==============================================
-- TRIGGERS FOR UPDATED_AT
-- ==============================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers where needed
CREATE TRIGGER update_user_security_preferences_updated_at 
    BEFORE UPDATE ON user_security_preferences 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ==============================================
-- FUNCTIONS FOR CLEANUP
-- ==============================================

-- Function to cleanup expired sessions
CREATE OR REPLACE FUNCTION cleanup_expired_sessions()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM user_sessions 
    WHERE expires_at < NOW() OR 
          (last_activity < NOW() - INTERVAL '24 hours' AND is_active = false);
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to cleanup old audit logs (keep 7 years by default)
CREATE OR REPLACE FUNCTION cleanup_old_audit_logs(retention_days INTEGER DEFAULT 2555)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM security_audit_log 
    WHERE created_at < NOW() - INTERVAL '1 day' * retention_days;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- ==============================================
-- COMMENTS FOR DOCUMENTATION
-- ==============================================

COMMENT ON TABLE security_audit_log IS 'Comprehensive audit trail for all system actions';
COMMENT ON TABLE api_keys IS 'API key management with rotation and permissions';
COMMENT ON TABLE user_roles IS 'User role assignments with expiration support';
COMMENT ON TABLE security_events IS 'Security violations and threat detection log';
COMMENT ON TABLE gdpr_requests IS 'GDPR compliance requests tracking';
COMMENT ON TABLE content_moderation_log IS 'Content moderation decisions and actions';

-- ==============================================
-- COMPLETION MESSAGE
-- ==============================================

DO $$
BEGIN
    RAISE NOTICE '✅ Security schema migration completed successfully!';
    RAISE NOTICE '🔒 Created % security tables with RLS policies', 
        (SELECT COUNT(*) FROM information_schema.tables 
         WHERE table_name IN ('security_audit_log', 'api_keys', 'user_roles', 
                              'security_events', 'gdpr_requests', 'user_consents',
                              'content_moderation_log', 'user_sessions'));
    RAISE NOTICE '🛡️ Security system ready for initialization';
END $$; 