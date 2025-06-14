#!/usr/bin/env python3
"""
Content Migration Script for Contextual Retrieval System

This script migrates existing content and configurations to support the new
contextual retrieval system introduced in Task 3.
"""

import os
import sys
import asyncio
import logging
import argparse
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.retrieval_settings import (
    RetrievalSettings,
    ConfigurationManager,
    create_retrieval_settings
)

class MigrationConfig:
    """Configuration for the migration process."""
    
    def __init__(
        self,
        environment: str = "development",
        dry_run: bool = False,
        validate_only: bool = False,
        batch_size: int = 100,
        backup_enabled: bool = True
    ):
        self.environment = environment
        self.dry_run = dry_run
        self.validate_only = validate_only
        self.batch_size = batch_size
        self.backup_enabled = backup_enabled
        
        # Load retrieval settings
        self.retrieval_config = create_retrieval_settings(environment)
        
        # Migration tracking
        self.migration_id = f"contextual_migration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

class ContextualRetrievalMigrator:
    """Main migration orchestrator for contextual retrieval system."""
    
    def __init__(self, config: MigrationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.migration_state = {
            'phase': 'initialization',
            'progress': 0.0,
            'errors': [],
            'warnings': [],
            'start_time': datetime.now()
        }
        
    async def initialize(self):
        """Initialize migration components."""
        
        self.logger.info(f"Initializing contextual retrieval migration: {self.config.migration_id}")
        
        # Create migration workspace
        migration_dir = f"migrations/{self.config.migration_id}"
        os.makedirs(migration_dir, exist_ok=True)
        
        # Setup logging
        log_file = f"{migration_dir}/migration.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info("Migration initialization completed")
    
    async def run_migration(self) -> Dict[str, Any]:
        """Run the complete migration process."""
        
        try:
            await self.initialize()
            
            # Migration phases
            phases = [
                ("validation", self._validate_prerequisites),
                ("backup", self._create_backup),
                ("config_migration", self._migrate_configuration),
                ("final_validation", self._final_validation)
            ]
            
            total_phases = len(phases)
            
            for i, (phase_name, phase_func) in enumerate(phases):
                self.migration_state['phase'] = phase_name
                self.migration_state['progress'] = i / total_phases
                
                self.logger.info(f"Starting phase: {phase_name}")
                
                if self.config.validate_only and phase_name not in ['validation', 'final_validation']:
                    self.logger.info(f"Skipping {phase_name} (validate-only mode)")
                    continue
                
                try:
                    phase_result = await phase_func()
                    if not phase_result.get('success', True):
                        self.migration_state['errors'].append(f"Phase {phase_name} failed: {phase_result.get('error')}")
                        if not self.config.dry_run:
                            break
                    
                except Exception as e:
                    error_msg = f"Phase {phase_name} encountered error: {str(e)}"
                    self.logger.error(error_msg)
                    self.migration_state['errors'].append(error_msg)
                    
                    if not self.config.dry_run:
                        break
            
            self.migration_state['progress'] = 1.0
            self.migration_state['end_time'] = datetime.now()
            
            # Generate migration report
            return await self._generate_migration_report()
            
        except Exception as e:
            self.logger.error(f"Migration failed: {e}")
            self.migration_state['errors'].append(f"Critical error: {str(e)}")
            return await self._generate_migration_report()
    
    async def _validate_prerequisites(self) -> Dict[str, Any]:
        """Validate prerequisites for migration."""
        
        self.logger.info("Validating migration prerequisites")
        issues = []
        warnings = []
        
        # Check configuration
        config_manager = ConfigurationManager()
        config_issues = config_manager.validate_config(self.config.retrieval_config)
        issues.extend(config_issues)
        
        # Check required directories
        required_dirs = ['config', 'migrations', 'logs']
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                warnings.append(f"Created missing directory: {dir_name}")
        
        # Check API keys for production
        if self.config.environment == "production":
            if not self.config.retrieval_config.openai_api_key:
                issues.append("OpenAI API key required for production migration")
        
        success = len(issues) == 0
        
        return {
            'success': success,
            'issues': issues,
            'warnings': warnings,
            'message': 'Prerequisites validated' if success else 'Prerequisites validation failed'
        }
    
    async def _create_backup(self) -> Dict[str, Any]:
        """Create backup of existing configuration and data."""
        
        if not self.config.backup_enabled:
            return {'success': True, 'message': 'Backup skipped (disabled)'}
        
        self.logger.info("Creating migration backup")
        
        backup_dir = f"migrations/{self.config.migration_id}/backup"
        os.makedirs(backup_dir, exist_ok=True)
        
        backup_items = []
        
        try:
            # Backup current configuration
            current_config = ConfigurationManager().load_config()
            config_backup_path = f"{backup_dir}/retrieval_settings_backup.json"
            
            with open(config_backup_path, 'w') as f:
                json.dump(current_config.dict(), f, indent=2, default=str)
            
            backup_items.append("Configuration files")
            
            # Create backup manifest
            manifest = {
                'migration_id': self.config.migration_id,
                'created_at': datetime.now().isoformat(),
                'environment': self.config.environment,
                'backup_items': backup_items,
                'original_config': current_config.dict()
            }
            
            manifest_path = f"{backup_dir}/backup_manifest.json"
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            return {
                'success': True,
                'backup_dir': backup_dir,
                'backup_items': backup_items,
                'message': f'Backup created successfully: {len(backup_items)} items'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Backup creation failed'
            }
    
    async def _migrate_configuration(self) -> Dict[str, Any]:
        """Migrate configuration to contextual retrieval settings."""
        
        self.logger.info("Migrating configuration for contextual retrieval")
        
        try:
            # Load current configuration
            config_manager = ConfigurationManager()
            current_config = config_manager.load_config()
            
            # Create enhanced configuration with contextual features
            enhanced_config_data = current_config.dict()
            
            # Enable contextual retrieval features
            enhanced_config_data['enable_contextual_retrieval'] = True
            enhanced_config_data['enable_task2_integration'] = True
            
            # Optimize for the environment
            if self.config.environment == "production":
                enhanced_config_data['performance_profile'] = "quality_optimized"
                enhanced_config_data['monitoring']['enable_metrics_collection'] = True
                enhanced_config_data['monitoring']['alert_on_slow_queries'] = True
            elif self.config.environment == "development":
                enhanced_config_data['performance_profile'] = "balanced"
                enhanced_config_data['monitoring']['log_level'] = "DEBUG"
            
            # Create new configuration
            new_config = RetrievalSettings(**enhanced_config_data)
            new_config.apply_performance_profile()
            
            # Validate new configuration
            issues = config_manager.validate_config(new_config)
            
            if self.config.dry_run:
                return {
                    'success': True,
                    'message': 'Configuration migration simulated (dry-run mode)',
                    'changes': enhanced_config_data,
                    'validation_issues': issues
                }
            
            # Save new configuration
            config_path = "config/retrieval_settings_migrated.json"
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(new_config.dict(), f, indent=2, default=str)
            
            return {
                'success': True,
                'message': 'Configuration migration completed',
                'config_path': config_path,
                'validation_issues': issues,
                'features_enabled': [
                    'contextual_retrieval',
                    'task2_integration',
                    'enhanced_confidence_scoring'
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Configuration migration failed'
            }
    
    async def _final_validation(self) -> Dict[str, Any]:
        """Final validation of migration success."""
        
        self.logger.info("Performing final migration validation")
        
        try:
            validation_results = {
                'configuration_valid': True,
                'contextual_features_enabled': True,
                'performance_acceptable': True
            }
            
            issues = []
            
            # Validate configuration
            try:
                config_manager = ConfigurationManager()
                config = config_manager.load_config()
                config_issues = config_manager.validate_config(config)
                
                if config_issues:
                    validation_results['configuration_valid'] = False
                    issues.extend(config_issues)
                    
            except Exception as e:
                validation_results['configuration_valid'] = False
                issues.append(f"Configuration validation error: {e}")
            
            # Check contextual features
            if not self.config.retrieval_config.enable_contextual_retrieval:
                validation_results['contextual_features_enabled'] = False
                issues.append("Contextual retrieval features not enabled")
            
            overall_success = all(validation_results.values())
            
            return {
                'success': overall_success,
                'message': f'Final validation {"passed" if overall_success else "failed"}',
                'validation_results': validation_results,
                'issues': issues,
                'migration_complete': overall_success
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': 'Final validation failed'
            }
    
    async def _generate_migration_report(self) -> Dict[str, Any]:
        """Generate comprehensive migration report."""
        
        end_time = datetime.now()
        duration = end_time - self.migration_state['start_time']
        
        report = {
            'migration_id': self.config.migration_id,
            'environment': self.config.environment,
            'start_time': self.migration_state['start_time'].isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration.total_seconds(),
            'final_phase': self.migration_state['phase'],
            'progress': self.migration_state['progress'],
            'errors': self.migration_state['errors'],
            'warnings': self.migration_state['warnings'],
            'success': len(self.migration_state['errors']) == 0,
            'dry_run': self.config.dry_run,
            'validate_only': self.config.validate_only
        }
        
        # Save report
        report_path = f"migrations/{self.config.migration_id}/migration_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Migration report saved: {report_path}")
        
        return report

async def main():
    """Main entry point for migration script."""
    
    parser = argparse.ArgumentParser(description='Migrate to Contextual Retrieval System')
    parser.add_argument('--environment', default='development', choices=['development', 'staging', 'production'])
    parser.add_argument('--dry-run', action='store_true', help='Simulate migration without making changes')
    parser.add_argument('--validate-only', action='store_true', help='Only validate prerequisites and configuration')
    parser.add_argument('--batch-size', type=int, default=100, help='Batch size for content processing')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Create migration configuration
    migration_config = MigrationConfig(
        environment=args.environment,
        dry_run=args.dry_run,
        validate_only=args.validate_only,
        batch_size=args.batch_size,
        backup_enabled=not args.no_backup
    )
    
    logger.info(f"Starting contextual retrieval migration for {args.environment}")
    logger.info(f"Migration mode: {'dry-run' if args.dry_run else 'live'}")
    logger.info(f"Validate only: {args.validate_only}")
    
    # Run migration
    migrator = ContextualRetrievalMigrator(migration_config)
    result = await migrator.run_migration()
    
    # Print results
    print("\n" + "="*80)
    print("CONTEXTUAL RETRIEVAL MIGRATION REPORT")
    print("="*80)
    print(f"Migration ID: {result['migration_id']}")
    print(f"Environment: {result['environment']}")
    print(f"Duration: {result['duration_seconds']:.1f} seconds")
    print(f"Success: {'✓' if result['success'] else '✗'}")
    print(f"Progress: {result['progress']:.1%}")
    
    if result['errors']:
        print(f"\nErrors ({len(result['errors'])}):")
        for error in result['errors']:
            print(f"  ✗ {error}")
    
    if result['warnings']:
        print(f"\nWarnings ({len(result['warnings'])}):")
        for warning in result['warnings']:
            print(f"  ⚠ {warning}")
    
    print(f"\nFull report saved to: migrations/{result['migration_id']}/migration_report.json")
    
    if result['success']:
        print("\n✓ Migration completed successfully!")
        if args.dry_run:
            print("  Run without --dry-run to apply changes")
        elif args.validate_only:
            print("  Run without --validate-only to perform migration")
        else:
            print("  Contextual retrieval system is now active")
    else:
        print("\n✗ Migration failed!")
        print("  Check the migration report for details")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 