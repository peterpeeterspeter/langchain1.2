#!/usr/bin/env python3
"""
Example: Advanced Configuration Management for Universal RAG CMS

This example demonstrates detailed usage of the ConfigurationManager for:
- Saving new configurations
- Validating configurations
- Retrieving configuration history
- Rolling back to previous configuration versions

Prerequisites:
- Set environment variables: SUPABASE_URL, SUPABASE_KEY
- Run database migrations if not already done
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from supabase import create_client
from src.config.prompt_config import ConfigurationManager, PromptOptimizationConfig, QueryType
from src.utils.integration_helpers import ConfigurationValidator

async def main():
    """Main example function demonstrating advanced ConfigurationManager usage."""
    
    print("🚀 Advanced Configuration Management Example")
    print("=" * 70)
    
    # Step 1: Validate environment
    print("
1. Validating Environment...")
    is_valid, issues = ConfigurationValidator.validate_runtime_environment()
    
    if not is_valid:
        print("❌ Environment validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        print("
Please set SUPABASE_URL and SUPABASE_KEY environment variables.")
        return
    
    print("✅ Environment validated successfully")
    
    # Step 2: Initialize ConfigurationManager
    print("
2. Initializing ConfigurationManager...")
    try:
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("Supabase URL and Key environment variables are not set.")
            
        supabase_client = create_client(supabase_url, supabase_key)
        config_manager = ConfigurationManager(supabase_client)
        print("✅ ConfigurationManager initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize ConfigurationManager: {e}")
        print("Note: This example requires a working Supabase instance with the 'prompt_configurations' table.")
        return
    
    # Step 3: Get current active configuration
    print("
3. Retrieving Current Active Configuration...")
    try:
        initial_config = await config_manager.get_active_config()
        print(f"📝 Initial active config version: {initial_config.version}")
        print(f"   - Cache TTL (News): {initial_config.cache_config.get_ttl(QueryType.NEWS)}h")
        print(f"   - Response Time Warning: {initial_config.performance.response_time_warning_ms}ms")
    except Exception as e:
        print(f"❌ Error retrieving initial config: {e}")
        return
    
    # Step 4: Save a new configuration version
    print("
4. Saving a New Configuration Version...")
    new_config_data = initial_config.model_dump()
    new_config_data["cache_config"]["news_ttl"] = 4 # Update news TTL to 4 hours
    new_config_data["performance"]["response_time_warning_ms"] = 1200 # Reduce warning threshold

    try:
        # Create a new config object from modified data
        new_config = PromptOptimizationConfig.model_validate(new_config_data)
        new_config.version = "1.0.1"
        new_config.change_notes = "Updated news cache TTL and reduced response time warning threshold."
        
        new_config_id = await config_manager.save_config(
            new_config,
            updated_by="advanced_example_script",
            change_notes=new_config.change_notes
        )
        print(f"✅ New configuration version '{new_config.version}' saved with ID: {new_config_id}")
        
        # Verify active config
        active_after_save = await config_manager.get_active_config(force_refresh=True)
        print(f"   Active config after save: {active_after_save.version}")
        print(f"   - Cache TTL (News): {active_after_save.cache_config.get_ttl(QueryType.NEWS)}h")
        print(f"   - Response Time Warning: {active_after_save.performance.response_time_warning_ms}ms")

    except Exception as e:
        print(f"❌ Error saving new config: {e}")
    
    # Step 5: Validate a configuration (without saving)
    print("
5. Validating a Configuration (Pre-deployment Check)...")
    invalid_config_data = initial_config.model_dump()
    invalid_config_data["cache_config"]["news_ttl"] = -10 # Invalid TTL

    validation_result = await config_manager.validate_config(invalid_config_data)
    if validation_result["valid"]:
        print("✅ Configuration is valid (unexpected for this example).")
    else:
        print("❌ Configuration validation failed (as expected):")
        print(f"   Error: {validation_result.get('error')}")
        if 'details' in validation_result:
            for detail in validation_result['details']:
                print(f"   - Field: {detail.get('loc')}, Message: {detail.get('msg')}, Type: {detail.get('type')}")
    
    valid_config_data = new_config.model_dump() # Use the previously saved valid config
    validation_result_valid = await config_manager.validate_config(valid_config_data)
    if validation_result_valid["valid"]:
        print("✅ Valid configuration passed validation.")
    else:
        print(f"❌ Valid configuration failed validation unexpectedly: {validation_result_valid.get('error')}")

    # Step 6: Retrieve Configuration History
    print("
6. Retrieving Configuration History...")
    try:
        history = await config_manager.get_config_history(limit=3)
        print("📜 Recent Configuration History (Last 3):")
        for entry in history:
            print(f"   - ID: {entry['id']}, Version: {entry['version']}, Active: {entry['is_active']}, Updated By: {entry['updated_by']}, Notes: {entry['change_notes']}")
        
        if len(history) < 2:
            print("Insufficient history to demonstrate rollback properly. Please run this example multiple times.")
            return

        # Identify a version to roll back to (e.g., the original active config)
        rollback_target_id = history[1]['id'] # Assuming the second most recent is the original
        print(f"
   Attempting to roll back to version ID: {rollback_target_id}...")
        
        # Step 7: Rollback to a previous version
        rolled_back_config = await config_manager.rollback_config(rollback_target_id, "advanced_example_script_rollback")
        print(f"✅ Successfully rolled back to version: {rolled_back_config.version}")
        print(f"   - Cache TTL (News): {rolled_back_config.cache_config.get_ttl(QueryType.NEWS)}h")

        # Verify active config after rollback
        active_after_rollback = await config_manager.get_active_config(force_refresh=True)
        print(f"   Active config after rollback: {active_after_rollback.version}")
        print(f"   - Cache TTL (News): {active_after_rollback.cache_config.get_ttl(QueryType.NEWS)}h")


    except Exception as e:
        print(f"❌ Error retrieving history or rolling back config: {e}")
        print("Ensure you have at least two configuration versions saved for rollback to work.")

    print("
✨ Advanced Configuration Management Example Completed.")

if __name__ == "__main__":
    asyncio.run(main()) 