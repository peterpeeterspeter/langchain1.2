#!/usr/bin/env python3
"""
Initialize Security System
Create first admin user and API key
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security.security_manager import SecurityManager
from src.security.models import Permission, UserRole
from supabase import create_client

async def main():
    """Initialize security system"""
    print("🔒 Security System Initialization")
    print("=" * 50)
    
    # Create Supabase client
    supabase = create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_SERVICE_KEY")
    )
    
    # Initialize security manager
    security = SecurityManager(supabase)
    
    print("
👤 Creating admin API key...")
    admin_email = input("Enter admin email: ").strip()
    
    try:
        # Create admin API key
        admin_key_data = await security.api_keys.create_api_key(
            user_id=admin_email,
            service_name="admin_access", 
            permissions=list(Permission)
        )
        
        print("
✅ Security system initialized!")
        print("=" * 50)
        print(f"Admin Email: {admin_email}")
        print(f"API Key ID: {admin_key_data['key_id']}")
        print(f"API Key: {admin_key_data['api_key']}")
        print("=" * 50)
        print("⚠️ SAVE THIS API KEY SECURELY!")
        print("
Usage:")
        print(f"Authorization: ApiKey {admin_key_data['api_key']}")
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
