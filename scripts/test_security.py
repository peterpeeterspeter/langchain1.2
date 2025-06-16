#!/usr/bin/env python3
"""
Test Security Implementation
Run basic security tests
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.security.security_manager import SecurityManager
from src.security.models import Permission, UserRole, SecurityContext
from supabase import create_client

async def test_security():
    """Run security tests"""
    print("🧪 Security System Tests")
    print("=" * 50)
    
    try:
        # Initialize
        supabase = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_SERVICE_KEY")
        )
        security = SecurityManager(supabase)
        
        # Test 1: Input Sanitization
        print("
📝 Test 1: Input Sanitization")
        try:
            security.sanitizer.sanitize_input("'; DROP TABLE users; --")
            print("❌ SQL injection not caught!")
        except ValueError:
            print("✅ SQL injection blocked")
        
        # Test 2: Encryption
        print("
🔐 Test 2: Encryption")
        test_data = "sensitive information"
        encrypted = security.encryption.encrypt(test_data)
        decrypted = security.encryption.decrypt(encrypted)
        
        if test_data == decrypted:
            print("✅ Encryption/decryption works")
        else:
            print("❌ Encryption failed")
        
        # Test 3: API Key Generation
        print("
🔑 Test 3: API Key Generation")
        key_data = await security.api_keys.create_api_key(
            user_id="test_user",
            service_name="test",
            permissions=[Permission.CONTENT_READ]
        )
        print(f"✅ API key created: {key_data['key_id']}")
        
        # Test 4: Key Validation
        validation = await security.api_keys.validate_api_key(key_data["api_key"])
        if validation:
            print("✅ API key validation works")
        else:
            print("❌ API key validation failed")
        
        print("
🎉 All tests completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_security())
