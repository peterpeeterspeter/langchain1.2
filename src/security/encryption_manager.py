"""
Encryption Manager for Universal RAG CMS
Handles all encryption/decryption operations with AES-256
"""

import hashlib
import hmac
import secrets
import base64
from typing import Tuple
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from .models import SecurityConfig


class EncryptionManager:
    """Handles all encryption/decryption operations"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self._setup_encryption()
    
    def _setup_encryption(self):
        """Setup encryption keys"""
        # Derive key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.config.salt_value.encode(),
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(
            kdf.derive(self.config.encryption_key.encode())
        )
        self.cipher = Fernet(key)
    
    def encrypt(self, data: str) -> str:
        """Encrypt sensitive data"""
        if not data:
            return data
        
        try:
            encrypted = self.cipher.encrypt(data.encode())
            return base64.urlsafe_b64encode(encrypted).decode()
        except Exception as e:
            raise ValueError(f"Encryption failed: {str(e)}")
    
    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        if not encrypted_data:
            return encrypted_data
        
        try:
            decoded = base64.urlsafe_b64decode(encrypted_data.encode())
            decrypted = self.cipher.decrypt(decoded)
            return decrypted.decode()
        except Exception as e:
            raise ValueError(f"Decryption failed: {str(e)}")
    
    def hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        pwdhash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return f"{salt}${pwdhash.hex()}"
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, pwdhash = password_hash.split('$')
            pwdhash_check = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return pwdhash_check.hex() == pwdhash
        except:
            return False
    
    def generate_api_key(self) -> Tuple[str, str]:
        """Generate API key and its hash"""
        api_key = secrets.token_urlsafe(self.config.api_key_length)
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        return api_key, api_key_hash
    
    def create_secure_token(self, length: int = 32) -> str:
        """Create secure random token"""
        return secrets.token_urlsafe(length)
    
    def verify_hmac_signature(self, message: str, signature: str, secret: str) -> bool:
        """Verify HMAC signature"""
        try:
            expected_signature = hmac.new(
                secret.encode(),
                message.encode(),
                hashlib.sha256
            ).hexdigest()
            return hmac.compare_digest(signature, expected_signature)
        except:
            return False 