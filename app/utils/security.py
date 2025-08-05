"""
Security utilities for the application.
"""

import hashlib
import hmac
import secrets
import base64
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from .exceptions import AuthenticationError, AuthorizationError

# Optional dependencies
try:
    import jwt
    JWT_AVAILABLE = True
except ImportError:
    JWT_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    FERNET_AVAILABLE = True
except ImportError:
    FERNET_AVAILABLE = False

try:
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    PASSLIB_AVAILABLE = True
except ImportError:
    pwd_context = None
    PASSLIB_AVAILABLE = False


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Hashed password
        
    Raises:
        ImportError: If passlib is not available
    """
    if not PASSLIB_AVAILABLE:
        raise ImportError("passlib is required for password hashing. Install with: pip install passlib[bcrypt]")
    
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password
        hashed_password: Hashed password
        
    Returns:
        True if password matches, False otherwise
        
    Raises:
        ImportError: If passlib is not available
    """
    if not PASSLIB_AVAILABLE:
        raise ImportError("passlib is required for password verification. Install with: pip install passlib[bcrypt]")
    
    return pwd_context.verify(plain_password, hashed_password)


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Length of the token in bytes
        
    Returns:
        Base64 encoded secure token
    """
    token_bytes = secrets.token_bytes(length)
    return base64.urlsafe_b64encode(token_bytes).decode('utf-8')


def generate_api_key(prefix: str = "ak", length: int = 32) -> str:
    """
    Generate an API key with a prefix.
    
    Args:
        prefix: Prefix for the API key
        length: Length of the random part
        
    Returns:
        API key string
    """
    random_part = secrets.token_hex(length)
    return f"{prefix}_{random_part}"


def create_jwt_token(
    payload: Dict[str, Any],
    secret_key: str,
    algorithm: str = "HS256",
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT token.
    
    Args:
        payload: Token payload
        secret_key: Secret key for signing
        algorithm: JWT algorithm
        expires_delta: Token expiration time
        
    Returns:
        JWT token string
        
    Raises:
        ImportError: If PyJWT is not available
    """
    if not JWT_AVAILABLE:
        raise ImportError("PyJWT is required for JWT tokens. Install with: pip install PyJWT")
    
    to_encode = payload.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
        to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm=algorithm)
    return encoded_jwt


def verify_jwt_token(
    token: str,
    secret_key: str,
    algorithm: str = "HS256"
) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        secret_key: Secret key for verification
        algorithm: JWT algorithm
        
    Returns:
        Decoded token payload
        
    Raises:
        AuthenticationError: If token is invalid or expired
        ImportError: If PyJWT is not available
    """
    if not JWT_AVAILABLE:
        raise ImportError("PyJWT is required for JWT tokens. Install with: pip install PyJWT")
    
    try:
        payload = jwt.decode(token, secret_key, algorithms=[algorithm])
        return payload
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("Token has expired")
    except jwt.JWTError as e:
        raise AuthenticationError(f"Invalid token: {str(e)}")


def create_hmac_signature(
    message: str,
    secret_key: str,
    algorithm: str = "sha256"
) -> str:
    """
    Create HMAC signature for a message.
    
    Args:
        message: Message to sign
        secret_key: Secret key for signing
        algorithm: Hash algorithm
        
    Returns:
        Base64 encoded HMAC signature
    """
    signature = hmac.new(
        secret_key.encode('utf-8'),
        message.encode('utf-8'),
        getattr(hashlib, algorithm)
    ).digest()
    
    return base64.b64encode(signature).decode('utf-8')


def verify_hmac_signature(
    message: str,
    signature: str,
    secret_key: str,
    algorithm: str = "sha256"
) -> bool:
    """
    Verify HMAC signature for a message.
    
    Args:
        message: Original message
        signature: Base64 encoded signature to verify
        secret_key: Secret key used for signing
        algorithm: Hash algorithm
        
    Returns:
        True if signature is valid, False otherwise
    """
    try:
        expected_signature = create_hmac_signature(message, secret_key, algorithm)
        return hmac.compare_digest(signature, expected_signature)
    except Exception:
        return False


def encrypt_data(data: str, key: bytes) -> str:
    """
    Encrypt data using Fernet symmetric encryption.
    
    Args:
        data: Data to encrypt
        key: Encryption key (32 bytes)
        
    Returns:
        Base64 encoded encrypted data
        
    Raises:
        ImportError: If cryptography is not available
    """
    if not FERNET_AVAILABLE:
        raise ImportError("cryptography is required for encryption. Install with: pip install cryptography")
    
    fernet = Fernet(key)
    encrypted_data = fernet.encrypt(data.encode('utf-8'))
    return base64.urlsafe_b64encode(encrypted_data).decode('utf-8')


def decrypt_data(encrypted_data: str, key: bytes) -> str:
    """
    Decrypt data using Fernet symmetric encryption.
    
    Args:
        encrypted_data: Base64 encoded encrypted data
        key: Decryption key (32 bytes)
        
    Returns:
        Decrypted data string
        
    Raises:
        ValueError: If decryption fails
        ImportError: If cryptography is not available
    """
    if not FERNET_AVAILABLE:
        raise ImportError("cryptography is required for decryption. Install with: pip install cryptography")
    
    try:
        fernet = Fernet(key)
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        decrypted_data = fernet.decrypt(encrypted_bytes)
        return decrypted_data.decode('utf-8')
    except Exception as e:
        raise ValueError(f"Decryption failed: {str(e)}")


def generate_encryption_key() -> bytes:
    """
    Generate a new encryption key for Fernet.
    
    Returns:
        32-byte encryption key
        
    Raises:
        ImportError: If cryptography is not available
    """
    if not FERNET_AVAILABLE:
        raise ImportError("cryptography is required for key generation. Install with: pip install cryptography")
    
    return Fernet.generate_key()


def sanitize_input(input_string: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        input_string: Input string to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if not isinstance(input_string, str):
        return ""
    
    # Truncate to max length
    sanitized = input_string[:max_length]
    
    # Remove null bytes and control characters
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\t\n\r')
    
    # Basic HTML/script tag removal (for basic XSS prevention)
    import re
    sanitized = re.sub(r'<script[^>]*>.*?</script>', '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    sanitized = re.sub(r'<[^>]+>', '', sanitized)
    
    return sanitized.strip()


def is_safe_path(path: str, base_path: str = ".") -> bool:
    """
    Check if a file path is safe (prevents directory traversal).
    
    Args:
        path: File path to check
        base_path: Base directory path
        
    Returns:
        True if path is safe, False otherwise
    """
    import os
    
    try:
        # Resolve the absolute path
        abs_base = os.path.abspath(base_path)
        abs_path = os.path.abspath(os.path.join(base_path, path))
        
        # Check if the resolved path is within the base directory
        return abs_path.startswith(abs_base)
    except (ValueError, OSError):
        return False


def generate_csrf_token() -> str:
    """
    Generate a CSRF token.
    
    Returns:
        CSRF token string
    """
    return secrets.token_urlsafe(32)


def constant_time_compare(a: str, b: str) -> bool:
    """
    Compare two strings in constant time to prevent timing attacks.
    
    Args:
        a: First string
        b: Second string
        
    Returns:
        True if strings are equal, False otherwise
    """
    return hmac.compare_digest(a, b)


def rate_limit_key(identifier: str, window: str = "hour") -> str:
    """
    Generate a rate limiting key.
    
    Args:
        identifier: Unique identifier (IP, user ID, etc.)
        window: Time window (minute, hour, day)
        
    Returns:
        Rate limiting key
    """
    now = datetime.utcnow()
    
    if window == "minute":
        time_key = now.strftime("%Y%m%d%H%M")
    elif window == "hour":
        time_key = now.strftime("%Y%m%d%H")
    elif window == "day":
        time_key = now.strftime("%Y%m%d")
    else:
        time_key = now.strftime("%Y%m%d%H")
    
    return f"rate_limit:{identifier}:{window}:{time_key}"


class SecurityHeaders:
    """Security headers for HTTP responses."""
    
    @staticmethod
    def get_security_headers() -> Dict[str, str]:
        """
        Get recommended security headers.
        
        Returns:
            Dictionary of security headers
        """
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()"
        }
    
    @staticmethod
    def get_cors_headers(
        allowed_origins: list = None,
        allowed_methods: list = None,
        allowed_headers: list = None
    ) -> Dict[str, str]:
        """
        Get CORS headers.
        
        Args:
            allowed_origins: List of allowed origins
            allowed_methods: List of allowed HTTP methods
            allowed_headers: List of allowed headers
            
        Returns:
            Dictionary of CORS headers
        """
        origins = allowed_origins or ["*"]
        methods = allowed_methods or ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        headers = allowed_headers or ["Content-Type", "Authorization"]
        
        return {
            "Access-Control-Allow-Origin": ", ".join(origins),
            "Access-Control-Allow-Methods": ", ".join(methods),
            "Access-Control-Allow-Headers": ", ".join(headers),
            "Access-Control-Max-Age": "86400"
        }