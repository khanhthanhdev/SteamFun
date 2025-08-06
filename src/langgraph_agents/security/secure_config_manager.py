"""Secure configuration management with encrypted API key storage."""

import os
import base64
import json
import logging
from typing import Any, Dict, Optional, Union
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from pydantic import BaseModel, Field, ValidationError


logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration management fails."""
    pass


class EncryptionConfig(BaseModel):
    """Configuration for encryption settings."""
    salt_length: int = Field(default=32, ge=16, le=64)
    iterations: int = Field(default=100000, ge=50000, le=1000000)
    key_length: int = Field(default=32, ge=16, le=64)


class SecureConfigManager:
    """
    Manages secure configuration with encrypted API key storage and retrieval.
    
    Features:
    - Encrypted API key storage using Fernet (AES 128)
    - Environment variable configuration management
    - Secure credential handling with key derivation
    - Configuration validation and type checking
    """
    
    def __init__(
        self,
        config_dir: Optional[Union[str, Path]] = None,
        master_password: Optional[str] = None,
        encryption_config: Optional[EncryptionConfig] = None
    ):
        """
        Initialize SecureConfigManager.
        
        Args:
            config_dir: Directory to store encrypted configuration files
            master_password: Master password for encryption (if None, uses env var)
            encryption_config: Configuration for encryption parameters
        """
        self.config_dir = Path(config_dir or os.path.expanduser("~/.langgraph_agents"))
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.encryption_config = encryption_config or EncryptionConfig()
        self._master_password = master_password or os.getenv("LANGGRAPH_MASTER_PASSWORD")
        self._encryption_key = None
        self._config_cache = {}
        
        # Paths for configuration files
        self.encrypted_config_path = self.config_dir / "encrypted_config.json"
        self.salt_path = self.config_dir / "config.salt"
        
        # Initialize encryption if master password is available
        if self._master_password:
            self._initialize_encryption()
    
    def _initialize_encryption(self) -> None:
        """Initialize encryption key from master password."""
        try:
            # Load or generate salt
            salt = self._get_or_create_salt()
            
            # Derive encryption key from master password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=self.encryption_config.key_length,
                salt=salt,
                iterations=self.encryption_config.iterations,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self._master_password.encode()))
            self._encryption_key = Fernet(key)
            
            logger.info("Encryption initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise ConfigError(f"Encryption initialization failed: {e}")
    
    def _get_or_create_salt(self) -> bytes:
        """Get existing salt or create a new one."""
        if self.salt_path.exists():
            try:
                with open(self.salt_path, 'rb') as f:
                    salt = f.read()
                if len(salt) == self.encryption_config.salt_length:
                    return salt
                else:
                    logger.warning("Invalid salt length, generating new salt")
            except Exception as e:
                logger.warning(f"Failed to read salt file: {e}")
        
        # Generate new salt
        salt = os.urandom(self.encryption_config.salt_length)
        try:
            with open(self.salt_path, 'wb') as f:
                f.write(salt)
            # Set restrictive permissions
            os.chmod(self.salt_path, 0o600)
        except Exception as e:
            logger.error(f"Failed to save salt: {e}")
            raise ConfigError(f"Salt generation failed: {e}")
        
        return salt
    
    def encrypt_value(self, value: str) -> str:
        """
        Encrypt a string value.
        
        Args:
            value: String to encrypt
            
        Returns:
            Base64-encoded encrypted string
            
        Raises:
            ConfigError: If encryption fails or is not initialized
        """
        if not self._encryption_key:
            raise ConfigError("Encryption not initialized. Provide master password.")
        
        try:
            encrypted_bytes = self._encryption_key.encrypt(value.encode())
            return base64.urlsafe_b64encode(encrypted_bytes).decode()
        except Exception as e:
            logger.error(f"Encryption failed: {e}")
            raise ConfigError(f"Failed to encrypt value: {e}")
    
    def decrypt_value(self, encrypted_value: str) -> str:
        """
        Decrypt an encrypted string value.
        
        Args:
            encrypted_value: Base64-encoded encrypted string
            
        Returns:
            Decrypted string
            
        Raises:
            ConfigError: If decryption fails or is not initialized
        """
        if not self._encryption_key:
            raise ConfigError("Encryption not initialized. Provide master password.")
        
        try:
            encrypted_bytes = base64.urlsafe_b64decode(encrypted_value.encode())
            decrypted_bytes = self._encryption_key.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            logger.error(f"Decryption failed: {e}")
            raise ConfigError(f"Failed to decrypt value: {e}")
    
    def store_api_key(self, provider: str, api_key: str, encrypt: bool = True) -> None:
        """
        Store an API key securely.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            api_key: API key to store
            encrypt: Whether to encrypt the API key
            
        Raises:
            ConfigError: If storage fails
        """
        try:
            # Load existing configuration
            config = self._load_encrypted_config()
            
            # Store API key (encrypted or plain)
            if encrypt and self._encryption_key:
                config[f"{provider}_api_key"] = {
                    "value": self.encrypt_value(api_key),
                    "encrypted": True
                }
            else:
                config[f"{provider}_api_key"] = {
                    "value": api_key,
                    "encrypted": False
                }
                if encrypt:
                    logger.warning(f"Storing {provider} API key unencrypted (encryption not available)")
            
            # Save configuration
            self._save_encrypted_config(config)
            
            # Update cache
            self._config_cache[f"{provider}_api_key"] = api_key
            
            logger.info(f"API key for {provider} stored successfully")
            
        except Exception as e:
            logger.error(f"Failed to store API key for {provider}: {e}")
            raise ConfigError(f"Failed to store API key: {e}")
    
    def get_api_key(self, provider: str, fallback_env_var: Optional[str] = None) -> Optional[str]:
        """
        Retrieve an API key.
        
        Args:
            provider: Provider name (e.g., 'openai', 'anthropic')
            fallback_env_var: Environment variable to check if key not found in config
            
        Returns:
            API key string or None if not found
            
        Raises:
            ConfigError: If retrieval fails
        """
        key_name = f"{provider}_api_key"
        
        # Check cache first
        if key_name in self._config_cache:
            return self._config_cache[key_name]
        
        try:
            # Load from encrypted config
            config = self._load_encrypted_config()
            
            if key_name in config:
                key_data = config[key_name]
                if isinstance(key_data, dict):
                    if key_data.get("encrypted", False):
                        api_key = self.decrypt_value(key_data["value"])
                    else:
                        api_key = key_data["value"]
                else:
                    # Legacy format - assume unencrypted
                    api_key = key_data
                
                # Cache the result
                self._config_cache[key_name] = api_key
                return api_key
            
            # Fallback to environment variable
            if fallback_env_var:
                env_key = os.getenv(fallback_env_var)
                if env_key:
                    # Cache the result
                    self._config_cache[key_name] = env_key
                    return env_key
            
            # Try standard environment variable naming
            standard_env_var = f"{provider.upper()}_API_KEY"
            env_key = os.getenv(standard_env_var)
            if env_key:
                # Cache the result
                self._config_cache[key_name] = env_key
                return env_key
            
            logger.warning(f"API key for {provider} not found")
            return None
            
        except Exception as e:
            logger.error(f"Failed to retrieve API key for {provider}: {e}")
            raise ConfigError(f"Failed to retrieve API key: {e}")
    
    def store_config_value(self, key: str, value: Any, encrypt: bool = False) -> None:
        """
        Store a configuration value.
        
        Args:
            key: Configuration key
            value: Configuration value
            encrypt: Whether to encrypt the value
            
        Raises:
            ConfigError: If storage fails
        """
        try:
            # Load existing configuration
            config = self._load_encrypted_config()
            
            # Convert value to string for encryption
            str_value = json.dumps(value) if not isinstance(value, str) else value
            
            # Store value (encrypted or plain)
            if encrypt and self._encryption_key:
                config[key] = {
                    "value": self.encrypt_value(str_value),
                    "encrypted": True,
                    "type": type(value).__name__
                }
            else:
                config[key] = {
                    "value": value,
                    "encrypted": False,
                    "type": type(value).__name__
                }
            
            # Save configuration
            self._save_encrypted_config(config)
            
            # Update cache
            self._config_cache[key] = value
            
            logger.info(f"Configuration value '{key}' stored successfully")
            
        except Exception as e:
            logger.error(f"Failed to store config value '{key}': {e}")
            raise ConfigError(f"Failed to store config value: {e}")
    
    def get_config_value(
        self,
        key: str,
        default: Any = None,
        fallback_env_var: Optional[str] = None
    ) -> Any:
        """
        Retrieve a configuration value.
        
        Args:
            key: Configuration key
            default: Default value if not found
            fallback_env_var: Environment variable to check if key not found
            
        Returns:
            Configuration value or default
            
        Raises:
            ConfigError: If retrieval fails
        """
        # Check cache first
        if key in self._config_cache:
            return self._config_cache[key]
        
        try:
            # Load from encrypted config
            config = self._load_encrypted_config()
            
            if key in config:
                value_data = config[key]
                if isinstance(value_data, dict):
                    if value_data.get("encrypted", False):
                        str_value = self.decrypt_value(value_data["value"])
                        # Try to restore original type
                        if value_data.get("type") != "str":
                            try:
                                value = json.loads(str_value)
                            except json.JSONDecodeError:
                                value = str_value
                        else:
                            value = str_value
                    else:
                        value = value_data["value"]
                else:
                    # Legacy format
                    value = value_data
                
                # Cache the result
                self._config_cache[key] = value
                return value
            
            # Fallback to environment variable
            if fallback_env_var:
                env_value = os.getenv(fallback_env_var, default)
                if env_value != default:
                    # Cache the result
                    self._config_cache[key] = env_value
                    return env_value
            
            return default
            
        except Exception as e:
            logger.error(f"Failed to retrieve config value '{key}': {e}")
            raise ConfigError(f"Failed to retrieve config value: {e}")
    
    def _load_encrypted_config(self) -> Dict[str, Any]:
        """Load encrypted configuration from file."""
        if not self.encrypted_config_path.exists():
            return {}
        
        try:
            with open(self.encrypted_config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load encrypted config: {e}")
            return {}
    
    def _save_encrypted_config(self, config: Dict[str, Any]) -> None:
        """Save encrypted configuration to file."""
        try:
            with open(self.encrypted_config_path, 'w') as f:
                json.dump(config, f, indent=2)
            # Set restrictive permissions
            os.chmod(self.encrypted_config_path, 0o600)
        except Exception as e:
            logger.error(f"Failed to save encrypted config: {e}")
            raise ConfigError(f"Failed to save configuration: {e}")
    
    def list_stored_keys(self) -> Dict[str, Dict[str, Any]]:
        """
        List all stored configuration keys and their metadata.
        
        Returns:
            Dictionary with key metadata (encrypted status, type, etc.)
        """
        try:
            config = self._load_encrypted_config()
            result = {}
            
            for key, value in config.items():
                if isinstance(value, dict):
                    result[key] = {
                        "encrypted": value.get("encrypted", False),
                        "type": value.get("type", "unknown"),
                        "has_value": bool(value.get("value"))
                    }
                else:
                    result[key] = {
                        "encrypted": False,
                        "type": type(value).__name__,
                        "has_value": bool(value)
                    }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to list stored keys: {e}")
            raise ConfigError(f"Failed to list stored keys: {e}")
    
    def delete_config_value(self, key: str) -> bool:
        """
        Delete a configuration value.
        
        Args:
            key: Configuration key to delete
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            ConfigError: If deletion fails
        """
        try:
            config = self._load_encrypted_config()
            
            if key in config:
                del config[key]
                self._save_encrypted_config(config)
                
                # Remove from cache
                if key in self._config_cache:
                    del self._config_cache[key]
                
                logger.info(f"Configuration value '{key}' deleted successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to delete config value '{key}': {e}")
            raise ConfigError(f"Failed to delete config value: {e}")
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._config_cache.clear()
        logger.info("Configuration cache cleared")
    
    def validate_encryption(self) -> bool:
        """
        Validate that encryption is working correctly.
        
        Returns:
            True if encryption is working, False otherwise
        """
        if not self._encryption_key:
            return False
        
        try:
            test_value = "test_encryption_validation"
            encrypted = self.encrypt_value(test_value)
            decrypted = self.decrypt_value(encrypted)
            return decrypted == test_value
        except Exception:
            return False
    
    def get_environment_config(self, prefix: str = "LANGGRAPH_") -> Dict[str, str]:
        """
        Get all environment variables with a specific prefix.
        
        Args:
            prefix: Environment variable prefix
            
        Returns:
            Dictionary of environment variables (without prefix)
        """
        env_config = {}
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                env_config[config_key] = value
        
        return env_config
    
    def merge_config(
        self,
        stored_config: Dict[str, Any],
        env_config: Dict[str, Any],
        prefer_env: bool = True
    ) -> Dict[str, Any]:
        """
        Merge stored configuration with environment variables.
        
        Args:
            stored_config: Configuration from secure storage
            env_config: Configuration from environment variables
            prefer_env: Whether to prefer environment variables over stored config
            
        Returns:
            Merged configuration dictionary
        """
        if prefer_env:
            merged = stored_config.copy()
            merged.update(env_config)
        else:
            merged = env_config.copy()
            merged.update(stored_config)
        
        return merged