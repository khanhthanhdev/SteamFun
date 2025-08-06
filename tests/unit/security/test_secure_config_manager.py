"""Unit tests for SecureConfigManager."""

import os
import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from src.langgraph_agents.security.secure_config_manager import (
    SecureConfigManager,
    ConfigError,
    EncryptionConfig
)


class TestSecureConfigManager:
    """Test cases for SecureConfigManager class."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    @pytest.fixture
    def config_manager(self, temp_config_dir):
        """Create a SecureConfigManager instance with temporary directory."""
        return SecureConfigManager(
            config_dir=temp_config_dir,
            master_password="test_password_123"
        )
    
    @pytest.fixture
    def config_manager_no_password(self, temp_config_dir):
        """Create a SecureConfigManager instance without master password."""
        return SecureConfigManager(config_dir=temp_config_dir)
    
    def test_initialization_with_password(self, temp_config_dir):
        """Test successful initialization with master password."""
        manager = SecureConfigManager(
            config_dir=temp_config_dir,
            master_password="test_password"
        )
        
        assert manager.config_dir == temp_config_dir
        assert manager._master_password == "test_password"
        assert manager._encryption_key is not None
        assert manager.salt_path.exists()
    
    def test_initialization_without_password(self, temp_config_dir):
        """Test initialization without master password."""
        manager = SecureConfigManager(config_dir=temp_config_dir)
        
        assert manager.config_dir == temp_config_dir
        assert manager._master_password is None
        assert manager._encryption_key is None
    
    @patch.dict(os.environ, {'LANGGRAPH_MASTER_PASSWORD': 'env_password'})
    def test_initialization_with_env_password(self, temp_config_dir):
        """Test initialization with password from environment variable."""
        manager = SecureConfigManager(config_dir=temp_config_dir)
        
        assert manager._master_password == "env_password"
        assert manager._encryption_key is not None
    
    def test_encryption_config_validation(self, temp_config_dir):
        """Test encryption configuration validation."""
        encryption_config = EncryptionConfig(
            salt_length=16,
            iterations=50000,
            key_length=32
        )
        
        manager = SecureConfigManager(
            config_dir=temp_config_dir,
            master_password="test_password",
            encryption_config=encryption_config
        )
        
        assert manager.encryption_config.salt_length == 16
        assert manager.encryption_config.iterations == 50000
        assert manager.encryption_config.key_length == 32
    
    def test_encrypt_decrypt_value(self, config_manager):
        """Test value encryption and decryption."""
        original_value = "test_secret_value"
        
        # Encrypt value
        encrypted = config_manager.encrypt_value(original_value)
        assert encrypted != original_value
        assert len(encrypted) > 0
        
        # Decrypt value
        decrypted = config_manager.decrypt_value(encrypted)
        assert decrypted == original_value
    
    def test_encrypt_without_key(self, config_manager_no_password):
        """Test encryption without encryption key raises error."""
        with pytest.raises(ConfigError, match="Encryption not initialized"):
            config_manager_no_password.encrypt_value("test")
    
    def test_decrypt_without_key(self, config_manager_no_password):
        """Test decryption without encryption key raises error."""
        with pytest.raises(ConfigError, match="Encryption not initialized"):
            config_manager_no_password.decrypt_value("test")
    
    def test_store_api_key_encrypted(self, config_manager):
        """Test storing API key with encryption."""
        provider = "openai"
        api_key = "sk-test123456789"
        
        config_manager.store_api_key(provider, api_key, encrypt=True)
        
        # Verify it's stored
        stored_key = config_manager.get_api_key(provider)
        assert stored_key == api_key
        
        # Verify it's encrypted in storage
        config = config_manager._load_encrypted_config()
        assert f"{provider}_api_key" in config
        assert config[f"{provider}_api_key"]["encrypted"] is True
        assert config[f"{provider}_api_key"]["value"] != api_key
    
    def test_store_api_key_unencrypted(self, config_manager):
        """Test storing API key without encryption."""
        provider = "anthropic"
        api_key = "sk-ant-test123"
        
        config_manager.store_api_key(provider, api_key, encrypt=False)
        
        # Verify it's stored
        stored_key = config_manager.get_api_key(provider)
        assert stored_key == api_key
        
        # Verify it's not encrypted in storage
        config = config_manager._load_encrypted_config()
        assert f"{provider}_api_key" in config
        assert config[f"{provider}_api_key"]["encrypted"] is False
        assert config[f"{provider}_api_key"]["value"] == api_key
    
    def test_get_api_key_not_found(self, config_manager):
        """Test getting API key that doesn't exist."""
        result = config_manager.get_api_key("nonexistent")
        assert result is None
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'env_api_key'})
    def test_get_api_key_from_env(self, config_manager):
        """Test getting API key from environment variable."""
        result = config_manager.get_api_key("openai")
        assert result == "env_api_key"
    
    @patch.dict(os.environ, {'CUSTOM_API_KEY': 'custom_env_key'})
    def test_get_api_key_with_fallback_env(self, config_manager):
        """Test getting API key with custom fallback environment variable."""
        result = config_manager.get_api_key("custom", fallback_env_var="CUSTOM_API_KEY")
        assert result == "custom_env_key"
    
    def test_store_config_value_encrypted(self, config_manager):
        """Test storing configuration value with encryption."""
        key = "database_password"
        value = "super_secret_password"
        
        config_manager.store_config_value(key, value, encrypt=True)
        
        # Verify it's stored
        stored_value = config_manager.get_config_value(key)
        assert stored_value == value
        
        # Verify it's encrypted in storage
        config = config_manager._load_encrypted_config()
        assert key in config
        assert config[key]["encrypted"] is True
        assert config[key]["value"] != value
    
    def test_store_config_value_complex_type(self, config_manager):
        """Test storing complex configuration value."""
        key = "model_config"
        value = {"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000}
        
        config_manager.store_config_value(key, value, encrypt=False)
        
        # Verify it's stored and retrieved correctly
        stored_value = config_manager.get_config_value(key)
        assert stored_value == value
    
    def test_get_config_value_with_default(self, config_manager):
        """Test getting configuration value with default."""
        default_value = "default_config"
        result = config_manager.get_config_value("nonexistent", default=default_value)
        assert result == default_value
    
    @patch.dict(os.environ, {'CUSTOM_CONFIG': 'env_config_value'})
    def test_get_config_value_from_env(self, config_manager):
        """Test getting configuration value from environment variable."""
        result = config_manager.get_config_value(
            "custom_config",
            fallback_env_var="CUSTOM_CONFIG"
        )
        assert result == "env_config_value"
    
    def test_list_stored_keys(self, config_manager):
        """Test listing stored configuration keys."""
        # Store some test values
        config_manager.store_api_key("openai", "sk-test", encrypt=True)
        config_manager.store_config_value("setting1", "value1", encrypt=False)
        config_manager.store_config_value("setting2", {"key": "value"}, encrypt=True)
        
        keys = config_manager.list_stored_keys()
        
        assert "openai_api_key" in keys
        assert "setting1" in keys
        assert "setting2" in keys
        
        assert keys["openai_api_key"]["encrypted"] is True
        assert keys["setting1"]["encrypted"] is False
        assert keys["setting2"]["encrypted"] is True
    
    def test_delete_config_value(self, config_manager):
        """Test deleting configuration value."""
        key = "test_key"
        value = "test_value"
        
        # Store value
        config_manager.store_config_value(key, value)
        assert config_manager.get_config_value(key) == value
        
        # Delete value
        result = config_manager.delete_config_value(key)
        assert result is True
        
        # Verify it's deleted
        assert config_manager.get_config_value(key) is None
        
        # Try to delete non-existent key
        result = config_manager.delete_config_value("nonexistent")
        assert result is False
    
    def test_clear_cache(self, config_manager):
        """Test clearing configuration cache."""
        # Store and retrieve value to populate cache
        config_manager.store_config_value("test_key", "test_value")
        config_manager.get_config_value("test_key")
        
        # Verify cache has content
        assert len(config_manager._config_cache) > 0
        
        # Clear cache
        config_manager.clear_cache()
        
        # Verify cache is empty
        assert len(config_manager._config_cache) == 0
    
    def test_validate_encryption(self, config_manager):
        """Test encryption validation."""
        assert config_manager.validate_encryption() is True
    
    def test_validate_encryption_no_key(self, config_manager_no_password):
        """Test encryption validation without key."""
        assert config_manager_no_password.validate_encryption() is False
    
    @patch.dict(os.environ, {
        'LANGGRAPH_MODEL': 'gpt-4',
        'LANGGRAPH_TEMPERATURE': '0.7',
        'LANGGRAPH_DEBUG': 'true'
    })
    def test_get_environment_config(self, config_manager):
        """Test getting environment configuration."""
        env_config = config_manager.get_environment_config("LANGGRAPH_")
        
        assert env_config["model"] == "gpt-4"
        assert env_config["temperature"] == "0.7"
        assert env_config["debug"] == "true"
    
    def test_merge_config_prefer_env(self, config_manager):
        """Test merging configuration with preference for environment."""
        stored_config = {"model": "gpt-3.5", "temperature": 0.5}
        env_config = {"model": "gpt-4", "debug": True}
        
        merged = config_manager.merge_config(stored_config, env_config, prefer_env=True)
        
        assert merged["model"] == "gpt-4"  # From env
        assert merged["temperature"] == 0.5  # From stored
        assert merged["debug"] is True  # From env
    
    def test_merge_config_prefer_stored(self, config_manager):
        """Test merging configuration with preference for stored."""
        stored_config = {"model": "gpt-3.5", "temperature": 0.5}
        env_config = {"model": "gpt-4", "debug": True}
        
        merged = config_manager.merge_config(stored_config, env_config, prefer_env=False)
        
        assert merged["model"] == "gpt-3.5"  # From stored
        assert merged["temperature"] == 0.5  # From stored
        assert merged["debug"] is True  # From env
    
    def test_salt_persistence(self, temp_config_dir):
        """Test that salt is persisted and reused."""
        # Create first manager
        manager1 = SecureConfigManager(
            config_dir=temp_config_dir,
            master_password="test_password"
        )
        
        # Store a value
        manager1.store_api_key("test", "value", encrypt=True)
        encrypted1 = manager1._load_encrypted_config()["test_api_key"]["value"]
        
        # Create second manager with same password
        manager2 = SecureConfigManager(
            config_dir=temp_config_dir,
            master_password="test_password"
        )
        
        # Should be able to decrypt the same value
        decrypted = manager2.get_api_key("test")
        assert decrypted == "value"
    
    def test_config_file_permissions(self, config_manager):
        """Test that configuration files have restrictive permissions."""
        # Store a value to create config file
        config_manager.store_config_value("test", "value")
        
        # Check file permissions (should be 0o600)
        config_stat = config_manager.encrypted_config_path.stat()
        assert oct(config_stat.st_mode)[-3:] == '600'
        
        salt_stat = config_manager.salt_path.stat()
        assert oct(salt_stat.st_mode)[-3:] == '600'


class TestSecureConfigManagerErrorHandling:
    """Test error handling in SecureConfigManager."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary directory for configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_invalid_encryption_config(self, temp_config_dir):
        """Test invalid encryption configuration."""
        with pytest.raises(Exception):  # Pydantic validation error
            EncryptionConfig(salt_length=8)  # Too small
    
    def test_corrupted_salt_file(self, temp_config_dir):
        """Test handling of corrupted salt file."""
        # Create manager to generate salt file
        manager = SecureConfigManager(
            config_dir=temp_config_dir,
            master_password="test_password"
        )
        
        # Corrupt the salt file
        with open(manager.salt_path, 'w') as f:
            f.write("corrupted_salt")
        
        # Create new manager - should generate new salt
        manager2 = SecureConfigManager(
            config_dir=temp_config_dir,
            master_password="test_password"
        )
        
        # Should still work
        assert manager2.validate_encryption() is True
    
    def test_corrupted_config_file(self, temp_config_dir):
        """Test handling of corrupted configuration file."""
        manager = SecureConfigManager(
            config_dir=temp_config_dir,
            master_password="test_password"
        )
        
        # Create corrupted config file
        with open(manager.encrypted_config_path, 'w') as f:
            f.write("invalid json")
        
        # Should handle gracefully and return empty config
        config = manager._load_encrypted_config()
        assert config == {}
    
    def test_decrypt_invalid_data(self, temp_config_dir):
        """Test decryption of invalid encrypted data."""
        manager = SecureConfigManager(
            config_dir=temp_config_dir,
            master_password="test_password"
        )
        
        with pytest.raises(ConfigError, match="Failed to decrypt value"):
            manager.decrypt_value("invalid_encrypted_data")
    
    def test_permission_error_on_save(self, temp_config_dir):
        """Test handling of permission errors when saving configuration."""
        manager = SecureConfigManager(
            config_dir=temp_config_dir,
            master_password="test_password"
        )
        
        # Make the config file read-only to simulate permission error
        config_file = manager.encrypted_config_path
        config_file.touch()
        config_file.chmod(0o444)  # Read-only
        
        with pytest.raises(ConfigError, match="Failed to save configuration"):
            manager.store_config_value("test", "value")


class TestEncryptionConfig:
    """Test EncryptionConfig model."""
    
    def test_default_values(self):
        """Test default encryption configuration values."""
        config = EncryptionConfig()
        
        assert config.salt_length == 32
        assert config.iterations == 100000
        assert config.key_length == 32
    
    def test_custom_values(self):
        """Test custom encryption configuration values."""
        config = EncryptionConfig(
            salt_length=16,
            iterations=50000,
            key_length=16
        )
        
        assert config.salt_length == 16
        assert config.iterations == 50000
        assert config.key_length == 16
    
    def test_validation_salt_length_too_small(self):
        """Test validation of salt length (too small)."""
        with pytest.raises(Exception):  # Pydantic validation error
            EncryptionConfig(salt_length=8)
    
    def test_validation_salt_length_too_large(self):
        """Test validation of salt length (too large)."""
        with pytest.raises(Exception):  # Pydantic validation error
            EncryptionConfig(salt_length=128)
    
    def test_validation_iterations_too_small(self):
        """Test validation of iterations (too small)."""
        with pytest.raises(Exception):  # Pydantic validation error
            EncryptionConfig(iterations=1000)
    
    def test_validation_iterations_too_large(self):
        """Test validation of iterations (too large)."""
        with pytest.raises(Exception):  # Pydantic validation error
            EncryptionConfig(iterations=2000000)