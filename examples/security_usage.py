"""Example usage of security and validation components."""

import os
import sys
sys.path.append('.')
from src.langgraph_agents.security import InputValidator, SecureConfigManager


def demonstrate_input_validation():
    """Demonstrate input validation features."""
    print("=== Input Validation Examples ===\n")
    
    # Valid topic validation
    print("1. Valid topic validation:")
    result = InputValidator.validate_topic("Introduction to Python Programming")
    print(f"   Valid: {result.is_valid}")
    print(f"   Sanitized: {result.sanitized_value}")
    print()
    
    # Invalid topic with suspicious content
    print("2. Invalid topic with suspicious content:")
    result = InputValidator.validate_topic("Learn Python <script>alert('xss')</script>")
    print(f"   Valid: {result.is_valid}")
    print(f"   Errors: {result.errors}")
    print()
    
    # Code validation - valid Manim code
    print("3. Valid Manim code validation:")
    code = """
from manim import *
import numpy as np

class TestScene(Scene):
    def construct(self):
        circle = Circle()
        self.play(Create(circle))
    """
    result = InputValidator.validate_code(code, allow_manim_only=True)
    print(f"   Valid: {result.is_valid}")
    print()
    
    # Code validation - dangerous code
    print("4. Dangerous code validation:")
    dangerous_code = """
import os
os.system('rm -rf /')
    """
    result = InputValidator.validate_code(dangerous_code, allow_manim_only=True)
    print(f"   Valid: {result.is_valid}")
    print(f"   Errors: {result.errors}")
    print()
    
    # Workflow input validation
    print("5. Complete workflow input validation:")
    workflow_data = {
        'topic': 'Python Basics',
        'description': 'Learn Python programming fundamentals',
        'session_id': 'session_123',
        'config': {'model': 'gpt-4'}
    }
    result = InputValidator.validate_workflow_input(workflow_data)
    print(f"   Valid: {result.is_valid}")
    if result.is_valid:
        print(f"   Sanitized data keys: {list(result.sanitized_value.keys())}")
    print()


def demonstrate_secure_config():
    """Demonstrate secure configuration management."""
    print("=== Secure Configuration Examples ===\n")
    
    # Create config manager with temporary directory
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        print("1. Initialize SecureConfigManager:")
        manager = SecureConfigManager(
            config_dir=temp_dir,
            master_password="demo_password_123"
        )
        print(f"   Encryption available: {manager.validate_encryption()}")
        print()
        
        # Store encrypted API key
        print("2. Store encrypted API key:")
        manager.store_api_key("openai", "sk-demo123456789", encrypt=True)
        print("   API key stored securely")
        print()
        
        # Retrieve API key
        print("3. Retrieve API key:")
        api_key = manager.get_api_key("openai")
        print(f"   Retrieved API key: {api_key[:10]}...")
        print()
        
        # Store configuration value
        print("4. Store configuration value:")
        config_value = {"model": "gpt-4", "temperature": 0.7}
        manager.store_config_value("model_config", config_value, encrypt=False)
        print("   Configuration stored")
        print()
        
        # List stored keys
        print("5. List stored keys:")
        keys = manager.list_stored_keys()
        for key, metadata in keys.items():
            print(f"   {key}: encrypted={metadata['encrypted']}, type={metadata['type']}")
        print()
        
        # Environment configuration
        print("6. Environment configuration:")
        # Set some demo environment variables
        os.environ['LANGGRAPH_MODEL'] = 'gpt-4'
        os.environ['LANGGRAPH_DEBUG'] = 'true'
        
        env_config = manager.get_environment_config("LANGGRAPH_")
        print(f"   Environment config: {env_config}")
        print()


def demonstrate_validation_rules():
    """Demonstrate validation rules and limits."""
    print("=== Validation Rules ===\n")
    
    rules = InputValidator.get_validation_rules()
    print("Current validation limits:")
    for rule, value in rules.items():
        print(f"   {rule}: {value}")
    print()


if __name__ == "__main__":
    try:
        demonstrate_input_validation()
        demonstrate_secure_config()
        demonstrate_validation_rules()
        
        print("✅ All security examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()