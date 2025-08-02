#!/usr/bin/env python3
"""
Setup script for LangGraph CLI integration
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            return True
        else:
            print(f"❌ {description} failed:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error during {description}: {e}")
        return False

def check_langgraph_cli() -> bool:
    """Check if LangGraph CLI is installed"""
    try:
        result = subprocess.run(["langgraph", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ LangGraph CLI is installed: {result.stdout.strip()}")
            return True
        else:
            return False
    except FileNotFoundError:
        return False

def install_langgraph_cli() -> bool:
    """Install LangGraph CLI"""
    return run_command("pip install langgraph-cli", "Installing LangGraph CLI")

def setup_environment() -> bool:
    """Setup environment file"""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("🔄 Creating .env file from .env.example...")
        try:
            with open(env_example, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            print("✅ Created .env file! Please update it with your actual values.")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    elif env_file.exists():
        print("✅ .env file already exists")
        return True
    else:
        print("⚠️  No .env.example file found")
        return False

def validate_config() -> bool:
    """Validate the langgraph.json configuration"""
    config_file = Path("langgraph.json")
    if not config_file.exists():
        print("❌ langgraph.json not found!")
        return False
    
    try:
        import json
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        required_fields = ["dependencies", "graphs"]
        missing = [field for field in required_fields if field not in config]
        
        if missing:
            print(f"❌ Missing required fields in langgraph.json: {missing}")
            return False
        
        print("✅ langgraph.json is valid!")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in langgraph.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Error validating langgraph.json: {e}")
        return False

def create_directories() -> bool:
    """Create necessary directories"""
    directories = ["output", "media", "code", "data/rag/chroma_db", "data/rag/manim_docs"]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 Setting up LangGraph CLI for Video Generation System")
    print("=" * 60)
    
    success = True
    
    # Check/install LangGraph CLI
    if not check_langgraph_cli():
        print("📦 LangGraph CLI not found, installing...")
        success &= install_langgraph_cli()
    
    # Setup environment
    success &= setup_environment()
    
    # Validate configuration
    success &= validate_config()
    
    # Create directories
    success &= create_directories()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Update your .env file with actual API keys and configuration")
        print("2. Run: python langgraph_cli_helper.py validate")
        print("3. Run: python langgraph_cli_helper.py dev")
        print("4. Visit http://localhost:8123 to access LangGraph Studio")
    else:
        print("❌ Setup completed with errors. Please check the messages above.")
    
    print("\nUseful commands:")
    print("  python langgraph_cli_helper.py info     - Show configuration info")
    print("  python langgraph_cli_helper.py graphs   - List available graphs")
    print("  python langgraph_cli_helper.py dev      - Start development server")
    print("  langgraph --help                        - Show LangGraph CLI help")

if __name__ == "__main__":
    main()