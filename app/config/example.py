#!/usr/bin/env python3
"""
Example usage of the configuration management system

This example demonstrates how to use the centralized configuration system
in your FastAPI application.
"""

from app.config import get_settings, validate_configuration

def main():
    """Example of using the configuration system"""
    
    print("🔧 FastAPI Configuration Management Example")
    print("=" * 50)
    
    # Load settings
    print("\n1. Loading application settings...")
    settings = get_settings()
    
    print(f"✅ Application: {settings.app.app_name}")
    print(f"   Environment: {settings.app.environment.value}")
    print(f"   Debug Mode: {settings.app.debug}")
    print(f"   Server: {settings.app.host}:{settings.app.port}")
    
    # Access different configuration sections
    print(f"\n2. Configuration sections:")
    print(f"   Database Host: {settings.database.database_host}")
    print(f"   RAG Enabled: {settings.rag.rag_enabled}")
    print(f"   Monitoring Enabled: {settings.monitoring.monitoring_enabled}")
    
    # Validate configuration
    print(f"\n3. Configuration validation:")
    validation_result = validate_configuration(settings)
    
    if validation_result.is_valid:
        print("✅ Configuration is valid!")
    else:
        print(f"⚠️  Configuration has {len(validation_result.errors)} errors and {len(validation_result.warnings)} warnings")
        
        if validation_result.errors:
            print("\nErrors:")
            for error in validation_result.errors[:3]:  # Show first 3
                print(f"   - {error}")
            if len(validation_result.errors) > 3:
                print(f"   ... and {len(validation_result.errors) - 3} more")
    
    # Show environment-specific behavior
    print(f"\n4. Environment-specific settings:")
    if settings.app.is_development:
        print("   🔧 Development mode: Debug enabled, auto-reload on")
    elif settings.app.is_production:
        print("   🚀 Production mode: Debug disabled, multiple workers")
    else:
        print("   🧪 Testing mode: Minimal logging, no external services")
    
    print(f"\n5. Usage in FastAPI:")
    print("""
    # In your FastAPI app:
    from fastapi import FastAPI, Depends
    from app.config import get_settings, Settings
    
    app = FastAPI()
    
    @app.get("/config")
    def get_config(settings: Settings = Depends(get_settings)):
        return {
            "app_name": settings.app.app_name,
            "environment": settings.app.environment,
            "debug": settings.app.debug
        }
    """)
    
    print("=" * 50)
    print("🎉 Configuration system ready for use!")

if __name__ == "__main__":
    main()