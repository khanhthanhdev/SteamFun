#!/usr/bin/env python3
"""
Example usage of configuration hot-reloading functionality.

This example demonstrates how to use the hot-reloading capabilities
of the centralized configuration system in a real application.
"""

import os
import sys
import time
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from config.manager import ConfigurationManager
from config.models import SystemConfig

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExampleApplication:
    """Example application that uses configuration hot-reloading."""
    
    def __init__(self):
        """Initialize the application with configuration manager."""
        # Ensure development mode for this example
        os.environ['ENVIRONMENT'] = 'development'
        os.environ['ENABLE_HOT_RELOAD'] = 'true'
        
        self.config_manager = ConfigurationManager()
        self.current_config = self.config_manager.config
        
        # Register for configuration change notifications
        self.config_manager.add_config_change_callback(self.on_configuration_changed)
        
        logger.info("Application initialized with hot-reload enabled")
    
    def on_configuration_changed(self, old_config: SystemConfig, new_config: SystemConfig):
        """Handle configuration changes.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
        """
        logger.info("Configuration changed - updating application settings")
        
        # Update internal configuration reference
        self.current_config = new_config
        
        # Example: React to specific configuration changes
        if old_config.debug != new_config.debug:
            self.update_debug_mode(new_config.debug)
        
        if old_config.default_llm_provider != new_config.default_llm_provider:
            self.update_llm_provider(new_config.default_llm_provider)
        
        # Example: Restart services if needed
        if self.needs_service_restart(old_config, new_config):
            self.restart_services()
    
    def update_debug_mode(self, debug_enabled: bool):
        """Update application debug mode.
        
        Args:
            debug_enabled: Whether debug mode is enabled
        """
        if debug_enabled:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.info("Debug mode enabled")
        else:
            logging.getLogger().setLevel(logging.INFO)
            logger.info("Debug mode disabled")
    
    def update_llm_provider(self, provider_name: str):
        """Update the default LLM provider.
        
        Args:
            provider_name: Name of the new default provider
        """
        logger.info(f"Switching to LLM provider: {provider_name}")
        
        # Example: Reinitialize LLM clients
        # In a real application, you would update your LLM client instances here
        provider_config = self.config_manager.get_provider_config(provider_name)
        if provider_config:
            logger.info(f"Provider {provider_name} configuration loaded successfully")
        else:
            logger.warning(f"No configuration found for provider: {provider_name}")
    
    def needs_service_restart(self, old_config: SystemConfig, new_config: SystemConfig) -> bool:
        """Check if services need to be restarted due to configuration changes.
        
        Args:
            old_config: Previous configuration
            new_config: New configuration
            
        Returns:
            True if services need restart, False otherwise
        """
        # Example: Restart if RAG configuration changed
        if old_config.rag_config != new_config.rag_config:
            return True
        
        # Example: Restart if monitoring configuration changed
        if old_config.monitoring_config != new_config.monitoring_config:
            return True
        
        return False
    
    def restart_services(self):
        """Restart application services after configuration change."""
        logger.info("Restarting services due to configuration changes...")
        
        # Example: In a real application, you would restart relevant services here
        # - RAG system
        # - Monitoring services
        # - Agent instances
        
        logger.info("Services restarted successfully")
    
    def run(self):
        """Run the example application."""
        logger.info("Starting example application...")
        
        print("=== Configuration Hot-Reload Example ===")
        print()
        print("This example demonstrates hot-reloading in action.")
        print("The application will react to configuration changes automatically.")
        print()
        
        # Show current configuration
        self.show_current_config()
        
        # Show file watcher status
        watcher_status = self.config_manager.get_file_watcher_status()
        print("File Watcher Status:")
        print(f"  Active: {watcher_status['is_watching']}")
        print(f"  Watched files: {', '.join(watcher_status['watched_files'])}")
        print()
        
        if watcher_status['is_watching']:
            print("Hot-reload is active! Try modifying your .env file to see changes.")
            print("Example changes to try:")
            print("  - Change DEBUG=true to DEBUG=false")
            print("  - Change DEFAULT_LLM_PROVIDER to a different provider")
            print("  - Modify any other configuration values")
            print()
            print("The application will run for 60 seconds...")
            print("Press Ctrl+C to stop early.")
            print()
            
            # Run for 60 seconds, showing periodic status
            for i in range(60):
                try:
                    time.sleep(1)
                    if i % 10 == 0 and i > 0:
                        print(f"Still running... {60-i} seconds remaining")
                except KeyboardInterrupt:
                    print("\nStopping application...")
                    break
        else:
            print("Hot-reload is not active. This could be due to:")
            print("  - Production environment")
            print("  - Hot-reload disabled in configuration")
            print("  - Missing watchdog library")
            print()
            print("You can still test manual reload:")
            
            input("Press Enter to test manual configuration reload...")
            success = self.config_manager.safe_reload_config()
            print(f"Manual reload successful: {success}")
        
        print()
        print("=== Example Completed ===")
    
    def show_current_config(self):
        """Display current configuration summary."""
        config = self.current_config
        
        print("Current Configuration:")
        print(f"  Environment: {config.environment}")
        print(f"  Debug: {config.debug}")
        print(f"  Default LLM Provider: {config.default_llm_provider}")
        print(f"  Available Providers: {list(config.llm_providers.keys())}")
        print(f"  RAG Enabled: {config.rag_config.enabled if config.rag_config else 'Not configured'}")
        print(f"  Monitoring Enabled: {config.monitoring_config.enabled}")
        print()


def main():
    """Main function to run the example."""
    try:
        app = ExampleApplication()
        app.run()
    except Exception as e:
        logger.error(f"Application failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()