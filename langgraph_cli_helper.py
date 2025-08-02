#!/usr/bin/env python3
"""
LangGraph CLI Helper Script
Provides utilities for working with LangGraph CLI and configuration
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

class LangGraphCLIHelper:
    def __init__(self, config_path: str = "langgraph.json"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """Load the langgraph.json configuration file"""
        if not self.config_path.exists():
            print(f"❌ Configuration file {self.config_path} not found!")
            return {}
        
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {self.config_path}: {e}")
            return {}
    
    def validate_config(self) -> bool:
        """Validate the configuration file"""
        if not self.config:
            return False
        
        required_fields = ["dependencies", "graphs"]
        missing_fields = [field for field in required_fields if field not in self.config]
        
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
        
        print("✅ Configuration is valid!")
        return True
    
    def list_graphs(self) -> None:
        """List all configured graphs"""
        if "graphs" not in self.config:
            print("❌ No graphs configured")
            return
        
        print("📊 Configured Graphs:")
        for graph_name, graph_path in self.config["graphs"].items():
            print(f"  • {graph_name}: {graph_path}")
    
    def check_dependencies(self) -> None:
        """Check if dependencies are available"""
        if "dependencies" not in self.config:
            print("❌ No dependencies configured")
            return
        
        print("📦 Checking Dependencies:")
        for dep in self.config["dependencies"]:
            if dep == ".":
                print(f"  ✅ {dep} (local package)")
            elif os.path.exists(dep):
                print(f"  ✅ {dep} (local path)")
            else:
                try:
                    import importlib
                    importlib.import_module(dep.replace("-", "_"))
                    print(f"  ✅ {dep} (installed)")
                except ImportError:
                    print(f"  ❌ {dep} (not found)")
    
    def run_langgraph_command(self, command: List[str]) -> None:
        """Run a langgraph CLI command"""
        try:
            cmd = ["langgraph"] + command
            print(f"🚀 Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ Command completed successfully!")
                if result.stdout:
                    print(result.stdout)
            else:
                print(f"❌ Command failed with exit code {result.returncode}")
                if result.stderr:
                    print(result.stderr)
        except FileNotFoundError:
            print("❌ LangGraph CLI not found. Install with: pip install langgraph-cli")
    
    def dev_server(self, port: int = 8123) -> None:
        """Start the development server"""
        self.run_langgraph_command(["dev", "--port", str(port)])
    
    def build(self) -> None:
        """Build the LangGraph application"""
        self.run_langgraph_command(["build"])
    
    def deploy(self, deployment_name: Optional[str] = None) -> None:
        """Deploy the LangGraph application"""
        cmd = ["deploy"]
        if deployment_name:
            cmd.append(deployment_name)
        self.run_langgraph_command(cmd)
    
    def show_info(self) -> None:
        """Show configuration information"""
        print("🔧 LangGraph Configuration Info:")
        print(f"  Config file: {self.config_path}")
        print(f"  Python version: {self.config.get('python_version', '3.11')}")
        print(f"  Environment file: {self.config.get('env', 'Not specified')}")
        
        if "store" in self.config:
            print("  Store configuration: ✅")
        if "checkpointer" in self.config:
            print("  Checkpointer configuration: ✅")
        if "http" in self.config:
            print("  HTTP configuration: ✅")

def main():
    """Main CLI interface"""
    helper = LangGraphCLIHelper()
    
    if len(sys.argv) < 2:
        print("🤖 LangGraph CLI Helper")
        print("\nUsage:")
        print("  python langgraph_cli_helper.py <command>")
        print("\nCommands:")
        print("  validate    - Validate configuration")
        print("  info        - Show configuration info")
        print("  graphs      - List configured graphs")
        print("  deps        - Check dependencies")
        print("  dev         - Start development server")
        print("  build       - Build application")
        print("  deploy      - Deploy application")
        print("  test        - Test the running workflow")
        print("  test full   - Test the full multi-agent workflow")
        return
    
    command = sys.argv[1].lower()
    
    if command == "validate":
        helper.validate_config()
    elif command == "info":
        helper.show_info()
    elif command == "graphs":
        helper.list_graphs()
    elif command == "deps":
        helper.check_dependencies()
    elif command == "dev":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8123
        helper.dev_server(port)
    elif command == "build":
        helper.build()
    elif command == "deploy":
        deployment_name = sys.argv[2] if len(sys.argv) > 2 else None
        helper.deploy(deployment_name)
    elif command == "test":
        test_type = sys.argv[2] if len(sys.argv) > 2 else "simple"
        if test_type == "full":
            print("🧪 Running full workflow test...")
            import subprocess
            subprocess.run([sys.executable, "test_full_workflow.py"])
        else:
            print("🧪 Running simple workflow test...")
            import subprocess
            subprocess.run([sys.executable, "test_langgraph_cli.py"])
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()