#!/usr/bin/env python3
"""
Example usage of the unified configuration system.

This script demonstrates how to use the new centralized configuration
system that replaces the old config.py and integrates with langgraph.
"""

import sys
import os
sys.path.append('src')

from config import get_config, get_agent_config, get_model_config, Config

def main():
    """Demonstrate configuration system usage."""
    
    print("=== Unified Configuration System Demo ===\n")
    
    # 1. Get the complete system configuration
    print("1. Loading system configuration...")
    config = get_config()
    print(f"   Environment: {config.environment}")
    print(f"   Debug mode: {config.debug}")
    print(f"   Default LLM provider: {config.default_llm_provider}")
    
    # 2. Access agent configurations
    print("\n2. Agent configurations:")
    for agent_name in ['planner_agent', 'code_generator_agent', 'rag_agent']:
        agent_config = get_agent_config(agent_name)
        if agent_config:
            print(f"   {agent_name}:")
            print(f"     - Temperature: {agent_config.temperature}")
            print(f"     - Max retries: {agent_config.max_retries}")
            print(f"     - Timeout: {agent_config.timeout_seconds}s")
            print(f"     - Tools: {', '.join(agent_config.tools)}")
    
    # 3. Get model configurations
    print("\n3. Model configurations:")
    models = ['openai/gpt-4o', 'openai/gpt-4o-mini']
    for model in models:
        model_config = get_model_config(model)
        print(f"   {model}:")
        print(f"     - Provider: {model_config.get('provider', 'N/A')}")
        print(f"     - API key present: {'Yes' if model_config.get('api_key') else 'No'}")
        print(f"     - Timeout: {model_config.get('timeout', 'N/A')}s")
    
    # 4. RAG configuration
    print("\n4. RAG configuration:")
    if config.rag_config:
        print(f"   Enabled: {config.rag_config.enabled}")
        print(f"   Embedding provider: {config.rag_config.embedding_config.provider}")
        print(f"   Vector store: {config.rag_config.vector_store_config.provider}")
        print(f"   Chunk size: {config.rag_config.chunk_size}")
    else:
        print("   RAG not configured")
    
    # 5. Workflow configuration
    print("\n5. Workflow configuration:")
    workflow = config.workflow_config
    print(f"   Output directory: {workflow.output_dir}")
    print(f"   Max scene concurrency: {workflow.max_scene_concurrency}")
    print(f"   Default quality: {workflow.default_quality}")
    print(f"   GPU acceleration: {workflow.use_gpu_acceleration}")
    
    # 6. Legacy compatibility
    print("\n6. Legacy Config compatibility:")
    legacy_config = Config()
    print(f"   OUTPUT_DIR: {legacy_config.OUTPUT_DIR}")
    print(f"   EMBEDDING_MODEL: {legacy_config.EMBEDDING_MODEL}")
    print(f"   CHROMA_DB_PATH: {legacy_config.CHROMA_DB_PATH}")
    
    # 7. External tool configurations
    print("\n7. External tool configurations:")
    print(f"   Docling enabled: {config.docling_config.enabled}")
    print(f"   Context7 enabled: {config.context7_config.enabled}")
    print(f"   Human loop enabled: {config.human_loop_config.enabled}")
    
    # 8. MCP servers
    print("\n8. MCP servers:")
    for server_name, server_config in config.mcp_servers.items():
        print(f"   {server_name}:")
        print(f"     - Command: {server_config.command}")
        print(f"     - Disabled: {server_config.disabled}")
        print(f"     - Auto-approve: {', '.join(server_config.auto_approve)}")
    
    print("\n=== Configuration Demo Complete ===")

if __name__ == "__main__":
    main()