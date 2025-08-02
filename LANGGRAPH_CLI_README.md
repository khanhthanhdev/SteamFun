# LangGraph CLI Integration for Video Generation System

This document explains how to use the LangGraph CLI with your multi-agent video generation system.

## Quick Start

1. **Run the setup script:**
   ```bash
   python setup_langgraph.py
   ```

2. **Update your environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your actual API keys and configuration
   ```

3. **Start the development server:**
   ```bash
   python langgraph_cli_helper.py dev
   # Or directly: langgraph dev
   ```

4. **Access LangGraph Studio:**
   Open http://localhost:8123 in your browser

## Configuration Overview

The `langgraph.json` file configures your multi-agent system with the following key sections:

### Dependencies
```json
{
  "dependencies": [
    ".",
    "langchain_openai",
    "langchain_anthropic", 
    "langgraph",
    "boto3",
    "python-dotenv"
  ]
}
```

### Graph Definitions
```json
{
  "graphs": {
    "video_generation_workflow": "./src/langgraph_agents/graph.py:workflow",
    "planner_agent": "./src/langgraph_agents/agents/planner_agent.py:agent",
    "enhanced_code_generator": "./src/langgraph_agents/agents/enhanced_code_generator_agent.py:agent"
  }
}
```

### Store Configuration (RAG & Memory)
```json
{
  "store": {
    "index": {
      "embed": "openai:text-embedding-3-small",
      "dims": 1536,
      "fields": ["$"]
    },
    "ttl": {
      "refresh_on_read": true,
      "default_ttl": 1440,
      "sweep_interval_minutes": 60
    }
  }
}
```

## Available Graphs

Your system exposes the following graphs through the CLI:

| Graph Name | Description | File Path |
|------------|-------------|-----------|
| `video_generation_workflow` | Main workflow orchestrator | `./src/langgraph_agents/graph.py:workflow` |
| `planner_agent` | Video planning and scene generation | `./src/langgraph_agents/agents/planner_agent.py:agent` |
| `enhanced_code_generator` | Manim code generation with AWS integration | `./src/langgraph_agents/agents/enhanced_code_generator_agent.py:agent` |
| `enhanced_renderer` | Video rendering with S3 upload | `./src/langgraph_agents/agents/enhanced_renderer_agent.py:agent` |
| `error_handler_agent` | Error handling and recovery | `./src/langgraph_agents/agents/error_handler_agent.py:agent` |
| `monitoring_agent` | System monitoring and metrics | `./src/langgraph_agents/agents/monitoring_agent.py:agent` |

## CLI Helper Commands

Use the provided helper script for common tasks:

```bash
# Validate configuration
python langgraph_cli_helper.py validate

# Show configuration info
python langgraph_cli_helper.py info

# List all graphs
python langgraph_cli_helper.py graphs

# Check dependencies
python langgraph_cli_helper.py deps

# Start development server
python langgraph_cli_helper.py dev

# Start on custom port
python langgraph_cli_helper.py dev 8080

# Build for deployment
python langgraph_cli_helper.py build

# Deploy to LangGraph Cloud
python langgraph_cli_helper.py deploy my-video-gen-app
```

## Direct LangGraph CLI Commands

You can also use the LangGraph CLI directly:

```bash
# Start development server
langgraph dev

# Start on specific port
langgraph dev --port 8080

# Build the application
langgraph build

# Deploy to LangGraph Cloud
langgraph deploy

# Show help
langgraph --help
```

## Environment Variables

Key environment variables for your system:

### LLM Providers
```bash
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### AWS Configuration
```bash
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_DEFAULT_REGION=us-east-1
AWS_S3_BUCKET_NAME=your_s3_bucket
```

### LangChain Tracing
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langchain_api_key
LANGCHAIN_PROJECT=video-generation-agents
```

## Development Workflow

1. **Local Development:**
   ```bash
   langgraph dev
   ```
   - Access LangGraph Studio at http://localhost:8123
   - Test individual agents and workflows
   - Debug with visual graph representation

2. **Testing:**
   ```bash
   # Test individual components
   python -m pytest src/langgraph_agents/tests/
   
   # Test full workflow
   python test_langgraph_simple.py
   ```

3. **Building:**
   ```bash
   langgraph build
   ```

4. **Deployment:**
   ```bash
   langgraph deploy video-generation-system
   ```

## Docker Configuration

The configuration includes Docker setup for:
- FFmpeg for video processing
- OpenCV for image processing
- Manim for mathematical animations
- Required system libraries

```json
{
  "dockerfile_lines": [
    "RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 libxrender-dev libglib2.0-0",
    "RUN pip install manim opencv-python-headless pillow",
    "RUN mkdir -p /app/output /app/media /app/code"
  ]
}
```

## Troubleshooting

### Common Issues

1. **LangGraph CLI not found:**
   ```bash
   pip install langgraph-cli
   ```

2. **Configuration validation errors:**
   ```bash
   python langgraph_cli_helper.py validate
   ```

3. **Missing dependencies:**
   ```bash
   python langgraph_cli_helper.py deps
   pip install -r requirements.txt
   ```

4. **Port already in use:**
   ```bash
   langgraph dev --port 8124
   ```

### Debug Mode

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
langgraph dev
```

## Integration with Existing Code

Your existing code can still be used alongside the CLI:

```python
# Direct usage (existing approach)
from src.langgraph_agents.graph import VideoGenerationWorkflow

workflow = VideoGenerationWorkflow(system_config)
result = workflow.invoke("Create a video about calculus", "Educational video", "session_123")

# CLI-compatible usage (new approach)
# The same workflow is now accessible via LangGraph Studio and API
```

## Next Steps

1. **Explore LangGraph Studio:** Visual debugging and testing interface
2. **Set up CI/CD:** Automated testing and deployment
3. **Monitor in Production:** Use LangChain tracing and monitoring
4. **Scale with LangGraph Cloud:** Deploy to managed infrastructure

## Resources

- [LangGraph CLI Documentation](https://langchain-ai.github.io/langgraph/cloud/reference/cli/)
- [LangGraph Studio Guide](https://langchain-ai.github.io/langgraph/concepts/langgraph_studio/)
- [Deployment Guide](https://langchain-ai.github.io/langgraph/cloud/deployment/setup/)
- [Configuration Reference](https://langchain-ai.github.io/langgraph/cloud/reference/cli/#configuration-file)