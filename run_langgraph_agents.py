#!/usr/bin/env python3
"""
LangGraph Agents Video Generation Runner
This script runs the complete LangGraph agent workflow from user input to video output.
Similar to generate_video.py but using the new multi-agent system.
"""

import os
import sys
import asyncio
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from langgraph_agents.graph import VideoGenerationWorkflow
from langgraph_agents.state import SystemConfig, AgentConfig, create_initial_state


def create_comprehensive_system_config(
    planner_model: str = "openai/gpt-4o-mini",
    scene_model: Optional[str] = None,
    helper_model: Optional[str] = None,
    output_dir: str = "output",
    use_rag: bool = False,
    use_visual_fix: bool = False,
    enable_aws: bool = False
) -> SystemConfig:
    """Create a comprehensive system configuration for all agents."""
    
    # Use same model for all if not specified
    scene_model = scene_model or planner_model
    helper_model = helper_model or planner_model
    
    # Define all available agents
    agent_configs = {
        "planner_agent": AgentConfig(
            name="planner_agent",
            model_config={
                "model_name": planner_model,
                "temperature": 0.7,
                "max_tokens": 4000
            },
            tools=["planning", "scene_generation", "plugin_detection"],
            max_retries=3,
            timeout_seconds=600,
            enable_human_loop=False,
            planner_model=planner_model,
            helper_model=helper_model
        ),
        
        "code_generator_agent": AgentConfig(
            name="code_generator_agent",
            model_config={
                "model_name": scene_model,
                "temperature": 0.7,
                "max_tokens": 6000
            },
            tools=["code_generation", "manim", "error_fixing"],
            max_retries=3,
            timeout_seconds=900,
            enable_human_loop=False,
            scene_model=scene_model,
            helper_model=helper_model
        ),
        
        "renderer_agent": AgentConfig(
            name="renderer_agent",
            model_config={
                "model_name": helper_model,
                "temperature": 0.7,
                "max_tokens": 2000
            },
            tools=["video_rendering", "manim_execution", "video_combination"],
            max_retries=2,
            timeout_seconds=1800,
            enable_human_loop=False
        ),
        
        "error_handler_agent": AgentConfig(
            name="error_handler_agent",
            model_config={
                "model_name": helper_model,
                "temperature": 0.7,
                "max_tokens": 3000
            },
            tools=["error_analysis", "recovery_planning"],
            max_retries=2,
            timeout_seconds=300,
            enable_human_loop=False
        ),
        
        "monitoring_agent": AgentConfig(
            name="monitoring_agent",
            model_config={
                "model_name": helper_model,
                "temperature": 0.7,
                "max_tokens": 2000
            },
            tools=["performance_monitoring", "system_analysis"],
            max_retries=1,
            timeout_seconds=120,
            enable_human_loop=False
        )
    }
    
    # Add RAG agent if enabled
    if use_rag:
        agent_configs["rag_agent"] = AgentConfig(
            name="rag_agent",
            model_config={
                "model_name": helper_model,
                "temperature": 0.7,
                "max_tokens": 4000
            },
            tools=["document_retrieval", "context_generation"],
            max_retries=2,
            timeout_seconds=300,
            enable_human_loop=False,
            helper_model=helper_model
        )
    
    # Add visual analysis agent if enabled
    if use_visual_fix:
        agent_configs["visual_analysis_agent"] = AgentConfig(
            name="visual_analysis_agent",
            model_config={
                "model_name": scene_model,
                "temperature": 0.7,
                "max_tokens": 3000
            },
            tools=["visual_analysis", "image_processing"],
            max_retries=2,
            timeout_seconds=600,
            enable_human_loop=False,
            scene_model=scene_model
        )
    
    # LLM Provider configurations
    llm_providers = {
        "openai": {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1",
            "timeout": 60,
            "max_retries": 3
        }
    }
    
    # Add other providers if API keys are available
    if os.getenv("ANTHROPIC_API_KEY"):
        llm_providers["anthropic"] = {
            "api_key": os.getenv("ANTHROPIC_API_KEY"),
            "base_url": "https://api.anthropic.com",
            "timeout": 60,
            "max_retries": 3
        }
    
    if os.getenv("OPENROUTER_API_KEY"):
        llm_providers["openrouter"] = {
            "api_key": os.getenv("OPENROUTER_API_KEY"),
            "base_url": "https://openrouter.ai/api/v1",
            "timeout": 60,
            "max_retries": 3
        }
    
    # External service configurations
    docling_config = {
        "enabled": False,  # Disable for simplicity
        "endpoint": "",
        "api_key": ""
    }
    
    mcp_servers = {}  # No MCP servers for basic test
    
    # Monitoring configuration
    monitoring_config = {
        "enable_monitoring": True,
        "log_level": "INFO",
        "performance_tracking": True,
        "error_tracking": True,
        "metrics_collection": True
    }
    
    # Human loop configuration
    human_loop_config = {
        "enable_interrupts": False,
        "auto_approve": True,
        "timeout_seconds": 300,
        "escalation_threshold": 3
    }
    
    return SystemConfig(
        agents=agent_configs,
        llm_providers=llm_providers,
        docling_config=docling_config,
        mcp_servers=mcp_servers,
        monitoring_config=monitoring_config,
        human_loop_config=human_loop_config,
        max_workflow_retries=3,
        workflow_timeout_seconds=3600,
        enable_checkpoints=True,
        checkpoint_interval=300
    )


async def run_video_generation_workflow(
    topic: str,
    description: str,
    session_id: str,
    config_overrides: Dict[str, Any] = None
) -> Dict[str, Any]:
    """Run the complete video generation workflow using LangGraph agents."""
    
    print(f"🎬 Starting LangGraph Video Generation Workflow")
    print(f"   Topic: {topic}")
    print(f"   Description: {description}")
    print(f"   Session ID: {session_id}")
    print("=" * 80)
    
    # Create system configuration
    system_config = create_comprehensive_system_config(
        planner_model=config_overrides.get("planner_model", "openai/gpt-4o-mini"),
        scene_model=config_overrides.get("scene_model"),
        helper_model=config_overrides.get("helper_model"),
        output_dir=config_overrides.get("output_dir", "output"),
        use_rag=config_overrides.get("use_rag", False),
        use_visual_fix=config_overrides.get("use_visual_fix", False),
        enable_aws=config_overrides.get("enable_aws", False)
    )
    
    # Initialize workflow
    print("🔧 Initializing LangGraph workflow...")
    try:
        workflow = VideoGenerationWorkflow(system_config)
        print("✅ Workflow initialized successfully")
        print(f"   Available agents: {list(workflow.agents.keys())}")
    except Exception as e:
        print(f"❌ Failed to initialize workflow: {e}")
        raise
    
    # Prepare runtime configuration
    runtime_config = {
        "output_dir": config_overrides.get("output_dir", "output"),
        "use_rag": config_overrides.get("use_rag", False),
        "use_visual_fix_code": config_overrides.get("use_visual_fix", False),
        "max_scene_concurrency": config_overrides.get("max_scene_concurrency", 2),
        "max_retries": config_overrides.get("max_retries", 3),
        "enable_caching": config_overrides.get("enable_caching", True),
        "default_quality": config_overrides.get("quality", "medium"),
        "preview_mode": config_overrides.get("preview_mode", False),
        "only_plan": config_overrides.get("only_plan", False)
    }
    
    print(f"\n⚙️ Runtime Configuration:")
    for key, value in runtime_config.items():
        print(f"   {key}: {value}")
    
    # Execute workflow
    print(f"\n🚀 Executing workflow...")
    try:
        final_state = await workflow.invoke(
            topic=topic,
            description=description,
            session_id=session_id,
            config=runtime_config
        )
        
        print(f"\n✅ Workflow completed successfully!")
        return final_state
        
    except Exception as e:
        print(f"\n❌ Workflow execution failed: {e}")
        import traceback
        traceback.print_exc()
        raise


async def run_streaming_workflow(
    topic: str,
    description: str,
    session_id: str,
    config_overrides: Dict[str, Any] = None
) -> None:
    """Run the workflow with streaming output to show progress."""
    
    print(f"🌊 Starting Streaming LangGraph Workflow")
    print(f"   Topic: {topic}")
    print(f"   Description: {description}")
    print("=" * 80)
    
    # Create system configuration
    system_config = create_comprehensive_system_config(
        planner_model=config_overrides.get("planner_model", "openai/gpt-4o-mini"),
        output_dir=config_overrides.get("output_dir", "output"),
        use_rag=config_overrides.get("use_rag", False),
        use_visual_fix=config_overrides.get("use_visual_fix", False)
    )
    
    # Initialize workflow
    workflow = VideoGenerationWorkflow(system_config)
    
    # Runtime configuration
    runtime_config = {
        "output_dir": config_overrides.get("output_dir", "output"),
        "use_rag": config_overrides.get("use_rag", False),
        "max_scene_concurrency": 1,
        "preview_mode": True,
        "only_plan": config_overrides.get("only_plan", False)
    }
    
    # Stream workflow execution
    chunk_count = 0
    try:
        async for chunk in workflow.stream(
            topic=topic,
            description=description,
            session_id=session_id,
            config=runtime_config
        ):
            chunk_count += 1
            
            print(f"\n📦 Chunk {chunk_count} - Status: {chunk.get('workflow_status', 'running')}")
            
            # Show agent updates
            for agent_name, state_update in chunk.items():
                if isinstance(state_update, dict) and agent_name not in [
                    'workflow_status', 'elapsed_time', 'timestamp', 'chunk_number', 'session_id'
                ]:
                    print(f"   🤖 {agent_name}: Processing...")
                    
                    # Show key state updates
                    if 'scene_outline' in state_update:
                        print(f"      📋 Scene outline generated ({len(state_update['scene_outline'])} chars)")
                    
                    if 'scene_implementations' in state_update:
                        impl_count = len([v for v in state_update['scene_implementations'].values() if v])
                        print(f"      🎯 Scene implementations: {impl_count}")
                    
                    if 'generated_code' in state_update:
                        code_count = len([v for v in state_update['generated_code'].values() if v])
                        print(f"      💻 Generated code: {code_count} scenes")
                    
                    if 'rendered_videos' in state_update:
                        video_count = len([v for v in state_update['rendered_videos'].values() if v])
                        print(f"      🎥 Rendered videos: {video_count} scenes")
                    
                    if 'combined_video_path' in state_update and state_update['combined_video_path']:
                        print(f"      🎬 Combined video: {state_update['combined_video_path']}")
            
            # Show timing info
            if 'elapsed_time' in chunk:
                print(f"   ⏱️ Elapsed time: {chunk['elapsed_time']:.2f}s")
            
            # Check for completion
            if chunk.get('workflow_status') == 'completed':
                print(f"\n🎉 Workflow completed successfully!")
                break
            elif chunk.get('workflow_status') == 'error':
                print(f"\n❌ Workflow encountered an error")
                break
                
    except Exception as e:
        print(f"\n❌ Streaming workflow failed: {e}")
        raise


def display_workflow_results(final_state: Dict[str, Any]) -> None:
    """Display the results of the workflow execution."""
    
    print(f"\n📊 Workflow Results Summary")
    print("=" * 80)
    
    # Basic info
    print(f"Topic: {final_state.get('topic', 'Unknown')}")
    print(f"Description: {final_state.get('description', 'Unknown')}")
    print(f"Session ID: {final_state.get('session_id', 'Unknown')}")
    print(f"Workflow Complete: {final_state.get('workflow_complete', False)}")
    print(f"Current Agent: {final_state.get('current_agent', 'None')}")
    print(f"Error Count: {final_state.get('error_count', 0)}")
    
    # Execution time
    if 'total_execution_time' in final_state:
        print(f"Total Execution Time: {final_state['total_execution_time']:.2f}s")
    
    # Planning results
    if final_state.get('scene_outline'):
        print(f"\n📋 Planning Results:")
        print(f"   Scene outline: {len(final_state['scene_outline'])} characters")
        
        if final_state.get('scene_implementations'):
            impl_count = len([v for v in final_state['scene_implementations'].values() if v])
            total_scenes = len(final_state['scene_implementations'])
            print(f"   Scene implementations: {impl_count}/{total_scenes}")
    
    # Code generation results
    if final_state.get('generated_code'):
        print(f"\n💻 Code Generation Results:")
        code_scenes = len([v for v in final_state['generated_code'].values() if v])
        total_scenes = len(final_state['generated_code'])
        print(f"   Generated code: {code_scenes}/{total_scenes} scenes")
        
        for scene_num, code in final_state['generated_code'].items():
            if code:
                print(f"   Scene {scene_num}: {len(code)} characters")
    
    # Rendering results
    if final_state.get('rendered_videos'):
        print(f"\n🎥 Rendering Results:")
        video_count = len([v for v in final_state['rendered_videos'].values() if v])
        total_scenes = len(final_state['rendered_videos'])
        print(f"   Rendered videos: {video_count}/{total_scenes} scenes")
        
        for scene_num, video_path in final_state['rendered_videos'].items():
            if video_path:
                print(f"   Scene {scene_num}: {video_path}")
    
    # Combined video
    if final_state.get('combined_video_path'):
        print(f"\n🎬 Final Output:")
        print(f"   Combined video: {final_state['combined_video_path']}")
    
    # Errors
    if final_state.get('escalated_errors'):
        print(f"\n⚠️ Errors:")
        for i, error in enumerate(final_state['escalated_errors'], 1):
            print(f"   {i}. {error}")
    
    # Performance metrics
    if final_state.get('performance_metrics'):
        print(f"\n📈 Performance Metrics:")
        metrics = final_state['performance_metrics']
        for key, value in metrics.items():
            print(f"   {key}: {value}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Run LangGraph Agents for Video Generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic video generation
  python run_langgraph_agents.py --topic "Python Functions" --description "Educational video about Python functions"
  
  # Planning only
  python run_langgraph_agents.py --topic "Math Basics" --description "Basic math operations" --only-plan
  
  # With streaming output
  python run_langgraph_agents.py --topic "Data Structures" --description "Arrays and lists" --stream
  
  # With RAG enabled
  python run_langgraph_agents.py --topic "Machine Learning" --description "ML basics" --use-rag
        """
    )
    
    # Required arguments
    parser.add_argument("--topic", required=True, help="Video topic")
    parser.add_argument("--description", required=True, help="Video description")
    
    # Optional arguments
    parser.add_argument("--session-id", help="Session ID (auto-generated if not provided)")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    
    # Model configuration
    parser.add_argument("--planner-model", default="openai/gpt-4o-mini", help="Planner model")
    parser.add_argument("--scene-model", help="Scene model (defaults to planner model)")
    parser.add_argument("--helper-model", help="Helper model (defaults to planner model)")
    
    # Workflow options
    parser.add_argument("--only-plan", action="store_true", help="Only generate planning")
    parser.add_argument("--stream", action="store_true", help="Use streaming workflow")
    parser.add_argument("--use-rag", action="store_true", help="Enable RAG")
    parser.add_argument("--use-visual-fix", action="store_true", help="Enable visual fixes")
    parser.add_argument("--enable-aws", action="store_true", help="Enable AWS integration")
    
    # Performance options
    parser.add_argument("--max-scene-concurrency", type=int, default=2, help="Max concurrent scenes")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries")
    parser.add_argument("--quality", choices=["preview", "low", "medium", "high"], default="medium", help="Render quality")
    
    # Flags
    parser.add_argument("--preview-mode", action="store_true", help="Enable preview mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    return parser


async def main():
    """Main function to run the LangGraph agents workflow."""
    
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is required")
        print("Please set your OpenAI API key:")
        print("export OPENAI_API_KEY='your-key-here'")
        return 1
    
    # Generate session ID if not provided
    session_id = args.session_id or f"langgraph_{int(asyncio.get_event_loop().time())}"
    
    # Prepare configuration overrides
    config_overrides = {
        "planner_model": args.planner_model,
        "scene_model": args.scene_model,
        "helper_model": args.helper_model,
        "output_dir": args.output_dir,
        "use_rag": args.use_rag,
        "use_visual_fix": args.use_visual_fix,
        "enable_aws": args.enable_aws,
        "max_scene_concurrency": args.max_scene_concurrency,
        "max_retries": args.max_retries,
        "quality": args.quality,
        "preview_mode": args.preview_mode,
        "only_plan": args.only_plan,
        "enable_caching": True
    }
    
    try:
        if args.stream:
            # Run streaming workflow
            await run_streaming_workflow(
                topic=args.topic,
                description=args.description,
                session_id=session_id,
                config_overrides=config_overrides
            )
        else:
            # Run standard workflow
            final_state = await run_video_generation_workflow(
                topic=args.topic,
                description=args.description,
                session_id=session_id,
                config_overrides=config_overrides
            )
            
            # Display results
            display_workflow_results(final_state)
        
        print(f"\n🎉 LangGraph agents workflow completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print(f"\n⚠️ Workflow interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)