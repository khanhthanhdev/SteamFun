"""
VideoGenerationState TypedDict for LangGraph multi-agent system.
Preserves current state structure while adding LangGraph compatibility.
"""

from typing import Annotated, List, Optional, Dict, Any, Union
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from dataclasses import dataclass
from datetime import datetime


class VideoGenerationState(TypedDict):
    """Central state for the video generation workflow.
    
    This state preserves the current workflow structure while adding
    LangGraph compatibility for multi-agent coordination.
    """
    
    # Core workflow data - compatible with existing initialization parameters
    messages: Annotated[List[Any], add_messages]
    topic: str
    description: str
    session_id: str
    
    # Configuration parameters - preserving existing initialization structure
    output_dir: str
    print_response: bool
    use_rag: bool
    use_context_learning: bool
    context_learning_path: str
    chroma_db_path: str
    manim_docs_path: str
    embedding_model: str
    use_visual_fix_code: bool
    use_langfuse: bool
    max_scene_concurrency: int
    max_topic_concurrency: int
    max_retries: int
    
    # Enhanced RAG Configuration - preserving existing structure
    use_enhanced_rag: bool
    enable_rag_caching: bool
    enable_quality_monitoring: bool
    enable_error_handling: bool
    rag_cache_ttl: int
    rag_max_cache_size: int
    rag_performance_threshold: float
    rag_quality_threshold: float
    
    # Renderer optimizations - preserving existing structure
    enable_caching: bool
    default_quality: str
    use_gpu_acceleration: bool
    preview_mode: bool
    max_concurrent_renders: int
    
    # Planning state - compatible with EnhancedVideoPlanner
    scene_outline: Optional[str]
    scene_implementations: Dict[int, str]
    detected_plugins: List[str]
    
    # Code generation state - compatible with CodeGenerator
    generated_code: Dict[int, str]
    code_errors: Dict[int, str]
    rag_context: Dict[str, Any]
    
    # Rendering state - compatible with OptimizedVideoRenderer
    rendered_videos: Dict[int, str]
    combined_video_path: Optional[str]
    rendering_errors: Dict[int, str]
    
    # Visual analysis state - compatible with existing visual analysis
    visual_analysis_results: Dict[int, Dict[str, Any]]
    visual_errors: Dict[int, List[str]]
    
    # Error handling state
    error_count: int
    retry_count: Dict[str, int]
    escalated_errors: List[Dict[str, Any]]
    
    # Human loop state
    pending_human_input: Optional[Dict[str, Any]]
    human_feedback: Optional[Dict[str, Any]]
    
    # Monitoring state
    performance_metrics: Dict[str, Any]
    execution_trace: List[Dict[str, Any]]
    
    # Current agent tracking
    current_agent: Optional[str]
    next_agent: Optional[str]
    
    # Workflow control
    workflow_complete: bool
    workflow_interrupted: bool


@dataclass
class AgentConfig:
    """Configuration for individual agents.
    
    Maintains compatibility with existing model configurations while
    adding LangGraph-specific settings.
    """
    name: str
    model_config: Dict[str, Any]  # Compatible with existing model wrapper configs
    tools: List[str]
    max_retries: int = 3
    timeout_seconds: int = 300
    enable_human_loop: bool = False
    
    # LLM Provider configurations - compatible with existing structure
    planner_model: Optional[str] = None
    scene_model: Optional[str] = None
    helper_model: Optional[str] = None
    
    # Model wrapper settings - preserving existing patterns
    temperature: float = 0.7
    print_cost: bool = True
    verbose: bool = False


@dataclass
class SystemConfig:
    """Overall system configuration.
    
    Preserves existing configuration patterns while adding multi-agent support.
    """
    agents: Dict[str, AgentConfig]
    
    # LLM Provider configurations - compatible with existing providers
    llm_providers: Dict[str, Dict[str, Any]]  # AWS Bedrock, OpenAI configs
    
    # External tool configurations
    docling_config: Dict[str, Any]
    mcp_servers: Dict[str, Dict[str, Any]]
    
    # System monitoring
    monitoring_config: Dict[str, Any]
    human_loop_config: Dict[str, Any]
    
    # Workflow settings
    max_workflow_retries: int = 3
    workflow_timeout_seconds: int = 3600
    enable_checkpoints: bool = True
    checkpoint_interval: int = 300  # seconds


@dataclass
class AgentError:
    """Structured error information for agent failures."""
    agent_name: str
    error_type: str
    error_message: str
    context: Dict[str, Any]
    timestamp: datetime
    retry_count: int
    stack_trace: Optional[str] = None


@dataclass
class RecoveryStrategy:
    """Error recovery strategy configuration."""
    error_pattern: str
    recovery_agent: str
    max_attempts: int
    escalation_threshold: int
    use_rag: bool = False
    use_alternative_quality: bool = False
    use_fallback_prompt: bool = False


def create_initial_state(
    topic: str,
    description: str,
    session_id: str,
    config: SystemConfig
) -> VideoGenerationState:
    """Create initial state for video generation workflow.
    
    Preserves existing initialization parameter structure while
    setting up LangGraph state management.
    """
    # Extract configuration values with defaults matching existing system
    agent_config = list(config.agents.values())[0] if config.agents else AgentConfig("default", {}, [])
    
    return VideoGenerationState(
        # Core workflow data
        messages=[],
        topic=topic,
        description=description,
        session_id=session_id,
        
        # Configuration parameters - preserving existing defaults
        output_dir="output",
        print_response=agent_config.verbose,
        use_rag=True,
        use_context_learning=True,
        context_learning_path="data/context_learning",
        chroma_db_path="data/rag/chroma_db",
        manim_docs_path="data/rag/manim_docs",
        embedding_model="hf:ibm-granite/granite-embedding-30m-english",
        use_visual_fix_code=False,
        use_langfuse=True,
        max_scene_concurrency=5,
        max_topic_concurrency=1,
        max_retries=5,
        
        # Enhanced RAG Configuration - preserving existing defaults
        use_enhanced_rag=True,
        enable_rag_caching=True,
        enable_quality_monitoring=True,
        enable_error_handling=True,
        rag_cache_ttl=3600,
        rag_max_cache_size=1000,
        rag_performance_threshold=2.0,
        rag_quality_threshold=0.7,
        
        # Renderer optimizations - preserving existing defaults
        enable_caching=True,
        default_quality="medium",
        use_gpu_acceleration=False,
        preview_mode=False,
        max_concurrent_renders=4,
        
        # Planning state
        scene_outline=None,
        scene_implementations={},
        detected_plugins=[],
        
        # Code generation state
        generated_code={},
        code_errors={},
        rag_context={},
        
        # Rendering state
        rendered_videos={},
        combined_video_path=None,
        rendering_errors={},
        
        # Visual analysis state
        visual_analysis_results={},
        visual_errors={},
        
        # Error handling state
        error_count=0,
        retry_count={},
        escalated_errors=[],
        
        # Human loop state
        pending_human_input=None,
        human_feedback=None,
        
        # Monitoring state
        performance_metrics={},
        execution_trace=[],
        
        # Current agent tracking
        current_agent=None,
        next_agent="planner_agent",
        
        # Workflow control
        workflow_complete=False,
        workflow_interrupted=False
    )