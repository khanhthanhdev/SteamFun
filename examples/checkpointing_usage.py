"""
Example usage of checkpointing and persistence functionality.

This example demonstrates how to use the checkpointing system for
development and production environments.
"""

import asyncio
import os
from datetime import datetime

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.langgraph_agents.workflow_graph import create_workflow
from src.langgraph_agents.models.config import WorkflowConfig
from src.langgraph_agents.models.state import VideoGenerationState
from src.langgraph_agents.checkpointing import (
    CheckpointConfig,
    CheckpointBackend,
    create_checkpoint_manager
)


async def demonstrate_memory_checkpointing():
    """Demonstrate memory-based checkpointing for development."""
    print("=== Memory Checkpointing Demo ===")
    
    # Create workflow configuration
    config = WorkflowConfig(
        max_retries=3,
        timeout_seconds=60,
        use_rag=False,
        use_visual_analysis=False
    )
    
    # Create checkpoint configuration for memory
    checkpoint_config = CheckpointConfig(
        backend=CheckpointBackend.MEMORY,
        memory_max_size=100,  # Max 100 checkpoints
        enable_compression=True
    )
    
    # Create workflow with memory checkpointing
    workflow = create_workflow(
        config=config,
        use_checkpointing=True,
        checkpoint_config=checkpoint_config
    )
    
    # Display checkpoint information
    info = workflow.get_checkpoint_info()
    print(f"Checkpointing enabled: {info['checkpointing_enabled']}")
    print(f"Backend: {info['backend']}")
    print(f"Persistent: {info['persistent']}")
    print(f"Compression enabled: {info['compression_enabled']}")
    
    # Get initial checkpoint statistics
    stats = await workflow.get_checkpoint_stats()
    print(f"Initial checkpoints: {stats['total_checkpoints']}")
    
    # Create a sample state
    sample_state = VideoGenerationState(
        topic="Python Programming Basics",
        description="An educational video about Python programming fundamentals",
        session_id="demo-session-001",
        scene_implementations={
            1: "Introduction to Python",
            2: "Variables and data types",
            3: "Control structures"
        },
        current_step="planning"
    )
    
    print(f"\nCreated sample state for session: {sample_state.session_id}")
    print(f"Current step: {sample_state.current_step}")
    print(f"Number of scenes: {len(sample_state.scene_implementations)}")
    
    return workflow, sample_state


async def demonstrate_checkpoint_recovery():
    """Demonstrate checkpoint recovery functionality."""
    print("\n=== Checkpoint Recovery Demo ===")
    
    config = WorkflowConfig(max_retries=3)
    
    # Create a recovery handler
    from src.langgraph_agents.checkpointing.recovery import create_recovery_handler
    recovery = create_recovery_handler(config)
    
    # Create a sample state with some issues
    problematic_state = VideoGenerationState(
        topic="Test Video",
        description="A test video with some issues",
        session_id="recovery-demo-001",
        scene_implementations={1: "Test scene"},
        current_step="code_generation",
        retry_counts={"code_generation": 2}  # Some retries
    )
    
    # Add some errors
    from src.langgraph_agents.models.errors import WorkflowError, ErrorType, ErrorSeverity
    error = WorkflowError(
        step="code_generation",
        error_type=ErrorType.CONTENT,
        message="Code generation failed",
        severity=ErrorSeverity.MEDIUM
    )
    problematic_state.add_error(error)
    
    # Analyze the checkpoint state
    print("Analyzing problematic checkpoint state...")
    analysis = recovery.analyze_checkpoint_state(problematic_state)
    
    print(f"Health score: {analysis['health_score']:.1f}/100")
    print(f"Recoverable: {analysis['recoverable']}")
    print(f"Issues found: {len(analysis['issues'])}")
    for issue in analysis['issues']:
        print(f"  - {issue}")
    
    print(f"Recommendations: {len(analysis['recommendations'])}")
    for rec in analysis['recommendations']:
        print(f"  - {rec}")
    
    # Determine recovery strategy
    print("\nDetermining recovery strategy...")
    decision = recovery.determine_recovery_strategy(problematic_state, analysis)
    
    print(f"Recovery strategy: {decision.strategy.value}")
    print(f"Reason: {decision.reason}")
    print(f"Requires user input: {decision.requires_user_input}")
    print(f"State modifications: {list(decision.state_modifications.keys()) if decision.state_modifications else 'None'}")
    
    # Apply recovery modifications
    if decision.state_modifications:
        print("\nApplying recovery modifications...")
        recovered_state = recovery.apply_recovery_modifications(problematic_state, decision)
        
        print(f"Errors after recovery: {len(recovered_state.errors)}")
        print(f"Retry counts after recovery: {len(recovered_state.retry_counts)}")
        print(f"Execution trace entries: {len(recovered_state.execution_trace)}")
    
    # Validate recovered state
    is_valid, issues = recovery.validate_recovery_state(
        recovered_state if decision.state_modifications else problematic_state
    )
    print(f"\nRecovered state valid: {is_valid}")
    if not is_valid:
        print("Validation issues:")
        for issue in issues:
            print(f"  - {issue}")
    
    return recovery


async def demonstrate_postgres_checkpointing():
    """Demonstrate PostgreSQL checkpointing (if available)."""
    print("\n=== PostgreSQL Checkpointing Demo ===")
    
    # Check if PostgreSQL connection is configured
    postgres_conn = os.getenv('POSTGRES_CONNECTION_STRING')
    if not postgres_conn:
        print("PostgreSQL connection string not configured (POSTGRES_CONNECTION_STRING)")
        print("Skipping PostgreSQL demo")
        return
    
    try:
        # Create PostgreSQL checkpoint configuration
        postgres_config = CheckpointConfig(
            backend=CheckpointBackend.POSTGRES,
            postgres_connection_string=postgres_conn,
            postgres_pool_size=5,
            postgres_max_overflow=10
        )
        
        # Create workflow configuration
        workflow_config = WorkflowConfig()
        
        # Create workflow with PostgreSQL checkpointing
        workflow = create_workflow(
            config=workflow_config,
            use_checkpointing=True,
            checkpoint_config=postgres_config
        )
        
        # Display checkpoint information
        info = workflow.get_checkpoint_info()
        print(f"Checkpointing enabled: {info['checkpointing_enabled']}")
        print(f"Backend: {info['backend']}")
        print(f"Persistent: {info['persistent']}")
        print(f"PostgreSQL available: {info['postgres_available']}")
        
        # Get checkpoint statistics
        try:
            stats = await workflow.get_checkpoint_stats()
            print(f"Total checkpoints: {stats.get('total_checkpoints', 'N/A')}")
            print(f"Connection healthy: {stats.get('connection_healthy', 'N/A')}")
        except Exception as e:
            print(f"Failed to get stats: {e}")
        
        # Test cleanup functionality
        try:
            cleaned = await workflow.cleanup_old_checkpoints(max_age_hours=24)
            print(f"Cleaned up {cleaned} old checkpoints")
        except Exception as e:
            print(f"Cleanup failed: {e}")
        
    except ImportError:
        print("PostgreSQL dependencies not available")
        print("Install with: pip install langgraph[postgres]")
    except Exception as e:
        print(f"PostgreSQL demo failed: {e}")


async def demonstrate_auto_backend_selection():
    """Demonstrate automatic backend selection."""
    print("\n=== Auto Backend Selection Demo ===")
    
    # Create configuration with auto backend
    auto_config = CheckpointConfig(backend=CheckpointBackend.AUTO)
    workflow_config = WorkflowConfig()
    
    # Create checkpoint manager
    manager = create_checkpoint_manager(workflow_config, auto_config)
    
    print(f"Auto-selected backend: {manager.backend_type.value}")
    print(f"Is persistent: {manager.is_persistent()}")
    
    # Get detailed info
    info = manager.get_checkpoint_info()
    print(f"PostgreSQL available: {info['postgres_available']}")
    print(f"Connection configured: {info['connection_configured']}")
    
    return manager


async def main():
    """Run all checkpointing demonstrations."""
    print("Checkpointing and Persistence Demo")
    print("=" * 50)
    
    try:
        # Memory checkpointing demo
        workflow, sample_state = await demonstrate_memory_checkpointing()
        
        # Recovery demo
        recovery = await demonstrate_checkpoint_recovery()
        
        # Auto backend selection demo
        manager = await demonstrate_auto_backend_selection()
        
        # PostgreSQL demo (if available)
        await demonstrate_postgres_checkpointing()
        
        print("\n=== Demo Complete ===")
        print("All checkpointing features demonstrated successfully!")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(main())