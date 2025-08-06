"""
Centralized error handling system with recovery strategies.

This module implements the core ErrorHandler class and recovery strategies
as specified in the design document, providing a centralized approach to
error handling and recovery in the video generation workflow.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Type, Union
from enum import Enum

from ..models.state import VideoGenerationState
from ..models.errors import WorkflowError, ErrorType, ErrorSeverity, RecoveryStrategy as RecoveryStrategyModel, ErrorRecoveryResult
from ..models.config import WorkflowConfig

logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Actions that can be taken during error recovery."""
    RETRY = "retry"
    ESCALATE = "escalate"
    SKIP = "skip"
    ABORT = "abort"
    FALLBACK = "fallback"


class BaseRecoveryStrategy(ABC):
    """Base class for all recovery strategies."""
    
    def __init__(self, config: RecoveryStrategyModel):
        """Initialize the recovery strategy with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def can_handle(self, error: WorkflowError) -> bool:
        """Check if this strategy can handle the given error."""
        pass
    
    @abstractmethod
    async def execute_recovery(
        self, 
        error: WorkflowError, 
        state: VideoGenerationState,
        context: Dict[str, Any]
    ) -> ErrorRecoveryResult:
        """Execute the recovery strategy."""
        pass
    
    def should_escalate(self, error: WorkflowError, attempt_count: int) -> bool:
        """Determine if the error should be escalated."""
        return (
            attempt_count >= self.config.escalation_threshold or
            error.severity == ErrorSeverity.CRITICAL or
            not error.recoverable
        )
    
    def get_backoff_delay(self, attempt: int) -> float:
        """Calculate backoff delay for the given attempt."""
        delay = min(self.config.backoff_factor ** attempt, 60.0)  # Cap at 60 seconds
        return delay


class RetryStrategy(BaseRecoveryStrategy):
    """Recovery strategy that implements retry with exponential backoff."""
    
    def can_handle(self, error: WorkflowError) -> bool:
        """Check if this strategy can handle the given error."""
        return error.error_type in [
            ErrorType.TRANSIENT,
            ErrorType.TIMEOUT,
            ErrorType.RATE_LIMIT,
            ErrorType.MODEL
        ]
    
    async def execute_recovery(
        self, 
        error: WorkflowError, 
        state: VideoGenerationState,
        context: Dict[str, Any]
    ) -> ErrorRecoveryResult:
        """Execute retry with exponential backoff."""
        start_time = time.time()
        attempt_count = context.get('attempt_count', 0) + 1
        
        self.logger.info(f"Executing retry strategy for error {error.get_error_code()}, attempt {attempt_count}")
        
        # Check if we should escalate
        if self.should_escalate(error, attempt_count):
            return ErrorRecoveryResult(
                success=False,
                strategy_used=self.config.name,
                attempts_made=attempt_count,
                time_taken=time.time() - start_time,
                escalated=True,
                escalation_reason=f"Max attempts reached ({attempt_count})"
            )
        
        # Calculate backoff delay
        delay = self.get_backoff_delay(attempt_count - 1)
        
        if delay > 0:
            self.logger.info(f"Waiting {delay:.2f} seconds before retry")
            await asyncio.sleep(delay)
        
        # Update retry count in error
        error.increment_retry()
        
        # Update state retry count
        operation_key = f"{error.step}_{error.operation or 'unknown'}"
        state.increment_retry_count(operation_key)
        
        return ErrorRecoveryResult(
            success=True,
            strategy_used=self.config.name,
            attempts_made=attempt_count,
            time_taken=time.time() - start_time,
            recovery_data={
                'backoff_delay': delay,
                'next_attempt': attempt_count + 1
            }
        )


class RAGEnhancementStrategy(BaseRecoveryStrategy):
    """Recovery strategy that uses RAG to enhance content for content-related errors."""
    
    def __init__(self, config: RecoveryStrategyModel, rag_service=None):
        """Initialize with RAG service dependency."""
        super().__init__(config)
        self.rag_service = rag_service
    
    def can_handle(self, error: WorkflowError) -> bool:
        """Check if this strategy can handle the given error."""
        return (
            error.error_type in [ErrorType.CONTENT, ErrorType.MODEL] and
            self.config.use_rag and
            self.rag_service is not None
        )
    
    async def execute_recovery(
        self, 
        error: WorkflowError, 
        state: VideoGenerationState,
        context: Dict[str, Any]
    ) -> ErrorRecoveryResult:
        """Execute RAG enhancement recovery."""
        start_time = time.time()
        attempt_count = context.get('attempt_count', 0) + 1
        
        self.logger.info(f"Executing RAG enhancement strategy for error {error.get_error_code()}")
        
        try:
            # Extract relevant context for RAG enhancement
            query_context = self._build_rag_query(error, state)
            
            # Get enhanced context from RAG
            if self.rag_service:
                enhanced_context = await self.rag_service.get_enhanced_context(
                    query=query_context,
                    error_type=error.error_type,
                    scene_number=error.scene_number
                )
                
                # Update state with enhanced context
                if enhanced_context:
                    state.rag_context.update({
                        f"error_recovery_{error.get_error_code()}": enhanced_context,
                        "last_rag_enhancement": datetime.now().isoformat()
                    })
                    
                    self.logger.info(f"RAG enhancement successful, added context for error {error.get_error_code()}")
                    
                    return ErrorRecoveryResult(
                        success=True,
                        strategy_used=self.config.name,
                        attempts_made=attempt_count,
                        time_taken=time.time() - start_time,
                        recovery_data={
                            'enhanced_context_keys': list(enhanced_context.keys()) if isinstance(enhanced_context, dict) else ['context'],
                            'rag_query': query_context
                        }
                    )
            
            # RAG service not available or no enhancement possible
            return ErrorRecoveryResult(
                success=False,
                strategy_used=self.config.name,
                attempts_made=attempt_count,
                time_taken=time.time() - start_time,
                new_error=WorkflowError(
                    step=error.step,
                    error_type=ErrorType.SYSTEM,
                    message="RAG enhancement failed: service unavailable or no relevant context found",
                    context={'original_error': error.get_error_code()}
                )
            )
            
        except Exception as e:
            self.logger.error(f"RAG enhancement failed: {str(e)}")
            
            return ErrorRecoveryResult(
                success=False,
                strategy_used=self.config.name,
                attempts_made=attempt_count,
                time_taken=time.time() - start_time,
                new_error=WorkflowError(
                    step=error.step,
                    error_type=ErrorType.SYSTEM,
                    message=f"RAG enhancement failed: {str(e)}",
                    context={'original_error': error.get_error_code()}
                )
            )
    
    def _build_rag_query(self, error: WorkflowError, state: VideoGenerationState) -> str:
        """Build a query for RAG enhancement based on the error context."""
        query_parts = []
        
        # Add error context
        query_parts.append(f"Error in {error.step}: {error.message}")
        
        # Add scene context if available
        if error.scene_number and error.scene_number in state.scene_implementations:
            scene_impl = state.scene_implementations[error.scene_number]
            query_parts.append(f"Scene implementation: {scene_impl[:500]}...")  # Truncate for query
        
        # Add topic context
        query_parts.append(f"Video topic: {state.topic}")
        query_parts.append(f"Video description: {state.description}")
        
        return " | ".join(query_parts)


class FallbackModelStrategy(BaseRecoveryStrategy):
    """Recovery strategy that switches to fallback models when primary models fail."""
    
    def can_handle(self, error: WorkflowError) -> bool:
        """Check if this strategy can handle the given error."""
        return (
            error.error_type in [ErrorType.MODEL, ErrorType.RATE_LIMIT, ErrorType.TIMEOUT] and
            self.config.use_fallback_model and
            len(self.config.fallback_models) > 0
        )
    
    async def execute_recovery(
        self, 
        error: WorkflowError, 
        state: VideoGenerationState,
        context: Dict[str, Any]
    ) -> ErrorRecoveryResult:
        """Execute fallback model recovery."""
        start_time = time.time()
        attempt_count = context.get('attempt_count', 0) + 1
        
        self.logger.info(f"Executing fallback model strategy for error {error.get_error_code()}")
        
        # Get current model from context or config
        current_model = context.get('current_model') or self._get_current_model(error, state)
        
        # Find next fallback model
        fallback_model = self._get_next_fallback_model(current_model, context)
        
        if not fallback_model:
            return ErrorRecoveryResult(
                success=False,
                strategy_used=self.config.name,
                attempts_made=attempt_count,
                time_taken=time.time() - start_time,
                escalated=True,
                escalation_reason="No more fallback models available"
            )
        
        # Update configuration with fallback model
        self._update_model_config(error, state, fallback_model)
        
        self.logger.info(f"Switched to fallback model: {fallback_model}")
        
        return ErrorRecoveryResult(
            success=True,
            strategy_used=self.config.name,
            attempts_made=attempt_count,
            time_taken=time.time() - start_time,
            recovery_data={
                'fallback_model': fallback_model,
                'previous_model': current_model,
                'remaining_fallbacks': len(self.config.fallback_models) - (attempt_count % len(self.config.fallback_models))
            }
        )
    
    def _get_current_model(self, error: WorkflowError, state: VideoGenerationState) -> str:
        """Get the current model being used."""
        # Determine model based on the step where error occurred
        if error.step == "planning":
            return f"{state.config.planner_model.provider}/{state.config.planner_model.model_name}"
        elif error.step == "code_generation":
            return f"{state.config.code_model.provider}/{state.config.code_model.model_name}"
        else:
            return f"{state.config.planner_model.provider}/{state.config.planner_model.model_name}"  # Default fallback
    
    def _get_next_fallback_model(self, current_model: str, context: Dict[str, Any]) -> Optional[str]:
        """Get the next fallback model to try."""
        used_models = context.get('used_fallback_models', set())
        used_models.add(current_model)
        
        for model in self.config.fallback_models:
            if model not in used_models:
                context['used_fallback_models'] = used_models
                return model
        
        return None
    
    def _update_model_config(self, error: WorkflowError, state: VideoGenerationState, fallback_model: str):
        """Update the model configuration with the fallback model."""
        # Parse fallback model string (format: "provider/model_name")
        if "/" in fallback_model:
            provider, model_name = fallback_model.split("/", 1)
        else:
            provider = "openai"
            model_name = fallback_model
        
        if error.step == "planning":
            state.config.planner_model.provider = provider
            state.config.planner_model.model_name = model_name
        elif error.step == "code_generation":
            state.config.code_model.provider = provider
            state.config.code_model.model_name = model_name
        
        # Add fallback tracking to state
        fallback_info = {
            'fallback_model': fallback_model,
            'original_error': error.get_error_code(),
            'timestamp': datetime.now().isoformat()
        }
        
        # Use the extra field in the config model
        if not hasattr(state.config, '__pydantic_extra__'):
            state.config.__pydantic_extra__ = {}
        
        if 'model_fallbacks' not in state.config.__pydantic_extra__:
            state.config.__pydantic_extra__['model_fallbacks'] = []
        
        state.config.__pydantic_extra__['model_fallbacks'].append(fallback_info)


class ErrorHandler:
    """
    Centralized error handler that manages recovery strategies and coordinates
    error handling across the workflow.
    """
    
    def __init__(self, config: WorkflowConfig, rag_service=None):
        """Initialize the error handler with configuration and dependencies."""
        self.config = config
        self.rag_service = rag_service
        self.logger = logging.getLogger(__name__)
        
        # Initialize recovery strategies
        self.strategies: List[BaseRecoveryStrategy] = []
        self._initialize_strategies()
        
        # Recovery context tracking
        self.recovery_contexts: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info(f"ErrorHandler initialized with {len(self.strategies)} recovery strategies")
    
    def _initialize_strategies(self):
        """Initialize all recovery strategies."""
        # Retry strategy
        retry_config = RecoveryStrategyModel(
            name="retry_with_backoff",
            error_types=[ErrorType.TRANSIENT, ErrorType.TIMEOUT, ErrorType.RATE_LIMIT, ErrorType.MODEL],
            max_attempts=self.config.max_retries,
            backoff_factor=2.0,
            timeout_seconds=60,
            escalation_threshold=self.config.max_retries
        )
        self.strategies.append(RetryStrategy(retry_config))
        
        # RAG enhancement strategy
        if self.config.use_rag and self.rag_service:
            rag_config = RecoveryStrategyModel(
                name="rag_enhancement",
                error_types=[ErrorType.CONTENT, ErrorType.MODEL],
                max_attempts=2,
                backoff_factor=1.5,
                timeout_seconds=120,
                use_rag=True,
                escalation_threshold=2
            )
            self.strategies.append(RAGEnhancementStrategy(rag_config, self.rag_service))
        
        # Fallback model strategy
        fallback_models = getattr(self.config, 'fallback_models', [
            "openai/gpt-3.5-turbo",
            "anthropic/claude-3-haiku"
        ])
        
        if fallback_models:
            fallback_config = RecoveryStrategyModel(
                name="fallback_model",
                error_types=[ErrorType.MODEL, ErrorType.RATE_LIMIT, ErrorType.TIMEOUT],
                max_attempts=len(fallback_models),
                backoff_factor=1.0,
                timeout_seconds=90,
                use_fallback_model=True,
                fallback_models=fallback_models,
                escalation_threshold=len(fallback_models)
            )
            self.strategies.append(FallbackModelStrategy(fallback_config))
    
    async def handle_error(
        self, 
        error: WorkflowError, 
        state: VideoGenerationState
    ) -> ErrorRecoveryResult:
        """
        Handle an error using the appropriate recovery strategy.
        
        Args:
            error: The error to handle
            state: Current workflow state
            
        Returns:
            Recovery result indicating success/failure and actions taken
        """
        error_code = error.get_error_code()
        self.logger.info(f"Handling error: {error_code}")
        
        # Get or create recovery context for this error
        context = self.recovery_contexts.get(error_code, {})
        
        # Find appropriate recovery strategy
        strategy = self._find_recovery_strategy(error)
        
        if not strategy:
            self.logger.warning(f"No recovery strategy found for error: {error_code}")
            return ErrorRecoveryResult(
                success=False,
                strategy_used="none",
                attempts_made=1,
                time_taken=0.0,
                escalated=True,
                escalation_reason="No applicable recovery strategy found"
            )
        
        try:
            # Execute recovery strategy
            result = await strategy.execute_recovery(error, state, context)
            
            # Update recovery context
            self.recovery_contexts[error_code] = context
            
            # Add error to state if recovery failed
            if not result.success:
                state.add_error(error)
                
                # Mark error as resolved if recovery succeeded
                if result.success and not result.escalated:
                    error.mark_resolved(result.strategy_used)
            
            # Log result
            self.logger.info(
                f"Error recovery completed: {error_code}, "
                f"strategy: {result.strategy_used}, "
                f"success: {result.success}, "
                f"escalated: {result.escalated}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error recovery failed with exception: {str(e)}")
            
            # Create failure result
            return ErrorRecoveryResult(
                success=False,
                strategy_used=strategy.config.name,
                attempts_made=context.get('attempt_count', 0) + 1,
                time_taken=0.0,
                new_error=WorkflowError(
                    step=error.step,
                    error_type=ErrorType.SYSTEM,
                    message=f"Recovery strategy failed: {str(e)}",
                    context={'original_error': error_code}
                )
            )
    
    def _find_recovery_strategy(self, error: WorkflowError) -> Optional[BaseRecoveryStrategy]:
        """Find the most appropriate recovery strategy for the given error."""
        # Find strategies that can handle this error
        applicable_strategies = [
            strategy for strategy in self.strategies
            if strategy.can_handle(error)
        ]
        
        if not applicable_strategies:
            return None
        
        # For now, return the first applicable strategy
        # In the future, this could be enhanced with priority-based selection
        return applicable_strategies[0]
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get statistics about error recovery attempts."""
        total_recoveries = len(self.recovery_contexts)
        
        if total_recoveries == 0:
            return {
                'total_recoveries': 0,
                'success_rate': 0.0,
                'average_attempts': 0.0,
                'strategy_usage': {}
            }
        
        # Calculate statistics from recovery contexts
        successful_recoveries = 0
        total_attempts = 0
        strategy_usage = {}
        
        for context in self.recovery_contexts.values():
            attempts = context.get('attempt_count', 1)
            total_attempts += attempts
            
            # This is a simplified calculation - in a real implementation,
            # we'd track success/failure in the context
            if attempts <= 3:  # Assume success if attempts are reasonable
                successful_recoveries += 1
        
        return {
            'total_recoveries': total_recoveries,
            'success_rate': successful_recoveries / total_recoveries if total_recoveries > 0 else 0.0,
            'average_attempts': total_attempts / total_recoveries if total_recoveries > 0 else 0.0,
            'strategy_usage': strategy_usage
        }
    
    def clear_recovery_context(self, error_code: str):
        """Clear recovery context for a specific error."""
        if error_code in self.recovery_contexts:
            del self.recovery_contexts[error_code]
    
    def clear_all_recovery_contexts(self):
        """Clear all recovery contexts."""
        self.recovery_contexts.clear()