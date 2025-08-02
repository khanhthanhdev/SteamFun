"""
Advanced error recovery strategies for production deployment.

This module implements sophisticated error pattern recognition, automatic recovery
workflow selection, escalation threshold management, and error analytics.
"""

import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import re
import statistics
from pathlib import Path

from ..state import VideoGenerationState, AgentError
from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ErrorPattern(Enum):
    """Types of error patterns that can be detected."""
    RECURRING = "recurring"
    CASCADING = "cascading"
    TEMPORAL = "temporal"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_FAILURE = "dependency_failure"
    CONFIGURATION_DRIFT = "configuration_drift"
    PERFORMANCE_DEGRADATION = "performance_degradation"


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    IMMEDIATE_RETRY = "immediate_retry"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK_PROVIDER = "fallback_provider"
    RESOURCE_SCALING = "resource_scaling"
    CONFIGURATION_RESET = "configuration_reset"
    DEPENDENCY_BYPASS = "dependency_bypass"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    HUMAN_ESCALATION = "human_escalation"


@dataclass
class ErrorAnalysis:
    """Comprehensive error analysis results."""
    error_id: str
    timestamp: datetime
    pattern_type: ErrorPattern
    confidence_score: float
    affected_components: List[str]
    root_cause_hypothesis: str
    recommended_strategy: RecoveryStrategy
    estimated_recovery_time: int  # seconds
    risk_assessment: str
    context: Dict[str, Any]


@dataclass
class RecoveryExecution:
    """Recovery execution tracking."""
    recovery_id: str
    strategy: RecoveryStrategy
    start_time: datetime
    end_time: Optional[datetime]
    success: Optional[bool]
    attempts: int
    error_analysis: ErrorAnalysis
    execution_log: List[str]
    metrics: Dict[str, Any]


class AdvancedErrorRecoverySystem:
    """Advanced error recovery system with pattern recognition and analytics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the advanced error recovery system.
        
        Args:
            config: Configuration for the recovery system
        """
        self.config = config
        self.error_history = deque(maxlen=config.get('max_error_history', 1000))
        self.recovery_history = deque(maxlen=config.get('max_recovery_history', 500))
        self.pattern_cache = {}
        self.circuit_breakers = {}
        self.escalation_thresholds = config.get('escalation_thresholds', {})
        
        # Pattern recognition settings
        self.pattern_window = timedelta(minutes=config.get('pattern_window_minutes', 30))
        self.min_pattern_occurrences = config.get('min_pattern_occurrences', 3)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        
        # Recovery strategy settings
        self.max_retry_attempts = config.get('max_retry_attempts', 3)
        self.backoff_multiplier = config.get('backoff_multiplier', 2.0)
        self.circuit_breaker_threshold = config.get('circuit_breaker_threshold', 5)
        self.circuit_breaker_timeout = config.get('circuit_breaker_timeout', 300)  # 5 minutes
        
        # Analytics settings
        self.analytics_enabled = config.get('analytics_enabled', True)
        self.analytics_export_path = Path(config.get('analytics_export_path', 'logs/error_analytics'))
        self.analytics_export_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("Advanced error recovery system initialized")
    
    async def analyze_error(self, error: AgentError, state: VideoGenerationState) -> ErrorAnalysis:
        """Perform comprehensive error analysis with pattern recognition.
        
        Args:
            error: The error to analyze
            state: Current workflow state
            
        Returns:
            Comprehensive error analysis
        """
        error_id = f"{error.agent_name}_{error.timestamp.isoformat()}_{hash(error.error_message)}"
        
        # Add to error history
        self.error_history.append({
            'id': error_id,
            'error': error,
            'state_snapshot': self._create_state_snapshot(state),
            'timestamp': error.timestamp
        })
        
        # Detect error patterns
        pattern_type, confidence = await self._detect_error_pattern(error, state)
        
        # Analyze root cause
        root_cause = await self._analyze_root_cause(error, state, pattern_type)
        
        # Determine recommended strategy
        strategy = await self._determine_recovery_strategy(error, pattern_type, confidence, state)
        
        # Estimate recovery time
        recovery_time = self._estimate_recovery_time(strategy, error, pattern_type)
        
        # Assess risk
        risk_assessment = self._assess_recovery_risk(strategy, error, state)
        
        # Get affected components
        affected_components = self._identify_affected_components(error, state)
        
        analysis = ErrorAnalysis(
            error_id=error_id,
            timestamp=datetime.now(),
            pattern_type=pattern_type,
            confidence_score=confidence,
            affected_components=affected_components,
            root_cause_hypothesis=root_cause,
            recommended_strategy=strategy,
            estimated_recovery_time=recovery_time,
            risk_assessment=risk_assessment,
            context={
                'error_count_last_hour': self._count_recent_errors(timedelta(hours=1)),
                'similar_errors_count': self._count_similar_errors(error),
                'system_load': self._assess_system_load(state),
                'workflow_progress': self._assess_workflow_progress(state)
            }
        )
        
        logger.info(f"Error analysis completed: {error_id}, pattern: {pattern_type.value}, "
                   f"confidence: {confidence:.2f}, strategy: {strategy.value}")
        
        return analysis
    
    async def execute_recovery(self, analysis: ErrorAnalysis, state: VideoGenerationState) -> RecoveryExecution:
        """Execute the recommended recovery strategy.
        
        Args:
            analysis: Error analysis results
            state: Current workflow state
            
        Returns:
            Recovery execution results
        """
        recovery_id = f"recovery_{analysis.error_id}_{datetime.now().isoformat()}"
        
        execution = RecoveryExecution(
            recovery_id=recovery_id,
            strategy=analysis.recommended_strategy,
            start_time=datetime.now(),
            end_time=None,
            success=None,
            attempts=0,
            error_analysis=analysis,
            execution_log=[],
            metrics={}
        )
        
        try:
            execution.execution_log.append(f"Starting recovery with strategy: {analysis.recommended_strategy.value}")
            
            # Execute the recovery strategy
            success = await self._execute_strategy(analysis.recommended_strategy, analysis, state, execution)
            
            execution.success = success
            execution.end_time = datetime.now()
            execution.metrics['duration_seconds'] = (execution.end_time - execution.start_time).total_seconds()
            
            # Record recovery attempt
            self.recovery_history.append(execution)
            
            # Update circuit breaker state if applicable
            self._update_circuit_breaker(analysis, success)
            
            # Export analytics if enabled
            if self.analytics_enabled:
                await self._export_recovery_analytics(execution)
            
            logger.info(f"Recovery execution completed: {recovery_id}, success: {success}")
            
            return execution
            
        except Exception as e:
            execution.success = False
            execution.end_time = datetime.now()
            execution.execution_log.append(f"Recovery failed with exception: {str(e)}")
            
            logger.error(f"Recovery execution failed: {recovery_id}, error: {str(e)}")
            
            # Record failed recovery
            self.recovery_history.append(execution)
            
            return execution  
  
    async def _detect_error_pattern(self, error: AgentError, state: VideoGenerationState) -> Tuple[ErrorPattern, float]:
        """Detect error patterns using advanced analysis.
        
        Args:
            error: Current error
            state: Current workflow state
            
        Returns:
            Tuple of (pattern_type, confidence_score)
        """
        # Get recent errors for pattern analysis
        recent_errors = [
            entry for entry in self.error_history
            if (datetime.now() - entry['timestamp']) <= self.pattern_window
        ]
        
        if len(recent_errors) < self.min_pattern_occurrences:
            return ErrorPattern.RECURRING, 0.3  # Low confidence for isolated errors
        
        # Check for recurring patterns
        recurring_confidence = self._analyze_recurring_pattern(error, recent_errors)
        if recurring_confidence > self.confidence_threshold:
            return ErrorPattern.RECURRING, recurring_confidence
        
        # Check for cascading failures
        cascading_confidence = self._analyze_cascading_pattern(error, recent_errors, state)
        if cascading_confidence > self.confidence_threshold:
            return ErrorPattern.CASCADING, cascading_confidence
        
        # Check for temporal patterns
        temporal_confidence = self._analyze_temporal_pattern(error, recent_errors)
        if temporal_confidence > self.confidence_threshold:
            return ErrorPattern.TEMPORAL, temporal_confidence
        
        # Check for resource exhaustion
        resource_confidence = self._analyze_resource_exhaustion(error, recent_errors, state)
        if resource_confidence > self.confidence_threshold:
            return ErrorPattern.RESOURCE_EXHAUSTION, resource_confidence
        
        # Check for dependency failures
        dependency_confidence = self._analyze_dependency_failure(error, recent_errors, state)
        if dependency_confidence > self.confidence_threshold:
            return ErrorPattern.DEPENDENCY_FAILURE, dependency_confidence
        
        # Check for configuration drift
        config_confidence = self._analyze_configuration_drift(error, recent_errors, state)
        if config_confidence > self.confidence_threshold:
            return ErrorPattern.CONFIGURATION_DRIFT, config_confidence
        
        # Check for performance degradation
        performance_confidence = self._analyze_performance_degradation(error, recent_errors, state)
        if performance_confidence > self.confidence_threshold:
            return ErrorPattern.PERFORMANCE_DEGRADATION, performance_confidence
        
        # Default to recurring with low confidence
        return ErrorPattern.RECURRING, max(recurring_confidence, 0.1)
    
    def _analyze_recurring_pattern(self, error: AgentError, recent_errors: List[Dict]) -> float:
        """Analyze for recurring error patterns."""
        similar_errors = [
            entry for entry in recent_errors
            if self._errors_similar(error, entry['error'])
        ]
        
        if len(similar_errors) < 2:
            return 0.0
        
        # Calculate confidence based on frequency and consistency
        frequency_score = min(len(similar_errors) / 10.0, 1.0)  # Max at 10 occurrences
        
        # Check for consistent intervals (temporal regularity)
        timestamps = [entry['timestamp'] for entry in similar_errors]
        timestamps.sort()
        
        if len(timestamps) > 2:
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            if statistics.mean(intervals) > 0:
                interval_consistency = 1.0 - (statistics.stdev(intervals) / statistics.mean(intervals))
                interval_consistency = max(0.0, min(1.0, interval_consistency))
            else:
                interval_consistency = 0.0
        else:
            interval_consistency = 0.5
        
        return (frequency_score * 0.7) + (interval_consistency * 0.3)
    
    def _analyze_cascading_pattern(self, error: AgentError, recent_errors: List[Dict], state: VideoGenerationState) -> float:
        """Analyze for cascading failure patterns."""
        # Look for errors in different agents within a short time window
        cascade_window = timedelta(minutes=5)
        cascade_errors = [
            entry for entry in recent_errors
            if abs((error.timestamp - entry['timestamp']).total_seconds()) <= cascade_window.total_seconds()
        ]
        
        if len(cascade_errors) < 2:
            return 0.0
        
        # Check for different agents involved
        agents_involved = set(entry['error'].agent_name for entry in cascade_errors)
        agent_diversity = len(agents_involved) / 5.0  # Normalize by typical agent count
        
        # Check for logical error progression (e.g., planner -> code_generator -> renderer)
        agent_sequence = ['planner_agent', 'code_generator_agent', 'renderer_agent', 'visual_analysis_agent']
        sequence_score = self._calculate_sequence_score(cascade_errors, agent_sequence)
        
        return min((agent_diversity * 0.6) + (sequence_score * 0.4), 1.0)    

    def _analyze_temporal_pattern(self, error: AgentError, recent_errors: List[Dict]) -> float:
        """Analyze for temporal error patterns."""
        if len(recent_errors) < 3:
            return 0.0
        
        # Group errors by hour of day
        hour_counts = defaultdict(int)
        for entry in recent_errors:
            hour = entry['timestamp'].hour
            hour_counts[hour] += 1
        
        if not hour_counts:
            return 0.0
        
        # Check for concentration in specific hours
        max_hour_count = max(hour_counts.values())
        total_errors = sum(hour_counts.values())
        concentration = max_hour_count / total_errors if total_errors > 0 else 0.0
        
        # Check for periodic patterns (e.g., every 24 hours)
        timestamps = [entry['timestamp'] for entry in recent_errors]
        timestamps.sort()
        
        if len(timestamps) > 3:
            # Look for 24-hour patterns
            daily_pattern_score = self._detect_daily_pattern(timestamps)
            return min((concentration * 0.5) + (daily_pattern_score * 0.5), 1.0)
        
        return concentration * 0.5
    
    def _analyze_resource_exhaustion(self, error: AgentError, recent_errors: List[Dict], state: VideoGenerationState) -> float:
        """Analyze for resource exhaustion patterns."""
        resource_keywords = [
            'memory', 'disk', 'cpu', 'timeout', 'limit', 'quota',
            'capacity', 'resource', 'exhausted', 'full'
        ]
        
        # Check error messages for resource-related keywords
        resource_errors = []
        for entry in recent_errors:
            error_msg = entry['error'].error_message.lower()
            if any(keyword in error_msg for keyword in resource_keywords):
                resource_errors.append(entry)
        
        if not resource_errors:
            return 0.0
        
        # Check for increasing frequency over time
        resource_errors.sort(key=lambda x: x['timestamp'])
        if len(resource_errors) > 2:
            recent_half = resource_errors[len(resource_errors)//2:]
            older_half = resource_errors[:len(resource_errors)//2]
            
            if recent_half and older_half:
                recent_duration = (recent_half[-1]['timestamp'] - recent_half[0]['timestamp']).total_seconds()
                older_duration = (older_half[-1]['timestamp'] - older_half[0]['timestamp']).total_seconds()
                
                recent_rate = len(recent_half) / max(recent_duration, 1)
                older_rate = len(older_half) / max(older_duration, 1)
                
                increasing_trend = min(recent_rate / max(older_rate, 0.001), 3.0) / 3.0
            else:
                increasing_trend = 0.5
        else:
            increasing_trend = 0.5
        
        resource_ratio = len(resource_errors) / len(recent_errors)
        
        return min((resource_ratio * 0.6) + (increasing_trend * 0.4), 1.0)
    
    def _analyze_dependency_failure(self, error: AgentError, recent_errors: List[Dict], state: VideoGenerationState) -> float:
        """Analyze for dependency failure patterns."""
        dependency_keywords = [
            'connection', 'network', 'api', 'service', 'unavailable',
            'unreachable', 'timeout', 'refused', 'dns', 'ssl'
        ]
        
        dependency_errors = []
        for entry in recent_errors:
            error_msg = entry['error'].error_message.lower()
            if any(keyword in error_msg for keyword in dependency_keywords):
                dependency_errors.append(entry)
        
        if not dependency_errors:
            return 0.0
        
        # Check for multiple agents affected by similar dependency issues
        affected_agents = set(entry['error'].agent_name for entry in dependency_errors)
        agent_spread = len(affected_agents) / 5.0  # Normalize by typical agent count
        
        dependency_ratio = len(dependency_errors) / len(recent_errors)
        
        return min((dependency_ratio * 0.7) + (agent_spread * 0.3), 1.0)
    
    def _analyze_configuration_drift(self, error: AgentError, recent_errors: List[Dict], state: VideoGenerationState) -> float:
        """Analyze for configuration drift patterns."""
        config_keywords = [
            'config', 'setting', 'parameter', 'environment', 'variable',
            'missing', 'invalid', 'format', 'parse', 'load'
        ]
        
        config_errors = []
        for entry in recent_errors:
            error_msg = entry['error'].error_message.lower()
            if any(keyword in error_msg for keyword in config_keywords):
                config_errors.append(entry)
        
        if not config_errors:
            return 0.0
        
        # Check for gradual increase in configuration-related errors
        config_errors.sort(key=lambda x: x['timestamp'])
        
        if len(config_errors) > 2:
            # Look for increasing trend
            time_diffs = [(config_errors[i+1]['timestamp'] - config_errors[i]['timestamp']).total_seconds() 
                         for i in range(len(config_errors)-1)]
            
            if time_diffs:
                avg_interval = statistics.mean(time_diffs)
                decreasing_intervals = sum(1 for i in range(1, len(time_diffs)) if time_diffs[i] < time_diffs[i-1])
                trend_score = decreasing_intervals / max(len(time_diffs) - 1, 1)
            else:
                trend_score = 0.0
        else:
            trend_score = 0.5
        
        config_ratio = len(config_errors) / len(recent_errors)
        
        return min((config_ratio * 0.6) + (trend_score * 0.4), 1.0)    

    def _analyze_performance_degradation(self, error: AgentError, recent_errors: List[Dict], state: VideoGenerationState) -> float:
        """Analyze for performance degradation patterns."""
        performance_keywords = [
            'slow', 'timeout', 'performance', 'latency', 'delay',
            'hang', 'freeze', 'unresponsive', 'bottleneck'
        ]
        
        performance_errors = []
        for entry in recent_errors:
            error_msg = entry['error'].error_message.lower()
            if any(keyword in error_msg for keyword in performance_keywords):
                performance_errors.append(entry)
        
        if not performance_errors:
            return 0.0
        
        # Check for increasing frequency of performance issues
        performance_errors.sort(key=lambda x: x['timestamp'])
        
        if len(performance_errors) > 3:
            # Calculate error frequency over time
            time_span = (performance_errors[-1]['timestamp'] - performance_errors[0]['timestamp']).total_seconds()
            if time_span > 0:
                error_rate = len(performance_errors) / time_span
                # Normalize rate (errors per minute)
                normalized_rate = min(error_rate * 60, 1.0)
            else:
                normalized_rate = 0.5
        else:
            normalized_rate = 0.3
        
        performance_ratio = len(performance_errors) / len(recent_errors)
        
        return min((performance_ratio * 0.5) + (normalized_rate * 0.5), 1.0)
    
    async def _determine_recovery_strategy(
        self, 
        error: AgentError, 
        pattern_type: ErrorPattern, 
        confidence: float, 
        state: VideoGenerationState
    ) -> RecoveryStrategy:
        """Determine the best recovery strategy based on error analysis."""
        
        # Check circuit breaker status
        if self._is_circuit_breaker_open(error.agent_name):
            return RecoveryStrategy.CIRCUIT_BREAKER
        
        # Strategy selection based on pattern type and confidence
        if pattern_type == ErrorPattern.RECURRING and confidence > 0.8:
            # High confidence recurring errors need circuit breaker or escalation
            if self._get_error_count_for_agent(error.agent_name) > self.circuit_breaker_threshold:
                return RecoveryStrategy.CIRCUIT_BREAKER
            else:
                return RecoveryStrategy.EXPONENTIAL_BACKOFF
        
        elif pattern_type == ErrorPattern.CASCADING:
            # Cascading failures need dependency bypass or graceful degradation
            return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        elif pattern_type == ErrorPattern.RESOURCE_EXHAUSTION:
            # Resource issues need scaling or circuit breaker
            return RecoveryStrategy.RESOURCE_SCALING
        
        elif pattern_type == ErrorPattern.DEPENDENCY_FAILURE:
            # Dependency issues need fallback or bypass
            return RecoveryStrategy.FALLBACK_PROVIDER
        
        elif pattern_type == ErrorPattern.CONFIGURATION_DRIFT:
            # Configuration issues need reset
            return RecoveryStrategy.CONFIGURATION_RESET
        
        elif pattern_type == ErrorPattern.PERFORMANCE_DEGRADATION:
            # Performance issues need scaling or circuit breaker
            return RecoveryStrategy.RESOURCE_SCALING
        
        elif pattern_type == ErrorPattern.TEMPORAL:
            # Temporal patterns might need scheduled recovery
            return RecoveryStrategy.EXPONENTIAL_BACKOFF
        
        # Check escalation thresholds
        if self._should_escalate_to_human(error, state):
            return RecoveryStrategy.HUMAN_ESCALATION
        
        # Default strategy based on error characteristics
        if error.retry_count == 0:
            return RecoveryStrategy.IMMEDIATE_RETRY
        elif error.retry_count < self.max_retry_attempts:
            return RecoveryStrategy.EXPONENTIAL_BACKOFF
        else:
            return RecoveryStrategy.HUMAN_ESCALATION  
  
    async def _execute_strategy(
        self, 
        strategy: RecoveryStrategy, 
        analysis: ErrorAnalysis, 
        state: VideoGenerationState,
        execution: RecoveryExecution
    ) -> bool:
        """Execute a specific recovery strategy."""
        
        execution.attempts += 1
        execution.execution_log.append(f"Executing strategy: {strategy.value} (attempt {execution.attempts})")
        
        try:
            if strategy == RecoveryStrategy.IMMEDIATE_RETRY:
                return await self._execute_immediate_retry(analysis, state, execution)
            
            elif strategy == RecoveryStrategy.EXPONENTIAL_BACKOFF:
                return await self._execute_exponential_backoff(analysis, state, execution)
            
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._execute_circuit_breaker(analysis, state, execution)
            
            elif strategy == RecoveryStrategy.FALLBACK_PROVIDER:
                return await self._execute_fallback_provider(analysis, state, execution)
            
            elif strategy == RecoveryStrategy.RESOURCE_SCALING:
                return await self._execute_resource_scaling(analysis, state, execution)
            
            elif strategy == RecoveryStrategy.CONFIGURATION_RESET:
                return await self._execute_configuration_reset(analysis, state, execution)
            
            elif strategy == RecoveryStrategy.DEPENDENCY_BYPASS:
                return await self._execute_dependency_bypass(analysis, state, execution)
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return await self._execute_graceful_degradation(analysis, state, execution)
            
            elif strategy == RecoveryStrategy.HUMAN_ESCALATION:
                return await self._execute_human_escalation(analysis, state, execution)
            
            else:
                execution.execution_log.append(f"Unknown strategy: {strategy.value}")
                return False
                
        except Exception as e:
            execution.execution_log.append(f"Strategy execution failed: {str(e)}")
            logger.error(f"Strategy execution failed: {strategy.value}, error: {str(e)}")
            return False
    
    async def _execute_immediate_retry(self, analysis: ErrorAnalysis, state: VideoGenerationState, execution: RecoveryExecution) -> bool:
        """Execute immediate retry strategy."""
        execution.execution_log.append("Executing immediate retry")
        
        # Simple immediate retry - just return True to indicate retry should happen
        # The actual retry logic is handled by the calling agent
        execution.metrics['retry_delay'] = 0
        return True
    
    async def _execute_exponential_backoff(self, analysis: ErrorAnalysis, state: VideoGenerationState, execution: RecoveryExecution) -> bool:
        """Execute exponential backoff strategy."""
        retry_count = execution.attempts - 1
        delay = min(self.backoff_multiplier ** retry_count, 300)  # Max 5 minutes
        
        execution.execution_log.append(f"Executing exponential backoff with {delay}s delay")
        execution.metrics['retry_delay'] = delay
        
        await asyncio.sleep(delay)
        return True
    
    async def _execute_circuit_breaker(self, analysis: ErrorAnalysis, state: VideoGenerationState, execution: RecoveryExecution) -> bool:
        """Execute circuit breaker strategy."""
        agent_name = analysis.affected_components[0] if analysis.affected_components else "unknown"
        
        execution.execution_log.append(f"Opening circuit breaker for {agent_name}")
        
        # Open circuit breaker
        self.circuit_breakers[agent_name] = {
            'state': 'open',
            'opened_at': datetime.now(),
            'failure_count': self.circuit_breakers.get(agent_name, {}).get('failure_count', 0) + 1
        }
        
        execution.metrics['circuit_breaker_opened'] = True
        return False  # Circuit breaker prevents further attempts    

    async def _execute_fallback_provider(self, analysis: ErrorAnalysis, state: VideoGenerationState, execution: RecoveryExecution) -> bool:
        """Execute fallback provider strategy."""
        execution.execution_log.append("Attempting fallback provider")
        
        # This would typically involve switching to a different LLM provider
        # For now, we'll simulate this by updating the state
        fallback_config = {
            'use_fallback_provider': True,
            'fallback_reason': 'primary_provider_failure',
            'fallback_timestamp': datetime.now().isoformat()
        }
        
        # Update state with fallback configuration
        state.update(fallback_config)
        execution.metrics['fallback_provider_used'] = True
        
        return True
    
    async def _execute_resource_scaling(self, analysis: ErrorAnalysis, state: VideoGenerationState, execution: RecoveryExecution) -> bool:
        """Execute resource scaling strategy."""
        execution.execution_log.append("Attempting resource scaling")
        
        # This would typically involve scaling compute resources
        # For now, we'll simulate by adjusting concurrency limits
        scaling_config = {
            'max_concurrent_scenes': max(1, state.get('max_concurrent_scenes', 3) - 1),
            'resource_scaling_applied': True,
            'scaling_timestamp': datetime.now().isoformat()
        }
        
        state.update(scaling_config)
        execution.metrics['resource_scaling_applied'] = True
        
        return True
    
    async def _execute_configuration_reset(self, analysis: ErrorAnalysis, state: VideoGenerationState, execution: RecoveryExecution) -> bool:
        """Execute configuration reset strategy."""
        execution.execution_log.append("Resetting configuration to defaults")
        
        # Reset configuration to known good state
        reset_config = {
            'configuration_reset': True,
            'reset_timestamp': datetime.now().isoformat(),
            'use_default_config': True
        }
        
        state.update(reset_config)
        execution.metrics['configuration_reset'] = True
        
        return True
    
    async def _execute_dependency_bypass(self, analysis: ErrorAnalysis, state: VideoGenerationState, execution: RecoveryExecution) -> bool:
        """Execute dependency bypass strategy."""
        execution.execution_log.append("Bypassing failed dependencies")
        
        # Bypass optional dependencies
        bypass_config = {
            'bypass_rag': True,
            'bypass_visual_analysis': True,
            'dependency_bypass_applied': True,
            'bypass_timestamp': datetime.now().isoformat()
        }
        
        state.update(bypass_config)
        execution.metrics['dependency_bypass_applied'] = True
        
        return True
    
    async def _execute_graceful_degradation(self, analysis: ErrorAnalysis, state: VideoGenerationState, execution: RecoveryExecution) -> bool:
        """Execute graceful degradation strategy."""
        execution.execution_log.append("Applying graceful degradation")
        
        # Reduce quality/features to ensure basic functionality
        degradation_config = {
            'quality_degradation': True,
            'skip_optional_features': True,
            'graceful_degradation_applied': True,
            'degradation_timestamp': datetime.now().isoformat()
        }
        
        state.update(degradation_config)
        execution.metrics['graceful_degradation_applied'] = True
        
        return True
    
    async def _execute_human_escalation(self, analysis: ErrorAnalysis, state: VideoGenerationState, execution: RecoveryExecution) -> bool:
        """Execute human escalation strategy."""
        execution.execution_log.append("Escalating to human intervention")
        
        # Prepare escalation context
        escalation_context = {
            'escalation_reason': 'automated_recovery_failed',
            'error_analysis': asdict(analysis),
            'recovery_attempts': execution.attempts,
            'escalation_timestamp': datetime.now().isoformat()
        }
        
        state.update({
            'pending_human_input': {
                'context': f"Advanced error recovery escalation: {analysis.root_cause_hypothesis}",
                'error_analysis': escalation_context,
                'options': ['retry_with_manual_intervention', 'skip_failed_operation', 'abort_workflow'],
                'requesting_agent': 'advanced_error_recovery'
            }
        })
        
        execution.metrics['escalated_to_human'] = True
        return False  # Human intervention required    

    # Helper methods for pattern analysis and system state assessment
    
    def _errors_similar(self, error1: AgentError, error2: AgentError) -> bool:
        """Check if two errors are similar."""
        # Simple similarity check based on agent and error message
        if error1.agent_name != error2.agent_name:
            return False
        
        # Normalize error messages for comparison
        msg1 = re.sub(r'\d+', 'N', error1.error_message.lower())
        msg2 = re.sub(r'\d+', 'N', error2.error_message.lower())
        
        # Calculate simple similarity score
        words1 = set(msg1.split())
        words2 = set(msg2.split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return similarity > 0.6
    
    def _calculate_sequence_score(self, errors: List[Dict], expected_sequence: List[str]) -> float:
        """Calculate how well errors follow an expected sequence."""
        if len(errors) < 2:
            return 0.0
        
        # Sort errors by timestamp
        sorted_errors = sorted(errors, key=lambda x: x['timestamp'])
        
        # Extract agent sequence
        agent_sequence = [entry['error'].agent_name for entry in sorted_errors]
        
        # Calculate sequence match score
        matches = 0
        for i in range(len(agent_sequence) - 1):
            current_agent = agent_sequence[i]
            next_agent = agent_sequence[i + 1]
            
            if current_agent in expected_sequence and next_agent in expected_sequence:
                current_idx = expected_sequence.index(current_agent)
                next_idx = expected_sequence.index(next_agent)
                
                if next_idx > current_idx:
                    matches += 1
        
        return matches / max(len(agent_sequence) - 1, 1)
    
    def _detect_daily_pattern(self, timestamps: List[datetime]) -> float:
        """Detect daily patterns in timestamps."""
        if len(timestamps) < 4:
            return 0.0
        
        # Calculate time differences
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
        
        # Look for ~24 hour patterns (within 2 hours tolerance)
        daily_seconds = 24 * 60 * 60
        tolerance = 2 * 60 * 60  # 2 hours
        
        daily_matches = sum(1 for diff in time_diffs if abs(diff - daily_seconds) <= tolerance)
        
        return daily_matches / len(time_diffs)
    
    def _create_state_snapshot(self, state: VideoGenerationState) -> Dict[str, Any]:
        """Create a snapshot of relevant state information."""
        return {
            'error_count': state.get('error_count', 0),
            'retry_count': state.get('retry_count', {}),
            'current_agent': state.get('current_agent'),
            'workflow_progress': {
                'scene_outline_complete': bool(state.get('scene_outline')),
                'scenes_generated': len(state.get('generated_code', {})),
                'scenes_rendered': len(state.get('rendered_videos', {}))
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def _count_recent_errors(self, time_window: timedelta) -> int:
        """Count errors within a time window."""
        cutoff_time = datetime.now() - time_window
        return sum(1 for entry in self.error_history if entry['timestamp'] > cutoff_time)
    
    def _count_similar_errors(self, error: AgentError) -> int:
        """Count similar errors in history."""
        return sum(1 for entry in self.error_history if self._errors_similar(error, entry['error']))
    
    def _assess_system_load(self, state: VideoGenerationState) -> str:
        """Assess current system load."""
        concurrent_scenes = len(state.get('generated_code', {}))
        max_concurrent = state.get('max_concurrent_scenes', 3)
        
        load_ratio = concurrent_scenes / max_concurrent if max_concurrent > 0 else 0
        
        if load_ratio > 0.8:
            return "high"
        elif load_ratio > 0.5:
            return "medium"
        else:
            return "low"
    
    def _assess_workflow_progress(self, state: VideoGenerationState) -> float:
        """Assess workflow completion progress."""
        total_scenes = len(state.get('scene_implementations', {}))
        if total_scenes == 0:
            return 0.0
        
        rendered_scenes = len(state.get('rendered_videos', {}))
        return rendered_scenes / total_scenes    

    def _is_circuit_breaker_open(self, agent_name: str) -> bool:
        """Check if circuit breaker is open for an agent."""
        breaker = self.circuit_breakers.get(agent_name)
        if not breaker or breaker['state'] != 'open':
            return False
        
        # Check if timeout has passed
        opened_at = breaker['opened_at']
        if (datetime.now() - opened_at).total_seconds() > self.circuit_breaker_timeout:
            # Move to half-open state
            self.circuit_breakers[agent_name]['state'] = 'half-open'
            return False
        
        return True
    
    def _get_error_count_for_agent(self, agent_name: str) -> int:
        """Get error count for a specific agent."""
        return sum(1 for entry in self.error_history if entry['error'].agent_name == agent_name)
    
    def _should_escalate_to_human(self, error: AgentError, state: VideoGenerationState) -> bool:
        """Check if error should be escalated to human."""
        # Check global escalation thresholds
        total_errors = state.get('error_count', 0)
        if total_errors >= self.escalation_thresholds.get('max_total_errors', 10):
            return True
        
        # Check agent-specific thresholds
        agent_errors = self._get_error_count_for_agent(error.agent_name)
        if agent_errors >= self.escalation_thresholds.get('max_agent_errors', 5):
            return True
        
        # Check time-based thresholds
        recent_errors = self._count_recent_errors(timedelta(minutes=30))
        if recent_errors >= self.escalation_thresholds.get('max_errors_per_30min', 8):
            return True
        
        return False
    
    def _update_circuit_breaker(self, analysis: ErrorAnalysis, success: bool):
        """Update circuit breaker state based on recovery result."""
        if not analysis.affected_components:
            return
        
        agent_name = analysis.affected_components[0]
        breaker = self.circuit_breakers.get(agent_name, {'state': 'closed', 'failure_count': 0})
        
        if success:
            if breaker['state'] == 'half-open':
                # Success in half-open state - close the breaker
                self.circuit_breakers[agent_name] = {'state': 'closed', 'failure_count': 0}
            elif breaker['state'] == 'closed':
                # Reset failure count on success
                breaker['failure_count'] = max(0, breaker['failure_count'] - 1)
        else:
            # Increment failure count
            breaker['failure_count'] += 1
            
            # Open breaker if threshold exceeded
            if breaker['failure_count'] >= self.circuit_breaker_threshold:
                breaker['state'] = 'open'
                breaker['opened_at'] = datetime.now()
        
        self.circuit_breakers[agent_name] = breaker
    
    async def _export_recovery_analytics(self, execution: RecoveryExecution):
        """Export recovery analytics for monitoring and analysis."""
        analytics_data = {
            'recovery_id': execution.recovery_id,
            'timestamp': execution.start_time.isoformat(),
            'strategy': execution.strategy.value,
            'success': execution.success,
            'duration_seconds': execution.metrics.get('duration_seconds', 0),
            'attempts': execution.attempts,
            'error_analysis': asdict(execution.error_analysis),
            'execution_log': execution.execution_log,
            'metrics': execution.metrics
        }
        
        # Export to JSON file
        export_file = self.analytics_export_path / f"recovery_{execution.recovery_id}.json"
        with open(export_file, 'w') as f:
            json.dump(analytics_data, f, indent=2, default=str)
        
        logger.debug(f"Recovery analytics exported to {export_file}")
    
    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        now = datetime.now()
        
        # Recovery success rates by strategy
        strategy_stats = defaultdict(lambda: {'total': 0, 'success': 0})
        for execution in self.recovery_history:
            strategy = execution.strategy.value
            strategy_stats[strategy]['total'] += 1
            if execution.success:
                strategy_stats[strategy]['success'] += 1
        
        # Calculate success rates
        strategy_success_rates = {
            strategy: stats['success'] / stats['total'] if stats['total'] > 0 else 0
            for strategy, stats in strategy_stats.items()
        }
        
        # Error pattern distribution
        pattern_counts = defaultdict(int)
        for execution in self.recovery_history:
            pattern_counts[execution.error_analysis.pattern_type.value] += 1
        
        # Recent performance metrics
        recent_recoveries = [
            execution for execution in self.recovery_history
            if (now - execution.start_time).total_seconds() < 3600  # Last hour
        ]
        
        avg_recovery_time = statistics.mean([
            execution.metrics.get('duration_seconds', 0)
            for execution in recent_recoveries
        ]) if recent_recoveries else 0
        
        return {
            'total_recoveries': len(self.recovery_history),
            'recent_recoveries': len(recent_recoveries),
            'strategy_success_rates': strategy_success_rates,
            'pattern_distribution': dict(pattern_counts),
            'average_recovery_time_seconds': avg_recovery_time,
            'circuit_breaker_states': {
                agent: breaker['state'] for agent, breaker in self.circuit_breakers.items()
            },
            'escalation_rate': sum(1 for execution in recent_recoveries 
                                 if execution.strategy == RecoveryStrategy.HUMAN_ESCALATION) / max(len(recent_recoveries), 1),
            'analytics_export_path': str(self.analytics_export_path)
        }    

    async def _analyze_root_cause(self, error: AgentError, state: VideoGenerationState, pattern_type: ErrorPattern) -> str:
        """Analyze root cause of error based on pattern and context."""
        
        if pattern_type == ErrorPattern.RECURRING:
            return f"Recurring error in {error.agent_name}: likely systematic issue with {error.error_type} handling"
        
        elif pattern_type == ErrorPattern.CASCADING:
            return f"Cascading failure starting from {error.agent_name}: upstream dependency or data quality issue"
        
        elif pattern_type == ErrorPattern.RESOURCE_EXHAUSTION:
            return f"Resource exhaustion in {error.agent_name}: insufficient compute, memory, or storage resources"
        
        elif pattern_type == ErrorPattern.DEPENDENCY_FAILURE:
            return f"External dependency failure affecting {error.agent_name}: network, API, or service unavailability"
        
        elif pattern_type == ErrorPattern.CONFIGURATION_DRIFT:
            return f"Configuration drift in {error.agent_name}: settings have changed or become inconsistent"
        
        elif pattern_type == ErrorPattern.PERFORMANCE_DEGRADATION:
            return f"Performance degradation in {error.agent_name}: system slowdown or resource contention"
        
        elif pattern_type == ErrorPattern.TEMPORAL:
            return f"Temporal pattern in {error.agent_name}: time-based or scheduled issue"
        
        else:
            return f"Unknown pattern in {error.agent_name}: requires manual investigation"
    
    def _estimate_recovery_time(self, strategy: RecoveryStrategy, error: AgentError, pattern_type: ErrorPattern) -> int:
        """Estimate recovery time in seconds based on strategy and error characteristics."""
        
        base_times = {
            RecoveryStrategy.IMMEDIATE_RETRY: 5,
            RecoveryStrategy.EXPONENTIAL_BACKOFF: 30,
            RecoveryStrategy.CIRCUIT_BREAKER: 300,  # 5 minutes
            RecoveryStrategy.FALLBACK_PROVIDER: 60,
            RecoveryStrategy.RESOURCE_SCALING: 120,
            RecoveryStrategy.CONFIGURATION_RESET: 90,
            RecoveryStrategy.DEPENDENCY_BYPASS: 45,
            RecoveryStrategy.GRACEFUL_DEGRADATION: 30,
            RecoveryStrategy.HUMAN_ESCALATION: 1800  # 30 minutes
        }
        
        base_time = base_times.get(strategy, 60)
        
        # Adjust based on pattern complexity
        pattern_multipliers = {
            ErrorPattern.RECURRING: 1.0,
            ErrorPattern.CASCADING: 1.5,
            ErrorPattern.RESOURCE_EXHAUSTION: 2.0,
            ErrorPattern.DEPENDENCY_FAILURE: 1.2,
            ErrorPattern.CONFIGURATION_DRIFT: 1.3,
            ErrorPattern.PERFORMANCE_DEGRADATION: 1.8,
            ErrorPattern.TEMPORAL: 1.1
        }
        
        multiplier = pattern_multipliers.get(pattern_type, 1.0)
        
        # Adjust based on retry count
        retry_multiplier = 1.0 + (error.retry_count * 0.2)
        
        return int(base_time * multiplier * retry_multiplier)
    
    def _assess_recovery_risk(self, strategy: RecoveryStrategy, error: AgentError, state: VideoGenerationState) -> str:
        """Assess risk level of recovery strategy."""
        
        high_risk_strategies = [
            RecoveryStrategy.CONFIGURATION_RESET,
            RecoveryStrategy.RESOURCE_SCALING,
            RecoveryStrategy.DEPENDENCY_BYPASS
        ]
        
        medium_risk_strategies = [
            RecoveryStrategy.FALLBACK_PROVIDER,
            RecoveryStrategy.GRACEFUL_DEGRADATION,
            RecoveryStrategy.CIRCUIT_BREAKER
        ]
        
        if strategy in high_risk_strategies:
            return "high - may affect system stability or functionality"
        elif strategy in medium_risk_strategies:
            return "medium - may reduce performance or features"
        else:
            return "low - minimal impact on system operation"
    
    def _identify_affected_components(self, error: AgentError, state: VideoGenerationState) -> List[str]:
        """Identify components affected by the error."""
        components = [error.agent_name]
        
        # Add downstream components based on workflow
        workflow_dependencies = {
            'planner_agent': ['code_generator_agent', 'renderer_agent'],
            'code_generator_agent': ['renderer_agent', 'visual_analysis_agent'],
            'renderer_agent': ['visual_analysis_agent'],
            'rag_agent': ['code_generator_agent', 'planner_agent']
        }
        
        downstream = workflow_dependencies.get(error.agent_name, [])
        components.extend(downstream)
        
        return list(set(components))  # Remove duplicates