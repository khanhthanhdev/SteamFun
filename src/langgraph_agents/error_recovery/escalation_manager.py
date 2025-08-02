"""
Escalation threshold management for advanced error recovery.

This module manages escalation thresholds, tracks error patterns over time,
and determines when human intervention is required.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

from .advanced_recovery import ErrorPattern, RecoveryStrategy, ErrorAnalysis

logger = logging.getLogger(__name__)


class EscalationLevel(Enum):
    """Escalation levels for error handling."""
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class EscalationThreshold:
    """Configuration for escalation thresholds."""
    name: str
    level: EscalationLevel
    condition: str
    threshold_value: float
    time_window_minutes: int
    description: str
    enabled: bool = True


@dataclass
class EscalationEvent:
    """Record of an escalation event."""
    event_id: str
    timestamp: datetime
    level: EscalationLevel
    threshold_name: str
    trigger_value: float
    threshold_value: float
    context: Dict[str, Any]
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None


class EscalationThresholdManager:
    """Manages escalation thresholds and determines when to escalate errors."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the escalation threshold manager.
        
        Args:
            config: Configuration for escalation management
        """
        self.config = config
        self.thresholds = self._initialize_thresholds(config.get('thresholds', {}))
        self.escalation_history = []
        self.active_escalations = {}
        
        # Threshold evaluation settings
        self.evaluation_interval = config.get('evaluation_interval_seconds', 60)
        self.max_history_size = config.get('max_history_size', 1000)
        
        # Export settings
        self.export_enabled = config.get('export_enabled', True)
        self.export_path = Path(config.get('export_path', 'logs/escalation_events'))
        self.export_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Escalation threshold manager initialized with {len(self.thresholds)} thresholds")
    
    def _initialize_thresholds(self, threshold_config: Dict[str, Any]) -> Dict[str, EscalationThreshold]:
        """Initialize escalation thresholds from configuration."""
        default_thresholds = {
            'error_rate_warning': EscalationThreshold(
                name='error_rate_warning',
                level=EscalationLevel.WARNING,
                condition='error_rate_per_minute',
                threshold_value=5.0,
                time_window_minutes=10,
                description='Warning when error rate exceeds 5 errors per minute'
            ),
            'error_rate_critical': EscalationThreshold(
                name='error_rate_critical',
                level=EscalationLevel.CRITICAL,
                condition='error_rate_per_minute',
                threshold_value=10.0,
                time_window_minutes=5,
                description='Critical when error rate exceeds 10 errors per minute'
            ),
            'agent_failure_rate': EscalationThreshold(
                name='agent_failure_rate',
                level=EscalationLevel.CRITICAL,
                condition='agent_failure_rate',
                threshold_value=0.8,
                time_window_minutes=15,
                description='Critical when any agent has >80% failure rate'
            ),
            'recovery_failure_rate': EscalationThreshold(
                name='recovery_failure_rate',
                level=EscalationLevel.CRITICAL,
                condition='recovery_failure_rate',
                threshold_value=0.7,
                time_window_minutes=20,
                description='Critical when recovery success rate drops below 30%'
            ),
            'cascading_failures': EscalationThreshold(
                name='cascading_failures',
                level=EscalationLevel.EMERGENCY,
                condition='cascading_failure_count',
                threshold_value=3.0,
                time_window_minutes=10,
                description='Emergency when 3+ cascading failures occur'
            ),
            'circuit_breaker_count': EscalationThreshold(
                name='circuit_breaker_count',
                level=EscalationLevel.CRITICAL,
                condition='open_circuit_breakers',
                threshold_value=2.0,
                time_window_minutes=5,
                description='Critical when 2+ circuit breakers are open'
            ),
            'human_escalation_rate': EscalationThreshold(
                name='human_escalation_rate',
                level=EscalationLevel.WARNING,
                condition='human_escalation_rate',
                threshold_value=0.3,
                time_window_minutes=30,
                description='Warning when >30% of errors require human escalation'
            )
        }
        
        # Merge with user configuration
        thresholds = {}
        for name, default_threshold in default_thresholds.items():
            if name in threshold_config:
                # Update with user configuration
                user_config = threshold_config[name]
                threshold = EscalationThreshold(
                    name=name,
                    level=EscalationLevel(user_config.get('level', default_threshold.level.value)),
                    condition=user_config.get('condition', default_threshold.condition),
                    threshold_value=user_config.get('threshold_value', default_threshold.threshold_value),
                    time_window_minutes=user_config.get('time_window_minutes', default_threshold.time_window_minutes),
                    description=user_config.get('description', default_threshold.description),
                    enabled=user_config.get('enabled', default_threshold.enabled)
                )
            else:
                threshold = default_threshold
            
            thresholds[name] = threshold
        
        return thresholds
    
    async def evaluate_escalation(
        self, 
        error_history: List[Dict[str, Any]], 
        recovery_history: List[Dict[str, Any]],
        circuit_breaker_states: Dict[str, str]
    ) -> List[EscalationEvent]:
        """Evaluate all escalation thresholds and return triggered events.
        
        Args:
            error_history: Recent error history
            recovery_history: Recent recovery history
            circuit_breaker_states: Current circuit breaker states
            
        Returns:
            List of triggered escalation events
        """
        triggered_events = []
        now = datetime.now()
        
        for threshold in self.thresholds.values():
            if not threshold.enabled:
                continue
            
            try:
                # Calculate current value for the threshold condition
                current_value = await self._calculate_threshold_value(
                    threshold, error_history, recovery_history, circuit_breaker_states
                )
                
                # Check if threshold is exceeded
                if current_value > threshold.threshold_value:
                    # Check if this escalation is already active
                    if threshold.name not in self.active_escalations:
                        event = EscalationEvent(
                            event_id=f"{threshold.name}_{now.isoformat()}",
                            timestamp=now,
                            level=threshold.level,
                            threshold_name=threshold.name,
                            trigger_value=current_value,
                            threshold_value=threshold.threshold_value,
                            context={
                                'condition': threshold.condition,
                                'time_window_minutes': threshold.time_window_minutes,
                                'description': threshold.description
                            }
                        )
                        
                        triggered_events.append(event)
                        self.active_escalations[threshold.name] = event
                        self.escalation_history.append(event)
                        
                        logger.warning(f"Escalation triggered: {threshold.name}, "
                                     f"value: {current_value:.2f}, threshold: {threshold.threshold_value}")
                
                else:
                    # Check if we can resolve an active escalation
                    if threshold.name in self.active_escalations:
                        active_event = self.active_escalations[threshold.name]
                        active_event.resolved = True
                        active_event.resolution_timestamp = now
                        del self.active_escalations[threshold.name]
                        
                        logger.info(f"Escalation resolved: {threshold.name}, "
                                   f"value: {current_value:.2f}, threshold: {threshold.threshold_value}")
            
            except Exception as e:
                logger.error(f"Error evaluating threshold {threshold.name}: {str(e)}")
        
        # Export escalation events if enabled
        if self.export_enabled and triggered_events:
            await self._export_escalation_events(triggered_events)
        
        return triggered_events
    
    async def _calculate_threshold_value(
        self,
        threshold: EscalationThreshold,
        error_history: List[Dict[str, Any]],
        recovery_history: List[Dict[str, Any]],
        circuit_breaker_states: Dict[str, str]
    ) -> float:
        """Calculate the current value for a threshold condition."""
        
        time_window = timedelta(minutes=threshold.time_window_minutes)
        cutoff_time = datetime.now() - time_window
        
        if threshold.condition == 'error_rate_per_minute':
            # Calculate errors per minute in the time window
            recent_errors = [
                entry for entry in error_history
                if entry.get('timestamp', datetime.min) > cutoff_time
            ]
            
            if not recent_errors:
                return 0.0
            
            time_span_minutes = threshold.time_window_minutes
            return len(recent_errors) / time_span_minutes
        
        elif threshold.condition == 'agent_failure_rate':
            # Calculate failure rate for each agent
            recent_errors = [
                entry for entry in error_history
                if entry.get('timestamp', datetime.min) > cutoff_time
            ]
            
            if not recent_errors:
                return 0.0
            
            # Group by agent
            agent_stats = {}
            for entry in recent_errors:
                agent_name = entry.get('error', {}).get('agent_name', 'unknown')
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = {'total': 0, 'failures': 0}
                
                agent_stats[agent_name]['total'] += 1
                # Assume all errors in history are failures
                agent_stats[agent_name]['failures'] += 1
            
            # Return the highest failure rate
            max_failure_rate = 0.0
            for stats in agent_stats.values():
                if stats['total'] > 0:
                    failure_rate = stats['failures'] / stats['total']
                    max_failure_rate = max(max_failure_rate, failure_rate)
            
            return max_failure_rate
        
        elif threshold.condition == 'recovery_failure_rate':
            # Calculate recovery failure rate
            recent_recoveries = [
                entry for entry in recovery_history
                if entry.get('start_time', datetime.min) > cutoff_time
            ]
            
            if not recent_recoveries:
                return 0.0
            
            failed_recoveries = sum(1 for entry in recent_recoveries if not entry.get('success', False))
            return failed_recoveries / len(recent_recoveries)
        
        elif threshold.condition == 'cascading_failure_count':
            # Count cascading failures in time window
            recent_errors = [
                entry for entry in error_history
                if entry.get('timestamp', datetime.min) > cutoff_time
            ]
            
            cascading_count = 0
            for entry in recent_errors:
                # Check if error analysis indicates cascading pattern
                error_analysis = entry.get('error_analysis', {})
                if error_analysis.get('pattern_type') == 'cascading':
                    cascading_count += 1
            
            return float(cascading_count)
        
        elif threshold.condition == 'open_circuit_breakers':
            # Count open circuit breakers
            open_breakers = sum(1 for state in circuit_breaker_states.values() if state == 'open')
            return float(open_breakers)
        
        elif threshold.condition == 'human_escalation_rate':
            # Calculate rate of human escalations
            recent_recoveries = [
                entry for entry in recovery_history
                if entry.get('start_time', datetime.min) > cutoff_time
            ]
            
            if not recent_recoveries:
                return 0.0
            
            human_escalations = sum(1 for entry in recent_recoveries 
                                  if entry.get('strategy') == 'human_escalation')
            return human_escalations / len(recent_recoveries)
        
        else:
            logger.warning(f"Unknown threshold condition: {threshold.condition}")
            return 0.0
    
    async def _export_escalation_events(self, events: List[EscalationEvent]):
        """Export escalation events to file for monitoring."""
        for event in events:
            event_data = {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'level': event.level.value,
                'threshold_name': event.threshold_name,
                'trigger_value': event.trigger_value,
                'threshold_value': event.threshold_value,
                'context': event.context,
                'resolved': event.resolved,
                'resolution_timestamp': event.resolution_timestamp.isoformat() if event.resolution_timestamp else None
            }
            
            export_file = self.export_path / f"escalation_{event.event_id}.json"
            with open(export_file, 'w') as f:
                json.dump(event_data, f, indent=2)
            
            logger.debug(f"Escalation event exported: {export_file}")
    
    def get_active_escalations(self) -> Dict[str, EscalationEvent]:
        """Get currently active escalations."""
        return self.active_escalations.copy()
    
    def get_escalation_summary(self) -> Dict[str, Any]:
        """Get summary of escalation activity."""
        now = datetime.now()
        
        # Recent escalations (last 24 hours)
        recent_escalations = [
            event for event in self.escalation_history
            if (now - event.timestamp).total_seconds() < 86400
        ]
        
        # Group by level
        level_counts = {}
        for level in EscalationLevel:
            level_counts[level.value] = sum(1 for event in recent_escalations if event.level == level)
        
        # Resolution statistics
        resolved_count = sum(1 for event in recent_escalations if event.resolved)
        resolution_rate = resolved_count / len(recent_escalations) if recent_escalations else 0.0
        
        # Average resolution time
        resolved_events = [event for event in recent_escalations if event.resolved and event.resolution_timestamp]
        if resolved_events:
            resolution_times = [
                (event.resolution_timestamp - event.timestamp).total_seconds()
                for event in resolved_events
            ]
            avg_resolution_time = sum(resolution_times) / len(resolution_times)
        else:
            avg_resolution_time = 0.0
        
        return {
            'total_escalations_24h': len(recent_escalations),
            'active_escalations': len(self.active_escalations),
            'escalations_by_level': level_counts,
            'resolution_rate': resolution_rate,
            'average_resolution_time_seconds': avg_resolution_time,
            'enabled_thresholds': sum(1 for t in self.thresholds.values() if t.enabled),
            'total_thresholds': len(self.thresholds)
        }
    
    def update_threshold(self, threshold_name: str, updates: Dict[str, Any]) -> bool:
        """Update an existing threshold configuration.
        
        Args:
            threshold_name: Name of the threshold to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if threshold was updated successfully
        """
        if threshold_name not in self.thresholds:
            logger.error(f"Threshold not found: {threshold_name}")
            return False
        
        threshold = self.thresholds[threshold_name]
        
        try:
            if 'level' in updates:
                threshold.level = EscalationLevel(updates['level'])
            if 'threshold_value' in updates:
                threshold.threshold_value = float(updates['threshold_value'])
            if 'time_window_minutes' in updates:
                threshold.time_window_minutes = int(updates['time_window_minutes'])
            if 'enabled' in updates:
                threshold.enabled = bool(updates['enabled'])
            if 'description' in updates:
                threshold.description = str(updates['description'])
            
            logger.info(f"Threshold updated: {threshold_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating threshold {threshold_name}: {str(e)}")
            return False
    
    def add_threshold(self, threshold: EscalationThreshold) -> bool:
        """Add a new escalation threshold.
        
        Args:
            threshold: The threshold to add
            
        Returns:
            True if threshold was added successfully
        """
        if threshold.name in self.thresholds:
            logger.error(f"Threshold already exists: {threshold.name}")
            return False
        
        self.thresholds[threshold.name] = threshold
        logger.info(f"Threshold added: {threshold.name}")
        return True
    
    def remove_threshold(self, threshold_name: str) -> bool:
        """Remove an escalation threshold.
        
        Args:
            threshold_name: Name of the threshold to remove
            
        Returns:
            True if threshold was removed successfully
        """
        if threshold_name not in self.thresholds:
            logger.error(f"Threshold not found: {threshold_name}")
            return False
        
        # Resolve any active escalation for this threshold
        if threshold_name in self.active_escalations:
            active_event = self.active_escalations[threshold_name]
            active_event.resolved = True
            active_event.resolution_timestamp = datetime.now()
            del self.active_escalations[threshold_name]
        
        del self.thresholds[threshold_name]
        logger.info(f"Threshold removed: {threshold_name}")
        return True
    
    def clear_escalation_history(self, older_than_hours: int = 168):  # Default 7 days
        """Clear old escalation events from history.
        
        Args:
            older_than_hours: Clear events older than this many hours
        """
        cutoff_time = datetime.now() - timedelta(hours=older_than_hours)
        
        original_count = len(self.escalation_history)
        self.escalation_history = [
            event for event in self.escalation_history
            if event.timestamp > cutoff_time
        ]
        
        cleared_count = original_count - len(self.escalation_history)
        if cleared_count > 0:
            logger.info(f"Cleared {cleared_count} old escalation events from history")