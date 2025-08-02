"""
Error recovery module for advanced error handling and recovery strategies.
"""

from .advanced_recovery import (
    AdvancedErrorRecoverySystem,
    ErrorPattern,
    RecoveryStrategy,
    ErrorAnalysis,
    RecoveryExecution
)

from .escalation_manager import (
    EscalationThresholdManager,
    EscalationLevel,
    EscalationThreshold,
    EscalationEvent
)

from .analytics import (
    ErrorAnalyticsSystem,
    AnalyticsReport,
    SystemHealthMetrics
)

__all__ = [
    'AdvancedErrorRecoverySystem',
    'ErrorPattern',
    'RecoveryStrategy', 
    'ErrorAnalysis',
    'RecoveryExecution',
    'EscalationThresholdManager',
    'EscalationLevel',
    'EscalationThreshold',
    'EscalationEvent',
    'ErrorAnalyticsSystem',
    'AnalyticsReport',
    'SystemHealthMetrics'
]