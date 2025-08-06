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

try:
    from .analytics import (
        ErrorAnalyticsSystem,
        AnalyticsReport,
        SystemHealthMetrics
    )
    _ANALYTICS_AVAILABLE = True
except ImportError:
    # Analytics module requires matplotlib, make it optional
    ErrorAnalyticsSystem = None
    AnalyticsReport = None
    SystemHealthMetrics = None
    _ANALYTICS_AVAILABLE = False

from .error_handler import (
    ErrorHandler,
    RetryStrategy,
    RAGEnhancementStrategy,
    FallbackModelStrategy,
    BaseRecoveryStrategy,
    RecoveryAction
)

from .circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerState,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    CircuitBreakerOpenError,
    CircuitBreakerManager,
    circuit_breaker,
    circuit_breaker_manager
)

__all__ = [
    # Advanced recovery system
    'AdvancedErrorRecoverySystem',
    'ErrorPattern',
    'RecoveryStrategy', 
    'ErrorAnalysis',
    'RecoveryExecution',
    
    # Escalation management
    'EscalationThresholdManager',
    'EscalationLevel',
    'EscalationThreshold',
    'EscalationEvent',
    
    # Error handler and recovery strategies
    'ErrorHandler',
    'RetryStrategy',
    'RAGEnhancementStrategy',
    'FallbackModelStrategy',
    'BaseRecoveryStrategy',
    'RecoveryAction',
    
    # Circuit breaker
    'CircuitBreaker',
    'CircuitBreakerState',
    'CircuitBreakerConfig',
    'CircuitBreakerMetrics',
    'CircuitBreakerOpenError',
    'CircuitBreakerManager',
    'circuit_breaker',
    'circuit_breaker_manager'
]

# Add analytics to __all__ only if available
if _ANALYTICS_AVAILABLE:
    __all__.extend([
        'ErrorAnalyticsSystem',
        'AnalyticsReport',
        'SystemHealthMetrics'
    ])