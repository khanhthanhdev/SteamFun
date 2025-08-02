"""
Embedding Provider Factory

This module implements the factory pattern for creating embedding providers
with automatic provider detection, configuration validation, and fallback handling.
"""

import os
import logging
from typing import Optional, List, Dict, Any, Type
from dataclasses import dataclass

from .embedding_providers import (
    EmbeddingProvider,
    EmbeddingConfig,
    ConfigurationManager,
    ProviderConfigurationError,
    EmbeddingGe