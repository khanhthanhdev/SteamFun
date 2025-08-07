"""
API client package for Gradio frontend communication with FastAPI backend.
"""

from .http_client import HTTPClient
from .websocket_client import WebSocketClient

__all__ = ["HTTPClient", "WebSocketClient"]