"""
WebSocket client for real-time communication with FastAPI backend.
Handles log streaming, connection management, and automatic reconnection.
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

import aiohttp
from aiohttp import ClientSession, WSMsgType, ClientWebSocketResponse


logger = logging.getLogger(__name__)


class MessageType(Enum):
    """WebSocket message types."""
    LOG = "log"
    PROGRESS = "progress"
    ERROR = "error"
    STATUS = "status"
    HEARTBEAT = "heartbeat"


@dataclass
class LogMessage:
    """Log message from backend."""
    timestamp: datetime
    level: str
    message: str
    component: str
    session_id: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogMessage':
        """Create LogMessage from dictionary."""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            level=data["level"],
            message=data["message"],
            component=data["component"],
            session_id=data["session_id"],
            metadata=data.get("metadata", {})
        )


@dataclass
class ProgressMessage:
    """Progress update message."""
    session_id: str
    step: str
    progress: float
    total_steps: int
    current_step: int
    message: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProgressMessage':
        """Create ProgressMessage from dictionary."""
        return cls(
            session_id=data["session_id"],
            step=data["step"],
            progress=data["progress"],
            total_steps=data["total_steps"],
            current_step=data["current_step"],
            message=data["message"],
            metadata=data.get("metadata", {})
        )


@dataclass
class ErrorMessage:
    """Error message from backend."""
    session_id: str
    error_type: str
    error_message: str
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorMessage':
        """Create ErrorMessage from dictionary."""
        return cls(
            session_id=data["session_id"],
            error_type=data["error_type"],
            error_message=data["error_message"],
            traceback=data.get("traceback"),
            metadata=data.get("metadata", {})
        )


@dataclass
class StatusMessage:
    """Status update message."""
    session_id: str
    status: str
    message: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StatusMessage':
        """Create StatusMessage from dictionary."""
        return cls(
            session_id=data["session_id"],
            status=data["status"],
            message=data["message"],
            metadata=data.get("metadata", {})
        )


class WebSocketClientError(Exception):
    """Base exception for WebSocket client errors."""
    pass


class ConnectionError(WebSocketClientError):
    """Raised when WebSocket connection fails."""
    pass


class MessageError(WebSocketClientError):
    """Raised when message processing fails."""
    pass


class WebSocketClient:
    """
    WebSocket client for real-time communication with FastAPI backend.
    Handles log streaming, connection management, and automatic reconnection.
    """
    
    def __init__(
        self,
        base_url: str,
        session_id: str,
        max_reconnect_attempts: int = 5,
        reconnect_delay: float = 1.0,
        heartbeat_interval: float = 30.0
    ):
        """
        Initialize WebSocket client.
        
        Args:
            base_url: Base URL of the FastAPI backend
            session_id: Session identifier for log streaming
            max_reconnect_attempts: Maximum reconnection attempts
            reconnect_delay: Delay between reconnection attempts
            heartbeat_interval: Heartbeat interval in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.session_id = session_id
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.heartbeat_interval = heartbeat_interval
        
        self._session: Optional[ClientSession] = None
        self._websocket: Optional[ClientWebSocketResponse] = None
        self._connected = False
        self._reconnect_count = 0
        self._should_reconnect = True
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
        
        # Message handlers
        self._log_handler: Optional[Callable[[LogMessage], Awaitable[None]]] = None
        self._progress_handler: Optional[Callable[[ProgressMessage], Awaitable[None]]] = None
        self._error_handler: Optional[Callable[[ErrorMessage], Awaitable[None]]] = None
        self._status_handler: Optional[Callable[[StatusMessage], Awaitable[None]]] = None
        self._connection_handler: Optional[Callable[[bool], Awaitable[None]]] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
        
    def set_log_handler(self, handler: Callable[[LogMessage], Awaitable[None]]):
        """Set handler for log messages."""
        self._log_handler = handler
        
    def set_progress_handler(self, handler: Callable[[ProgressMessage], Awaitable[None]]):
        """Set handler for progress messages."""
        self._progress_handler = handler
        
    def set_error_handler(self, handler: Callable[[ErrorMessage], Awaitable[None]]):
        """Set handler for error messages."""
        self._error_handler = handler
        
    def set_status_handler(self, handler: Callable[[StatusMessage], Awaitable[None]]):
        """Set handler for status messages."""
        self._status_handler = handler
        
    def set_connection_handler(self, handler: Callable[[bool], Awaitable[None]]):
        """Set handler for connection status changes."""
        self._connection_handler = handler
        
    async def connect(self) -> bool:
        """
        Connect to WebSocket endpoint.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if self._session is None:
                self._session = ClientSession()
                
            ws_url = f"{self.base_url.replace('http', 'ws')}/ws/{self.session_id}"
            logger.info(f"Connecting to WebSocket: {ws_url}")
            
            self._websocket = await self._session.ws_connect(ws_url)
            self._connected = True
            self._reconnect_count = 0
            
            # Start background tasks
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            logger.info(f"WebSocket connected for session: {self.session_id}")
            
            # Notify connection handler
            if self._connection_handler:
                await self._connection_handler(True)
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            self._connected = False
            
            # Notify connection handler
            if self._connection_handler:
                await self._connection_handler(False)
                
            return False
            
    async def disconnect(self):
        """Disconnect from WebSocket."""
        self._should_reconnect = False
        self._connected = False
        
        # Cancel background tasks
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
                
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
                
        # Close WebSocket connection
        if self._websocket and not self._websocket.closed:
            await self._websocket.close()
            
        # Close session
        if self._session and not self._session.closed:
            await self._session.close()
            
        logger.info(f"WebSocket disconnected for session: {self.session_id}")
        
        # Notify connection handler
        if self._connection_handler:
            await self._connection_handler(False)
            
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages."""
        while self._connected and self._should_reconnect:
            try:
                if self._websocket and not self._websocket.closed:
                    heartbeat_msg = {
                        "type": MessageType.HEARTBEAT.value,
                        "session_id": self.session_id,
                        "timestamp": datetime.now().isoformat()
                    }
                    await self._websocket.send_str(json.dumps(heartbeat_msg))
                    
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                logger.warning(f"Heartbeat failed: {e}")
                break
                
    async def _receive_loop(self):
        """Receive and process messages from WebSocket."""
        while self._connected and self._should_reconnect:
            try:
                if not self._websocket or self._websocket.closed:
                    break
                    
                msg = await self._websocket.receive()
                
                if msg.type == WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._websocket.exception()}")
                    break
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
                    logger.info("WebSocket connection closed by server")
                    break
                    
            except Exception as e:
                logger.error(f"Error in receive loop: {e}")
                break
                
        # Connection lost, attempt reconnection
        if self._should_reconnect and self._reconnect_count < self.max_reconnect_attempts:
            await self._attempt_reconnection()
            
    async def _handle_message(self, message_data: str):
        """
        Handle incoming WebSocket message.
        
        Args:
            message_data: Raw message data
        """
        try:
            data = json.loads(message_data)
            message_type = data.get("type")
            
            if message_type == MessageType.LOG.value:
                if self._log_handler:
                    log_msg = LogMessage.from_dict(data)
                    await self._log_handler(log_msg)
                    
            elif message_type == MessageType.PROGRESS.value:
                if self._progress_handler:
                    progress_msg = ProgressMessage.from_dict(data)
                    await self._progress_handler(progress_msg)
                    
            elif message_type == MessageType.ERROR.value:
                if self._error_handler:
                    error_msg = ErrorMessage.from_dict(data)
                    await self._error_handler(error_msg)
                    
            elif message_type == MessageType.STATUS.value:
                if self._status_handler:
                    status_msg = StatusMessage.from_dict(data)
                    await self._status_handler(status_msg)
                    
            elif message_type == MessageType.HEARTBEAT.value:
                logger.debug("Received heartbeat response")
                
            else:
                logger.warning(f"Unknown message type: {message_type}")
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode message: {e}")
            raise MessageError(f"Invalid JSON message: {e}")
        except Exception as e:
            logger.error(f"Failed to handle message: {e}")
            raise MessageError(f"Message handling error: {e}")
            
    async def _attempt_reconnection(self):
        """Attempt to reconnect to WebSocket."""
        self._connected = False
        self._reconnect_count += 1
        
        logger.info(f"Attempting reconnection {self._reconnect_count}/{self.max_reconnect_attempts}")
        
        # Notify connection handler
        if self._connection_handler:
            await self._connection_handler(False)
            
        # Wait before reconnecting
        await asyncio.sleep(self.reconnect_delay * self._reconnect_count)
        
        # Attempt reconnection
        if await self.connect():
            logger.info("Reconnection successful")
        else:
            logger.error(f"Reconnection failed (attempt {self._reconnect_count})")
            
            if self._reconnect_count >= self.max_reconnect_attempts:
                logger.error("Max reconnection attempts reached, giving up")
                self._should_reconnect = False
                
    async def send_message(self, message: Dict[str, Any]):
        """
        Send message to WebSocket.
        
        Args:
            message: Message to send
            
        Raises:
            ConnectionError: If not connected
            MessageError: If message sending fails
        """
        if not self._connected or not self._websocket or self._websocket.closed:
            raise ConnectionError("WebSocket not connected")
            
        try:
            message_str = json.dumps(message)
            await self._websocket.send_str(message_str)
            logger.debug(f"Sent message: {message.get('type', 'unknown')}")
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise MessageError(f"Message sending failed: {e}")
            
    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        return self._connected and self._websocket and not self._websocket.closed
        
    @property
    def reconnect_count(self) -> int:
        """Get current reconnection count."""
        return self._reconnect_count