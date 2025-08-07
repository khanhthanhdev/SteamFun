"""
WebSocket Client for Real-time Log Streaming

This module provides WebSocket client functionality for real-time communication
with the FastAPI backend. It handles log streaming, progress updates, and
connection management.
"""

import asyncio
import websockets
import json
import threading
from typing import Dict, Any, Callable, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class WebSocketLogClient:
    """WebSocket client for real-time log streaming."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url.replace('http://', 'ws://').replace('https://', 'wss://').rstrip('/')
        self.connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
    def get_websocket_url(self, session_id: str) -> str:
        """Get WebSocket URL for a session."""
        return f"{self.base_url}/test/ws/logs/{session_id}"
    
    async def connect_to_session(
        self, 
        session_id: str, 
        message_handler: Callable[[Dict[str, Any]], None]
    ) -> bool:
        """
        Connect to a session's log stream.
        
        Args:
            session_id: Session ID to connect to
            message_handler: Function to handle incoming messages
            
        Returns:
            True if connection successful, False otherwise
        """
        try:
            if session_id in self.connections:
                await self.disconnect_from_session(session_id)
            
            websocket_url = self.get_websocket_url(session_id)
            logger.info(f"Connecting to WebSocket: {websocket_url}")
            
            websocket = await websockets.connect(websocket_url)
            self.connections[session_id] = websocket
            self.message_handlers[session_id] = message_handler
            
            # Start listening task
            task = asyncio.create_task(self._listen_to_session(session_id))
            self.running_tasks[session_id] = task
            
            logger.info(f"Connected to session {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to session {session_id}: {e}")
            return False
    
    async def disconnect_from_session(self, session_id: str):
        """
        Disconnect from a session's log stream.
        
        Args:
            session_id: Session ID to disconnect from
        """
        try:
            # Cancel listening task
            if session_id in self.running_tasks:
                task = self.running_tasks[session_id]
                if not task.done():
                    task.cancel()
                del self.running_tasks[session_id]
            
            # Close WebSocket connection
            if session_id in self.connections:
                websocket = self.connections[session_id]
                if not websocket.closed:
                    await websocket.close()
                del self.connections[session_id]
            
            # Remove message handler
            if session_id in self.message_handlers:
                del self.message_handlers[session_id]
            
            logger.info(f"Disconnected from session {session_id}")
            
        except Exception as e:
            logger.error(f"Error disconnecting from session {session_id}: {e}")
    
    async def _listen_to_session(self, session_id: str):
        """
        Listen to messages from a session's WebSocket.
        
        Args:
            session_id: Session ID to listen to
        """
        try:
            websocket = self.connections.get(session_id)
            message_handler = self.message_handlers.get(session_id)
            
            if not websocket or not message_handler:
                logger.error(f"No connection or handler for session {session_id}")
                return
            
            # Send initial ping to establish connection
            await websocket.send(json.dumps({"type": "ping"}))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    # Handle pong responses
                    if data.get('type') == 'pong':
                        continue
                    
                    # Call the message handler
                    message_handler(data)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse WebSocket message: {e}")
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed for session {session_id}")
        except asyncio.CancelledError:
            logger.info(f"WebSocket listening cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket listener for session {session_id}: {e}")
        finally:
            # Clean up connection
            if session_id in self.connections:
                del self.connections[session_id]
            if session_id in self.message_handlers:
                del self.message_handlers[session_id]
    
    async def send_message(self, session_id: str, message: Dict[str, Any]) -> bool:
        """
        Send a message to a session's WebSocket.
        
        Args:
            session_id: Session ID to send message to
            message: Message dictionary to send
            
        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            websocket = self.connections.get(session_id)
            if not websocket or websocket.closed:
                logger.error(f"No active connection for session {session_id}")
                return False
            
            await websocket.send(json.dumps(message))
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message to session {session_id}: {e}")
            return False
    
    async def ping_session(self, session_id: str) -> bool:
        """
        Send a ping to a session to check connection.
        
        Args:
            session_id: Session ID to ping
            
        Returns:
            True if ping sent successfully, False otherwise
        """
        return await self.send_message(session_id, {"type": "ping"})
    
    def is_connected(self, session_id: str) -> bool:
        """
        Check if connected to a session.
        
        Args:
            session_id: Session ID to check
            
        Returns:
            True if connected, False otherwise
        """
        websocket = self.connections.get(session_id)
        return websocket is not None and not websocket.closed
    
    async def disconnect_all(self):
        """Disconnect from all sessions."""
        session_ids = list(self.connections.keys())
        for session_id in session_ids:
            await self.disconnect_from_session(session_id)
    
    def get_connected_sessions(self) -> list:
        """Get list of currently connected session IDs."""
        return [
            session_id for session_id, websocket in self.connections.items()
            if not websocket.closed
        ]


class LogMessageFormatter:
    """Utility class for formatting log messages."""
    
    @staticmethod
    def format_log_entry(log_data: Dict[str, Any]) -> str:
        """
        Format a log entry for display.
        
        Args:
            log_data: Log data dictionary
            
        Returns:
            Formatted log string
        """
        try:
            timestamp = log_data.get('timestamp', datetime.now().isoformat())
            level = log_data.get('level', 'INFO').upper()
            message = log_data.get('message', '')
            component = log_data.get('component', 'system')
            session_id = log_data.get('session_id', '')
            
            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime('%H:%M:%S')
            except:
                time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
            
            # Color coding for different log levels
            level_colors = {
                'DEBUG': '🔍',
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'SUCCESS': '✅'
            }
            
            level_icon = level_colors.get(level, 'ℹ️')
            
            return f"[{time_str}] {level_icon} [{component}] {message}"
            
        except Exception as e:
            logger.error(f"Error formatting log entry: {e}")
            return f"[ERROR] Failed to format log entry: {log_data}"
    
    @staticmethod
    def format_status_update(status_data: Dict[str, Any]) -> str:
        """
        Format a status update for display.
        
        Args:
            status_data: Status update dictionary
            
        Returns:
            Formatted status string
        """
        try:
            session_id = status_data.get('session_id', '')
            status = status_data.get('status', 'unknown')
            progress = status_data.get('progress', 0)
            current_step = status_data.get('current_step', '')
            
            status_icons = {
                'running': '⚙️',
                'completed': '✅',
                'failed': '❌',
                'cancelled': '⏹️',
                'pending': '⏳'
            }
            
            status_icon = status_icons.get(status, '❓')
            
            status_line = f"{status_icon} Status: {status.title()}"
            
            if progress > 0:
                status_line += f" ({progress:.1f}%)"
            
            if current_step:
                status_line += f" - {current_step}"
            
            return status_line
            
        except Exception as e:
            logger.error(f"Error formatting status update: {e}")
            return f"[ERROR] Failed to format status update: {status_data}"


class ThreadedWebSocketClient:
    """Threaded wrapper for WebSocket client to work with Gradio."""
    
    def __init__(self, base_url: str):
        self.client = WebSocketLogClient(base_url)
        self.loop = None
        self.thread = None
        self.log_buffer = {}
        self.status_buffer = {}
        
    def start(self):
        """Start the WebSocket client in a separate thread."""
        if self.thread and self.thread.is_alive():
            return
        
        self.thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the WebSocket client."""
        if self.loop:
            asyncio.run_coroutine_threadsafe(self.client.disconnect_all(), self.loop)
        
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
    
    def _run_event_loop(self):
        """Run the asyncio event loop in the thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()
    
    def connect_to_session(self, session_id: str, message_callback: Callable[[str], None]):
        """
        Connect to a session with a message callback.
        
        Args:
            session_id: Session ID to connect to
            message_callback: Callback function for formatted messages
        """
        if not self.loop:
            self.start()
        
        def handle_message(data: Dict[str, Any]):
            if data.get('type') == 'log':
                formatted_msg = LogMessageFormatter.format_log_entry(data)
                message_callback(formatted_msg)
            elif data.get('type') == 'status_update':
                formatted_msg = LogMessageFormatter.format_status_update(data.get('data', {}))
                message_callback(formatted_msg)
        
        future = asyncio.run_coroutine_threadsafe(
            self.client.connect_to_session(session_id, handle_message),
            self.loop
        )
        
        return future.result(timeout=10)
    
    def disconnect_from_session(self, session_id: str):
        """Disconnect from a session."""
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.client.disconnect_from_session(session_id),
                self.loop
            )
    
    def is_connected(self, session_id: str) -> bool:
        """Check if connected to a session."""
        return self.client.is_connected(session_id)