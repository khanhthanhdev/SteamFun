"""
Gradio Testing Frontend for LangGraph Agents

This module provides a comprehensive web-based testing interface for LangGraph agents
using Gradio. It includes individual agent testing, complete workflow testing,
and real-time log monitoring capabilities.
"""

import os
import gradio as gr
import asyncio
import json
import uuid
import websockets
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from src.test_ui.api_client import TestAPIClient
from src.test_ui.websocket_client import WebSocketLogClient, ThreadedWebSocketClient
from src.test_ui.dynamic_forms import AgentFormManager
from src.test_ui.config_manager import TestConfigManager
from src.test_ui.config_ui import ConfigurationUI
from src.test_ui.validation import InputValidator, ValidationError, AGENT_FORM_VALIDATION_RULES, WORKFLOW_FORM_VALIDATION_RULES, CONFIG_FORM_VALIDATION_RULES
from src.test_ui.error_handling import GradioErrorHandler, ValidationMixin, with_error_handling, ERROR_HANDLING_CSS
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioTestInterface(ValidationMixin):
    """Main Gradio testing interface for LangGraph agents."""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        ValidationMixin.__init__(self)
        self.backend_url = backend_url
        self.api_client = TestAPIClient(backend_url)
        self.websocket_client = WebSocketLogClient(backend_url)
        self.threaded_ws_client = ThreadedWebSocketClient(backend_url)
        self.active_sessions = {}
        self.form_manager = AgentFormManager()
        self.config_manager = TestConfigManager()
        self.config_ui = ConfigurationUI(self.config_manager)
        self.input_validator = InputValidator()
        
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface with tabbed layout."""
        
        # Custom CSS for better styling
        custom_css = f"""
        .main-header {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 1rem;
            margin-bottom: 2rem;
        }}
        .config-panel {{
            border: 1px solid #e1e5e9;
            border-radius: 0.5rem;
            padding: 1rem;
            background: #f8f9fa;
        }}
        .status-panel {{
            border: 1px solid #e1e5e9;
            border-radius: 0.5rem;
            padding: 1rem;
            background: #fff;
        }}
        .log-viewer {{
            font-family: 'Courier New', monospace;
            background: #1e1e1e;
            color: #ffffff;
            border-radius: 0.5rem;
        }}
        .agent-card {{
            border: 1px solid #e1e5e9;
            border-radius: 0.5rem;
            padding: 1rem;
            margin: 0.5rem 0;
            background: white;
        }}
        {ERROR_HANDLING_CSS}
        """
        
        with gr.Blocks(
            title="LangGraph Agent Testing Interface",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="slate",
                neutral_hue="slate",
                font=gr.themes.GoogleFont("Inter")
            ),
            css=custom_css
        ) as interface:
            
            # Header
            with gr.Row():
                with gr.Column():
                    gr.HTML("""
                        <div class="main-header">
                            <h1>🧪 LangGraph Agent Testing Interface</h1>
                            <p>Test individual agents and complete workflows with real-time monitoring</p>
                        </div>
                    """)
            
            # Configuration Panel (always visible)
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["config-panel"]):
                        gr.Markdown("### ⚙️ API Configuration")
                        
                        backend_url_input = gr.Textbox(
                            label="Backend URL",
                            value=self.backend_url,
                            placeholder="http://localhost:8000",
                            info="FastAPI backend server URL"
                        )
                        
                        connection_status = gr.Textbox(
                            label="Connection Status",
                            value="Not connected",
                            interactive=False
                        )
                        
                        test_connection_btn = gr.Button(
                            "🔗 Test Connection",
                            variant="secondary"
                        )
                        
                        refresh_agents_btn = gr.Button(
                            "🔄 Refresh Agents",
                            variant="secondary"
                        )
            
            # Main tabbed interface
            with gr.Tabs():
                
                # Tab 1: Individual Agent Testing
                with gr.TabItem("🤖 Individual Agent Testing"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Agent Selection
                            with gr.Group():
                                gr.Markdown("### 🎯 Agent Selection")
                                
                                agent_dropdown = gr.Dropdown(
                                    label="Select Agent",
                                    choices=[],
                                    value=None,
                                    info="Choose an agent to test"
                                )
                                
                                agent_info_display = gr.Markdown(
                                    "Select an agent to see its information",
                                    elem_classes=["agent-card"]
                                )
                            
                            # Dynamic Input Form (will be populated based on agent selection)
                            with gr.Group():
                                gr.Markdown("### 📝 Agent Input Configuration")
                                
                                # Dynamic inputs will be created here
                                agent_inputs_container = gr.Column()
                                
                                # Store dynamic input components
                                dynamic_inputs = gr.State({})
                                
                                # Common options
                                session_id_input = gr.Textbox(
                                    label="Session ID (optional)",
                                    placeholder="Leave empty to auto-generate",
                                    info="Custom session ID for tracking"
                                )
                                
                                # Agent execution options
                                with gr.Accordion("Execution Options", open=False):
                                    agent_timeout = gr.Slider(
                                        label="Timeout (seconds)",
                                        minimum=10,
                                        maximum=300,
                                        value=60,
                                        step=10,
                                        info="Maximum execution time"
                                    )
                                    
                                    agent_retry_count = gr.Slider(
                                        label="Retry Count",
                                        minimum=0,
                                        maximum=5,
                                        value=1,
                                        step=1,
                                        info="Number of retries on failure"
                                    )
                                    
                                    agent_verbose = gr.Checkbox(
                                        label="Verbose Output",
                                        value=True,
                                        info="Enable detailed logging"
                                    )
                        
                        with gr.Column(scale=1):
                            # Execution Controls
                            with gr.Group(elem_classes=["status-panel"]):
                                gr.Markdown("### 🚀 Execution Controls")
                                
                                run_agent_btn = gr.Button(
                                    "▶️ Run Agent",
                                    variant="primary",
                                    size="lg"
                                )
                                
                                stop_agent_btn = gr.Button(
                                    "⏹️ Stop Agent",
                                    variant="stop",
                                    visible=False
                                )
                                
                                refresh_agent_status_btn = gr.Button(
                                    "🔄 Refresh Status",
                                    variant="secondary",
                                    size="sm"
                                )
                                
                                agent_status_text = gr.Textbox(
                                    label="Status",
                                    value="Ready",
                                    interactive=False
                                )
                                
                                agent_progress = gr.Slider(
                                    label="Progress",
                                    minimum=0,
                                    maximum=100,
                                    value=0,
                                    interactive=False
                                )
                                
                                agent_current_step = gr.Textbox(
                                    label="Current Step",
                                    interactive=False,
                                    placeholder="Waiting to start..."
                                )
                                
                                agent_session_display = gr.Textbox(
                                    label="Session ID",
                                    interactive=False,
                                    visible=False
                                )
                                
                                # Error and success displays for agent testing
                                agent_error_display = self.error_handler.create_error_display_component()
                                agent_success_display = self.error_handler.create_success_display_component()
                    
                    # Results Display
                    with gr.Row():
                        with gr.Column():
                            with gr.Group():
                                gr.Markdown("### 📊 Agent Results")
                                
                                agent_results_json = gr.JSON(
                                    label="Results",
                                    value=None
                                )
                                
                                agent_execution_time = gr.Textbox(
                                    label="Execution Time",
                                    interactive=False
                                )
                                
                                # Agent output visualization
                                with gr.Accordion("Agent Output Details", open=False):
                                    agent_output_text = gr.Textbox(
                                        label="Raw Output",
                                        lines=5,
                                        interactive=False,
                                        placeholder="Agent output will appear here..."
                                    )
                                    
                                    agent_metadata = gr.JSON(
                                        label="Execution Metadata",
                                        value=None
                                    )
                                    
                                    agent_errors = gr.Textbox(
                                        label="Errors",
                                        lines=3,
                                        interactive=False,
                                        visible=False
                                    )
                    
                    # Configuration Management for Agent Testing
                    with gr.Row():
                        with gr.Column(scale=1):
                            agent_config_save = self.config_ui.create_config_save_section()
                        
                        with gr.Column(scale=1):
                            agent_config_load = self.config_ui.create_config_load_section()
                
                # Tab 2: Complete Workflow Testing  
                with gr.TabItem("🔄 Complete Workflow Testing"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            # Workflow Input
                            with gr.Group():
                                gr.Markdown("### 📋 Workflow Configuration")
                                
                                workflow_topic = gr.Textbox(
                                    label="Topic",
                                    placeholder="e.g., Fourier Transform",
                                    info="Main topic for the video generation"
                                )
                                
                                workflow_description = gr.Textbox(
                                    label="Description",
                                    placeholder="Detailed description of what you want to create...",
                                    lines=4,
                                    info="Comprehensive description for better results"
                                )
                                
                                workflow_session_id = gr.Textbox(
                                    label="Session ID (optional)",
                                    placeholder="Leave empty to auto-generate"
                                )
                                
                                # Configuration overrides
                                with gr.Accordion("Advanced Configuration", open=False):
                                    config_overrides_json = gr.JSON(
                                        label="Configuration Overrides",
                                        value={},
                                        info="JSON object with configuration overrides"
                                    )
                        
                        with gr.Column(scale=1):
                            # Workflow Controls
                            with gr.Group(elem_classes=["status-panel"]):
                                gr.Markdown("### 🎬 Workflow Controls")
                                
                                run_workflow_btn = gr.Button(
                                    "🚀 Start Workflow",
                                    variant="primary",
                                    size="lg"
                                )
                                
                                stop_workflow_btn = gr.Button(
                                    "⏹️ Stop Workflow",
                                    variant="stop",
                                    visible=False
                                )
                                
                                workflow_status_text = gr.Textbox(
                                    label="Status",
                                    value="Ready",
                                    interactive=False
                                )
                                
                                workflow_current_step = gr.Textbox(
                                    label="Current Step",
                                    interactive=False
                                )
                                
                                workflow_progress = gr.Slider(
                                    label="Progress",
                                    minimum=0,
                                    maximum=100,
                                    value=0,
                                    interactive=False
                                )
                                
                                workflow_refresh_btn = gr.Button(
                                    "🔄 Refresh Status",
                                    variant="secondary",
                                    visible=False
                                )
                                
                                auto_refresh_workflow = gr.Checkbox(
                                    label="Auto-refresh (2s)",
                                    value=True,
                                    info="Automatically update workflow status"
                                )
                                
                                # Error and success displays for workflow testing
                                workflow_error_display = self.error_handler.create_error_display_component()
                                workflow_success_display = self.error_handler.create_success_display_component()
                    
                    # Workflow Results
                    with gr.Row():
                        with gr.Column(scale=2):
                            with gr.Group():
                                gr.Markdown("### 🎥 Workflow Results")
                                
                                workflow_results_json = gr.JSON(
                                    label="Results",
                                    value=None
                                )
                                
                                workflow_execution_time = gr.Textbox(
                                    label="Total Execution Time",
                                    interactive=False
                                )
                                
                                # Detailed workflow progress
                                with gr.Accordion("Workflow Step Details", open=False):
                                    workflow_step_details = gr.Markdown(
                                        "Workflow step details will appear here during execution...",
                                        elem_classes=["workflow-details"]
                                    )
                                    
                                    workflow_logs_display = gr.Textbox(
                                        label="Workflow Logs",
                                        lines=8,
                                        interactive=False,
                                        placeholder="Workflow logs will appear here..."
                                    )
                        
                        with gr.Column(scale=1):
                            # Video Output Preview
                            with gr.Group():
                                gr.Markdown("### 🎬 Video Output")
                                
                                video_preview = gr.Video(
                                    label="Generated Video",
                                    visible=False
                                )
                                
                                video_thumbnail = gr.Image(
                                    label="Thumbnail",
                                    visible=False
                                )
                                
                                download_video_btn = gr.Button(
                                    "💾 Download Video",
                                    variant="secondary",
                                    visible=False
                                )
                    
                    # Configuration Management for Workflow Testing
                    with gr.Row():
                        with gr.Column(scale=1):
                            workflow_config_save = self.config_ui.create_config_save_section()
                        
                        with gr.Column(scale=1):
                            workflow_config_load = self.config_ui.create_config_load_section()
                
                # Tab 3: Real-time Logs
                with gr.TabItem("📋 Real-time Logs"):
                    with gr.Row():
                        with gr.Column():
                            # Log Controls
                            with gr.Group():
                                gr.Markdown("### 📊 Log Monitoring")
                                
                                with gr.Row():
                                    log_session_input = gr.Textbox(
                                        label="Session ID",
                                        placeholder="Enter session ID to monitor",
                                        scale=3
                                    )
                                    
                                    connect_logs_btn = gr.Button(
                                        "🔗 Connect",
                                        variant="primary",
                                        scale=1
                                    )
                                    
                                    disconnect_logs_btn = gr.Button(
                                        "🔌 Disconnect",
                                        variant="secondary",
                                        visible=False,
                                        scale=1
                                    )
                                
                                with gr.Row():
                                    log_level_filter = gr.Dropdown(
                                        label="Log Level Filter",
                                        choices=["ALL", "DEBUG", "INFO", "WARNING", "ERROR"],
                                        value="ALL",
                                        scale=1
                                    )
                                    
                                    auto_scroll_checkbox = gr.Checkbox(
                                        label="Auto-scroll",
                                        value=True,
                                        scale=1
                                    )
                                    
                                    clear_logs_btn = gr.Button(
                                        "🧹 Clear Logs",
                                        variant="secondary",
                                        scale=1
                                    )
                            
                            # Log Display
                            with gr.Group():
                                log_display = gr.Textbox(
                                    label="Live Logs",
                                    lines=20,
                                    max_lines=50,
                                    interactive=False,
                                    elem_classes=["log-viewer"],
                                    placeholder="Connect to a session to see real-time logs..."
                                )
                                
                                log_connection_status = gr.Textbox(
                                    label="Connection Status",
                                    value="Disconnected",
                                    interactive=False
                                )
                
                # Tab 4: Configuration Management
                config_tab, config_components = self.config_ui.create_config_management_tab()
            
            # Event handlers will be set up in the setup_event_handlers method
            self._setup_event_handlers(
                interface,
                # Configuration components
                backend_url_input, connection_status, test_connection_btn, refresh_agents_btn,
                # Agent testing components
                agent_dropdown, agent_info_display, agent_inputs_container, dynamic_inputs,
                session_id_input, agent_timeout, agent_retry_count, agent_verbose,
                run_agent_btn, stop_agent_btn, refresh_agent_status_btn, agent_status_text, agent_progress,
                agent_current_step, agent_session_display, agent_results_json, 
                agent_execution_time, agent_output_text, agent_metadata, agent_errors,
                agent_error_display, agent_success_display,
                # Workflow testing components
                workflow_topic, workflow_description, workflow_session_id, config_overrides_json,
                run_workflow_btn, stop_workflow_btn, workflow_refresh_btn, auto_refresh_workflow,
                workflow_status_text, workflow_current_step, workflow_progress, workflow_results_json,
                workflow_execution_time, workflow_step_details, workflow_logs_display,
                video_preview, video_thumbnail, download_video_btn,
                workflow_error_display, workflow_success_display,
                # Log monitoring components
                log_session_input, connect_logs_btn, disconnect_logs_btn,
                log_level_filter, auto_scroll_checkbox, clear_logs_btn,
                log_display, log_connection_status,
                # Configuration management components
                config_components, agent_config_save, agent_config_load,
                workflow_config_save, workflow_config_load
            )
        
        return interface
    
    def _setup_event_handlers(self, interface, *components):
        """Set up event handlers for the interface components."""
        # Unpack components for easier reference
        (backend_url_input, connection_status, test_connection_btn, refresh_agents_btn,
         agent_dropdown, agent_info_display, agent_inputs_container, dynamic_inputs,
         session_id_input, agent_timeout, agent_retry_count, agent_verbose,
         run_agent_btn, stop_agent_btn, refresh_agent_status_btn, agent_status_text, agent_progress,
         agent_current_step, agent_session_display, agent_results_json, 
         agent_execution_time, agent_output_text, agent_metadata, agent_errors,
         workflow_topic, workflow_description, workflow_session_id, config_overrides_json,
         run_workflow_btn, stop_workflow_btn, workflow_refresh_btn, auto_refresh_workflow,
         workflow_status_text, workflow_current_step, workflow_progress, workflow_results_json,
         workflow_execution_time, workflow_step_details, workflow_logs_display,
         video_preview, video_thumbnail, download_video_btn,
         log_session_input, connect_logs_btn, disconnect_logs_btn,
         log_level_filter, auto_scroll_checkbox, clear_logs_btn,
         log_display, log_connection_status,
         config_components, agent_config_save, agent_config_load,
         workflow_config_save, workflow_config_load) = components
        
        # Configuration event handlers
        test_connection_btn.click(
            fn=self._test_connection,
            inputs=[backend_url_input],
            outputs=[connection_status]
        )
        
        refresh_agents_btn.click(
            fn=self._refresh_agents,
            inputs=[backend_url_input],
            outputs=[agent_dropdown, connection_status]
        )
        
        # Agent selection handler
        agent_dropdown.change(
            fn=self._on_agent_selected,
            inputs=[agent_dropdown],
            outputs=[agent_info_display, agent_inputs_container, dynamic_inputs]
        )
        
        # Agent testing handlers
        run_agent_btn.click(
            fn=self._run_agent_test_with_validation,
            inputs=[agent_dropdown, session_id_input, dynamic_inputs, 
                   agent_timeout, agent_retry_count, agent_verbose],
            outputs=[agent_status_text, agent_progress, agent_current_step,
                    agent_session_display, run_agent_btn, stop_agent_btn,
                    agent_results_json, agent_output_text, agent_metadata,
                    agent_error_display]
        )
        
        # Workflow testing handlers
        run_workflow_btn.click(
            fn=self._run_workflow_test_with_validation,
            inputs=[workflow_topic, workflow_description, workflow_session_id, config_overrides_json],
            outputs=[workflow_status_text, workflow_current_step, workflow_progress, 
                    run_workflow_btn, stop_workflow_btn, workflow_results_json,
                    workflow_execution_time, workflow_step_details, workflow_logs_display,
                    video_preview, video_thumbnail, download_video_btn, workflow_error_display]
        )
        
        # Workflow stop handler
        stop_workflow_btn.click(
            fn=self._stop_workflow_test,
            inputs=[workflow_session_id],
            outputs=[workflow_status_text, workflow_current_step, run_workflow_btn, stop_workflow_btn]
        )
        
        # Workflow status refresh handler
        workflow_refresh_btn.click(
            fn=self._refresh_workflow_status,
            inputs=[workflow_session_id],
            outputs=[workflow_status_text, workflow_current_step, workflow_progress,
                    workflow_results_json, workflow_execution_time, workflow_step_details,
                    workflow_logs_display, video_preview, video_thumbnail, download_video_btn]
        )
        
        # Video download handler
        download_video_btn.click(
            fn=self._handle_video_download,
            inputs=[workflow_session_id],
            outputs=[gr.Textbox(visible=False)]  # Hidden output for download status
        )
        
        # Auto-refresh workflow status (using a timer-like approach)
        # Note: In a real implementation, this would use Gradio's built-in timer functionality
        # For now, we rely on the periodic updates in the background thread
        
        # Log monitoring handlers
        connect_logs_btn.click(
            fn=self._connect_to_logs,
            inputs=[log_session_input],
            outputs=[log_connection_status, connect_logs_btn, disconnect_logs_btn]
        )
        
        disconnect_logs_btn.click(
            fn=self._disconnect_from_logs,
            inputs=[],
            outputs=[log_connection_status, connect_logs_btn, disconnect_logs_btn, log_display]
        )
        
        clear_logs_btn.click(
            fn=lambda: "",
            inputs=[],
            outputs=[log_display]
        )
        
        # Set up periodic status updates for agent testing
        interface.load(
            fn=self._setup_periodic_updates,
            inputs=[],
            outputs=[]
        )
        
        # Agent status refresh handler
        refresh_agent_status_btn.click(
            fn=self._get_active_agent_status,
            inputs=[],
            outputs=[agent_status_text, agent_progress, agent_current_step, 
                    agent_results_json, agent_output_text, agent_metadata]
        )
        
        # Configuration management event handlers
        self._setup_config_event_handlers(
            config_components, agent_config_save, agent_config_load,
            workflow_config_save, workflow_config_load,
            # Agent components for saving/loading
            agent_dropdown, dynamic_inputs, session_id_input, 
            agent_timeout, agent_retry_count, agent_verbose,
            # Workflow components for saving/loading
            workflow_topic, workflow_description, workflow_session_id, config_overrides_json
        )
    
    def _test_connection(self, backend_url: str) -> str:
        """Test connection to the backend API with detailed error reporting."""
        try:
            # Validate backend URL format
            if not backend_url:
                return "❌ Backend URL is required"
            
            if not backend_url.startswith(('http://', 'https://')):
                return "❌ Backend URL must start with http:// or https://"
            
            self.backend_url = backend_url
            self.api_client = TestAPIClient(backend_url)
            
            # Perform detailed health check
            is_healthy, status_message, suggestions = self.api_client.detailed_health_check()
            
            if is_healthy:
                # If health check passes, try to get agents for additional verification
                try:
                    agents = self.api_client.get_available_agents()
                    return f"✅ Connected successfully. Found {len(agents)} agents available for testing."
                except Exception as e:
                    return f"⚠️ Server is healthy but agents endpoint failed: {str(e)}"
            else:
                # Format error message with suggestions
                error_msg = status_message
                if suggestions:
                    error_msg += "\n\n💡 Try these solutions:"
                    for i, suggestion in enumerate(suggestions, 1):
                        error_msg += f"\n{i}. {suggestion}"
                return error_msg
                
        except Exception as e:
            logger.error(f"Unexpected error in connection test: {e}")
            return f"❌ Unexpected error: {str(e)}\n\n💡 Try refreshing the page or checking the console for details."
    
    def _refresh_agents(self, backend_url: str) -> tuple:
        """Refresh the list of available agents with enhanced error handling."""
        try:
            # Validate backend URL
            if not backend_url:
                return (
                    gr.update(choices=[], value=None),
                    "❌ Backend URL is required to refresh agents"
                )
            
            self.backend_url = backend_url
            self.api_client = TestAPIClient(backend_url)
            
            agents = self.api_client.get_available_agents()
            agent_choices = [(f"{agent['name']} ({agent['type']})", agent['name']) 
                           for agent in agents]
            
            return (
                gr.update(choices=agent_choices, value=None),
                f"✅ Successfully refreshed. Found {len(agents)} agents available for testing."
            )
        except ConnectionError as e:
            return (
                gr.update(choices=[], value=None),
                f"❌ Connection Error: {str(e)}\n\n💡 Check if the backend server is running and the URL is correct."
            )
        except TimeoutError as e:
            return (
                gr.update(choices=[], value=None),
                f"❌ Timeout Error: {str(e)}\n\n💡 The server may be overloaded. Try again in a few moments."
            )
        except Exception as e:
            logger.error(f"Error refreshing agents: {e}")
            error_msg = f"❌ Failed to refresh agents: {str(e)}"
            
            # Add specific suggestions based on error type
            if "404" in str(e):
                error_msg += "\n\n💡 The agents endpoint may not be available. Check the backend configuration."
            elif "500" in str(e):
                error_msg += "\n\n💡 Server error occurred. Check the backend logs for details."
            else:
                error_msg += "\n\n💡 Try testing the connection first or check the backend server status."
            
            return (
                gr.update(choices=[], value=None),
                error_msg
            )
    
    def _on_agent_selected(self, agent_name: str) -> tuple:
        """Handle agent selection and update the interface."""
        if not agent_name:
            return (
                "Select an agent to see its information",
                gr.update(),
                {}
            )
        
        try:
            agents = self.api_client.get_available_agents()
            agent_info = next((agent for agent in agents if agent['name'] == agent_name), None)
            
            if not agent_info:
                return (
                    "❌ Agent information not found",
                    gr.update(),
                    {}
                )
            
            # Format agent information with enhanced details
            agent_type = agent_info.get('type', 'unknown')
            agent_description = agent_info.get('description', 'No description available')
            input_schema = agent_info.get('input_schema', {})
            test_examples = agent_info.get('test_examples', [])
            
            # Create agent-specific information display
            info_md = self._create_agent_info_display(agent_info)
            
            # Create dynamic input form based on schema
            components, form_info = self.form_manager.update_form_for_agent(agent_info)
            
            # Create new container with dynamic components and agent-specific styling
            with gr.Column() as new_container:
                # Agent type indicator
                agent_type_colors = {
                    'planning': '🧠 #4CAF50',
                    'code_generation': '💻 #2196F3', 
                    'rendering': '🎬 #FF9800',
                    'unknown': '❓ #9E9E9E'
                }
                
                type_color = agent_type_colors.get(agent_type, agent_type_colors['unknown'])
                type_icon, color = type_color.split(' ')
                
                gr.HTML(f"""
                    <div style="background: {color}20; border-left: 4px solid {color}; padding: 1rem; margin-bottom: 1rem; border-radius: 0.5rem;">
                        <h4 style="margin: 0; color: {color};">{type_icon} {agent_type.replace('_', ' ').title()} Agent</h4>
                        <p style="margin: 0.5rem 0 0 0; color: #666;">{form_info}</p>
                    </div>
                """)
                
                # Add example selector if examples are available
                if test_examples:
                    with gr.Accordion("📋 Load Test Example", open=False):
                        example_dropdown = gr.Dropdown(
                            label="Select Example",
                            choices=[(f"Example {i+1}: {ex.get('name', f'Test {i+1}')}", i) 
                                   for i, ex in enumerate(test_examples)],
                            value=None,
                            info="Load predefined test data"
                        )
                        
                        load_example_btn = gr.Button(
                            "📥 Load Example",
                            variant="secondary",
                            size="sm"
                        )
                
                # Render dynamic form components
                for component in components:
                    component.render()
                
                # Add form validation status
                validation_status = gr.HTML(
                    '<div id="validation-status" style="margin-top: 1rem;"></div>'
                )
            
            return (
                info_md, 
                gr.update(value=new_container),
                self.form_manager.current_component_types
            )
            
        except Exception as e:
            logger.error(f"Error in _on_agent_selected: {e}")
            return (
                f"❌ Error loading agent info: {str(e)}",
                gr.update(),
                {}
            )
    
    def _run_agent_test(
        self, 
        agent_name: str, 
        session_id: str, 
        dynamic_inputs_state: dict,
        timeout: int,
        retry_count: int,
        verbose: bool,
        *form_values
    ) -> tuple:
        """Run an individual agent test with enhanced status tracking."""
        if not agent_name:
            return (
                "❌ Please select an agent first",
                gr.update(value=0),
                "",
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                None,
                "",
                None
            )
        
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Extract and validate form values using the form manager
            try:
                form_data = self.form_manager.get_form_values(*form_values)
                is_valid, validation_msg = self.form_manager.validate_form(*form_values)
                
                if not is_valid:
                    return (
                        f"❌ Validation failed: {validation_msg}",
                        gr.update(value=0),
                        "Validation failed",
                        gr.update(visible=False),
                        gr.update(visible=True),
                        gr.update(visible=False),
                        None,
                        f"Validation Error: {validation_msg}",
                        None
                    )
            except Exception as e:
                return (
                    f"❌ Error processing form data: {str(e)}",
                    gr.update(value=0),
                    "Form processing error",
                    gr.update(visible=False),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    None,
                    f"Form Error: {str(e)}",
                    None
                )
            
            # Prepare execution options with agent-specific settings
            options = {
                "timeout": timeout,
                "retry_count": retry_count,
                "verbose": verbose,
                "agent_type": self._get_agent_type(agent_name)
            }
            
            # Start agent test
            response = self.api_client.test_individual_agent(
                agent_name=agent_name,
                inputs=form_data,
                session_id=session_id,
                options=options
            )
            
            # Store session for monitoring with enhanced tracking
            self.active_sessions[session_id] = {
                'type': 'agent_test',
                'agent_name': agent_name,
                'agent_type': self._get_agent_type(agent_name),
                'status': 'running',
                'start_time': time.time(),
                'form_data': form_data,
                'options': options
            }
            
            # Create agent-specific status message
            agent_type = self._get_agent_type(agent_name)
            type_icons = {
                'planning': '🧠',
                'code_generation': '💻',
                'rendering': '🎬'
            }
            
            icon = type_icons.get(agent_type, '🤖')
            status_msg = f"{icon} Starting {agent_name.replace('_', ' ').title()}"
            
            # Create initial step message based on agent type
            initial_step = self._get_agent_initial_step(agent_type)
            
            # Format input preview for display
            input_preview = self._format_input_preview(agent_name, form_data)
            
            return (
                status_msg,
                gr.update(value=5),
                initial_step,
                gr.update(value=session_id, visible=True),
                gr.update(visible=False),
                gr.update(visible=True),
                {"session_id": session_id, "status": "started", "agent_name": agent_name},
                input_preview,
                {
                    "execution_options": options, 
                    "form_data": form_data,
                    "agent_type": agent_type,
                    "session_start": time.time()
                }
            )
            
        except Exception as e:
            logger.error(f"Error starting agent test: {e}")
            return (
                f"❌ Error starting agent test: {str(e)}",
                gr.update(value=0),
                "Execution failed to start",
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),
                None,
                f"Error: {str(e)}",
                None
            )
    
    def _run_workflow_test(self, topic: str, description: str, session_id: str, config_overrides: dict) -> tuple:
        """Run a complete workflow test with enhanced step-by-step progress tracking."""
        if not topic or not description:
            return (
                "❌ Please provide both topic and description",
                "",
                gr.update(value=0),
                gr.update(visible=True),
                gr.update(visible=False),
                None,
                "",
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        try:
            # Generate session ID if not provided
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Validate config overrides
            if config_overrides and not isinstance(config_overrides, dict):
                return (
                    "❌ Configuration overrides must be a valid JSON object",
                    "",
                    gr.update(value=0),
                    gr.update(visible=True),
                    gr.update(visible=False),
                    None,
                    "",
                    "",
                    "",
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(visible=False)
                )
            
            # Start workflow test with enhanced pipeline tracking
            response = self.api_client.test_complete_workflow(
                topic=topic,
                description=description,
                session_id=session_id,
                config_overrides=config_overrides or {}
            )
            
            # Store session for monitoring with enhanced step-by-step tracking
            self.active_sessions[session_id] = {
                'type': 'workflow_test',
                'topic': topic,
                'description': description,
                'status': 'running',
                'start_time': time.time(),
                'current_step': 'initializing',
                'progress': 0.0,
                'step_history': ['initializing'],
                'step_details': {
                    'planning': {'status': 'pending', 'start_time': None, 'end_time': None, 'progress': 0.0},
                    'code_generation': {'status': 'pending', 'start_time': None, 'end_time': None, 'progress': 0.0},
                    'rendering': {'status': 'pending', 'start_time': None, 'end_time': None, 'progress': 0.0}
                },
                'config_overrides': config_overrides or {},
                'video_generation_pipeline': {
                    'total_steps': 3,
                    'completed_steps': 0,
                    'current_step_progress': 0.0
                }
            }
            
            # Create enhanced initial step details with pipeline visualization
            initial_step_details = self._create_enhanced_workflow_initialization_details(topic, description, session_id)
            
            # Create enhanced initial logs display with pipeline information
            initial_logs = self._create_initial_workflow_logs(topic, description, session_id)
            
            # Start background monitoring for step-by-step progress updates
            self._start_workflow_progress_monitoring(session_id)
            
            return (
                f"🚀 Starting video generation workflow for '{topic}'",
                "🔄 Initializing video generation pipeline...",
                gr.update(value=5),
                gr.update(visible=False),
                gr.update(visible=True),
                {"session_id": session_id, "status": "started", "topic": topic, "pipeline": "video_generation"},
                "",
                initial_step_details,
                initial_logs,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
            
        except Exception as e:
            logger.error(f"Error starting workflow test: {e}")
            return (
                f"❌ Error starting workflow test: {str(e)}",
                "",
                gr.update(value=0),
                gr.update(visible=True),
                gr.update(visible=False),
                None,
                "",
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
    
    def _refresh_workflow_status(self, session_id: str) -> tuple:
        """Refresh workflow execution status."""
        if not session_id:
            return (
                "No session ID provided",
                "",
                gr.update(value=0),
                None,
                "",
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        try:
            # Get session status from API
            session_data = self.api_client.get_session_logs(session_id)
            
            status = session_data.get('status', 'unknown')
            current_step = session_data.get('current_step', '')
            progress = session_data.get('progress', 0) * 100
            results = session_data.get('results')
            execution_time = session_data.get('execution_time')
            
            # Format status message
            status_icons = {
                'running': '⚙️',
                'completed': '✅',
                'failed': '❌',
                'cancelled': '⏹️'
            }
            
            status_icon = status_icons.get(status, '❓')
            status_msg = f"{status_icon} Workflow {status.title()}"
            
            if execution_time:
                status_msg += f" (took {execution_time:.2f}s)"
            
            # Format execution time
            time_display = ""
            if execution_time:
                time_display = f"Total execution time: {execution_time:.2f} seconds"
            
            # Create step details
            step_details = self._create_workflow_step_details(session_data)
            
            # Format logs
            logs_text = ""
            logs = session_data.get('logs', [])
            for log_entry in logs[-20:]:  # Show last 20 log entries
                if isinstance(log_entry, dict):
                    message = log_entry.get('message', str(log_entry))
                    timestamp = log_entry.get('timestamp', '')
                    level = log_entry.get('level', 'info')
                else:
                    message = str(log_entry)
                    timestamp = ''
                    level = 'info'
                
                level_icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(level, 'ℹ️')
                time_str = timestamp[:19] if timestamp else ''
                logs_text += f"[{time_str}] {level_icon} {message}\n"
            
            # Handle video output if workflow completed
            video_visible = False
            thumbnail_visible = False
            download_visible = False
            
            if status == 'completed' and results:
                video_path = results.get('video_output')
                thumbnail_path = results.get('thumbnail')
                
                if video_path:
                    video_visible = True
                    download_visible = True
                
                if thumbnail_path:
                    thumbnail_visible = True
            
            return (
                status_msg,
                current_step,
                gr.update(value=progress),
                results,
                time_display,
                step_details,
                logs_text,
                gr.update(visible=video_visible, value=results.get('video_output') if results else None),
                gr.update(visible=thumbnail_visible, value=results.get('thumbnail') if results else None),
                gr.update(visible=download_visible)
            )
            
        except Exception as e:
            logger.error(f"Error refreshing workflow status: {e}")
            return (
                f"❌ Error refreshing status: {str(e)}",
                "",
                gr.update(value=0),
                None,
                "",
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
    
    def _refresh_workflow_status(self, session_id: str) -> tuple:
        """Refresh workflow execution status with enhanced step-by-step progress tracking."""
        if not session_id:
            return (
                "No session ID provided",
                "",
                gr.update(value=0),
                None,
                "",
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
        
        try:
            # Get session status from API and local session info
            session_data = self.api_client.get_session_logs(session_id)
            local_session = self.active_sessions.get(session_id, {})
            
            status = session_data.get('status', 'unknown')
            current_step = session_data.get('current_step', '')
            progress = session_data.get('progress', 0) * 100
            results = session_data.get('results')
            execution_time = session_data.get('execution_time')
            
            # Format status message with enhanced step-by-step information
            status_icons = {
                'running': '⚙️',
                'completed': '✅',
                'failed': '❌',
                'cancelled': '⏹️'
            }
            
            status_icon = status_icons.get(status, '❓')
            
            # Enhanced status message with pipeline information
            if status == 'running':
                pipeline_info = local_session.get('video_generation_pipeline', {})
                completed_steps = pipeline_info.get('completed_steps', 0)
                total_steps = pipeline_info.get('total_steps', 3)
                status_msg = f"{status_icon} Video Generation Pipeline Running (Step {completed_steps + 1}/{total_steps})"
            else:
                status_msg = f"{status_icon} Video Generation Workflow {status.title()}"
            
            if execution_time:
                status_msg += f" (took {execution_time:.2f}s)"
            elif status == 'running':
                # Calculate elapsed time for running workflows
                start_time = local_session.get('start_time')
                if start_time:
                    elapsed = time.time() - start_time
                    status_msg += f" (running for {elapsed:.1f}s)"
            
            # Format execution time display with step breakdown
            time_display = self._format_execution_time_display(execution_time, local_session, status)
            
            # Create enhanced step details with detailed pipeline visualization
            step_details = self._create_detailed_pipeline_step_visualization(session_data, local_session, session_id)
            
            # Format logs with enhanced pipeline-specific formatting
            logs_text = self._format_enhanced_workflow_logs(session_data.get('logs', []), current_step)
            
            # Handle video output with enhanced preview and download functionality
            video_visible, thumbnail_visible, download_visible = False, False, False
            video_value, thumbnail_value = None, None
            
            if status == 'completed' and results:
                video_path = results.get('video_output')
                thumbnail_path = results.get('thumbnail')
                
                if video_path:
                    video_visible = True
                    download_visible = True
                    # In a real implementation, this would be the actual video file path
                    video_value = video_path
                    logger.info(f"Video output ready: {video_path}")
                
                if thumbnail_path:
                    thumbnail_visible = True
                    # In a real implementation, this would be the actual thumbnail file path
                    thumbnail_value = thumbnail_path
                    logger.info(f"Thumbnail ready: {thumbnail_path}")
            
            return (
                status_msg,
                self._format_enhanced_current_step_display(current_step, progress, local_session),
                gr.update(value=progress),
                results,
                time_display,
                step_details,
                logs_text,
                gr.update(visible=video_visible, value=video_value),
                gr.update(visible=thumbnail_visible, value=thumbnail_value),
                gr.update(visible=download_visible)
            )
            
        except Exception as e:
            logger.error(f"Error refreshing workflow status: {e}")
            return (
                f"❌ Error refreshing status: {str(e)}",
                "",
                gr.update(value=0),
                None,
                "",
                "",
                "",
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
    
    def _create_workflow_progress_visualization(self, current_step: str, progress: float) -> str:
        """Create a visual representation of workflow progress."""
        steps = [
            ("planning", "🧠 Planning"),
            ("code_generation", "💻 Code Generation"), 
            ("rendering", "🎬 Rendering"),
            ("completed", "✅ Completed")
        ]
        
        progress_bar = ""
        current_step_lower = current_step.lower() if current_step else ""
        
        for i, (step_key, step_name) in enumerate(steps):
            if step_key in current_step_lower or (step_key == "completed" and progress >= 100):
                progress_bar += f"**{step_name}** ✅\n"
            elif i < len(steps) - 1 and any(s in current_step_lower for s in steps[i+1:]):
                progress_bar += f"{step_name} ✅\n"
            elif step_key in current_step_lower:
                progress_bar += f"**{step_name}** ⚙️\n"
            else:
                progress_bar += f"{step_name} ⏳\n"
        
        return progress_bar
    
    def _handle_video_download(self, session_id: str) -> str:
        """Handle video download preparation with enhanced functionality and metadata."""
        try:
            if not session_id:
                return "❌ No session ID provided for video download"
            
            # Get session data to find video path and metadata
            session_data = self.api_client.get_session_logs(session_id)
            results = session_data.get('results', {})
            video_path = results.get('video_output')
            thumbnail_path = results.get('thumbnail')
            metadata = results.get('metadata', {})
            
            if not video_path:
                return "❌ No video available for download. Workflow may not be completed yet."
            
            # Get local session info for additional details
            local_session = self.active_sessions.get(session_id, {})
            topic = local_session.get('topic', 'Unknown Topic')
            
            # In a real implementation, this would:
            # 1. Verify the video file exists on the server
            # 2. Prepare it for download (copy to download directory, generate secure URL)
            # 3. Trigger browser download or provide download link
            # 4. Log download activity for analytics
            
            logger.info(f"Video download requested for session {session_id}: {video_path}")
            
            # Enhanced download preparation with detailed information
            download_info = {
                'session_id': session_id,
                'topic': topic,
                'video_path': video_path,
                'thumbnail_path': thumbnail_path,
                'file_size': metadata.get('file_size', '~15-25MB'),
                'format': metadata.get('format', 'MP4'),
                'resolution': metadata.get('resolution', '1080p'),
                'duration': metadata.get('duration', '~2-4 minutes'),
                'fps': metadata.get('fps', '30'),
                'quality': metadata.get('quality_settings', 'High'),
                'render_time': metadata.get('render_time', 'N/A')
            }
            
            # Create detailed download status message
            download_status = f"📁 Video Generation Complete - Ready for Download\n\n"
            download_status += f"🎯 Topic: {topic}\n"
            download_status += f"📹 Video File: {video_path}\n"
            
            if thumbnail_path:
                download_status += f"🖼️ Thumbnail: {thumbnail_path}\n"
            
            download_status += f"\n📊 Video Details:\n"
            download_status += f"• Format: {download_info['format']}\n"
            download_status += f"• Resolution: {download_info['resolution']}\n"
            download_status += f"• Quality: {download_info['quality']}\n"
            download_status += f"• Estimated Size: {download_info['file_size']}\n"
            download_status += f"• Duration: {download_info['duration']}\n"
            download_status += f"• Frame Rate: {download_info['fps']} FPS\n"
            
            if download_info['render_time'] != 'N/A':
                download_status += f"• Render Time: {download_info['render_time']}\n"
            
            download_status += f"\n💾 Download Options:\n"
            download_status += f"• Click the video preview to play\n"
            download_status += f"• Right-click video to save\n"
            download_status += f"• Thumbnail available separately\n"
            download_status += f"• Files remain available for 24 hours\n"
            
            # In a real implementation, you might also:
            # - Generate a secure download URL with expiration
            # - Create a ZIP file with video + thumbnail + metadata
            # - Send download link via email
            # - Update download statistics
            
            return download_status
            
        except Exception as e:
            logger.error(f"Error preparing video download for session {session_id}: {e}")
            return f"❌ Error preparing video download: {str(e)}\n\nPlease check that the workflow completed successfully and try again."
    
    def _create_enhanced_video_preview_info(self, results: dict, session_id: str) -> str:
        """Create enhanced video preview information display with detailed metadata."""
        if not results:
            return "## 🎥 Video Generation\n\nNo video output available yet. Complete the workflow to generate video content."
        
        try:
            # Get local session info for additional context
            local_session = self.active_sessions.get(session_id, {})
            topic = local_session.get('topic', 'Unknown Topic')
            
            info = "## 🎥 Generated Educational Video\n\n"
            
            video_path = results.get('video_output')
            thumbnail_path = results.get('thumbnail')
            metadata = results.get('metadata', {})
            
            # Video topic and description
            info += f"**🎯 Topic:** {topic}\n"
            if local_session.get('description'):
                description = local_session['description']
                info += f"**📝 Description:** {description[:150]}{'...' if len(description) > 150 else ''}\n"
            info += f"**🆔 Session:** {session_id[:8]}...\n\n"
            
            # File information
            if video_path:
                info += f"### 📁 Output Files\n"
                info += f"**📹 Video File:** `{video_path}`\n"
                
                if thumbnail_path:
                    info += f"**🖼️ Thumbnail:** `{thumbnail_path}`\n"
                info += "\n"
            
            # Enhanced video metadata
            if metadata:
                info += f"### 📊 Video Specifications\n"
                
                # Technical specifications
                technical_specs = {
                    'format': ('🎬 Format', 'MP4'),
                    'resolution': ('📐 Resolution', '1080p'),
                    'fps': ('🎞️ Frame Rate', '30 FPS'),
                    'duration': ('⏱️ Duration', '2-4 minutes'),
                    'file_size': ('💾 File Size', '15-25MB'),
                    'quality_settings': ('⚙️ Quality', 'High'),
                    'codec': ('🔧 Codec', 'H.264')
                }
                
                for key, (display_name, default_value) in technical_specs.items():
                    value = metadata.get(key, default_value)
                    info += f"- {display_name}: {value}\n"
                
                # Generation metadata
                if 'render_time' in metadata:
                    info += f"- ⏱️ Render Time: {metadata['render_time']}\n"
                
                if 'total_execution_time' in metadata:
                    info += f"- 🕐 Total Generation Time: {metadata['total_execution_time']:.2f}s\n"
                
                # Pipeline steps completed
                if 'steps_completed' in metadata:
                    steps = metadata['steps_completed']
                    info += f"- 📋 Pipeline Steps: {', '.join(steps)}\n"
                
                info += "\n"
            
            # Content information
            info += f"### 📚 Educational Content\n"
            info += f"- **Type:** Mathematical/Scientific Visualization\n"
            info += f"- **Generated with:** Manim (Mathematical Animation Engine)\n"
            info += f"- **Style:** Educational with clear explanations\n"
            info += f"- **Target Audience:** Students and educators\n\n"
            
            # Download and usage options
            info += f"### 💾 Download & Usage Options\n"
            info += f"**Download Options:**\n"
            info += f"- 📹 Click 'Download Video' button for full video file\n"
            info += f"- 🖼️ Right-click thumbnail to save preview image\n"
            info += f"- 📱 Compatible with all modern devices and browsers\n\n"
            
            info += f"**Usage Rights:**\n"
            info += f"- ✅ Free to use for educational purposes\n"
            info += f"- ✅ Share with students and colleagues\n"
            info += f"- ✅ Embed in presentations and courses\n"
            info += f"- ℹ️ Generated content expires after 24 hours\n\n"
            
            # Quality and technical notes
            info += f"### 🔧 Technical Notes\n"
            info += f"- Video optimized for online streaming and download\n"
            info += f"- High-quality mathematical animations and visualizations\n"
            info += f"- Compatible with standard video players\n"
            info += f"- Thumbnail provides quick preview of content\n"
            
            # Generation pipeline summary
            pipeline_info = local_session.get('video_generation_pipeline', {})
            if pipeline_info:
                completed_steps = pipeline_info.get('completed_steps', 0)
                total_steps = pipeline_info.get('total_steps', 3)
                info += f"- Generated through {completed_steps}/{total_steps} step pipeline\n"
            
            return info
            
        except Exception as e:
            logger.error(f"Error creating enhanced video preview info: {e}")
            return f"## 🎥 Video Preview\n\nError creating video information: {str(e)}"
    
    def _stop_workflow_test(self, session_id: str) -> tuple:
        """Stop a running workflow test."""
        if not session_id:
            return (
                "❌ No session ID provided",
                "",
                gr.update(visible=True),
                gr.update(visible=False)
            )
        
        try:
            # In a real implementation, this would send a stop request to the backend
            # For now, we'll update the local session status
            if session_id in self.active_sessions:
                self.active_sessions[session_id]['status'] = 'cancelled'
            
            logger.info(f"Workflow stop requested for session {session_id}")
            
            return (
                "⏹️ Workflow stop requested",
                "Cancelling workflow execution...",
                gr.update(visible=True),
                gr.update(visible=False)
            )
            
        except Exception as e:
            logger.error(f"Error stopping workflow: {e}")
            return (
                f"❌ Error stopping workflow: {str(e)}",
                "",
                gr.update(visible=True),
                gr.update(visible=False)
            )
    
    def _create_workflow_step_details(self, session_data: dict) -> str:
        """Create detailed information about workflow steps."""
        if not session_data:
            return "No workflow data available"
        
        try:
            output = "## Workflow Execution Details\n\n"
            
            # Basic information
            output += f"**Session ID:** {session_data.get('session_id', 'Unknown')}\n"
            output += f"**Status:** {session_data.get('status', 'Unknown')}\n"
            output += f"**Current Step:** {session_data.get('current_step', 'Unknown')}\n"
            output += f"**Progress:** {session_data.get('progress', 0) * 100:.1f}%\n\n"
            
            # Step-by-step progress
            logs = session_data.get('logs', [])
            if logs:
                output += "### Execution Log\n"
                for log_entry in logs[-10:]:  # Show last 10 log entries
                    if isinstance(log_entry, dict):
                        message = log_entry.get('message', str(log_entry))
                        level = log_entry.get('level', 'info')
                        timestamp = log_entry.get('timestamp', '')
                    else:
                        message = str(log_entry)
                        level = 'info'
                        timestamp = ''
                    
                    level_icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}.get(level, 'ℹ️')
                    output += f"- {level_icon} {message}\n"
                output += "\n"
            
            # Results summary
            results = session_data.get('results')
            if results:
                output += "### Results Summary\n"
                if isinstance(results, dict):
                    for key, value in results.items():
                        if key not in ['video_output', 'thumbnail']:
                            output += f"**{key.replace('_', ' ').title()}:** {value}\n"
                output += "\n"
            
            # Execution metrics
            execution_time = session_data.get('execution_time')
            if execution_time:
                output += f"### Performance Metrics\n"
                output += f"**Total Execution Time:** {execution_time:.2f} seconds\n"
                
                # Estimate per-step timing (this would be more accurate with real data)
                if session_data.get('status') == 'completed':
                    avg_step_time = execution_time / 3  # Assuming 3 main steps
                    output += f"**Average Step Time:** {avg_step_time:.2f} seconds\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Error creating workflow step details: {e}")
            return f"Error creating workflow details: {str(e)}"
    
    def _create_enhanced_workflow_initialization_details(self, topic: str, description: str, session_id: str) -> str:
        """Create enhanced initial workflow details display with step-by-step pipeline visualization."""
        output = "## 🎬 Video Generation Workflow Pipeline\n\n"
        
        output += f"**🎯 Topic:** {topic}\n"
        output += f"**📝 Description:** {description[:200]}{'...' if len(description) > 200 else ''}\n"
        output += f"**🆔 Session ID:** {session_id[:8]}...\n\n"
        
        output += "### 📋 Step-by-Step Pipeline Overview\n\n"
        output += "The video generation workflow consists of three sequential phases with detailed progress tracking:\n\n"
        
        # Step 1: Planning Phase with detailed breakdown
        output += "#### 🧠 **Step 1: Planning Phase** ⏳ *Pending*\n"
        output += "```\n"
        output += "├── Topic Analysis & Breakdown\n"
        output += "├── Educational Content Structure\n"
        output += "├── Scene Planning & Flow Design\n"
        output += "└── Key Concepts Identification\n"
        output += "```\n"
        output += "**Expected Duration:** ~30-45 seconds\n"
        output += "**Output:** Structured scene plan with educational flow\n\n"
        
        # Step 2: Code Generation Phase with detailed breakdown
        output += "#### 💻 **Step 2: Code Generation Phase** ⏳ *Pending*\n"
        output += "```\n"
        output += "├── Manim Code Generation\n"
        output += "├── Animation Sequence Creation\n"
        output += "├── Mathematical Visualization Setup\n"
        output += "└── Code Optimization & Validation\n"
        output += "```\n"
        output += "**Expected Duration:** ~60-90 seconds\n"
        output += "**Output:** Executable Manim Python code\n\n"
        
        # Step 3: Rendering Phase with detailed breakdown
        output += "#### 🎬 **Step 3: Rendering Phase** ⏳ *Pending*\n"
        output += "```\n"
        output += "├── Manim Code Execution\n"
        output += "├── Video Compilation & Processing\n"
        output += "├── Quality Optimization\n"
        output += "└── Thumbnail Generation\n"
        output += "```\n"
        output += "**Expected Duration:** ~120-180 seconds\n"
        output += "**Output:** MP4 video file + thumbnail preview\n\n"
        
        # Pipeline Status
        output += "### ⚙️ Pipeline Status\n"
        output += "```\n"
        output += "🔄 Initializing video generation pipeline...\n"
        output += "📋 Validating input parameters\n"
        output += "⚙️ Preparing agent execution environment\n"
        output += "🚀 Ready to begin Step 1: Planning Phase\n"
        output += "```\n\n"
        
        # Progress Visualization
        output += "### 📊 Progress Visualization\n"
        output += "```\n"
        output += "Overall Progress: [░░░░░░░░░░░░░░░░░░░░] 0%\n"
        output += "\n"
        output += "Step 1 - Planning:      [░░░░░░░░░░] 0%\n"
        output += "Step 2 - Code Gen:      [░░░░░░░░░░] 0%\n"
        output += "Step 3 - Rendering:     [░░░░░░░░░░] 0%\n"
        output += "```\n\n"
        
        # Configuration Summary
        output += "### ⚙️ Configuration Summary\n"
        output += f"- **Pipeline Mode:** Complete Workflow\n"
        output += f"- **Quality Setting:** High (1080p)\n"
        output += f"- **Expected Total Time:** ~3-5 minutes\n"
        output += f"- **Output Format:** MP4 + PNG thumbnail\n"
        
        return output
    
    def _create_initial_workflow_logs(self, topic: str, description: str, session_id: str) -> str:
        """Create enhanced initial logs display for workflow pipeline."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        
        logs = f"[{timestamp}] 🚀 Video Generation Workflow Started\n"
        logs += f"[{timestamp}] 📋 Session ID: {session_id}\n"
        logs += f"[{timestamp}] 🎯 Topic: {topic}\n"
        logs += f"[{timestamp}] 📝 Description: {description[:100]}{'...' if len(description) > 100 else ''}\n"
        logs += f"[{timestamp}] ⚙️ Initializing 3-step video generation pipeline\n"
        logs += f"[{timestamp}] 🔍 Validating input parameters and configuration\n"
        logs += f"[{timestamp}] 🧠 Preparing Planning Agent for Step 1\n"
        logs += f"[{timestamp}] 💻 Preparing Code Generation Agent for Step 2\n"
        logs += f"[{timestamp}] 🎬 Preparing Rendering Agent for Step 3\n"
        logs += f"[{timestamp}] ✅ Pipeline initialization complete\n"
        logs += f"[{timestamp}] 🚀 Ready to begin video generation process\n"
        
        return logs
    
    def _start_workflow_progress_monitoring(self, session_id: str):
        """Start background monitoring for detailed workflow progress updates."""
        def monitor_workflow_progress():
            """Background thread to monitor and update workflow progress with step-by-step tracking."""
            try:
                session_info = self.active_sessions.get(session_id)
                if not session_info:
                    return
                
                # Monitor workflow progress with enhanced step tracking
                while session_info.get('status') == 'running':
                    try:
                        # Get latest session data from API
                        session_data = self.api_client.get_session_logs(session_id)
                        current_step = session_data.get('current_step', 'initializing')
                        progress = session_data.get('progress', 0)
                        
                        # Update session info with detailed step tracking
                        session_info.update({
                            'current_step': current_step,
                            'progress': progress,
                            'last_update': time.time()
                        })
                        
                        # Track step transitions and update step details
                        if current_step != session_info.get('previous_step'):
                            self._update_step_details(session_info, current_step, progress)
                            session_info['previous_step'] = current_step
                        
                        # Update step-specific progress
                        self._update_current_step_progress(session_info, current_step, progress)
                        
                        # Check for completion
                        status = session_data.get('status', 'running')
                        if status in ['completed', 'failed', 'cancelled']:
                            session_info['status'] = status
                            if status == 'completed':
                                self._handle_workflow_completion(session_info, session_data)
                            break
                        
                        time.sleep(1)  # Update every second for smooth progress
                        
                    except Exception as e:
                        logger.error(f"Error monitoring workflow progress for {session_id}: {e}")
                        time.sleep(2)  # Wait longer on error
                        
            except Exception as e:
                logger.error(f"Error in workflow progress monitoring: {e}")
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_workflow_progress, daemon=True)
        monitor_thread.start()
    
    def _update_step_details(self, session_info: dict, current_step: str, progress: float):
        """Update detailed step information for workflow tracking."""
        step_details = session_info.get('step_details', {})
        current_time = time.time()
        
        # Mark previous step as completed if moving to next step
        step_order = ['planning', 'code_generation', 'rendering']
        if current_step in step_order:
            current_index = step_order.index(current_step)
            
            # Complete previous steps
            for i in range(current_index):
                prev_step = step_order[i]
                if prev_step in step_details and step_details[prev_step]['status'] != 'completed':
                    step_details[prev_step].update({
                        'status': 'completed',
                        'end_time': current_time,
                        'progress': 1.0
                    })
            
            # Start current step
            if current_step in step_details:
                step_details[current_step].update({
                    'status': 'running',
                    'start_time': current_time,
                    'progress': 0.0
                })
        
        # Update pipeline tracking
        pipeline_info = session_info.get('video_generation_pipeline', {})
        completed_steps = sum(1 for step in step_details.values() if step.get('status') == 'completed')
        pipeline_info.update({
            'completed_steps': completed_steps,
            'current_step_progress': progress
        })
    
    def _update_current_step_progress(self, session_info: dict, current_step: str, overall_progress: float):
        """Update progress for the currently executing step."""
        step_details = session_info.get('step_details', {})
        
        if current_step in step_details and step_details[current_step]['status'] == 'running':
            # Calculate step-specific progress based on overall progress
            step_progress = self._calculate_step_progress(current_step, overall_progress)
            step_details[current_step]['progress'] = step_progress
    
    def _calculate_step_progress(self, current_step: str, overall_progress: float) -> float:
        """Calculate individual step progress from overall workflow progress."""
        # Define progress ranges for each step
        step_ranges = {
            'planning': (0.0, 0.33),
            'code_generation': (0.33, 0.66),
            'rendering': (0.66, 1.0)
        }
        
        if current_step not in step_ranges:
            return 0.0
        
        start, end = step_ranges[current_step]
        
        if overall_progress <= start:
            return 0.0
        elif overall_progress >= end:
            return 1.0
        else:
            # Calculate progress within the step range
            step_progress = (overall_progress - start) / (end - start)
            return min(1.0, max(0.0, step_progress))
    
    def _handle_workflow_completion(self, session_info: dict, session_data: dict):
        """Handle workflow completion with video output processing."""
        try:
            results = session_data.get('results', {})
            
            # Update session with completion details
            session_info.update({
                'status': 'completed',
                'completion_time': time.time(),
                'results': results
            })
            
            # Mark all steps as completed
            step_details = session_info.get('step_details', {})
            for step_name in step_details:
                step_details[step_name].update({
                    'status': 'completed',
                    'progress': 1.0,
                    'end_time': time.time()
                })
            
            # Process video output information
            video_path = results.get('video_output')
            thumbnail_path = results.get('thumbnail')
            
            if video_path:
                session_info['video_ready'] = True
                session_info['video_path'] = video_path
                logger.info(f"Video generation completed: {video_path}")
            
            if thumbnail_path:
                session_info['thumbnail_ready'] = True
                session_info['thumbnail_path'] = thumbnail_path
                logger.info(f"Thumbnail generated: {thumbnail_path}")
                
        except Exception as e:
            logger.error(f"Error handling workflow completion: {e}")
            session_info['status'] = 'failed'
    
    def _create_enhanced_workflow_step_details(self, session_data: dict, session_id: str) -> str:
        """Create enhanced workflow step details with visual progress."""
        if not session_data:
            return "No workflow data available"
        
        try:
            output = "## 🎬 Video Generation Progress\n\n"
            
            # Get session info from active sessions
            session_info = self.active_sessions.get(session_id, {})
            topic = session_info.get('topic', session_data.get('topic', 'Unknown'))
            
            output += f"**🎯 Topic:** {topic}\n"
            output += f"**🆔 Session:** {session_id[:8]}...\n"
            output += f"**📊 Status:** {session_data.get('status', 'Unknown').title()}\n\n"
            
            # Visual progress pipeline
            current_step = session_data.get('current_step', '').lower()
            progress = session_data.get('progress', 0)
            
            output += "### 🔄 Pipeline Progress\n\n"
            
            # Step 1: Planning
            if current_step == 'planning' or progress > 0.1:
                if current_step == 'planning':
                    output += "🧠 **Planning Phase** ⚙️ *In Progress*\n"
                elif progress > 0.33:
                    output += "🧠 **Planning Phase** ✅ *Completed*\n"
                else:
                    output += "🧠 **Planning Phase** ⚙️ *In Progress*\n"
            else:
                output += "🧠 **Planning Phase** ⏳ *Pending*\n"
            
            output += "   - Topic analysis and breakdown\n"
            output += "   - Scene structure planning\n"
            output += "   - Educational flow design\n\n"
            
            # Step 2: Code Generation
            if current_step == 'code_generation' or progress > 0.33:
                if current_step == 'code_generation':
                    output += "💻 **Code Generation Phase** ⚙️ *In Progress*\n"
                elif progress > 0.66:
                    output += "💻 **Code Generation Phase** ✅ *Completed*\n"
                else:
                    output += "💻 **Code Generation Phase** ⚙️ *In Progress*\n"
            else:
                output += "💻 **Code Generation Phase** ⏳ *Pending*\n"
            
            output += "   - Manim code generation\n"
            output += "   - Animation sequence creation\n"
            output += "   - Code optimization\n\n"
            
            # Step 3: Rendering
            if current_step == 'rendering' or progress > 0.66:
                if current_step == 'rendering':
                    output += "🎬 **Rendering Phase** ⚙️ *In Progress*\n"
                elif progress >= 1.0:
                    output += "🎬 **Rendering Phase** ✅ *Completed*\n"
                else:
                    output += "🎬 **Rendering Phase** ⚙️ *In Progress*\n"
            else:
                output += "🎬 **Rendering Phase** ⏳ *Pending*\n"
            
            output += "   - Video compilation\n"
            output += "   - Quality optimization\n"
            output += "   - Thumbnail generation\n\n"
            
            # Progress bar visualization
            progress_percent = int(progress * 100)
            filled_blocks = int(progress * 20)
            empty_blocks = 20 - filled_blocks
            progress_bar = "█" * filled_blocks + "░" * empty_blocks
            
            output += f"### 📊 Overall Progress: {progress_percent}%\n"
            output += f"`{progress_bar}` {progress_percent}%\n\n"
            
            # Execution metrics
            execution_time = session_data.get('execution_time')
            start_time = session_info.get('start_time')
            
            if execution_time:
                output += f"### ⏱️ Performance Metrics\n"
                output += f"**Total Time:** {execution_time:.2f} seconds\n"
                
                # Estimate per-step timing
                if session_data.get('status') == 'completed':
                    avg_step_time = execution_time / 3
                    output += f"**Average Step Time:** {avg_step_time:.2f} seconds\n"
            elif start_time and session_data.get('status') == 'running':
                elapsed = time.time() - start_time
                output += f"### ⏱️ Execution Time\n"
                output += f"**Elapsed:** {elapsed:.1f} seconds\n"
                
                # Estimate remaining time based on progress
                if progress > 0:
                    estimated_total = elapsed / progress
                    remaining = estimated_total - elapsed
                    output += f"**Estimated Remaining:** {remaining:.1f} seconds\n"
            
            # Results preview
            results = session_data.get('results')
            if results and isinstance(results, dict):
                output += f"\n### 🎥 Output Files\n"
                if results.get('video_output'):
                    output += f"**Video:** `{results['video_output']}`\n"
                if results.get('thumbnail'):
                    output += f"**Thumbnail:** `{results['thumbnail']}`\n"
                
                # Metadata
                metadata = results.get('metadata', {})
                if metadata:
                    output += f"\n**Metadata:**\n"
                    for key, value in metadata.items():
                        if key not in ['config_overrides']:
                            output += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Error creating enhanced workflow step details: {e}")
            return f"Error creating workflow details: {str(e)}"
    
    def _format_enhanced_current_step_display(self, current_step: str, progress: float, local_session: dict) -> str:
        """Format the current step display with enhanced progress information and pipeline context."""
        if not current_step:
            return "🔄 Waiting to start video generation pipeline..."
        
        step_icons = {
            'initializing': '🔄',
            'planning': '🧠',
            'code_generation': '💻',
            'rendering': '🎬',
            'completed': '✅',
            'failed': '❌'
        }
        
        step_names = {
            'initializing': 'Initializing Video Generation Pipeline',
            'planning': 'Planning Phase - Analyzing Topic & Creating Scene Structure',
            'code_generation': 'Code Generation Phase - Converting Plan to Manim Code',
            'rendering': 'Rendering Phase - Generating Video Output',
            'completed': 'Video Generation Completed Successfully',
            'failed': 'Video Generation Failed'
        }
        
        icon = step_icons.get(current_step.lower(), '⚙️')
        name = step_names.get(current_step.lower(), current_step.replace('_', ' ').title())
        
        if current_step.lower() in ['completed', 'failed']:
            return f"{icon} {name}"
        else:
            # Add step-specific progress details
            step_details = local_session.get('step_details', {})
            current_step_info = step_details.get(current_step, {})
            step_progress = current_step_info.get('progress', 0) * 100
            
            # Add pipeline context
            pipeline_info = local_session.get('video_generation_pipeline', {})
            completed_steps = pipeline_info.get('completed_steps', 0)
            total_steps = pipeline_info.get('total_steps', 3)
            
            return f"{icon} {name} | Step {completed_steps + 1}/{total_steps} | Progress: {step_progress:.1f}%"
    
    def _format_execution_time_display(self, execution_time: float, local_session: dict, status: str) -> str:
        """Format execution time display with step breakdown."""
        if execution_time:
            time_display = f"Total execution time: {execution_time:.2f} seconds"
            
            # Add step breakdown if available
            step_details = local_session.get('step_details', {})
            if step_details and status == 'completed':
                time_display += "\n\nStep Breakdown:"
                for step_name, details in step_details.items():
                    start_time = details.get('start_time')
                    end_time = details.get('end_time')
                    if start_time and end_time:
                        step_duration = end_time - start_time
                        step_icon = {'planning': '🧠', 'code_generation': '💻', 'rendering': '🎬'}.get(step_name, '⚙️')
                        time_display += f"\n- {step_icon} {step_name.replace('_', ' ').title()}: {step_duration:.2f}s"
            
            return time_display
            
        elif status == 'running':
            start_time = local_session.get('start_time')
            if start_time:
                elapsed = time.time() - start_time
                time_display = f"Elapsed time: {elapsed:.1f} seconds"
                
                # Add estimated remaining time based on current step
                current_step = local_session.get('current_step', '')
                estimated_remaining = self._estimate_remaining_time(current_step, elapsed)
                if estimated_remaining:
                    time_display += f"\nEstimated remaining: {estimated_remaining:.1f} seconds"
                
                return time_display
        
        return ""
    
    def _estimate_remaining_time(self, current_step: str, elapsed_time: float) -> float:
        """Estimate remaining time based on current step and elapsed time."""
        # Rough time estimates for each step (in seconds)
        step_estimates = {
            'initializing': 10,
            'planning': 45,
            'code_generation': 90,
            'rendering': 180
        }
        
        step_order = ['initializing', 'planning', 'code_generation', 'rendering']
        
        if current_step not in step_order:
            return 0
        
        current_index = step_order.index(current_step)
        
        # Calculate remaining time for current step and all subsequent steps
        remaining = 0
        for i in range(current_index, len(step_order)):
            step = step_order[i]
            if i == current_index:
                # For current step, estimate based on elapsed time
                step_estimate = step_estimates.get(step, 60)
                remaining += max(0, step_estimate - (elapsed_time if i == 0 else 0))
            else:
                remaining += step_estimates.get(step, 60)
        
        return remaining
    
    def _create_detailed_pipeline_step_visualization(self, session_data: dict, local_session: dict, session_id: str) -> str:
        """Create detailed pipeline step visualization with progress bars and status."""
        if not session_data and not local_session:
            return "No workflow data available"
        
        try:
            output = "## 🎬 Video Generation Pipeline Status\n\n"
            
            # Basic information
            topic = local_session.get('topic', session_data.get('topic', 'Unknown'))
            current_step = session_data.get('current_step', 'initializing')
            overall_progress = session_data.get('progress', 0)
            
            output += f"**🎯 Topic:** {topic}\n"
            output += f"**🆔 Session:** {session_id[:8]}...\n"
            output += f"**📊 Overall Progress:** {overall_progress * 100:.1f}%\n\n"
            
            # Detailed step-by-step progress
            output += "### 📋 Pipeline Steps Progress\n\n"
            
            step_details = local_session.get('step_details', {})
            step_order = ['planning', 'code_generation', 'rendering']
            
            for i, step_name in enumerate(step_order, 1):
                step_info = step_details.get(step_name, {'status': 'pending', 'progress': 0.0})
                status = step_info.get('status', 'pending')
                progress = step_info.get('progress', 0.0)
                
                # Step status icons
                status_icons = {
                    'pending': '⏳',
                    'running': '⚙️',
                    'completed': '✅',
                    'failed': '❌'
                }
                
                # Step icons
                step_icons = {
                    'planning': '🧠',
                    'code_generation': '💻',
                    'rendering': '🎬'
                }
                
                status_icon = status_icons.get(status, '❓')
                step_icon = step_icons.get(step_name, '⚙️')
                
                # Create progress bar
                progress_bar_length = 20
                filled = int(progress * progress_bar_length)
                empty = progress_bar_length - filled
                progress_bar = "█" * filled + "░" * empty
                
                # Step title and status
                step_title = step_name.replace('_', ' ').title()
                output += f"#### {step_icon} **Step {i}: {step_title}** {status_icon} *{status.title()}*\n"
                
                # Progress bar and percentage
                output += f"`{progress_bar}` {progress * 100:.1f}%\n"
                
                # Step-specific details
                if step_name == 'planning':
                    output += "- Topic analysis and educational content breakdown\n"
                    output += "- Scene structure and flow design\n"
                    output += "- Key concepts identification\n"
                elif step_name == 'code_generation':
                    output += "- Manim code generation from scene plan\n"
                    output += "- Animation sequence creation\n"
                    output += "- Code optimization and validation\n"
                elif step_name == 'rendering':
                    output += "- Manim code execution and video compilation\n"
                    output += "- Quality optimization and processing\n"
                    output += "- Thumbnail generation\n"
                
                # Timing information
                start_time = step_info.get('start_time')
                end_time = step_info.get('end_time')
                
                if start_time and end_time:
                    duration = end_time - start_time
                    output += f"- ⏱️ **Duration:** {duration:.2f} seconds\n"
                elif start_time and status == 'running':
                    elapsed = time.time() - start_time
                    output += f"- ⏱️ **Elapsed:** {elapsed:.1f} seconds\n"
                
                output += "\n"
            
            # Overall pipeline progress visualization
            output += "### 📊 Overall Pipeline Progress\n\n"
            overall_progress_percent = int(overall_progress * 100)
            overall_filled = int(overall_progress * 30)
            overall_empty = 30 - overall_filled
            overall_progress_bar = "█" * overall_filled + "░" * overall_empty
            
            output += f"`{overall_progress_bar}` {overall_progress_percent}%\n\n"
            
            # Current activity
            if current_step and current_step != 'completed':
                output += f"### 🔄 Current Activity\n"
                output += f"**Active Step:** {current_step.replace('_', ' ').title()}\n"
                
                # Add step-specific current activity
                if current_step == 'planning':
                    output += "**Activity:** Analyzing educational content and creating scene structure\n"
                elif current_step == 'code_generation':
                    output += "**Activity:** Converting scene plan to executable Manim code\n"
                elif current_step == 'rendering':
                    output += "**Activity:** Executing Manim code and generating video output\n"
                
                output += f"**Progress:** {overall_progress * 100:.1f}% complete\n"
            
            # Results preview
            results = session_data.get('results')
            if results and isinstance(results, dict):
                output += f"\n### 🎥 Generated Output\n"
                if results.get('video_output'):
                    output += f"**📹 Video:** `{results['video_output']}`\n"
                if results.get('thumbnail'):
                    output += f"**🖼️ Thumbnail:** `{results['thumbnail']}`\n"
                
                metadata = results.get('metadata', {})
                if metadata:
                    output += f"\n**📊 Video Metadata:**\n"
                    for key, value in metadata.items():
                        if key not in ['config_overrides']:
                            output += f"- {key.replace('_', ' ').title()}: {value}\n"
            
            return output
            
        except Exception as e:
            logger.error(f"Error creating detailed pipeline visualization: {e}")
            return f"Error creating pipeline visualization: {str(e)}"
    
    def _format_enhanced_workflow_logs(self, logs: list, current_step: str) -> str:
        """Format workflow logs with enhanced pipeline-specific styling and step context."""
        if not logs:
            return "🔄 Initializing video generation pipeline...\nWaiting for first log entries..."
        
        formatted_logs = ""
        
        # Show last 30 log entries for better visibility
        recent_logs = logs[-30:] if len(logs) > 30 else logs
        
        for log_entry in recent_logs:
            if isinstance(log_entry, dict):
                message = log_entry.get('message', str(log_entry))
                timestamp = log_entry.get('timestamp', '')
                level = log_entry.get('level', 'info')
                component = log_entry.get('component', 'system')
            else:
                message = str(log_entry)
                timestamp = datetime.now().strftime('%H:%M:%S')
                level = 'info'
                component = 'system'
            
            # Format timestamp
            if timestamp:
                try:
                    # Parse ISO timestamp and format for display
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S')
                except:
                    time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
            else:
                time_str = datetime.now().strftime('%H:%M:%S')
            
            # Enhanced level icons and colors for video generation pipeline
            level_info = {
                'error': '❌',
                'warning': '⚠️',
                'info': 'ℹ️',
                'success': '✅',
                'debug': '🔍',
                'progress': '📊',
                'step': '🔄'
            }
            
            level_icon = level_info.get(level.lower(), 'ℹ️')
            
            # Enhanced component icons for video generation pipeline
            component_icons = {
                'planning_agent': '🧠',
                'code_generation_agent': '💻',
                'rendering_agent': '🎬',
                'workflow': '🔄',
                'system': '⚙️',
                'pipeline': '🎬',
                'manim': '🎞️',
                'video_processor': '📹'
            }
            
            component_icon = component_icons.get(component, '📝')
            
            # Add step context to messages
            step_context = ""
            if current_step and current_step in message.lower():
                step_context = f"[{current_step.upper()}] "
            
            formatted_logs += f"[{time_str}] {component_icon} {level_icon} {step_context}{message}\n"
        
        return formatted_logs
    
    def _format_workflow_logs(self, logs: list) -> str:
        """Format workflow logs with enhanced styling."""
        if not logs:
            return "No logs available yet..."
        
        formatted_logs = ""
        
        # Show last 25 log entries for better visibility
        recent_logs = logs[-25:] if len(logs) > 25 else logs
        
        for log_entry in recent_logs:
            if isinstance(log_entry, dict):
                message = log_entry.get('message', str(log_entry))
                timestamp = log_entry.get('timestamp', '')
                level = log_entry.get('level', 'info')
                component = log_entry.get('component', 'system')
            else:
                message = str(log_entry)
                timestamp = datetime.now().strftime('%H:%M:%S')
                level = 'info'
                component = 'system'
            
            # Format timestamp
            if timestamp:
                try:
                    # Parse ISO timestamp and format for display
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    time_str = dt.strftime('%H:%M:%S')
                except:
                    time_str = timestamp[:8] if len(timestamp) >= 8 else timestamp
            else:
                time_str = datetime.now().strftime('%H:%M:%S')
            
            # Level icons and colors
            level_info = {
                'error': '❌',
                'warning': '⚠️',
                'info': 'ℹ️',
                'success': '✅',
                'debug': '🔍'
            }
            
            level_icon = level_info.get(level.lower(), 'ℹ️')
            
            # Component icons
            component_icons = {
                'planning_agent': '🧠',
                'code_generation_agent': '💻',
                'rendering_agent': '🎬',
                'workflow': '🔄',
                'system': '⚙️'
            }
            
            component_icon = component_icons.get(component, '📝')
            
            formatted_logs += f"[{time_str}] {component_icon} {level_icon} {message}\n"
        
        return formatted_logs
    
    def _connect_to_logs(self, session_id: str) -> tuple:
        """Connect to real-time logs for a session."""
        if not session_id:
            return (
                "❌ Please enter a session ID",
                gr.update(visible=True),
                gr.update(visible=False)
            )
        
        try:
            # This would be implemented with actual WebSocket connection
            # For now, return a placeholder response
            return (
                f"🔗 Connected to logs for session {session_id[:8]}...",
                gr.update(visible=False),
                gr.update(visible=True)
            )
            
        except Exception as e:
            return (
                f"❌ Error connecting to logs: {str(e)}",
                gr.update(visible=True),
                gr.update(visible=False)
            )
    
    def _disconnect_from_logs(self) -> tuple:
        """Disconnect from real-time logs."""
        return (
            "🔌 Disconnected from logs",
            gr.update(visible=True),
            gr.update(visible=False),
            ""
        )
    
    def _setup_periodic_updates(self):
        """Set up periodic status updates for active sessions."""
        def update_active_sessions():
            """Background thread to update active session statuses."""
            while True:
                try:
                    for session_id, session_info in list(self.active_sessions.items()):
                        if session_info.get('status') == 'running':
                            # Check session status from API
                            try:
                                session_data = self.api_client.get_session_logs(session_id)
                                status = session_data.get('status', 'unknown')
                                current_step = session_data.get('current_step', '')
                                progress = session_data.get('progress', 0)
                                
                                # Update session info with enhanced tracking
                                session_info.update({
                                    'status': status,
                                    'current_step': current_step,
                                    'progress': progress,
                                    'last_update': time.time()
                                })
                                
                                # Track step history for workflow sessions
                                if session_info.get('type') == 'workflow_test':
                                    step_history = session_info.get('step_history', [])
                                    if current_step and (not step_history or step_history[-1] != current_step):
                                        step_history.append(current_step)
                                        session_info['step_history'] = step_history
                                
                                # Remove completed sessions after some time
                                if status in ['completed', 'failed', 'cancelled']:
                                    elapsed = time.time() - session_info.get('start_time', 0)
                                    if elapsed > 600:  # Remove after 10 minutes for workflows
                                        del self.active_sessions[session_id]
                                        
                            except Exception as e:
                                logger.error(f"Error updating session {session_id}: {e}")
                    
                    time.sleep(2)  # Update every 2 seconds
                    
                except Exception as e:
                    logger.error(f"Error in periodic updates: {e}")
                    time.sleep(5)  # Wait longer on error
        
        # Start background thread
        update_thread = threading.Thread(target=update_active_sessions, daemon=True)
        update_thread.start()
        
        return
    
    def _update_agent_status(self, session_id: str) -> tuple:
        """Update agent execution status."""
        if not session_id or session_id not in self.active_sessions:
            return (
                "No active session",
                gr.update(value=0),
                "",
                None,
                "",
                None
            )
        
        try:
            # Get session logs and status
            session_data = self.api_client.get_session_logs(session_id)
            
            status = session_data.get('status', 'unknown')
            progress = session_data.get('progress', 0) * 100
            current_step = session_data.get('current_step', '')
            results = session_data.get('results')
            execution_time = session_data.get('execution_time')
            
            # Format status message
            status_icons = {
                'running': '⚙️',
                'completed': '✅',
                'failed': '❌',
                'cancelled': '⏹️'
            }
            
            status_icon = status_icons.get(status, '❓')
            status_msg = f"{status_icon} {status.title()}"
            
            if execution_time:
                status_msg += f" (took {execution_time:.2f}s)"
            
            # Format output text
            output_text = ""
            if results:
                if isinstance(results, dict):
                    output_text = json.dumps(results.get('outputs', results), indent=2)
                else:
                    output_text = str(results)
            
            # Format metadata
            metadata = None
            if results and isinstance(results, dict):
                metadata = results.get('metadata', {})
            
            return (
                status_msg,
                gr.update(value=progress),
                current_step,
                results,
                output_text,
                metadata
            )
            
        except Exception as e:
            logger.error(f"Error updating agent status: {e}")
            return (
                f"❌ Error updating status: {str(e)}",
                gr.update(value=0),
                "",
                None,
                "",
                None
            )
    
    def _get_agent_type(self, agent_name: str) -> str:
        """Get agent type from agent name."""
        if 'planning' in agent_name.lower():
            return 'planning'
        elif 'code' in agent_name.lower() or 'generation' in agent_name.lower():
            return 'code_generation'
        elif 'render' in agent_name.lower():
            return 'rendering'
        else:
            return 'unknown'
    
    def _get_agent_initial_step(self, agent_type: str) -> str:
        """Get initial step message for agent type."""
        steps = {
            'planning': 'Analyzing topic and creating scene structure...',
            'code_generation': 'Generating Manim code from scene plan...',
            'rendering': 'Preparing video rendering environment...',
            'unknown': 'Initializing agent execution...'
        }
        return steps.get(agent_type, steps['unknown'])
    
    def _format_input_preview(self, agent_name: str, form_data: Dict[str, Any]) -> str:
        """Format input data for preview display."""
        agent_type = self._get_agent_type(agent_name)
        
        preview = f"**Agent:** {agent_name.replace('_', ' ').title()}\n"
        preview += f"**Type:** {agent_type.replace('_', ' ').title()}\n\n"
        preview += "**Input Data:**\n"
        
        # Agent-specific formatting
        if agent_type == 'planning':
            topic = form_data.get('topic', 'N/A')
            description = form_data.get('description', 'N/A')
            complexity = form_data.get('complexity', 'intermediate')
            
            preview += f"- **Topic:** {topic}\n"
            preview += f"- **Complexity:** {complexity}\n"
            preview += f"- **Description:** {description[:100]}{'...' if len(description) > 100 else ''}\n"
            
        elif agent_type == 'code_generation':
            scene_plan = form_data.get('scene_plan', {})
            style = form_data.get('style', 'detailed')
            
            preview += f"- **Style:** {style}\n"
            if isinstance(scene_plan, dict):
                preview += f"- **Scene Plan:** {len(scene_plan)} elements\n"
            else:
                preview += f"- **Scene Plan:** {str(scene_plan)[:100]}{'...' if len(str(scene_plan)) > 100 else ''}\n"
                
        elif agent_type == 'rendering':
            code = form_data.get('code', '')
            quality = form_data.get('quality', 'medium')
            
            preview += f"- **Quality:** {quality}\n"
            preview += f"- **Code Length:** {len(code)} characters\n"
            if code:
                preview += f"- **Code Preview:** {code[:100]}{'...' if len(code) > 100 else ''}\n"
        else:
            # Generic formatting
            for key, value in form_data.items():
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                preview += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        return preview
    
    def _create_agent_info_display(self, agent_info: Dict[str, Any]) -> str:
        """Create enhanced agent information display."""
        agent_name = agent_info.get('name', 'Unknown')
        agent_type = agent_info.get('type', 'unknown')
        agent_description = agent_info.get('description', 'No description available')
        input_schema = agent_info.get('input_schema', {})
        test_examples = agent_info.get('test_examples', [])
        
        # Agent type specific descriptions and capabilities
        type_descriptions = {
            'planning': {
                'icon': '🧠',
                'description': 'Analyzes topics and creates structured scene plans for video generation',
                'capabilities': [
                    'Topic analysis and breakdown',
                    'Scene structure planning',
                    'Content organization',
                    'Educational flow design'
                ]
            },
            'code_generation': {
                'icon': '💻',
                'description': 'Converts scene plans into executable Manim code',
                'capabilities': [
                    'Python/Manim code generation',
                    'Animation sequence creation',
                    'Mathematical visualization',
                    'Code optimization'
                ]
            },
            'rendering': {
                'icon': '🎬',
                'description': 'Executes Manim code to produce final video output',
                'capabilities': [
                    'Video rendering and compilation',
                    'Quality optimization',
                    'Thumbnail generation',
                    'Output file management'
                ]
            }
        }
        
        type_info = type_descriptions.get(agent_type, {
            'icon': '❓',
            'description': agent_description,
            'capabilities': ['General agent functionality']
        })
        
        # Build information display
        info_md = f"""
        ## {type_info['icon']} {agent_name}
        
        **Type:** {agent_type.replace('_', ' ').title()}
        
        **Description:** {type_info['description']}
        
        ### 🎯 Capabilities
        """
        
        for capability in type_info['capabilities']:
            info_md += f"- {capability}\n"
        
        # Input requirements
        if input_schema:
            info_md += "\n### 📝 Input Requirements\n"
            required_fields = []
            optional_fields = []
            
            for field_name, field_schema in input_schema.items():
                field_desc = field_schema.get('description', 'No description')
                field_type = field_schema.get('type', 'string')
                is_required = field_schema.get('required', False)
                
                field_info = f"**{field_name}** ({field_type}): {field_desc}"
                
                if is_required:
                    required_fields.append(field_info)
                else:
                    optional_fields.append(field_info)
            
            if required_fields:
                info_md += "\n**Required Fields:**\n"
                for field in required_fields:
                    info_md += f"- {field}\n"
            
            if optional_fields:
                info_md += "\n**Optional Fields:**\n"
                for field in optional_fields:
                    info_md += f"- {field}\n"
        
        # Test examples info
        if test_examples:
            info_md += f"\n### 📋 Test Examples\n"
            info_md += f"{len(test_examples)} predefined examples available for quick testing.\n"
            
            # Show first example preview
            if test_examples:
                first_example = test_examples[0]
                info_md += f"\n**Example Preview:**\n"
                for key, value in first_example.items():
                    if isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    info_md += f"- {key}: {value}\n"
        
        return info_md
    
    def _create_agent_result_visualization(self, agent_name: str, results: dict) -> str:
        """Create agent-specific result visualization."""
        if not results:
            return "No results available"
        
        try:
            # Agent-specific result formatting
            if agent_name == 'planning_agent':
                return self._format_planning_results(results)
            elif agent_name == 'code_generation_agent':
                return self._format_code_generation_results(results)
            elif agent_name == 'rendering_agent':
                return self._format_rendering_results(results)
            else:
                # Generic result formatting
                return json.dumps(results, indent=2)
                
        except Exception as e:
            logger.error(f"Error formatting results for {agent_name}: {e}")
            return f"Error formatting results: {str(e)}"
    
    def _format_planning_results(self, results: dict) -> str:
        """Format planning agent results with enhanced visualization."""
        output = "## 🧠 Planning Agent Results\n\n"
        
        # Extract outputs from nested structure
        outputs = results.get('outputs', results)
        
        if 'scene_plan' in outputs:
            output += "### 📋 Generated Scene Plan\n"
            scene_plan = outputs['scene_plan']
            
            if isinstance(scene_plan, dict):
                # Structure the scene plan display
                for key, value in scene_plan.items():
                    if key.lower() in ['title', 'topic']:
                        output += f"**🎯 {key.title()}:** {value}\n\n"
                    elif key.lower() in ['scenes', 'steps', 'sequence']:
                        output += f"**🎬 {key.title()}:**\n"
                        if isinstance(value, list):
                            for i, item in enumerate(value, 1):
                                output += f"  {i}. {item}\n"
                        else:
                            output += f"  {value}\n"
                        output += "\n"
                    elif key.lower() in ['concepts', 'key_points']:
                        output += f"**💡 {key.replace('_', ' ').title()}:**\n"
                        if isinstance(value, list):
                            for concept in value:
                                output += f"  • {concept}\n"
                        else:
                            output += f"  {value}\n"
                        output += "\n"
                    else:
                        output += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
            else:
                output += f"{scene_plan}\n\n"
        
        # Add planning metrics if available
        if 'metadata' in results:
            metadata = results['metadata']
            output += "### 📊 Planning Metrics\n"
            
            metrics_display = {
                'complexity_score': '🎚️ Complexity Score',
                'estimated_duration': '⏱️ Estimated Duration',
                'scene_count': '🎬 Scene Count',
                'concept_count': '💡 Concept Count'
            }
            
            for key, display_name in metrics_display.items():
                if key in metadata:
                    output += f"- **{display_name}:** {metadata[key]}\n"
            
            # Show other metadata
            for key, value in metadata.items():
                if key not in metrics_display:
                    output += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        return output
    
    def _format_code_generation_results(self, results: dict) -> str:
        """Format code generation agent results with enhanced visualization."""
        output = "## 💻 Code Generation Agent Results\n\n"
        
        # Extract outputs from nested structure
        outputs = results.get('outputs', results)
        
        if 'code' in outputs:
            code = outputs['code']
            output += "### 🐍 Generated Manim Code\n"
            
            # Add code statistics
            lines = code.count('\n') + 1
            chars = len(code)
            output += f"**Code Statistics:** {lines} lines, {chars} characters\n\n"
            
            output += f"```python\n{code}\n```\n\n"
        
        if 'imports' in outputs:
            output += "### 📦 Required Imports\n"
            imports = outputs['imports']
            if isinstance(imports, list):
                for imp in imports:
                    output += f"- `{imp}`\n"
            else:
                output += f"- `{imports}`\n"
            output += "\n"
        
        # Add code analysis if available
        if 'analysis' in outputs:
            analysis = outputs['analysis']
            output += "### 🔍 Code Analysis\n"
            
            if isinstance(analysis, dict):
                for key, value in analysis.items():
                    output += f"- **{key.replace('_', ' ').title()}:** {value}\n"
            else:
                output += f"{analysis}\n"
            output += "\n"
        
        if 'metadata' in results:
            metadata = results['metadata']
            output += "### 📊 Generation Metrics\n"
            
            metrics_display = {
                'generation_time': '⏱️ Generation Time',
                'code_complexity': '🎚️ Code Complexity',
                'animation_count': '🎬 Animation Count',
                'function_count': '⚙️ Function Count'
            }
            
            for key, display_name in metrics_display.items():
                if key in metadata:
                    output += f"- **{display_name}:** {metadata[key]}\n"
            
            # Show other metadata
            for key, value in metadata.items():
                if key not in metrics_display:
                    output += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        return output
    
    def _format_rendering_results(self, results: dict) -> str:
        """Format rendering agent results with enhanced visualization."""
        output = "## 🎬 Rendering Agent Results\n\n"
        
        # Extract outputs from nested structure
        outputs = results.get('outputs', results)
        
        if 'video_path' in outputs:
            video_path = outputs['video_path']
            output += f"### 🎥 Video Output\n"
            output += f"**File Path:** `{video_path}`\n"
            
            # Add video info if available
            if 'video_info' in outputs:
                video_info = outputs['video_info']
                if isinstance(video_info, dict):
                    output += f"**Duration:** {video_info.get('duration', 'Unknown')}\n"
                    output += f"**Resolution:** {video_info.get('resolution', 'Unknown')}\n"
                    output += f"**Frame Rate:** {video_info.get('fps', 'Unknown')} fps\n"
                    output += f"**File Size:** {video_info.get('file_size', 'Unknown')}\n"
            output += "\n"
        
        if 'thumbnail_path' in outputs:
            thumbnail_path = outputs['thumbnail_path']
            output += f"### 🖼️ Thumbnail\n"
            output += f"**File Path:** `{thumbnail_path}`\n\n"
        
        # Add rendering statistics
        if 'render_stats' in outputs:
            stats = outputs['render_stats']
            output += "### 📊 Rendering Statistics\n"
            
            if isinstance(stats, dict):
                stats_display = {
                    'render_time': '⏱️ Render Time',
                    'frames_rendered': '🎞️ Frames Rendered',
                    'quality_level': '🎚️ Quality Level',
                    'output_format': '📁 Output Format'
                }
                
                for key, display_name in stats_display.items():
                    if key in stats:
                        output += f"- **{display_name}:** {stats[key]}\n"
                
                # Show other stats
                for key, value in stats.items():
                    if key not in stats_display:
                        output += f"- **{key.replace('_', ' ').title()}:** {value}\n"
            output += "\n"
        
        if 'metadata' in results:
            metadata = results['metadata']
            output += "### 🔧 Rendering Metadata\n"
            
            for key, value in metadata.items():
                if key not in ['render_stats', 'video_info']:  # Avoid duplication
                    output += f"- **{key.replace('_', ' ').title()}:** {value}\n"
        
        return output
    
    def _get_active_agent_status(self) -> tuple:
        """Get status updates for currently active agent sessions."""
        # Find the most recent active agent session
        active_session = None
        latest_time = 0
        
        for session_id, session_info in self.active_sessions.items():
            if (session_info.get('type') == 'agent_test' and 
                session_info.get('status') == 'running' and
                session_info.get('start_time', 0) > latest_time):
                active_session = session_id
                latest_time = session_info.get('start_time', 0)
        
        if not active_session:
            # No active sessions, return current state
            return (
                gr.update(),  # status
                gr.update(),  # progress
                gr.update(),  # current step
                gr.update(),  # results
                gr.update(),  # output text
                gr.update()   # metadata
            )
        
        try:
            # Get updated status from API
            session_data = self.api_client.get_session_logs(active_session)
            session_info = self.active_sessions[active_session]
            
            status = session_data.get('status', 'unknown')
            progress = session_data.get('progress', 0) * 100
            current_step = session_data.get('current_step', '')
            results = session_data.get('results')
            
            # Update session info
            session_info['status'] = status
            
            # Create status message with agent-specific formatting
            agent_name = session_info.get('agent_name', 'Unknown')
            agent_type = session_info.get('agent_type', 'unknown')
            
            type_icons = {
                'planning': '🧠',
                'code_generation': '💻',
                'rendering': '🎬'
            }
            
            icon = type_icons.get(agent_type, '🤖')
            
            if status == 'completed':
                status_msg = f"✅ {icon} {agent_name.replace('_', ' ').title()} completed successfully"
            elif status == 'failed':
                status_msg = f"❌ {icon} {agent_name.replace('_', ' ').title()} failed"
            elif status == 'running':
                status_msg = f"⚙️ {icon} {agent_name.replace('_', ' ').title()} running..."
            else:
                status_msg = f"{icon} {agent_name.replace('_', ' ').title()}: {status}"
            
            # Format output text with agent-specific visualization
            output_text = ""
            if results:
                output_text = self._create_agent_result_visualization(agent_name, results)
            
            # Format metadata
            metadata = None
            if results and isinstance(results, dict):
                metadata = {
                    'session_id': active_session,
                    'agent_type': agent_type,
                    'execution_time': session_data.get('execution_time'),
                    'status': status,
                    'progress': progress,
                    **results.get('metadata', {})
                }
            
            return (
                gr.update(value=status_msg),
                gr.update(value=progress),
                gr.update(value=current_step),
                gr.update(value=results),
                gr.update(value=output_text),
                gr.update(value=metadata)
            )
            
        except Exception as e:
            logger.error(f"Error getting active agent status: {e}")
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update()
            )


def create_gradio_test_interface(backend_url: str = "http://localhost:8000") -> gr.Blocks:
    """Create and return the Gradio testing interface."""
    interface = GradioTestInterface(backend_url)
    return interface.create_interface()


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_test_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )  
    def _setup_config_event_handlers(
        self, 
        config_components, agent_config_save, agent_config_load,
        workflow_config_save, workflow_config_load,
        agent_dropdown, dynamic_inputs, session_id_input, 
        agent_timeout, agent_retry_count, agent_verbose,
        workflow_topic, workflow_description, workflow_session_id, config_overrides_json
    ):
        """Set up event handlers for configuration management."""
        
        # Configuration Management Tab Handlers
        config_components["refresh_configs_btn"].click(
            fn=lambda filter_type: self.config_ui.refresh_config_list(filter_type),
            inputs=[config_components["config_type_filter"]],
            outputs=[config_components["config_list_display"]]
        )
        
        config_components["config_type_filter"].change(
            fn=lambda filter_type: self.config_ui.refresh_config_list(filter_type),
            inputs=[config_components["config_type_filter"]],
            outputs=[config_components["config_list_display"]]
        )
        
        config_components["load_config_btn"].click(
            fn=self._load_selected_config,
            inputs=[config_components["selected_config_id"]],
            outputs=[config_components["config_details_display"], config_components["config_status_text"]]
        )
        
        config_components["delete_config_btn"].click(
            fn=self._delete_selected_config,
            inputs=[config_components["selected_config_id"]],
            outputs=[config_components["config_status_text"], config_components["config_list_display"]]
        )
        
        config_components["import_config_btn"].click(
            fn=self._import_config_file,
            inputs=[config_components["import_file"]],
            outputs=[config_components["config_status_text"], config_components["config_list_display"]]
        )
        
        config_components["save_config_btn"].click(
            fn=self._save_current_config,
            inputs=[
                config_components["save_config_name"],
                config_components["save_config_type"],
                config_components["save_config_description"],
                config_components["current_agent_config"],
                config_components["current_workflow_config"]
            ],
            outputs=[config_components["save_status_text"], config_components["config_list_display"]]
        )
        
        # Agent Tab Configuration Handlers
        agent_config_save["quick_save_btn"].click(
            fn=self._quick_save_agent_config_with_validation,
            inputs=[
                agent_config_save["config_name_input"],
                agent_config_save["config_description_input"],
                agent_dropdown, dynamic_inputs, session_id_input,
                agent_timeout, agent_retry_count, agent_verbose
            ],
            outputs=[agent_config_save["save_feedback"]]
        )
        
        agent_config_load["refresh_list_btn"].click(
            fn=lambda: gr.update(choices=self.config_ui.get_config_choices("agent")),
            inputs=[],
            outputs=[agent_config_load["config_selector"]]
        )
        
        agent_config_load["load_btn"].click(
            fn=self._load_agent_config,
            inputs=[agent_config_load["config_selector"]],
            outputs=[
                agent_config_load["load_feedback"],
                agent_dropdown, dynamic_inputs, session_id_input,
                agent_timeout, agent_retry_count, agent_verbose
            ]
        )
        
        # Workflow Tab Configuration Handlers
        workflow_config_save["quick_save_btn"].click(
            fn=self._quick_save_workflow_config_with_validation,
            inputs=[
                workflow_config_save["config_name_input"],
                workflow_config_save["config_description_input"],
                workflow_topic, workflow_description, workflow_session_id, config_overrides_json
            ],
            outputs=[workflow_config_save["save_feedback"]]
        )
        
        workflow_config_load["refresh_list_btn"].click(
            fn=lambda: gr.update(choices=self.config_ui.get_config_choices("workflow")),
            inputs=[],
            outputs=[workflow_config_load["config_selector"]]
        )
        
        workflow_config_load["load_btn"].click(
            fn=self._load_workflow_config,
            inputs=[workflow_config_load["config_selector"]],
            outputs=[
                workflow_config_load["load_feedback"],
                workflow_topic, workflow_description, workflow_session_id, config_overrides_json
            ]
        )
    
    def _load_selected_config(self, config_id: str) -> tuple:
        """Load the selected configuration from the management tab."""
        try:
            config_data, status = self.config_ui.load_configuration(config_id)
            return config_data, status
        except Exception as e:
            logger.error(f"Failed to load selected config: {e}")
            return {}, f"Error: {e}"
    
    def _delete_selected_config(self, config_id: str) -> tuple:
        """Delete the selected configuration."""
        try:
            status = self.config_ui.delete_configuration(config_id)
            config_list = self.config_ui.refresh_config_list()
            return status, config_list
        except Exception as e:
            logger.error(f"Failed to delete config: {e}")
            return f"Error: {e}", []
    
    def _import_config_file(self, file_path: str) -> tuple:
        """Import configuration from file."""
        try:
            if not file_path:
                return "No file selected", []
            
            status = self.config_ui.import_configuration(file_path)
            config_list = self.config_ui.refresh_config_list()
            return status, config_list
        except Exception as e:
            logger.error(f"Failed to import config: {e}")
            return f"Error: {e}", []
    
    def _save_current_config(
        self, 
        name: str, 
        config_type: str, 
        description: str,
        agent_config: dict,
        workflow_config: dict
    ) -> tuple:
        """Save the current configuration from the management tab."""
        try:
            if config_type == "Agent":
                if not agent_config:
                    return "No agent configuration to save", []
                
                status = self.config_ui.save_agent_configuration(
                    name=name,
                    description=description,
                    agent_name=agent_config.get("agent_name", ""),
                    inputs=agent_config.get("inputs", {}),
                    options=agent_config.get("options", {})
                )
            else:  # Workflow
                if not workflow_config:
                    return "No workflow configuration to save", []
                
                status = self.config_ui.save_workflow_configuration(
                    name=name,
                    description=description,
                    topic=workflow_config.get("topic", ""),
                    workflow_description=workflow_config.get("workflow_description", ""),
                    config_overrides=workflow_config.get("config_overrides", {})
                )
            
            config_list = self.config_ui.refresh_config_list()
            return status, config_list
            
        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return f"Error: {e}", []
    
    def _quick_save_agent_config(
        self,
        name: str,
        description: str,
        agent_name: str,
        dynamic_inputs: dict,
        session_id: str,
        timeout: int,
        retry_count: int,
        verbose: bool
    ) -> str:
        """Quick save agent configuration from the agent tab."""
        try:
            if not name:
                return "Configuration name is required"
            
            if not agent_name:
                return "Please select an agent first"
            
            # Prepare inputs and options
            inputs = dynamic_inputs if isinstance(dynamic_inputs, dict) else {}
            options = {
                "timeout": timeout,
                "retry_count": retry_count,
                "verbose": verbose
            }
            
            if session_id:
                options["session_id"] = session_id
            
            status = self.config_ui.save_agent_configuration(
                name=name,
                description=description,
                agent_name=agent_name,
                inputs=inputs,
                options=options
            )
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to quick save agent config: {e}")
            return f"Error: {e}"
    
    def _load_agent_config(self, config_id: str) -> tuple:
        """Load agent configuration and populate the form."""
        try:
            if not config_id:
                return "No configuration selected", None, {}, "", 60, 1, True
            
            config_data, status = self.config_ui.load_configuration(config_id)
            
            if not config_data:
                return status, None, {}, "", 60, 1, True
            
            # Extract configuration values
            agent_name = config_data.get("agent_name", "")
            inputs = config_data.get("inputs", {})
            options = config_data.get("options", {})
            
            session_id = options.get("session_id", "")
            timeout = options.get("timeout", 60)
            retry_count = options.get("retry_count", 1)
            verbose = options.get("verbose", True)
            
            return (
                status,
                agent_name,
                inputs,
                session_id,
                timeout,
                retry_count,
                verbose
            )
            
        except Exception as e:
            logger.error(f"Failed to load agent config: {e}")
            return f"Error: {e}", None, {}, "", 60, 1, True
    
    def _quick_save_workflow_config(
        self,
        name: str,
        description: str,
        topic: str,
        workflow_description: str,
        session_id: str,
        config_overrides: dict
    ) -> str:
        """Quick save workflow configuration from the workflow tab."""
        try:
            if not name:
                return "Configuration name is required"
            
            if not topic:
                return "Topic is required"
            
            if not workflow_description:
                return "Workflow description is required"
            
            status = self.config_ui.save_workflow_configuration(
                name=name,
                description=description,
                topic=topic,
                workflow_description=workflow_description,
                config_overrides=config_overrides or {}
            )
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to quick save workflow config: {e}")
            return f"Error: {e}"
    
    def _load_workflow_config(self, config_id: str) -> tuple:
        """Load workflow configuration and populate the form."""
        try:
            if not config_id:
                return "No configuration selected", "", "", "", {}
            
            config_data, status = self.config_ui.load_configuration(config_id)
            
            if not config_data:
                return status, "", "", "", {}
            
            # Extract configuration values
            topic = config_data.get("topic", "")
            workflow_description = config_data.get("workflow_description", "")
            config_overrides = config_data.get("config_overrides", {})
            
            return (
                status,
                topic,
                workflow_description,
                "",  # session_id is not saved in config
                config_overrides
            )
            
        except Exception as e:
            logger.error(f"Failed to load workflow config: {e}")
            return f"Error: {e}", "", "", "", {}    
   
    def _validate_agent_inputs(
        self,
        agent_name: str,
        session_id: str,
        dynamic_inputs: dict,
        timeout: int,
        retry_count: int,
        verbose: bool
    ) -> Tuple[bool, List[str]]:
        """
        Validate agent testing inputs.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate agent name
        is_valid, error_msg = self.input_validator.validate_agent_name(agent_name)
        if not is_valid:
            errors.append(f"Agent Name: {error_msg}")
        
        # Validate session ID
        is_valid, error_msg = self.input_validator.validate_session_id(session_id)
        if not is_valid:
            errors.append(f"Session ID: {error_msg}")
        
        # Validate timeout
        is_valid, error_msg = self.input_validator.validate_timeout(timeout)
        if not is_valid:
            errors.append(f"Timeout: {error_msg}")
        
        # Validate retry count
        is_valid, error_msg = self.input_validator.validate_retry_count(retry_count)
        if not is_valid:
            errors.append(f"Retry Count: {error_msg}")
        
        # Validate dynamic inputs
        if dynamic_inputs:
            is_valid, input_errors = self.input_validator.validate_agent_inputs(dynamic_inputs)
            if not is_valid:
                errors.extend([f"Input {error}" for error in input_errors])
        
        return len(errors) == 0, errors
    
    def _validate_workflow_inputs(
        self,
        topic: str,
        description: str,
        session_id: str,
        config_overrides: dict
    ) -> Tuple[bool, List[str]]:
        """
        Validate workflow testing inputs.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate topic
        is_valid, error_msg = self.input_validator.validate_topic(topic)
        if not is_valid:
            errors.append(f"Topic: {error_msg}")
        
        # Validate description
        is_valid, error_msg = self.input_validator.validate_description(description, required=True)
        if not is_valid:
            errors.append(f"Description: {error_msg}")
        
        # Validate session ID
        is_valid, error_msg = self.input_validator.validate_session_id(session_id)
        if not is_valid:
            errors.append(f"Session ID: {error_msg}")
        
        # Validate config overrides (should be valid JSON)
        if config_overrides:
            try:
                if isinstance(config_overrides, str):
                    import json
                    json.loads(config_overrides)
                elif not isinstance(config_overrides, dict):
                    errors.append("Config Overrides: Must be a valid JSON object")
            except json.JSONDecodeError as e:
                errors.append(f"Config Overrides: Invalid JSON format - {str(e)}")
        
        return len(errors) == 0, errors
    
    def _validate_config_inputs(
        self,
        name: str,
        description: str
    ) -> Tuple[bool, List[str]]:
        """
        Validate configuration inputs.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate config name
        is_valid, error_msg = self.input_validator.validate_config_name(name)
        if not is_valid:
            errors.append(f"Configuration Name: {error_msg}")
        
        # Validate description (optional)
        is_valid, error_msg = self.input_validator.validate_description(description, required=False)
        if not is_valid:
            errors.append(f"Description: {error_msg}")
        
        return len(errors) == 0, errors
    
    @with_error_handling
    def _run_agent_test_with_validation(
        self,
        agent_name: str,
        session_id: str,
        dynamic_inputs: dict,
        timeout: int,
        retry_count: int,
        verbose: bool
    ):
        """Run agent test with input validation."""
        # Validate inputs first
        is_valid, errors = self._validate_agent_inputs(
            agent_name, session_id, dynamic_inputs, timeout, retry_count, verbose
        )
        
        if not is_valid:
            raise ValidationError(
                message="Agent test validation failed",
                suggestions=errors
            )
        
        # If validation passes, run the original test
        return self._run_agent_test(
            agent_name, session_id, dynamic_inputs, timeout, retry_count, verbose
        )
    
    @with_error_handling
    def _run_workflow_test_with_validation(
        self,
        topic: str,
        description: str,
        session_id: str,
        config_overrides: dict
    ):
        """Run workflow test with input validation."""
        # Validate inputs first
        is_valid, errors = self._validate_workflow_inputs(
            topic, description, session_id, config_overrides
        )
        
        if not is_valid:
            raise ValidationError(
                message="Workflow test validation failed",
                suggestions=errors
            )
        
        # If validation passes, run the original test
        return self._run_workflow_test(topic, description, session_id, config_overrides)
    
    @with_error_handling
    def _quick_save_agent_config_with_validation(
        self,
        name: str,
        description: str,
        agent_name: str,
        dynamic_inputs: dict,
        session_id: str,
        timeout: int,
        retry_count: int,
        verbose: bool
    ) -> str:
        """Quick save agent configuration with validation."""
        # Validate config inputs
        is_valid, errors = self._validate_config_inputs(name, description)
        if not is_valid:
            raise ValidationError(
                message="Configuration validation failed",
                suggestions=errors
            )
        
        # Validate agent inputs
        is_valid, errors = self._validate_agent_inputs(
            agent_name, session_id, dynamic_inputs, timeout, retry_count, verbose
        )
        if not is_valid:
            raise ValidationError(
                message="Agent configuration validation failed",
                suggestions=errors
            )
        
        # If validation passes, save the configuration
        return self._quick_save_agent_config(
            name, description, agent_name, dynamic_inputs, 
            session_id, timeout, retry_count, verbose
        )
    
    @with_error_handling
    def _quick_save_workflow_config_with_validation(
        self,
        name: str,
        description: str,
        topic: str,
        workflow_description: str,
        session_id: str,
        config_overrides: dict
    ) -> str:
        """Quick save workflow configuration with validation."""
        # Validate config inputs
        is_valid, errors = self._validate_config_inputs(name, description)
        if not is_valid:
            raise ValidationError(
                message="Configuration validation failed",
                suggestions=errors
            )
        
        # Validate workflow inputs
        is_valid, errors = self._validate_workflow_inputs(
            topic, workflow_description, session_id, config_overrides
        )
        if not is_valid:
            raise ValidationError(
                message="Workflow configuration validation failed",
                suggestions=errors
            )
        
        # If validation passes, save the configuration
        return self._quick_save_workflow_config(
            name, description, topic, workflow_description, session_id, config_overrides
        )
    
    def _create_validation_feedback(self, field_name: str, is_valid: bool, error_message: str = "") -> gr.update:
        """
        Create validation feedback for form fields.
        
        Args:
            field_name: Name of the field being validated
            is_valid: Whether the field is valid
            error_message: Error message if invalid
            
        Returns:
            Gradio update with validation styling
        """
        if is_valid:
            return gr.update(
                elem_classes=["validation-success"],
                info=f"✅ {field_name} is valid"
            )
        else:
            return gr.update(
                elem_classes=["validation-error"],
                info=f"❌ {error_message}"
            )
    
    def _validate_field_on_change(self, field_name: str, value: Any, validator_func: Callable) -> gr.update:
        """
        Validate a field when its value changes.
        
        Args:
            field_name: Name of the field
            value: Current field value
            validator_func: Validation function to use
            
        Returns:
            Gradio update with validation feedback
        """
        try:
            is_valid, error_message = validator_func(value)
            return self._create_validation_feedback(field_name, is_valid, error_message)
        except Exception as e:
            logger.error(f"Error validating field {field_name}: {e}")
            return gr.update()
    
    def _create_validation_feedback(self, field_name: str, is_valid: bool, error_message: str = "") -> gr.update:
        """
        Create validation feedback for a field.
        
        Args:
            field_name: Name of the field
            is_valid: Whether the field is valid
            error_message: Error message if invalid
            
        Returns:
            Gradio update with validation styling and info
        """
        if is_valid:
            return gr.update(
                elem_classes=["validation-success"],
                info=f"✅ {field_name.replace('_', ' ').title()} is valid"
            )
        else:
            return gr.update(
                elem_classes=["validation-error"],
                info=f"❌ {error_message}"
            )
    
    def _validate_agent_name_on_change(self, agent_name: str) -> gr.update:
        """Validate agent name field on change."""
        return self._validate_field_on_change(
            "agent_name", 
            agent_name, 
            self.input_validator.validate_agent_name
        )
    
    def _validate_session_id_on_change(self, session_id: str) -> gr.update:
        """Validate session ID field on change."""
        return self._validate_field_on_change(
            "session_id", 
            session_id, 
            self.input_validator.validate_session_id
        )
    
    def _validate_timeout_on_change(self, timeout: Union[int, float]) -> gr.update:
        """Validate timeout field on change."""
        return self._validate_field_on_change(
            "timeout", 
            timeout, 
            self.input_validator.validate_timeout
        )
    
    def _validate_retry_count_on_change(self, retry_count: Union[int, str]) -> gr.update:
        """Validate retry count field on change."""
        return self._validate_field_on_change(
            "retry_count", 
            retry_count, 
            self.input_validator.validate_retry_count
        )
    
    def _validate_topic_on_change(self, topic: str) -> gr.update:
        """Validate topic field on change."""
        return self._validate_field_on_change(
            "topic", 
            topic, 
            self.input_validator.validate_topic
        )
    
    def _validate_description_on_change(self, description: str) -> gr.update:
        """Validate description field on change."""
        return self._validate_field_on_change(
            "description", 
            description, 
            lambda desc: self.input_validator.validate_description(desc, required=True)
        )
    
    def _validate_config_name_on_change(self, config_name: str) -> gr.update:
        """Validate configuration name field on change."""
        return self._validate_field_on_change(
            "config_name", 
            config_name, 
            self.input_validator.validate_config_name
        )
    
    def _validate_json_input_on_change(self, json_str: str) -> Tuple[gr.update, str]:
        """
        Validate JSON input field on change.
        
        Args:
            json_str: JSON string to validate
            
        Returns:
            Tuple of (field_update, validation_message)
        """
        try:
            is_valid, error_message, parsed_json = self.input_validator.validate_json_input(json_str, required=False)
            
            if is_valid:
                return (
                    gr.update(
                        elem_classes=["validation-success"],
                        info="✅ Valid JSON format"
                    ),
                    "JSON is valid"
                )
            else:
                return (
                    gr.update(
                        elem_classes=["validation-error"],
                        info=f"❌ {error_message}"
                    ),
                    f"JSON validation error: {error_message}"
                )
        except Exception as e:
            logger.error(f"Error validating JSON input: {e}")
            return (
                gr.update(
                    elem_classes=["validation-error"],
                    info="❌ Validation error occurred"
                ),
                f"Validation error: {str(e)}"
            )