"""
Configuration UI Components

This module provides Gradio UI components for managing test configurations,
including save/load/delete functionality and configuration management interface.
"""

import gradio as gr
import json
from typing import Dict, List, Optional, Any, Tuple
import logging

from src.test_ui.config_manager import TestConfigManager

logger = logging.getLogger(__name__)


class ConfigurationUI:
    """UI components for test configuration management."""
    
    def __init__(self, config_manager: TestConfigManager):
        """
        Initialize the configuration UI.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager
    
    def create_config_management_tab(self) -> Tuple[gr.TabItem, Dict[str, Any]]:
        """
        Create the configuration management tab.
        
        Returns:
            Tuple of (TabItem, components dictionary)
        """
        components = {}
        
        with gr.TabItem("⚙️ Configuration Management") as tab:
            with gr.Row():
                with gr.Column(scale=2):
                    # Configuration List
                    with gr.Group():
                        gr.Markdown("### 📋 Saved Configurations")
                        
                        with gr.Row():
                            config_type_filter = gr.Dropdown(
                                label="Filter by Type",
                                choices=["All", "Agent", "Workflow"],
                                value="All",
                                scale=2
                            )
                            
                            refresh_configs_btn = gr.Button(
                                "🔄 Refresh",
                                variant="secondary",
                                scale=1
                            )
                        
                        config_list_display = gr.Dataframe(
                            headers=["Name", "Type", "Description", "Created", "Actions"],
                            datatype=["str", "str", "str", "str", "str"],
                            interactive=False,
                            wrap=True,
                            max_rows=10
                        )
                        
                        selected_config_id = gr.Textbox(
                            label="Selected Configuration ID",
                            visible=False
                        )
                
                with gr.Column(scale=1):
                    # Configuration Actions
                    with gr.Group():
                        gr.Markdown("### 🛠️ Configuration Actions")
                        
                        load_config_btn = gr.Button(
                            "📂 Load Configuration",
                            variant="primary",
                            size="lg"
                        )
                        
                        delete_config_btn = gr.Button(
                            "🗑️ Delete Configuration",
                            variant="stop",
                            size="lg"
                        )
                        
                        export_config_btn = gr.Button(
                            "📤 Export Configuration",
                            variant="secondary"
                        )
                        
                        # Import section
                        gr.Markdown("#### Import Configuration")
                        
                        import_file = gr.File(
                            label="Select Configuration File",
                            file_types=[".json"],
                            file_count="single"
                        )
                        
                        import_config_btn = gr.Button(
                            "📥 Import Configuration",
                            variant="secondary"
                        )
                    
                    # Configuration Details
                    with gr.Group():
                        gr.Markdown("### 📄 Configuration Details")
                        
                        config_details_display = gr.JSON(
                            label="Configuration Data",
                            value=None
                        )
                        
                        config_status_text = gr.Textbox(
                            label="Status",
                            value="No configuration selected",
                            interactive=False
                        )
            
            # Save Configuration Section
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        gr.Markdown("### 💾 Save Current Configuration")
                        
                        with gr.Row():
                            save_config_name = gr.Textbox(
                                label="Configuration Name",
                                placeholder="Enter a name for this configuration",
                                scale=2
                            )
                            
                            save_config_type = gr.Dropdown(
                                label="Type",
                                choices=["Agent", "Workflow"],
                                value="Agent",
                                scale=1
                            )
                        
                        save_config_description = gr.Textbox(
                            label="Description (optional)",
                            placeholder="Describe this configuration...",
                            lines=2
                        )
                        
                        # Hidden fields to store current configuration data
                        current_agent_config = gr.State({})
                        current_workflow_config = gr.State({})
                        
                        save_config_btn = gr.Button(
                            "💾 Save Configuration",
                            variant="primary",
                            size="lg"
                        )
                        
                        save_status_text = gr.Textbox(
                            label="Save Status",
                            interactive=False,
                            visible=False
                        )
        
        # Store components for external access
        components.update({
            "tab": tab,
            "config_type_filter": config_type_filter,
            "refresh_configs_btn": refresh_configs_btn,
            "config_list_display": config_list_display,
            "selected_config_id": selected_config_id,
            "load_config_btn": load_config_btn,
            "delete_config_btn": delete_config_btn,
            "export_config_btn": export_config_btn,
            "import_file": import_file,
            "import_config_btn": import_config_btn,
            "config_details_display": config_details_display,
            "config_status_text": config_status_text,
            "save_config_name": save_config_name,
            "save_config_type": save_config_type,
            "save_config_description": save_config_description,
            "current_agent_config": current_agent_config,
            "current_workflow_config": current_workflow_config,
            "save_config_btn": save_config_btn,
            "save_status_text": save_status_text
        })
        
        return tab, components
    
    def create_config_save_section(self) -> Dict[str, Any]:
        """
        Create a compact configuration save section for other tabs.
        
        Returns:
            Dictionary of components
        """
        components = {}
        
        with gr.Accordion("💾 Save Configuration", open=False) as accordion:
            with gr.Row():
                config_name_input = gr.Textbox(
                    label="Configuration Name",
                    placeholder="Enter configuration name",
                    scale=2
                )
                
                quick_save_btn = gr.Button(
                    "💾 Save",
                    variant="secondary",
                    scale=1
                )
            
            config_description_input = gr.Textbox(
                label="Description (optional)",
                placeholder="Describe this configuration...",
                lines=2
            )
            
            save_feedback = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False
            )
        
        components.update({
            "accordion": accordion,
            "config_name_input": config_name_input,
            "quick_save_btn": quick_save_btn,
            "config_description_input": config_description_input,
            "save_feedback": save_feedback
        })
        
        return components
    
    def create_config_load_section(self) -> Dict[str, Any]:
        """
        Create a compact configuration load section for other tabs.
        
        Returns:
            Dictionary of components
        """
        components = {}
        
        with gr.Accordion("📂 Load Configuration", open=False) as accordion:
            config_selector = gr.Dropdown(
                label="Select Configuration",
                choices=[],
                value=None,
                info="Choose a saved configuration to load"
            )
            
            with gr.Row():
                load_btn = gr.Button(
                    "📂 Load",
                    variant="primary",
                    scale=1
                )
                
                refresh_list_btn = gr.Button(
                    "🔄 Refresh",
                    variant="secondary",
                    scale=1
                )
            
            load_feedback = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False
            )
        
        components.update({
            "accordion": accordion,
            "config_selector": config_selector,
            "load_btn": load_btn,
            "refresh_list_btn": refresh_list_btn,
            "load_feedback": load_feedback
        })
        
        return components
    
    def refresh_config_list(self, config_type_filter: str = "All") -> List[List[str]]:
        """
        Refresh the configuration list display.
        
        Args:
            config_type_filter: Filter by configuration type
            
        Returns:
            List of configuration rows for display
        """
        try:
            # Map filter to internal type
            type_filter = None
            if config_type_filter == "Agent":
                type_filter = "agent"
            elif config_type_filter == "Workflow":
                type_filter = "workflow"
            
            configs = self.config_manager.list_configs(type_filter)
            
            # Format for display
            rows = []
            for config in configs:
                created_date = config.get("created_at", "")[:10]  # Just the date part
                config_type = config["type"].title()
                
                # Create action buttons (represented as text for now)
                actions = f"ID: {config['id'][:8]}..."
                
                rows.append([
                    config["name"],
                    config_type,
                    config.get("description", "")[:50] + ("..." if len(config.get("description", "")) > 50 else ""),
                    created_date,
                    actions
                ])
            
            return rows
            
        except Exception as e:
            logger.error(f"Failed to refresh config list: {e}")
            return [["Error loading configurations", "", str(e), "", ""]]
    
    def get_config_choices(self, config_type: Optional[str] = None) -> List[str]:
        """
        Get configuration choices for dropdown.
        
        Args:
            config_type: Optional filter by type
            
        Returns:
            List of configuration choices
        """
        try:
            configs = self.config_manager.list_configs(config_type)
            choices = []
            
            for config in configs:
                choice_text = f"{config['name']} ({config['type'].title()})"
                if config.get("description"):
                    choice_text += f" - {config['description'][:30]}..."
                choices.append((choice_text, config["id"]))
            
            return choices
            
        except Exception as e:
            logger.error(f"Failed to get config choices: {e}")
            return [("Error loading configurations", "")]
    
    def load_configuration(self, config_id: str) -> Tuple[Dict[str, Any], str]:
        """
        Load a configuration by ID.
        
        Args:
            config_id: Configuration ID to load
            
        Returns:
            Tuple of (configuration data, status message)
        """
        try:
            if not config_id:
                return {}, "No configuration selected"
            
            config_data = self.config_manager.load_config(config_id)
            status = f"Loaded configuration: {config_data['name']}"
            
            return config_data, status
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return {}, f"Error loading configuration: {e}"
    
    def save_agent_configuration(
        self,
        name: str,
        description: str,
        agent_name: str,
        inputs: Dict[str, Any],
        options: Dict[str, Any] = None
    ) -> str:
        """
        Save an agent configuration.
        
        Args:
            name: Configuration name
            description: Configuration description
            agent_name: Agent name
            inputs: Agent inputs
            options: Execution options
            
        Returns:
            Status message
        """
        try:
            if not name:
                return "Configuration name is required"
            
            if not agent_name:
                return "Agent name is required"
            
            config_id = self.config_manager.save_agent_config(
                name=name,
                agent_name=agent_name,
                inputs=inputs,
                options=options,
                description=description
            )
            
            return f"Configuration saved successfully (ID: {config_id[:8]}...)"
            
        except Exception as e:
            logger.error(f"Failed to save agent configuration: {e}")
            return f"Error saving configuration: {e}"
    
    def save_workflow_configuration(
        self,
        name: str,
        description: str,
        topic: str,
        workflow_description: str,
        config_overrides: Dict[str, Any] = None
    ) -> str:
        """
        Save a workflow configuration.
        
        Args:
            name: Configuration name
            description: Configuration description
            topic: Workflow topic
            workflow_description: Workflow description
            config_overrides: Configuration overrides
            
        Returns:
            Status message
        """
        try:
            if not name:
                return "Configuration name is required"
            
            if not topic:
                return "Topic is required"
            
            if not workflow_description:
                return "Workflow description is required"
            
            config_id = self.config_manager.save_workflow_config(
                name=name,
                topic=topic,
                description=workflow_description,
                config_overrides=config_overrides,
                config_description=description
            )
            
            return f"Configuration saved successfully (ID: {config_id[:8]}...)"
            
        except Exception as e:
            logger.error(f"Failed to save workflow configuration: {e}")
            return f"Error saving configuration: {e}"
    
    def delete_configuration(self, config_id: str) -> str:
        """
        Delete a configuration.
        
        Args:
            config_id: Configuration ID to delete
            
        Returns:
            Status message
        """
        try:
            if not config_id:
                return "No configuration selected"
            
            self.config_manager.delete_config(config_id)
            return f"Configuration deleted successfully"
            
        except Exception as e:
            logger.error(f"Failed to delete configuration: {e}")
            return f"Error deleting configuration: {e}"
    
    def import_configuration(self, file_path: str) -> str:
        """
        Import a configuration from file.
        
        Args:
            file_path: Path to configuration file
            
        Returns:
            Status message
        """
        try:
            if not file_path:
                return "No file selected"
            
            config_id = self.config_manager.import_config(file_path)
            return f"Configuration imported successfully (ID: {config_id[:8]}...)"
            
        except Exception as e:
            logger.error(f"Failed to import configuration: {e}")
            return f"Error importing configuration: {e}"
    
    def export_configuration(self, config_id: str, export_path: str) -> str:
        """
        Export a configuration to file.
        
        Args:
            config_id: Configuration ID to export
            export_path: Export file path
            
        Returns:
            Status message
        """
        try:
            if not config_id:
                return "No configuration selected"
            
            if not export_path:
                export_path = f"config_{config_id[:8]}.json"
            
            self.config_manager.export_config(config_id, export_path)
            return f"Configuration exported to: {export_path}"
            
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
            return f"Error exporting configuration: {e}"