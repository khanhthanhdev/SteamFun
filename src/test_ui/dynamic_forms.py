"""
Dynamic Form Generation for Agent Testing

This module provides functionality to dynamically generate Gradio input forms
based on agent input schemas. It supports various input types and validation.
"""

import gradio as gr
import json
from typing import Dict, Any, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DynamicFormGenerator:
    """Generator for dynamic Gradio forms based on agent schemas."""
    
    @staticmethod
    def create_input_component(
        field_name: str, 
        field_schema: Dict[str, Any]
    ) -> Tuple[gr.Component, str]:
        """
        Create a Gradio input component based on field schema.
        
        Args:
            field_name: Name of the input field
            field_schema: Schema definition for the field
            
        Returns:
            Tuple of (Gradio component, component type)
        """
        field_type = field_schema.get('type', 'string')
        required = field_schema.get('required', False)
        description = field_schema.get('description', '')
        default_value = field_schema.get('default')
        enum_values = field_schema.get('enum')
        
        # Create label with required indicator
        label = field_name.replace('_', ' ').title()
        if required:
            label += " *"
        
        # Create component based on type
        if enum_values:
            # Dropdown for enum values
            component = gr.Dropdown(
                label=label,
                choices=enum_values,
                value=default_value or (enum_values[0] if enum_values else None),
                info=description,
                interactive=True
            )
            return component, 'dropdown'
            
        elif field_type == 'boolean':
            # Checkbox for boolean
            component = gr.Checkbox(
                label=label,
                value=default_value or False,
                info=description,
                interactive=True
            )
            return component, 'checkbox'
            
        elif field_type == 'number' or field_type == 'integer':
            # Number input
            minimum = field_schema.get('minimum', 0)
            maximum = field_schema.get('maximum', 100)
            step = 1 if field_type == 'integer' else 0.1
            
            component = gr.Slider(
                label=label,
                minimum=minimum,
                maximum=maximum,
                value=default_value or minimum,
                step=step,
                info=description,
                interactive=True
            )
            return component, 'slider'
            
        elif field_type == 'array':
            # Text area for array input (JSON format)
            placeholder = "Enter JSON array, e.g., [\"item1\", \"item2\"]"
            component = gr.Textbox(
                label=label,
                placeholder=placeholder,
                lines=3,
                info=f"{description} (JSON format)",
                interactive=True
            )
            return component, 'json_array'
            
        elif field_type == 'object':
            # Text area for object input (JSON format)
            placeholder = "Enter JSON object, e.g., {\"key\": \"value\"}"
            component = gr.Textbox(
                label=label,
                placeholder=placeholder,
                lines=5,
                info=f"{description} (JSON format)",
                interactive=True
            )
            return component, 'json_object'
            
        else:
            # Default to text input
            lines = 1
            if 'description' in field_name.lower() or len(description) > 100:
                lines = 3
            
            component = gr.Textbox(
                label=label,
                placeholder=f"Enter {field_name.replace('_', ' ')}...",
                lines=lines,
                info=description,
                interactive=True,
                value=default_value or ""
            )
            return component, 'textbox'
    
    @staticmethod
    def create_agent_form(agent_info: Dict[str, Any]) -> Tuple[List[gr.Component], Dict[str, str]]:
        """
        Create a complete form for an agent based on its input schema.
        
        Args:
            agent_info: Agent information including input schema
            
        Returns:
            Tuple of (list of components, component type mapping)
        """
        components = []
        component_types = {}
        
        input_schema = agent_info.get('input_schema', {})
        agent_type = agent_info.get('type', 'unknown')
        
        if not input_schema:
            # No schema available, create a generic input
            component = gr.Textbox(
                label="Agent Input (JSON)",
                placeholder="Enter input data as JSON object",
                lines=5,
                info="No schema available. Enter input as JSON object."
            )
            components.append(component)
            component_types['generic_input'] = 'json_object'
            return components, component_types
        
        # Sort fields to put required fields first
        fields = list(input_schema.items())
        fields.sort(key=lambda x: not x[1].get('required', False))
        
        # Agent-specific form enhancements
        for field_name, field_schema in fields:
            try:
                # Create enhanced component with agent-specific customizations
                component, comp_type = DynamicFormGenerator.create_enhanced_input_component(
                    field_name, field_schema, agent_type
                )
                components.append(component)
                component_types[field_name] = comp_type
            except Exception as e:
                logger.error(f"Error creating component for field {field_name}: {e}")
                # Fallback to text input
                component = gr.Textbox(
                    label=field_name.replace('_', ' ').title(),
                    placeholder=f"Enter {field_name}...",
                    info=f"Error in schema: {str(e)}"
                )
                components.append(component)
                component_types[field_name] = 'textbox'
        
        return components, component_types
    
    @staticmethod
    def create_enhanced_input_component(
        field_name: str, 
        field_schema: Dict[str, Any],
        agent_type: str = 'unknown'
    ) -> Tuple[gr.Component, str]:
        """
        Create an enhanced Gradio input component with agent-specific customizations.
        
        Args:
            field_name: Name of the input field
            field_schema: Schema definition for the field
            agent_type: Type of agent (planning, code_generation, rendering)
            
        Returns:
            Tuple of (Gradio component, component type)
        """
        field_type = field_schema.get('type', 'string')
        required = field_schema.get('required', False)
        description = field_schema.get('description', '')
        default_value = field_schema.get('default')
        enum_values = field_schema.get('enum')
        
        # Create label with required indicator and agent-specific icons
        label = field_name.replace('_', ' ').title()
        if required:
            label += " *"
        
        # Add agent-specific field icons
        field_icons = {
            'planning': {
                'topic': '🎯',
                'description': '📝',
                'complexity': '🎚️',
                'duration': '⏱️'
            },
            'code_generation': {
                'scene_plan': '📋',
                'style': '🎨',
                'code': '💻',
                'imports': '📦'
            },
            'rendering': {
                'code': '🐍',
                'quality': '🎬',
                'resolution': '📺',
                'fps': '🎞️'
            }
        }
        
        agent_icons = field_icons.get(agent_type, {})
        if field_name in agent_icons:
            label = f"{agent_icons[field_name]} {label}"
        
        # Agent-specific placeholder and help text
        placeholder_text = DynamicFormGenerator._get_agent_specific_placeholder(
            field_name, agent_type, field_schema
        )
        
        # Create component based on type with enhancements
        if enum_values:
            # Dropdown for enum values
            component = gr.Dropdown(
                label=label,
                choices=enum_values,
                value=default_value or (enum_values[0] if enum_values else None),
                info=description,
                interactive=True
            )
            return component, 'dropdown'
            
        elif field_type == 'boolean':
            # Checkbox for boolean
            component = gr.Checkbox(
                label=label,
                value=default_value or False,
                info=description,
                interactive=True
            )
            return component, 'checkbox'
            
        elif field_type == 'number' or field_type == 'integer':
            # Enhanced number input with agent-specific ranges
            minimum, maximum, step = DynamicFormGenerator._get_agent_specific_number_range(
                field_name, agent_type, field_schema
            )
            
            component = gr.Slider(
                label=label,
                minimum=minimum,
                maximum=maximum,
                value=default_value or minimum,
                step=step,
                info=description,
                interactive=True
            )
            return component, 'slider'
            
        elif field_type == 'array':
            # Enhanced text area for array input
            component = gr.Textbox(
                label=label,
                placeholder=placeholder_text,
                lines=3,
                info=f"{description} (JSON format)",
                interactive=True
            )
            return component, 'json_array'
            
        elif field_type == 'object':
            # Enhanced text area for object input with agent-specific sizing
            lines = 5
            if agent_type == 'code_generation' and 'plan' in field_name.lower():
                lines = 8
            elif agent_type == 'rendering' and 'code' in field_name.lower():
                lines = 10
            
            component = gr.Textbox(
                label=label,
                placeholder=placeholder_text,
                lines=lines,
                info=f"{description} (JSON format)",
                interactive=True
            )
            return component, 'json_object'
            
        else:
            # Enhanced text input with agent-specific sizing
            lines = 1
            if 'description' in field_name.lower():
                lines = 4
            elif agent_type == 'planning' and field_name in ['topic']:
                lines = 1
            elif agent_type == 'code_generation' and 'code' in field_name.lower():
                lines = 15
            elif len(description) > 100:
                lines = 3
            
            component = gr.Textbox(
                label=label,
                placeholder=placeholder_text,
                lines=lines,
                info=description,
                interactive=True,
                value=default_value or ""
            )
            return component, 'textbox'
    
    @staticmethod
    def _get_agent_specific_placeholder(field_name: str, agent_type: str, field_schema: Dict[str, Any]) -> str:
        """Get agent-specific placeholder text for form fields."""
        placeholders = {
            'planning': {
                'topic': 'e.g., Fourier Transform, Linear Algebra, Calculus',
                'description': 'Provide a detailed explanation of what you want to teach or demonstrate...',
                'complexity': 'Select the appropriate complexity level for your audience',
                'duration': 'Estimated video duration in minutes'
            },
            'code_generation': {
                'scene_plan': 'Paste the scene plan JSON from the planning agent...',
                'style': 'Choose the visual style for your animation',
                'code': 'Enter existing Manim code to modify or extend...'
            },
            'rendering': {
                'code': 'Paste the complete Manim code to render...',
                'quality': 'Select rendering quality (affects processing time)',
                'resolution': 'Video resolution (e.g., 1920x1080)',
                'fps': 'Frames per second (typically 30 or 60)'
            }
        }
        
        agent_placeholders = placeholders.get(agent_type, {})
        return agent_placeholders.get(field_name, f"Enter {field_name.replace('_', ' ')}...")
    
    @staticmethod
    def _get_agent_specific_number_range(field_name: str, agent_type: str, field_schema: Dict[str, Any]) -> Tuple[float, float, float]:
        """Get agent-specific number ranges for numeric inputs."""
        # Default ranges
        minimum = field_schema.get('minimum', 0)
        maximum = field_schema.get('maximum', 100)
        step = 1 if field_schema.get('type') == 'integer' else 0.1
        
        # Agent-specific ranges
        ranges = {
            'planning': {
                'complexity': (1, 5, 1),
                'duration': (1, 30, 1),
                'scene_count': (1, 20, 1)
            },
            'code_generation': {
                'animation_speed': (0.1, 3.0, 0.1),
                'font_size': (12, 72, 2)
            },
            'rendering': {
                'quality': (1, 10, 1),
                'fps': (15, 60, 15),
                'resolution_width': (480, 1920, 240),
                'resolution_height': (360, 1080, 180)
            }
        }
        
        agent_ranges = ranges.get(agent_type, {})
        if field_name in agent_ranges:
            return agent_ranges[field_name]
        
        return minimum, maximum, step
    
    @staticmethod
    def extract_form_values(
        components: List[gr.Component], 
        component_types: Dict[str, str],
        values: List[Any]
    ) -> Dict[str, Any]:
        """
        Extract and validate values from form components.
        
        Args:
            components: List of Gradio components
            component_types: Mapping of field names to component types
            values: List of values from the components
            
        Returns:
            Dictionary of validated field values
        """
        result = {}
        field_names = list(component_types.keys())
        
        for i, (field_name, comp_type) in enumerate(component_types.items()):
            if i >= len(values):
                continue
                
            value = values[i]
            
            try:
                if comp_type == 'json_array':
                    if value and value.strip():
                        result[field_name] = json.loads(value)
                    else:
                        result[field_name] = []
                        
                elif comp_type == 'json_object':
                    if value and value.strip():
                        if field_name == 'generic_input':
                            # For generic input, return the parsed object directly
                            return json.loads(value)
                        else:
                            result[field_name] = json.loads(value)
                    else:
                        result[field_name] = {}
                        
                elif comp_type == 'checkbox':
                    result[field_name] = bool(value)
                    
                elif comp_type in ['slider', 'number']:
                    result[field_name] = value
                    
                else:
                    # Text input or dropdown
                    result[field_name] = value if value is not None else ""
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error for field {field_name}: {e}")
                raise ValueError(f"Invalid JSON in field '{field_name}': {str(e)}")
            except Exception as e:
                logger.error(f"Error processing field {field_name}: {e}")
                result[field_name] = value
        
        return result
    
    @staticmethod
    def create_example_form(agent_info: Dict[str, Any], example_index: int = 0) -> Dict[str, Any]:
        """
        Create form values from an agent's test example.
        
        Args:
            agent_info: Agent information including test examples
            example_index: Index of the example to use
            
        Returns:
            Dictionary of example values
        """
        test_examples = agent_info.get('test_examples', [])
        
        if not test_examples or example_index >= len(test_examples):
            return {}
        
        return test_examples[example_index]
    
    @staticmethod
    def validate_required_fields(
        input_schema: Dict[str, Any], 
        form_values: Dict[str, Any]
    ) -> List[str]:
        """
        Validate that all required fields are provided.
        
        Args:
            input_schema: Agent input schema
            form_values: Form values to validate
            
        Returns:
            List of missing required field names
        """
        missing_fields = []
        
        for field_name, field_schema in input_schema.items():
            if field_schema.get('required', False):
                value = form_values.get(field_name)
                if value is None or value == "" or (isinstance(value, (list, dict)) and not value):
                    missing_fields.append(field_name)
        
        return missing_fields


class AgentFormManager:
    """Manager for agent-specific form handling."""
    
    def __init__(self):
        self.current_agent = None
        self.current_components = []
        self.current_component_types = {}
    
    def update_form_for_agent(self, agent_info: Dict[str, Any]) -> Tuple[List[gr.Component], str]:
        """
        Update the form for a specific agent.
        
        Args:
            agent_info: Agent information
            
        Returns:
            Tuple of (components list, info message)
        """
        try:
            self.current_agent = agent_info
            self.current_components, self.current_component_types = (
                DynamicFormGenerator.create_agent_form(agent_info)
            )
            
            # Create info message about the form
            schema = agent_info.get('input_schema', {})
            required_fields = [
                name for name, field in schema.items() 
                if field.get('required', False)
            ]
            
            info_msg = f"Form created for {agent_info['name']}. "
            if required_fields:
                info_msg += f"Required fields: {', '.join(required_fields)}"
            else:
                info_msg += "No required fields."
            
            return self.current_components, info_msg
            
        except Exception as e:
            logger.error(f"Error updating form for agent: {e}")
            return [], f"Error creating form: {str(e)}"
    
    def get_form_values(self, *values) -> Dict[str, Any]:
        """
        Get validated form values.
        
        Args:
            *values: Values from form components
            
        Returns:
            Dictionary of validated values
        """
        if not self.current_component_types:
            return {}
        
        return DynamicFormGenerator.extract_form_values(
            self.current_components,
            self.current_component_types,
            list(values)
        )
    
    def validate_form(self, *values) -> Tuple[bool, str]:
        """
        Validate form values.
        
        Args:
            *values: Values from form components
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            form_values = self.get_form_values(*values)
            
            if not self.current_agent:
                return False, "No agent selected"
            
            input_schema = self.current_agent.get('input_schema', {})
            missing_fields = DynamicFormGenerator.validate_required_fields(
                input_schema, form_values
            )
            
            if missing_fields:
                return False, f"Missing required fields: {', '.join(missing_fields)}"
            
            return True, "Form validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"