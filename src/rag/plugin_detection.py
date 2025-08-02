"""
Enhanced Plugin Detection and Query Generation

This module provides context-aware plugin detection and generates plugin-specific
queries with proper namespacing and relevance scoring.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, Tuple
from enum import Enum

class PluginRelevanceLevel(Enum):
    """Plugin relevance levels"""
    HIGH = 0.9
    MEDIUM = 0.6
    LOW = 0.3
    NONE = 0.0

@dataclass
class PluginInfo:
    """Information about a plugin"""
    name: str
    description: str
    namespace: str
    keywords: List[str]
    categories: List[str]
    version: Optional[str] = None
    dependencies: Optional[List[str]] = None

@dataclass
class PluginRelevanceScore:
    """Plugin relevance scoring information"""
    plugin_name: str
    relevance_level: PluginRelevanceLevel
    confidence_score: float
    matching_keywords: List[str]
    context_factors: Dict[str, float]
    reasoning: str

class ContextAwarePluginDetector:
    """
    Enhanced plugin detector that analyzes context to determine plugin relevance
    with improved accuracy and context awareness.
    """
    
    def __init__(self, manim_docs_path: str):
        self.manim_docs_path = manim_docs_path
        self.plugins_info = self._load_plugin_information()
        self.context_patterns = self._initialize_context_patterns()
        self.keyword_weights = self._initialize_keyword_weights()
    
    def _load_plugin_information(self) -> Dict[str, PluginInfo]:
        """Load comprehensive plugin information"""
        plugins_info = {}
        
        try:
            plugin_config_path = os.path.join(
                self.manim_docs_path,
                "plugin_docs",
                "plugins.json"
            )
            
            if os.path.exists(plugin_config_path):
                with open(plugin_config_path, "r") as f:
                    plugins_data = json.load(f)
                
                for plugin_data in plugins_data:
                    plugin_info = PluginInfo(
                        name=plugin_data.get("name", ""),
                        description=plugin_data.get("description", ""),
                        namespace=plugin_data.get("namespace", plugin_data.get("name", "")),
                        keywords=plugin_data.get("keywords", []),
                        categories=plugin_data.get("categories", []),
                        version=plugin_data.get("version"),
                        dependencies=plugin_data.get("dependencies", [])
                    )
                    plugins_info[plugin_info.name] = plugin_info
            
            # Add enhanced plugin information with better keyword mapping
            self._enhance_plugin_information(plugins_info)
            
        except Exception as e:
            print(f"Error loading plugin information: {e}")
        
        return plugins_info
    
    def _enhance_plugin_information(self, plugins_info: Dict[str, PluginInfo]):
        """Enhance plugin information with additional context-aware keywords"""
        
        # Enhanced keyword mappings based on common use cases
        enhanced_keywords = {
            "manim_slides": {
                "keywords": ["slide", "presentation", "slideshow", "next_slide", "previous_slide", 
                           "slide_transition", "presenter", "deck", "slides", "present"],
                "categories": ["presentation", "education", "slides"],
                "namespace": "manim_slides"
            },
            "manim_physics": {
                "keywords": ["physics", "gravity", "collision", "rigid_body", "force", "velocity",
                           "acceleration", "momentum", "energy", "simulation", "dynamics"],
                "categories": ["physics", "simulation", "science"],
                "namespace": "manim_physics"
            },
            "manim_chemistry": {
                "keywords": ["molecule", "atom", "bond", "reaction", "chemical", "chemistry",
                           "periodic_table", "compound", "element", "molecular"],
                "categories": ["chemistry", "science", "molecular"],
                "namespace": "manim_chemistry"
            },
            "manim_ml": {
                "keywords": ["neural_network", "machine_learning", "deep_learning", "layer",
                           "activation", "gradient", "training", "model", "ai", "ml"],
                "categories": ["machine_learning", "ai", "neural_networks"],
                "namespace": "manim_ml"
            },
            "manim_data_structures": {
                "keywords": ["array", "list", "tree", "graph", "stack", "queue", "heap",
                           "data_structure", "algorithm", "sorting", "searching"],
                "categories": ["data_structures", "algorithms", "computer_science"],
                "namespace": "manim_data_structures"
            }
        }
        
        for plugin_name, enhanced_info in enhanced_keywords.items():
            if plugin_name in plugins_info:
                plugin = plugins_info[plugin_name]
                # Merge keywords
                existing_keywords = set(plugin.keywords)
                new_keywords = set(enhanced_info["keywords"])
                plugin.keywords = list(existing_keywords.union(new_keywords))
                
                # Update categories
                existing_categories = set(plugin.categories)
                new_categories = set(enhanced_info["categories"])
                plugin.categories = list(existing_categories.union(new_categories))
                
                # Update namespace if not set
                if not plugin.namespace:
                    plugin.namespace = enhanced_info["namespace"]
            else:
                # Create new plugin info if not exists
                plugins_info[plugin_name] = PluginInfo(
                    name=plugin_name,
                    description=f"Enhanced {plugin_name} plugin",
                    namespace=enhanced_info["namespace"],
                    keywords=enhanced_info["keywords"],
                    categories=enhanced_info["categories"]
                )
    
    def _initialize_context_patterns(self) -> Dict[str, List[str]]:
        """Initialize context patterns for better plugin detection"""
        return {
            "presentation_context": [
                r"slide\s*\d+", r"next\s+slide", r"presentation", r"slideshow",
                r"presenter", r"deck", r"slides"
            ],
            "physics_context": [
                r"gravity", r"collision", r"physics", r"force", r"velocity",
                r"acceleration", r"simulation", r"rigid\s*body"
            ],
            "chemistry_context": [
                r"molecule", r"atom", r"chemical", r"reaction", r"bond",
                r"periodic\s*table", r"compound", r"element"
            ],
            "ml_context": [
                r"neural\s*network", r"machine\s*learning", r"deep\s*learning",
                r"layer", r"activation", r"gradient", r"training", r"model"
            ],
            "data_structures_context": [
                r"array", r"list", r"tree", r"graph", r"stack", r"queue",
                r"heap", r"algorithm", r"sorting", r"searching"
            ]
        }
    
    def _initialize_keyword_weights(self) -> Dict[str, float]:
        """Initialize keyword weights for relevance scoring"""
        return {
            "exact_match": 1.0,
            "partial_match": 0.7,
            "context_match": 0.5,
            "category_match": 0.4,
            "description_match": 0.3
        }
    
    def detect_relevant_plugins(self, topic: str, description: str, 
                              implementation_plan: Optional[str] = None,
                              error_context: Optional[str] = None,
                              storyboard_context: Optional[str] = None,
                              task_type: Optional[str] = None) -> List[PluginRelevanceScore]:
        """
        Detect relevant plugins with enhanced context awareness and scoring.
        
        Args:
            topic: Video topic
            description: Video description
            implementation_plan: Optional implementation plan text
            error_context: Optional error context for debugging
            storyboard_context: Optional storyboard context for visual elements
            task_type: Optional task type (animation, presentation, simulation, etc.)
            
        Returns:
            List of plugin relevance scores sorted by relevance
        """
        if not self.plugins_info:
            return []
        
        # Combine all context information with enhanced analysis
        full_context = self._combine_context(topic, description, implementation_plan, 
                                           error_context, storyboard_context)
        
        # Analyze context for better plugin detection
        context_analysis = self._analyze_context_deeply(
            full_context, implementation_plan, error_context, storyboard_context, task_type
        )
        
        # Score each plugin with enhanced context awareness
        plugin_scores = []
        for plugin_name, plugin_info in self.plugins_info.items():
            score = self._score_plugin_relevance_enhanced(
                plugin_info, full_context, context_analysis, topic, description
            )
            if score.confidence_score > 0.1:  # Only include plugins with some relevance
                plugin_scores.append(score)
        
        # Sort by confidence score
        plugin_scores.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return plugin_scores
    
    def _combine_context(self, topic: str, description: str, 
                        implementation_plan: Optional[str] = None,
                        error_context: Optional[str] = None,
                        storyboard_context: Optional[str] = None) -> str:
        """Combine all context information into a single text"""
        context_parts = [topic, description]
        
        if implementation_plan:
            context_parts.append(implementation_plan)
        if error_context:
            context_parts.append(error_context)
        if storyboard_context:
            context_parts.append(storyboard_context)
        
        return " ".join(filter(None, context_parts)).lower()
    
    def _analyze_context_deeply(self, full_context: str, implementation_plan: Optional[str],
                               error_context: Optional[str], storyboard_context: Optional[str],
                               task_type: Optional[str]) -> Dict[str, any]:
        """
        Perform deep context analysis to better understand plugin relevance.
        
        Returns:
            Dictionary containing context analysis results
        """
        analysis = {
            "context_type": self._determine_context_type(full_context, task_type),
            "visual_elements": self._extract_visual_elements(storyboard_context),
            "technical_requirements": self._extract_technical_requirements(implementation_plan),
            "error_patterns": self._analyze_error_patterns(error_context),
            "complexity_level": self._assess_complexity_level(full_context),
            "domain_indicators": self._identify_domain_indicators(full_context)
        }
        
        return analysis
    
    def _determine_context_type(self, context: str, task_type: Optional[str]) -> str:
        """Determine the primary context type"""
        if task_type:
            return task_type.lower()
        
        # Analyze context to determine type
        if any(word in context for word in ["slide", "presentation", "deck"]):
            return "presentation"
        elif any(word in context for word in ["physics", "simulation", "gravity", "collision"]):
            return "physics"
        elif any(word in context for word in ["molecule", "chemistry", "chemical", "atom"]):
            return "chemistry"
        elif any(word in context for word in ["neural", "machine learning", "ml", "ai"]):
            return "machine_learning"
        elif any(word in context for word in ["algorithm", "data structure", "tree", "graph"]):
            return "algorithms"
        else:
            return "general"
    
    def _extract_visual_elements(self, storyboard_context: Optional[str]) -> List[str]:
        """Extract visual elements from storyboard context"""
        if not storyboard_context:
            return []
        
        visual_patterns = [
            r"circle", r"square", r"triangle", r"arrow", r"line", r"curve",
            r"graph", r"chart", r"diagram", r"animation", r"transition",
            r"color", r"movement", r"rotation", r"scaling"
        ]
        
        elements = []
        for pattern in visual_patterns:
            if re.search(pattern, storyboard_context, re.IGNORECASE):
                elements.append(pattern)
        
        return elements
    
    def _extract_technical_requirements(self, implementation_plan: Optional[str]) -> List[str]:
        """Extract technical requirements from implementation plan"""
        if not implementation_plan:
            return []
        
        tech_patterns = [
            r"import\s+(\w+)", r"from\s+(\w+)", r"class\s+(\w+)", r"def\s+(\w+)",
            r"\.(\w+)\(", r"(\w+)\.(\w+)", r"@(\w+)"
        ]
        
        requirements = []
        for pattern in tech_patterns:
            matches = re.findall(pattern, implementation_plan, re.IGNORECASE)
            requirements.extend([match if isinstance(match, str) else match[0] for match in matches])
        
        return list(set(requirements))  # Remove duplicates
    
    def _analyze_error_patterns(self, error_context: Optional[str]) -> Dict[str, any]:
        """Analyze error patterns to suggest relevant plugins"""
        if not error_context:
            return {}
        
        error_analysis = {
            "import_errors": [],
            "attribute_errors": [],
            "missing_dependencies": []
        }
        
        # Look for import errors
        import_error_pattern = r"ModuleNotFoundError.*'(\w+)'"
        import_matches = re.findall(import_error_pattern, error_context)
        error_analysis["import_errors"] = import_matches
        
        # Look for attribute errors
        attr_error_pattern = r"AttributeError.*'(\w+)'"
        attr_matches = re.findall(attr_error_pattern, error_context)
        error_analysis["attribute_errors"] = attr_matches
        
        return error_analysis
    
    def _assess_complexity_level(self, context: str) -> int:
        """Assess the complexity level of the task (1-5)"""
        complexity_indicators = {
            "basic": ["simple", "basic", "easy", "intro", "beginner"],
            "intermediate": ["medium", "intermediate", "moderate"],
            "advanced": ["complex", "advanced", "sophisticated", "detailed"],
            "expert": ["expert", "professional", "production", "enterprise"]
        }
        
        context_lower = context.lower()
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in context_lower for indicator in indicators):
                if level == "basic":
                    return 1
                elif level == "intermediate":
                    return 3
                elif level == "advanced":
                    return 4
                elif level == "expert":
                    return 5
        
        return 2  # Default to basic-intermediate
    
    def _identify_domain_indicators(self, context: str) -> List[str]:
        """Identify domain-specific indicators in the context"""
        domain_keywords = {
            "education": ["teach", "learn", "student", "course", "lesson", "tutorial"],
            "science": ["research", "experiment", "data", "analysis", "study"],
            "business": ["presentation", "meeting", "report", "dashboard", "metrics"],
            "entertainment": ["game", "story", "animation", "visual", "creative"]
        }
        
        identified_domains = []
        context_lower = context.lower()
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in context_lower for keyword in keywords):
                identified_domains.append(domain)
        
        return identified_domains
    
    def _score_plugin_relevance_enhanced(self, plugin_info: PluginInfo, full_context: str,
                                        context_analysis: Dict[str, any], topic: str, 
                                        description: str) -> PluginRelevanceScore:
        """Enhanced plugin relevance scoring with deep context analysis"""
        
        matching_keywords = []
        context_factors = {}
        total_score = 0.0
        
        # 1. Direct keyword matching (enhanced)
        keyword_score = self._score_keyword_matches_enhanced(
            plugin_info.keywords, full_context, context_analysis, matching_keywords
        )
        context_factors["keyword_match"] = keyword_score
        total_score += keyword_score * 0.3
        
        # 2. Context type alignment
        context_type_score = self._score_context_type_alignment(plugin_info, context_analysis)
        context_factors["context_type_match"] = context_type_score
        total_score += context_type_score * 0.25
        
        # 3. Technical requirements matching
        tech_score = self._score_technical_requirements(plugin_info, context_analysis)
        context_factors["technical_match"] = tech_score
        total_score += tech_score * 0.2
        
        # 4. Visual elements alignment
        visual_score = self._score_visual_elements_alignment(plugin_info, context_analysis)
        context_factors["visual_match"] = visual_score
        total_score += visual_score * 0.15
        
        # 5. Error pattern matching
        error_score = self._score_error_pattern_relevance(plugin_info, context_analysis)
        context_factors["error_match"] = error_score
        total_score += error_score * 0.1
        
        # Determine relevance level with enhanced thresholds
        if total_score >= 0.75:
            relevance_level = PluginRelevanceLevel.HIGH
        elif total_score >= 0.45:
            relevance_level = PluginRelevanceLevel.MEDIUM
        elif total_score >= 0.15:
            relevance_level = PluginRelevanceLevel.LOW
        else:
            relevance_level = PluginRelevanceLevel.NONE
        
        # Generate enhanced reasoning
        reasoning = self._generate_enhanced_relevance_reasoning(
            plugin_info.name, matching_keywords, context_factors, context_analysis, total_score
        )
        
        return PluginRelevanceScore(
            plugin_name=plugin_info.name,
            relevance_level=relevance_level,
            confidence_score=min(total_score, 1.0),
            matching_keywords=matching_keywords,
            context_factors=context_factors,
            reasoning=reasoning
        )
    
    def _score_keyword_matches_enhanced(self, keywords: List[str], context: str,
                                       context_analysis: Dict[str, any], 
                                       matching_keywords: List[str]) -> float:
        """Enhanced keyword matching with context weighting"""
        if not keywords:
            return 0.0
        
        matches = 0
        weighted_matches = 0.0
        
        for keyword in keywords:
            if keyword.lower() in context:
                matches += 1
                matching_keywords.append(keyword)
                
                # Weight matches based on context analysis
                weight = 1.0
                if keyword.lower() in context_analysis.get("technical_requirements", []):
                    weight += 0.5  # Technical requirement matches are more important
                if keyword.lower() in context_analysis.get("visual_elements", []):
                    weight += 0.3  # Visual element matches are moderately important
                
                weighted_matches += weight
        
        # Normalize by total possible weighted score
        max_possible_score = len(keywords) * 1.5  # Assuming average weight of 1.5
        return min(weighted_matches / max_possible_score, 1.0)
    
    def _score_context_type_alignment(self, plugin_info: PluginInfo, 
                                     context_analysis: Dict[str, any]) -> float:
        """Score alignment between plugin and context type"""
        context_type = context_analysis.get("context_type", "general")
        plugin_name = plugin_info.name.lower()
        
        # Define plugin-context type alignments
        alignments = {
            "manim_slides": ["presentation", "education", "business"],
            "manim_physics": ["physics", "science", "simulation"],
            "manim_chemistry": ["chemistry", "science", "molecular"],
            "manim_ml": ["machine_learning", "ai", "data"],
            "manim_data_structures": ["algorithms", "computer_science", "education"]
        }
        
        if plugin_name in alignments:
            if context_type in alignments[plugin_name]:
                return 1.0
            elif any(cat in plugin_info.categories for cat in alignments[plugin_name]):
                return 0.7
        
        return 0.0
    
    def _score_technical_requirements(self, plugin_info: PluginInfo,
                                     context_analysis: Dict[str, any]) -> float:
        """Score based on technical requirements matching"""
        tech_requirements = context_analysis.get("technical_requirements", [])
        if not tech_requirements:
            return 0.0
        
        plugin_namespace = plugin_info.namespace.lower()
        plugin_keywords = [kw.lower() for kw in plugin_info.keywords]
        
        matches = 0
        for req in tech_requirements:
            req_lower = req.lower()
            if (plugin_namespace in req_lower or 
                any(keyword in req_lower for keyword in plugin_keywords)):
                matches += 1
        
        return matches / len(tech_requirements) if tech_requirements else 0.0
    
    def _score_visual_elements_alignment(self, plugin_info: PluginInfo,
                                        context_analysis: Dict[str, any]) -> float:
        """Score alignment with visual elements"""
        visual_elements = context_analysis.get("visual_elements", [])
        if not visual_elements:
            return 0.0
        
        # Define visual element alignments for plugins
        visual_alignments = {
            "manim_slides": ["transition", "animation"],
            "manim_physics": ["movement", "rotation", "collision"],
            "manim_chemistry": ["molecule", "bond", "reaction"],
            "manim_ml": ["graph", "chart", "diagram"],
            "manim_data_structures": ["tree", "graph", "diagram"]
        }
        
        plugin_name = plugin_info.name.lower()
        if plugin_name not in visual_alignments:
            return 0.0
        
        aligned_elements = visual_alignments[plugin_name]
        matches = sum(1 for element in visual_elements 
                     if any(aligned in element for aligned in aligned_elements))
        
        return matches / len(visual_elements) if visual_elements else 0.0
    
    def _score_error_pattern_relevance(self, plugin_info: PluginInfo,
                                      context_analysis: Dict[str, any]) -> float:
        """Score relevance based on error patterns"""
        error_patterns = context_analysis.get("error_patterns", {})
        if not error_patterns:
            return 0.0
        
        plugin_namespace = plugin_info.namespace.lower()
        relevance_score = 0.0
        
        # Check import errors
        import_errors = error_patterns.get("import_errors", [])
        for error in import_errors:
            if plugin_namespace in error.lower():
                relevance_score += 0.8
        
        # Check attribute errors
        attr_errors = error_patterns.get("attribute_errors", [])
        for error in attr_errors:
            if any(keyword.lower() in error.lower() for keyword in plugin_info.keywords):
                relevance_score += 0.6
        
        return min(relevance_score, 1.0)
    
    def _generate_enhanced_relevance_reasoning(self, plugin_name: str, matching_keywords: List[str],
                                             context_factors: Dict[str, float], 
                                             context_analysis: Dict[str, any],
                                             total_score: float) -> str:
        """Generate enhanced reasoning for plugin relevance"""
        reasoning_parts = []
        
        # Add context type information
        context_type = context_analysis.get("context_type", "general")
        if context_type != "general":
            reasoning_parts.append(f"Context type: {context_type}")
        
        # Add keyword matches
        if matching_keywords:
            reasoning_parts.append(f"Matched keywords: {', '.join(matching_keywords[:3])}")
        
        # Add strong factor matches
        strong_factors = [factor for factor, score in context_factors.items() if score > 0.6]
        if strong_factors:
            reasoning_parts.append(f"Strong matches: {', '.join(strong_factors)}")
        
        # Add complexity and domain information
        complexity = context_analysis.get("complexity_level", 2)
        if complexity >= 4:
            reasoning_parts.append("High complexity task")
        
        domains = context_analysis.get("domain_indicators", [])
        if domains:
            reasoning_parts.append(f"Domains: {', '.join(domains[:2])}")
        
        # Add confidence assessment
        if total_score > 0.75:
            reasoning_parts.append("High confidence match")
        elif total_score > 0.45:
            reasoning_parts.append("Moderate confidence match")
        else:
            reasoning_parts.append("Low confidence match")
        
        return "; ".join(reasoning_parts)
    
    def _score_plugin_relevance(self, plugin_info: PluginInfo, full_context: str,
                               topic: str, description: str) -> PluginRelevanceScore:
        """Score plugin relevance based on multiple factors"""
        
        matching_keywords = []
        context_factors = {}
        total_score = 0.0
        
        # 1. Direct keyword matching
        keyword_score = self._score_keyword_matches(plugin_info.keywords, full_context, matching_keywords)
        context_factors["keyword_match"] = keyword_score
        total_score += keyword_score * 0.4
        
        # 2. Context pattern matching
        pattern_score = self._score_context_patterns(plugin_info, full_context)
        context_factors["pattern_match"] = pattern_score
        total_score += pattern_score * 0.3
        
        # 3. Category relevance
        category_score = self._score_category_relevance(plugin_info.categories, full_context)
        context_factors["category_match"] = category_score
        total_score += category_score * 0.2
        
        # 4. Description similarity
        description_score = self._score_description_similarity(plugin_info.description, full_context)
        context_factors["description_match"] = description_score
        total_score += description_score * 0.1
        
        # Determine relevance level
        if total_score >= 0.8:
            relevance_level = PluginRelevanceLevel.HIGH
        elif total_score >= 0.5:
            relevance_level = PluginRelevanceLevel.MEDIUM
        elif total_score >= 0.2:
            relevance_level = PluginRelevanceLevel.LOW
        else:
            relevance_level = PluginRelevanceLevel.NONE
        
        # Generate reasoning
        reasoning = self._generate_relevance_reasoning(
            plugin_info.name, matching_keywords, context_factors, total_score
        )
        
        return PluginRelevanceScore(
            plugin_name=plugin_info.name,
            relevance_level=relevance_level,
            confidence_score=min(total_score, 1.0),
            matching_keywords=matching_keywords,
            context_factors=context_factors,
            reasoning=reasoning
        )
    
    def _score_keyword_matches(self, keywords: List[str], context: str, 
                              matching_keywords: List[str]) -> float:
        """Score based on keyword matches"""
        if not keywords:
            return 0.0
        
        matches = 0
        for keyword in keywords:
            if keyword.lower() in context:
                matches += 1
                matching_keywords.append(keyword)
        
        return matches / len(keywords)
    
    def _score_context_patterns(self, plugin_info: PluginInfo, context: str) -> float:
        """Score based on context pattern matching"""
        plugin_name = plugin_info.name.lower()
        
        # Map plugin names to context patterns
        pattern_mapping = {
            "manim_slides": "presentation_context",
            "manim_physics": "physics_context",
            "manim_chemistry": "chemistry_context",
            "manim_ml": "ml_context",
            "manim_data_structures": "data_structures_context"
        }
        
        if plugin_name not in pattern_mapping:
            return 0.0
        
        patterns = self.context_patterns.get(pattern_mapping[plugin_name], [])
        matches = 0
        
        for pattern in patterns:
            if re.search(pattern, context, re.IGNORECASE):
                matches += 1
        
        return matches / len(patterns) if patterns else 0.0
    
    def _score_category_relevance(self, categories: List[str], context: str) -> float:
        """Score based on category relevance"""
        if not categories:
            return 0.0
        
        matches = 0
        for category in categories:
            if category.lower() in context:
                matches += 1
        
        return matches / len(categories)
    
    def _score_description_similarity(self, description: str, context: str) -> float:
        """Score based on description similarity"""
        if not description:
            return 0.0
        
        description_words = set(description.lower().split())
        context_words = set(context.split())
        
        intersection = description_words.intersection(context_words)
        union = description_words.union(context_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_relevance_reasoning(self, plugin_name: str, matching_keywords: List[str],
                                    context_factors: Dict[str, float], total_score: float) -> str:
        """Generate human-readable reasoning for plugin relevance"""
        reasoning_parts = []
        
        if matching_keywords:
            reasoning_parts.append(f"Matched keywords: {', '.join(matching_keywords)}")
        
        high_factors = [factor for factor, score in context_factors.items() if score > 0.5]
        if high_factors:
            reasoning_parts.append(f"Strong matches in: {', '.join(high_factors)}")
        
        if total_score > 0.8:
            reasoning_parts.append("High confidence match")
        elif total_score > 0.5:
            reasoning_parts.append("Moderate confidence match")
        else:
            reasoning_parts.append("Low confidence match")
        
        return "; ".join(reasoning_parts)

class PluginSpecificQueryGenerator:
    """
    Generates plugin-specific queries with proper namespacing and context awareness.
    """
    
    def __init__(self, plugin_detector: ContextAwarePluginDetector):
        self.plugin_detector = plugin_detector
        self.plugin_query_templates = self._initialize_plugin_templates()
    
    def _initialize_plugin_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize plugin-specific query templates"""
        return {
            "manim_slides": {
                "api_reference": [
                    "{namespace}.Slide class documentation",
                    "{namespace}.next_slide method",
                    "{namespace} slide transitions",
                    "{namespace} presentation setup"
                ],
                "examples": [
                    "{namespace} slide creation example",
                    "{namespace} presentation workflow",
                    "{namespace} slide navigation example",
                    "creating slides with {namespace}"
                ],
                "tutorials": [
                    "{namespace} getting started guide",
                    "{namespace} presentation tutorial",
                    "how to use {namespace} for presentations"
                ]
            },
            "manim_physics": {
                "api_reference": [
                    "{namespace}.RigidBody class",
                    "{namespace}.Gravity documentation",
                    "{namespace} collision detection API",
                    "{namespace} physics simulation methods"
                ],
                "examples": [
                    "{namespace} gravity simulation example",
                    "{namespace} collision detection example",
                    "{namespace} rigid body physics",
                    "physics animation with {namespace}"
                ],
                "tutorials": [
                    "{namespace} physics tutorial",
                    "{namespace} simulation guide",
                    "physics animations with {namespace}"
                ]
            },
            "manim_chemistry": {
                "api_reference": [
                    "{namespace}.Molecule class",
                    "{namespace}.ChemicalReaction API",
                    "{namespace} molecular visualization",
                    "{namespace} chemical bond methods"
                ],
                "examples": [
                    "{namespace} molecule creation example",
                    "{namespace} chemical reaction animation",
                    "{namespace} molecular structure example",
                    "chemistry visualization with {namespace}"
                ],
                "tutorials": [
                    "{namespace} chemistry tutorial",
                    "{namespace} molecular visualization guide",
                    "chemical animations with {namespace}"
                ]
            },
            "manim_ml": {
                "api_reference": [
                    "{namespace}.NeuralNetwork class",
                    "{namespace}.Layer documentation",
                    "{namespace} activation functions",
                    "{namespace} gradient visualization"
                ],
                "examples": [
                    "{namespace} neural network example",
                    "{namespace} layer visualization",
                    "{namespace} training animation",
                    "machine learning with {namespace}"
                ],
                "tutorials": [
                    "{namespace} ML tutorial",
                    "{namespace} neural network guide",
                    "ML animations with {namespace}"
                ]
            }
        }
    
    def generate_plugin_queries(self, relevant_plugins: List[PluginRelevanceScore],
                               base_queries: List[str], context: str,
                               context_analysis: Optional[Dict[str, any]] = None) -> List[Dict[str, any]]:
        """
        Generate plugin-specific queries with enhanced context awareness and proper namespacing.
        
        Args:
            relevant_plugins: List of relevant plugins with scores
            base_queries: Base queries to enhance with plugin context
            context: Context for query generation
            context_analysis: Optional deep context analysis results
            
        Returns:
            List of enhanced queries with plugin-specific information
        """
        plugin_queries = []
        
        # Sort plugins by relevance and limit to top performers
        sorted_plugins = sorted(relevant_plugins, key=lambda x: x.confidence_score, reverse=True)
        
        for plugin_score in sorted_plugins[:5]:  # Limit to top 5 most relevant plugins
            if plugin_score.confidence_score < 0.2:
                continue  # Skip low-relevance plugins
            
            plugin_name = plugin_score.plugin_name
            plugin_info = self.plugin_detector.plugins_info.get(plugin_name)
            
            if not plugin_info:
                continue
            
            # Generate context-aware queries for this plugin
            queries = self._generate_context_aware_queries_for_plugin(
                plugin_info, plugin_score, base_queries, context, context_analysis
            )
            plugin_queries.extend(queries)
        
        # Deduplicate and rank queries
        plugin_queries = self._deduplicate_and_rank_queries(plugin_queries)
        
        return plugin_queries
    
    def _generate_context_aware_queries_for_plugin(self, plugin_info: PluginInfo, 
                                                  plugin_score: PluginRelevanceScore,
                                                  base_queries: List[str], context: str,
                                                  context_analysis: Optional[Dict[str, any]]) -> List[Dict[str, any]]:
        """Generate context-aware queries for a specific plugin"""
        queries = []
        plugin_name = plugin_info.name
        namespace = plugin_info.namespace
        
        # Get plugin-specific templates
        templates = self.plugin_query_templates.get(plugin_name, {})
        
        # Generate API reference queries with context awareness
        api_templates = templates.get("api_reference", [])
        for template in api_templates:
            query = template.format(namespace=namespace)
            
            # Enhance query based on context analysis
            if context_analysis:
                enhanced_query = self._enhance_query_with_context(query, context_analysis, plugin_info)
                queries.append({
                    "query": enhanced_query,
                    "plugin": plugin_name,
                    "namespace": namespace,
                    "type": "api_reference",
                    "confidence": plugin_score.confidence_score * 0.9,
                    "reasoning": f"Context-aware API query for {plugin_name}",
                    "context_factors": plugin_score.context_factors
                })
            else:
                queries.append({
                    "query": query,
                    "plugin": plugin_name,
                    "namespace": namespace,
                    "type": "api_reference",
                    "confidence": plugin_score.confidence_score * 0.9,
                    "reasoning": f"Plugin-specific API query for {plugin_name}"
                })
        
        # Generate example queries with context-specific focus
        example_templates = templates.get("examples", [])
        for template in example_templates:
            query = template.format(namespace=namespace)
            
            # Prioritize examples based on context
            if context_analysis:
                enhanced_query = self._enhance_query_with_context(query, context_analysis, plugin_info)
                confidence_boost = self._calculate_context_confidence_boost(context_analysis, plugin_info)
                queries.append({
                    "query": enhanced_query,
                    "plugin": plugin_name,
                    "namespace": namespace,
                    "type": "example",
                    "confidence": plugin_score.confidence_score * (0.8 + confidence_boost),
                    "reasoning": f"Context-enhanced example query for {plugin_name}",
                    "context_factors": plugin_score.context_factors
                })
            else:
                queries.append({
                    "query": query,
                    "plugin": plugin_name,
                    "namespace": namespace,
                    "type": "example",
                    "confidence": plugin_score.confidence_score * 0.8,
                    "reasoning": f"Plugin-specific example query for {plugin_name}"
                })
        
        # Generate tutorial queries for high-relevance plugins
        if plugin_score.confidence_score > 0.5:
            tutorial_templates = templates.get("tutorials", [])
            for template in tutorial_templates:
                query = template.format(namespace=namespace)
                queries.append({
                    "query": query,
                    "plugin": plugin_name,
                    "namespace": namespace,
                    "type": "tutorial",
                    "confidence": plugin_score.confidence_score * 0.7,
                    "reasoning": f"Plugin-specific tutorial query for {plugin_name}"
                })
        
        # Generate context-specific queries based on matching keywords
        if plugin_score.matching_keywords:
            for keyword in plugin_score.matching_keywords[:2]:  # Top 2 matching keywords
                context_query = f"{namespace} {keyword} implementation"
                queries.append({
                    "query": context_query,
                    "plugin": plugin_name,
                    "namespace": namespace,
                    "type": "context_specific",
                    "confidence": plugin_score.confidence_score * 0.75,
                    "reasoning": f"Keyword-specific query for {keyword} in {plugin_name}"
                })
        
        # Enhance base queries with plugin context (limited to avoid query explosion)
        for base_query in base_queries[:2]:  # Limit to top 2 base queries
            enhanced_query = f"{base_query} using {namespace}"
            queries.append({
                "query": enhanced_query,
                "plugin": plugin_name,
                "namespace": namespace,
                "type": "enhanced_base",
                "confidence": plugin_score.confidence_score * 0.6,
                "reasoning": f"Base query enhanced with {plugin_name} context"
            })
        
        return queries
    
    def _enhance_query_with_context(self, base_query: str, context_analysis: Dict[str, any],
                                   plugin_info: PluginInfo) -> str:
        """Enhance a query with context-specific information"""
        enhanced_query = base_query
        
        # Add context type specific terms
        context_type = context_analysis.get("context_type", "general")
        if context_type != "general":
            enhanced_query += f" for {context_type}"
        
        # Add complexity level indicators
        complexity = context_analysis.get("complexity_level", 2)
        if complexity >= 4:
            enhanced_query += " advanced"
        elif complexity == 1:
            enhanced_query += " beginner"
        
        # Add domain-specific terms
        domains = context_analysis.get("domain_indicators", [])
        if domains:
            primary_domain = domains[0]
            enhanced_query += f" {primary_domain}"
        
        return enhanced_query
    
    def _calculate_context_confidence_boost(self, context_analysis: Dict[str, any],
                                          plugin_info: PluginInfo) -> float:
        """Calculate confidence boost based on context alignment"""
        boost = 0.0
        
        # Boost for strong context type alignment
        context_type = context_analysis.get("context_type", "general")
        plugin_name = plugin_info.name.lower()
        
        strong_alignments = {
            "manim_slides": "presentation",
            "manim_physics": "physics",
            "manim_chemistry": "chemistry",
            "manim_ml": "machine_learning"
        }
        
        if plugin_name in strong_alignments and context_type == strong_alignments[plugin_name]:
            boost += 0.15
        
        # Boost for technical requirements match
        tech_requirements = context_analysis.get("technical_requirements", [])
        if any(plugin_info.namespace.lower() in req.lower() for req in tech_requirements):
            boost += 0.1
        
        # Boost for visual elements alignment
        visual_elements = context_analysis.get("visual_elements", [])
        if visual_elements and plugin_name in ["manim_physics", "manim_chemistry", "manim_ml"]:
            boost += 0.05
        
        return min(boost, 0.2)  # Cap boost at 0.2
    
    def _deduplicate_and_rank_queries(self, queries: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Remove duplicate queries and rank by confidence"""
        # Remove exact duplicates
        seen_queries = set()
        unique_queries = []
        
        for query_dict in queries:
            query_text = query_dict["query"].lower().strip()
            if query_text not in seen_queries:
                seen_queries.add(query_text)
                unique_queries.append(query_dict)
        
        # Sort by confidence score
        unique_queries.sort(key=lambda x: x["confidence"], reverse=True)
        
        # Limit total queries to prevent overwhelming the system
        return unique_queries[:20]
    
    def _generate_queries_for_plugin(self, plugin_info: PluginInfo, 
                                   plugin_score: PluginRelevanceScore,
                                   base_queries: List[str], context: str) -> List[Dict[str, any]]:
        """Generate queries for a specific plugin"""
        queries = []
        plugin_name = plugin_info.name
        namespace = plugin_info.namespace
        
        # Get plugin-specific templates
        templates = self.plugin_query_templates.get(plugin_name, {})
        
        # Generate API reference queries
        api_templates = templates.get("api_reference", [])
        for template in api_templates:
            query = template.format(namespace=namespace)
            queries.append({
                "query": query,
                "plugin": plugin_name,
                "namespace": namespace,
                "type": "api_reference",
                "confidence": plugin_score.confidence_score * 0.9,
                "reasoning": f"Plugin-specific API query for {plugin_name}"
            })
        
        # Generate example queries
        example_templates = templates.get("examples", [])
        for template in example_templates:
            query = template.format(namespace=namespace)
            queries.append({
                "query": query,
                "plugin": plugin_name,
                "namespace": namespace,
                "type": "example",
                "confidence": plugin_score.confidence_score * 0.8,
                "reasoning": f"Plugin-specific example query for {plugin_name}"
            })
        
        # Generate tutorial queries if high relevance
        if plugin_score.confidence_score > 0.6:
            tutorial_templates = templates.get("tutorials", [])
            for template in tutorial_templates:
                query = template.format(namespace=namespace)
                queries.append({
                    "query": query,
                    "plugin": plugin_name,
                    "namespace": namespace,
                    "type": "tutorial",
                    "confidence": plugin_score.confidence_score * 0.7,
                    "reasoning": f"Plugin-specific tutorial query for {plugin_name}"
                })
        
        # Enhance base queries with plugin context
        for base_query in base_queries[:3]:  # Limit to top 3 base queries
            enhanced_query = f"{base_query} {namespace}"
            queries.append({
                "query": enhanced_query,
                "plugin": plugin_name,
                "namespace": namespace,
                "type": "enhanced_base",
                "confidence": plugin_score.confidence_score * 0.6,
                "reasoning": f"Base query enhanced with {plugin_name} context"
            })
        
        return queries
    
    def score_plugin_query_relevance(self, query: str, plugin_info: PluginInfo,
                                   task_context: str) -> float:
        """Score the relevance of a plugin query for the given task context"""
        relevance_score = 0.0
        
        # Check if plugin namespace is in query
        if plugin_info.namespace.lower() in query.lower():
            relevance_score += 0.4
        
        # Check if plugin keywords are in task context
        context_lower = task_context.lower()
        matching_keywords = 0
        for keyword in plugin_info.keywords:
            if keyword.lower() in context_lower:
                matching_keywords += 1
        
        if plugin_info.keywords:
            keyword_relevance = matching_keywords / len(plugin_info.keywords)
            relevance_score += keyword_relevance * 0.6
        
        return min(relevance_score, 1.0)