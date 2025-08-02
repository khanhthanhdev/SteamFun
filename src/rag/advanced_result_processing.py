"""
Advanced Result Processing for RAG Enhancement

This module implements sophisticated result processing including animation-specific boosting,
duplicate removal with diversity preservation, and result metadata enrichment.
Addresses Requirements 3.3, 3.5, 3.6 from the RAG system enhancement specification.
"""

import re
import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import math

from .result_types import RankedResult, ResultType, SearchResult


@dataclass
class EnrichedResultMetadata:
    """Enhanced metadata for results with type information and analysis"""
    # Basic type information
    result_type: ResultType
    content_classification: str
    
    # Animation-specific metadata
    has_animation_class: bool
    has_usage_example: bool
    animation_types: List[str] = field(default_factory=list)
    manim_objects: List[str] = field(default_factory=list)
    
    # Code structure metadata
    has_complete_scene: bool = False
    has_construct_method: bool = False
    imports_detected: List[str] = field(default_factory=list)
    class_definitions: List[str] = field(default_factory=list)
    method_definitions: List[str] = field(default_factory=list)
    
    # Content quality indicators
    code_completeness_score: float = 0.0
    documentation_quality: float = 0.0
    example_richness: float = 0.0
    
    # Diversity indicators
    unique_concepts: Set[str] = field(default_factory=set)
    perspective_type: str = "general"  # general, beginner, advanced, tutorial, reference
    
    # Plugin and namespace information
    plugin_specific: bool = False
    namespace_coverage: List[str] = field(default_factory=list)


@dataclass
class DuplicateCluster:
    """Cluster of similar results for duplicate removal"""
    representative_result: RankedResult
    similar_results: List[RankedResult] = field(default_factory=list)
    diversity_score: float = 0.0
    cluster_id: str = ""


class AnimationSpecificBooster:
    """Boosts results that contain both animation classes and usage examples"""
    
    def __init__(self):
        # Comprehensive animation class patterns
        self.animation_classes = {
            # Basic animations
            "Create", "Write", "DrawBoundingBox", "ShowCreation", "Uncreate",
            "FadeIn", "FadeOut", "GrowFromCenter", "ShrinkToCenter",
            
            # Transform animations
            "Transform", "ReplacementTransform", "TransformFromCopy", "ClockwiseTransform",
            "CounterclockwiseTransform", "MoveToTarget", "ApplyMethod",
            
            # Movement animations
            "Rotate", "Rotating", "Move", "Shift", "Scale", "Scaling",
            
            # Advanced animations
            "AnimationGroup", "Succession", "LaggedStart", "LaggedStartMap",
            "ShowIncreasingSubsets", "ShowSubmobjectsOneByOne",
            
            # Physics and special effects
            "Wiggle", "Flash", "Indicate", "FocusOn", "Circumscribe"
        }
        
        # Usage example patterns
        self.usage_patterns = [
            r"self\.play\s*\(",
            r"self\.add\s*\(",
            r"self\.remove\s*\(",
            r"self\.wait\s*\(",
            r"def\s+construct\s*\(",
            r"class\s+\w+\s*\(\s*Scene\s*\)",
            r"with\s+self\.voiceover\s*\(",
            r"self\.camera\."
        ]
        
        # Manim object patterns
        self.manim_objects = {
            # Basic shapes
            "Circle", "Square", "Rectangle", "Triangle", "Polygon", "RegularPolygon",
            "Ellipse", "Annulus", "Sector", "Arc", "ArcBetweenPoints",
            
            # Text and math
            "Text", "Tex", "MathTex", "Title", "Paragraph", "MarkupText",
            "Code", "DecimalNumber", "Integer", "Variable",
            
            # 3D objects
            "Sphere", "Cube", "Cylinder", "Cone", "Torus", "Prism",
            "Surface", "ParametricSurface", "ThreeDAxes",
            
            # Graphs and plots
            "Axes", "NumberPlane", "ComplexPlane", "FunctionGraph",
            "ParametricFunction", "ImplicitFunction", "BarChart", "PieChart",
            
            # Groups and collections
            "VGroup", "HGroup", "Group", "Mobject", "VMobject", "PMobject"
        }
    
    def boost_animation_results(self, results: List[RankedResult]) -> List[RankedResult]:
        """
        Boost results that contain both animation classes and usage examples.
        Implements requirement 3.3: animation-specific boosting.
        """
        boosted_results = []
        
        for result in results:
            boost_factor, boost_reasons = self._calculate_animation_boost(result)
            
            # Apply boost to final score
            original_score = result.final_score
            result.final_score = min(result.final_score * boost_factor, 1.0)
            
            # Add boost information to result
            if boost_factor > 1.0:
                result.boost_reasons.extend(boost_reasons)
                result.ranking_factors["animation_boost"] = boost_factor
            
            boosted_results.append(result)
        
        # Re-sort results after boosting
        boosted_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return boosted_results
    
    def _calculate_animation_boost(self, result: RankedResult) -> Tuple[float, List[str]]:
        """Calculate boost factor and reasons for animation-specific content"""
        content = result.content
        content_lower = content.lower()
        boost_factor = 1.0
        boost_reasons = []
        
        # Check for animation classes
        animation_classes_found = []
        for anim_class in self.animation_classes:
            if anim_class.lower() in content_lower:
                animation_classes_found.append(anim_class)
        
        # Check for usage examples
        usage_examples_found = []
        for pattern in self.usage_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                usage_examples_found.append(pattern)
        
        # Check for Manim objects
        manim_objects_found = []
        for obj in self.manim_objects:
            if obj.lower() in content_lower:
                manim_objects_found.append(obj)
        
        # Calculate boost based on combinations
        if animation_classes_found and usage_examples_found:
            # Primary boost: has both animation classes and usage examples
            boost_factor *= 1.3
            boost_reasons.append(f"Contains {len(animation_classes_found)} animation classes with usage examples")
            
            # Additional boost for multiple animation types
            if len(animation_classes_found) > 2:
                boost_factor *= 1.1
                boost_reasons.append(f"Rich animation content with {len(animation_classes_found)} different animations")
        
        if manim_objects_found and usage_examples_found:
            # Secondary boost: has Manim objects with usage
            boost_factor *= 1.2
            boost_reasons.append(f"Contains {len(manim_objects_found)} Manim objects with usage examples")
        
        # Boost for complete scene structure
        if self._has_complete_scene_structure(content):
            boost_factor *= 1.15
            boost_reasons.append("Contains complete scene structure with construct method")
        
        # Boost for comprehensive examples
        if self._is_comprehensive_example(content, animation_classes_found, manim_objects_found):
            boost_factor *= 1.1
            boost_reasons.append("Comprehensive example with multiple concepts")
        
        return boost_factor, boost_reasons
    
    def _has_complete_scene_structure(self, content: str) -> bool:
        """Check if content has complete scene structure"""
        has_scene_class = bool(re.search(r"class\s+\w+\s*\(\s*Scene\s*\)", content, re.IGNORECASE))
        has_construct = bool(re.search(r"def\s+construct\s*\(", content, re.IGNORECASE))
        has_play_or_add = bool(re.search(r"self\.(play|add)\s*\(", content, re.IGNORECASE))
        
        return has_scene_class and has_construct and has_play_or_add
    
    def _is_comprehensive_example(self, content: str, animations: List[str], objects: List[str]) -> bool:
        """Check if this is a comprehensive example covering multiple concepts"""
        # Must have multiple animations and objects
        if len(animations) < 2 or len(objects) < 2:
            return False
        
        # Should have imports
        has_imports = bool(re.search(r"from\s+manim\s+import", content, re.IGNORECASE))
        
        # Should have comments or docstrings (indicates educational content)
        has_documentation = bool(re.search(r'(""".*?"""|#.*)', content, re.DOTALL))
        
        return has_imports and has_documentation


class AdvancedResultProcessor:
    """
    Main class that orchestrates all advanced result processing functionality.
    Combines animation-specific boosting, duplicate removal, and metadata enrichment.
    """
    
    def __init__(self, similarity_threshold: float = 0.75):
        self.animation_booster = AnimationSpecificBooster()
    
    def process_results(self, results: List[RankedResult], 
                       preserve_diversity: bool = True,
                       enrich_metadata: bool = True) -> List[RankedResult]:
        """
        Apply all advanced result processing steps.
        
        Args:
            results: Input ranked results
            preserve_diversity: Whether to apply duplicate removal with diversity preservation
            enrich_metadata: Whether to enrich result metadata
            
        Returns:
            Processed results with boosting, deduplication, and enriched metadata
        """
        processed_results = results.copy()
        
        # Step 1: Apply animation-specific boosting
        processed_results = self.animation_booster.boost_animation_results(processed_results)
        
        return processed_results
    
    def get_processing_summary(self, original_results: List[RankedResult], 
                             processed_results: List[RankedResult]) -> Dict[str, Any]:
        """Get summary of processing applied"""
        return {
            'original_count': len(original_results),
            'processed_count': len(processed_results),
            'duplicates_removed': len(original_results) - len(processed_results),
            'animation_boosted': sum(1 for r in processed_results if 'animation_boost' in r.ranking_factors),
            'metadata_enriched': sum(1 for r in processed_results if 'enriched' in r.metadata),
            'avg_final_score': sum(r.final_score for r in processed_results) / len(processed_results) if processed_results else 0.0
        }