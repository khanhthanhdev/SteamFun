"""
Local Embedding Provider

This module implements the LocalEmbeddingProvider class that wraps existing
local embedding functionality (HuggingFace embeddings) in the provider interface.
This ensures backward compatibility while enabling the provider switching mechanism.
"""

import logging
import time
from typing import List, Dict, Any, Optional
import os

from .embedding_providers import (
    EmbeddingProvider,
    EmbeddingConfig,
    EmbeddingGenerationError
)

# Import HuggingFace embeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    HuggingFaceEmbeddings = None


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local embedding provider using HuggingFace embeddings.
    
    This provider wraps the existing local embedding functionality to conform
    to the EmbeddingProvider interface, maintaining backward compatibility
    while enabling provider switching capabilities.
    """
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize the local embedding provider.
        
        Args:
            config: EmbeddingConfig object containing provider settings
            
        Raises:
            EmbeddingGenerationError: If HuggingFace embeddings are not available
        """
        super().__init__(config)
        
        if not HUGGINGFACE_AVAILABLE:
            raise EmbeddingGenerationError(
                "HuggingFace embeddings not available. "
                "Please install: pip install langchain-community sentence-transformers"
            )
        
        # Initialize the HuggingFace embedding model
        self.embedding_function = self._initialize_local_model()
        
        # Track performance metrics
        self._total_requests = 0
        self._total_texts_processed = 0
        self._total_processing_time = 0.0
        self._last_error = None
        
        self.logger.info(f"Initialized LocalEmbeddingProvider with model: {config.model_name}")
    
    def _initialize_local_model(self) -> HuggingFaceEmbeddings:
        """Initialize the local HuggingFace embedding model.
        
        Returns:
            HuggingFaceEmbeddings instance configured for the model
            
        Raises:
            EmbeddingGenerationError: If model initialization fails
        """
        try:
            # Extract model name (remove 'hf:' prefix if present)
            model_name = self.config.model_name
            if model_name.startswith('hf:'):
                model_name = model_name[3:]
            
            self.logger.info(f"Initializing HuggingFace model: {model_name}")
            
            # Configure model parameters for optimal performance
            model_kwargs = {
                'device': 'cpu',  # Use CPU by default for compatibility
                'trust_remote_code': False  # Security best practice
            }
            
            # Configure encoding parameters
            encode_kwargs = {
                'normalize_embeddings': True,  # Normalize for better similarity search
                'batch_size': min(self.config.batch_size, 32)  # Reasonable batch size for local processing
            }
            
            # Check if GPU is available and configure accordingly
            try:
                import torch
                if torch.cuda.is_available():
                    model_kwargs['device'] = 'cuda'
                    self.logger.info("CUDA available, using GPU for embeddings")
                else:
                    self.logger.info("CUDA not available, using CPU for embeddings")
            except ImportError:
                self.logger.info("PyTorch not available, using CPU for embeddings")
            
            # Create the embedding function
            embedding_function = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            
            # Test the model with a simple embedding to ensure it works
            try:
                test_embedding = embedding_function.embed_query("test")
                if not test_embedding or len(test_embedding) == 0:
                    raise EmbeddingGenerationError("Model test failed: empty embedding returned")
                
                # Update config dimensions based on actual model output
                actual_dimensions = len(test_embedding)
                if self.config.dimensions != actual_dimensions:
                    self.logger.warning(
                        f"Config dimensions ({self.config.dimensions}) don't match "
                        f"model dimensions ({actual_dimensions}). Using actual: {actual_dimensions}"
                    )
                    self.config.dimensions = actual_dimensions
                
                self.logger.info(f"Model test successful, embedding dimension: {actual_dimensions}")
                
            except Exception as e:
                raise EmbeddingGenerationError(f"Model test failed: {e}")
            
            return embedding_function
            
        except Exception as e:
            error_msg = f"Failed to initialize local embedding model '{self.config.model_name}': {e}"
            self.logger.error(error_msg)
            raise EmbeddingGenerationError(error_msg)
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts using the local model.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors, where each vector is a list of floats
            
        Raises:
            EmbeddingGenerationError: If embedding generation fails
        """
        if not texts:
            return []
        
        # Validate input
        if not isinstance(texts, list):
            raise EmbeddingGenerationError("Input must be a list of strings")
        
        # Filter out empty or invalid texts
        valid_texts = []
        text_indices = []  # Track original indices for result mapping
        
        for i, text in enumerate(texts):
            if isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                text_indices.append(i)
            else:
                self.logger.warning(f"Skipping invalid text at index {i}: {type(text)}")
        
        if not valid_texts:
            self.logger.warning("No valid texts to process")
            return [[0.0] * self.config.dimensions] * len(texts)  # Return zero embeddings
        
        try:
            start_time = time.time()
            
            # Process texts in batches to manage memory usage
            all_embeddings = []
            batch_size = min(self.config.batch_size, 32)  # Reasonable batch size for local processing
            
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i + batch_size]
                
                try:
                    # Generate embeddings for the batch
                    batch_embeddings = self.embedding_function.embed_documents(batch_texts)
                    
                    # Validate batch results
                    if not batch_embeddings or len(batch_embeddings) != len(batch_texts):
                        raise EmbeddingGenerationError(
                            f"Batch embedding failed: expected {len(batch_texts)} embeddings, "
                            f"got {len(batch_embeddings) if batch_embeddings else 0}"
                        )
                    
                    # Validate embedding dimensions
                    for j, embedding in enumerate(batch_embeddings):
                        if not isinstance(embedding, list) or len(embedding) != self.config.dimensions:
                            raise EmbeddingGenerationError(
                                f"Invalid embedding at batch {i//batch_size}, item {j}: "
                                f"expected dimension {self.config.dimensions}, "
                                f"got {len(embedding) if isinstance(embedding, list) else type(embedding)}"
                            )
                    
                    all_embeddings.extend(batch_embeddings)
                    
                except Exception as e:
                    error_msg = f"Batch processing failed for texts {i}-{i+len(batch_texts)}: {e}"
                    self.logger.error(error_msg)
                    raise EmbeddingGenerationError(error_msg)
            
            # Create result array with proper mapping back to original indices
            result_embeddings = []
            valid_idx = 0
            
            for i in range(len(texts)):
                if i in text_indices:
                    result_embeddings.append(all_embeddings[valid_idx])
                    valid_idx += 1
                else:
                    # Return zero embedding for invalid texts
                    result_embeddings.append([0.0] * self.config.dimensions)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._total_requests += 1
            self._total_texts_processed += len(texts)
            self._total_processing_time += processing_time
            self._last_error = None
            
            self.logger.debug(
                f"Generated {len(result_embeddings)} embeddings in {processing_time:.2f}s "
                f"({len(texts)/processing_time:.1f} texts/sec)"
            )
            
            return result_embeddings
            
        except EmbeddingGenerationError:
            # Re-raise embedding generation errors as-is
            raise
        except Exception as e:
            error_msg = f"Local embedding generation failed: {e}"
            self.logger.error(error_msg)
            self._last_error = str(e)
            raise EmbeddingGenerationError(error_msg)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider.
        
        Returns:
            Integer representing the embedding dimension
        """
        return self.config.dimensions
    
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured.
        
        Returns:
            True if provider is available, False otherwise
        """
        try:
            # Test with a simple embedding without affecting metrics
            if not hasattr(self, 'embedding_function') or self.embedding_function is None:
                return False
            
            # Use the embedding function directly to avoid incrementing metrics
            test_result = self.embedding_function.embed_query("test")
            return (
                test_result is not None and
                isinstance(test_result, list) and
                len(test_result) == self.config.dimensions and
                all(isinstance(x, (int, float)) for x in test_result)
            )
        except Exception as e:
            self.logger.warning(f"Availability check failed: {e}")
            return False
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the provider for logging and debugging.
        
        Returns:
            Dictionary containing provider information
        """
        # Calculate average processing time
        avg_processing_time = (
            self._total_processing_time / self._total_requests 
            if self._total_requests > 0 else 0.0
        )
        
        # Calculate average texts per second
        avg_texts_per_second = (
            self._total_texts_processed / self._total_processing_time 
            if self._total_processing_time > 0 else 0.0
        )
        
        # Get model information
        model_name = self.config.model_name
        if model_name.startswith('hf:'):
            model_name = model_name[3:]
        
        # Check device information
        device_info = "cpu"
        try:
            import torch
            if torch.cuda.is_available() and hasattr(self.embedding_function, 'client'):
                device_info = "cuda"
        except ImportError:
            pass
        
        return {
            "provider": "local",
            "model": model_name,
            "dimensions": self.config.dimensions,
            "available": self.is_available(),
            "device": device_info,
            "batch_size": self.config.batch_size,
            "performance_metrics": {
                "total_requests": self._total_requests,
                "total_texts_processed": self._total_texts_processed,
                "total_processing_time": round(self._total_processing_time, 2),
                "avg_processing_time": round(avg_processing_time, 3),
                "avg_texts_per_second": round(avg_texts_per_second, 1)
            },
            "last_error": self._last_error,
            "huggingface_available": HUGGINGFACE_AVAILABLE,
            "configuration": {
                "model_name": self.config.model_name,
                "timeout": self.config.timeout,
                "normalize_embeddings": True
            }
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get detailed information about the underlying model.
        
        Returns:
            Dictionary with model-specific information
        """
        model_name = self.config.model_name
        if model_name.startswith('hf:'):
            model_name = model_name[3:]
        
        # Provide information about common models
        model_info = {
            "model_name": model_name,
            "model_type": "sentence-transformer",
            "embedding_dimension": self.config.dimensions,
            "max_sequence_length": 512,  # Common default
            "language_support": "multilingual" if "multilingual" in model_name.lower() else "english",
            "specialized_for": []
        }
        
        # Add specialization information based on model name
        if "code" in model_name.lower():
            model_info["specialized_for"].append("code")
        if "granite" in model_name.lower():
            model_info["specialized_for"].extend(["code", "technical_documentation"])
        if "minilm" in model_name.lower():
            model_info["specialized_for"].append("general_purpose")
        
        # Add known model-specific information
        known_models = {
            "ibm-granite/granite-embedding-30m-english": {
                "embedding_dimension": 384,
                "max_sequence_length": 512,
                "specialized_for": ["code", "technical_documentation"],
                "language_support": "english"
            },
            "sentence-transformers/all-MiniLM-L6-v2": {
                "embedding_dimension": 384,
                "max_sequence_length": 256,
                "specialized_for": ["general_purpose"],
                "language_support": "multilingual"
            },
            "microsoft/codebert-base": {
                "embedding_dimension": 768,
                "max_sequence_length": 512,
                "specialized_for": ["code"],
                "language_support": "code"
            }
        }
        
        if model_name in known_models:
            model_info.update(known_models[model_name])
        
        return model_info
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self._total_requests = 0
        self._total_texts_processed = 0
        self._total_processing_time = 0.0
        self._last_error = None
        self.logger.info("Performance metrics reset")
    
    def warm_up(self, sample_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """Warm up the model by processing sample texts.
        
        Args:
            sample_texts: Optional list of texts to use for warm-up.
                         If None, uses default sample texts.
        
        Returns:
            Dictionary with warm-up results
        """
        if sample_texts is None:
            sample_texts = [
                "This is a test sentence for model warm-up.",
                "def example_function(): return 'Hello, World!'",
                "Manim is a Python library for creating mathematical animations."
            ]
        
        try:
            start_time = time.time()
            embeddings = self.generate_embeddings(sample_texts)
            warm_up_time = time.time() - start_time
            
            return {
                "success": True,
                "warm_up_time": round(warm_up_time, 3),
                "texts_processed": len(sample_texts),
                "embeddings_generated": len(embeddings),
                "avg_time_per_text": round(warm_up_time / len(sample_texts), 4)
            }
            
        except Exception as e:
            self.logger.error(f"Model warm-up failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }