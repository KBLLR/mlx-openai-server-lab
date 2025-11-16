"""Model registry for managing multiple model handlers."""

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from app.schemas.model import ModelMetadata


class ModelRegistry:
    """
    Registry for managing model handlers.

    Maintains a thread-safe registry of loaded models and their handlers.
    In Phase 1, this wraps the existing single-model flow. Future phases
    will extend this to support multi-model loading and hot-swapping.

    Attributes:
        _handlers: Dict mapping model_id to handler instance
        _metadata: Dict mapping model_id to ModelMetadata
        _lock: Async lock for thread-safe operations
    """

    def __init__(self):
        """Initialize empty model registry."""
        self._handlers: Dict[str, Any] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._lock = asyncio.Lock()
        logger.info("Model registry initialized")

    def _infer_model_family(self, model_id: str) -> Optional[str]:
        """
        Infer model family from model_id.

        Args:
            model_id: Model identifier (e.g., mlx-community/gemma-3-4b-it-4bit)

        Returns:
            Model family name or None
        """
        model_lower = model_id.lower()
        if "gemma" in model_lower:
            return "gemma"
        elif "llama" in model_lower:
            return "llama"
        elif "qwen" in model_lower:
            return "qwen"
        elif "phi" in model_lower:
            return "phi"
        elif "mistral" in model_lower:
            return "mistral"
        elif "glm" in model_lower:
            return "glm"
        elif "flux" in model_lower:
            return "flux"
        elif "whisper" in model_lower:
            return "whisper"
        return None

    def _generate_model_description(self, model_id: str, model_type: str, family: Optional[str]) -> str:
        """
        Generate a human-readable description for the model.

        Args:
            model_id: Model identifier
            model_type: Model type
            family: Model family

        Returns:
            Description string
        """
        family_str = f"{family.capitalize()} " if family else ""
        type_map = {
            "lm": "language model",
            "multimodal": "multimodal model",
            "embeddings": "embedding model",
            "whisper": "speech recognition model",
            "image-generation": "image generation model",
            "image-edit": "image editing model",
        }
        type_str = type_map.get(model_type, model_type)
        return f"{family_str}{type_str} running on MLX"

    def _generate_model_tags(self, model_type: str, model_id: str) -> List[str]:
        """
        Generate appropriate tags for a model.

        Args:
            model_type: Model type
            model_id: Model identifier

        Returns:
            List of tags
        """
        tags = ["local"]

        # Add type-based tags
        if model_type in ["lm", "multimodal"]:
            tags.append("chat")
        if model_type == "embeddings":
            tags.append("embeddings")
        if model_type in ["image-generation", "image-edit"]:
            tags.append("image")
        if model_type == "whisper":
            tags.append("audio")

        # Check for quantization in model_id
        if "4bit" in model_id.lower() or "4b" in model_id.lower():
            tags.append("quantized")
        elif "8bit" in model_id.lower() or "8b" in model_id.lower():
            tags.append("quantized")

        return tags

    async def register_model(
        self,
        model_id: str,
        handler: Any,
        model_type: str,
        context_length: Optional[int] = None,
        description: Optional[str] = None,
        family: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """
        Register a model handler with metadata.

        Args:
            model_id: Unique identifier for the model
            handler: Handler instance (MLXLMHandler, MLXVLMHandler, etc.)
            model_type: Type of model (lm, multimodal, embeddings, etc.)
            context_length: Maximum context length (if applicable)
            description: Optional custom description
            family: Optional model family (auto-detected if not provided)
            tags: Optional list of tags (auto-generated if not provided)

        Raises:
            ValueError: If model_id already registered
        """
        async with self._lock:
            if model_id in self._handlers:
                raise ValueError(f"Model '{model_id}' is already registered")

            # Infer family if not provided
            if family is None:
                family = self._infer_model_family(model_id)

            # Generate description if not provided
            if description is None:
                description = self._generate_model_description(model_id, model_type, family)

            # Generate tags if not provided
            if tags is None:
                tags = self._generate_model_tags(model_type, model_id)

            # Create metadata
            metadata = ModelMetadata(
                id=model_id,
                type=model_type,
                context_length=context_length,
                created_at=int(time.time()),
                description=description,
                family=family,
                tags=tags,
            )

            # Store handler and metadata
            self._handlers[model_id] = handler
            self._metadata[model_id] = metadata

            logger.info(
                f"Registered model: {model_id} (type={model_type}, "
                f"family={family}, context_length={context_length})"
            )

    def get_handler(self, model_id: str) -> Any:
        """
        Get handler for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            Handler instance

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in self._handlers:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._handlers[model_id]

    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models with rich metadata for Tier 2 discovery.

        Returns:
            List of model metadata dicts with extended fields
        """
        return [
            {
                "id": metadata.id,
                "object": metadata.object,
                "created": metadata.created_at,
                "owned_by": metadata.owned_by,
                "description": metadata.description,
                "context_length": metadata.context_length,
                "family": metadata.family,
                "tags": metadata.tags,
                "tier": metadata.tier,
            }
            for metadata in self._metadata.values()
        ]

    def get_metadata(self, model_id: str) -> ModelMetadata:
        """
        Get metadata for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            ModelMetadata instance

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in self._metadata:
            raise KeyError(f"Model '{model_id}' not found in registry")
        return self._metadata[model_id]

    async def unregister_model(self, model_id: str) -> None:
        """
        Unregister a model (stub for future implementation).

        In Phase 1, this just removes from registry. Future phases will
        implement proper cleanup (handler.cleanup(), memory release, etc.).

        Args:
            model_id: Model identifier

        Raises:
            KeyError: If model_id not found
        """
        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")

            # TODO Phase 2: Call handler.cleanup() before removing
            del self._handlers[model_id]
            del self._metadata[model_id]

            logger.info(f"Unregistered model: {model_id}")

    def has_model(self, model_id: str) -> bool:
        """
        Check if a model is registered.

        Args:
            model_id: Model identifier

        Returns:
            True if model is registered, False otherwise
        """
        return model_id in self._handlers

    def get_model_count(self) -> int:
        """
        Get count of registered models.

        Returns:
            Number of registered models
        """
        return len(self._handlers)
