"""Model registry for managing multiple model handlers."""

import asyncio
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from app.core.lru_cache import LRUModelCache
from app.core.vram_tracker import VRAMTracker
from app.schemas.model import ModelMetadata


class ModelRegistry:
    """
    Registry for managing model handlers.

    Maintains a thread-safe registry of loaded models and their handlers.
    Phase-4 enhancement: Supports multiple concurrent models with VRAM
    management and LRU eviction.

    Attributes:
        _handlers: Dict mapping model_id to handler instance
        _metadata: Dict mapping model_id to ModelMetadata
        _lock: Async lock for thread-safe operations
        _vram_tracker: VRAM usage tracker
        _lru_cache: LRU cache for model eviction
        _max_models: Maximum number of models to keep loaded
        _request_counts: Dict mapping model_id to request count
    """

    def __init__(
        self,
        max_vram_gb: float = 32.0,
        max_models: Optional[int] = None,
        enable_vram_tracking: bool = True,
    ):
        """
        Initialize model registry.

        Args:
            max_vram_gb: Maximum VRAM limit in GB
            max_models: Maximum number of models to keep loaded (None = unlimited)
            enable_vram_tracking: Enable VRAM tracking and eviction
        """
        self._handlers: Dict[str, Any] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._lock = asyncio.Lock()
        self._request_counts: Dict[str, int] = {}

        # Phase-4 enhancements
        self._vram_tracker = VRAMTracker(max_vram_gb) if enable_vram_tracking else None
        self._lru_cache = LRUModelCache()
        self._max_models = max_models
        self._enable_vram_tracking = enable_vram_tracking

        logger.info(
            f"Model registry initialized (max_vram={max_vram_gb}GB, "
            f"max_models={max_models}, vram_tracking={enable_vram_tracking})"
        )

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
        vram_usage_gb: float = 0.0,
        capabilities: Optional[List[str]] = None,
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
            vram_usage_gb: VRAM usage in GB (for tracking)
            capabilities: Optional list of model capabilities

        Raises:
            ValueError: If model_id already registered or VRAM limit exceeded
        """
        async with self._lock:
            if model_id in self._handlers:
                raise ValueError(f"Model '{model_id}' is already registered")

            # Check if we need to evict models
            if self._max_models is not None and len(self._handlers) >= self._max_models:
                await self._evict_least_used_model()

            # Check VRAM and evict if needed
            if self._enable_vram_tracking and self._vram_tracker:
                while not self._vram_tracker.can_fit_model(vram_usage_gb):
                    evicted = await self._evict_least_used_model()
                    if evicted is None:
                        raise ValueError(
                            f"Cannot fit model (requires {vram_usage_gb:.2f}GB, "
                            f"available: {self._vram_tracker.get_available_vram():.2f}GB)"
                        )

            # Infer family if not provided
            if family is None:
                family = self._infer_model_family(model_id)

            # Generate description if not provided
            if description is None:
                description = self._generate_model_description(model_id, model_type, family)

            # Generate tags if not provided
            if tags is None:
                tags = self._generate_model_tags(model_type, model_id)

            # Generate capabilities if not provided
            if capabilities is None:
                capabilities = self._generate_model_capabilities(model_type)

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
            self._request_counts[model_id] = 0

            # Track VRAM
            if self._enable_vram_tracking and self._vram_tracker:
                await self._vram_tracker.register_model(model_id, vram_usage_gb)

            # Update LRU cache
            await self._lru_cache.access_model(model_id)

            logger.info(
                f"Registered model: {model_id} (type={model_type}, "
                f"family={family}, context_length={context_length}, "
                f"vram={vram_usage_gb:.2f}GB)"
            )

    def _generate_model_capabilities(self, model_type: str) -> List[str]:
        """
        Generate model capabilities based on model type.

        Args:
            model_type: Model type

        Returns:
            List of capabilities
        """
        capabilities_map = {
            "lm": ["chat", "embeddings", "text"],
            "multimodal": ["chat", "vision", "audio", "embeddings"],
            "embeddings": ["embeddings"],
            "whisper": ["audio", "transcription"],
            "image-generation": ["image_generation"],
            "image-edit": ["image_editing"],
        }
        return capabilities_map.get(model_type, [])

    async def _evict_least_used_model(self) -> Optional[str]:
        """
        Evict the least recently used model.

        Returns:
            Evicted model ID, or None if no model to evict

        Note:
            Must be called with self._lock held
        """
        lru_model = self._lru_cache.get_least_recently_used()
        if lru_model is None:
            logger.warning("No model to evict (registry empty)")
            return None

        logger.info(f"Evicting least recently used model: {lru_model}")
        await self.unregister_model(lru_model)
        return lru_model

    async def get_handler(self, model_id: str) -> Any:
        """
        Get handler for a specific model and update access time.

        Args:
            model_id: Model identifier

        Returns:
            Handler instance

        Raises:
            KeyError: If model_id not found
        """
        if model_id not in self._handlers:
            raise KeyError(f"Model '{model_id}' not found in registry")

        # Update LRU cache and request count
        await self._lru_cache.access_model(model_id)
        if model_id in self._request_counts:
            self._request_counts[model_id] += 1

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

    async def unregister_model(self, model_id: str) -> float:
        """
        Unregister a model and clean up resources.

        Phase-4 enhancement: Properly cleanup VRAM and LRU cache.

        Args:
            model_id: Model identifier

        Returns:
            VRAM freed in GB

        Raises:
            KeyError: If model_id not found
        """
        async with self._lock:
            if model_id not in self._handlers:
                raise KeyError(f"Model '{model_id}' not found in registry")

            vram_freed = 0.0

            # Unregister from VRAM tracker
            if self._enable_vram_tracking and self._vram_tracker:
                try:
                    vram_freed = await self._vram_tracker.unregister_model(model_id)
                except KeyError:
                    pass  # Model not in VRAM tracker

            # Remove from LRU cache
            await self._lru_cache.remove_model(model_id)

            # Remove from registries
            # TODO: Call handler.cleanup() if method exists
            del self._handlers[model_id]
            del self._metadata[model_id]
            if model_id in self._request_counts:
                del self._request_counts[model_id]

            logger.info(f"Unregistered model: {model_id} (freed {vram_freed:.2f}GB)")
            return vram_freed

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

    async def get_models_by_capability(self, capability: str) -> List[str]:
        """
        Get models that support a specific capability.

        Args:
            capability: Capability to filter by

        Returns:
            List of model IDs with the capability
        """
        models = []
        for model_id, metadata in self._metadata.items():
            capabilities = self._generate_model_capabilities(metadata.type)
            if capability in capabilities:
                models.append(model_id)
        return models

    def get_vram_usage_summary(self) -> Dict[str, float]:
        """
        Get VRAM usage summary.

        Returns:
            Dict with VRAM statistics
        """
        if not self._enable_vram_tracking or not self._vram_tracker:
            return {
                "enabled": False,
                "total_gb": 0.0,
                "available_gb": 0.0,
                "max_gb": 0.0,
                "usage_percent": 0.0,
                "model_count": 0,
            }

        summary = self._vram_tracker.get_usage_summary()
        summary["enabled"] = True
        return summary

    def get_model_stats(self, model_id: str) -> Dict[str, Any]:
        """
        Get usage statistics for a model.

        Args:
            model_id: Model identifier

        Returns:
            Dict with model statistics

        Raises:
            KeyError: If model not found
        """
        if model_id not in self._metadata:
            raise KeyError(f"Model '{model_id}' not found in registry")

        metadata = self._metadata[model_id]
        vram_usage = 0.0
        if self._enable_vram_tracking and self._vram_tracker:
            vram_usage = self._vram_tracker.get_model_usage(model_id) or 0.0

        last_access = self._lru_cache.get_access_time(model_id)

        return {
            "model_id": model_id,
            "type": metadata.type,
            "family": metadata.family,
            "request_count": self._request_counts.get(model_id, 0),
            "vram_usage_gb": vram_usage,
            "last_access": last_access,
            "created_at": metadata.created_at,
        }
