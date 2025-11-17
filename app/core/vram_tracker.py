"""VRAM tracking and management for multi-model support."""

import asyncio
from typing import Dict, Optional

from loguru import logger


class VRAMTracker:
    """
    VRAM usage tracker for multi-model management.

    Tracks VRAM usage per model and enforces global VRAM limits.
    Implements simple monitoring without MLX-specific calls for now.

    Attributes:
        max_vram_gb: Maximum total VRAM allowed
        _usage: Dict mapping model_id to VRAM usage in GB
        _lock: Async lock for thread-safe operations
    """

    def __init__(self, max_vram_gb: float = 32.0):
        """
        Initialize VRAM tracker.

        Args:
            max_vram_gb: Maximum VRAM limit in GB
        """
        self.max_vram_gb = max_vram_gb
        self._usage: Dict[str, float] = {}
        self._lock = asyncio.Lock()
        logger.info(f"VRAM tracker initialized with max {max_vram_gb}GB")

    async def register_model(self, model_id: str, vram_usage_gb: float) -> None:
        """
        Register a model's VRAM usage.

        Args:
            model_id: Model identifier
            vram_usage_gb: VRAM usage in GB

        Raises:
            ValueError: If VRAM limit would be exceeded
        """
        async with self._lock:
            current_total = self.get_total_usage()
            new_total = current_total + vram_usage_gb

            if new_total > self.max_vram_gb:
                raise ValueError(
                    f"VRAM limit exceeded: {new_total:.2f}GB > {self.max_vram_gb}GB "
                    f"(current: {current_total:.2f}GB, requested: {vram_usage_gb:.2f}GB)"
                )

            self._usage[model_id] = vram_usage_gb
            logger.info(
                f"Registered model '{model_id}' VRAM usage: {vram_usage_gb:.2f}GB "
                f"(total: {new_total:.2f}GB / {self.max_vram_gb}GB)"
            )

    async def unregister_model(self, model_id: str) -> float:
        """
        Unregister a model and free its VRAM.

        Args:
            model_id: Model identifier

        Returns:
            VRAM freed in GB

        Raises:
            KeyError: If model not found
        """
        async with self._lock:
            if model_id not in self._usage:
                raise KeyError(f"Model '{model_id}' not found in VRAM tracker")

            vram_freed = self._usage.pop(model_id)
            new_total = self.get_total_usage()

            logger.info(
                f"Unregistered model '{model_id}', freed {vram_freed:.2f}GB "
                f"(total: {new_total:.2f}GB / {self.max_vram_gb}GB)"
            )

            return vram_freed

    def get_total_usage(self) -> float:
        """
        Get total VRAM usage across all models.

        Returns:
            Total VRAM usage in GB
        """
        return sum(self._usage.values())

    def get_model_usage(self, model_id: str) -> Optional[float]:
        """
        Get VRAM usage for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            VRAM usage in GB, or None if model not found
        """
        return self._usage.get(model_id)

    def get_available_vram(self) -> float:
        """
        Get available VRAM.

        Returns:
            Available VRAM in GB
        """
        return max(0.0, self.max_vram_gb - self.get_total_usage())

    def can_fit_model(self, vram_required_gb: float) -> bool:
        """
        Check if a model can fit in available VRAM.

        Args:
            vram_required_gb: Required VRAM in GB

        Returns:
            True if model can fit, False otherwise
        """
        return self.get_available_vram() >= vram_required_gb

    def get_usage_summary(self) -> Dict[str, float]:
        """
        Get VRAM usage summary.

        Returns:
            Dict with usage statistics
        """
        total = self.get_total_usage()
        available = self.get_available_vram()
        usage_percent = (total / self.max_vram_gb) * 100 if self.max_vram_gb > 0 else 0

        return {
            "total_gb": total,
            "available_gb": available,
            "max_gb": self.max_vram_gb,
            "usage_percent": usage_percent,
            "model_count": len(self._usage),
        }
