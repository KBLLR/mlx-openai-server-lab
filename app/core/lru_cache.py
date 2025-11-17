"""LRU cache for model eviction policy."""

import asyncio
import time
from collections import OrderedDict
from typing import List, Optional

from loguru import logger


class LRUModelCache:
    """
    LRU (Least Recently Used) cache for model eviction.

    Tracks model access times and provides LRU ordering for eviction decisions.

    Attributes:
        _access_times: OrderedDict mapping model_id to last access timestamp
        _lock: Async lock for thread-safe operations
    """

    def __init__(self):
        """Initialize LRU cache."""
        self._access_times: OrderedDict[str, float] = OrderedDict()
        self._lock = asyncio.Lock()
        logger.info("LRU model cache initialized")

    async def access_model(self, model_id: str) -> None:
        """
        Record model access (move to end of LRU queue).

        Args:
            model_id: Model identifier
        """
        async with self._lock:
            # Remove if exists (to reorder)
            if model_id in self._access_times:
                del self._access_times[model_id]

            # Add to end (most recently used)
            self._access_times[model_id] = time.time()

    async def remove_model(self, model_id: str) -> None:
        """
        Remove model from cache.

        Args:
            model_id: Model identifier
        """
        async with self._lock:
            if model_id in self._access_times:
                del self._access_times[model_id]
                logger.debug(f"Removed model '{model_id}' from LRU cache")

    def get_least_recently_used(self) -> Optional[str]:
        """
        Get the least recently used model.

        Returns:
            Model ID of LRU model, or None if cache is empty
        """
        if not self._access_times:
            return None

        # First item in OrderedDict is least recently used
        return next(iter(self._access_times))

    def get_lru_models(self, count: int) -> List[str]:
        """
        Get N least recently used models.

        Args:
            count: Number of models to return

        Returns:
            List of model IDs in LRU order
        """
        return list(self._access_times.keys())[:count]

    def get_access_time(self, model_id: str) -> Optional[float]:
        """
        Get last access time for a model.

        Args:
            model_id: Model identifier

        Returns:
            Unix timestamp of last access, or None if not found
        """
        return self._access_times.get(model_id)

    def get_model_count(self) -> int:
        """
        Get number of models in cache.

        Returns:
            Number of models
        """
        return len(self._access_times)

    def clear(self) -> None:
        """Clear all models from cache."""
        self._access_times.clear()
        logger.info("LRU cache cleared")
