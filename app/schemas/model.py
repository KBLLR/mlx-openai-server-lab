"""Model metadata schemas for model registry."""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """
    Metadata for a registered model.

    Attributes:
        id: Unique identifier for the model
        type: Model type (lm, multimodal, embeddings, etc.)
        context_length: Maximum context length (if applicable)
        created_at: Timestamp when model was loaded
        description: Human-readable description of the model
        family: Model family (e.g., "gemma", "llama", "qwen")
        tags: List of tags for categorization (e.g., ["local", "default", "chat"])
        tier: Service tier (e.g., "3A" for this MLX server)
        capabilities: Optional dict of model capabilities
    """

    id: str = Field(..., description="Unique model identifier")
    type: str = Field(..., description="Model type (lm, multimodal, embeddings, whisper, image-generation, image-edit)")
    context_length: Optional[int] = Field(None, description="Maximum context length for language models")
    created_at: int = Field(..., description="Unix timestamp when model was loaded")
    object: str = Field(default="model", description="Object type, always 'model'")
    owned_by: str = Field(default="local-mlx", description="Model owner/organization")
    description: str = Field(default="", description="Human-readable model description")
    family: Optional[str] = Field(None, description="Model family (e.g., gemma, llama, qwen)")
    tags: List[str] = Field(default_factory=lambda: ["local"], description="Tags for categorization")
    tier: str = Field(default="3A", description="Service tier identifier")

    class Config:
        """Pydantic configuration."""
        frozen = False
