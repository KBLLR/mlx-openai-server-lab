"""Tests for API endpoints."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import argparse

from app.main import create_app
from app.core.model_registry import ModelRegistry
from app.schemas.model import ModelMetadata


@pytest.fixture
def mock_handler():
    """Create a mock handler for testing."""
    handler = MagicMock()
    handler.model_path = "mlx-community/gemma-3-4b-it-4bit"
    handler.warmup_done = True
    return handler


@pytest.fixture
def mock_registry():
    """Create a mock registry with test model metadata."""
    registry = ModelRegistry()
    # We'll populate this in tests as needed
    return registry


@pytest.fixture
def test_app(mock_handler, mock_registry):
    """Create a test FastAPI app with mocked dependencies."""
    # Create minimal args
    args = argparse.Namespace(
        model_path="mlx-community/gemma-3-4b-it-4bit",
        model_type="lm",
        context_length=8192,
        port=8000,
        host="0.0.0.0",
        max_concurrency=1,
        queue_timeout=300,
        queue_size=100,
        quantize=8,
        config_name=None,
        lora_paths=None,
        lora_scales=None,
        disable_auto_resize=False,
        log_file=None,
        no_log_file=True,
        log_level="ERROR",
        enable_auto_tool_choice=False,
        tool_call_parser=None,
        reasoning_parser=None,
        trust_remote_code=False,
        mlx_warmup=True,
    )

    # Create app without lifespan (we'll mock the state)
    with patch('app.main.create_lifespan'):
        app = create_app(args, configure_log=False)

    # Manually set the state
    app.state.handler = mock_handler
    app.state.registry = mock_registry
    app.state.config = args

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_check_success(self, client):
        """Test health check with initialized handler."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"
        assert data["model_status"] == "initialized"
        assert data["model_id"] == "mlx-community/gemma-3-4b-it-4bit"
        assert data["models_healthy"] is True
        assert "warmup_enabled" in data
        assert "warmup_completed" in data

    def test_health_check_no_handler(self, test_app):
        """Test health check when handler is not initialized."""
        # Remove handler
        test_app.state.handler = None

        with TestClient(test_app) as client:
            response = client.get("/health")
            assert response.status_code == 503

            data = response.json()
            assert data["model_status"] == "uninitialized"
            assert data["models_healthy"] is False


class TestModelsEndpoint:
    """Tests for /v1/models endpoints."""

    @pytest.mark.asyncio
    async def test_list_models_success(self, client, test_app):
        """Test listing models with rich metadata."""
        # Register a test model in the registry
        registry = test_app.state.registry
        handler = test_app.state.handler

        await registry.register_model(
            model_id="mlx-community/gemma-3-4b-it-4bit",
            handler=handler,
            model_type="lm",
            context_length=8192,
        )

        response = client.get("/v1/models")
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1

        model = data["data"][0]
        assert model["id"] == "mlx-community/gemma-3-4b-it-4bit"
        assert model["object"] == "model"
        assert model["owned_by"] == "local-mlx"
        assert "description" in model
        assert "context_length" in model
        assert model["context_length"] == 8192
        assert "family" in model
        assert model["family"] == "gemma"
        assert "tags" in model
        assert "local" in model["tags"]
        assert "chat" in model["tags"]
        assert "tier" in model
        assert model["tier"] == "3A"

    @pytest.mark.asyncio
    async def test_get_model_by_id_success(self, client, test_app):
        """Test getting a specific model by ID."""
        # Register a test model
        registry = test_app.state.registry
        handler = test_app.state.handler

        await registry.register_model(
            model_id="mlx-community/gemma-3-4b-it-4bit",
            handler=handler,
            model_type="lm",
            context_length=8192,
        )

        response = client.get("/v1/models/mlx-community/gemma-3-4b-it-4bit")
        assert response.status_code == 200

        model = response.json()
        assert model["id"] == "mlx-community/gemma-3-4b-it-4bit"
        assert model["object"] == "model"
        assert model["context_length"] == 8192
        assert model["family"] == "gemma"
        assert "tags" in model
        assert model["tier"] == "3A"

    def test_get_model_not_found(self, client):
        """Test getting a non-existent model."""
        response = client.get("/v1/models/nonexistent-model")
        assert response.status_code == 404

        data = response.json()
        assert "error" in data


class TestModelMetadataGeneration:
    """Tests for model metadata generation logic."""

    @pytest.mark.asyncio
    async def test_family_inference(self, mock_registry):
        """Test that model family is correctly inferred."""
        test_cases = [
            ("mlx-community/gemma-3-4b-it-4bit", "gemma"),
            ("mlx-community/Meta-Llama-3.1-8B-Instruct", "llama"),
            ("mlx-community/Qwen2.5-7B-Instruct", "qwen"),
            ("mlx-community/Phi-3-mini-4k-instruct", "phi"),
            ("mlx-community/Mistral-7B-Instruct-v0.2", "mistral"),
        ]

        for model_id, expected_family in test_cases:
            inferred_family = mock_registry._infer_model_family(model_id)
            assert inferred_family == expected_family, f"Failed for {model_id}"

    @pytest.mark.asyncio
    async def test_tags_generation(self, mock_registry):
        """Test that appropriate tags are generated."""
        # Test LM model with quantization
        tags = mock_registry._generate_model_tags("lm", "mlx-community/gemma-3-4b-it-4bit")
        assert "local" in tags
        assert "chat" in tags
        assert "quantized" in tags

        # Test embeddings model
        tags = mock_registry._generate_model_tags("embeddings", "mlx-community/bge-base-en-v1.5")
        assert "local" in tags
        assert "embeddings" in tags

        # Test image generation model
        tags = mock_registry._generate_model_tags("image-generation", "flux-schnell")
        assert "local" in tags
        assert "image" in tags

    @pytest.mark.asyncio
    async def test_description_generation(self, mock_registry):
        """Test that model descriptions are generated correctly."""
        description = mock_registry._generate_model_description(
            "mlx-community/gemma-3-4b-it-4bit",
            "lm",
            "gemma"
        )
        assert "Gemma" in description
        assert "language model" in description
        assert "MLX" in description


# Note: Full integration tests for chat/completions and embeddings would require
# actual model loading, which is expensive. These tests focus on the API contract
# and metadata handling. For smoke tests with real models, run manual tests.
