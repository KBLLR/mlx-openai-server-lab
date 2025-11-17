"""Phase-4 integration tests for MLX Tier-3A provider."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import argparse

from app.main import create_app
from app.core.model_registry import ModelRegistry
from app.schemas.model import ModelMetadata


@pytest.fixture
def mock_handler_with_model():
    """Create a mock handler with proper model structure."""
    handler = MagicMock()
    handler.model_path = "mlx-community/qwen2.5-7b-instruct-4bit"
    handler.warmup_done = True

    # Mock the model structure for validation
    mock_model = MagicMock()
    mock_model.model = MagicMock()  # Inner model (for corrupted weights check)
    mock_model.max_kv_size = 8192
    mock_model.model_type = "qwen"
    handler.model = mock_model

    return handler


@pytest.fixture
def mock_handler_corrupted():
    """Create a mock handler with corrupted model (for validation tests)."""
    handler = MagicMock()
    handler.model_path = "mlx-community/corrupted-model"
    handler.model = None  # Simulate corrupted/failed load
    return handler


@pytest.fixture
def mock_registry():
    """Create a mock registry for testing."""
    registry = ModelRegistry()
    return registry


@pytest.fixture
def test_app(mock_handler_with_model, mock_registry):
    """Create a test FastAPI app with mocked dependencies."""
    args = argparse.Namespace(
        model_path="mlx-community/qwen2.5-7b-instruct-4bit",
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

    with patch('app.main.create_lifespan'):
        app = create_app(args, configure_log=False)

    app.state.handler = mock_handler_with_model
    app.state.registry = mock_registry
    app.state.config = args

    return app


@pytest.fixture
def client(test_app):
    """Create a test client."""
    return TestClient(test_app)


class TestPhase4HealthEndpoint:
    """Phase-4 health endpoint tests with latency measurement."""

    def test_health_includes_latency_ms(self, client):
        """Test that health endpoint includes latency_ms field."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "latency_ms" in data
        assert isinstance(data["latency_ms"], (int, float))
        assert data["latency_ms"] >= 0

    def test_health_contract_complete(self, client):
        """Test that health endpoint returns all required Phase-4 fields."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        required_fields = [
            "status",
            "model_id",
            "model_status",
            "models_healthy",
            "warmup_enabled",
            "warmup_completed",
            "latency_ms",
        ]

        for field in required_fields:
            assert field in data, f"Missing required field: {field}"


class TestPhase4RequestIdPropagation:
    """Phase-4 requestId propagation tests."""

    def test_health_accepts_request_id_header(self, client):
        """Test that health endpoint accepts X-Request-ID header."""
        request_id = "test-request-123"
        response = client.get("/health", headers={"X-Request-ID": request_id})

        assert response.status_code == 200
        assert response.headers.get("X-Request-ID") == request_id

    def test_health_generates_request_id(self, client):
        """Test that health endpoint generates request ID if not provided."""
        response = client.get("/health")

        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) > 0


class TestPhase4DiagnosticsEndpoint:
    """Phase-4 diagnostics endpoint tests."""

    @pytest.mark.asyncio
    async def test_diagnostics_endpoint_exists(self, client, test_app):
        """Test that /internal/diagnostics endpoint exists."""
        # Register a model first
        registry = test_app.state.registry
        handler = test_app.state.handler

        await registry.register_model(
            model_id="mlx-community/qwen2.5-7b-instruct-4bit",
            handler=handler,
            model_type="lm",
            context_length=8192,
        )

        response = client.get("/internal/diagnostics")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_diagnostics_includes_all_sections(self, client, test_app):
        """Test that diagnostics includes all required sections."""
        # Register a model
        registry = test_app.state.registry
        handler = test_app.state.handler

        await registry.register_model(
            model_id="mlx-community/qwen2.5-7b-instruct-4bit",
            handler=handler,
            model_type="lm",
            context_length=8192,
        )

        response = client.get("/internal/diagnostics")
        assert response.status_code == 200

        data = response.json()

        # Check required sections
        assert "timestamp" in data
        assert "status" in data
        assert "handler_initialized" in data
        assert "registry_initialized" in data
        assert "vram" in data
        assert "models" in data
        assert "config" in data
        assert "diagnostics_latency_ms" in data

    @pytest.mark.asyncio
    async def test_diagnostics_model_stats(self, client, test_app):
        """Test that diagnostics returns per-model statistics."""
        registry = test_app.state.registry
        handler = test_app.state.handler

        await registry.register_model(
            model_id="mlx-community/qwen2.5-7b-instruct-4bit",
            handler=handler,
            model_type="lm",
            context_length=8192,
        )

        response = client.get("/internal/diagnostics")
        data = response.json()

        assert "models" in data
        assert "count" in data["models"]
        assert data["models"]["count"] == 1
        assert "stats" in data["models"]
        assert len(data["models"]["stats"]) == 1

        model_stats = data["models"]["stats"][0]
        assert "model_id" in model_stats
        assert "type" in model_stats
        assert "request_count" in model_stats


class TestPhase4CorruptedWeightsDetection:
    """Phase-4 corrupted weights detection tests."""

    @pytest.mark.asyncio
    async def test_registry_rejects_corrupted_handler(self, mock_handler_corrupted):
        """Test that registry rejects handler with corrupted weights."""
        registry = ModelRegistry()

        with pytest.raises(ValueError, match="Handler validation failed"):
            await registry.register_model(
                model_id="corrupted-model",
                handler=mock_handler_corrupted,
                model_type="lm",
            )

    @pytest.mark.asyncio
    async def test_registry_validates_handler_structure(self, mock_handler_with_model):
        """Test that registry validates handler has proper structure."""
        registry = ModelRegistry()

        # This should succeed (valid handler)
        await registry.register_model(
            model_id="valid-model",
            handler=mock_handler_with_model,
            model_type="lm",
            context_length=8192,
        )

        # Verify model was registered
        assert registry.has_model("valid-model")


class TestPhase4EmbeddingsRequestId:
    """Phase-4 embeddings endpoint requestId tests."""

    @pytest.mark.asyncio
    async def test_embeddings_response_includes_request_id(self, test_app):
        """Test that embeddings response includes request_id field."""
        # Mock the handler's generate_embeddings_response method
        mock_embeddings = [[0.1, 0.2, 0.3]]
        test_app.state.handler.generate_embeddings_response = AsyncMock(
            return_value=mock_embeddings
        )

        with TestClient(test_app) as client:
            request_id = "embed-test-123"
            response = client.post(
                "/v1/embeddings",
                json={"model": "local-embedding-model", "input": "test text"},
                headers={"X-Request-ID": request_id},
            )

            assert response.status_code == 200
            data = response.json()

            # Check response has request_id field
            assert "request_id" in data
            assert data["request_id"] == request_id


class TestPhase4IntegrationFlow:
    """Phase-4 end-to-end integration flow tests."""

    @pytest.mark.asyncio
    async def test_health_to_models_flow(self, client, test_app):
        """Test health → models discovery flow."""
        # Register a model
        registry = test_app.state.registry
        handler = test_app.state.handler

        await registry.register_model(
            model_id="mlx-community/qwen2.5-7b-instruct-4bit",
            handler=handler,
            model_type="lm",
            context_length=8192,
        )

        # Step 1: Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_data = health_response.json()
        assert health_data["models_healthy"] is True
        assert "latency_ms" in health_data

        # Step 2: List models
        models_response = client.get("/v1/models")
        assert models_response.status_code == 200
        models_data = models_response.json()
        assert models_data["object"] == "list"
        assert len(models_data["data"]) == 1

        # Step 3: Get specific model
        model_id = models_data["data"][0]["id"]
        model_response = client.get(f"/v1/models/{model_id}")
        assert model_response.status_code == 200
        model_data = model_response.json()
        assert model_data["id"] == model_id
        assert model_data["context_length"] == 8192

    @pytest.mark.asyncio
    async def test_request_id_tracking_across_endpoints(self, client, test_app):
        """Test that request ID is tracked across multiple endpoint calls."""
        request_id = "phase4-flow-test"

        # Health endpoint
        health_resp = client.get("/health", headers={"X-Request-ID": request_id})
        assert health_resp.headers.get("X-Request-ID") == request_id

        # Models endpoint
        models_resp = client.get("/v1/models", headers={"X-Request-ID": request_id})
        assert models_resp.headers.get("X-Request-ID") == request_id

        # Diagnostics endpoint
        diag_resp = client.get(
            "/internal/diagnostics", headers={"X-Request-ID": request_id}
        )
        assert diag_resp.headers.get("X-Request-ID") == request_id
