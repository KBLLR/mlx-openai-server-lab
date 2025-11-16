# Tier 2 Integration Guide

This document describes how `mlx-openai-server` (Tier 3A) integrates with `gen-idea-lab` (Tier 2) for dynamic model discovery and request tracking.

## Overview

The MLX OpenAI Server acts as **Tier 3A** in the MLX-first stack, providing local OpenAI-compatible LLM endpoints that Tier 2 services can discover and use dynamically.

## Architecture

```
┌─────────────────┐
│  gen-idea-lab   │  Tier 2: Idea generation & orchestration
│    (Tier 2)     │  - Dynamic model discovery via /v1/models
└────────┬────────┘  - Health gating via /health
         │           - Request tracking via X-Request-ID
         │
         ▼
┌─────────────────┐
│ mlx-openai-     │  Tier 3A: Local MLX LLM server
│    server       │  - OpenAI-compatible endpoints
│   (Tier 3A)     │  - Rich model metadata
└─────────────────┘  - KV cache warmup
                     - Latency observability
```

## Key Features for Tier 2

### 1. Rich Model Discovery

#### GET /v1/models

Returns a list of available models with metadata suitable for UI-driven discovery:

**Response Fields:**
- `id` (string): Model identifier (e.g., `mlx-community/gemma-3-4b-it-4bit`)
- `object` (string): Always `"model"`
- `created` (int): Unix timestamp when model was loaded
- `owned_by` (string): Always `"local-mlx"`
- `description` (string): Human-readable description
- `context_length` (int): Maximum context window (e.g., `8192`)
- `family` (string): Model family (e.g., `"gemma"`, `"llama"`, `"qwen"`)
- `tags` (array): Tags for categorization (e.g., `["local", "chat", "quantized"]`)
- `tier` (string): Service tier, always `"3A"`

**Example Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/gemma-3-4b-it-4bit",
      "object": "model",
      "created": 1700000000,
      "owned_by": "local-mlx",
      "description": "Gemma language model running on MLX",
      "context_length": 8192,
      "family": "gemma",
      "tags": ["local", "chat", "quantized"],
      "tier": "3A"
    }
  ]
}
```

#### GET /v1/models/{id}

Returns detailed information about a specific model.

**Example:**
```bash
curl http://localhost:8000/v1/models/mlx-community/gemma-3-4b-it-4bit
```

**Response:**
```json
{
  "id": "mlx-community/gemma-3-4b-it-4bit",
  "object": "model",
  "created": 1700000000,
  "owned_by": "local-mlx",
  "description": "Gemma language model running on MLX",
  "context_length": 8192,
  "family": "gemma",
  "tags": ["local", "chat", "quantized"],
  "tier": "3A"
}
```

### 2. Request ID Propagation

The server supports `X-Request-ID` for end-to-end request tracking:

**Request:**
```bash
curl -H "X-Request-ID: req-tier2-abc123" \
  http://localhost:8000/v1/chat/completions \
  -d '{"model": "gemma", "messages": [...]}'
```

**Behavior:**
- If `X-Request-ID` header is provided, it's used for logging and returned in response
- If not provided, a UUID is generated automatically
- Request ID is logged with all operations for traceability
- Latency metrics are associated with the request ID in logs

**Response Headers:**
```
X-Request-ID: req-tier2-abc123
X-Process-Time: 0.234
```

**Logs:**
```
Request started: POST /v1/chat/completions [request_id=req-tier2-abc123]
Processing text request [request_id=req-tier2-abc123]
Request completed: POST /v1/chat/completions status=200 duration=0.234s [request_id=req-tier2-abc123]
```

### 3. Health Check with Warmup Status

#### GET /health

Returns comprehensive health information for Tier 2 to gate on:

**Response Fields:**
- `status` (string): Always `"ok"` if server is running
- `model_id` (string): ID of loaded model
- `model_status` (string): `"initialized"` or `"uninitialized"`
- `models_healthy` (bool): Whether all models are ready
- `warmup_enabled` (bool): Whether KV cache warmup is configured
- `warmup_completed` (bool): Whether warmup has finished

**Example Response (Ready):**
```json
{
  "status": "ok",
  "model_id": "mlx-community/gemma-3-4b-it-4bit",
  "model_status": "initialized",
  "models_healthy": true,
  "warmup_enabled": true,
  "warmup_completed": true
}
```

**Example Response (Not Ready):**
```json
{
  "status": "ok",
  "model_id": null,
  "model_status": "uninitialized",
  "models_healthy": false,
  "warmup_enabled": null,
  "warmup_completed": null
}
```

**HTTP Status Codes:**
- `200`: Server and models are healthy
- `503`: Server is running but models not initialized

**Tier 2 Health Gating Logic:**
```python
import requests

def wait_for_mlx_server(url: str, timeout: int = 60):
    """Wait for MLX server to be fully ready."""
    import time
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/health")
            if response.status_code == 200:
                data = response.json()
                if data.get("models_healthy"):
                    # Additional check for warmup if needed
                    if data.get("warmup_enabled"):
                        if data.get("warmup_completed"):
                            return True
                    else:
                        return True
            time.sleep(2)
        except requests.RequestException:
            time.sleep(2)

    raise TimeoutError("MLX server not ready")
```

### 4. KV Cache Warmup Configuration

The server supports KV cache warmup to reduce first-token latency:

**Environment Variable:**
```bash
export MLX_WARMUP=true  # default
# or
export MLX_WARMUP=false  # disable warmup
```

**Command Line:**
```bash
python -m app.main \
  --model-path mlx-community/gemma-3-4b-it-4bit \
  --model-type lm \
  --mlx-warmup true
```

**Behavior:**
- When enabled (default), the server performs a dummy forward pass on startup
- This pre-populates the KV cache and reduces latency for the first real request
- Warmup status is visible in `/health` endpoint
- Only applies to `lm` and `multimodal` model types

## Configuration for Tier 2

### Recommended Server Configuration

For optimal Tier 2 integration:

```bash
# Start MLX server with warmup and appropriate context length
python -m app.main \
  --model-path mlx-community/gemma-3-4b-it-4bit \
  --model-type lm \
  --context-length 8192 \
  --mlx-warmup true \
  --max-concurrency 1 \
  --queue-timeout 300 \
  --port 8000 \
  --log-level INFO
```

### Environment Variables

```bash
# Server configuration
export MODEL_PATH=mlx-community/gemma-3-4b-it-4bit
export MODEL_TYPE=lm
export CONTEXT_LENGTH=8192
export MLX_WARMUP=true

# Server binding
export SERVER_HOST=0.0.0.0
export SERVER_PORT=8000

# Queue configuration
export MAX_CONCURRENCY=1
export QUEUE_TIMEOUT=300
export QUEUE_SIZE=100

# Logging
export LOG_LEVEL=INFO
```

## Integration Checklist

When integrating Tier 2 with Tier 3A, ensure:

- [ ] `/v1/models` returns rich metadata with all required fields
- [ ] `/health` endpoint shows `models_healthy: true` before sending requests
- [ ] `X-Request-ID` is propagated from Tier 2 for request tracking
- [ ] Warmup is enabled and completed (check `/health`)
- [ ] Model context length matches Tier 2 expectations
- [ ] Appropriate tags are used for model filtering (e.g., `["chat", "local"]`)
- [ ] Latency metrics are monitored via `X-Process-Time` header and logs

## Observability

### Request Tracking

All requests are logged with:
- Request ID (from `X-Request-ID` header or generated)
- Model ID
- Endpoint
- Duration
- Status code

Example log format:
```
2025-01-16 10:30:45 | INFO | Request started: POST /v1/chat/completions [request_id=req-123]
2025-01-16 10:30:45 | INFO | Processing text request [request_id=req-123]
2025-01-16 10:30:45 | INFO | Request completed: POST /v1/chat/completions status=200 duration=0.234s [request_id=req-123]
```

### Latency Monitoring

Response headers include timing information:
- `X-Request-ID`: Correlation ID for tracking
- `X-Process-Time`: Total request processing time in seconds

### Health Monitoring

Tier 2 should:
1. Poll `/health` on startup until `models_healthy: true`
2. Periodically check `/health` during operation
3. Gracefully handle 503 responses (server not ready)
4. Monitor `warmup_completed` if warmup is expected

## Common Integration Patterns

### Pattern 1: Dynamic Model Registry

Tier 2 can query `/v1/models` on startup to build a dynamic registry:

```python
import requests

def discover_models(mlx_server_url: str):
    """Discover available models from MLX server."""
    response = requests.get(f"{mlx_server_url}/v1/models")
    response.raise_for_status()

    models = response.json()["data"]

    # Filter by tags
    chat_models = [m for m in models if "chat" in m.get("tags", [])]

    # Build registry
    registry = {
        model["id"]: {
            "family": model.get("family"),
            "context_length": model.get("context_length"),
            "description": model.get("description"),
            "tier": model.get("tier"),
        }
        for model in chat_models
    }

    return registry
```

### Pattern 2: Health-Gated Startup

Tier 2 should wait for Tier 3A to be ready:

```python
import time
import requests

def wait_for_mlx_ready(url: str, timeout: int = 120):
    """Wait for MLX server with warmup to be ready."""
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()

                # Check all health criteria
                if (health.get("models_healthy") and
                    health.get("model_status") == "initialized"):

                    # If warmup is enabled, ensure it's done
                    if health.get("warmup_enabled"):
                        if not health.get("warmup_completed"):
                            print("Waiting for KV cache warmup...")
                            time.sleep(2)
                            continue

                    print(f"MLX server ready: {health['model_id']}")
                    return health

        except requests.RequestException as e:
            print(f"Health check failed: {e}")

        time.sleep(2)

    raise TimeoutError("MLX server not ready within timeout")
```

### Pattern 3: Request ID Propagation

Tier 2 should propagate request IDs for tracing:

```python
import requests
import uuid

def call_mlx_with_tracking(url: str, request_id: str = None):
    """Call MLX server with request tracking."""
    if request_id is None:
        request_id = f"tier2-{uuid.uuid4()}"

    headers = {
        "X-Request-ID": request_id,
        "Content-Type": "application/json"
    }

    response = requests.post(
        f"{url}/v1/chat/completions",
        json={
            "model": "gemma",
            "messages": [{"role": "user", "content": "Hello"}]
        },
        headers=headers
    )

    # Extract timing from response
    process_time = response.headers.get("X-Process-Time")
    returned_request_id = response.headers.get("X-Request-ID")

    print(f"Request {returned_request_id} completed in {process_time}s")

    return response.json()
```

## Troubleshooting

### Models Not Showing Up

Check that:
1. Server is started with correct model path
2. `/health` shows `models_healthy: true`
3. `/v1/models` returns non-empty list

### Warmup Not Completing

If `warmup_completed` stays `false`:
1. Check server logs for errors
2. Ensure enough memory for model + cache
3. Try disabling warmup with `--mlx-warmup false`

### Request IDs Not Propagating

Ensure:
1. `X-Request-ID` header is set in Tier 2 requests
2. Check server logs show the expected request ID
3. Verify response headers include `X-Request-ID`

## Summary

The MLX OpenAI Server provides:

✅ **Rich model metadata** via `/v1/models` for dynamic discovery
✅ **Request ID propagation** via `X-Request-ID` header
✅ **Health gating** via `/health` with warmup status
✅ **Latency observability** via `X-Process-Time` and logs
✅ **Configurable warmup** for optimal first-token latency
✅ **Strictly local** - no cloud calls, all processing on-device

This enables Tier 2 (gen-idea-lab) to dynamically discover, health-check, and interact with local MLX models through a stable OpenAI-compatible interface.
