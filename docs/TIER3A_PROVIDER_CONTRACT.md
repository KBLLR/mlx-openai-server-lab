# Tier-3A MLX Provider Contract

**Document Version**: 1.0
**Date**: 2025-11-17
**Phase**: Phase-4 Fusion Orchestrator
**Provider Type**: MLX LLM Backend (Local Inference)

---

## Overview

This document specifies the **Tier-3A MLX Provider Contract** for integration with the Tier-2 Fusion Orchestrator (gen-idea-lab). The mlx-openai-server-lab provides local MLX-based model inference with OpenAI-compatible API endpoints.

### Architecture Position

```
┌─────────────────────────────────────────┐
│   Tier-2: gen-idea-lab                  │
│   (Fusion Orchestrator / Gateway)       │
└──────────────┬──────────────────────────┘
               │
               │ HTTP/JSON
               │
┌──────────────▼──────────────────────────┐
│   Tier-3A: mlx-openai-server-lab        │
│   (MLX Provider - THIS SERVICE)         │
│   - Local LLM inference                 │
│   - Embeddings generation               │
│   - OpenAI-compatible API               │
└─────────────────────────────────────────┘
```

---

## Service Information

**Base URL**: `http://localhost:8000` (configurable via `SERVER_PORT` environment variable)

**Protocol**: HTTP/1.1
**Content-Type**: `application/json`
**Character Encoding**: UTF-8

---

## 1. Health Check Endpoint

### `GET /health`

**Purpose**: Verify service and model initialization status. Used by Tier-2 for health gating.

#### Request

```http
GET /health HTTP/1.1
Host: localhost:8000
X-Request-ID: optional-request-id
```

#### Response (200 OK - Service Healthy)

```json
{
  "status": "ok",
  "model_id": "mlx-community/qwen2.5-7b-instruct-4bit",
  "model_status": "initialized",
  "models_healthy": true,
  "warmup_enabled": true,
  "warmup_completed": true,
  "latency_ms": 2.34
}
```

#### Response (503 Service Unavailable - Service Degraded)

```json
{
  "status": "ok",
  "model_id": null,
  "model_status": "uninitialized",
  "models_healthy": false,
  "warmup_enabled": null,
  "warmup_completed": null,
  "latency_ms": 1.23
}
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `status` | string | Yes | Always `"ok"` |
| `model_id` | string\|null | Yes | ID of loaded model, null if uninitialized |
| `model_status` | string | Yes | `"initialized"` or `"uninitialized"` |
| `models_healthy` | boolean | Yes | Whether all models are healthy and ready |
| `warmup_enabled` | boolean\|null | Yes | Whether KV cache warmup is enabled |
| `warmup_completed` | boolean\|null | Yes | Whether warmup has completed |
| `latency_ms` | float | Yes | Health check latency in milliseconds |

#### Headers

- **Response Header**: `X-Request-ID` - Request correlation ID (echoed or generated)

---

## 2. Model Discovery Endpoint

### `GET /v1/models`

**Purpose**: List all available models with rich metadata for dynamic discovery.

#### Request

```http
GET /v1/models HTTP/1.1
Host: localhost:8000
X-Request-ID: optional-request-id
```

#### Response (200 OK)

```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/qwen2.5-7b-instruct-4bit",
      "object": "model",
      "created": 1700234567,
      "owned_by": "local-mlx",
      "description": "Qwen language model running on MLX",
      "context_length": 8192,
      "family": "qwen",
      "tags": ["local", "chat", "quantized"],
      "tier": "3A"
    }
  ]
}
```

#### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `object` | string | Yes | Always `"list"` |
| `data` | array | Yes | Array of model objects |
| `data[].id` | string | Yes | Model identifier |
| `data[].object` | string | Yes | Always `"model"` |
| `data[].created` | integer | Yes | Unix timestamp |
| `data[].owned_by` | string | Yes | Always `"local-mlx"` |
| `data[].description` | string | Yes | Human-readable description |
| `data[].context_length` | integer\|null | Yes | Maximum context length |
| `data[].family` | string\|null | Yes | Model family (qwen, llama, gemma, etc.) |
| `data[].tags` | array | Yes | Tags (local, chat, quantized, etc.) |
| `data[].tier` | string | Yes | Always `"3A"` |

---

## 3. Chat Completions Endpoint

### `POST /v1/chat/completions`

**Purpose**: Generate chat completions using local MLX models.

#### Request

```http
POST /v1/chat/completions HTTP/1.1
Host: localhost:8000
Content-Type: application/json
X-Request-ID: optional-request-id

{
  "model": "mlx-community/qwen2.5-7b-instruct-4bit",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is MLX?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

#### Response (200 OK - Non-Streaming)

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700234567,
  "model": "mlx-community/qwen2.5-7b-instruct-4bit",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "MLX is Apple's machine learning framework..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 128,
    "total_tokens": 170
  },
  "request_id": "optional-request-id"
}
```

#### Response (200 OK - Streaming)

When `stream: true`, server sends Server-Sent Events (SSE):

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700234567,"model":"mlx-community/qwen2.5-7b-instruct-4bit","choices":[{"index":0,"delta":{"role":"assistant","content":"MLX"},"finish_reason":null}],"request_id":"optional-request-id"}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700234567,"model":"mlx-community/qwen2.5-7b-instruct-4bit","choices":[{"index":0,"delta":{"content":" is"},"finish_reason":null}],"request_id":"optional-request-id"}

data: [DONE]
```

#### Key Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string\|null | Yes | Request correlation ID for tracing |
| `choices[].message.content` | string | Yes | Generated response text |
| `usage` | object\|null | No | Token usage statistics |

---

## 4. Embeddings Endpoint

### `POST /v1/embeddings`

**Purpose**: Generate embeddings for text inputs.

#### Request

```http
POST /v1/embeddings HTTP/1.1
Host: localhost:8000
Content-Type: application/json
X-Request-ID: optional-request-id

{
  "model": "local-embedding-model",
  "input": "The quick brown fox jumps over the lazy dog",
  "encoding_format": "float"
}
```

#### Response (200 OK)

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, 0.789, ...],
      "index": 0
    }
  ],
  "model": "local-embedding-model",
  "usage": {
    "prompt_tokens": 10,
    "total_tokens": 10
  },
  "request_id": "optional-request-id"
}
```

#### Key Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `request_id` | string\|null | Yes | Request correlation ID for tracing |
| `data[].embedding` | array\|string | Yes | Embedding vector (float array or base64) |
| `encoding_format` | string | No | `"float"` (default) or `"base64"` |

---

## 5. Internal Diagnostics Endpoint

### `GET /internal/diagnostics`

**Purpose**: Comprehensive system diagnostics for observability and debugging.

#### Request

```http
GET /internal/diagnostics HTTP/1.1
Host: localhost:8000
X-Request-ID: optional-request-id
```

#### Response (200 OK)

```json
{
  "timestamp": 1700234567,
  "status": "ok",
  "handler_initialized": true,
  "registry_initialized": true,
  "vram": {
    "enabled": true,
    "total_gb": 12.5,
    "available_gb": 7.8,
    "max_gb": 32.0,
    "usage_percent": 39.1,
    "model_count": 1
  },
  "models": {
    "count": 1,
    "stats": [
      {
        "model_id": "mlx-community/qwen2.5-7b-instruct-4bit",
        "type": "lm",
        "family": "qwen",
        "request_count": 42,
        "vram_usage_gb": 4.7,
        "last_access": 1700234560,
        "created_at": 1700234500
      }
    ]
  },
  "queue": {
    "active": 0,
    "pending": 0,
    "max_concurrency": 1
  },
  "config": {
    "model_type": "lm",
    "max_concurrency": 1,
    "context_length": 8192,
    "mlx_warmup": true
  },
  "diagnostics_latency_ms": 3.45
}
```

#### Sections

| Section | Description |
|---------|-------------|
| `vram` | VRAM usage tracking (if enabled) |
| `models` | Per-model statistics and metadata |
| `queue` | Request queue status |
| `config` | Service configuration |

---

## 6. Request/Response Contract

### Request ID Propagation

All endpoints support the `X-Request-ID` header for request correlation:

1. **Client provides header**: Server echoes it in response
2. **No header provided**: Server generates UUID4 and returns it

#### Example

```
Request:  X-Request-ID: fusion-req-abc123
Response: X-Request-ID: fusion-req-abc123
```

All response schemas include optional `request_id` field in JSON body.

---

## 7. Error Handling

### Error Response Format

All errors follow OpenAI-compatible format:

```json
{
  "error": {
    "message": "Model handler not initialized",
    "type": "service_unavailable",
    "code": 503
  }
}
```

### Status Codes

| Code | Meaning | Usage |
|------|---------|-------|
| 200 | OK | Successful response |
| 400 | Bad Request | Invalid request parameters |
| 404 | Not Found | Model not found |
| 500 | Internal Server Error | Unexpected error |
| 503 | Service Unavailable | Service not initialized |

---

## 8. Model Registry Validation

### Corrupted Weights Detection

The service validates model handlers during registration:

- **Check 1**: Handler has `model` attribute
- **Check 2**: Model is not `None`
- **Check 3**: Inner model structure is valid

If validation fails:
- Model registration raises `ValueError`
- Service startup fails
- Health endpoint returns 503

---

## 9. Tier-2 Integration Guidelines

### Health Gating

Tier-2 MUST check `/health` before routing requests:

```python
health_response = requests.get(f"{MLX_URL}/health")
if health_response.status_code != 200:
    # Degrade to mlx_only mode or return error
    pass
elif not health_response.json()["models_healthy"]:
    # Handler exists but model unhealthy
    pass
```

### Request ID Propagation

Tier-2 MUST propagate `requestId` from client through to Tier-3A:

```python
headers = {"X-Request-ID": request_id}
mlx_response = requests.post(f"{MLX_URL}/v1/chat/completions",
                              json=payload,
                              headers=headers)
```

### Model Discovery

Tier-2 SHOULD cache model metadata from `/v1/models`:

```python
models = requests.get(f"{MLX_URL}/v1/models").json()["data"]
# Cache locally for dynamic routing
```

---

## 10. Provider Guarantees

### Guarantees

✅ **Strict OpenAI API Compatibility**: All endpoints match OpenAI response shapes
✅ **Request ID Tracking**: All requests/responses include correlation IDs
✅ **Health Contract**: `/health` always returns within 100ms
✅ **Model Validation**: Corrupted weights detected at startup
✅ **Metadata Richness**: Model discovery includes family, tags, context_length
✅ **Local-First**: No cloud dependencies

### Limitations

⚠️ **Single Model at Startup**: Current implementation loads one model
⚠️ **No Dynamic Loading**: Models cannot be loaded/unloaded at runtime (Phase-4 future work)
⚠️ **No Multi-GPU**: Currently single-device inference

---

## 11. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | - | Path to model (required) |
| `MODEL_TYPE` | `"lm"` | Model type (lm, multimodal, embeddings, etc.) |
| `SERVER_PORT` | `8000` | HTTP server port |
| `SERVER_HOST` | `"0.0.0.0"` | HTTP server host |
| `MAX_CONCURRENCY` | `1` | Max concurrent requests |
| `CONTEXT_LENGTH` | `32768` | Model context length |
| `MLX_WARMUP` | `true` | Enable KV cache warmup |

---

## 12. Testing Contract Compliance

Run Phase-4 integration tests:

```bash
pytest tests/test_phase4_integration.py -v
```

Tests verify:
- ✅ Health endpoint includes `latency_ms`
- ✅ Request ID propagation across all endpoints
- ✅ Diagnostics endpoint returns all sections
- ✅ Corrupted weights detection
- ✅ Embeddings response includes `request_id`

---

## 13. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-11-17 | Initial Phase-4 contract specification |

---

## Contact

For Tier-2 integration support, refer to:
- **Architecture Docs**: `docs/PHASE4_ARCHITECTURE.md`
- **API Spec**: `docs/PHASE4_API_SPEC.md`
- **Integration Tests**: `tests/test_phase4_integration.py`
