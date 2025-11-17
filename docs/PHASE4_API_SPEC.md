# Phase-4 API Specification

**Document Version**: 1.0
**Date**: 2025-11-17
**Status**: API Definition
**Base URL**: `http://localhost:8000`

---

## Table of Contents

1. [Overview](#1-overview)
2. [Model Management API](#2-model-management-api)
3. [Fusion Orchestration API](#3-fusion-orchestration-api)
4. [RAG Provider API](#4-rag-provider-api)
5. [MCP Server API](#5-mcp-server-api)
6. [Observability & Diagnostics API](#6-observability--diagnostics-api)
7. [Error Handling](#7-error-handling)
8. [Rate Limiting](#8-rate-limiting)
9. [Versioning](#9-versioning)

---

## 1. Overview

### API Groups

| Group | Base Path | Description |
|-------|-----------|-------------|
| **Model Management** | `/v1/models` | Load, unload, and manage models |
| **Fusion Orchestration** | `/api/fusion` | Multi-model workflow execution |
| **RAG Provider** | `/api/rag` | Document ingestion and retrieval-augmented generation |
| **MCP Server** | `/api/mcp` | Job tracking and state management |
| **Diagnostics** | `/internal` | System diagnostics and metrics |

### Authentication

**Current**: No authentication (local development)
**Future**: API key or JWT token authentication

### Request/Response Format

- **Content-Type**: `application/json`
- **Character Encoding**: UTF-8
- **Date Format**: ISO 8601 (`2025-11-17T10:00:00Z`)

---

## 2. Model Management API

### 2.1 Load Model

Load a new model into the multi-model registry.

**Endpoint**: `POST /v1/models/load`

**Request Body**:
```json
{
  "model_path": "mlx-community/qwen2.5-7b-instruct-4bit",
  "model_type": "lm",
  "model_id": "qwen-7b",
  "context_length": 8192,
  "priority": "normal",
  "metadata": {
    "description": "Qwen 2.5 7B Instruct quantized to 4-bit"
  }
}
```

**Request Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_path` | string | Yes | HuggingFace model path or local path |
| `model_type` | string | Yes | Model type: `lm`, `multimodal`, `embeddings`, `whisper`, `image-generation`, `image-edit` |
| `model_id` | string | No | Custom model ID (auto-generated if not provided) |
| `context_length` | integer | No | Context length (default: model's default) |
| `priority` | string | No | Loading priority: `low`, `normal`, `high` (default: `normal`) |
| `metadata` | object | No | Additional metadata |

**Response** (200 OK):
```json
{
  "model_id": "qwen-7b",
  "status": "loaded",
  "vram_usage_gb": 4.2,
  "load_time_ms": 1234,
  "metadata": {
    "id": "qwen-7b",
    "family": "qwen",
    "type": "lm",
    "context_length": 8192,
    "tags": ["local", "chat", "quantized"],
    "tier": "3A",
    "created_at": 1700234567
  }
}
```

**Error Responses**:
- `400 Bad Request`: Invalid model_path or model_type
- `409 Conflict`: Model ID already exists
- `503 Service Unavailable`: Insufficient VRAM (may trigger eviction)

---

### 2.2 Unload Model

Unload a model from the registry to free VRAM.

**Endpoint**: `DELETE /v1/models/{model_id}/unload`

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `model_id` | string | Model identifier |

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force` | boolean | `false` | Force unload even if model is in use |

**Response** (200 OK):
```json
{
  "model_id": "qwen-7b",
  "status": "unloaded",
  "vram_freed_gb": 4.2,
  "unload_time_ms": 156
}
```

**Error Responses**:
- `404 Not Found`: Model ID not found
- `409 Conflict`: Model is in use (unless `force=true`)

---

### 2.3 Get Model Info

Retrieve detailed information about a loaded model.

**Endpoint**: `GET /v1/models/{model_id}/info`

**Response** (200 OK):
```json
{
  "model_id": "qwen-7b",
  "status": "loaded",
  "metadata": {
    "id": "qwen-7b",
    "family": "qwen",
    "type": "lm",
    "context_length": 8192,
    "tags": ["local", "chat", "quantized"],
    "tier": "3A"
  },
  "vram_usage_gb": 4.2,
  "load_time": "2025-11-17T10:00:00Z",
  "last_used": "2025-11-17T10:15:00Z",
  "request_count": 42
}
```

---

### 2.4 Get Model Stats

Get usage statistics for a model.

**Endpoint**: `GET /v1/models/{model_id}/stats`

**Response** (200 OK):
```json
{
  "model_id": "qwen-7b",
  "request_count": 42,
  "total_tokens_processed": 12345,
  "avg_latency_ms": 234.5,
  "p50_latency_ms": 210,
  "p95_latency_ms": 450,
  "p99_latency_ms": 600,
  "error_count": 2,
  "last_error": "2025-11-17T10:05:00Z",
  "vram_usage_gb": 4.2
}
```

---

### 2.5 List Loaded Models

List all currently loaded models.

**Endpoint**: `GET /v1/models`

**Response** (200 OK):
```json
{
  "object": "list",
  "data": [
    {
      "id": "qwen-7b",
      "object": "model",
      "created": 1700234567,
      "owned_by": "local-mlx",
      "family": "qwen",
      "type": "lm",
      "context_length": 8192,
      "tags": ["local", "chat", "quantized"],
      "tier": "3A",
      "vram_usage_gb": 4.2,
      "status": "loaded"
    },
    {
      "id": "embedding-model",
      "object": "model",
      "created": 1700234600,
      "owned_by": "local-mlx",
      "type": "embeddings",
      "tags": ["local", "embeddings"],
      "tier": "3A",
      "vram_usage_gb": 0.8,
      "status": "loaded"
    }
  ]
}
```

---

### 2.6 Model Warmup

Pre-warm a model's KV cache without generating output.

**Endpoint**: `POST /v1/models/{model_id}/warmup`

**Request Body**:
```json
{
  "prompt": "Hello world",
  "max_length": 100
}
```

**Response** (200 OK):
```json
{
  "model_id": "qwen-7b",
  "status": "warmed",
  "warmup_time_ms": 45
}
```

---

## 3. Fusion Orchestration API

### 3.1 Execute Fusion Workflow

Execute a multi-model fusion workflow.

**Endpoint**: `POST /api/fusion/orchestrate`

**Request Body**:
```json
{
  "workflow": "rag_with_rerank",
  "input": {
    "query": "What are the benefits of MLX?",
    "collection": "technical_docs"
  },
  "models": {
    "retrieval": "embedding-model",
    "generation": "qwen-7b",
    "rerank": "rerank-model"
  },
  "config": {
    "top_k": 5,
    "temperature": 0.7
  }
}
```

**Request Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `workflow` | string | Yes | Workflow ID (predefined) or `"custom"` |
| `input` | object | Yes | Workflow input data |
| `models` | object | No | Override default models for workflow steps |
| `config` | object | No | Workflow-specific configuration |

**Response** (200 OK):
```json
{
  "workflow_id": "wf-123456",
  "status": "completed",
  "result": {
    "answer": "MLX provides several benefits: 1) Optimized for Apple Silicon...",
    "sources": [
      {
        "chunk_id": "doc1_chunk_3",
        "text": "MLX is optimized for Apple Silicon...",
        "score": 0.92
      }
    ]
  },
  "execution_trace": [
    {
      "step": "retrieval",
      "model": "embedding-model",
      "status": "completed",
      "duration_ms": 45
    },
    {
      "step": "generation",
      "model": "qwen-7b",
      "status": "completed",
      "duration_ms": 189
    }
  ],
  "total_latency_ms": 234,
  "created_at": "2025-11-17T10:00:00Z",
  "completed_at": "2025-11-17T10:00:00.234Z"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid workflow or input
- `404 Not Found`: Workflow not found
- `503 Service Unavailable`: Required model not loaded

---

### 3.2 List Available Workflows

List all predefined fusion workflows.

**Endpoint**: `GET /api/fusion/workflows`

**Response** (200 OK):
```json
{
  "workflows": [
    {
      "id": "rag_with_rerank",
      "name": "RAG with Reranking",
      "description": "Retrieve documents, rerank by relevance, and generate answer",
      "required_models": ["embeddings", "lm"],
      "optional_models": ["rerank"],
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "collection": {"type": "string"}
        },
        "required": ["query"]
      }
    },
    {
      "id": "multimodal_analysis",
      "name": "Multimodal Analysis",
      "description": "Analyze image + text input with vision model",
      "required_models": ["multimodal"],
      "optional_models": [],
      "input_schema": {
        "type": "object",
        "properties": {
          "image_url": {"type": "string"},
          "prompt": {"type": "string"}
        },
        "required": ["image_url", "prompt"]
      }
    },
    {
      "id": "sequential_summarization",
      "name": "Sequential Summarization",
      "description": "Summarize long document in multiple passes",
      "required_models": ["lm"],
      "optional_models": [],
      "input_schema": {
        "type": "object",
        "properties": {
          "document": {"type": "string"},
          "max_summary_length": {"type": "integer"}
        },
        "required": ["document"]
      }
    }
  ]
}
```

---

### 3.3 Model Capability Discovery

Discover which models support specific capabilities.

**Endpoint**: `GET /api/fusion/capabilities`

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `capability` | string | Filter by specific capability (optional) |

**Response** (200 OK):
```json
{
  "capabilities": {
    "chat": ["qwen-7b", "gemma-4b"],
    "vision": ["qwen-vl-3b"],
    "embeddings": ["embedding-model", "qwen-7b"],
    "image_generation": ["flux-schnell"],
    "tool_calling": ["qwen-7b"],
    "long_context": ["qwen-7b"]
  },
  "models": [
    {
      "id": "qwen-7b",
      "capabilities": ["chat", "embeddings", "tool_calling", "long_context"],
      "context_length": 8192,
      "vram_usage_gb": 4.2,
      "status": "loaded"
    },
    {
      "id": "embedding-model",
      "capabilities": ["embeddings"],
      "context_length": 512,
      "vram_usage_gb": 0.8,
      "status": "loaded"
    }
  ]
}
```

---

### 3.4 Compose Custom Workflow

Create a custom workflow from steps.

**Endpoint**: `POST /api/fusion/compose`

**Request Body**:
```json
{
  "name": "custom_rag_workflow",
  "description": "Custom RAG workflow with specific models",
  "steps": [
    {
      "step_id": "embed_query",
      "type": "model_call",
      "model_id": "embedding-model",
      "operation": "embed",
      "input": "{{ input.query }}"
    },
    {
      "step_id": "retrieve_docs",
      "type": "vector_search",
      "collection": "{{ input.collection }}",
      "query_embedding": "{{ steps.embed_query.output }}",
      "top_k": 5
    },
    {
      "step_id": "generate_answer",
      "type": "model_call",
      "model_id": "qwen-7b",
      "operation": "chat",
      "input": {
        "messages": [
          {
            "role": "system",
            "content": "Answer based on the following context: {{ steps.retrieve_docs.output }}"
          },
          {
            "role": "user",
            "content": "{{ input.query }}"
          }
        ]
      }
    }
  ]
}
```

**Response** (201 Created):
```json
{
  "workflow_id": "custom_rag_workflow",
  "name": "custom_rag_workflow",
  "status": "created",
  "created_at": "2025-11-17T10:00:00Z"
}
```

---

## 4. RAG Provider API

### 4.1 Ingest Documents

Ingest documents into a RAG collection.

**Endpoint**: `POST /api/rag/ingest`

**Request** (multipart/form-data):
```
collection: "technical_docs"
files: [document1.pdf, document2.pdf]
chunk_size: 512
overlap: 64
embedding_model: "embedding-model"
metadata: {"source": "user_upload", "category": "documentation"}
```

**Form Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `collection` | string | Yes | Collection name |
| `files` | file[] | Yes | Document files (PDF, TXT, MD, HTML) |
| `chunk_size` | integer | No | Chunk size in tokens (default: 512) |
| `overlap` | integer | No | Overlap between chunks (default: 64) |
| `embedding_model` | string | No | Model for embeddings (default: auto-select) |
| `metadata` | object | No | Additional metadata for documents |

**Response** (200 OK):
```json
{
  "collection": "technical_docs",
  "documents_ingested": 2,
  "chunks_created": 156,
  "embedding_time_ms": 1234,
  "document_ids": ["doc-123", "doc-456"],
  "status": "completed"
}
```

**Error Responses**:
- `400 Bad Request`: Invalid file format or parameters
- `413 Payload Too Large`: File size exceeds limit (default: 10MB per file)
- `503 Service Unavailable`: Embedding model not available

---

### 4.2 RAG Query

Perform retrieval-augmented generation query.

**Endpoint**: `POST /api/rag/query`

**Request Body**:
```json
{
  "query": "What are the benefits of MLX?",
  "collection": "technical_docs",
  "top_k": 5,
  "model_id": "qwen-7b",
  "embedding_model": "embedding-model",
  "include_sources": true,
  "similarity_threshold": 0.7,
  "rerank": false,
  "generation_config": {
    "temperature": 0.7,
    "max_tokens": 512
  }
}
```

**Request Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | Yes | User query |
| `collection` | string | No | Collection name (default: `"default"`) |
| `top_k` | integer | No | Number of documents to retrieve (default: 5) |
| `model_id` | string | No | Generation model (default: auto-select) |
| `embedding_model` | string | No | Embedding model (default: auto-select) |
| `include_sources` | boolean | No | Include source chunks in response (default: `true`) |
| `similarity_threshold` | float | No | Minimum similarity score (default: 0.7) |
| `rerank` | boolean | No | Enable reranking (default: `false`) |
| `generation_config` | object | No | Generation parameters |

**Response** (200 OK):
```json
{
  "answer": "MLX provides several benefits: 1) Optimized for Apple Silicon with unified memory...",
  "sources": [
    {
      "chunk_id": "doc1_chunk_3",
      "text": "MLX is optimized for Apple Silicon, leveraging unified memory...",
      "score": 0.92,
      "metadata": {
        "file": "mlx_intro.pdf",
        "page": 3,
        "document_id": "doc-123"
      }
    },
    {
      "chunk_id": "doc1_chunk_7",
      "text": "The framework provides automatic differentiation...",
      "score": 0.87,
      "metadata": {
        "file": "mlx_intro.pdf",
        "page": 7,
        "document_id": "doc-123"
      }
    }
  ],
  "retrieval_latency_ms": 45,
  "generation_latency_ms": 234,
  "total_latency_ms": 279,
  "metadata": {
    "collection": "technical_docs",
    "top_k": 5,
    "model_id": "qwen-7b",
    "embedding_model": "embedding-model"
  }
}
```

---

### 4.3 List RAG Collections

List all RAG collections.

**Endpoint**: `GET /api/rag/collections`

**Response** (200 OK):
```json
{
  "collections": [
    {
      "name": "technical_docs",
      "document_count": 156,
      "chunk_count": 3421,
      "created_at": "2025-11-17T10:00:00Z",
      "updated_at": "2025-11-17T10:30:00Z",
      "metadata": {
        "description": "Technical documentation",
        "source": "PDF upload"
      }
    },
    {
      "name": "research_papers",
      "document_count": 42,
      "chunk_count": 1234,
      "created_at": "2025-11-16T15:00:00Z",
      "updated_at": "2025-11-17T09:00:00Z",
      "metadata": {
        "description": "Research papers on ML",
        "source": "Arxiv"
      }
    }
  ]
}
```

---

### 4.4 Get Collection Info

Get detailed information about a collection.

**Endpoint**: `GET /api/rag/collections/{collection_name}`

**Response** (200 OK):
```json
{
  "name": "technical_docs",
  "document_count": 156,
  "chunk_count": 3421,
  "created_at": "2025-11-17T10:00:00Z",
  "updated_at": "2025-11-17T10:30:00Z",
  "embedding_model": "embedding-model",
  "chunk_size": 512,
  "overlap": 64,
  "metadata": {
    "description": "Technical documentation",
    "source": "PDF upload"
  },
  "documents": [
    {
      "document_id": "doc-123",
      "filename": "mlx_intro.pdf",
      "chunk_count": 78,
      "ingested_at": "2025-11-17T10:00:00Z"
    }
  ]
}
```

---

### 4.5 Delete Documents

Delete documents from a collection.

**Endpoint**: `DELETE /api/rag/documents/{collection}/{document_id}`

**Path Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `collection` | string | Collection name |
| `document_id` | string | Document identifier |

**Response** (200 OK):
```json
{
  "collection": "technical_docs",
  "document_id": "doc-123",
  "chunks_deleted": 78,
  "status": "deleted"
}
```

---

### 4.6 Delete Collection

Delete an entire RAG collection.

**Endpoint**: `DELETE /api/rag/collections/{collection_name}`

**Response** (200 OK):
```json
{
  "collection": "technical_docs",
  "documents_deleted": 156,
  "chunks_deleted": 3421,
  "status": "deleted"
}
```

---

## 5. MCP Server API

### 5.1 Submit Job

Submit a job to the MCP queue.

**Endpoint**: `POST /api/mcp/jobs`

**Request Body**:
```json
{
  "model_id": "qwen-7b",
  "operation": "chat",
  "input": {
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  },
  "callback_url": "https://tier2.example.com/webhooks/job-complete",
  "priority": "normal",
  "metadata": {
    "user_id": "user123",
    "session_id": "sess456"
  }
}
```

**Request Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | Model identifier |
| `operation` | string | Yes | Operation: `chat`, `embed`, `image_gen`, etc. |
| `input` | object | Yes | Operation-specific input |
| `callback_url` | string | No | Webhook URL for job completion |
| `priority` | string | No | Priority: `low`, `normal`, `high` (default: `normal`) |
| `metadata` | object | No | Additional metadata |

**Response** (202 Accepted):
```json
{
  "job_id": "job-123456",
  "status": "queued",
  "created_at": "2025-11-17T10:00:00Z",
  "estimated_completion_ms": 500,
  "position_in_queue": 3
}
```

**Error Responses**:
- `400 Bad Request`: Invalid operation or input
- `404 Not Found`: Model not found
- `429 Too Many Requests`: Queue is full

---

### 5.2 Get Job Status

Get the status of a submitted job.

**Endpoint**: `GET /api/mcp/jobs/{job_id}`

**Response** (200 OK):

**Queued**:
```json
{
  "job_id": "job-123456",
  "status": "queued",
  "created_at": "2025-11-17T10:00:00Z",
  "position_in_queue": 3,
  "estimated_completion_ms": 500
}
```

**Processing**:
```json
{
  "job_id": "job-123456",
  "status": "processing",
  "created_at": "2025-11-17T10:00:00Z",
  "started_at": "2025-11-17T10:00:01Z",
  "progress": 0.5
}
```

**Completed**:
```json
{
  "job_id": "job-123456",
  "status": "completed",
  "created_at": "2025-11-17T10:00:00Z",
  "started_at": "2025-11-17T10:00:01Z",
  "completed_at": "2025-11-17T10:00:02Z",
  "result": {
    "content": "Hello! I'm doing well, thank you for asking. How can I help you today?",
    "usage": {
      "prompt_tokens": 10,
      "completion_tokens": 20,
      "total_tokens": 30
    }
  },
  "metadata": {
    "user_id": "user123",
    "session_id": "sess456"
  }
}
```

**Failed**:
```json
{
  "job_id": "job-123456",
  "status": "failed",
  "created_at": "2025-11-17T10:00:00Z",
  "started_at": "2025-11-17T10:00:01Z",
  "failed_at": "2025-11-17T10:00:02Z",
  "error": "Model inference failed: Out of memory"
}
```

**Error Responses**:
- `404 Not Found`: Job ID not found

---

### 5.3 List Jobs

List jobs with optional filters.

**Endpoint**: `GET /api/mcp/jobs`

**Query Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status: `queued`, `processing`, `completed`, `failed`, `cancelled` |
| `model_id` | string | Filter by model ID |
| `limit` | integer | Max results (default: 100, max: 1000) |
| `offset` | integer | Pagination offset (default: 0) |
| `sort_by` | string | Sort by: `created_at`, `completed_at` (default: `created_at`) |
| `order` | string | Sort order: `asc`, `desc` (default: `desc`) |

**Response** (200 OK):
```json
{
  "jobs": [
    {
      "job_id": "job-123456",
      "status": "completed",
      "model_id": "qwen-7b",
      "operation": "chat",
      "created_at": "2025-11-17T10:00:00Z",
      "completed_at": "2025-11-17T10:00:02Z",
      "latency_ms": 234
    },
    {
      "job_id": "job-123455",
      "status": "processing",
      "model_id": "qwen-7b",
      "operation": "chat",
      "created_at": "2025-11-17T09:59:58Z",
      "started_at": "2025-11-17T09:59:59Z"
    }
  ],
  "total": 42,
  "limit": 100,
  "offset": 0
}
```

---

### 5.4 Cancel Job

Cancel a queued or processing job.

**Endpoint**: `DELETE /api/mcp/jobs/{job_id}`

**Response** (200 OK):
```json
{
  "job_id": "job-123456",
  "status": "cancelled",
  "cancelled_at": "2025-11-17T10:00:05Z"
}
```

**Error Responses**:
- `404 Not Found`: Job ID not found
- `409 Conflict`: Job already completed or failed

---

### 5.5 Register Webhook

Register a webhook for job notifications.

**Endpoint**: `POST /api/mcp/webhooks`

**Request Body**:
```json
{
  "url": "https://tier2.example.com/webhooks/job-complete",
  "events": ["job.completed", "job.failed"],
  "secret": "webhook-secret-123",
  "metadata": {
    "description": "Tier 2 job completion webhook"
  }
}
```

**Request Parameters**:
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | Webhook URL |
| `events` | string[] | Yes | Events to subscribe: `job.queued`, `job.processing`, `job.completed`, `job.failed`, `job.cancelled` |
| `secret` | string | No | Secret for HMAC signature verification |
| `metadata` | object | No | Additional metadata |

**Response** (201 Created):
```json
{
  "webhook_id": "wh-123456",
  "url": "https://tier2.example.com/webhooks/job-complete",
  "events": ["job.completed", "job.failed"],
  "status": "active",
  "created_at": "2025-11-17T10:00:00Z"
}
```

---

### 5.6 List Webhooks

List all registered webhooks.

**Endpoint**: `GET /api/mcp/webhooks`

**Response** (200 OK):
```json
{
  "webhooks": [
    {
      "webhook_id": "wh-123456",
      "url": "https://tier2.example.com/webhooks/job-complete",
      "events": ["job.completed", "job.failed"],
      "status": "active",
      "created_at": "2025-11-17T10:00:00Z",
      "last_triggered_at": "2025-11-17T10:15:00Z",
      "success_count": 42,
      "failure_count": 0
    }
  ]
}
```

---

### 5.7 Delete Webhook

Delete a webhook.

**Endpoint**: `DELETE /api/mcp/webhooks/{webhook_id}`

**Response** (200 OK):
```json
{
  "webhook_id": "wh-123456",
  "status": "deleted"
}
```

---

### 5.8 Webhook Payload Format

When a job event occurs, a POST request is sent to the webhook URL.

**Headers**:
```
Content-Type: application/json
X-Webhook-Event: job.completed
X-Webhook-Signature: sha256=<HMAC signature>
X-Webhook-ID: wh-123456
```

**Payload** (job.completed):
```json
{
  "event": "job.completed",
  "timestamp": "2025-11-17T10:00:02Z",
  "job": {
    "job_id": "job-123456",
    "status": "completed",
    "model_id": "qwen-7b",
    "operation": "chat",
    "created_at": "2025-11-17T10:00:00Z",
    "started_at": "2025-11-17T10:00:01Z",
    "completed_at": "2025-11-17T10:00:02Z",
    "result": {
      "content": "Hello! I'm doing well...",
      "usage": {"prompt_tokens": 10, "completion_tokens": 20}
    },
    "metadata": {
      "user_id": "user123",
      "session_id": "sess456"
    }
  }
}
```

**Payload** (job.failed):
```json
{
  "event": "job.failed",
  "timestamp": "2025-11-17T10:00:02Z",
  "job": {
    "job_id": "job-123456",
    "status": "failed",
    "model_id": "qwen-7b",
    "operation": "chat",
    "created_at": "2025-11-17T10:00:00Z",
    "started_at": "2025-11-17T10:00:01Z",
    "failed_at": "2025-11-17T10:00:02Z",
    "error": "Model inference failed: Out of memory"
  }
}
```

---

## 6. Observability & Diagnostics API

### 6.1 Prometheus Metrics

Export Prometheus-compatible metrics.

**Endpoint**: `GET /metrics`

**Response** (200 OK, text/plain):
```
# HELP mlx_requests_total Total number of requests
# TYPE mlx_requests_total counter
mlx_requests_total{model="qwen-7b",endpoint="/v1/chat/completions",status="200"} 42

# HELP mlx_request_latency_seconds Request latency in seconds
# TYPE mlx_request_latency_seconds histogram
mlx_request_latency_seconds_bucket{model="qwen-7b",le="0.1"} 10
mlx_request_latency_seconds_bucket{model="qwen-7b",le="0.5"} 35
mlx_request_latency_seconds_bucket{model="qwen-7b",le="1.0"} 42
mlx_request_latency_seconds_sum{model="qwen-7b"} 12.34
mlx_request_latency_seconds_count{model="qwen-7b"} 42

# HELP mlx_vram_usage_gb Current VRAM usage in GB
# TYPE mlx_vram_usage_gb gauge
mlx_vram_usage_gb{model="qwen-7b"} 4.2
mlx_vram_usage_gb{model="embedding-model"} 0.8

# HELP mlx_queue_depth Current queue depth
# TYPE mlx_queue_depth gauge
mlx_queue_depth{model="qwen-7b"} 3

# HELP mlx_model_load_time_seconds Model load time in seconds
# TYPE mlx_model_load_time_seconds gauge
mlx_model_load_time_seconds{model="qwen-7b"} 1.234
```

---

### 6.2 System Diagnostics

Get comprehensive system diagnostics.

**Endpoint**: `GET /internal/diagnostics`

**Response** (200 OK):
```json
{
  "timestamp": "2025-11-17T10:00:00Z",
  "uptime_seconds": 3600,
  "system": {
    "platform": "darwin",
    "platform_version": "macOS 14.5",
    "processor": "Apple M1 Ultra",
    "python_version": "3.11.5",
    "mlx_version": "0.28.3"
  },
  "resources": {
    "cpu_usage_percent": 45.2,
    "memory_usage_gb": 12.5,
    "memory_total_gb": 128.0,
    "vram_usage_gb": 8.7,
    "vram_total_gb": 64.0
  },
  "models": {
    "loaded_count": 2,
    "total_vram_usage_gb": 5.0,
    "models": [
      {
        "id": "qwen-7b",
        "status": "loaded",
        "vram_usage_gb": 4.2,
        "request_count": 42,
        "avg_latency_ms": 234.5
      },
      {
        "id": "embedding-model",
        "status": "loaded",
        "vram_usage_gb": 0.8,
        "request_count": 156,
        "avg_latency_ms": 45.2
      }
    ]
  },
  "queues": {
    "qwen-7b": {
      "queue_size": 3,
      "max_queue_size": 100,
      "active_requests": 1,
      "max_concurrency": 2
    }
  },
  "jobs": {
    "queued": 5,
    "processing": 2,
    "completed": 142,
    "failed": 3,
    "cancelled": 1
  },
  "rag": {
    "collections": 2,
    "total_documents": 198,
    "total_chunks": 4655
  },
  "errors": {
    "last_24h": 3,
    "recent_errors": [
      {
        "timestamp": "2025-11-17T09:45:00Z",
        "error": "Model inference timeout",
        "model": "qwen-7b"
      }
    ]
  }
}
```

---

### 6.3 Health Check (Extended)

Enhanced health check with detailed status.

**Endpoint**: `GET /health`

**Response** (200 OK):
```json
{
  "status": "ok",
  "timestamp": "2025-11-17T10:00:00Z",
  "models": {
    "loaded_count": 2,
    "models_healthy": true,
    "models": [
      {
        "id": "qwen-7b",
        "status": "loaded",
        "warmup_completed": true
      }
    ]
  },
  "persistence": {
    "mongodb_connected": true,
    "redis_connected": true,
    "vector_db_connected": true
  },
  "services": {
    "fusion_enabled": true,
    "rag_enabled": true,
    "mcp_enabled": true
  }
}
```

---

## 7. Error Handling

### Standard Error Response

All errors return a consistent format:

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Model 'nonexistent-model' not found",
    "param": "model_id",
    "code": "model_not_found"
  }
}
```

### Error Types

| HTTP Status | Error Type | Description |
|-------------|-----------|-------------|
| 400 | `invalid_request_error` | Invalid parameters or request format |
| 401 | `authentication_error` | Invalid API key or token |
| 403 | `permission_error` | Insufficient permissions |
| 404 | `not_found_error` | Resource not found |
| 409 | `conflict_error` | Resource conflict (e.g., duplicate ID) |
| 413 | `payload_too_large_error` | Request payload exceeds limit |
| 429 | `rate_limit_error` | Too many requests |
| 500 | `internal_error` | Server error |
| 503 | `service_unavailable_error` | Service temporarily unavailable |

### Error Codes

| Code | Description |
|------|-------------|
| `model_not_found` | Model ID not found in registry |
| `model_already_loaded` | Model ID already exists |
| `insufficient_vram` | Not enough VRAM to load model |
| `job_not_found` | Job ID not found |
| `queue_full` | Request queue is full |
| `timeout` | Operation timed out |
| `invalid_workflow` | Workflow ID not found or invalid |
| `collection_not_found` | RAG collection not found |

---

## 8. Rate Limiting

### Limits

| Endpoint Group | Rate Limit | Window |
|---------------|------------|--------|
| Model Management | 10 req/min | Per IP |
| Fusion Orchestration | 60 req/min | Per IP |
| RAG Provider | 30 req/min | Per IP |
| MCP Jobs | 100 req/min | Per IP |

### Rate Limit Headers

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1700234567
```

### Rate Limit Error

**Response** (429 Too Many Requests):
```json
{
  "error": {
    "type": "rate_limit_error",
    "message": "Rate limit exceeded. Try again in 30 seconds.",
    "param": null,
    "code": "rate_limit_exceeded"
  }
}
```

---

## 9. Versioning

### API Version

Current API version: **v1**

### Version Header

Include API version in requests (optional):
```
X-API-Version: 1
```

### Deprecation Policy

- Breaking changes result in new API version (v2, v3, etc.)
- Previous versions supported for 6 months after deprecation announcement
- Deprecation warnings included in response headers:
  ```
  X-API-Deprecated: true
  X-API-Sunset: 2025-12-31
  ```

---

**End of Phase-4 API Specification**
