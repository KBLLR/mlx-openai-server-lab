# Phase-4 Fusion Orchestration Architecture

**Document Version**: 1.0
**Date**: 2025-11-17
**Status**: Architecture Definition
**Session**: Phase-4 Orchestration & RAG Implementation

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Multi-Model Registry](#3-multi-model-registry)
4. [Fusion Orchestration Layer](#4-fusion-orchestration-layer)
5. [RAG Provider Service](#5-rag-provider-service)
6. [MCP Server Layer](#6-mcp-server-layer)
7. [State Management](#7-state-management)
8. [Request Flow](#8-request-flow)
9. [Data Schemas](#9-data-schemas)
10. [Deployment Architecture](#10-deployment-architecture)
11. [Security & Access Control](#11-security--access-control)
12. [Performance Considerations](#12-performance-considerations)
13. [Future Enhancements](#13-future-enhancements)

---

## 1. Executive Summary

### Purpose

Phase-4 transforms the mlx-openai-server from a **stateless single-model inference server** into a **multi-model fusion orchestration platform** with RAG capabilities and persistent state management.

### Key Objectives

1. **Multi-Model Support**: Load and manage multiple MLX models simultaneously
2. **Fusion Orchestration**: Route requests intelligently based on model capabilities
3. **RAG Integration**: Provide document retrieval and generation capabilities
4. **State Persistence**: Track jobs, maintain history, and enable crash recovery
5. **MCP Protocol**: Enable Tier 2 ↔ Tier 3 state synchronization

### Architectural Principles

- **Local-First**: All inference happens locally on Apple Silicon via MLX
- **Modular Design**: Clear separation between fusion, RAG, and MCP layers
- **Backward Compatible**: Existing `/v1/*` endpoints continue to work
- **Async-Native**: Built on asyncio for efficient concurrency
- **Observable**: Rich metrics, tracing, and diagnostics

---

## 2. Architecture Overview

### System Layers

```
┌─────────────────────────────────────────────────────────────────┐
│                          Tier 2 (MCP)                           │
│                    (External Orchestrator)                       │
└─────────────────────────────────────────────────────────────────┘
                              ↕ HTTP/Webhooks
┌─────────────────────────────────────────────────────────────────┐
│                       MCP Server Layer                          │
│  Job Tracking | State Sync | Webhook Management | Job History  │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                   Fusion Orchestration Layer                    │
│  Multi-Model Routing | Workflow Execution | Capability Discovery│
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌──────────────────────┬──────────────────────┬───────────────────┐
│   MLX Provider       │   RAG Provider       │  Observability    │
│   (Tier 3A)         │                      │   & Diagnostics   │
│                     │                      │                   │
│ • Text Models       │ • Document Ingest    │ • Metrics         │
│ • Multimodal        │ • Vector Store       │ • Tracing         │
│ • Embeddings        │ • Retrieval          │ • Logging         │
│ • Image Gen/Edit    │ • RAG Generation     │ • Diagnostics     │
│ • Audio (Whisper)   │                      │                   │
└──────────────────────┴──────────────────────┴───────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                      Persistence Layer                          │
│         MongoDB (State) | Redis (Queue) | Vector DB            │
└─────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibilities | Technology |
|-----------|-----------------|------------|
| **MCP Server** | Job tracking, state sync, webhooks | FastAPI, MongoDB |
| **Fusion Orchestrator** | Multi-model routing, workflow execution | Python async |
| **MLX Provider** | Local model inference (existing) | MLX, FastAPI |
| **RAG Provider** | Document ingestion, retrieval, generation | ChromaDB, MLX |
| **State Manager** | Job persistence, history, recovery | MongoDB (motor) |
| **Persistent Queue** | Request queue with crash recovery | Redis or MongoDB |
| **Observability** | Metrics, tracing, diagnostics | Prometheus, OpenTelemetry |

---

## 3. Multi-Model Registry

### Current State

**Limitation**: `app/core/model_registry.py` supports only **one model per instance**

### Phase-4 Enhancement

**Goal**: Support **multiple concurrent models** with dynamic loading/unloading

### Architecture

```python
class EnhancedModelRegistry:
    """
    Multi-model registry with VRAM management.

    Features:
    - Load/unload models dynamically (hot-swap)
    - VRAM monitoring and model eviction (LRU)
    - Model capability discovery and tagging
    - Per-model concurrency limits
    - Model health tracking
    """

    def __init__(self, max_vram_gb: float = 32.0):
        self._handlers: Dict[str, ModelHandler] = {}
        self._metadata: Dict[str, ModelMetadata] = {}
        self._vram_tracker: VRAMTracker = VRAMTracker(max_vram_gb)
        self._lru_cache: LRUModelCache = LRUModelCache()
        self._lock: asyncio.Lock = asyncio.Lock()

    async def load_model(
        self,
        model_id: str,
        model_path: str,
        model_type: str,
        **kwargs
    ) -> ModelMetadata:
        """Load a new model into registry."""

    async def unload_model(self, model_id: str, force: bool = False) -> None:
        """Unload a model from registry."""

    async def get_model_by_capability(
        self,
        capability: str
    ) -> List[str]:
        """Get models matching a specific capability."""

    async def evict_least_used_model(self) -> Optional[str]:
        """Evict LRU model to free VRAM."""
```

### VRAM Management Strategy

1. **Monitoring**: Track VRAM usage per model via MLX API
2. **Limits**: Configure max total VRAM (default: 32GB for M1 Ultra)
3. **Eviction Policy**: LRU (Least Recently Used) with configurable TTL
4. **Preemption**: Support priority-based model loading (evict low-priority models first)

### Model Loading API

**Endpoint**: `POST /v1/models/load`

**Request**:
```json
{
  "model_path": "mlx-community/qwen2.5-7b-instruct-4bit",
  "model_type": "lm",
  "model_id": "qwen-7b",
  "context_length": 8192,
  "priority": "normal"
}
```

**Response**:
```json
{
  "model_id": "qwen-7b",
  "status": "loaded",
  "vram_usage_gb": 4.2,
  "metadata": {
    "id": "qwen-7b",
    "family": "qwen",
    "type": "lm",
    "context_length": 8192,
    "tags": ["local", "chat", "quantized"]
  }
}
```

### Model Unloading API

**Endpoint**: `DELETE /v1/models/{model_id}/unload`

**Response**:
```json
{
  "model_id": "qwen-7b",
  "status": "unloaded",
  "vram_freed_gb": 4.2
}
```

---

## 4. Fusion Orchestration Layer

### Purpose

**Route requests intelligently** across multiple loaded models based on:
- Model capabilities (text, vision, audio, embeddings)
- Request requirements (e.g., long context, speed, quality)
- System resource availability (VRAM, queue depth)

### Architecture

```
app/fusion/
├── __init__.py
├── coordinator.py          # Main orchestration coordinator
├── router.py               # Model routing logic
├── workflows.py            # Predefined workflow templates
├── capabilities.py         # Model capability discovery
└── schemas.py             # Fusion request/response schemas
```

### Fusion Coordinator

```python
class FusionCoordinator:
    """
    Multi-model workflow orchestrator.

    Supports:
    - Sequential workflows (model A → model B)
    - Parallel workflows (model A + model B)
    - Conditional workflows (route based on input)
    - Custom user-defined workflows
    """

    def __init__(self, registry: EnhancedModelRegistry):
        self.registry = registry
        self.router = ModelRouter(registry)
        self.workflow_engine = WorkflowEngine()

    async def orchestrate(
        self,
        workflow: FusionWorkflow,
        input_data: Dict[str, Any]
    ) -> FusionResult:
        """Execute a multi-model fusion workflow."""
```

### Routing Strategies

1. **Capability-Based**: Route to model with required capability
   - Example: Image analysis → multimodal model

2. **Load-Based**: Route to least-loaded model
   - Example: High request volume → distribute across replicas

3. **Quality-Based**: Route to highest-quality model
   - Example: Critical requests → largest model

4. **Custom**: User-defined routing logic
   - Example: Route based on request metadata

### Fusion API Endpoints

#### 1. Orchestrate Multi-Model Workflow

**Endpoint**: `POST /api/fusion/orchestrate`

**Request**:
```json
{
  "workflow": "rag_with_rerank",
  "input": {
    "query": "What are the benefits of MLX?",
    "documents": ["doc1", "doc2"]
  },
  "models": {
    "retrieval": "embedding-model",
    "generation": "qwen-7b",
    "rerank": "rerank-model"
  }
}
```

**Response**:
```json
{
  "workflow_id": "wf-123456",
  "status": "completed",
  "result": {
    "answer": "MLX provides...",
    "sources": ["doc1", "doc2"],
    "latency_ms": 234
  },
  "execution_trace": [
    {"step": "retrieval", "model": "embedding-model", "duration_ms": 45},
    {"step": "generation", "model": "qwen-7b", "duration_ms": 189}
  ]
}
```

#### 2. List Available Workflows

**Endpoint**: `GET /api/fusion/workflows`

**Response**:
```json
{
  "workflows": [
    {
      "id": "rag_with_rerank",
      "name": "RAG with Reranking",
      "description": "Retrieve documents, rerank, and generate answer",
      "required_models": ["embeddings", "lm"],
      "optional_models": ["rerank"]
    },
    {
      "id": "multimodal_analysis",
      "name": "Multimodal Analysis",
      "description": "Analyze image + text input",
      "required_models": ["multimodal"]
    }
  ]
}
```

#### 3. Model Capability Discovery

**Endpoint**: `GET /api/fusion/capabilities`

**Response**:
```json
{
  "capabilities": {
    "chat": ["qwen-7b", "gemma-4b"],
    "vision": ["qwen-vl-3b"],
    "embeddings": ["embedding-model"],
    "image_generation": ["flux-schnell"]
  },
  "models": [
    {
      "id": "qwen-7b",
      "capabilities": ["chat", "embeddings", "tool_calling"],
      "context_length": 8192,
      "vram_usage_gb": 4.2
    }
  ]
}
```

#### 4. Compose Custom Workflow

**Endpoint**: `POST /api/fusion/compose`

**Request**:
```json
{
  "name": "custom_workflow",
  "steps": [
    {
      "type": "model_call",
      "model_id": "embedding-model",
      "operation": "embed",
      "input": "{{ input.query }}"
    },
    {
      "type": "model_call",
      "model_id": "qwen-7b",
      "operation": "chat",
      "input": "{{ steps[0].output }}"
    }
  ]
}
```

---

## 5. RAG Provider Service

### Purpose

**Retrieval-Augmented Generation** over document collections using local MLX models.

### Architecture

```
app/rag/
├── __init__.py
├── document_processor.py   # PDF/text chunking
├── vector_store.py         # ChromaDB/FAISS wrapper
├── retriever.py            # Semantic search
├── generator.py            # RAG generation with MLX
├── reranker.py            # Optional reranking
└── schemas.py             # RAG request/response schemas
```

### Components

#### 1. Document Processor

```python
class DocumentProcessor:
    """
    Document ingestion and chunking.

    Supports:
    - PDF, TXT, MD, HTML formats
    - Recursive chunking with overlap
    - Metadata extraction (title, author, date)
    - Deduplication
    """

    def chunk_document(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 64
    ) -> List[DocumentChunk]:
        """Split document into overlapping chunks."""
```

#### 2. Vector Store

```python
class VectorStore:
    """
    Vector database interface.

    Backends:
    - ChromaDB (default, easy setup)
    - FAISS (fast, in-memory)
    - MLX-native (future: custom MLX vector index)
    """

    async def add_documents(
        self,
        documents: List[DocumentChunk],
        collection: str = "default"
    ) -> None:
        """Add documents to vector store."""

    async def search(
        self,
        query: str,
        k: int = 5,
        collection: str = "default",
        similarity_threshold: float = 0.7
    ) -> List[DocumentChunk]:
        """Semantic search over documents."""
```

#### 3. RAG Generator

```python
class RAGGenerator:
    """
    RAG generation with MLX models.

    Process:
    1. Embed query using local embeddings model
    2. Retrieve top-k documents from vector store
    3. Rerank documents (optional)
    4. Generate answer using LM with context
    """

    async def generate(
        self,
        query: str,
        collection: str = "default",
        top_k: int = 5,
        model_id: Optional[str] = None
    ) -> RAGResult:
        """Generate RAG answer with citations."""
```

### RAG API Endpoints

#### 1. Ingest Documents

**Endpoint**: `POST /api/rag/ingest`

**Request** (multipart/form-data):
```
collection: "technical_docs"
files: [document1.pdf, document2.pdf]
chunk_size: 512
overlap: 64
```

**Response**:
```json
{
  "collection": "technical_docs",
  "documents_ingested": 2,
  "chunks_created": 156,
  "embedding_time_ms": 1234
}
```

#### 2. RAG Query

**Endpoint**: `POST /api/rag/query`

**Request**:
```json
{
  "query": "What are the benefits of MLX?",
  "collection": "technical_docs",
  "top_k": 5,
  "model_id": "qwen-7b",
  "include_sources": true
}
```

**Response**:
```json
{
  "answer": "MLX provides several benefits: 1) Optimized for Apple Silicon...",
  "sources": [
    {
      "chunk_id": "doc1_chunk_3",
      "text": "MLX is optimized for Apple Silicon...",
      "score": 0.92,
      "metadata": {"file": "mlx_intro.pdf", "page": 3}
    }
  ],
  "retrieval_latency_ms": 45,
  "generation_latency_ms": 234
}
```

#### 3. List Collections

**Endpoint**: `GET /api/rag/collections`

**Response**:
```json
{
  "collections": [
    {
      "name": "technical_docs",
      "document_count": 156,
      "chunk_count": 3421,
      "created_at": "2025-11-17T10:00:00Z"
    }
  ]
}
```

#### 4. Delete Documents

**Endpoint**: `DELETE /api/rag/documents/{collection}/{document_id}`

**Response**:
```json
{
  "collection": "technical_docs",
  "document_id": "doc1",
  "chunks_deleted": 78,
  "status": "deleted"
}
```

---

## 6. MCP Server Layer

### Purpose

**Model Context Protocol** server for Tier 2 ↔ Tier 3 integration with persistent state management.

### Architecture

```
app/mcp/
├── __init__.py
├── protocol.py       # MCP protocol implementation
├── handlers.py       # Request/response handlers
├── state_sync.py     # Tier 2 state synchronization
├── webhooks.py       # Webhook management
└── schemas.py        # MCP request/response schemas
```

### MCP Protocol

**Core Concepts**:
1. **Job**: A unit of work submitted to Tier 3
2. **State**: Job lifecycle (queued → processing → completed/failed)
3. **Webhook**: Callback URL for job completion notifications
4. **History**: Persistent job history with TTL

### Job Lifecycle

```
[Tier 2] POST /api/mcp/jobs
    ↓
[queued] → [processing] → [completed/failed]
    ↓           ↓              ↓
 MongoDB    MongoDB       Webhook → [Tier 2]
```

### MCP API Endpoints

#### 1. Submit Job

**Endpoint**: `POST /api/mcp/jobs`

**Request**:
```json
{
  "model_id": "qwen-7b",
  "operation": "chat",
  "input": {
    "messages": [{"role": "user", "content": "Hello"}]
  },
  "callback_url": "https://tier2.example.com/webhooks/job-complete",
  "priority": "normal",
  "metadata": {
    "user_id": "user123",
    "session_id": "sess456"
  }
}
```

**Response**:
```json
{
  "job_id": "job-123456",
  "status": "queued",
  "created_at": "2025-11-17T10:00:00Z",
  "estimated_completion_ms": 500
}
```

#### 2. Get Job Status

**Endpoint**: `GET /api/mcp/jobs/{job_id}`

**Response**:
```json
{
  "job_id": "job-123456",
  "status": "completed",
  "created_at": "2025-11-17T10:00:00Z",
  "started_at": "2025-11-17T10:00:01Z",
  "completed_at": "2025-11-17T10:00:02Z",
  "result": {
    "content": "Hello! How can I help you today?",
    "usage": {"prompt_tokens": 10, "completion_tokens": 20}
  },
  "metadata": {
    "user_id": "user123",
    "session_id": "sess456"
  }
}
```

#### 3. List Jobs

**Endpoint**: `GET /api/mcp/jobs`

**Query Parameters**:
- `status`: Filter by status (queued, processing, completed, failed)
- `model_id`: Filter by model
- `limit`: Max results (default: 100)
- `offset`: Pagination offset

**Response**:
```json
{
  "jobs": [
    {
      "job_id": "job-123456",
      "status": "completed",
      "model_id": "qwen-7b",
      "created_at": "2025-11-17T10:00:00Z"
    }
  ],
  "total": 1,
  "limit": 100,
  "offset": 0
}
```

#### 4. Cancel Job

**Endpoint**: `DELETE /api/mcp/jobs/{job_id}`

**Response**:
```json
{
  "job_id": "job-123456",
  "status": "cancelled",
  "cancelled_at": "2025-11-17T10:00:05Z"
}
```

#### 5. Register Webhook

**Endpoint**: `POST /api/mcp/webhook`

**Request**:
```json
{
  "url": "https://tier2.example.com/webhooks/job-complete",
  "events": ["job.completed", "job.failed"],
  "secret": "webhook-secret-123"
}
```

**Response**:
```json
{
  "webhook_id": "wh-123456",
  "url": "https://tier2.example.com/webhooks/job-complete",
  "status": "active"
}
```

---

## 7. State Management

### Database Schema (MongoDB)

#### Jobs Collection

```javascript
{
  "_id": ObjectId("..."),
  "job_id": "job-123456",
  "status": "completed",  // queued, processing, completed, failed, cancelled
  "model_id": "qwen-7b",
  "operation": "chat",
  "priority": "normal",
  "created_at": ISODate("2025-11-17T10:00:00Z"),
  "started_at": ISODate("2025-11-17T10:00:01Z"),
  "completed_at": ISODate("2025-11-17T10:00:02Z"),
  "input": {
    // Request data (sanitized, no sensitive info)
  },
  "result": {
    // Response data
  },
  "metadata": {
    "user_id": "user123",
    "session_id": "sess456"
  },
  "error": null,  // Error message if failed
  "ttl_expire_at": ISODate("2025-11-24T10:00:00Z")  // 7-day TTL
}
```

**Indexes**:
- `job_id` (unique)
- `status, created_at` (compound)
- `model_id, created_at` (compound)
- `ttl_expire_at` (TTL index for auto-cleanup)

#### Webhooks Collection

```javascript
{
  "_id": ObjectId("..."),
  "webhook_id": "wh-123456",
  "url": "https://tier2.example.com/webhooks/job-complete",
  "events": ["job.completed", "job.failed"],
  "secret": "webhook-secret-123",
  "status": "active",  // active, disabled
  "created_at": ISODate("2025-11-17T10:00:00Z"),
  "last_triggered_at": ISODate("2025-11-17T10:00:05Z"),
  "failure_count": 0
}
```

#### RAG Collections Collection

```javascript
{
  "_id": ObjectId("..."),
  "collection_name": "technical_docs",
  "document_count": 156,
  "chunk_count": 3421,
  "created_at": ISODate("2025-11-17T10:00:00Z"),
  "updated_at": ISODate("2025-11-17T10:30:00Z"),
  "metadata": {
    "description": "Technical documentation",
    "source": "PDF upload"
  }
}
```

### State Management API

```python
class StateManager:
    """
    MongoDB-backed state management.

    Features:
    - Async job CRUD operations
    - Job status transitions with validation
    - Job history with TTL (default: 7 days)
    - Webhook registration and triggering
    - Atomic operations for consistency
    """

    def __init__(self, mongo_uri: str, db_name: str = "mlx_fusion"):
        self.client = motor.motor_asyncio.AsyncIOMotorClient(mongo_uri)
        self.db = self.client[db_name]
        self.jobs = self.db["jobs"]
        self.webhooks = self.db["webhooks"]
        self.rag_collections = self.db["rag_collections"]

    async def create_job(self, job: JobCreate) -> Job:
        """Create new job in database."""

    async def update_job_status(
        self,
        job_id: str,
        status: JobStatus,
        result: Optional[Dict] = None,
        error: Optional[str] = None
    ) -> Job:
        """Update job status with atomic operation."""

    async def get_job(self, job_id: str) -> Job:
        """Retrieve job by ID."""

    async def list_jobs(
        self,
        status: Optional[JobStatus] = None,
        model_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Job]:
        """List jobs with filters."""
```

---

## 8. Request Flow

### Standard Request Flow (Existing `/v1/*` Endpoints)

```
Client → FastAPI → Middleware (Request ID) → Endpoint → Handler → Model → Response
                                                  ↓
                                            Queue (if busy)
```

### Fusion Request Flow

```
Client → POST /api/fusion/orchestrate
    ↓
Fusion Coordinator
    ↓
Model Router (select models based on workflow)
    ↓
Workflow Engine (execute steps)
    ├─→ Model A (retrieval)
    ├─→ Model B (rerank)
    └─→ Model C (generation)
    ↓
Aggregate Results → Response
```

### RAG Request Flow

```
Client → POST /api/rag/query
    ↓
RAG Generator
    ├─→ Embedding Model (embed query)
    ├─→ Vector Store (semantic search)
    ├─→ Reranker (optional)
    └─→ LM (generate with context)
    ↓
RAG Result with Citations → Response
```

### MCP Job Flow

```
Tier 2 → POST /api/mcp/jobs
    ↓
MCP Handler
    ├─→ Create job in MongoDB (status: queued)
    ├─→ Add to persistent queue (Redis)
    └─→ Return job_id
    ↓
Background Worker
    ├─→ Dequeue job
    ├─→ Update status: processing
    ├─→ Execute model inference
    ├─→ Update status: completed/failed
    └─→ Trigger webhook → Tier 2
```

---

## 9. Data Schemas

### Fusion Schemas

```python
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class FusionWorkflow(BaseModel):
    workflow_id: str
    name: str
    steps: List[WorkflowStep]

class WorkflowStep(BaseModel):
    step_id: str
    type: str  # "model_call", "conditional", "parallel"
    model_id: Optional[str] = None
    operation: str
    input: Any

class FusionRequest(BaseModel):
    workflow: str  # Predefined workflow ID or "custom"
    input: Dict[str, Any]
    models: Optional[Dict[str, str]] = None  # Override default models

class FusionResult(BaseModel):
    workflow_id: str
    status: str
    result: Any
    execution_trace: List[ExecutionStep]
    total_latency_ms: int
```

### RAG Schemas

```python
class RAGIngestRequest(BaseModel):
    collection: str
    chunk_size: int = 512
    overlap: int = 64

class RAGQueryRequest(BaseModel):
    query: str
    collection: str = "default"
    top_k: int = 5
    model_id: Optional[str] = None
    include_sources: bool = True
    similarity_threshold: float = 0.7

class RAGResult(BaseModel):
    answer: str
    sources: List[DocumentChunk]
    retrieval_latency_ms: int
    generation_latency_ms: int
```

### MCP Schemas

```python
from enum import Enum

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobCreate(BaseModel):
    model_id: str
    operation: str  # "chat", "embed", "image_gen", etc.
    input: Dict[str, Any]
    callback_url: Optional[str] = None
    priority: str = "normal"  # "low", "normal", "high"
    metadata: Optional[Dict[str, Any]] = None

class Job(BaseModel):
    job_id: str
    status: JobStatus
    model_id: str
    operation: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
```

---

## 10. Deployment Architecture

### Single-Node Deployment (Phase-4 Initial)

```
┌────────────────────────────────────────────────┐
│         MacBook Pro (Apple Silicon)            │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │      mlx-openai-server (FastAPI)         │ │
│  │                                          │ │
│  │  ┌─────────────┬─────────────┬─────────┐│ │
│  │  │ MLX Provider│ RAG Provider│   MCP   ││ │
│  │  └─────────────┴─────────────┴─────────┘│ │
│  │                                          │ │
│  │  ┌─────────────────────────────────────┐│ │
│  │  │    Fusion Orchestration Layer       ││ │
│  │  └─────────────────────────────────────┘│ │
│  └──────────────────────────────────────────┘ │
│                                                │
│  ┌──────────────────────────────────────────┐ │
│  │  MongoDB (local)  │  Redis (local)       │ │
│  │  ChromaDB (local) │  MLX Models (local)  │ │
│  └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────┘
```

### Multi-Node Deployment (Future)

```
┌────────────────────┐     ┌────────────────────┐
│   Node 1 (M1 Pro)  │     │  Node 2 (M1 Ultra) │
│                    │     │                    │
│  MLX Provider      │     │  MLX Provider      │
│  + Fusion Layer    │     │  + Fusion Layer    │
└────────────────────┘     └────────────────────┘
         ↓                          ↓
         └──────────┬───────────────┘
                    ↓
         ┌──────────────────────┐
         │  Shared MongoDB      │
         │  Shared Redis Queue  │
         └──────────────────────┘
```

---

## 11. Security & Access Control

### Authentication (Future Enhancement)

- **API Keys**: Optional API key authentication for production deployments
- **JWT Tokens**: Support for JWT-based auth from Tier 2
- **Rate Limiting**: Per-key or per-IP rate limits

### Data Privacy

- **No Logging of Sensitive Data**: Request bodies not logged by default
- **Sanitized Job Storage**: Only metadata stored in MongoDB, not full payloads
- **TTL on Jobs**: Auto-delete job history after configurable period (default: 7 days)

### Network Security

- **HTTPS Support**: TLS/SSL for production deployments
- **CORS**: Configurable CORS policies
- **Webhook Signing**: HMAC signatures for webhook payloads

---

## 12. Performance Considerations

### Concurrency

- **Per-Model Limits**: Each model can have independent concurrency limits
- **Global VRAM Limit**: Prevent OOM by monitoring total VRAM usage
- **Adaptive Concurrency**: Auto-tune based on latency and error rates

### Caching

- **KV Cache Warmup**: Reduce first-token latency (existing)
- **Embedding Cache**: Cache embeddings for frequently accessed documents
- **Model Weight Cache**: Prevent re-downloading models

### Resource Management

- **Model Eviction**: LRU policy with configurable TTL
- **Queue Backpressure**: Reject requests when queue is full (HTTP 429)
- **Graceful Degradation**: Return partial results if some workflow steps fail

### Benchmarks (Target)

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Chat completion (7B model) | < 200ms | First token |
| Embedding (512 tokens) | < 50ms | Single batch |
| RAG query (with retrieval) | < 300ms | Top-5 retrieval |
| Fusion workflow (3 steps) | < 500ms | Sequential |
| Job submission (MCP) | < 10ms | Queue only |

---

## 13. Future Enhancements

### Phase-5+ Roadmap

1. **Distributed Inference**
   - Multi-node model serving
   - Work stealing for load balancing
   - Consensus-based job scheduling

2. **Advanced Fusion**
   - Conditional branching in workflows
   - Looping and recursion
   - User-defined workflow DSL

3. **Enhanced RAG**
   - Multi-collection search
   - Hybrid search (semantic + keyword)
   - RAG with memory (conversation history)
   - Parent document retrieval

4. **Model Optimization**
   - Model quantization on-the-fly
   - LoRA adapter management
   - Model compression and pruning

5. **Observability**
   - Real-time dashboard (web UI)
   - Distributed tracing (Jaeger)
   - Anomaly detection (latency spikes)

6. **Enterprise Features**
   - Multi-tenancy support
   - Role-based access control (RBAC)
   - Audit logging
   - SLA monitoring and alerting

---

## Appendix A: Configuration

### Environment Variables

```bash
# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=mlx_fusion

# Redis
REDIS_URI=redis://localhost:6379
REDIS_DB=0

# Vector Database
VECTOR_DB_TYPE=chromadb  # chromadb, faiss, mlx
VECTOR_DB_PATH=/data/vector_store

# VRAM Management
MAX_VRAM_GB=32.0
MODEL_EVICTION_POLICY=lru  # lru, priority, manual
MODEL_TTL_SECONDS=3600

# Job Management
JOB_TTL_DAYS=7
WEBHOOK_TIMEOUT_SECONDS=10
WEBHOOK_RETRY_COUNT=3

# Performance
DEFAULT_CONCURRENCY=2
MAX_QUEUE_SIZE=100
QUEUE_TIMEOUT_SECONDS=300
```

### CLI Configuration

```bash
# Launch with Phase-4 features enabled
python -m app.main \
  --enable-fusion \
  --enable-rag \
  --enable-mcp \
  --mongo-uri mongodb://localhost:27017 \
  --redis-uri redis://localhost:6379 \
  --max-models 3 \
  --max-vram-gb 32
```

---

## Appendix B: Migration Path

### From Phase-3 to Phase-4

1. **Backward Compatibility**: All existing `/v1/*` endpoints continue to work
2. **Gradual Adoption**: New `/api/fusion/*`, `/api/rag/*`, `/api/mcp/*` endpoints are additive
3. **Feature Flags**: Enable Phase-4 features via CLI flags or env vars
4. **Data Migration**: No breaking changes to existing data structures

### Migration Steps

1. **Install Dependencies**:
   ```bash
   pip install motor redis chromadb
   ```

2. **Start MongoDB and Redis**:
   ```bash
   brew install mongodb-community redis
   brew services start mongodb-community
   brew services start redis
   ```

3. **Launch with Phase-4 Enabled**:
   ```bash
   python -m app.main \
     --enable-fusion \
     --enable-rag \
     --enable-mcp \
     --model-path mlx-community/qwen2.5-7b-instruct-4bit \
     --model-type lm
   ```

4. **Test Phase-4 Features**:
   ```bash
   # Load additional model
   curl -X POST http://localhost:8000/v1/models/load \
     -d '{"model_path": "mlx-community/gemma-3-4b-it-4bit", "model_type": "lm"}'

   # Run fusion workflow
   curl -X POST http://localhost:8000/api/fusion/orchestrate \
     -d '{"workflow": "sequential", "input": {...}}'
   ```

---

**End of Phase-4 Architecture Document**
