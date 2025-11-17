# HANDOFFS – Session Log

This document tracks session-to-session handoffs for the `mlx-openai-server-lab` fusion engine project. Each session appends a new entry with discoveries, actions, and next steps.

---

## Session 1: Phase 0 – Engine Warm-Up
**Date**: 2025-11-16
**Branch**: `claude/phase0-engine-warmup-012M2QbGXpRukoMy8x4jyVrg`
**Goal**: Map the existing codebase and document architecture for Phase 1 transformation

### Discoveries

**Architecture Overview:**
- FastAPI-based OpenAI-compatible API server running on Apple Silicon (MLX)
- Stateless design: no database, no persistent storage, all config via CLI
- Single-model-per-instance: model loaded at startup, no runtime switching
- Async request queue with semaphore-based concurrency control
- Supports 6 model types: lm, multimodal, embeddings, whisper, image-generation, image-edit

**Key Components Identified:**
1. **Entrypoints**: CLI via Click (`app/cli.py:145`) or argparse (`app/main.py:50`), script entry in `pyproject.toml:56`
2. **Model Loading**: Model wrappers in `app/models/` (mlx_lm, mlx_vlm, etc.), handlers in `app/handler/` wrapping models with queue logic
3. **Configuration**: All via CLI args (no config files), defaults from env vars in `app/models/mlx_lm.py:15-21`
4. **HTTP Routing**: Single router in `app/api/endpoints.py` with 8 endpoints (health, models, queue stats, chat, embeddings, images, audio)
5. **Concurrency**: `app/core/queue.py` RequestQueue with `asyncio.Semaphore`, defaults: max_concurrency=1, timeout=300s, queue_size=100

**Current Limitations:**
- **No model registry**: Can't switch models without restarting server
- **No persistent queue**: Queue is in-memory, lost on restart
- **No state management**: Completely stateless (good for Tier 3, but needs Tier 2 integration)
- **Sequential by default**: max_concurrency=1 means only one request processed at a time
- **No multi-model support**: Can only load one model per server instance
- **No job tracking**: No request history, logs, or status persistence

**Code Quality Observations:**
- Well-structured, clear separation of concerns (models, handlers, API)
- Good error handling and logging (loguru)
- Memory-conscious: explicit garbage collection and MLX cache clearing
- OpenAI compatibility: Follows OpenAI API schemas closely
- Extensible: Easy to add new model types via handler pattern

### Actions Taken

1. ✅ Scanned repository structure (42 Python files, 8 handler types, 5 model wrappers)
2. ✅ Identified main entrypoints: `app/cli.py`, `app/main.py`, `pyproject.toml` script entry
3. ✅ Mapped model loading flow: `app/models/` → `app/handler/` → `app/main.py` lifespan
4. ✅ Documented config handling: CLI args only, no config files
5. ✅ Catalogued HTTP endpoints: 8 routes in `app/api/endpoints.py`
6. ✅ Analyzed concurrency system: RequestQueue with semaphore in `app/core/queue.py`
7. ✅ Created `docs/FUSION_PHASE0.md` with comprehensive architecture documentation:
   - Executive summary
   - High-level code map with exact file paths and line numbers
   - Dataflow diagram
   - Complete API surface table
   - Concurrency configuration details
   - Model lifecycle and memory management
   - Dependencies and tech stack
   - Gaps and future work (TODOs for Phase 1)
8. ✅ Created `docs/HANDOFFS.md` (this file) for session tracking

### Next Actions for Phase 1: Fusion Engine Transformation

**Goal**: Transform this stateless inference server into a **multi-model fusion engine** that integrates with Tier 2 (MCP) for state management and orchestration.

#### Priority 1: Model Registry & Management
1. **Implement model registry** (`app/core/model_registry.py`):
   - Registry class to track loaded models by ID
   - Support multiple models loaded simultaneously
   - Model metadata: type, capabilities, context length, VRAM usage
   - Model loading/unloading API (hot-swap without restart)

2. **Add model routing logic** (`app/api/endpoints.py`):
   - Route requests to appropriate model by `model` parameter
   - Validate model exists before processing request
   - Return 404 if model not found

3. **Create model management endpoints**:
   - `POST /v1/models/load` – Load a new model
   - `DELETE /v1/models/{model_id}/unload` – Unload a model
   - `GET /v1/models/{model_id}/info` – Get model metadata
   - `GET /v1/models/{model_id}/stats` – Get model usage stats

#### Priority 2: Persistent Queue & Job Tracking
4. **Integrate MongoDB for job tracking**:
   - Job schema: request ID, model ID, status, timestamps, input/output
   - Status enum: queued, processing, completed, failed, cancelled
   - Store request metadata (no full request body for privacy)

5. **Implement job status API**:
   - `GET /v1/jobs/{job_id}` – Get job status
   - `GET /v1/jobs` – List jobs (with filters: status, model, time range)
   - `DELETE /v1/jobs/{job_id}` – Cancel a job

6. **Persist queue to Redis** (optional):
   - Replace in-memory queue with Redis-backed queue
   - Survive restarts without losing pending requests
   - Enable distributed queue for multi-node setup

#### Priority 3: Tier 2 Integration
7. **Define Tier 2 ↔ Tier 3 protocol**:
   - MCP (Tier 2) sends job requests to Tier 3 via HTTP
   - Tier 3 reports job status back to MCP (webhooks or polling)
   - Shared MongoDB for state synchronization

8. **Add health metrics endpoint** (`/v1/health/metrics`):
   - VRAM usage (current, peak)
   - Model inference latency (p50, p95, p99)
   - Queue depth and throughput
   - Active models and concurrency

9. **Implement callback/webhook system**:
   - Tier 3 notifies Tier 2 when job completes
   - POST to configurable webhook URL with job result
   - Retry logic for failed notifications

#### Priority 4: Observability & Performance
10. **Add Prometheus metrics export** (`/metrics`):
    - Request rate, error rate, latency
    - Queue stats (depth, wait time)
    - Model stats (inference time, VRAM usage)
    - Memory stats (gc count, cache clears)

11. **Implement request ID tracking**:
    - Generate unique request ID for each request
    - Pass request ID to Tier 2 for correlation
    - Include request ID in all logs

12. **Optimize concurrency defaults**:
    - Benchmark different concurrency levels for common models
    - Document recommended concurrency by model size
    - Consider adaptive concurrency (auto-tune based on VRAM/latency)

#### Priority 5: Code Refactoring (Low Priority)
13. **Extract config to dataclass** (optional):
    - Centralize all config in `app/core/config.py`
    - Replace arg parsing with pydantic settings
    - Support config file loading (YAML/TOML)

14. **Add integration tests**:
    - Test multi-model loading and routing
    - Test job tracking and status updates
    - Test Tier 2 integration (mock MCP)

15. **Document Tier 2 ↔ Tier 3 contract**:
    - API spec for job submission
    - Job status schema
    - Webhook payload format

### Risks & Open Questions

**Risks:**
1. **VRAM constraints**: Loading multiple models simultaneously may exceed available VRAM
   - *Mitigation*: Implement model LRU eviction, lazy loading, or VRAM monitoring
2. **Concurrency complexity**: Managing multiple models with different concurrency limits
   - *Mitigation*: Per-model queue configuration, global semaphore for VRAM
3. **State synchronization**: Keeping Tier 2 and Tier 3 state consistent
   - *Mitigation*: Single source of truth (MongoDB), atomic operations, idempotency

**Open Questions:**
1. Should Tier 3 own the model registry, or should Tier 2 dictate which models to load?
   - *Recommendation*: Tier 2 owns configuration, Tier 3 manages lifecycle
2. How to handle model loading failures (e.g., out of VRAM)?
   - *Recommendation*: Return 503 Service Unavailable, log error, notify Tier 2
3. Should job history be stored indefinitely, or pruned after N days?
   - *Recommendation*: Configurable TTL (e.g., 7 days), archive to S3/disk if needed
4. Should Tier 3 support streaming to Tier 2, or only non-streaming?
   - *Recommendation*: Support both, use SSE for streaming, JSON for non-streaming
5. How to handle model version updates (e.g., model repo changes)?
   - *Recommendation*: Treat as new model ID, allow side-by-side deployment, manual cutover

### Files Changed
- ✅ Created `docs/FUSION_PHASE0.md` (architecture documentation)
- ✅ Created `docs/HANDOFFS.md` (session log, this file)

### Files to Change in Phase 1
- `app/core/model_registry.py` (NEW) – Model registry implementation
- `app/core/job_tracker.py` (NEW) – MongoDB job tracking
- `app/api/endpoints.py` – Add model management and job status endpoints
- `app/main.py` – Update lifespan to support multi-model loading
- `app/handler/*.py` – Update handlers to report job status
- `app/schemas/openai.py` – Add job status schemas
- `README.md` – Update with new multi-model capabilities
- `docs/TIER2_INTEGRATION.md` (NEW) – Document Tier 2 ↔ Tier 3 protocol

---

## Session 2: Phase-4 Readiness Snapshot
**Date**: 2025-11-17
**Branch**: `claude/phase4-repo-snapshot-01F2YXdaBv8rGTWHnc8MPqPv`
**Goal**: Generate complete current-state repository snapshot to enable Phase-4 fusion implementation without guesswork

### Discoveries

**Current State Summary:**
- Repository is in **clean state** (no uncommitted changes, Phase-3 branches merged)
- **Readiness Score: 5/10** ⚠️ Foundation ready, but fusion layer completely missing
- Tier 3A MLX provider is **production-ready** (8 stable OpenAI-compatible endpoints)
- **Perfect local-first implementation** (zero cloud dependencies, 100% local MLX inference)
- Tier 2 integration features **fully implemented** (health endpoints, request tracking, rich metadata)
- Model registry exists but **single-model limitation** (cannot load multiple models simultaneously)

**Critical Gaps Identified:**
1. ❌ **No Fusion Orchestrator** - No `/api/fusion/*` endpoints, single-model-per-instance only
2. ❌ **No RAG Provider** - No `/api/rag/*` endpoints, only example notebooks
3. ❌ **No MCP Server** - No `/api/mcp/*` endpoints, no state management layer
4. ❌ **No Persistent State** - Completely stateless (MongoDB/Redis not integrated)
5. ❌ **No Job Tracking** - No job history, status persistence, or database integration
6. ❌ **No Multi-Model Support** - Registry exists but only supports one model at a time

**Service Architecture Analysis:**
- **MLX Provider (Tier 3A)**: ✅ Fully operational, OpenAI-compatible, request queue with concurrency control
- **RAG Provider**: ❌ Not present (only `simple_rag_demo.ipynb` example)
- **Fusion Orchestrator**: ❌ Not present (FUSION_PHASE0.md documents Phase 0 only)
- **MCP Server**: ❌ Not present (Tier 2 integration planned but not implemented)
- **Health Endpoints**: ✅ Tier 2-compliant (`/health` with warmup status, `/v1/queue/stats`)

**Registry System Audit:**
- **Models Registry**: ✅ Implemented (`app/core/model_registry.py`) but single-model only
  - Rich metadata: `id`, `type`, `family`, `description`, `context_length`, `tags`, `tier`
  - Local-first entries only (no Ollama/OpenAI/Anthropic/Gemini artifacts)
  - Registry endpoints correct (`GET /v1/models`, `GET /v1/models/{id}`)
- **Tools Registry**: ❌ Not present (tool calling via OpenAI format, but no registry)
- **Apps Registry**: ❌ Not present (no application-level orchestration)

**API Surface Inventory:**
- **Fully Implemented**: 9 endpoints (health, models, queue stats, chat, embeddings, images, audio)
- **Missing Phase-4 Routes**: `/api/fusion/*`, `/api/rag/*`, `/api/mcp/*`, `/internal/*` (diagnostics)
- **OpenAI Compatibility**: ✅ Stable, production-ready (streaming & non-streaming chat completions)

**Tests & Tooling:**
- 3 test files (517 lines total): endpoints, model registry, tool parser
- ⚠️ pytest not available in environment (cannot verify test execution)
- Missing: integration tests, fusion/orchestration tests, multi-model scenarios
- Dependencies clean, no MongoDB/Redis clients installed yet
- Scripts: health dashboard and contract validation (not integrated)

### Actions Taken

1. ✅ Analyzed git state (clean working tree, Phase-3 branches merged)
2. ✅ Audited service architecture (MLX provider operational, fusion/RAG/MCP absent)
3. ✅ Examined API surface (9 stable endpoints, Phase-4 routes missing)
4. ✅ Reviewed registry system (models registry implemented, tools/apps registries absent)
5. ✅ Assessed local-first purity (100% local, zero cloud dependencies)
6. ✅ Evaluated test coverage (basic tests present, integration tests missing)
7. ✅ Calculated Phase-4 readiness score (5/10 - foundation ready, fusion layer missing)
8. ✅ Created **comprehensive snapshot report** (`docs/PHASE4_SNAPSHOT.md`):
   - Repository metadata and git state
   - Service status (Tier 2 & Tier 3)
   - Complete API surface audit with route-by-route status
   - Registry system state (models, tools, apps)
   - Tests & tooling infrastructure review
   - Phase-4 readiness score with dimension breakdown
   - Prioritized next steps (Critical → Low priority)
   - Key file locations appendix
9. ✅ Updated session handoff log (`docs/HANDOFFS.md` - this entry)

### Next Actions for Phase-4 Implementation

**CRITICAL PATH (Must Complete Before Any Phase-4 Coding):**

#### 1. Architecture Definition (Week 1)
- [ ] **Document Fusion Architecture**: Create `docs/PHASE4_ARCHITECTURE.md`
  - Define fusion orchestration service contract
  - Specify multi-model routing strategy
  - Document RAG provider interface
  - Define MCP server protocol and Tier 2 ↔ Tier 3 integration
  - Design state management schema (MongoDB collections)

- [ ] **Create API Specifications**: Create `docs/PHASE4_API_SPEC.md`
  - `/api/fusion/*` endpoints specification
  - `/api/rag/*` endpoints specification
  - `/api/mcp/*` endpoints specification
  - `/internal/diagnostics` endpoint specification
  - Request/response schemas with examples

#### 2. Foundation Infrastructure (Week 1-2)
- [ ] **Multi-Model Registry Enhancement**: Extend `app/core/model_registry.py`
  - Support multiple concurrent models (currently single-model only)
  - Add VRAM monitoring and model eviction policies
  - Implement model loading queue (prevent VRAM exhaustion)
  - Add model capability discovery and tagging

- [ ] **State Management Integration**: Create `app/core/state_manager.py`
  - Integrate MongoDB client (motor) for job tracking
  - Define job schema: `{job_id, model_id, status, created_at, completed_at, metadata}`
  - Status enum: `queued`, `processing`, `completed`, `failed`, `cancelled`
  - Implement job CRUD operations

- [ ] **Persistent Queue**: Create `app/core/persistent_queue.py`
  - Redis or MongoDB-backed queue (survive restarts)
  - Queue persistence for pending requests
  - Job recovery on server restart

#### 3. Fusion Orchestrator (Week 2-3)
- [ ] **Fusion Coordinator Service**: Create `app/fusion/coordinator.py`
  - Multi-model workflow orchestration
  - Model capability-based routing
  - Request decomposition and parallel execution
  - Response aggregation and formatting

- [ ] **Fusion API Endpoints**: Update `app/api/endpoints.py`
  - `POST /api/fusion/orchestrate` - Multi-model workflow execution
  - `GET /api/fusion/workflows` - List available fusion workflows
  - `GET /api/fusion/capabilities` - Model capability discovery
  - `POST /api/fusion/compose` - Compose custom workflows

- [ ] **Model Management API**: Update `app/api/endpoints.py`
  - `POST /v1/models/load` - Load additional model (multi-model support)
  - `DELETE /v1/models/{id}/unload` - Unload specific model
  - `GET /v1/models/{id}/stats` - Model usage statistics
  - `POST /v1/models/{id}/warmup` - Warmup model without inference

#### 4. RAG Provider (Week 3-4)
- [ ] **RAG Service Layer**: Create `app/rag/` module
  - `app/rag/document_processor.py` - Document ingestion and chunking
  - `app/rag/vector_store.py` - Vector database integration (ChromaDB or FAISS)
  - `app/rag/retriever.py` - Semantic search and retrieval
  - `app/rag/generator.py` - RAG generation with MLX models

- [ ] **RAG API Endpoints**: Update `app/api/endpoints.py`
  - `POST /api/rag/ingest` - Ingest documents into vector store
  - `POST /api/rag/query` - RAG query with retrieval + generation
  - `GET /api/rag/documents` - List ingested documents
  - `DELETE /api/rag/documents/{id}` - Remove document from store
  - `GET /api/rag/collections` - List vector collections

- [ ] **RAG Configuration**: Add to `app/core/config.py`
  - Vector store backend selection (ChromaDB, FAISS, MLX-native)
  - Chunk size and overlap configuration
  - Retrieval parameters (top-k, similarity threshold)
  - Embedding model selection

#### 5. MCP Server (Week 4-5)
- [ ] **MCP Protocol Handlers**: Create `app/mcp/` module
  - `app/mcp/protocol.py` - MCP protocol implementation
  - `app/mcp/handlers.py` - Request/response handlers
  - `app/mcp/state_sync.py` - Tier 2 state synchronization

- [ ] **MCP API Endpoints**: Update `app/api/endpoints.py`
  - `POST /api/mcp/jobs` - Submit job to MCP queue
  - `GET /api/mcp/jobs/{id}` - Get job status
  - `GET /api/mcp/jobs` - List jobs (filterable by status, model, time)
  - `DELETE /api/mcp/jobs/{id}` - Cancel pending job
  - `POST /api/mcp/webhook` - Register webhook for job notifications

- [ ] **Job Tracking System**: Create `app/core/job_tracker.py`
  - Job lifecycle management (queued → processing → completed/failed)
  - Job history and audit trail
  - Webhook/callback system for Tier 2 notifications
  - Job retention policy and cleanup

#### 6. Observability & Diagnostics (Week 5-6)
- [ ] **Metrics Export**: Create `app/observability/metrics.py`
  - Prometheus metrics export (`/metrics` endpoint)
  - Request rate, error rate, latency (p50, p95, p99)
  - Queue depth, throughput, concurrency
  - VRAM usage, model inference time, memory stats

- [ ] **Distributed Tracing**: Create `app/observability/tracing.py`
  - OpenTelemetry integration
  - Request ID propagation across services
  - Trace collection and export

- [ ] **Diagnostics Endpoint**: Create `/internal/diagnostics`
  - System health (VRAM, CPU, memory)
  - Model status and statistics
  - Queue health and backlog
  - Recent errors and warnings
  - Configuration dump

#### 7. Testing & Documentation (Week 6)
- [ ] **Integration Tests**: Expand `tests/`
  - `tests/test_multi_model.py` - Multi-model loading and routing
  - `tests/test_fusion.py` - Fusion orchestration workflows
  - `tests/test_rag.py` - RAG retrieval and generation
  - `tests/test_mcp.py` - MCP protocol and job tracking
  - `tests/test_tier2_integration.py` - End-to-end Tier 2 ↔ Tier 3

- [ ] **Documentation Updates**:
  - `docs/PHASE4_ARCHITECTURE.md` - Architecture overview
  - `docs/PHASE4_API_SPEC.md` - API specifications
  - `docs/FUSION_GUIDE.md` - Fusion orchestration guide
  - `docs/RAG_GUIDE.md` - RAG provider integration guide
  - `docs/MCP_PROTOCOL.md` - MCP server protocol specification
  - Update `README.md` - Phase-4 capabilities and examples

### Risks & Open Questions

**Risks:**
1. **VRAM Exhaustion**: Loading multiple models may exceed available VRAM
   - *Mitigation*: Implement LRU model eviction, lazy loading, VRAM monitoring, model pooling
2. **State Consistency**: Keeping Tier 2 and Tier 3 state synchronized
   - *Mitigation*: MongoDB as single source of truth, atomic operations, idempotency
3. **Complexity Explosion**: Fusion orchestration adds significant complexity
   - *Mitigation*: Start with simple workflows, incremental feature rollout, comprehensive testing
4. **Performance Degradation**: Multi-model routing overhead
   - *Mitigation*: Benchmark early, optimize hot paths, implement request batching

**Open Questions:**
1. **Model Ownership**: Should Tier 3 own model registry, or Tier 2?
   - *Recommendation*: Tier 2 dictates which models to load, Tier 3 manages lifecycle
2. **Model Loading Failures**: How to handle out-of-VRAM errors?
   - *Recommendation*: Return 503, log error, notify Tier 2, suggest model eviction
3. **Job History Retention**: Store indefinitely or prune?
   - *Recommendation*: Configurable TTL (default 7 days), archive to cold storage if needed
4. **Streaming Support**: Should Tier 3 stream to Tier 2?
   - *Recommendation*: Support both streaming (SSE) and non-streaming (JSON)
5. **Vector Store**: Which vector database for RAG?
   - *Recommendation*: Start with ChromaDB (easy setup), add FAISS/MLX-native as alternatives
6. **Fusion Complexity**: How complex should fusion workflows be?
   - *Recommendation*: Start simple (sequential, parallel), add conditional/branching later
7. **MCP Protocol**: Should we implement full MCP spec or subset?
   - *Recommendation*: Minimal viable protocol first, expand based on Tier 2 needs

### Files Changed

**Created:**
- ✅ `docs/PHASE4_SNAPSHOT.md` - Complete Phase-4 readiness snapshot report

**Updated:**
- ✅ `docs/HANDOFFS.md` - Added Session 2 handoff entry (this section)

### Files to Create in Phase-4

**Architecture & Specs:**
- `docs/PHASE4_ARCHITECTURE.md` - Phase-4 architecture design
- `docs/PHASE4_API_SPEC.md` - Complete API specifications
- `docs/FUSION_GUIDE.md` - Fusion orchestration guide
- `docs/RAG_GUIDE.md` - RAG provider integration guide
- `docs/MCP_PROTOCOL.md` - MCP server protocol spec

**Core Infrastructure:**
- `app/core/state_manager.py` - MongoDB state management
- `app/core/persistent_queue.py` - Redis/MongoDB queue
- `app/core/job_tracker.py` - Job lifecycle management
- `app/core/config.py` - Centralized configuration

**Fusion Layer:**
- `app/fusion/coordinator.py` - Fusion orchestration coordinator
- `app/fusion/workflows.py` - Workflow definitions
- `app/fusion/router.py` - Model routing logic

**RAG Layer:**
- `app/rag/document_processor.py` - Document ingestion
- `app/rag/vector_store.py` - Vector database interface
- `app/rag/retriever.py` - Semantic retrieval
- `app/rag/generator.py` - RAG generation

**MCP Layer:**
- `app/mcp/protocol.py` - MCP protocol implementation
- `app/mcp/handlers.py` - Request handlers
- `app/mcp/state_sync.py` - State synchronization

**Observability:**
- `app/observability/metrics.py` - Prometheus metrics
- `app/observability/tracing.py` - OpenTelemetry tracing

**Testing:**
- `tests/test_multi_model.py` - Multi-model tests
- `tests/test_fusion.py` - Fusion orchestration tests
- `tests/test_rag.py` - RAG provider tests
- `tests/test_mcp.py` - MCP protocol tests
- `tests/test_tier2_integration.py` - Tier 2 ↔ Tier 3 integration tests

**Schemas:**
- `app/schemas/fusion.py` - Fusion request/response schemas
- `app/schemas/rag.py` - RAG request/response schemas
- `app/schemas/mcp.py` - MCP protocol schemas
- `app/schemas/jobs.py` - Job tracking schemas

---

## Session 3: TBD
**Date**: TBD
**Branch**: TBD
**Goal**: TBD

### Discoveries
(Append here)

### Actions Taken
(Append here)

### Next Actions
(Append here)

### Risks & Open Questions
(Append here)

### Files Changed
(Append here)
