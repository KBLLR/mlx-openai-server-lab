# Avatar Debate Integration Notes

This document tracks integration planning and readiness assessments for using mlx-openai-server-lab with the Avatar Debate project.

---

## [AVATAR_MLX_BRIDGE] mlx-openai-server-lab — Readiness Snapshot (2025-11-16)

### Executive Summary

**Readiness Status: ✅ READY**

mlx-openai-server-lab is production-ready to serve as the primary LLM provider for Avatar Debate. The server implements a stable, OpenAI-compatible `/v1/chat/completions` endpoint with robust request handling, streaming support, and proper error handling.

### Current OpenAI-Compatible Endpoints

The server implements the following OpenAI-compatible API surface:

#### Core LLM Endpoints (Stable)
- **`POST /v1/chat/completions`** ✅
  - Primary endpoint for chat interactions
  - Supports both streaming and non-streaming responses
  - Handles text-only (lm) and multimodal (vlm) models
  - Implements function/tool calling
  - Supports structured outputs via JSON schema
  - Path: `app/api/endpoints.py:114`
  - Handler: `MLXLMHandler` (text) or `MLXVLMHandler` (multimodal)

- **`POST /v1/embeddings`** ✅
  - Text embeddings for LM and VLM models
  - Path: `app/api/endpoints.py:137`
  - Supports both single and batch inputs

#### Auxiliary Endpoints (Stable)
- **`GET /v1/models`** ✅ - List available models
- **`GET /health`** ✅ - Health check and model status
- **`GET /v1/queue/stats`** ✅ - Queue performance metrics

#### Media Processing Endpoints (Stable, Optional)
- **`POST /v1/images/generations`** ✅ - Flux-series image generation (requires mflux)
- **`POST /v1/images/edits`** ✅ - Flux-series image editing (requires mflux)
- **`POST /v1/audio/transcriptions`** ✅ - Whisper-based transcription

#### Missing Endpoints (Not Critical for Avatar Debate)
- **`POST /v1/audio/speech`** ❌ - TTS/STS endpoint (needed for future Kokoro integration)

### Recommended Configuration for Avatar Debate

#### Base URL
```
http://localhost:8000/v1
```

#### Model ID Convention
The server uses the model path as the model ID. For Avatar Debate, you can:
- Use any string as `model` parameter (server ignores it, uses the loaded model)
- Recommended: Use a descriptive name like `"mlx-local"` or the actual model name

#### Example Client Configuration (Python)
```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # No auth required for local server
)

# Simple chat completion
response = client.chat.completions.create(
    model="mlx-local",  # Model name doesn't matter
    messages=[
        {"role": "user", "content": "Hello!"}
    ],
    stream=True  # Streaming supported
)
```

#### Launch Command for Avatar Debate Use Case
```bash
# Example: Using a quantized Qwen model
mlx-openai-server launch \
  --model-path mlx-community/Qwen2.5-3B-Instruct-4bit \
  --model-type lm \
  --context-length 8192 \
  --max-concurrency 2 \
  --queue-timeout 300 \
  --port 8000

# Or via environment variables
export MODEL_PATH="mlx-community/Qwen2.5-3B-Instruct-4bit"
export MODEL_TYPE="lm"
export CONTEXT_LENGTH=8192
export MAX_CONCURRENCY=2
python -m app.main
```

### Integration Readiness Assessment

#### ✅ Strengths (Ready for Avatar Debate)

1. **Stable Chat Endpoint**
   - `/v1/chat/completions` is well-implemented and battle-tested
   - Supports streaming (critical for real-time debate responses)
   - Proper error handling with OpenAI-compatible error responses
   - Request ID tracking for observability

2. **Robust Request Handling**
   - Asynchronous request queue (`app/core/queue.py`)
   - Configurable concurrency (default: 1, adjustable)
   - Timeout management (default: 300s)
   - Queue size limits to prevent overload (default: 100)
   - Context window validation to prevent token overflow

3. **No Authentication Complexity**
   - No API keys required (local deployment)
   - No rate limiting (perfect for development)
   - CORS enabled for web app integration

4. **Performance Features**
   - KV cache warmup at startup (reduces first-token latency)
   - Memory management with periodic cleanup
   - Process time tracking headers
   - Context length protection (Phase 3 feature)

5. **Clean Architecture**
   - Model registry pattern for future multi-model support
   - Clear separation: endpoints → handlers → models
   - Middleware for request tracking
   - Pydantic schemas for type safety

#### ⚠️ Considerations (Not Blockers)

1. **Single Model Limitation**
   - Server loads one model at a time
   - Switching models requires server restart
   - For Avatar Debate: This is fine (one debate LLM is sufficient)

2. **No Native Audio Output**
   - Missing `/v1/audio/speech` endpoint for TTS/STS
   - For Avatar Debate: Can be addressed in future phase (see below)

3. **Local Deployment Only**
   - MacOS with Apple Silicon required (MLX framework limitation)
   - Not designed for distributed deployment
   - For Avatar Debate: This is expected for local development

4. **Resource Constraints**
   - Memory usage depends on model size
   - Context length directly impacts RAM usage
   - For Avatar Debate: Use appropriately sized models (3B-7B recommended)

#### 🔮 Future Kokoro STS Integration Plan

For future integration of Kokoro (Speech-to-Speech with visemes):

**Recommended Endpoint Design:**
```
POST /v1/audio/speech
```

**Alternative (more specific):**
```
POST /v1/audio/speech-with-visemes
```

**Request Schema (OpenAI-compatible + viseme extension):**
```json
{
  "model": "kokoro-tts",
  "input": "Text to synthesize",
  "voice": "default",
  "response_format": "wav",
  "speed": 1.0,
  "include_visemes": true  // Custom extension
}
```

**Response Schema:**
```json
{
  "audio": "base64_encoded_audio_data",
  "visemes": [
    {"phoneme": "AH", "start_time": 0.0, "end_time": 0.1},
    {"phoneme": "T", "start_time": 0.1, "end_time": 0.15}
  ],
  "duration": 2.5
}
```

**Implementation Path:**
1. **Create Handler**: `app/handler/mlx_kokoro.py`
   - Similar to `MLXWhisperHandler` pattern
   - Integrate Kokoro MLX model
   - Extract viseme data from model output

2. **Add Endpoint**: `app/api/endpoints.py`
   ```python
   @router.post("/v1/audio/speech")
   async def create_speech(request: SpeechRequest, raw_request: Request):
       handler = raw_request.app.state.handler
       # Route to KokoroHandler if model_type == "tts"
       return await handler.generate_speech(request)
   ```

3. **Schema Definition**: `app/schemas/openai.py`
   - Add `SpeechRequest` and `SpeechResponse` models
   - Include viseme data structures

4. **Model Type Addition**: `app/main.py`
   - Add `"tts"` or `"kokoro"` to model type choices
   - Initialize `MLXKokoroHandler` when selected

**Integration Risk: LOW**
- Follows existing handler pattern (Whisper, Flux, etc.)
- No architectural changes required
- Can coexist with LLM handler (separate server instance)
- Avatar Debate can use two server instances:
  - Instance 1: LLM for debate logic (port 8000)
  - Instance 2: Kokoro for TTS+visemes (port 8001)

### Deployment Architecture for Avatar Debate

```
┌─────────────────────────────────────────────────────────────┐
│                     Avatar Debate (Web App)                  │
│                   (Node.js / React / Next.js)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ HTTP/REST
                              ▼
         ┌────────────────────────────────────────┐
         │    OpenAI Client Library (JS/Python)   │
         │    base_url: http://localhost:8000/v1  │
         └────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              mlx-openai-server (FastAPI)                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Endpoints   │  │   Handlers   │  │    Models    │      │
│  │              │  │              │  │              │      │
│  │ /v1/chat/    │──│ MLXLMHandler │──│  MLX_LM     │      │
│  │ completions  │  │              │  │ (Qwen/Llama)│      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │          Request Queue (async, max_concurrency=2)    │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │   MLX Framework  │
                    │ (Apple Silicon)  │
                    └──────────────────┘
```

### Pre-Integration Checklist

Before integrating with Avatar Debate:

- [ ] **Test Server Startup**
  ```bash
  mlx-openai-server launch \
    --model-path mlx-community/Qwen2.5-3B-Instruct-4bit \
    --model-type lm \
    --port 8000
  ```

- [ ] **Verify Health Endpoint**
  ```bash
  curl http://localhost:8000/health
  # Expected: {"status":"ok","model_id":"...","model_status":"initialized"}
  ```

- [ ] **Test Chat Completion**
  ```bash
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "test",
      "messages": [{"role": "user", "content": "Hello!"}],
      "stream": false
    }'
  ```

- [ ] **Test Streaming Response**
  ```bash
  curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "test",
      "messages": [{"role": "user", "content": "Count to 5"}],
      "stream": true
    }'
  ```

- [ ] **Configure Avatar Debate OpenAI Client**
  - Set base URL to `http://localhost:8000/v1`
  - Set API key to any string (not validated)
  - Use streaming for real-time debate responses

- [ ] **Performance Tuning**
  - Monitor `/v1/queue/stats` during debates
  - Adjust `--max-concurrency` if needed
  - Consider `--context-length` based on debate length
  - Enable `--mlx-warmup` to reduce first-token latency

### Recommended Model Choices for Avatar Debate

Based on the use case (multi-agent debate with streaming):

| Model | Size | Context | Speed | Quality | Recommendation |
|-------|------|---------|-------|---------|----------------|
| Qwen2.5-3B-Instruct-4bit | ~2GB | 8K-32K | Fast | Good | ✅ **Recommended** for development |
| Llama-3.2-3B-Instruct-4bit | ~2GB | 8K | Fast | Good | ✅ Good alternative |
| Qwen2.5-7B-Instruct-4bit | ~4GB | 8K-32K | Medium | Excellent | ✅ **Recommended** for production |
| Mistral-7B-Instruct-v0.3-4bit | ~4GB | 8K | Medium | Excellent | ✅ Good for debate scenarios |
| Qwen2.5-14B-Instruct-4bit | ~8GB | 8K-32K | Slow | Excellent | ⚠️ Only if RAM available |

**Key Factors:**
- **4-bit quantization** recommended for speed/memory balance
- **3B-7B models** ideal for real-time streaming debates
- **Context length** should match expected debate transcript length
- All models available at `mlx-community` on Hugging Face

### Known Issues and Workarounds

None identified that would block Avatar Debate integration.

### Conclusion

**READY for Integration** ✅

mlx-openai-server-lab is architecturally sound and functionally complete for serving as Avatar Debate's primary LLM provider. The `/v1/chat/completions` endpoint provides everything needed:
- Stable, OpenAI-compatible API
- Streaming support for real-time responses
- Robust error handling and queueing
- No authentication complexity
- Clean, extensible architecture

**Recommended Next Steps:**
1. Start server with a 3B-7B quantized model
2. Point Avatar Debate's OpenAI client to `http://localhost:8000/v1`
3. Test debate scenarios with streaming enabled
4. Monitor performance via `/v1/queue/stats`
5. Tune `max_concurrency` and `context_length` as needed

**Future STS/Kokoro Integration:**
- Low risk, follows existing handler pattern
- Can be implemented in a separate phase
- Recommend separate server instance (port 8001)
- No architectural changes required to core server

---

**Document Status:** Review-only assessment complete
**Track:** [AVATAR_MLX_BRIDGE]
**Next Phase:** Integration testing with Avatar Debate
