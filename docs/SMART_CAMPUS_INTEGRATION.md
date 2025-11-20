# Smart Campus Integration Guide

**Version**: 1.0
**Date**: 2025-11-20
**Status**: Ready for Integration

---

## 📋 Overview

This document provides step-by-step instructions for integrating **mlx-openai-server** with the **Smart Campus** platform to enable local, privacy-first AI-powered classroom assistance.

### Architecture

```
┌─────────────────────────────────────┐
│  Smart Campus Frontend              │
│  (ClassroomSelector, ChatSection)   │
└──────────────┬──────────────────────┘
               │ WebSocket/HTTP
               ▼
┌─────────────────────────────────────┐
│  Smart Campus Backend API           │
│  POST /api/classrooms/:id/chat      │
└──────────────┬──────────────────────┘
               │ HTTP + X-Request-ID
               ▼
┌─────────────────────────────────────┐
│  mlx-openai-server (THIS REPO)      │
│  http://localhost:8000              │
│  - Local LLM inference              │
│  - Tool calling support             │
│  - OpenAI-compatible API            │
└─────────────────────────────────────┘
```

### Key Features

✅ **Local-First**: All AI processing happens on-device, no cloud dependencies
✅ **Privacy-Preserving**: Student data never leaves the local network
✅ **Classroom-Scoped**: AI can only access data for the specific classroom
✅ **Tool Integration**: AI can interact with classroom sensors, events, schedules
✅ **Fallback Support**: Graceful degradation when mlx-server is unavailable
✅ **Request Tracking**: End-to-end observability with request IDs

---

## 🚀 Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11+
- Node.js 18+ (for Smart Campus)
- ~8GB free disk space (for model)
- ~8GB free RAM

### Step 1: Install mlx-openai-server

```bash
# Clone this repository (if not already done)
git clone https://github.com/KBLLR/mlx-openai-server-lab.git
cd mlx-openai-server-lab

# Install dependencies (use uv or pip)
pip install -e .

# Verify installation
python -m app.main --help
```

### Step 2: Configure Environment

```bash
# Copy example configuration
cp .env.campus-example .env

# Edit .env and customize settings
# At minimum, verify:
# - MODEL_PATH (default is good for most users)
# - SERVER_HOST=127.0.0.1 (localhost only for security)
# - SERVER_PORT=8000
# - CAMPUS_FRONTEND_URL (match your Smart Campus dev server)
```

### Step 3: Start mlx-openai-server

```bash
# Use the provided startup script
./start-campus-mode.sh

# The first run will download the model (~4GB)
# Subsequent starts are much faster

# You should see:
# ✅ Model found in cache
# 🚀 Starting server...
# INFO | MLX handler initialized successfully
# INFO | CORS allowed origins: ['http://localhost:5173', ...]
```

### Step 4: Verify Server Health

In a separate terminal:

```bash
# Check health endpoint
curl http://localhost:8000/health | jq

# Expected response:
# {
#   "status": "ok",
#   "model_id": "mlx-community/qwen2.5-7b-instruct-4bit",
#   "model_status": "initialized",
#   "models_healthy": true,
#   "warmup_enabled": true,
#   "warmup_completed": true,
#   "latency_ms": 2.34
# }

# Check available models
curl http://localhost:8000/v1/models | jq

# Test basic chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/qwen2.5-7b-instruct-4bit",
    "messages": [
      {"role": "user", "content": "Hello! Can you help with classroom management?"}
    ]
  }' | jq '.choices[0].message.content'
```

If all tests pass, **mlx-openai-server is ready!**

---

## 🔧 Smart Campus Integration

### Step 5: Integrate with Smart Campus API

**Location**: `code-smart-campus/server/api/classrooms.js` (or equivalent)

We've provided a complete reference implementation in:
```
examples/smart-campus-api-reference.js
```

#### Integration Steps:

1. **Copy the reference implementation** to your Smart Campus project
2. **Adapt to your framework** (Express, Fastify, Next.js, etc.)
3. **Replace mock functions** with your actual data sources:
   - `getClassroomById()` → your classroom database query
   - `handleAppendEvent()` → your event storage
   - `handleGetSensorData()` → your sensor API
   - `handleGetSchedule()` → your calendar integration

#### Minimal Integration Example (Express):

```javascript
// In your Smart Campus API routes
const { handleClassroomChat } = require('./smart-campus-api-reference');

// POST /api/classrooms/:id/chat
app.post('/api/classrooms/:id/chat', handleClassroomChat);
```

### Step 6: Configure Smart Campus Environment

Add to `code-smart-campus/.env`:

```bash
# MLX Server Configuration
MLX_SERVER_URL=http://localhost:8000
ENABLE_LOCAL_AI=true
ENABLE_CLOUD_FALLBACK=false
```

### Step 7: Update Smart Campus Frontend (Optional)

If you want to show when AI is using tools:

```jsx
// In ChatSection.jsx or similar
function ChatMessage({ message }) {
  // Detect tool calls
  if (message.tool_calls) {
    return (
      <div className="tool-usage-indicator">
        🔧 AI is accessing classroom data...
        {message.tool_calls.map(tc => (
          <span key={tc.id}>{tc.function.name}</span>
        ))}
      </div>
    );
  }

  return <div>{message.content}</div>;
}
```

---

## 🔒 Security & Privacy

### Security Checklist

- [x] **CORS Restricted**: Only allowed origins can access mlx-server
- [x] **Localhost Binding**: Server binds to 127.0.0.1 by default
- [x] **No Authentication**: mlx-server has no auth (by design - local only)
- [x] **Classroom Scoping**: Tools enforce classroom-level data isolation
- [x] **No Cloud Calls**: All processing happens locally

### Privacy Guarantees

✅ **Student data never leaves the device**
✅ **No telemetry or external API calls**
✅ **Message content not logged** (only metadata)
✅ **Request IDs are opaque UUIDs** (no PII)

### Production Recommendations

1. **Network Isolation**: Run mlx-server on `127.0.0.1` (localhost only)
2. **Reverse Proxy**: If exposing externally, use nginx/caddy with auth
3. **CORS Configuration**: Only allow your Smart Campus frontend origin
4. **Monitoring**: Use `/internal/diagnostics` for observability
5. **Fallback Strategy**: Decide on behavior when mlx-server is unavailable

---

## 🧪 Testing the Integration

### Test 1: Health Check

```bash
# Should return healthy status
curl http://localhost:8000/health
```

### Test 2: Basic Chat (No Tools)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/qwen2.5-7b-instruct-4bit",
    "messages": [
      {"role": "system", "content": "You are a classroom assistant."},
      {"role": "user", "content": "What can you help me with?"}
    ]
  }' | jq
```

### Test 3: Chat with Tool Calling

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: test-tool-call" \
  -d '{
    "model": "mlx-community/qwen2.5-7b-instruct-4bit",
    "messages": [
      {"role": "system", "content": "You manage classroom events."},
      {"role": "user", "content": "Add an announcement that class is cancelled tomorrow"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "append_classroom_event",
        "description": "Add event to classroom timeline",
        "parameters": {
          "type": "object",
          "properties": {
            "event_type": {"type": "string", "enum": ["announcement"]},
            "description": {"type": "string"}
          },
          "required": ["event_type", "description"]
        }
      }
    }]
  }' | jq '.choices[0].message.tool_calls'
```

**Expected**: Response should include `tool_calls` array with function name and arguments.

### Test 4: Streaming Response

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -N \
  -d '{
    "model": "mlx-community/qwen2.5-7b-instruct-4bit",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "stream": true
  }'
```

**Expected**: Server-Sent Events (SSE) stream with incremental tokens.

### Test 5: End-to-End (via Smart Campus)

Once Smart Campus integration is complete:

```bash
# Start both servers
# Terminal 1: mlx-openai-server
./start-campus-mode.sh

# Terminal 2: Smart Campus
cd ../code-smart-campus
npm run dev

# Terminal 3: Test
curl -X POST http://localhost:3000/api/classrooms/room-101/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the current temperature?"}
    ]
  }' | jq
```

**Expected**: AI should call `get_sensor_data` tool and return temperature.

---

## 📊 Monitoring & Observability

### Available Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Basic health check with model status |
| `GET /internal/diagnostics` | Comprehensive system diagnostics |
| `GET /v1/queue/stats` | Request queue statistics |
| `GET /v1/models` | List available models |

### Diagnostics Dashboard

```bash
# Get comprehensive diagnostics
curl http://localhost:8000/internal/diagnostics | jq

# Key metrics:
# - vram.usage_percent: VRAM utilization
# - models[].request_count: Requests per model
# - queue.active: Currently processing requests
# - queue.pending: Queued requests
```

### Log Monitoring

```bash
# Watch logs in real-time
tail -f logs/app.log

# Look for:
# - Request IDs: [request_id=classroom-101-...]
# - Latency: duration=0.234s
# - Tool calls: Processing text request with tools
# - Errors: ERROR level messages
```

### Request ID Tracking

Every request includes a correlation ID:

```javascript
// Smart Campus sets request ID
const requestId = `classroom-${classroomId}-${Date.now()}`;

fetch(MLX_SERVER_URL, {
  headers: { 'X-Request-ID': requestId }
});

// mlx-server logs show same ID
// "Request started: POST /v1/chat/completions [request_id=classroom-101-1700234567]"
```

This enables end-to-end request tracing across services.

---

## 🛠️ Advanced Configuration

### Custom Models

To use a different model:

```bash
# Edit .env
MODEL_PATH=mlx-community/llama-3.2-8b-instruct-4bit

# Or pass as argument
./start-campus-mode.sh
# Then edit the script to use your model
```

### Performance Tuning

```bash
# Adjust context length based on conversation needs
CONTEXT_LENGTH=4096   # Shorter context = less memory
CONTEXT_LENGTH=16384  # Longer context = more memory

# Enable/disable warmup
MLX_WARMUP=true   # Faster first request (default)
MLX_WARMUP=false  # Slower first request, faster startup

# Concurrency (advanced - usually keep at 1)
MAX_CONCURRENCY=1  # Process one request at a time (recommended)
MAX_CONCURRENCY=2  # Process two requests simultaneously (needs more VRAM)
```

### CORS Configuration

```bash
# Single origin
CAMPUS_FRONTEND_URL=http://localhost:5173

# Multiple origins (comma-separated)
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000,https://campus.example.com
```

---

## 🐛 Troubleshooting

### Problem: "MLX server unavailable"

**Symptoms**: Smart Campus shows "AI temporarily unavailable"

**Diagnosis**:
```bash
# Check if server is running
curl http://localhost:8000/health

# Check logs
tail -20 logs/app.log
```

**Solutions**:
- Ensure mlx-server is started: `./start-campus-mode.sh`
- Check `MLX_SERVER_URL` in Smart Campus `.env`
- Verify CORS allows Smart Campus origin

---

### Problem: "Model not loading" or "Weights corrupted"

**Symptoms**: Server starts but health check shows `models_healthy: false`

**Diagnosis**:
```bash
# Check health
curl http://localhost:8000/health | jq '.model_status'

# Check logs for errors
grep -i "error\|failed" logs/app.log
```

**Solutions**:
```bash
# Clear model cache and re-download
rm -rf ~/.cache/huggingface/hub/models--mlx-community--qwen2.5-7b-instruct-4bit
./start-campus-mode.sh  # Will re-download
```

---

### Problem: CORS errors in browser console

**Symptoms**: Browser shows "CORS policy" error

**Diagnosis**: Check browser console network tab

**Solutions**:
```bash
# Verify CORS configuration
curl http://localhost:8000/v1/models -H "Origin: http://localhost:5173" -I

# Should include: Access-Control-Allow-Origin: http://localhost:5173

# If not, update .env
CAMPUS_FRONTEND_URL=http://localhost:5173  # Match your dev server
```

---

### Problem: "Tool calls not working"

**Symptoms**: AI doesn't use tools even when appropriate

**Diagnosis**:
```bash
# Check if tool calling is enabled
grep "ENABLE_AUTO_TOOL_CHOICE" .env
grep "TOOL_CALL_PARSER" .env

# Test tool calling directly
curl -X POST http://localhost:8000/v1/chat/completions \
  -d '{"tools": [...], "messages": [...]}' | jq '.choices[0].message.tool_calls'
```

**Solutions**:
```bash
# Ensure .env has:
ENABLE_AUTO_TOOL_CHOICE=true
TOOL_CALL_PARSER=qwen3  # Must match your model

# Restart server
./start-campus-mode.sh
```

---

### Problem: High memory usage

**Symptoms**: System slows down, memory pressure warnings

**Diagnosis**:
```bash
# Check VRAM usage
curl http://localhost:8000/internal/diagnostics | jq '.vram'
```

**Solutions**:
```bash
# Use smaller context length
CONTEXT_LENGTH=4096  # Instead of 8192

# Use smaller model
MODEL_PATH=mlx-community/qwen2.5-3b-instruct-4bit  # 3B instead of 7B

# Disable warmup (saves ~1GB during startup)
MLX_WARMUP=false
```

---

## 📚 Reference Documentation

### Related Docs

- **[TIER3A_PROVIDER_CONTRACT.md](./TIER3A_PROVIDER_CONTRACT.md)** - Complete API specification
- **[TIER2_INTEGRATION.md](./TIER2_INTEGRATION.md)** - Tier-2 integration patterns
- **[PHASE4_ARCHITECTURE.md](./PHASE4_ARCHITECTURE.md)** - Phase-4 architecture details
- **[README.md](../README.md)** - General mlx-openai-server documentation

### Example Code

- **[examples/smart-campus-api-reference.js](../examples/smart-campus-api-reference.js)** - Complete API implementation
- **[start-campus-mode.sh](../start-campus-mode.sh)** - Startup script
- **[.env.campus-example](../.env.campus-example)** - Configuration template

### Tool Schemas

See `examples/smart-campus-api-reference.js` for complete tool definitions:
- `append_classroom_event` - Add events to classroom timeline
- `get_sensor_data` - Query classroom sensors
- `get_classroom_schedule` - Retrieve schedules
- `search_classroom_events` - Search historical events

---

## 🎯 Integration Checklist

Use this checklist to track your integration progress:

### mlx-openai-server Setup
- [ ] Repository cloned
- [ ] Dependencies installed (`pip install -e .`)
- [ ] Environment configured (`.env` file created)
- [ ] CORS origins configured for Smart Campus
- [ ] Server starts successfully (`./start-campus-mode.sh`)
- [ ] Health check passes (`curl /health`)
- [ ] Model loaded successfully (`models_healthy: true`)
- [ ] Tool calling tested

### Smart Campus Integration
- [ ] Reference implementation copied to Smart Campus repo
- [ ] Mock functions replaced with actual data sources
- [ ] Environment variables added (`MLX_SERVER_URL`, etc.)
- [ ] API route created/updated for classroom chat
- [ ] Classroom tools defined (events, sensors, schedule)
- [ ] Tool execution implemented with classroom scoping
- [ ] Error handling and fallback logic implemented
- [ ] Frontend updated to show tool usage (optional)

### Testing
- [ ] Basic chat works (no tools)
- [ ] Tool calling works (with tools)
- [ ] Streaming responses work
- [ ] End-to-end test via Smart Campus frontend
- [ ] Classroom scoping verified (no cross-classroom leaks)
- [ ] Fallback behavior tested (mlx-server down)
- [ ] Request ID propagation verified in logs

### Production Readiness
- [ ] CORS restricted to production origins
- [ ] Logging configured appropriately
- [ ] Monitoring endpoints accessible
- [ ] Error handling robust
- [ ] Privacy audit completed
- [ ] Performance tested under load
- [ ] Documentation updated for team

---

## 💡 Best Practices

### Do's ✅

- **Always scope tools to classroom ID** - Prevent data leakage
- **Use request IDs** - Enable end-to-end tracing
- **Handle mlx-server failures gracefully** - Show friendly error messages
- **Monitor diagnostics endpoint** - Track VRAM, queue, request counts
- **Log tool usage** - Audit AI actions for compliance
- **Test with real classroom data** - Ensure tools work as expected
- **Restrict CORS** - Only allow trusted origins

### Don'ts ❌

- **Don't log message content** - Privacy risk
- **Don't use student names in request IDs** - Use opaque UUIDs
- **Don't expose mlx-server to public internet** - Localhost only
- **Don't skip classroom validation** - Always verify classroom exists
- **Don't hard-code model names** - Use environment variables
- **Don't ignore health checks** - Gate requests on `models_healthy`
- **Don't disable CORS in production** - Security risk

---

## 🤝 Support & Contributing

### Getting Help

1. **Check logs**: `tail -f logs/app.log`
2. **Review diagnostics**: `curl http://localhost:8000/internal/diagnostics`
3. **Search docs**: Check `docs/` directory
4. **Open an issue**: [GitHub Issues](https://github.com/KBLLR/mlx-openai-server-lab/issues)

### Contributing

Improvements welcome! Areas of interest:
- Additional classroom tools (attendance, assignments, etc.)
- Performance optimizations
- Better error messages
- Integration examples for other platforms

---

## 📝 License

MIT License - See [LICENSE](../LICENSE) for details.

---

**Last Updated**: 2025-11-20
**Maintainer**: Smart Campus Integration Team
**Status**: ✅ Ready for Production Integration
