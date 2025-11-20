#!/bin/bash
# Start mlx-openai-server in Smart Campus mode
# This script configures the server for local classroom AI integration

set -e

echo "🎓 Starting mlx-openai-server for Smart Campus integration..."
echo ""

# Default configuration
MODEL_PATH="${MODEL_PATH:-mlx-community/qwen2.5-7b-instruct-4bit}"
MODEL_TYPE="${MODEL_TYPE:-lm}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-8192}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8000}"
ENABLE_AUTO_TOOL_CHOICE="${ENABLE_AUTO_TOOL_CHOICE:-true}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
MLX_WARMUP="${MLX_WARMUP:-true}"

echo "📋 Configuration:"
echo "   Model: $MODEL_PATH"
echo "   Type: $MODEL_TYPE"
echo "   Context: $CONTEXT_LENGTH tokens"
echo "   Host: $SERVER_HOST:$SERVER_PORT"
echo "   Tool Calling: $ENABLE_AUTO_TOOL_CHOICE (parser: $TOOL_CALL_PARSER)"
echo "   Warmup: $MLX_WARMUP"
echo ""

# Check if model directory exists (cached)
if [ -d "$HOME/.cache/huggingface/hub/models--mlx-community--qwen2.5-7b-instruct-4bit" ]; then
  echo "✅ Model found in cache"
else
  echo "📥 Model not cached - will download on first run (~4GB)"
  echo "   This may take several minutes..."
fi

echo ""
echo "🚀 Starting server..."
echo ""

# Start server with Smart Campus optimized settings
python -m app.main \
  --model-path "$MODEL_PATH" \
  --model-type "$MODEL_TYPE" \
  --context-length "$CONTEXT_LENGTH" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  --port "$SERVER_PORT" \
  --host "$SERVER_HOST" \
  --log-level "$LOG_LEVEL" \
  --mlx-warmup "$MLX_WARMUP" \
  --max-concurrency 1 \
  --queue-timeout 300

echo ""
echo "✅ Server stopped"
