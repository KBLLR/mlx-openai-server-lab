/**
 * Smart Campus API Integration Reference Implementation
 *
 * This file provides a reference implementation for integrating mlx-openai-server
 * with Smart Campus classroom chat functionality.
 *
 * TARGET FILE: code-smart-campus/server/api/classrooms.js (or similar)
 *
 * USAGE:
 * 1. Copy relevant functions to your Smart Campus API route handler
 * 2. Adapt to your existing classroom data models
 * 3. Ensure MLX_SERVER_URL environment variable is set
 * 4. Test with your frontend
 */

// ============================================================================
// CONFIGURATION
// ============================================================================

const MLX_SERVER_URL = process.env.MLX_SERVER_URL || 'http://localhost:8000';
const ENABLE_LOCAL_AI = process.env.ENABLE_LOCAL_AI !== 'false'; // default true
const ENABLE_CLOUD_FALLBACK = process.env.ENABLE_CLOUD_FALLBACK === 'true'; // default false

// ============================================================================
// CLASSROOM TOOLS DEFINITION
// ============================================================================

/**
 * Get tool definitions scoped to a specific classroom.
 * These tools allow the AI to interact with classroom data safely.
 */
function getClassroomTools(classroomId) {
  return [
    {
      type: 'function',
      function: {
        name: 'append_classroom_event',
        description: 'Add an event to the classroom timeline. Use this to record significant classroom activities, announcements, or changes.',
        parameters: {
          type: 'object',
          properties: {
            event_type: {
              type: 'string',
              enum: ['activity', 'announcement', 'sensor_alert', 'schedule_change', 'note'],
              description: 'Type of event'
            },
            description: {
              type: 'string',
              description: 'Detailed description of the event'
            },
            metadata: {
              type: 'object',
              description: 'Additional event metadata (optional)',
              properties: {
                priority: { type: 'string', enum: ['low', 'medium', 'high'] },
                tags: { type: 'array', items: { type: 'string' } }
              }
            }
          },
          required: ['event_type', 'description']
        }
      }
    },
    {
      type: 'function',
      function: {
        name: 'get_sensor_data',
        description: 'Retrieve current sensor readings for the classroom.',
        parameters: {
          type: 'object',
          properties: {
            sensor_type: {
              type: 'string',
              enum: ['temperature', 'humidity', 'occupancy', 'co2', 'light', 'noise'],
              description: 'Type of sensor to query'
            }
          },
          required: ['sensor_type']
        }
      }
    },
    {
      type: 'function',
      function: {
        name: 'get_classroom_schedule',
        description: 'Get the classroom schedule for a specific date or date range.',
        parameters: {
          type: 'object',
          properties: {
            date: {
              type: 'string',
              description: 'Date in YYYY-MM-DD format (defaults to today)'
            },
            days_ahead: {
              type: 'number',
              description: 'Number of days ahead to include (default: 0 = today only)'
            }
          }
        }
      }
    },
    {
      type: 'function',
      function: {
        name: 'search_classroom_events',
        description: 'Search historical classroom events by type, date range, or keyword.',
        parameters: {
          type: 'object',
          properties: {
            event_type: {
              type: 'string',
              description: 'Filter by event type (optional)'
            },
            keyword: {
              type: 'string',
              description: 'Search keyword in event descriptions (optional)'
            },
            limit: {
              type: 'number',
              description: 'Maximum number of events to return (default: 10)'
            }
          }
        }
      }
    }
  ];
}

// ============================================================================
// TOOL EXECUTION
// ============================================================================

/**
 * Execute a classroom tool call with proper scoping and validation.
 *
 * IMPORTANT: This enforces classroom-level isolation - tools can only
 * access data for the specified classroomId.
 */
async function executeClassroomTool(classroomId, toolName, toolArgs, classroom) {
  console.log(`[Tool Execution] classroom=${classroomId} tool=${toolName}`);

  // Classroom scoping: ensure tool only accesses this classroom's data
  if (!classroom) {
    throw new Error(`Classroom ${classroomId} not found`);
  }

  switch (toolName) {
    case 'append_classroom_event':
      return await handleAppendEvent(classroom, toolArgs);

    case 'get_sensor_data':
      return await handleGetSensorData(classroom, toolArgs);

    case 'get_classroom_schedule':
      return await handleGetSchedule(classroom, toolArgs);

    case 'search_classroom_events':
      return await handleSearchEvents(classroom, toolArgs);

    default:
      throw new Error(`Unknown tool: ${toolName}`);
  }
}

/**
 * Tool Handler: Append classroom event
 */
async function handleAppendEvent(classroom, args) {
  const event = {
    id: generateEventId(),
    type: args.event_type,
    description: args.description,
    timestamp: new Date().toISOString(),
    source: 'ai_assistant',
    metadata: args.metadata || {},
    classroomId: classroom.id
  };

  // TODO: Replace with your actual classroom event storage
  // Example: await classroom.addEvent(event);
  // Example: await db.classroomEvents.create(event);

  console.log(`[Event Created] classroom=${classroom.id} type=${event.type}`);

  return {
    success: true,
    event_id: event.id,
    message: `Event "${event.type}" added to classroom timeline`
  };
}

/**
 * Tool Handler: Get sensor data
 */
async function handleGetSensorData(classroom, args) {
  const sensorType = args.sensor_type;

  // TODO: Replace with your actual sensor data retrieval
  // Example: const sensorData = await classroom.getSensorData(sensorType);
  // Example: const sensorData = await db.sensors.findLatest({ classroomId: classroom.id, type: sensorType });

  // Mock response - replace with actual data
  const mockSensorData = {
    temperature: { value: 22.5, unit: '°C', timestamp: new Date().toISOString() },
    humidity: { value: 45, unit: '%', timestamp: new Date().toISOString() },
    occupancy: { value: 24, unit: 'people', timestamp: new Date().toISOString() },
    co2: { value: 450, unit: 'ppm', timestamp: new Date().toISOString() },
    light: { value: 350, unit: 'lux', timestamp: new Date().toISOString() },
    noise: { value: 42, unit: 'dB', timestamp: new Date().toISOString() }
  };

  const sensorReading = mockSensorData[sensorType];

  if (!sensorReading) {
    return {
      success: false,
      error: `Sensor type "${sensorType}" not available in this classroom`
    };
  }

  return {
    success: true,
    sensor_type: sensorType,
    reading: sensorReading
  };
}

/**
 * Tool Handler: Get classroom schedule
 */
async function handleGetSchedule(classroom, args) {
  const date = args.date || new Date().toISOString().split('T')[0];
  const daysAhead = args.days_ahead || 0;

  // TODO: Replace with your actual schedule retrieval
  // Example: const schedule = await classroom.getSchedule(date, daysAhead);

  // Mock response
  return {
    success: true,
    classroom: classroom.name,
    date_range: { start: date, days: daysAhead + 1 },
    schedule: [
      { time: '09:00-10:30', subject: 'Mathematics', teacher: 'Ms. Johnson' },
      { time: '10:45-12:15', subject: 'Science', teacher: 'Mr. Chen' },
      { time: '13:00-14:30', subject: 'Literature', teacher: 'Dr. Williams' }
    ]
  };
}

/**
 * Tool Handler: Search classroom events
 */
async function handleSearchEvents(classroom, args) {
  const { event_type, keyword, limit = 10 } = args;

  // TODO: Replace with your actual event search
  // Example: const events = await classroom.searchEvents({ type: event_type, keyword, limit });

  // Mock response
  return {
    success: true,
    events: [
      {
        id: 'evt-123',
        type: 'announcement',
        description: 'Class project presentations next week',
        timestamp: '2025-01-15T10:00:00Z'
      }
    ],
    count: 1,
    filters_applied: { event_type, keyword, limit }
  };
}

// ============================================================================
// MLX SERVER COMMUNICATION
// ============================================================================

/**
 * Call mlx-openai-server with proper error handling and fallback.
 */
async function callMLXServer(classroom, messages, tools, options = {}) {
  const {
    temperature = 0.7,
    max_tokens = 2048,
    stream = false
  } = options;

  const requestId = `classroom-${classroom.id}-${Date.now()}`;

  const payload = {
    model: 'mlx-community/qwen2.5-7b-instruct-4bit', // Or make configurable
    messages,
    temperature,
    max_tokens,
    stream
  };

  // Add tools if provided
  if (tools && tools.length > 0) {
    payload.tools = tools;
  }

  try {
    const response = await fetch(`${MLX_SERVER_URL}/v1/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Request-ID': requestId
      },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(60000) // 60 second timeout
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(`MLX server error: ${error.error?.message || response.statusText}`);
    }

    return response;

  } catch (error) {
    console.error('[MLX Error]', error.message);
    throw error;
  }
}

/**
 * Health check for mlx-openai-server
 */
async function checkMLXServerHealth() {
  try {
    const response = await fetch(`${MLX_SERVER_URL}/health`, {
      signal: AbortSignal.timeout(5000)
    });

    if (!response.ok) {
      return { healthy: false, status: response.status };
    }

    const health = await response.json();
    return {
      healthy: health.models_healthy === true,
      model_id: health.model_id,
      warmup_completed: health.warmup_completed
    };

  } catch (error) {
    console.error('[MLX Health Check Failed]', error.message);
    return { healthy: false, error: error.message };
  }
}

// ============================================================================
// MAIN API HANDLER
// ============================================================================

/**
 * Main handler for POST /api/classrooms/:id/chat
 *
 * This is the main integration point - adapt this to your API framework
 * (Express, Fastify, Next.js API routes, etc.)
 */
async function handleClassroomChat(req, res) {
  const { id: classroomId } = req.params;
  const { messages, stream = false } = req.body;

  try {
    // 1. Validate and load classroom
    const classroom = await getClassroomById(classroomId);
    if (!classroom) {
      return res.status(404).json({
        error: 'Classroom not found',
        classroom_id: classroomId
      });
    }

    // 2. Check if local AI is enabled
    if (!ENABLE_LOCAL_AI) {
      return res.status(503).json({
        error: 'Local AI is disabled',
        message: 'Enable ENABLE_LOCAL_AI environment variable'
      });
    }

    // 3. Build system message with classroom context
    const systemMessage = buildClassroomSystemMessage(classroom);
    const fullMessages = [systemMessage, ...messages];

    // 4. Get classroom tools
    const tools = getClassroomTools(classroomId);

    // 5. Call MLX server
    let response;
    try {
      response = await callMLXServer(classroom, fullMessages, tools, { stream });
    } catch (mlxError) {
      // Handle MLX server unavailable
      if (ENABLE_CLOUD_FALLBACK) {
        console.warn('[Fallback] Using cloud AI due to MLX error:', mlxError.message);
        // TODO: Implement cloud fallback (OpenAI, etc.)
        return res.status(503).json({
          error: 'Local AI unavailable, cloud fallback not implemented',
          fallback: true
        });
      } else {
        return res.status(503).json({
          error: 'AI assistant temporarily unavailable',
          message: 'The local AI server is not responding. Please try again later.',
          details: mlxError.message
        });
      }
    }

    // 6. Handle streaming response
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      // Pipe SSE stream to client
      response.body.pipe(res);
      return;
    }

    // 7. Handle non-streaming response
    const data = await response.json();

    // 8. Handle tool calls if present
    if (data.choices[0].message.tool_calls) {
      const processedResponse = await processToolCalls(
        classroom,
        fullMessages,
        data,
        tools
      );
      return res.json(processedResponse);
    }

    // 9. No tool calls - return response directly
    return res.json(data);

  } catch (error) {
    console.error('[Classroom Chat Error]', error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
}

/**
 * Process tool calls and continue conversation
 */
async function processToolCalls(classroom, messages, initialResponse, tools) {
  const toolCalls = initialResponse.choices[0].message.tool_calls;

  // Add assistant message with tool calls
  messages.push(initialResponse.choices[0].message);

  // Execute each tool call
  for (const toolCall of toolCalls) {
    const toolName = toolCall.function.name;
    const toolArgs = JSON.parse(toolCall.function.arguments);

    try {
      // Execute tool with classroom scoping
      const result = await executeClassroomTool(
        classroom.id,
        toolName,
        toolArgs,
        classroom
      );

      // Add tool result to conversation
      messages.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        name: toolName,
        content: JSON.stringify(result)
      });

    } catch (toolError) {
      console.error(`[Tool Error] ${toolName}:`, toolError.message);

      // Add error result
      messages.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        name: toolName,
        content: JSON.stringify({
          success: false,
          error: toolError.message
        })
      });
    }
  }

  // Continue conversation with tool results
  const followUpResponse = await callMLXServer(classroom, messages, tools);
  return await followUpResponse.json();
}

/**
 * Build system message with classroom context
 */
function buildClassroomSystemMessage(classroom) {
  return {
    role: 'system',
    content: `You are an AI assistant for ${classroom.name}.

**Classroom Information:**
- Room: ${classroom.room_number || 'N/A'}
- Capacity: ${classroom.capacity || 'N/A'} students
- Current Enrollment: ${classroom.student_count || 'N/A'} students
- Active Sensors: ${classroom.sensors?.map(s => s.type).join(', ') || 'None'}

**Your Capabilities:**
- Answer questions about the classroom
- Check sensor readings (temperature, humidity, occupancy, etc.)
- View and manage the classroom schedule
- Add events to the classroom timeline
- Search historical classroom events

**Guidelines:**
- Always maintain privacy - do not share individual student information
- Be helpful and educational
- Use tools when appropriate to access real-time classroom data
- Scope all actions to this classroom only (ID: ${classroom.id})`
  };
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

function generateEventId() {
  return `evt-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Mock function - replace with your actual classroom retrieval
 */
async function getClassroomById(classroomId) {
  // TODO: Replace with actual database query
  // Example: return await db.classrooms.findById(classroomId);

  return {
    id: classroomId,
    name: 'Room 101 - Advanced Mathematics',
    room_number: '101',
    capacity: 30,
    student_count: 24,
    sensors: [
      { type: 'temperature', status: 'active' },
      { type: 'humidity', status: 'active' },
      { type: 'occupancy', status: 'active' }
    ],
    schedule: 'Mon-Fri 9:00-15:00'
  };
}

// ============================================================================
// EXPORTS (adapt to your framework)
// ============================================================================

module.exports = {
  // Main handler
  handleClassroomChat,

  // Helper functions
  getClassroomTools,
  executeClassroomTool,
  callMLXServer,
  checkMLXServerHealth,
  buildClassroomSystemMessage,

  // Configuration
  MLX_SERVER_URL,
  ENABLE_LOCAL_AI,
  ENABLE_CLOUD_FALLBACK
};
