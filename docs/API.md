# AIMS API Documentation

## Overview

AIMS provides both REST API endpoints and WebSocket connections for real-time consciousness updates. All API endpoints are accessible at `http://localhost:8000/api/`.

## Authentication

Currently, AIMS uses session-based authentication with encrypted cookies. Each user is automatically assigned a unique user ID on first visit.

## REST API Endpoints

### Chat Endpoint

Send a message to the consciousness-aware AI.

**Endpoint:** `POST /api/chat`

**Request Body:**
```json
{
  "message": "Hello, how are you today?"
}
```

**Response:**
```json
{
  "response": "I'm doing well, thank you for asking! I'm currently in a positive emotional state...",
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Status Codes:**
- `200 OK` - Message processed successfully
- `400 Bad Request` - Invalid message format
- `500 Internal Server Error` - Processing error

### Session Information

Get current session information including consciousness state.

**Endpoint:** `GET /api/session`

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": "user_123",
  "consciousness": {
    "coherence": 0.85,
    "attention_focus": "conversation",
    "working_memory_items": 5,
    "interaction_count": 23
  },
  "emotional_state": {
    "current_emotion": "content",
    "confidence": 0.78,
    "pad_values": {
      "pleasure": 0.7,
      "arousal": 0.4,
      "dominance": 0.6
    },
    "intensity": 0.65
  },
  "personality": {
    "traits": {
      "openness": 0.8,
      "conscientiousness": 0.7,
      "extraversion": 0.6,
      "agreeableness": 0.8,
      "neuroticism": 0.3
    },
    "behavioral_modifiers": {
      "response_length": 0.65,
      "emotional_expression": 0.72,
      "formality": 0.58
    }
  },
  "memory_stats": {
    "user_memory_count": 145,
    "average_user_importance": 0.68
  }
}
```

### Consciousness State

Get real-time consciousness state information.

**Endpoint:** `GET /api/consciousness/state`

**Response:**
```json
{
  "coherence": 0.85,
  "attention": "user_question",
  "emotion": {
    "pleasure": 0.7,
    "arousal": 0.4,
    "dominance": 0.6,
    "label": "content",
    "confidence": 0.78,
    "color": [178, 153, 153]
  },
  "personality": {
    "openness": 0.8,
    "conscientiousness": 0.7,
    "extraversion": 0.6,
    "agreeableness": 0.8,
    "neuroticism": 0.3
  },
  "recent_memories": [
    "User asked about consciousness",
    "Explained emotional state",
    "Discussed personality traits"
  ],
  "interaction_count": 23,
  "goals": [
    "maintain_coherence",
    "be_helpful",
    "learn_continuously"
  ]
}
```

### Memory Statistics

Get memory system statistics.

**Endpoint:** `GET /api/memory/stats`

**Response:**
```json
{
  "total_memories": 1523,
  "total_sessions": 45,
  "oldest_memory": "2024-01-01T00:00:00Z",
  "average_importance": 0.65,
  "memory_distribution": {
    "high_importance": 234,
    "medium_importance": 789,
    "low_importance": 500
  }
}
```

### State Management

#### Save Current State

Create a checkpoint of the current system state.

**Endpoint:** `POST /api/state/save`

**Response:**
```json
{
  "success": true,
  "state_id": "20240115_103000"
}
```

#### Load Saved State

Restore system from a saved checkpoint.

**Endpoint:** `POST /api/state/load`

**Request Body:**
```json
{
  "state_id": "20240115_103000"
}
```

**Response:**
```json
{
  "success": true,
  "state_id": "20240115_103000"
}
```

#### List Available States

Get list of all saved states.

**Endpoint:** `GET /api/state/list`

**Response:**
```json
[
  {
    "state_id": "20240115_103000",
    "location": "local",
    "size": 125634,
    "modified": "2024-01-15T10:30:00Z",
    "compressed": true
  },
  {
    "state_id": "20240114_153000",
    "location": "local",
    "size": 118234,
    "modified": "2024-01-14T15:30:00Z",
    "compressed": true
  }
]
```

### File Upload

Upload files for processing.

**Endpoint:** `POST /api/upload`

**Request:** Multipart form data with file field

**Response:**
```json
{
  "success": true,
  "filename": "document.pdf",
  "size": 2048576
}
```

## WebSocket API

Connect to the WebSocket server for real-time consciousness updates.

### Connection

**URL:** `ws://localhost:8765`

For reconnection support, include client ID:
`ws://localhost:8765?client_id=your_client_id`

### Connection Established

Upon successful connection, you'll receive:
```json
{
  "type": "connection_established",
  "client_id": "client_1705315800000_127.0.0.1"
}
```

### Message Types

#### Send Messages

##### Get Current State
```json
{
  "type": "get_state"
}
```

##### Get Memory Statistics
```json
{
  "type": "get_memory_stats"
}
```

##### Get Emotion Trajectory
```json
{
  "type": "get_emotion_trajectory"
}
```

##### Ping (for connection health)
```json
{
  "type": "ping"
}
```

#### Receive Messages

##### Consciousness Update (sent every 500ms)
```json
{
  "type": "consciousness_update",
  "timestamp": "2024-01-15T10:30:00Z",
  "data": {
    "coherence": 0.85,
    "attention_focus": "conversation",
    "emotional_state": {
      "pleasure": 0.7,
      "arousal": 0.4,
      "dominance": 0.6
    },
    "emotion_label": "content",
    "working_memory_size": 5
  }
}
```

##### Memory Statistics
```json
{
  "type": "memory_stats",
  "data": {
    "total_memories": 1523,
    "total_sessions": 45,
    "oldest_memory": "2024-01-01T00:00:00Z",
    "average_importance": 0.65
  }
}
```

##### Emotion Trajectory
```json
{
  "type": "emotion_trajectory",
  "data": [
    {"pleasure": 0.5, "arousal": 0.5, "dominance": 0.5},
    {"pleasure": 0.6, "arousal": 0.5, "dominance": 0.5},
    {"pleasure": 0.7, "arousal": 0.4, "dominance": 0.6}
  ]
}
```

##### Pong (response to ping)
```json
{
  "type": "pong",
  "timestamp": 1705315800.123
}
```

### Reconnection Support

The WebSocket server supports automatic reconnection with message queuing:

1. Save the `client_id` from the initial connection
2. On disconnect, reconnect using: `ws://localhost:8765?client_id=your_saved_client_id`
3. Any messages sent while disconnected will be delivered upon reconnection

### Rate Limiting

- Maximum 60 messages per minute per client
- Rate limit errors return:
```json
{
  "type": "error",
  "message": "Rate limit exceeded"
}
```

## Error Handling

All API endpoints follow consistent error response format:

```json
{
  "error": "Error message",
  "details": "Additional error details",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

Common HTTP status codes:
- `400` - Bad Request (invalid input)
- `401` - Unauthorized (authentication required)
- `404` - Not Found (resource doesn't exist)
- `429` - Too Many Requests (rate limited)
- `500` - Internal Server Error

## Best Practices

1. **Session Management**: Store session IDs for continuous conversations
2. **WebSocket Reconnection**: Always save and reuse client IDs
3. **Rate Limiting**: Implement exponential backoff on rate limit errors
4. **Error Handling**: Always check response status and handle errors gracefully
5. **State Persistence**: Regularly save state for important conversations

## Example Code

### Python Client Example

```python
import asyncio
import aiohttp
import json

async def chat_with_aims():
    async with aiohttp.ClientSession() as session:
        # Send a message
        async with session.post(
            'http://localhost:8000/api/chat',
            json={'message': 'Hello AIMS!'}
        ) as response:
            result = await response.json()
            print(f"AIMS: {result['response']}")
            
        # Get consciousness state
        async with session.get(
            'http://localhost:8000/api/consciousness/state'
        ) as response:
            state = await response.json()
            print(f"Coherence: {state['coherence']}")
            print(f"Emotion: {state['emotion']['label']}")

# Run the example
asyncio.run(chat_with_aims())
```

### JavaScript WebSocket Example

```javascript
class AIMSWebSocketClient {
    constructor() {
        this.ws = null;
        this.clientId = localStorage.getItem('aims_client_id');
    }
    
    connect() {
        const url = this.clientId 
            ? `ws://localhost:8765?client_id=${this.clientId}`
            : 'ws://localhost:8765';
            
        this.ws = new WebSocket(url);
        
        this.ws.onopen = () => {
            console.log('Connected to AIMS');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'connection_established') {
                this.clientId = data.client_id;
                localStorage.setItem('aims_client_id', this.clientId);
            } else if (data.type === 'consciousness_update') {
                console.log('Consciousness state:', data.data);
            }
        };
        
        this.ws.onclose = () => {
            console.log('Disconnected, reconnecting...');
            setTimeout(() => this.connect(), 3000);
        };
    }
    
    getState() {
        this.ws.send(JSON.stringify({ type: 'get_state' }));
    }
}

// Usage
const client = new AIMSWebSocketClient();
client.connect();
```

## Changelog

### Version 1.0.0 (Current)
- Initial API release
- REST endpoints for chat, state, and memory
- WebSocket support with reconnection
- Rate limiting and error handling