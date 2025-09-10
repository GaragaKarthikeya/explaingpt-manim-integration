# API Reference

This document provides a complete reference for the ExplainGPT-Manim Integration REST API. All endpoints return JSON responses and use standard HTTP status codes.

## Base URL

The API is typically available at:
- **Development**: `http://localhost:8000`
- **Production**: Your deployed URL or ngrok tunnel URL

## Authentication

Currently, the API uses rate limiting but does not require authentication tokens. Rate limiting is applied per IP address.

**Rate Limits:**
- 5 requests per minute per IP address
- HTTP 429 status returned when rate limit exceeded

## Content Types

- **Request**: `application/json`
- **Response**: `application/json`
- **Video Files**: `video/mp4`

## Core Endpoints

### Generate Animation

Creates a new animation job and adds it to the processing queue.

```http
POST /generate
```

**Request Body:**

```json
{
  "prompt": "string",
  "animate": true,
  "complexity": 2
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `prompt` | string | Yes | The description of what to animate (e.g., "Show how derivatives work") |
| `animate` | boolean | No | Whether to create animation (default: true) |
| `complexity` | integer | No | Animation complexity level 1-3 (default: 2) |

**Complexity Levels:**
- `1`: Simple - Basic visualization with minimal detail
- `2`: Moderate - Balanced visualization with good detail 
- `3`: Complex - Comprehensive visualization with extensive detail

**Response:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Animation job created and queued",
  "status": "queued"
}
```

**Status Codes:**
- `200`: Job created successfully
- `400`: Invalid request parameters
- `429`: Rate limit exceeded
- `500`: Internal server error

### Check Job Status

Retrieves the current status and result of an animation job.

```http
GET /status/{job_id}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `job_id` | string | Yes | The unique job identifier returned from `/generate` |

**Response:**

```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "success": true,
  "video_url": "http://localhost:8000/videos/550e8400-e29b-41d4-a716-446655440000.mp4",
  "error": null
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | The job identifier |
| `success` | boolean | Whether the job completed successfully |
| `video_url` | string | URL to the generated video (only when successful) |
| `error` | string | Error message if the job failed |

**Job Statuses:**
- `queued`: Job is waiting in queue
- `processing`: Job is being processed by AI
- `rendering`: Video is being rendered
- `completed`: Job completed successfully
- `failed`: Job failed with error

**Status Codes:**
- `200`: Status retrieved successfully
- `404`: Job not found
- `500`: Internal server error

### Download Video

Downloads the generated video file directly.

```http
GET /video/{job_id}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `job_id` | string | Yes | The unique job identifier |

**Response:**
- **Content-Type**: `video/mp4`
- **Content-Disposition**: `attachment; filename="{job_id}.mp4"`

**Status Codes:**
- `200`: Video file returned
- `400`: Job not completed yet
- `404`: Job or video file not found
- `500`: Internal server error

## System Endpoints

### Health Check

Verifies that the API is running and accessible.

```http
GET /healthcheck
```

**Response:**

```json
{
  "status": "ok",
  "ngrok_url": "https://abc123.ngrok.io"
}
```

**Status Codes:**
- `200`: Service is healthy

### System Resources

Provides information about system resource usage and worker configuration.

```http
GET /system/resources
```

**Response:**

```json
{
  "worker_configuration": {
    "current_workers": 2,
    "max_workers": 3,
    "memory_per_worker_mb": 2048,
    "dynamic_scaling_enabled": true
  },
  "system_resources": {
    "cpu_percent": 15.2,
    "memory_percent": 62.8,
    "system_load": [0.5, 0.6, 0.4],
    "total_memory_mb": 8192.0,
    "available_memory_mb": 3072.0
  },
  "job_metrics": {
    "total_jobs": 150,
    "queued_jobs": 2,
    "processing_jobs": 1,
    "completed_jobs": 140,
    "failed_jobs": 7
  },
  "top_memory_processes": [
    {
      "pid": 1234,
      "name": "python",
      "memory_mb": 512.5
    }
  ]
}
```

**Status Codes:**
- `200`: System information retrieved

### Job Performance Stats

Provides performance metrics for recent animation jobs.

```http
GET /system/jobs/performance
```

**Response:**

```json
{
  "recent_jobs": [
    {
      "job_id": "550e8400-e29b-41d4-a716-446655440000",
      "complexity": 2,
      "total_time_seconds": 45.2,
      "queue_time_seconds": 2.1,
      "processing_time_seconds": 43.1
    }
  ],
  "complexity_metrics": {
    "1": {
      "count": 50,
      "avg_total_time": 25.5,
      "avg_queue_time": 1.2,
      "avg_processing_time": 24.3
    },
    "2": {
      "count": 75,
      "avg_total_time": 45.8,
      "avg_queue_time": 1.8,
      "avg_processing_time": 44.0
    },
    "3": {
      "count": 25,
      "avg_total_time": 85.2,
      "avg_queue_time": 2.5,
      "avg_processing_time": 82.7
    }
  }
}
```

**Status Codes:**
- `200`: Performance stats retrieved
- `200`: Returns `{"message": "No job statistics available yet"}` if no jobs processed

## CORS Support

The API includes CORS support for browser-based clients:

- **Allowed Origins**: `*` (all origins in development)
- **Allowed Methods**: `GET`, `POST`, `OPTIONS`
- **Allowed Headers**: `*` (all headers)
- **Exposed Headers**: `Content-Disposition`

## Error Responses

All error responses follow this format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**
- `400 Bad Request`: Invalid request parameters
- `404 Not Found`: Resource not found
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server-side error

## Example Usage

### Complete Animation Workflow

1. **Generate Animation:**
```bash
curl -X POST "http://localhost:8000/generate" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Show the concept of a derivative as the slope of a tangent line",
       "complexity": 2
     }'
```

2. **Check Status (repeat until completed):**
```bash
curl "http://localhost:8000/status/550e8400-e29b-41d4-a716-446655440000"
```

3. **Download Video:**
```bash
curl -o "animation.mp4" \
     "http://localhost:8000/video/550e8400-e29b-41d4-a716-446655440000"
```

### JavaScript Example

```javascript
// Generate animation
const response = await fetch('/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: 'Visualize the Pythagorean theorem',
    complexity: 2
  })
});

const { job_id } = await response.json();

// Poll for completion
const pollStatus = async () => {
  const statusResponse = await fetch(`/status/${job_id}`);
  const status = await statusResponse.json();
  
  if (status.success) {
    console.log('Video ready:', status.video_url);
    return status.video_url;
  } else if (status.error) {
    console.error('Job failed:', status.error);
    return null;
  } else {
    // Still processing, check again in 2 seconds
    setTimeout(pollStatus, 2000);
  }
};

pollStatus();
```

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Limit**: 5 requests per minute per IP address
- **Window**: Rolling 60-second window
- **Response**: HTTP 429 when exceeded
- **Headers**: Rate limit info in response headers (if implemented)

To handle rate limiting in your application:

```javascript
const makeRequest = async (url, options) => {
  const response = await fetch(url, options);
  
  if (response.status === 429) {
    console.log('Rate limited, waiting 60 seconds...');
    await new Promise(resolve => setTimeout(resolve, 60000));
    return makeRequest(url, options); // Retry
  }
  
  return response;
};
```