# Manim API

A containerized API service that enables generation of mathematical animations via HTTP requests using Manim.

## Overview

This API wraps [Manim](https://github.com/ManimCommunity/manim), the mathematical animation engine, into a web service. It allows you to create mathematical animations programmatically without needing to write Python code or install Manim locally.

## Features

- FastAPI-based REST interface for Manim
- Supports creation of various mathematical animations and visualizations
- Docker containerization for easy deployment
- Asynchronous job processing with status tracking
- CORS support for browser-based applications
- Optional ngrok tunnel for public access

## Getting Started

### Prerequisites

- Docker

### Installation and Running

#### Option 1: Using Docker Hub Image (Recommended)

```bash
docker pull garagakarthikeya/manim-api_backend
docker run -p 8000:8000 garagakarthikeya/manim-api_backend
```

#### Option 2: Using the Source Code

```bash
git clone https://github.com/GaragaKarthikeya/explaingpt-manim-integration.git
cd explaingpt-manim-integration
docker compose up
```

The API will be available at `http://localhost:8000`

## API Documentation

### Endpoints

#### 1. Generate Animation

```
POST /generate
```

Creates a new animation job based on the provided prompt.

**Request Body:**
```json
{
  "prompt": "Create a circle that appears with a pink fill",
  "quality": "medium_quality"
}
```

**Response:**
```json
{
  "job_id": "12345678-1234-5678-1234-567812345678",
  "message": "Animation job created and queued",
  "status": "PENDING"
}
```

#### 2. Check Job Status

```
GET /status/{job_id}
```

Get the status of an animation job.

**Response:**
```json
{
  "job_id": "12345678-1234-5678-1234-567812345678",
  "success": true,
  "error": null,
  "video_url": "http://localhost:8000/videos/12345678-1234-5678-1234-567812345678.mp4"
}
```

#### 3. Download Video

```
GET /video/{job_id}
```

Download the rendered video file for a completed job.

#### 4. Health Check

```
GET /healthcheck
```

Check if the API is running and get the ngrok URL if available.

**Response:**
```json
{
  "status": "ok",
  "ngrok_url": "https://absolute-seriously-shrew.ngrok-free.app"
}
```

### Usage Examples

#### Generate an Animation

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Create a circle that grows from the center of the screen",
    "quality": "medium_quality"
  }'
```

#### Check Job Status

```bash
curl -X GET http://localhost:8000/status/12345678-1234-5678-1234-567812345678
```

#### Download Generated Video

```bash
curl -X GET http://localhost:8000/video/12345678-1234-5678-1234-567812345678 -o animation.mp4
```

## Advanced Configuration

The API supports additional configuration through environment variables:

- `NGROK_AUTHTOKEN`: To enable public access via ngrok
- `PORT`: To change the default port (8000)
- `APP_NAME`: To change the application name

## Technical Details

The service is built with:
- FastAPI - For the API framework
- Manim Community Edition - For animation rendering
- Pyngrok - For optional public access
- Docker - For containerization

## Docker Hub Repository

The latest Docker image is available at:
[https://hub.docker.com/r/garagakarthikeya/manim-api_backend](https://hub.docker.com/r/garagakarthikeya/manim-api_backend)

## GitHub Repository

[https://github.com/GaragaKarthikeya/explaingpt-manim-integration](https://github.com/GaragaKarthikeya/explaingpt-manim-integration)

## License

MIT License