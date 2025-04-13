# Manim API

A containerized API service that provides access to Manim's mathematical animation capabilities through simple HTTP requests.

## Overview

Manim API is a web service wrapper around [Manim](https://github.com/ManimCommunity/manim), the mathematical animation engine. It allows users to generate complex mathematical animations and visualizations without having to set up Manim locally or write Python code directly.

## Features

- RESTful API for creating Manim animations
- Docker containerization for easy deployment
- Support for various Manim scene types and configurations
- Video output in multiple formats
- Asynchronous processing of animation requests

## Project Structure

```
manim-api/
├── app/                  # Main application code
│   ├── api/              # API routes and handlers
│   ├── core/             # Core functionality
│   ├── models/           # Data models
│   └── utils/            # Utility functions
├── tests/                # Test suite
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-container orchestration
├── requirements.txt      # Python dependencies
└── README.md             # This documentation
```

## Getting Started

### Prerequisites

- Docker
- Docker Compose (optional)

### Installation and Running

#### Option 1: Using Docker Hub Image

You can directly pull and run the pre-built Docker image:

```bash
docker pull garagakarthikeya/manim-api_backend
docker run -p 8000:8000 garagakarthikeya/manim-api_backend
```

The API will be available at `http://localhost:8000`

#### Option 2: Using the Source Code

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/manim-api.git
   cd manim-api
   ```

2. Build and start the service:
   ```bash
   docker-compose up
   ```

3. The API will be available at `http://localhost:8000`

## Usage

### Example API Request

```bash
curl -X POST http://localhost:8000/animations/create \
  -H "Content-Type: application/json" \
  -d '{
    "scene_type": "TextExample",
    "properties": {
      "text": "Hello, Manim!",
      "color": "#FFFFFF",
      "position": [0, 0, 0]
    },
    "output_format": "mp4"
  }'
```

## API Documentation

### Endpoints

- `GET /health` - Check if the API is running
- `POST /animations/create` - Create a new animation
- `GET /animations/{id}` - Get the status of an animation
- `GET /animations/{id}/download` - Download the generated animation

## Development

### Local Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Manim Community](https://www.manim.community/) for the amazing animation engine
- All contributors to this project