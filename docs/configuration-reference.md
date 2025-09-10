# Configuration Reference

This document provides a comprehensive reference for all configuration options available in the ExplainGPT-Manim Integration system.

## Configuration Methods

Configuration can be set through:

1. **Environment Variables** - Recommended for production
2. **`.env` File** - Convenient for development
3. **Default Values** - Built-in fallbacks

The system uses Pydantic Settings for configuration management, which automatically loads from environment variables and `.env` files.

## Basic Configuration

### Application Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `APP_NAME` | `APP_NAME` | `"Manim Animation Server"` | Application name for logging and API docs |
| `PORT` | `PORT` | `8000` | Port for the web server |
| `MANIM_OUTPUT_DIR` | `MANIM_OUTPUT_DIR` | `"./output_videos"` | Directory for generated video files |

### Example `.env` file:
```bash
# Basic Settings
APP_NAME="My Manim Server"
PORT=8080
MANIM_OUTPUT_DIR="/app/videos"
```

## Performance and Processing

### Worker Configuration

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `MAX_PARALLEL_RENDERINGS` | `MAX_PARALLEL_RENDERINGS` | `CPU_COUNT - 1` | Maximum concurrent rendering jobs |
| `MIN_MEMORY_PER_WORKER_MB` | `MIN_MEMORY_PER_WORKER_MB` | `2048` | Minimum memory required per worker (MB) |
| `MAX_MEMORY_PERCENT` | `MAX_MEMORY_PERCENT` | `80` | Maximum system memory to use (%) |

### Dynamic Scaling

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `ENABLE_DYNAMIC_SCALING` | `ENABLE_DYNAMIC_SCALING` | `true` | Enable automatic worker scaling |
| `HIGH_LOAD_THRESHOLD` | `HIGH_LOAD_THRESHOLD` | `0.8` | CPU load threshold to reduce workers |
| `LOW_LOAD_THRESHOLD` | `LOW_LOAD_THRESHOLD` | `0.3` | CPU load threshold to add workers |
| `MONITORING_INTERVAL_SEC` | `MONITORING_INTERVAL_SEC` | `30` | Resource monitoring interval (seconds) |

### Example Configuration:
```bash
# Performance Settings
MAX_PARALLEL_RENDERINGS=4
MIN_MEMORY_PER_WORKER_MB=1024
MAX_MEMORY_PERCENT=75
ENABLE_DYNAMIC_SCALING=true
HIGH_LOAD_THRESHOLD=0.85
LOW_LOAD_THRESHOLD=0.25
```

## Queue Management

### Queue Settings

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `MAX_QUEUE_SIZE` | `MAX_QUEUE_SIZE` | `10` | Maximum number of jobs in queue |
| `CACHE_EXPIRY_HOURS` | `CACHE_EXPIRY_HOURS` | `24` | Video cache expiration (hours) |

### Rate Limiting

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `RATE_LIMIT_PER_MINUTE` | `RATE_LIMIT_PER_MINUTE` | `5` | API requests per minute per IP |

### Example:
```bash
# Queue Configuration
MAX_QUEUE_SIZE=20
CACHE_EXPIRY_HOURS=48
RATE_LIMIT_PER_MINUTE=10
```

## AI Integration

### Gemini API Configuration

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `GEMINI_API_KEY` | `GEMINI_API_KEY` | `""` | Google Gemini API key (required) |
| `GEMINI_MODEL` | `GEMINI_MODEL` | `"models/gemini-2.0-flash"` | Gemini model to use |

### Example:
```bash
# AI Configuration
GEMINI_API_KEY="your_gemini_api_key_here"
GEMINI_MODEL="models/gemini-2.0-flash"
```

**Note**: The Gemini API key is required for the system to function. Get your API key from the [Google AI Studio](https://makersuite.google.com/app/apikey).

## Network and Tunneling

### Ngrok Configuration

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `ENABLE_NGROK` | `ENABLE_NGROK` | `true` | Enable ngrok tunneling |
| `NGROK_AUTHTOKEN` | `NGROK_AUTHTOKEN` | `""` | Ngrok authentication token |
| `NGROK_DOMAIN` | `NGROK_DOMAIN` | `""` | Static ngrok domain (optional) |
| `NGROK_MAX_RETRIES` | `NGROK_MAX_RETRIES` | `2` | Maximum ngrok connection retries |
| `NGROK_RETRY_DELAY` | `NGROK_RETRY_DELAY` | `2` | Delay between ngrok retries (seconds) |
| `NGROK_BASE_URL` | `NGROK_BASE_URL` | `""` | Current ngrok URL (auto-set) |

### Example:
```bash
# Ngrok Configuration
ENABLE_NGROK=true
NGROK_AUTHTOKEN="your_ngrok_token"
NGROK_DOMAIN="your-static-domain.ngrok.io"
```

**Ngrok Setup:**
1. Sign up at [ngrok.com](https://ngrok.com)
2. Get your authtoken from the dashboard
3. Set `NGROK_AUTHTOKEN` in your environment

## Error Recovery

### Error Handling Configuration

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `ERROR_RECOVERY_ENABLED` | `ERROR_RECOVERY_ENABLED` | `true` | Enable automatic error recovery |
| `ERROR_RECOVERY_MAX_RETRIES` | `ERROR_RECOVERY_MAX_RETRIES` | `3` | Maximum retry attempts |
| `ERROR_RECOVERY_SANDBOX_TIMEOUT` | `ERROR_RECOVERY_SANDBOX_TIMEOUT` | `30` | Sandbox timeout (seconds) |

### Example:
```bash
# Error Recovery
ERROR_RECOVERY_ENABLED=true
ERROR_RECOVERY_MAX_RETRIES=5
ERROR_RECOVERY_SANDBOX_TIMEOUT=45
```

## RAG (Retrieval Augmented Generation)

### RAG Configuration

| Setting | Environment Variable | Default | Description |
|---------|---------------------|---------|-------------|
| `RAG_ENABLED` | `RAG_ENABLED` | `true` | Enable RAG for code generation |
| `RAG_INDEX_PATH` | `RAG_INDEX_PATH` | `"./data/manim_index.faiss"` | FAISS index file path |
| `RAG_BLOCKS_PATH` | `RAG_BLOCKS_PATH` | `"./data/manim_blocks.npy"` | Code blocks file path |
| `RAG_MIN_SCORE` | `RAG_MIN_SCORE` | `0.5` | Minimum similarity score |
| `RAG_MAX_EXAMPLES` | `RAG_MAX_EXAMPLES` | `3` | Maximum examples to retrieve |

### Example:
```bash
# RAG Configuration
RAG_ENABLED=true
RAG_INDEX_PATH="/app/data/manim_index.faiss"
RAG_BLOCKS_PATH="/app/data/manim_blocks.npy"
RAG_MIN_SCORE=0.6
RAG_MAX_EXAMPLES=5
```

## Complete Configuration Example

Here's a complete `.env` file with all configuration options:

```bash
# Application Settings
APP_NAME="ExplainGPT Manim Server"
PORT=8000
MANIM_OUTPUT_DIR="./output_videos"

# Performance and Workers
MAX_PARALLEL_RENDERINGS=4
MIN_MEMORY_PER_WORKER_MB=2048
MAX_MEMORY_PERCENT=80
ENABLE_DYNAMIC_SCALING=true
HIGH_LOAD_THRESHOLD=0.8
LOW_LOAD_THRESHOLD=0.3
MONITORING_INTERVAL_SEC=30

# Queue Management
MAX_QUEUE_SIZE=20
CACHE_EXPIRY_HOURS=24
RATE_LIMIT_PER_MINUTE=10

# AI Configuration (Required)
GEMINI_API_KEY="your_gemini_api_key_here"
GEMINI_MODEL="models/gemini-2.0-flash"

# Ngrok Tunneling
ENABLE_NGROK=true
NGROK_AUTHTOKEN="your_ngrok_token"
NGROK_DOMAIN=""
NGROK_MAX_RETRIES=2
NGROK_RETRY_DELAY=2

# Error Recovery
ERROR_RECOVERY_ENABLED=true
ERROR_RECOVERY_MAX_RETRIES=3
ERROR_RECOVERY_SANDBOX_TIMEOUT=30

# RAG Configuration
RAG_ENABLED=true
RAG_INDEX_PATH="./data/manim_index.faiss"
RAG_BLOCKS_PATH="./data/manim_blocks.npy"
RAG_MIN_SCORE=0.5
RAG_MAX_EXAMPLES=3
```

## Environment-Specific Configurations

### Development Configuration

```bash
# Development optimized settings
PORT=8000
MAX_PARALLEL_RENDERINGS=2
MIN_MEMORY_PER_WORKER_MB=1024
MAX_MEMORY_PERCENT=70
ENABLE_DYNAMIC_SCALING=false
RATE_LIMIT_PER_MINUTE=20
ENABLE_NGROK=true
```

### Production Configuration

```bash
# Production optimized settings
PORT=8000
MAX_PARALLEL_RENDERINGS=8
MIN_MEMORY_PER_WORKER_MB=4096
MAX_MEMORY_PERCENT=85
ENABLE_DYNAMIC_SCALING=true
HIGH_LOAD_THRESHOLD=0.9
LOW_LOAD_THRESHOLD=0.2
RATE_LIMIT_PER_MINUTE=5
ENABLE_NGROK=false
```

### Resource-Constrained Environment

```bash
# Low-resource settings (e.g., small VPS)
MAX_PARALLEL_RENDERINGS=1
MIN_MEMORY_PER_WORKER_MB=1024
MAX_MEMORY_PERCENT=60
ENABLE_DYNAMIC_SCALING=false
MAX_QUEUE_SIZE=5
RATE_LIMIT_PER_MINUTE=2
```

## Configuration Validation

The system validates configuration on startup:

- **Memory Validation**: Ensures sufficient memory for configured workers
- **Model Validation**: Verifies Gemini model availability
- **Path Validation**: Checks that output directories are writable
- **Network Validation**: Tests ngrok connectivity if enabled

### Common Configuration Errors

1. **Insufficient Memory**: Reduce `MAX_PARALLEL_RENDERINGS` or increase system RAM
2. **Invalid Gemini Key**: Verify API key at [Google AI Studio](https://makersuite.google.com/app/apikey)
3. **Ngrok Connection Failed**: Check authtoken and network connectivity
4. **Permission Denied**: Ensure output directory is writable

## Advanced Configuration

### Custom Worker Scaling Logic

For advanced users, worker scaling can be customized by modifying the resource monitoring thresholds:

```bash
# Fine-tuned scaling
ENABLE_DYNAMIC_SCALING=true
HIGH_LOAD_THRESHOLD=0.85  # Scale down when CPU > 85%
LOW_LOAD_THRESHOLD=0.15   # Scale up when CPU < 15%
MONITORING_INTERVAL_SEC=15  # Check every 15 seconds
```

### RAG Fine-Tuning

Optimize RAG performance for your use case:

```bash
# Conservative RAG (higher quality, fewer examples)
RAG_MIN_SCORE=0.7
RAG_MAX_EXAMPLES=2

# Aggressive RAG (more examples, lower threshold)
RAG_MIN_SCORE=0.3
RAG_MAX_EXAMPLES=5
```

## Docker Environment Variables

When using Docker, pass environment variables using the `--env-file` flag:

```bash
docker run --env-file .env -p 8000:8000 explaingpt-manim
```

Or set individual variables:

```bash
docker run \
  -e GEMINI_API_KEY="your_key" \
  -e MAX_PARALLEL_RENDERINGS=4 \
  -p 8000:8000 \
  explaingpt-manim
```

## Configuration Best Practices

1. **Security**: Never commit API keys to version control
2. **Performance**: Start with default settings and tune based on monitoring
3. **Resources**: Monitor system resources and adjust worker limits accordingly  
4. **Rate Limiting**: Set appropriate limits based on expected usage
5. **Error Recovery**: Enable in production for better reliability
6. **RAG**: Disable if not using AI-assisted code generation to save resources