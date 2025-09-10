# System Architecture

This document describes the technical architecture of the ExplainGPT-Manim Integration system, including its components, data flow, and design patterns.

## Overview

The ExplainGPT-Manim Integration is a microservice-based system that generates mathematical animations using AI-powered code generation and the Manim animation library.

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Client/UI     │───▶│   FastAPI    │───▶│   Job Queue     │
│   (ExplainGPT)  │    │   Server     │    │   System        │
└─────────────────┘    └──────────────┘    └─────────────────┘
                              │                      │
                              ▼                      ▼
                       ┌─────────────┐    ┌─────────────────┐
                       │   Static    │    │   Renderer      │
                       │   Files     │    │   Workers       │
                       └─────────────┘    └─────────────────┘
                              │                      │
                              │                      ▼
                              │            ┌─────────────────┐
                              │            │   AI Service    │
                              │            │   (Gemini)      │
                              │            └─────────────────┘
                              │                      │
                              │                      ▼
                              │            ┌─────────────────┐
                              │            │   Manim         │
                              │            │   Engine        │
                              └────────────┴─────────────────┘
```

## Core Components

### 1. FastAPI Server (`app/main.py`)

**Purpose**: HTTP API server and request handling

**Key Responsibilities**:
- HTTP endpoint management
- Request validation and rate limiting
- CORS handling for web clients
- Static file serving
- Health checks and system monitoring

**Key Features**:
- RESTful API with automatic OpenAPI documentation
- Middleware for rate limiting and CORS
- Background task management
- Ngrok integration for external access

```python
# Key endpoints structure
POST /generate        # Create animation job
GET  /status/{job_id} # Check job status  
GET  /video/{job_id}  # Download video
GET  /healthcheck     # System health
GET  /system/*        # Monitoring endpoints
```

### 2. Job Queue System (`app/services/job_queue.py`)

**Purpose**: Asynchronous job processing and queue management

**Architecture**:
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Client    │───▶│    Queue    │───▶│   Workers   │
│   Request   │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       │                   ▼                   ▼
       │            ┌─────────────┐    ┌─────────────┐
       │            │   Status    │    │   Results   │
       │            │   Tracking  │    │   Storage   │
       └────────────┴─────────────┴────┴─────────────┘
```

**Key Features**:
- Thread-safe job queue with status tracking
- Dynamic worker scaling based on system resources
- Job persistence and error recovery
- Performance metrics collection

**Job Lifecycle**:
```
QUEUED → PROCESSING → RENDERING → COMPLETED
   │         │            │          │
   └─────────┴────────────┴──────────┴─→ FAILED
```

### 3. Renderer Service (`app/services/renderer.py`)

**Purpose**: Animation rendering orchestration

**Components**:
- **Worker Pool**: Dynamic thread pool for parallel rendering
- **Resource Monitor**: System resource monitoring and scaling
- **Code Generator**: Integration with AI service for code generation
- **Manim Executor**: Safe execution of generated Manim code

**Worker Management**:
```python
# Dynamic scaling logic
if cpu_load > HIGH_THRESHOLD and workers > 1:
    scale_down_workers()
elif cpu_load < LOW_THRESHOLD and memory_available > MIN_PER_WORKER:
    scale_up_workers()
```

### 4. AI Integration Service (`app/services/llm_service.py`)

**Purpose**: AI-powered code generation using Google Gemini

**Architecture**:
```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Prompt    │───▶│    RAG      │───▶│   Gemini    │
│ Processing  │    │   System    │    │    API      │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Context   │    │  Examples   │    │   Manim     │
│  Analysis   │    │ Retrieval   │    │    Code     │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Key Features**:
- Prompt engineering for mathematical content
- RAG (Retrieval Augmented Generation) for better code quality
- Context-aware code generation
- Error recovery and code refinement

### 5. Storage Service (`app/services/storage.py`)

**Purpose**: Video file management and URL generation

**Features**:
- Local filesystem storage
- URL generation for video access
- File cleanup and cache management
- Ngrok URL integration

### 6. Error Recovery System (`app/services/error_recovery.py`)

**Purpose**: Automatic error handling and code repair

**Process Flow**:
```
Code Generation → Execution → Error? → Analysis → Repair → Retry
      │              │         │          │         │       │
      │              │         └──────────┴─────────┘       │
      │              └─────────────────────────────────────────┘
      └──────────────────── Success ─────────────────────────→
```

**Recovery Strategies**:
- Syntax error correction
- Import statement fixes
- Timeout handling
- Resource limitation workarounds

## Data Flow

### 1. Animation Request Flow

```
1. Client Request
   ├── Validation (rate limiting, format)
   ├── Job Creation (UUID generation)
   └── Queue Insertion

2. Queue Processing
   ├── Worker Assignment
   ├── Status Update (PROCESSING)
   └── AI Service Call

3. Code Generation
   ├── Prompt Analysis
   ├── RAG Context Retrieval
   ├── Gemini API Call
   └── Code Validation

4. Animation Rendering  
   ├── Manim Scene Creation
   ├── Video Rendering
   ├── File Storage
   └── URL Generation

5. Client Response
   ├── Status Update (COMPLETED)
   ├── Video URL Return
   └── Metrics Collection
```

### 2. Resource Monitoring Flow

```
System Monitor (30s intervals)
├── CPU Load Check
├── Memory Usage Check
├── Worker Count Assessment
└── Scaling Decision
    ├── Scale Up (if resources available)
    ├── Scale Down (if overloaded)
    └── Maintain Current Level
```

## Design Patterns

### 1. Producer-Consumer Pattern
- **Producers**: HTTP request handlers creating jobs
- **Queue**: Thread-safe job queue
- **Consumers**: Worker threads processing jobs

### 2. Observer Pattern
- **Subject**: Job status changes
- **Observers**: Client polling, monitoring systems
- **Notifications**: Status updates, completion events

### 3. Strategy Pattern
- **Context**: Animation generation
- **Strategies**: Different complexity levels (1-3)
- **Selection**: Based on user requirements and system load

### 4. Factory Pattern
- **Factory**: Manim scene creation
- **Products**: Different animation types (math, physics, etc.)
- **Creation Logic**: Based on AI-generated code

## Concurrency and Threading

### Thread Safety

```python
# Key thread-safe components
- Job queue: threading.Lock() for operations
- Status tracking: concurrent.futures for async operations  
- Resource monitoring: separate thread with locks
- Worker pool: ThreadPoolExecutor for job processing
```

### Resource Management

```python
# Dynamic worker scaling
class ResourceMonitor:
    def __init__(self):
        self.monitor_thread = threading.Thread(
            target=self._monitor_resources, 
            daemon=True
        )
    
    def _monitor_resources(self):
        while True:
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            
            # Scale workers based on resources
            self._adjust_worker_count(cpu_percent, memory_info)
            time.sleep(MONITORING_INTERVAL_SEC)
```

## Security Architecture

### API Security
- Rate limiting per IP address
- Input validation and sanitization
- CORS configuration for web clients
- No authentication required (internal service)

### Code Execution Security
- Sandboxed Manim execution
- Timeout protection (30 seconds default)
- Resource limitation (memory, CPU)
- Code validation before execution

### Data Security
- No persistent user data storage
- Temporary file cleanup
- Environment variable configuration
- API key security (Gemini)

## Scalability Considerations

### Vertical Scaling
- Dynamic worker count adjustment
- Memory-based scaling decisions
- CPU load-based throttling
- Resource monitoring and optimization

### Horizontal Scaling (Future)
```
Load Balancer
├── Instance 1 (primary)
├── Instance 2 (worker)
└── Instance N (worker)
     │
     ▼
Shared Storage (NFS/S3)
     │
     ▼  
Shared Queue (Redis/RabbitMQ)
```

### Performance Optimization
- **Caching**: Generated code patterns
- **Connection Pooling**: AI API connections  
- **Resource Reuse**: Manim environment
- **Batch Processing**: Multiple simple animations

## Monitoring and Observability

### Metrics Collection
```python
# Performance metrics tracked
- Job processing times by complexity
- Queue depth and wait times
- System resource utilization
- Error rates and types
- AI API response times
```

### Health Checks
- **Basic**: HTTP endpoint availability
- **Deep**: Database connections, AI API
- **Resource**: Memory, CPU, disk space
- **Queue**: Job processing capability

### Logging Strategy
```
Level DEBUG: Detailed execution flow
Level INFO:  Job lifecycle events
Level WARN:  Resource constraints, retries
Level ERROR: Failures, exceptions
```

## Integration Points

### External Dependencies
- **Google Gemini API**: AI code generation
- **Ngrok**: External access tunneling
- **FFmpeg**: Video processing
- **LaTeX**: Mathematical rendering

### Client Integration
- **REST API**: JSON request/response
- **WebSocket** (future): Real-time status updates
- **Webhook** (future): Job completion notifications

## Configuration Management

### Environment-Based Configuration
```python
# Pydantic Settings with validation
class Settings(BaseSettings):
    # Automatic environment variable loading
    # Type validation and conversion
    # Default value management
    # .env file support
```

### Runtime Configuration
- Dynamic worker scaling parameters
- Resource usage thresholds
- Rate limiting rules
- Feature flags (ngrok, RAG, etc.)

## Error Handling Strategy

### Error Categories
1. **Client Errors** (400-499): Invalid requests, rate limits
2. **Server Errors** (500-599): Internal failures, resource issues
3. **AI API Errors**: Quota, authentication, model issues
4. **Rendering Errors**: Manim execution failures

### Recovery Mechanisms
- **Automatic Retry**: Transient failures (network, resources)
- **Code Repair**: Syntax errors, import issues
- **Graceful Degradation**: Simplified animations on failure
- **Circuit Breaker**: AI API failure protection

This architecture provides a robust, scalable foundation for generating high-quality mathematical animations through AI-powered code generation and the Manim animation library.