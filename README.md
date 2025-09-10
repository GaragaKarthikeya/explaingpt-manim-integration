# ExplainGPT-Manim Integration

![Manim Integration](https://img.shields.io/badge/Manim-Integration-blue)
![Python](https://img.shields.io/badge/Python-99.3%25-green)
![Docker](https://img.shields.io/badge/Docker-0.7%25-blue)
![Version](https://img.shields.io/badge/version-1.0.0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)

## 📝 Description

ExplainGPT-Manim Integration is an AI-powered service that automatically generates mathematical and scientific animations using Google Gemini and the [Manim](https://www.manim.community/) animation library. This project transforms text-based explanations into dynamic visual content through intelligent code generation and automated video rendering.

Key innovations:
- **AI-Driven Code Generation**: Uses Google Gemini to create Manim animation code from natural language prompts
- **Real-Time Animation**: Generates custom animations on-demand via REST API
- **Educational Focus**: Optimized for mathematical and scientific concept visualization
- **Production Ready**: Built with FastAPI, includes job queuing, resource management, and error recovery

The result is an educational experience where abstract concepts become concrete through visual representation, making complex topics more accessible and engaging.

## ✨ Key Features

- **AI-Powered Generation**: Uses Google Gemini to automatically generate Manim code from text prompts
- **RESTful API**: Clean, documented API endpoints for easy integration with any frontend
- **Asynchronous Processing**: Job queue system handles multiple animation requests efficiently  
- **Dynamic Resource Management**: Automatic worker scaling based on system resources and load
- **Error Recovery**: Built-in error detection, code repair, and retry mechanisms
- **Production Ready**: FastAPI-based with rate limiting, CORS support, and monitoring endpoints
- **RAG Integration**: Retrieval Augmented Generation for improved code quality using example database
- **Ngrok Support**: Automatic tunnel setup for external access during development
- **Docker Deployment**: Containerized for consistent, easy deployment across environments

## 🐳 Quick Start

### Prerequisites
- Docker (recommended) OR Python 3.8+
- Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))

### Docker Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/GaragaKarthikeya/explaingpt-manim-integration.git
cd explaingpt-manim-integration

# Create environment file
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Build and run
docker-compose up --build
```

### Manual Installation

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt install python3 python3-pip ffmpeg texlive-latex-recommended

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies  
pip install -r requirements.txt

# Run the service
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Verify Installation

```bash
# Check health
curl http://localhost:8000/healthcheck

# Test animation generation
python test_client.py --prompt "Show a simple circle" --complexity 1
```

**Full setup instructions**: [Installation Guide](./docs/installation.md)


## 🚀 Usage

### Using the Test Client

```bash
# Interactive mode
python test_client.py

# Command line mode  
python test_client.py --prompt "Show how derivatives work" --complexity 2
```

### API Usage

```bash
# Generate animation
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Visualize the Pythagorean theorem",
       "complexity": 2
     }'

# Check status  
curl http://localhost:8000/status/{job_id}

# Download video
curl -o animation.mp4 http://localhost:8000/video/{job_id}
```

### JavaScript Integration

```javascript
// Generate animation
const response = await fetch('/generate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    prompt: 'Show how sine and cosine relate',
    complexity: 2
  })
});

const { job_id } = await response.json();

// Poll for completion
const checkStatus = async () => {
  const status = await fetch(`/status/${job_id}`).then(r => r.json());
  if (status.success) {
    // Display video at status.video_url
    document.getElementById('video').src = status.video_url;
  } else if (!status.error) {
    setTimeout(checkStatus, 2000); // Check again in 2s
  }
};
checkStatus();
```

**Complete usage examples**: [API Examples](./docs/api-examples.md)

## 🔄 How It Works

1. **Request Processing**: Client submits animation prompt via REST API
2. **Job Queue**: Request is queued and assigned a unique job ID  
3. **AI Code Generation**: Google Gemini generates Manim code based on the prompt
4. **Code Enhancement**: RAG system retrieves relevant examples to improve code quality
5. **Error Recovery**: Built-in systems detect and fix common code issues
6. **Animation Rendering**: Manim renders the code into MP4 video
7. **Resource Management**: Dynamic worker scaling based on system load
8. **Delivery**: Video URL returned to client for embedding or download

```
Client Request → Queue → AI Generation → Code Enhancement → Rendering → Video URL
     ↓              ↓         ↓              ↓              ↓           ↓
Rate Limiting → Job Tracking → RAG Context → Error Recovery → Resource → Caching
                                                              Scaling
```

This architecture ensures reliable, scalable animation generation with robust error handling and optimal resource utilization.

## ⚙️ Configuration

### Essential Settings

```bash
# Required - Get from https://makersuite.google.com/app/apikey
GEMINI_API_KEY="your_gemini_api_key"

# Basic Configuration  
PORT=8000
MAX_PARALLEL_RENDERINGS=2
MIN_MEMORY_PER_WORKER_MB=2048

# Optional - Ngrok for external access
ENABLE_NGROK=true
NGROK_AUTHTOKEN="your_ngrok_token"
```

### Performance Tuning

```bash
# Resource Management
MAX_MEMORY_PERCENT=80
ENABLE_DYNAMIC_SCALING=true
HIGH_LOAD_THRESHOLD=0.8
LOW_LOAD_THRESHOLD=0.3

# Rate Limiting
RATE_LIMIT_PER_MINUTE=5
MAX_QUEUE_SIZE=10

# RAG (Retrieval Augmented Generation)
RAG_ENABLED=true
RAG_MIN_SCORE=0.5
RAG_MAX_EXAMPLES=3
```

**Complete configuration reference**: [Configuration Guide](./docs/configuration-reference.md)


## 📊 Supported Animation Types

The service can generate animations for various mathematical and scientific concepts:

### Mathematics
- **Calculus**: Derivatives, integrals, limits, series expansions
- **Algebra**: Function transformations, equation solving, polynomial graphs  
- **Geometry**: Shape properties, transformations, proofs
- **Linear Algebra**: Vector operations, matrix transformations, eigenvalues

### Physics & Science  
- **Mechanics**: Motion, forces, energy, momentum
- **Waves**: Oscillations, interference, wave propagation
- **Electromagnetism**: Fields, circuits, electromagnetic waves

### Computer Science
- **Algorithms**: Sorting, searching, graph traversal
- **Data Structures**: Trees, graphs, hash tables
- **Machine Learning**: Neural networks, optimization

### Statistics & Data
- **Probability**: Distributions, random variables, Bayes' theorem
- **Data Visualization**: Charts, regression, correlation
- **Statistical Inference**: Hypothesis testing, confidence intervals

**Animation examples**: [API Examples](./docs/api-examples.md)

## 📚 Documentation

Comprehensive documentation is available in the [docs](./docs) directory:

### Getting Started
- [Installation Guide](./docs/installation.md) - Complete setup instructions
- [Quick Start Guide](./docs/quick-start.md) - Get running in 5 minutes  
- [Configuration Reference](./docs/configuration-reference.md) - All configuration options

### API and Integration  
- [API Reference](./docs/api-reference.md) - Complete REST API documentation
- [Integration Guide](./docs/integration-guide.md) - Frontend integration examples
- [API Examples](./docs/api-examples.md) - Practical usage examples

### Advanced Topics
- [Architecture Overview](./docs/architecture.md) - System design and components
- [Resource Management](./docs/resource-management.md) - Scaling and optimization
- [RAG Integration](./docs/rag-integration.md) - AI enhancement features
- [Error Recovery](./docs/error-recovery.md) - Fault tolerance mechanisms

### Operations
- [Production Deployment](./docs/production-deployment.md) - Production setup guide
- [Monitoring](./docs/monitoring.md) - Observability and alerting
- [Troubleshooting](./docs/troubleshooting.md) - Common issues and solutions
- [FAQ](./docs/faq.md) - Frequently asked questions

### Development
- [Development Setup](./docs/development-setup.md) - Local development guide  
- [Contributing](./docs/contributing.md) - How to contribute
- [Testing](./docs/testing.md) - Testing strategies

## 🔍 System Monitoring

The service provides built-in monitoring endpoints for production use:

### Health and Status
```bash
# Service health
curl http://localhost:8000/healthcheck

# System resources  
curl http://localhost:8000/system/resources

# Job performance metrics
curl http://localhost:8000/system/jobs/performance
```

### Key Metrics
- **Resource Usage**: CPU, memory, worker utilization
- **Job Statistics**: Queue depth, completion rates, processing times
- **Error Tracking**: Failed jobs, error rates, recovery attempts
- **Performance**: Animation generation times by complexity level

### Resource Management
- **Dynamic Scaling**: Automatic worker adjustment based on system load
- **Memory Protection**: Prevents system overload with configurable limits  
- **Queue Management**: Handles burst traffic with job queuing
- **Error Recovery**: Automatic retry and code repair mechanisms

**Complete monitoring guide**: [Monitoring Documentation](./docs/monitoring.md)

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Areas of Interest
- **New Animation Types**: Add support for additional mathematical concepts
- **Performance Improvements**: Optimize rendering speed and resource usage
- **AI Enhancement**: Improve prompt engineering and code generation quality
- **Integration Examples**: Add examples for new frameworks and platforms
- **Documentation**: Help improve and expand documentation

### How to Contribute
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Commit your changes (`git commit -m 'Add some amazing feature'`)  
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/explaingpt-manim-integration.git
cd explaingpt-manim-integration

# Set up development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest

# Start development server
uvicorn app.main:app --reload
```

**Detailed contribution guide**: [Contributing Documentation](./docs/contributing.md)

## 📝 Changelog

### Version 1.0.0 (Current)
- Initial release with Google Gemini integration
- FastAPI-based REST API with async job processing
- Dynamic resource management and worker scaling
- RAG-enhanced code generation for improved quality
- Built-in error recovery and retry mechanisms  
- Docker deployment with ngrok support
- Comprehensive monitoring and observability

**Full version history**: [Changelog](./docs/changelog.md)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Manim Community](https://www.manim.community/) for the incredible animation library
- [Google AI](https://ai.google.dev/) for the Gemini API enabling intelligent code generation  
- [FastAPI](https://fastapi.tiangolo.com/) for the excellent web framework
- All contributors who help improve this project
- The open source community for inspiration and support

## 📞 Contact & Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/GaragaKarthikeya/explaingpt-manim-integration/issues)
- **Documentation**: Check the [docs](./docs) directory for detailed guides
- **Author**: Karthikeya Garaga - [@GaragaKarthikeya](https://github.com/GaragaKarthikeya)

Project Link: [https://github.com/GaragaKarthikeya/explaingpt-manim-integration](https://github.com/GaragaKarthikeya/explaingpt-manim-integration)

---

**Made with ❤️ for mathematical education and visual learning**

*Last updated: 2024-12-19*