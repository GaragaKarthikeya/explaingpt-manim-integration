# Installation Guide

This guide covers all the steps needed to install and set up the ExplainGPT-Manim Integration system.

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows with WSL2
- **RAM**: 4GB (8GB recommended)
- **CPU**: 2 cores (4 cores recommended)  
- **Storage**: 2GB free space
- **Python**: 3.8+ (if installing without Docker)

### Recommended Requirements
- **OS**: Linux (Ubuntu 20.04+) or macOS
- **RAM**: 8GB+ (16GB for production)
- **CPU**: 4+ cores
- **Storage**: 10GB+ free space
- **Network**: Stable internet connection for AI API calls

## Installation Methods

### Method 1: Docker (Recommended)

Docker provides the easiest and most reliable installation method.

#### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+ (optional but recommended)

#### Installation Steps

1. **Install Docker** (if not already installed):

   **Ubuntu/Debian:**
   ```bash
   sudo apt update
   sudo apt install docker.io docker-compose
   sudo systemctl start docker
   sudo systemctl enable docker
   sudo usermod -aG docker $USER
   ```
   
   **macOS:**
   ```bash
   # Install Docker Desktop from https://docker.com/products/docker-desktop
   # Or using Homebrew:
   brew install --cask docker
   ```
   
   **Windows:**
   Download and install Docker Desktop from https://docker.com/products/docker-desktop

2. **Clone the Repository:**
   ```bash
   git clone https://github.com/GaragaKarthikeya/explaingpt-manim-integration.git
   cd explaingpt-manim-integration
   ```

3. **Create Environment File:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration (see Configuration section below)
   nano .env
   ```

4. **Build and Run:**
   ```bash
   # Using Docker Compose (recommended)
   docker-compose up --build
   
   # Or using Docker directly
   docker build -t explaingpt-manim .
   docker run -p 8000:8000 --env-file .env explaingpt-manim
   ```

5. **Verify Installation:**
   ```bash
   curl http://localhost:8000/healthcheck
   ```

### Method 2: Manual Installation

For development or when Docker is not available.

#### Prerequisites
- Python 3.8+
- pip package manager
- FFmpeg (for video processing)
- LaTeX distribution (for math rendering)

#### System Dependencies

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
sudo apt install ffmpeg
sudo apt install texlive-full  # Or texlive-latex-recommended for minimal install
sudo apt install libcairo2-dev libpango1.0-dev
```

**macOS:**
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python3 ffmpeg
brew install --cask mactex  # Or brew install texlive for minimal install
```

**Windows (WSL2 recommended):**
```bash
# In WSL2 Ubuntu
sudo apt update
sudo apt install python3 python3-pip python3-venv ffmpeg
sudo apt install texlive-latex-recommended
```

#### Python Installation

1. **Clone Repository:**
   ```bash
   git clone https://github.com/GaragaKarthikeya/explaingpt-manim-integration.git
   cd explaingpt-manim-integration
   ```

2. **Create Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create Configuration:**
   ```bash
   cp .env.example .env
   # Edit configuration file
   nano .env
   ```

5. **Initialize Data Directories:**
   ```bash
   mkdir -p output_videos data
   ```

6. **Run the Application:**
   ```bash
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Required Configuration

### Essential Settings

Before running the system, you must configure these required settings:

1. **Gemini API Key** (Required):
   ```bash
   # Get your API key from https://makersuite.google.com/app/apikey
   GEMINI_API_KEY="your_gemini_api_key_here"
   ```

2. **Basic Application Settings:**
   ```bash
   APP_NAME="Manim Animation Server"
   PORT=8000
   MANIM_OUTPUT_DIR="./output_videos"
   ```

### Optional but Recommended

1. **Ngrok Configuration** (for external access):
   ```bash
   ENABLE_NGROK=true
   NGROK_AUTHTOKEN="your_ngrok_token"  # Get from https://ngrok.com
   ```

2. **Performance Tuning:**
   ```bash
   MAX_PARALLEL_RENDERINGS=2  # Adjust based on your system
   MIN_MEMORY_PER_WORKER_MB=2048
   ```

## Verification

### Test Basic Functionality

1. **Health Check:**
   ```bash
   curl http://localhost:8000/healthcheck
   ```
   
   Expected response:
   ```json
   {
     "status": "ok",
     "ngrok_url": "https://abc123.ngrok.io"
   }
   ```

2. **System Resources:**
   ```bash
   curl http://localhost:8000/system/resources
   ```

3. **Generate Test Animation:**
   ```bash
   curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{
          "prompt": "Show a simple circle",
          "complexity": 1
        }'
   ```

### Using the Test Client

The repository includes a test client for easy testing:

```bash
# Interactive mode
python test_client.py

# Command line mode
python test_client.py --prompt "Show a simple square" --complexity 1
```

## Troubleshooting Installation

### Common Issues

#### Docker Issues

**Problem**: Permission denied when running Docker
```bash
# Solution: Add user to docker group
sudo usermod -aG docker $USER
# Log out and log back in
```

**Problem**: Port 8000 already in use
```bash
# Solution: Use different port
docker run -p 8080:8000 --env-file .env explaingpt-manim
```

#### Python Installation Issues

**Problem**: ModuleNotFoundError for manim
```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Problem**: FFmpeg not found
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

**Problem**: LaTeX errors
```bash
# Install full LaTeX distribution
# Ubuntu/Debian
sudo apt install texlive-full

# macOS
brew install --cask mactex
```

#### Configuration Issues

**Problem**: Gemini API errors
- Verify API key is correct
- Check API key permissions
- Ensure you have API credits

**Problem**: Ngrok connection failed
- Verify authtoken is correct
- Check internet connectivity
- Try different ngrok region

### Performance Issues

**Problem**: High memory usage
```bash
# Reduce parallel workers
MAX_PARALLEL_RENDERINGS=1
MIN_MEMORY_PER_WORKER_MB=1024
```

**Problem**: Slow rendering
- Increase worker count if you have resources
- Check CPU usage with `htop` or `top`
- Consider using SSD storage

## Production Installation

### Additional Considerations

1. **Reverse Proxy** (Nginx/Apache):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

2. **Process Management** (systemd):
   ```ini
   [Unit]
   Description=Manim Animation Server
   After=network.target
   
   [Service]
   Type=exec
   User=manim
   WorkingDirectory=/opt/explaingpt-manim-integration
   Environment=PATH=/opt/explaingpt-manim-integration/venv/bin
   ExecStart=/opt/explaingpt-manim-integration/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
   Restart=always
   
   [Install]
   WantedBy=multi-user.target
   ```

3. **Security**:
   - Use environment-specific configuration
   - Set up proper firewall rules
   - Use HTTPS in production
   - Regular security updates

4. **Monitoring**:
   - Set up log aggregation
   - Monitor system resources
   - Set up alerting for failures

## Next Steps

After successful installation:

1. Read the [Quick Start Guide](quick-start.md) to create your first animation
2. Review the [Configuration Reference](configuration-reference.md) for optimization
3. Check the [API Reference](api-reference.md) for integration details
4. See [Production Deployment](production-deployment.md) for production setup

## Getting Help

If you encounter issues during installation:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Review the [FAQ](faq.md)
3. Search existing [GitHub Issues](https://github.com/GaragaKarthikeya/explaingpt-manim-integration/issues)
4. Create a new issue with:
   - Your operating system
   - Installation method used
   - Complete error messages
   - Configuration (with sensitive data removed)