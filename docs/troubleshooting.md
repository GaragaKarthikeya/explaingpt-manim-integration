# Troubleshooting Guide

This guide helps you diagnose and fix common issues with the ExplainGPT-Manim Integration system.

## Quick Diagnostics

### Health Check

First, verify the service is running:

```bash
curl http://localhost:8000/healthcheck
```

**Expected Response:**
```json
{
  "status": "ok",
  "ngrok_url": "https://abc123.ngrok.io"
}
```

### System Resources

Check system resource usage:

```bash
curl http://localhost:8000/system/resources
```

This shows memory usage, CPU load, and worker status.

## Common Issues

### 1. Installation and Setup Issues

#### Docker Issues

**Problem**: `docker: permission denied`
```bash
# Add user to docker group
sudo usermod -aG docker $USER
# Log out and back in, or run:
newgrp docker
```

**Problem**: `port 8000 already in use`
```bash
# Find what's using the port
sudo lsof -i :8000

# Kill the process or use different port
PORT=8080 docker run -p 8080:8000 --env-file .env explaingpt-manim
```

**Problem**: Docker build fails
```bash
# Clean Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t explaingpt-manim .
```

#### Python Installation Issues

**Problem**: `ModuleNotFoundError: No module named 'manim'`
```bash
# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

**Problem**: `FFmpeg not found`
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows (in WSL)
sudo apt install ffmpeg
```

**Problem**: LaTeX compilation errors
```bash
# Install full LaTeX distribution
# Ubuntu/Debian
sudo apt install texlive-full

# macOS
brew install --cask mactex

# Verify installation
latex --version
```

### 2. Configuration Issues

#### API Key Problems

**Problem**: `Authentication error` or `Invalid API key`

1. **Verify API Key Format:**
   ```bash
   # Should start with 'AI' and be about 40 characters
   echo $GEMINI_API_KEY | wc -c
   ```

2. **Test API Key:**
   ```python
   import google.generativeai as genai
   genai.configure(api_key="your_api_key_here")
   models = list(genai.list_models())
   print(f"Available models: {len(models)}")
   ```

3. **Check API Quota:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Verify you have available quota
   - Check billing settings

**Problem**: `Model not found: models/gemini-2.0-flash`

1. **List Available Models:**
   ```bash
   python list_gemini_models.py
   ```

2. **Update Model Name:**
   ```bash
   # In .env file, use exact model name from list
   GEMINI_MODEL="models/gemini-1.5-flash"
   ```

#### Ngrok Issues

**Problem**: `Ngrok tunnel failed to start`

1. **Verify Auth Token:**
   ```bash
   # Test ngrok manually
   ngrok config add-authtoken your_auth_token
   ngrok http 8000
   ```

2. **Check Network Connectivity:**
   ```bash
   ping ngrok.com
   ```

3. **Disable Ngrok (temporary fix):**
   ```bash
   ENABLE_NGROK=false
   ```

### 3. Performance Issues

#### High Memory Usage

**Problem**: System runs out of memory

1. **Check Current Usage:**
   ```bash
   free -h
   htop  # or top
   ```

2. **Reduce Workers:**
   ```bash
   MAX_PARALLEL_RENDERINGS=1
   MIN_MEMORY_PER_WORKER_MB=1024
   ```

3. **Enable Dynamic Scaling:**
   ```bash
   ENABLE_DYNAMIC_SCALING=true
   MAX_MEMORY_PERCENT=70
   ```

#### Slow Animation Generation

**Problem**: Animations take too long to generate

1. **Check System Load:**
   ```bash
   curl http://localhost:8000/system/resources | jq '.system_resources'
   ```

2. **Optimize Settings:**
   ```bash
   # Reduce complexity for testing
   MAX_PARALLEL_RENDERINGS=2
   MONITORING_INTERVAL_SEC=15
   ```

3. **Use Simpler Prompts:**
   ```bash
   # Start with complexity 1
   curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "Show a circle", "complexity": 1}'
   ```

#### Queue Issues

**Problem**: Jobs stuck in queue

1. **Check Queue Status:**
   ```bash
   curl http://localhost:8000/system/resources | jq '.job_metrics'
   ```

2. **Restart Workers:**
   ```bash
   # In Docker
   docker restart container_name
   
   # Manual installation
   # Restart the uvicorn process
   ```

3. **Clear Queue (if needed):**
   ```bash
   # This requires manual intervention - restart the service
   ```

### 4. API and Integration Issues

#### CORS Errors

**Problem**: `CORS policy: No 'Access-Control-Allow-Origin' header`

1. **Verify CORS Settings:**
   - The API should allow all origins by default
   - Check browser console for specific error

2. **Test with curl (bypasses CORS):**
   ```bash
   curl -X POST http://localhost:8000/generate \
        -H "Content-Type: application/json" \
        -d '{"prompt": "test", "complexity": 1}'
   ```

3. **Use Ngrok for external access:**
   ```bash
   ENABLE_NGROK=true
   # Use the ngrok URL instead of localhost
   ```

#### Rate Limiting

**Problem**: `HTTP 429 Too Many Requests`

1. **Check Rate Limit:**
   ```bash
   # Default is 5 requests per minute per IP
   echo "Current rate limit: $RATE_LIMIT_PER_MINUTE per minute"
   ```

2. **Increase Rate Limit:**
   ```bash
   RATE_LIMIT_PER_MINUTE=10
   ```

3. **Wait Before Retrying:**
   ```bash
   # Wait 60 seconds before next request
   sleep 60
   ```

#### Job Status Issues

**Problem**: Job never completes or shows wrong status

1. **Check Logs:**
   ```bash
   # Docker logs
   docker logs container_name -f
   
   # Manual installation - check console output
   ```

2. **Verify Job ID:**
   ```bash
   curl http://localhost:8000/status/your_job_id_here
   ```

3. **Check for Errors:**
   ```bash
   curl http://localhost:8000/system/jobs/performance
   ```

### 5. Video and Output Issues

#### Video Not Generated

**Problem**: Job completes but no video URL

1. **Check Output Directory:**
   ```bash
   ls -la ./output_videos/
   ```

2. **Verify Permissions:**
   ```bash
   # Ensure directory is writable
   chmod 755 ./output_videos
   ```

3. **Check FFmpeg:**
   ```bash
   ffmpeg -version
   ```

#### Video Cannot Be Played

**Problem**: Video file exists but won't play

1. **Check Video File:**
   ```bash
   file ./output_videos/your_job_id.mp4
   ffprobe ./output_videos/your_job_id.mp4
   ```

2. **Test Video Locally:**
   ```bash
   # Try playing with VLC or other player
   vlc ./output_videos/your_job_id.mp4
   ```

3. **Verify Encoding:**
   - Manim outputs should be H.264 MP4
   - Compatible with most browsers

### 6. Development and Debugging

#### Enable Debug Logging

1. **Set Log Level:**
   ```bash
   LOG_LEVEL=DEBUG
   ```

2. **Python Debug Mode:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
   ```

3. **Docker Debug:**
   ```bash
   docker run -it --env-file .env explaingpt-manim /bin/bash
   # Then run commands manually to debug
   ```

#### Test Components Individually

1. **Test Gemini API:**
   ```python
   import google.generativeai as genai
   genai.configure(api_key="your_key")
   model = genai.GenerativeModel("gemini-2.0-flash")
   response = model.generate_content("Write a simple Python script")
   print(response.text)
   ```

2. **Test Manim Installation:**
   ```bash
   python -c "from manim import *; print('Manim imported successfully')"
   ```

3. **Test File I/O:**
   ```bash
   # Check if output directory is writable
   touch ./output_videos/test.txt && rm ./output_videos/test.txt
   echo "Directory is writable"
   ```

## Error Code Reference

### HTTP Status Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 400 | Bad Request | Invalid JSON, missing required fields |
| 404 | Not Found | Job ID doesn't exist, video file missing |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Gemini API error, system resource issue |

### Common Error Messages

#### `Failed to create animation job`
- **Cause**: System resource exhaustion, queue full
- **Fix**: Reduce load, wait for queue to clear

#### `Job with ID xyz not found`  
- **Cause**: Invalid job ID, job expired
- **Fix**: Use correct job ID, check if job was cleaned up

#### `Gemini API error`
- **Cause**: API key invalid, quota exceeded, model unavailable
- **Fix**: Check API key, verify quota, update model name

#### `FFmpeg error`
- **Cause**: Video encoding failure, corrupted intermediate files
- **Fix**: Restart service, check disk space, verify FFmpeg installation

## Performance Monitoring

### Key Metrics to Watch

1. **Memory Usage:**
   ```bash
   # Should stay below 80% typically
   free | grep Mem | awk '{print ($3/$2) * 100.0}'
   ```

2. **CPU Load:**
   ```bash
   # Load average should be < number of CPU cores
   uptime
   ```

3. **Queue Depth:**
   ```bash
   curl http://localhost:8000/system/resources | jq '.job_metrics.queued_jobs'
   ```

4. **Error Rate:**
   ```bash
   curl http://localhost:8000/system/resources | jq '.job_metrics.failed_jobs'
   ```

### Performance Tuning

1. **For High-Memory Systems:**
   ```bash
   MAX_PARALLEL_RENDERINGS=4
   MIN_MEMORY_PER_WORKER_MB=4096
   MAX_MEMORY_PERCENT=85
   ```

2. **For Low-Memory Systems:**
   ```bash
   MAX_PARALLEL_RENDERINGS=1
   MIN_MEMORY_PER_WORKER_MB=1024
   MAX_MEMORY_PERCENT=60
   ```

3. **For CPU-Intensive Workloads:**
   ```bash
   ENABLE_DYNAMIC_SCALING=true
   HIGH_LOAD_THRESHOLD=0.9
   LOW_LOAD_THRESHOLD=0.2
   ```

## Getting Help

### Information to Gather

When reporting issues, include:

1. **System Information:**
   ```bash
   uname -a
   python --version
   docker --version
   ```

2. **Configuration (sanitized):**
   ```bash
   # Remove sensitive values like API keys
   env | grep -E "(GEMINI|NGROK|MANIM)" | sed 's/=.*/=***/'
   ```

3. **Error Messages:**
   ```bash
   # Recent logs
   docker logs container_name --tail 100
   ```

4. **System Resources:**
   ```bash
   curl http://localhost:8000/system/resources
   ```

### Where to Get Help

1. **Documentation**: Check other guides in this documentation
2. **GitHub Issues**: [Create an issue](https://github.com/GaragaKarthikeya/explaingpt-manim-integration/issues)
3. **FAQ**: Check the [FAQ](faq.md) for common questions

### Before Opening an Issue

1. Search existing issues for similar problems
2. Try the solutions in this troubleshooting guide
3. Test with minimal configuration
4. Gather all relevant information listed above

## Preventive Maintenance

### Regular Checks

1. **Update Dependencies:**
   ```bash
   pip list --outdated
   pip install --upgrade -r requirements.txt
   ```

2. **Clean Old Files:**
   ```bash
   # Remove videos older than 7 days
   find ./output_videos -name "*.mp4" -mtime +7 -delete
   ```

3. **Monitor System Resources:**
   ```bash
   # Set up monitoring alerts for memory/CPU usage
   ```

4. **Backup Configuration:**
   ```bash
   cp .env .env.backup.$(date +%Y%m%d)
   ```

### Security Updates

1. **Keep Docker Images Updated:**
   ```bash
   docker pull python:3.11-slim
   docker-compose build --no-cache
   ```

2. **Update System Packages:**
   ```bash
   sudo apt update && sudo apt upgrade
   ```

3. **Rotate API Keys Periodically:**
   - Generate new Gemini API key
   - Update configuration
   - Verify functionality