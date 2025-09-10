# Quick Start Guide

Get up and running with the ExplainGPT-Manim Integration in just a few minutes!

## Prerequisites

Before starting, ensure you have:
- Docker installed (recommended) OR Python 3.8+
- A Google Gemini API key ([Get one here](https://makersuite.google.com/app/apikey))
- 4GB+ RAM available
- Stable internet connection

## 5-Minute Setup

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/GaragaKarthikeya/explaingpt-manim-integration.git
cd explaingpt-manim-integration

# Create configuration file
cp .env.example .env
```

### Step 2: Configure API Key

Edit the `.env` file and add your Gemini API key:

```bash
# Essential configuration
GEMINI_API_KEY="your_gemini_api_key_here"
PORT=8000
MAX_PARALLEL_RENDERINGS=2
```

### Step 3: Run with Docker (Recommended)

```bash
# Build and start the service
docker-compose up --build

# Or using Docker directly
docker build -t explaingpt-manim .
docker run -p 8000:8000 --env-file .env explaingpt-manim
```

**Alternative: Manual Installation**
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Step 4: Verify Installation

```bash
# Check if the service is running
curl http://localhost:8000/healthcheck
```

Expected response:
```json
{
  "status": "ok",
  "ngrok_url": "https://abc123.ngrok.io"
}
```

## Create Your First Animation

### Using the Test Client (Easiest)

The repository includes a user-friendly test client:

```bash
# Interactive mode
python test_client.py

# Command line mode
python test_client.py --prompt "Show how a derivative works" --complexity 2
```

**Example Output:**
```
==========================================
          MANIM ANIMATION API CLIENT
==========================================

Sending animation request to http://localhost:8000...
Prompt: 'Show how a derivative works'
Complexity: 2

Job submitted with ID: 550e8400-e29b-41d4-a716-446655440000

✅ ⚡ Status: Processing
✅ Animation Complete!

🎬 Video available at: https://abc123.ngrok.io/videos/550e8400-e29b-41d4-a716-446655440000.mp4
```

### Using curl Commands

```bash
# Step 1: Request animation generation
curl -X POST http://localhost:8000/generate \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Visualize the Pythagorean theorem with a right triangle",
       "complexity": 2
     }'
```

Response:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Animation job created and queued",
  "status": "queued"
}
```

```bash
# Step 2: Check status (repeat until completed)
curl http://localhost:8000/status/550e8400-e29b-41d4-a716-446655440000
```

Response when completed:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "success": true,
  "video_url": "http://localhost:8000/videos/550e8400-e29b-41d4-a716-446655440000.mp4",
  "error": null
}
```

```bash
# Step 3: Download the video
curl -o "my_animation.mp4" \
     "http://localhost:8000/videos/550e8400-e29b-41d4-a716-446655440000.mp4"
```

### Using JavaScript/Web Browser

```html
<!DOCTYPE html>
<html>
<head>
    <title>Manim Animation Test</title>
</head>
<body>
    <h1>Create Animation</h1>
    <input type="text" id="prompt" placeholder="Enter your animation prompt" value="Show a sine wave">
    <select id="complexity">
        <option value="1">Simple</option>
        <option value="2" selected>Moderate</option>
        <option value="3">Complex</option>
    </select>
    <button onclick="generateAnimation()">Generate</button>
    
    <div id="status"></div>
    <video id="video" controls style="display:none; width:100%; max-width:800px;"></video>

    <script>
        async function generateAnimation() {
            const prompt = document.getElementById('prompt').value;
            const complexity = parseInt(document.getElementById('complexity').value);
            const statusDiv = document.getElementById('status');
            const video = document.getElementById('video');
            
            statusDiv.innerHTML = 'Requesting animation...';
            
            try {
                // Generate animation
                const response = await fetch('http://localhost:8000/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ prompt, complexity })
                });
                
                const result = await response.json();
                const jobId = result.job_id;
                
                statusDiv.innerHTML = `Job created: ${jobId}<br>Checking status...`;
                
                // Poll for completion
                const pollStatus = async () => {
                    const statusResponse = await fetch(`http://localhost:8000/status/${jobId}`);
                    const status = await statusResponse.json();
                    
                    if (status.success && status.video_url) {
                        statusDiv.innerHTML = 'Animation ready!';
                        video.src = status.video_url;
                        video.style.display = 'block';
                    } else if (status.error) {
                        statusDiv.innerHTML = `Error: ${status.error}`;
                    } else {
                        statusDiv.innerHTML = 'Processing... (this may take 1-2 minutes)';
                        setTimeout(pollStatus, 3000);
                    }
                };
                
                pollStatus();
                
            } catch (error) {
                statusDiv.innerHTML = `Error: ${error.message}`;
            }
        }
    </script>
</body>
</html>
```

## Animation Complexity Levels

Choose the right complexity for your needs:

### Level 1: Simple
- **Time**: 15-30 seconds to generate
- **Best for**: Basic shapes, simple concepts
- **Example**: "Show a circle" or "Draw a triangle"

### Level 2: Moderate (Recommended)
- **Time**: 30-60 seconds to generate  
- **Best for**: Mathematical concepts, educational content
- **Example**: "Explain derivatives" or "Show the Pythagorean theorem"

### Level 3: Complex
- **Time**: 60-120 seconds to generate
- **Best for**: Detailed explanations, multi-step processes
- **Example**: "Show how Fourier transforms work with multiple examples"

## Example Animation Prompts

### Mathematics
```bash
"Show how limits work as x approaches infinity"
"Visualize the chain rule for derivatives"
"Demonstrate the relationship between sin and cos functions"
"Show how matrix multiplication works geometrically"
"Visualize the concept of eigenvectors and eigenvalues"
```

### Physics
```bash
"Show simple harmonic motion with a spring"
"Visualize electromagnetic waves propagating"
"Demonstrate conservation of momentum with colliding spheres"
"Show how potential energy converts to kinetic energy"
```

### Computer Science
```bash
"Visualize how quicksort algorithm works"
"Show how binary search trees are constructed"
"Demonstrate breadth-first search on a graph"
"Visualize how neural network training works"
```

## Integration with ExplainGPT

If you're integrating with an ExplainGPT frontend:

```javascript
// Example integration code
class ManimAnimationService {
    constructor(apiUrl = 'http://localhost:8000') {
        this.apiUrl = apiUrl;
    }
    
    async generateAnimation(prompt, complexity = 2) {
        const response = await fetch(`${this.apiUrl}/generate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ prompt, complexity })
        });
        
        return response.json();
    }
    
    async getAnimationStatus(jobId) {
        const response = await fetch(`${this.apiUrl}/status/${jobId}`);
        return response.json();
    }
    
    async waitForAnimation(jobId, maxWait = 300000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < maxWait) {
            const status = await this.getAnimationStatus(jobId);
            
            if (status.success) {
                return status.video_url;
            } else if (status.error) {
                throw new Error(status.error);
            }
            
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
        
        throw new Error('Animation generation timed out');
    }
}

// Usage in your ExplainGPT integration
const manimService = new ManimAnimationService();

async function enhanceExplanation(prompt) {
    try {
        const animation = await manimService.generateAnimation(prompt);
        const videoUrl = await manimService.waitForAnimation(animation.job_id);
        
        // Display video in your UI
        displayAnimation(videoUrl);
    } catch (error) {
        console.error('Animation failed:', error);
        // Fallback to text-only explanation
    }
}
```

## Monitoring and System Info

### Check System Resources
```bash
curl http://localhost:8000/system/resources
```

### View Job Performance
```bash
curl http://localhost:8000/system/jobs/performance
```

### Access Ngrok URL
If ngrok is enabled, your animations will be accessible via the public URL:
```bash
# Check your current ngrok URL
curl http://localhost:8000/healthcheck
```

## Common Issues and Quick Fixes

### Animation Generation Fails
1. **Check Gemini API key**: Ensure it's valid and has credits
2. **Verify prompt**: Try simpler prompts first
3. **Check logs**: Look at console output for error details

### Slow Performance
1. **Reduce complexity**: Start with level 1 animations
2. **Lower worker count**: Set `MAX_PARALLEL_RENDERINGS=1` 
3. **Check system resources**: Ensure sufficient RAM/CPU

### Port Already in Use
```bash
# Use different port
PORT=8080 uvicorn app.main:app --host 0.0.0.0 --port 8080
```

### Can't Access from Other Devices
1. **Enable ngrok**: Set `ENABLE_NGROK=true` in `.env`
2. **Or bind to all interfaces**: Use `--host 0.0.0.0`

## Next Steps

Now that you have the system running:

1. **Explore the API**: Check out the [API Reference](api-reference.md)
2. **Optimize Performance**: Review [Configuration Reference](configuration-reference.md)
3. **Production Setup**: See [Production Deployment](production-deployment.md)
4. **Integration**: Read [Integration Guide](integration-guide.md)

## Need Help?

- **Documentation**: Check out the full [documentation](README.md)
- **Issues**: Report problems on [GitHub](https://github.com/GaragaKarthikeya/explaingpt-manim-integration/issues)
- **Examples**: More examples in [API Examples](api-examples.md)

Happy animating! 🎬✨