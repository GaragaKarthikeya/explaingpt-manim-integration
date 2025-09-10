# Frequently Asked Questions (FAQ)

## General Questions

### What is the ExplainGPT-Manim Integration?

The ExplainGPT-Manim Integration is a service that automatically generates mathematical and scientific animations using AI-powered code generation and the Manim animation library. It's designed to enhance text-based explanations with visual content.

### What makes this different from other animation tools?

- **AI-Powered**: Uses Google Gemini to generate Manim code automatically
- **API-First**: RESTful API designed for easy integration
- **Education-Focused**: Specifically optimized for mathematical and scientific content
- **Real-Time**: Generates animations on-demand based on text prompts
- **Scalable**: Built with microservices architecture for production use

### Who is this for?

- **Educational Platforms**: Enhance math and science explanations
- **Content Creators**: Generate visual content for tutorials
- **Developers**: Integrate animation generation into applications
- **Researchers**: Visualize mathematical concepts quickly

## Installation and Setup

### What are the system requirements?

**Minimum:**
- 4GB RAM, 2 CPU cores
- Python 3.8+ OR Docker
- 2GB free disk space

**Recommended:**
- 8GB+ RAM, 4+ CPU cores
- Linux or macOS
- SSD storage
- Stable internet connection

### Do I need to install Manim separately?

No, Manim is included in the Docker image and Python requirements. The service handles all Manim dependencies automatically.

### What if I don't have Docker?

You can install manually using Python, but Docker is strongly recommended for consistency. See the [Installation Guide](installation.md) for manual installation steps.

### How do I get a Gemini API key?

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Add it to your `.env` file as `GEMINI_API_KEY`

### Is the Gemini API free?

Google Gemini has a free tier with usage limits. Check [Google's pricing](https://ai.google.dev/pricing) for current rates and limits.

## Usage and Features

### What types of animations can it generate?

- **Mathematics**: Derivatives, integrals, limits, functions, geometry
- **Physics**: Motion, waves, forces, energy transformations
- **Computer Science**: Algorithms, data structures, sorting
- **Statistics**: Distributions, data visualization, regression

### How long does it take to generate an animation?

- **Simple (Level 1)**: 15-30 seconds
- **Moderate (Level 2)**: 30-60 seconds  
- **Complex (Level 3)**: 60-120 seconds

Times vary based on system resources and prompt complexity.

### What's the difference between complexity levels?

- **Level 1**: Basic shapes and simple concepts
- **Level 2**: Detailed mathematical explanations (recommended)
- **Level 3**: Comprehensive multi-step visualizations

### Can I customize the animation style?

Currently, animations use Manim's default styling. Future versions may include style customization options.

### What video formats are supported?

The service outputs MP4 videos with H.264 encoding, compatible with all modern browsers and devices.

### How long are the generated videos?

Typically 10-30 seconds, depending on complexity. The AI determines appropriate duration based on content.

## API and Integration

### Is there authentication required?

No authentication is required currently. The service uses rate limiting (5 requests per minute per IP) to prevent abuse.

### What's the API rate limit?

Default is 5 requests per minute per IP address. This can be configured via the `RATE_LIMIT_PER_MINUTE` environment variable.

### Can I use this in production?

Yes, but consider:
- Setting up proper monitoring
- Configuring appropriate resource limits
- Implementing caching for frequently requested animations
- Using HTTPS and security best practices

### How do I integrate with a web application?

Check the [Integration Guide](integration-guide.md) for detailed examples including JavaScript, React, and Vue.js implementations.

### Can I use this with mobile applications?

Yes, the REST API works with any HTTP client. Mobile apps can request animations and display the returned video URLs.

### Is there a JavaScript SDK?

Not officially, but the [Integration Guide](integration-guide.md) includes complete JavaScript examples you can use as a starting point.

## Performance and Scaling

### How many animations can it generate simultaneously?

By default, it processes 2-4 animations concurrently (based on CPU cores). This is configurable via `MAX_PARALLEL_RENDERINGS`.

### My system is running out of memory. What can I do?

1. Reduce `MAX_PARALLEL_RENDERINGS` to 1
2. Lower `MIN_MEMORY_PER_WORKER_MB` to 1024
3. Set `MAX_MEMORY_PERCENT` to 60-70
4. Enable dynamic scaling with `ENABLE_DYNAMIC_SCALING=true`

### Can I run multiple instances?

Currently, each instance runs independently. Future versions may support distributed processing across multiple servers.

### How do I optimize performance?

1. **Use SSD storage** for faster video writing
2. **Increase RAM** to allow more concurrent workers  
3. **Enable caching** to reuse common animations
4. **Use ngrok** to reduce local network overhead
5. **Monitor resources** via `/system/resources` endpoint

## Troubleshooting

### Animations are not generating. What's wrong?

1. **Check the health endpoint**: `curl http://localhost:8000/healthcheck`
2. **Verify Gemini API key** is correct and has quota
3. **Check system resources** - ensure sufficient memory
4. **Look at logs** for specific error messages

### I get "Rate limit exceeded" errors

Wait 60 seconds between requests, or increase `RATE_LIMIT_PER_MINUTE` in your configuration.

### The service starts but animations fail

1. **Verify FFmpeg is installed**: `ffmpeg -version`
2. **Check LaTeX installation**: `latex --version`  
3. **Ensure output directory is writable**: `ls -la ./output_videos`
4. **Test with simple prompt**: "Show a circle"

### Ngrok tunnel fails to connect

1. **Verify auth token** is correct from ngrok dashboard
2. **Check internet connectivity**: `ping ngrok.com`
3. **Disable temporarily**: Set `ENABLE_NGROK=false`
4. **Try different region** if available

### Videos won't play in browser

1. **Check video file exists**: `ls ./output_videos/`
2. **Test video locally**: Try opening with VLC or similar
3. **Verify URL format** is correct (should end in `.mp4`)
4. **Check CORS settings** if accessing from web page

## Development and Customization

### Can I modify the generated code?

The service generates Manim code automatically. You could modify the AI prompts or add post-processing, but direct code editing isn't supported in the API.

### How can I add new animation types?

Currently, animation types are determined by the AI model. You can influence output through prompt engineering in your integration.

### Is the source code available?

Yes, the project is open source. You can contribute improvements or customize for your needs.

### How do I contribute?

1. Fork the repository on GitHub
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

See the [Contributing Guide](contributing.md) for detailed instructions.

### Can I use a different AI model?

Currently only Google Gemini is supported. The architecture could support other models with code changes.

## Security and Privacy

### Is my data stored permanently?

No, generated videos are temporary and cleaned up based on `CACHE_EXPIRY_HOURS` setting (default 24 hours).

### Are prompts sent to external services?

Yes, prompts are sent to Google Gemini API for code generation. Review Google's privacy policy for their data handling.

### Is this safe to run in production?

The service includes security measures:
- Sandboxed code execution
- Timeout protection
- Resource limitations
- Input validation

However, review the security considerations in the [Production Deployment Guide](production-deployment.md).

### Can I run this offline?

No, the service requires internet access for the Gemini API. Future versions might support offline models.

## Pricing and Licensing

### What does this cost to run?

- **Service**: Free (open source)
- **Gemini API**: Pay-per-use (has free tier)
- **Infrastructure**: Your hosting costs (minimal for light usage)

### What's the license?

The project uses the MIT License, allowing free use, modification, and distribution.

### Can I use this commercially?

Yes, the MIT license permits commercial use. However, review Google Gemini's terms of service for their usage restrictions.

## Getting Help

### Where can I find more documentation?

- [Installation Guide](installation.md)
- [API Reference](api-reference.md)  
- [Configuration Reference](configuration-reference.md)
- [Troubleshooting Guide](troubleshooting.md)

### How do I report bugs?

1. Check existing [GitHub Issues](https://github.com/GaragaKarthikeya/explaingpt-manim-integration/issues)
2. Create a new issue with:
   - System information
   - Steps to reproduce
   - Error messages
   - Configuration (without API keys)

### Where can I ask questions?

- GitHub Issues for bug reports and feature requests
- GitHub Discussions for general questions
- Check this FAQ first for common questions

### Is there community support?

The project is community-driven. Help improve it by:
- Reporting issues
- Contributing code improvements
- Updating documentation
- Sharing usage examples

### Can I get commercial support?

Currently no commercial support is available. The project relies on community contributions and open source collaboration.

## Future Plans

### What features are planned?

Potential future features include:
- Style customization options
- Multiple output formats (GIF, WebM)
- Webhook notifications
- Distributed processing
- Additional AI model support

### How can I influence the roadmap?

- Open feature request issues on GitHub
- Contribute code for desired features
- Participate in community discussions
- Sponsor development if you represent an organization

### Is there a mobile app?

No mobile app is planned. The REST API can be used by mobile applications to generate and display animations.

---

## Still Need Help?

If your question isn't answered here:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search [existing GitHub issues](https://github.com/GaragaKarthikeya/explaingpt-manim-integration/issues)
3. Create a new issue with detailed information
4. Join community discussions

We're here to help make mathematical visualization accessible to everyone! 🎬📐