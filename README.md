# ExplainGPT-Manim Integration

![Manim Integration](https://img.shields.io/badge/Manim-Integration-blue)
![Python](https://img.shields.io/badge/Python-99.3%25-green)
![Docker](https://img.shields.io/badge/Docker-0.7%25-blue)
![Version](https://img.shields.io/badge/version-1.0.0-brightgreen)
![License](https://img.shields.io/badge/license-MIT-orange)

## 📝 Description

ExplainGPT-Manim Integration connects the powerful mathematical animation library [Manim](https://www.manim.community/) with the ExplainGPT frontend. This project enables dynamic, visually engaging mathematical and scientific explanations through programmatically generated animations that can be seamlessly delivered through the ExplainGPT interface.

This integration transforms how complex mathematical concepts are explained by combining:
- *Visual Learning*: Manim's ability to create clear, elegant mathematical animations
- *Interactive Explanations*: ExplainGPT's natural language processing and explanation capabilities
- *Seamless Delivery*: A robust API-driven pipeline that connects these technologies

The result is an educational experience where abstract concepts become concrete through visual representation, helping users build deeper understanding through multiple learning modalities.

## ✨ Key Features

- *Dynamic Animation Generation*: Creates mathematical visualizations based on user queries in real-time
- *Seamless Frontend Integration*: Designed specifically to work with ExplainGPT's explanation framework
- *Template System*: Customizable animation templates for common mathematical concepts
- *Multi-format Support*: Outputs in MP4, GIF, and WebM formats for flexible embedding
- *Animation Library*: Pre-built collection of common mathematical visualizations
- *Efficient Processing*: Request queuing system with caching to optimize performance
- *Containerized Deployment*: Docker-based for simple, consistent deployment
- *RESTful API*: Well-documented endpoints for straightforward integration

## 🐳 Docker Installation (Recommended)

Docker is the recommended and supported deployment method for this integration. This ensures consistent environments and simplifies the setup process by bundling all dependencies.

bash
# Clone the repository
git clone https://github.com/GaragaKarthikeya/explaingpt-manim-integration.git
cd explaingpt-manim-integration

# Build and run the Docker container
docker build -t explaingpt-manim .
docker run -p 8000:8000 --env-file .env explaingpt-manim


### Environment Configuration

Create an environment file (.env) with your configuration:


# .env file example
API_KEY=your_secure_api_key
STORAGE_PATH=/app/storage
LOG_LEVEL=INFO
MAX_QUEUE_SIZE=20
CACHE_ENABLED=true
CACHE_EXPIRY_SECONDS=86400


## 🚀 Usage

### Integration Code Example

python
from manim_integration import ExplainGPTAnimator

# Create an animator instance
animator = ExplainGPTAnimator(
    quality="production",  # Options: "draft", "medium", "production"
    cache_enabled=True,
    output_format="mp4"    # Options: "mp4", "gif", "webm"
)

# Generate an animation from a mathematical concept
animation = animator.create_animation(
    concept="derivative of sin(x)",
    duration=5,
    resolution="1080p",
    background_color="#1A1A1A",
    text_color="#FFFFFF",
    highlight_color="#3498DB",
    show_step_by_step=True
)

# Get the animation URL to pass to ExplainGPT frontend
animation_url = animation.get_url()

# Or embed directly in HTML
embed_code = animation.get_embed_code()


### Working with Templates

The system includes pre-built templates for common mathematical concepts:

python
# Using a template for limit explanation
animation = animator.use_template(
    template_name="limit_visualization",
    function="x^2",
    approach_point=3,
    duration=8
)

# Using a template for derivative visualization
animation = animator.use_template(
    template_name="derivative_geometric",
    function="sin(x)",
    point=Math.PI/4,
    show_tangent=True
)


### API Endpoints

The service exposes RESTful endpoints for the ExplainGPT frontend:

#### Generate Animation
http
POST /api/generate-animation
Content-Type: application/json

{
  "concept": "integration by parts",
  "duration": 8,
  "resolution": "1080p",
  "format": "mp4",
  "show_steps": true,
  "colors": {
    "background": "#1A1A1A",
    "text": "#FFFFFF",
    "highlight": "#E74C3C"
  }
}


#### Other Key Endpoints
- GET /api/animations/{animation_id} - Retrieves a specific animation
- GET /api/templates - Lists all available animation templates with parameters
- POST /api/custom-scene - Creates an animation from a custom scene definition
- GET /api/status/{job_id} - Checks the rendering status of a submitted job
- POST /api/feedback - Submits user feedback on an animation

## 🔄 How It Works

1. *Request Initiation*: ExplainGPT identifies a concept that would benefit from visual explanation
2. *Animation Request*: The frontend sends an API request specifying the concept and parameters
3. *Template Selection*: The system selects the appropriate animation template based on the concept
4. *Parameter Customization*: The template is populated with specific parameters for the request
5. *Manim Scene Creation*: A Manim scene is programmatically constructed with the required elements
6. *Rendering*: The scene is rendered into the requested format (MP4, GIF, WebM)
7. *Delivery*: The rendered animation is made available via URL or direct embedding
8. *Display*: ExplainGPT presents the animation alongside textual explanation

This workflow allows for seamless integration of visual explanations within the conversational flow of ExplainGPT, enhancing user understanding through multiple learning modalities.

## ⚙ Configuration Options

The integration can be configured through environment variables or a configuration file:

yaml
# config.yaml
rendering:
  quality: production  # draft, medium, production
  fps: 60
  resolution: 1080p
  cache_enabled: true
  cache_expiry: 604800  # 7 days in seconds

api:
  host: 0.0.0.0
  port: 8000
  rate_limit: 100  # requests per minute
  max_queue_size: 50

storage:
  type: s3  # local, s3, azure
  s3_bucket: animations-bucket
  s3_region: us-west-2
  local_path: /app/storage/animations
  
manim:
  custom_directories:
    - /app/templates
    - /app/custom_scenes
  plugins:
    - manim_voiceover


## 📊 Supported Mathematical Concepts

The integration currently supports animations for:

| Category | Supported Concepts |
|----------|-------------------|
| Calculus | Limits, Derivatives, Integrals, Series |
| Linear Algebra | Vectors, Matrices, Transformations, Eigenvalues |
| Probability | Distributions, Random Variables, Bayes' Theorem |
| Geometry | Shapes, Transformations, Projections |
| Statistics | Data Visualization, Regression, Hypothesis Testing |

Each concept has customizable parameters to tailor the explanation to the specific context.

## 📚 Documentation

For detailed documentation, refer to the [docs](./docs) directory:

- [API Reference](./docs/api-reference.md): Complete API documentation
- [Animation Templates](./docs/templates.md): Available templates and parameters
- [Integration Guide](./docs/integration-guide.md): How to integrate with ExplainGPT
- [Configuration Guide](./docs/configuration.md): All configuration options
- [Troubleshooting](./docs/troubleshooting.md): Common issues and solutions

## 🔍 Performance Optimization

For optimal performance:
- Enable caching to reuse commonly requested animations
- Configure appropriate queue sizes based on your traffic patterns
- Use the template system instead of custom scenes when possible
- Monitor API usage and adjust rate limits accordingly
- Consider horizontal scaling for high-traffic deployments

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create your feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add some amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

Areas we're particularly interested in:
- New animation templates for mathematical concepts
- Performance improvements
- Enhanced integration capabilities
- Documentation improvements

## 📝 Changelog

See the [CHANGELOG.md](CHANGELOG.md) file for details on version history and updates.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Manim Community](https://www.manim.community/) for the amazing animation engine
- ExplainGPT team for the frontend integration support
- All contributors who have helped shape this project

## 📞 Contact

Karthikeya Garaga - [@GaragaKarthikeya](https://github.com/GaragaKarthikeya)

Project Link: [https://github.com/GaragaKarthikeya/explaingpt-manim-integration](https://github.com/GaragaKarthikeya/explaingpt-manim-integration)

---
Last updated: 2025-04-19
```