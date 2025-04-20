FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for Manim and audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libcairo2-dev \
    libpango1.0-dev \
    texlive-full \
    build-essential \
    pkg-config \
    python3-dev \
    git \
    curl \
    # Add audio codecs for FLAC file support
    libavcodec-extra \
    libflac-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Copy assets directory (containing bgmusic)
COPY assets/ ./assets/

# Create output directory for videos
RUN mkdir -p ./output_videos && chmod 777 ./output_videos

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]