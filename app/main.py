from fastapi import FastAPI, Query, Path, HTTPException, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import os
import uvicorn
import logging
import psutil
from pyngrok import ngrok
import time

from app.models import AnimationRequest, AnimationResponse, JobStatus, AnimationResult
from app.services.job_queue import job_queue
from app.services.renderer import renderer
from app.services.storage import storage_service
from app.utils.error_handler import app_error_handler, AppError
from app.utils.validators import validate_animation_request, rate_limit_middleware
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description="A FastAPI backend for generating animations using Manim",
    version="1.0.0",
)

# Add CORS middleware - CRITICALLY IMPORTANT for browser API calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],  # Explicitly list allowed methods
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],  # Necessary for file downloads
)

# Add exception handlers
app.add_exception_handler(AppError, app_error_handler)

# Mount static files directory for video serving
app.mount("/videos", StaticFiles(directory=settings.MANIM_OUTPUT_DIR), name="videos")

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add process time header and handle rate limiting."""
    try:
        # Log requests for debugging
        logger.debug(f"Request: {request.method} {request.url}")
        
        await rate_limit_middleware(request)
        response = await call_next(request)
        
        # Add CORS headers to all responses as a fallback
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    except HTTPException as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

@app.on_event("startup")
async def startup_event():
    """Start background services on application startup."""
    # Start existing services
    renderer.start()
    storage_service.start()
    job_queue.start()  # Start the job queue service
    
    # Set up ngrok only if enabled
    if settings.ENABLE_NGROK:
        ngrok_auth_token = os.getenv("NGROK_AUTHTOKEN") or settings.NGROK_AUTHTOKEN
        
        if ngrok_auth_token:
            logger.info("Setting up ngrok tunnel...")
            max_retries = settings.NGROK_MAX_RETRIES
            retry_count = 0
            retry_delay = settings.NGROK_RETRY_DELAY  # seconds
            
            while retry_count < max_retries:
                try:
                    # Configure ngrok
                    ngrok.set_auth_token(ngrok_auth_token)
                    
                    # Kill any existing tunnels to prevent conflicts
                    ngrok.kill()
                    
                    # Check if we have a static domain in the environment variables
                    static_domain = os.getenv("NGROK_DOMAIN")
                    
                    try:
                        # If we have a static domain, use it
                        if static_domain:
                            logger.info(f"Connecting to ngrok with static domain: {static_domain}")
                            # Use static domain with the connect call
                            http_tunnel = ngrok.connect(8000, domain=static_domain)
                        else:
                            # Otherwise connect with a random domain
                            logger.info("Connecting to ngrok with random domain")
                            http_tunnel = ngrok.connect(8000)
                        
                        tunnel_url = http_tunnel.public_url
                        logger.info(f"Ngrok tunnel established at: {tunnel_url}")
                        
                        # Update the NGROK_BASE_URL environment variable and settings
                        os.environ["NGROK_BASE_URL"] = tunnel_url
                        settings.NGROK_BASE_URL = tunnel_url
                        logger.info(f"Updated NGROK_BASE_URL to: {tunnel_url}")
                        
                        # Update storage service with base URL if needed
                        if hasattr(storage_service, 'set_base_url'):
                            storage_service.set_base_url(tunnel_url)
                        
                        break  # Success, exit retry loop
                    except Exception as e:
                        logger.error(f"Failed to establish ngrok tunnel: {str(e)}")
                        retry_count += 1
                        
                        if retry_count >= max_retries:
                            logger.warning(f"Max retries ({max_retries}) reached for ngrok setup. Continuing without tunnel.")
                        else:
                            logger.info(f"Retry {retry_count}/{max_retries} in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                
                except Exception as e:
                    logger.error(f"Error configuring ngrok: {str(e)}")
                    retry_count += 1
                    
                    if retry_count >= max_retries:
                        logger.warning(f"Max retries ({max_retries}) reached for ngrok setup. Continuing without tunnel.")
                    else:
                        logger.info(f"Retry {retry_count}/{max_retries} in {retry_delay} seconds...")
                        time.sleep(retry_delay)
        else:
            logger.warning("NGROK_AUTHTOKEN not set. Running without ngrok tunnel.")
    else:
        logger.info("Ngrok disabled via ENABLE_NGROK setting. Running without ngrok tunnel.")
    
    logger.info("Application started")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop background services on application shutdown."""
    # Stop existing services
    renderer.stop()
    storage_service.stop()
    job_queue.stop()  # Stop the job queue service
    
    # Stop ngrok
    try:
        ngrok.kill()
        logger.info("Ngrok tunnel closed")
    except Exception as e:
        logger.error(f"Error closing ngrok tunnel: {str(e)}")
    
    logger.info("Application shutdown")

@app.post("/generate", response_model=AnimationResponse)
async def generate_animation(
    request: AnimationRequest,
    _: None = Depends(validate_animation_request)
):
    """Generate an animation from a prompt."""
    try:
        logger.info(f"Generating animation with prompt: {request.prompt}")
        job_info = job_queue.add_job(request)
        
        return AnimationResponse(
            job_id=job_info.job_id,
            message="Animation job created and queued",
            status=job_info.status
        )
    except ValueError as e:
        logger.warning(f"Rate limit exceeded: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e)
        )
    except Exception as e:
        logger.exception(f"Error creating animation job: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create animation job: {str(e)}"
        )

@app.get("/status/{job_id}", response_model=AnimationResult)
async def get_job_status(job_id: str = Path(..., description="The ID of the job")):
    """Get the status of an animation job."""
    logger.debug(f"Checking status for job: {job_id}")
    job_info = job_queue.get_job_status(job_id)
    
    if job_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found"
        )
    
    result = AnimationResult(
        job_id=job_id,
        success=job_info.status == JobStatus.COMPLETED,
        error=job_info.error
    )
    
    if job_info.status == JobStatus.COMPLETED and storage_service.video_exists(job_id):
        video_url = storage_service.get_video_url(job_id)
        
        # Ensure we have an absolute URL
        if not video_url.startswith(('http://', 'https://')):
            base_url = settings.NGROK_BASE_URL or f"http://localhost:{settings.PORT}"
            video_url = f"{base_url}/videos/{job_id}.mp4"
            
        result.video_url = video_url
        logger.debug(f"Video URL for job {job_id}: {video_url}")
    
    return result

@app.get("/video/{job_id}")
async def get_video(job_id: str = Path(..., description="The ID of the job")):
    """Get the video file for a completed job."""
    job_info = job_queue.get_job_status(job_id)
    
    if job_info is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job with ID {job_id} not found"
        )
    
    if job_info.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job with ID {job_id} is not completed yet"
        )
    
    video_path = storage_service.get_video_path(job_id)
    
    if not video_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Video file for job {job_id} not found"
        )
    
    return FileResponse(
        path=str(video_path),
        media_type="video/mp4",
        filename=f"{job_id}.mp4"
    )

# Add OPTIONS route handlers for CORS preflight requests
@app.options("/{path:path}")
async def options_route(path: str):
    """Handle OPTIONS requests for CORS preflight."""
    return {}

@app.get("/healthcheck")
async def healthcheck():
    """Healthcheck endpoint."""
    return {
        "status": "ok",
        "ngrok_url": settings.NGROK_BASE_URL or "Not configured"
    }

@app.get("/system/resources")
async def system_resources():
    """System resource utilization endpoint."""
    # Get current renderer state
    worker_stats = {
        "current_workers": renderer.current_workers,
        "max_workers": renderer.max_workers,
        "memory_per_worker_mb": settings.MIN_MEMORY_PER_WORKER_MB,
        "dynamic_scaling_enabled": settings.ENABLE_DYNAMIC_SCALING
    }
    
    # Get system resource information
    system_stats = renderer.resource_monitor.get_system_resources()
    
    # Get job queue metrics
    job_metrics = {
        "total_jobs": len(job_queue.jobs),
        "queued_jobs": sum(1 for info in job_queue.jobs.values() if info.status == JobStatus.QUEUED),
        "processing_jobs": sum(1 for info in job_queue.jobs.values() if info.status == JobStatus.PROCESSING),
        "completed_jobs": sum(1 for info in job_queue.jobs.values() if info.status == JobStatus.COMPLETED),
        "failed_jobs": sum(1 for info in job_queue.jobs.values() if info.status == JobStatus.FAILED)
    }
    
    # Calculate memory usage per process
    memory_per_process = []
    for proc in psutil.process_iter(['name', 'pid', 'memory_info']):
        try:
            if 'python' in proc.info['name'] or 'manim' in proc.info['name']:
                memory_mb = proc.info['memory_info'].rss / (1024 * 1024)
                if memory_mb > 10:  # Only include processes using >10MB
                    memory_per_process.append({
                        'pid': proc.info['pid'],
                        'name': proc.info['name'],
                        'memory_mb': round(memory_mb, 2)
                    })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    # Format the output
    return {
        "worker_configuration": worker_stats,
        "system_resources": {
            "cpu_percent": system_stats['cpu_percent'],
            "memory_percent": system_stats['memory_percent'],
            "system_load": system_stats['load_avg'],
            "total_memory_mb": round(system_stats['total_memory'] / (1024 * 1024), 2),
            "available_memory_mb": round(system_stats['available_memory'] / (1024 * 1024), 2),
        },
        "job_metrics": job_metrics,
        "top_memory_processes": sorted(memory_per_process, key=lambda x: x['memory_mb'], reverse=True)[:10]
    }

@app.get("/system/jobs/performance")
async def job_performance_stats():
    """Get performance metrics for recent jobs."""
    if not hasattr(renderer, 'job_stats') or not renderer.job_stats:
        return {"message": "No job statistics available yet"}
    
    # Process job statistics
    job_metrics = []
    for job_id, stats in renderer.job_stats.items():
        if 'end_time' in stats and 'start_time' in stats:
            total_time = stats['end_time'] - stats['start_time']
            queue_time = stats.get('dequeue_time', stats['start_time']) - stats['queue_time']
            processing_time = stats['end_time'] - stats.get('dequeue_time', stats['start_time'])
            
            job_metrics.append({
                'job_id': job_id,
                'complexity': stats['complexity'],
                'total_time_seconds': round(total_time, 2),
                'queue_time_seconds': round(queue_time, 2),
                'processing_time_seconds': round(processing_time, 2)
            })
    
    # Calculate averages by complexity
    complexity_metrics = {}
    for metric in job_metrics:
        complexity = metric['complexity']
        if complexity not in complexity_metrics:
            complexity_metrics[complexity] = {
                'count': 0,
                'total_time': 0,
                'queue_time': 0,
                'processing_time': 0
            }
        
        complexity_metrics[complexity]['count'] += 1
        complexity_metrics[complexity]['total_time'] += metric['total_time_seconds']
        complexity_metrics[complexity]['queue_time'] += metric['queue_time_seconds']
        complexity_metrics[complexity]['processing_time'] += metric['processing_time_seconds']
    
    # Calculate averages
    for complexity, data in complexity_metrics.items():
        count = data['count']
        complexity_metrics[complexity]['avg_total_time'] = round(data['total_time'] / count, 2)
        complexity_metrics[complexity]['avg_queue_time'] = round(data['queue_time'] / count, 2)
        complexity_metrics[complexity]['avg_processing_time'] = round(data['processing_time'] / count, 2)
    
    return {
        "recent_jobs": sorted(job_metrics, key=lambda x: x['total_time_seconds'], reverse=True)[:20],
        "complexity_metrics": complexity_metrics
    }

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.PORT, reload=True)