import os
import threading
import subprocess
import tempfile
import time
from pathlib import Path
from app.models import JobStatus, AnimationRequest
from app.services.job_queue import job_queue
from app.services.llm_service import gemini_service
from app.config import settings
import logging
import asyncio
import shutil

logger = logging.getLogger(__name__)

class ManimRenderer:
    def __init__(self):
        self.output_dir = Path(settings.MANIM_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.running = False
    
    def start(self):
        """Start the renderer worker thread."""
        self.running = True
        self.worker_thread.start()
        logger.info("Manim renderer worker started")
    
    def stop(self):
        """Stop the renderer worker thread."""
        self.running = False
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5.0)
        logger.info("Manim renderer worker stopped")
    
    def _process_queue(self):
        """Process jobs from the queue."""
        while self.running:
            job_id = job_queue.get_next_job()
            
            if not job_id:
                time.sleep(1)
                continue
            
            try:
                job_request = job_queue.get_job_request(job_id)
                if not job_request:
                    continue
                
                # Update status to processing
                job_queue.update_job_status(
                    job_id, 
                    JobStatus.PROCESSING,
                    progress=0.1,
                    message="Preparing animation"
                )
                
                # Process the job
                self._render_animation(job_id, job_request)
                
            except Exception as e:
                logger.exception(f"Error processing job {job_id}: {str(e)}")
                job_queue.update_job_status(
                    job_id,
                    JobStatus.FAILED,
                    error=str(e)
                )
    
    async def _generate_manim_code_async(self, request: AnimationRequest) -> str:
        """Generate Manim Python code based on the prompt using Gemini."""
        return await gemini_service.generate_manim_code(request.prompt, request.complexity)
    
    def _render_animation(self, job_id: str, request: AnimationRequest):
        """Render an animation using Manim."""
        try:
            # Create a temp Python file with the Manim code
            job_queue.update_job_status(
                job_id,
                JobStatus.RENDERING,
                progress=0.3,
                message="Generating animation script with Gemini"
            )
            
            # Use Gemini to generate Manim code
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                manim_code = loop.run_until_complete(self._generate_manim_code_async(request))
            finally:
                loop.close()
            
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(manim_code.encode('utf-8'))
                
            # Update status
            job_queue.update_job_status(
                job_id,
                JobStatus.RENDERING,
                progress=0.5,
                message="Rendering animation with Manim"
            )
            
            # Create a temporary directory for Manim output
            with tempfile.TemporaryDirectory() as temp_media_dir:
                # Run Manim to generate the animation
                output_filename = f"{job_id}.mp4"
                cmd = [
                    "manim", 
                    tmp_path, 
                    "AnimationScene", 
                    "-q", "h", 
                    "-o", output_filename,
                    "--media_dir", temp_media_dir
                ]
                
                logger.info(f"Running Manim command: {' '.join(cmd)}")
                
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    logger.error(f"Manim rendering failed: {stderr.decode('utf-8')}")
                    raise RuntimeError(f"Manim rendering failed: {stderr.decode('utf-8')}")
                
                # Find the generated file - Manim typically puts it in a videos subdirectory
                # within the media directory
                expected_path = Path(temp_media_dir) / "videos" / "720p30" / output_filename
                if not expected_path.exists():
                    # Look for file in different locations
                    found_files = list(Path(temp_media_dir).glob(f"**/{output_filename}"))
                    if not found_files:
                        raise FileNotFoundError(f"Output video file not found in {temp_media_dir}")
                    expected_path = found_files[0]
                
                # Copy to final destination
                final_path = self.output_dir / output_filename
                shutil.copy2(expected_path, final_path)
                
                if not final_path.exists():
                    raise FileNotFoundError(f"Failed to copy output file to {final_path}")
                
                # Update status to completed
                video_url = f"{settings.NGROK_BASE_URL}/videos/{output_filename}"
                
                job_queue.update_job_status(
                    job_id,
                    JobStatus.COMPLETED,
                    progress=1.0,
                    message=f"Animation completed: {video_url}"
                )
            
            # Cleanup temp file
            os.unlink(tmp_path)
            
        except Exception as e:
            logger.exception(f"Error rendering animation for job {job_id}: {str(e)}")
            job_queue.update_job_status(
                job_id,
                JobStatus.FAILED,
                error=str(e)
            )

# Create a singleton instance
renderer = ManimRenderer()