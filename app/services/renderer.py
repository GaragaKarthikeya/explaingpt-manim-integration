import os
import threading
import subprocess
import tempfile
import time
import json
import logging
import asyncio
import shutil
import multiprocessing
import psutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import google.generativeai as genai
from google.generativeai import types
from typing import Dict, Optional
import re

from app.models import JobStatus, AnimationRequest
from app.services.job_queue import job_queue
from app.services.llm_service import gemini_service, GeminiService
from app.services.error_recovery import ErrorRecovery, CodeAnalyzer  # Import the new error recovery framework
from app.config import settings

logger = logging.getLogger(__name__)

# Create a global instance of the error recovery framework
error_recovery = ErrorRecovery()

class ResourceMonitor:
    """Monitors system resource usage and recommends optimal worker count."""
    
    def __init__(self):
        self.cpu_history = []  # Store recent CPU usage percentages
        self.memory_history = []  # Store recent memory usage percentages
        self.history_size = 5  # Number of samples to keep
    
    def get_system_resources(self):
        """Get current system resource usage."""
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory_percent = psutil.virtual_memory().percent
        load_avg = os.getloadavg()[0] / os.cpu_count()  # Normalize load average
        
        # Update history
        self.cpu_history.append(cpu_percent)
        self.memory_history.append(memory_percent)
        if len(self.cpu_history) > self.history_size:
            self.cpu_history.pop(0)
        if len(self.memory_history) > self.history_size:
            self.memory_history.pop(0)
            
        # Calculate averages
        avg_cpu = sum(self.cpu_history) / len(self.cpu_history)
        avg_memory = sum(self.memory_history) / len(self.memory_history)
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'load_avg': load_avg,
            'avg_cpu': avg_cpu,
            'avg_memory': avg_memory,
            'total_memory': psutil.virtual_memory().total,
            'available_memory': psutil.virtual_memory().available
        }
    
    def calculate_optimal_workers(self, current_workers):
        """Calculate the optimal number of worker processes based on system resources."""
        resources = self.get_system_resources()
        
        # Calculate workers based on memory
        available_memory_mb = resources['available_memory'] / (1024 * 1024)
        memory_based_workers = int(available_memory_mb / settings.MIN_MEMORY_PER_WORKER_MB)
        
        # Calculate workers based on CPU load
        max_cpu_workers = max(1, multiprocessing.cpu_count() - 1)
        if resources['avg_cpu'] > 80:  # High CPU usage
            cpu_based_workers = max(1, current_workers - 1)
        elif resources['avg_cpu'] < 50:  # Low CPU usage
            cpu_based_workers = min(max_cpu_workers, current_workers + 1)
        else:
            cpu_based_workers = current_workers
            
        # Take the minimum to ensure we don't overload either CPU or memory
        optimal_workers = min(memory_based_workers, cpu_based_workers, settings.MAX_PARALLEL_RENDERINGS)
        
        # Always ensure at least 1 worker
        return max(1, optimal_workers)
    
    def log_resource_usage(self):
        """Log current resource usage."""
        resources = self.get_system_resources()
        logger.info(
            f"System resources: CPU: {resources['cpu_percent']}%, "
            f"Memory: {resources['memory_percent']}%, "
            f"Load: {resources['load_avg']:.2f}"
        )


# Enhanced worker process function with error recovery
def worker_process(job_id, request_dict, output_dir, temp_dir, result_queue):
    """Worker process to render an animation."""
    try:
        # Setup logging for the worker process
        logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=logging_format)
        worker_logger = logging.getLogger(f"worker-{job_id}")
        
        # Start monitoring process resources
        current_process = psutil.Process()
        initial_memory = current_process.memory_info().rss / (1024 * 1024)  # MB
        start_time = time.time()
        
        # Update status via result queue
        result_queue.put((job_id, JobStatus.PROCESSING, 0.1, "Preparing animation", None))
        
        # Parse the request dictionary
        request = AnimationRequest(**request_dict)
        prompt = request.prompt
        complexity = request.complexity
        worker_logger.info(f"Processing prompt: '{prompt}' with complexity level {complexity}")
        
        # Create GeminiService instance for code generation
        gemini = GeminiService()
        
        try:
            # Create an event loop for async code generation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Generate Manim code using our service
            manim_code = loop.run_until_complete(gemini.generate_manim_code(prompt, complexity))
            loop.close()
            
            if not manim_code:
                raise RuntimeError("Failed to generate valid Manim code")
            
            # Write the code to a file for rendering
            max_rendering_attempts = 1
            if settings.ERROR_RECOVERY_ENABLED:
                max_rendering_attempts = settings.ERROR_RECOVERY_MAX_RETRIES + 1
            
            for attempt in range(max_rendering_attempts):
                with tempfile.NamedTemporaryFile(suffix='.py', delete=False, dir=temp_dir) as tmp:
                    tmp_path = tmp.name
                    worker_logger.info(f"Writing Manim code to temporary file for job {job_id} (attempt {attempt+1}/{max_rendering_attempts})")
                    tmp.write(manim_code.encode('utf-8'))
                
                # Update status
                progress = 0.5 + (0.1 * attempt)
                result_queue.put((job_id, JobStatus.RENDERING, progress, 
                                f"Rendering animation with Manim (attempt {attempt+1})", None))
                
                # Create a temporary directory for Manim output
                with tempfile.TemporaryDirectory(dir=temp_dir) as temp_media_dir:
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
                    
                    worker_logger.info(f"Running Manim command: {' '.join(cmd)}")
                    
                    manim_process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    
                    stdout, stderr = manim_process.communicate()
                    
                    if manim_process.returncode != 0:
                        error_output = stderr.decode('utf-8')
                        worker_logger.error(f"Manim rendering failed (attempt {attempt+1}): {error_output}")
                        
                        # Only try error recovery if we have attempts left and it's enabled
                        if (attempt < max_rendering_attempts - 1) and settings.ERROR_RECOVERY_ENABLED:
                            worker_logger.info(f"Attempting error recovery for job {job_id}")
                            result_queue.put((job_id, JobStatus.PROCESSING, 0.6 + (0.1 * attempt), 
                                            "Attempting to fix rendering errors", None))
                            
                            try:
                                # Create an event loop for async recovery
                                loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(loop)
                                
                                # Run the recovery process to get improved code
                                manim_code = loop.run_until_complete(
                                    gemini.generate_manim_code(
                                        f"Fix this Manim code error: {error_output}\n\nOriginal prompt: {prompt}",
                                        complexity
                                    )
                                )
                                loop.close()
                                
                                # Clean up the current temporary file before the next attempt
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                                    
                                continue  # Try again with the improved code
                                
                            except Exception as recovery_error:
                                worker_logger.error(f"Error during recovery attempt: {str(recovery_error)}")
                                # Fall through to failure
                        
                        # If we're out of attempts or recovery is disabled, report failure
                        if attempt == max_rendering_attempts - 1:
                            error_msg = f"Manim rendering failed after {max_rendering_attempts} attempts: {error_output}"
                            result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))
                            # Clean up temp file
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                            return
                    else:
                        # Success! We have a rendered animation
                        # Find the generated file
                        expected_path = Path(temp_media_dir) / "videos" / "720p30" / output_filename
                        if not expected_path.exists():
                            # Look for file in different locations
                            found_files = list(Path(temp_media_dir).glob(f"**/{output_filename}"))
                            if not found_files:
                                error_msg = f"Output video file not found in {temp_media_dir}"
                                result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))
                                # Clean up temp file
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                                return
                            expected_path = found_files[0]
                        
                        # Copy to final destination
                        final_path = Path(output_dir) / output_filename
                        shutil.copy2(expected_path, final_path)
                        
                        if not final_path.exists():
                            error_msg = f"Failed to copy output file to {final_path}"
                            result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))
                            # Clean up temp file
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
                            return
                        
                        # Update status to completed
                        completion_msg = f"Animation completed successfully after {attempt+1} attempts: {output_filename}"
                        result_queue.put((job_id, JobStatus.COMPLETED, 1.0, completion_msg, None))
                        
                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                            
                        # Log resource usage statistics
                        end_time = time.time()
                        final_memory = current_process.memory_info().rss / (1024 * 1024)  # MB
                        total_memory_used = final_memory - initial_memory
                        execution_time = end_time - start_time
                        worker_logger.info(
                            f"Job {job_id} completed successfully after {attempt+1} attempts. "
                            f"Time: {execution_time:.2f}s, "
                            f"Memory: {total_memory_used:.2f}MB"
                        )
                        return  # Success, exit the function
            
        except Exception as e:
            worker_logger.error(f"Error generating code with Gemini: {str(e)}")
            error_msg = f"Error rendering animation for job {job_id}: {str(e)}"
            result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))
    except Exception as e:
        error_msg = f"Error rendering animation for job {job_id}: {str(e)}"
        result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))


class ManimRenderer:
    def __init__(self):
        self.output_dir = Path(settings.MANIM_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a temporary directory for worker process files
        self.temp_dir = tempfile.mkdtemp(prefix="manim_workers_")
        
        # Set up multiprocessing with optimal worker count
        self.max_workers = min(multiprocessing.cpu_count(), settings.MAX_PARALLEL_RENDERINGS)
        self.current_workers = self.max_workers
        self.process_pool = None
        
        # Initialize resource monitor
        self.resource_monitor = ResourceMonitor()
        
        # Set up a result processing thread
        self.result_thread = threading.Thread(target=self._process_results, daemon=True)
        self.running = False
        
        # Set up a job submission thread
        self.submit_thread = threading.Thread(target=self._submit_jobs, daemon=True)
        
        # Set up a resource monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        
        # Job statistics for performance analysis
        self.job_stats: Dict[str, dict] = {}
    
    def render_animation(self, job_id, request: AnimationRequest):
        """Add a job to the multiprocessing queue for rendering.
        
        This method is called by the JobQueue to submit jobs for rendering.
        The actual rendering happens in worker processes.
        """
        if job_queue.mp_job_queue:
            # Record initial job stats for performance tracking
            self.job_stats[job_id] = {
                'start_time': time.time(),
                'complexity': request.complexity,
                'queue_time': time.time()
            }
            
            # Add job to queue with complexity level for better scheduling
            job_queue.mp_job_queue.put((job_id, request))
            logger.info(f"Job {job_id} added to multiprocessing queue for rendering")
        else:
            raise RuntimeError("Multiprocessing queue not initialized. Call start() first.")
    
    def start(self):
        """Start the renderer worker processes and threads."""
        # Initialize the job queue's multiprocessing manager
        job_queue.start_mp_manager()
        
        # Calculate initial worker count based on available system resources
        resources = self.resource_monitor.get_system_resources()
        available_memory_mb = resources['available_memory'] / (1024 * 1024)
        memory_based_workers = int(available_memory_mb / settings.MIN_MEMORY_PER_WORKER_MB)
        self.current_workers = min(self.max_workers, memory_based_workers)
        self.current_workers = max(1, self.current_workers)  # Ensure at least 1 worker
        
        # Start the process pool with calculated worker count
        self.process_pool = ProcessPoolExecutor(max_workers=self.current_workers)
        
        # Start the threads
        self.running = True
        self.result_thread.start()
        self.submit_thread.start()
        self.monitor_thread.start()
        
        logger.info(f"Manim renderer started with {self.current_workers} worker processes "
                   f"(max: {self.max_workers}, memory per worker: {settings.MIN_MEMORY_PER_WORKER_MB}MB)")
    
    def stop(self):
        """Stop the renderer worker processes and threads."""
        self.running = False
        
        # Shutdown the process pool
        if self.process_pool:
            self.process_pool.shutdown(wait=False)
        
        # Wait for threads to finish
        if self.result_thread.is_alive():
            self.result_thread.join(timeout=5.0)
        
        if self.submit_thread.is_alive():
            self.submit_thread.join(timeout=5.0)
            
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
            
        # Stop the job queue's multiprocessing manager
        job_queue.stop_mp_manager()
        
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
        logger.info("Manim renderer stopped")
    
    def _adjust_worker_count(self, new_count):
        """Adjust the number of worker processes."""
        if new_count == self.current_workers:
            return
            
        logger.info(f"Adjusting worker count from {self.current_workers} to {new_count}")
        
        # Shutdown existing pool
        if self.process_pool:
            self.process_pool.shutdown(wait=False)
            
        # Create new pool with adjusted worker count
        self.current_workers = new_count
        self.process_pool = ProcessPoolExecutor(max_workers=self.current_workers)
        
        logger.info(f"Worker count adjusted to {self.current_workers}")
    
    def _monitor_resources(self):
        """Monitor system resources and adjust worker count if needed."""
        last_check = time.time()
        
        while self.running:
            try:
                # Check system resources periodically
                current_time = time.time()
                if current_time - last_check >= settings.MONITORING_INTERVAL_SEC:
                    last_check = current_time
                    
                    # Log current resource usage
                    self.resource_monitor.log_resource_usage()
                    
                    # Only adjust worker count if dynamic scaling is enabled
                    if settings.ENABLE_DYNAMIC_SCALING:
                        # Calculate optimal worker count
                        optimal_workers = self.resource_monitor.calculate_optimal_workers(self.current_workers)
                        
                        # Adjust worker count if necessary
                        if optimal_workers != self.current_workers:
                            self._adjust_worker_count(optimal_workers)
                
                # Sleep for a short time to reduce CPU usage of monitoring thread
                time.sleep(5)
                
            except Exception as e:
                logger.exception(f"Error in resource monitoring: {str(e)}")
                time.sleep(10)
    
    def _submit_jobs(self):
        """Submit jobs to the process pool from the queue."""
        while self.running:
            try:
                if job_queue.mp_job_queue and not job_queue.mp_job_queue.empty():
                    job_id, request = job_queue.mp_job_queue.get()
                    
                    # Update job stats with dequeue time
                    if job_id in self.job_stats:
                        self.job_stats[job_id]['dequeue_time'] = time.time()
                        queue_duration = self.job_stats[job_id]['dequeue_time'] - self.job_stats[job_id]['queue_time']
                        logger.info(f"Job {job_id} waited in queue for {queue_duration:.2f} seconds")
                    
                    # Check if we have enough memory before submitting
                    resources = self.resource_monitor.get_system_resources()
                    available_memory_mb = resources['available_memory'] / (1024 * 1024)
                    
                    if available_memory_mb < settings.MIN_MEMORY_PER_WORKER_MB:
                        logger.warning(
                            f"Low memory: {available_memory_mb:.2f}MB available, "
                            f"need {settings.MIN_MEMORY_PER_WORKER_MB}MB. "
                            f"Delaying job {job_id} submission."
                        )
                        # Put the job back in the queue and wait
                        job_queue.mp_job_queue.put((job_id, request))
                        time.sleep(10)  # Wait before trying again
                        continue
                    
                    # Convert request to dictionary for serialization
                    request_dict = request.dict()
                    
                    # Submit job to process pool
                    self.process_pool.submit(
                        worker_process,
                        job_id,
                        request_dict,
                        str(self.output_dir),
                        self.temp_dir,
                        job_queue.mp_result_queue
                    )
                    logger.info(f"Submitted job {job_id} to process pool")
                else:
                    time.sleep(1)
            except Exception as e:
                logger.exception(f"Error submitting job to process pool: {str(e)}")
                time.sleep(1)
    
    def _process_results(self):
        """Process results from worker processes."""
        while self.running:
            try:
                # Process any results from the multiprocessing jobs
                job_queue.process_results()
                
                # Check for completed jobs and update statistics
                with job_queue.lock:
                    for job_id, job_info in job_queue.jobs.items():
                        if (job_info.status in [JobStatus.COMPLETED, JobStatus.FAILED] and 
                            job_id in self.job_stats and
                            'end_time' not in self.job_stats[job_id]):
                            
                            # Record job completion statistics
                            self.job_stats[job_id]['end_time'] = time.time()
                            total_duration = self.job_stats[job_id]['end_time'] - self.job_stats[job_id]['start_time']
                            
                            logger.info(
                                f"Job {job_id} completed with status {job_info.status}. "
                                f"Total time: {total_duration:.2f} seconds. "
                                f"Complexity: {self.job_stats[job_id]['complexity']}"
                            )
                            
                            # Clean up old stats to prevent memory leak (keep only the last 100)
                            if len(self.job_stats) > 100:
                                oldest_job = min(self.job_stats, key=lambda k: self.job_stats[k].get('end_time', float('inf')))
                                self.job_stats.pop(oldest_job, None)
                
                time.sleep(0.5)
            except Exception as e:
                logger.exception(f"Error processing results: {str(e)}")
                time.sleep(1)

    # ------------------- Enhanced Asynchronous Code Generation Methods ------------------- #
    async def _assess_topic_complexity(self, topic: str) -> tuple[str, int, float]:
        """Use Gemini to dynamically evaluate topic complexity, duration, and resource needs."""
        PROMPT = (
            "Rate the conceptual complexity of this topic for a computer science animation, "
            f"suggest video duration (in seconds), and estimate resource intensity: '{topic}'\n\n"
            "Respond ONLY in this JSON format:\n"
            '{"complexity": "low/medium/high", "duration_seconds": int, "resource_intensity": float(0.1-1.0)}\n'
            "Guidelines:\n"
            "- 'low' (2-4 min): Simple topics (e.g., linear search, basic sorting)\n"
            "- 'medium' (5-8 min): Moderate topics (e.g., Dijkstra's, recursion)\n"
            "- 'high' (9-15 min): Complex topics (e.g., AVL trees, dynamic programming)\n"
            "- 'resource_intensity': Estimate from 0.1 (simple shapes) to 1.0 (complex physics simulations)"
        )

        response = await gemini_service.generate_text(PROMPT)
        try:
            assessment = json.loads(response.strip())
            return (
                assessment["complexity"], 
                assessment["duration_seconds"], 
                assessment["resource_intensity"]
            )
        except Exception as e:
            logger.error(f"Error parsing complexity assessment: {e}")
            return "medium", 300, 0.5  # Fallback default (5 min, medium resource usage)
    
    def _build_manim_prompt(self, user_prompt: str, duration_sec: int, resource_intensity: float) -> str:
        """Generate a prompt that enforces duration-aware and resource-aware animations."""
        resource_guidance = (
            "Use simple shapes and minimal animations" if resource_intensity < 0.3 else
            "Balance visual complexity with performance" if resource_intensity < 0.7 else
            "Optimize complex animations and be mindful of performance"
        )
        
        return (
            f"Generate a Manim animation explaining: '{user_prompt}'\n"
            f"- *Video length*: Exactly {duration_sec // 60} minutes ({duration_sec} sec).\n"
            "- *Key Requirements*:\n"
            "  1. Adjust animation speed to fit the duration.\n"
            "  2. Break complex steps into subtasks if needed.\n"
            "  3. Use annotations/voiceover hints (no audio code).\n"
            f"- *Performance note*: {resource_guidance}.\n"
            "- *Output*: ONLY Python code (no explanations)."
        )
    
    async def _generate_manim_code_async(self, request: AnimationRequest) -> str:
        """Dynamic duration-adjusted and resource-aware code generation."""
        # Step 1: Let Gemini judge topic complexity and suggest duration and resource needs
        complexity, duration_sec, resource_intensity = await self._assess_topic_complexity(request.prompt)
        
        # Step 2: Build a prompt with duration and resource constraints
        modified_prompt = self._build_manim_prompt(request.prompt, duration_sec, resource_intensity)
        
        # Step 3: Generate Manim code using Gemini with the updated prompt and complexity value
        return await gemini_service.generate_manim_code(modified_prompt, complexity)

# Create a singleton instance of the renderer
renderer = ManimRenderer()
