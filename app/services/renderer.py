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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import google.generativeai as genai

from app.models import JobStatus, AnimationRequest
from app.services.job_queue import job_queue
from app.services.llm_service import gemini_service
from app.config import settings

logger = logging.getLogger(__name__)

# Define worker process function outside of any class (to be picklable)
def worker_process(job_id, request_dict, output_dir, temp_dir, result_queue):
    """Worker process to render an animation."""
    try:
        # Setup logging for the worker process
        logging_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=logging_format)
        worker_logger = logging.getLogger(f"worker-{job_id}")
        
        # Update status via result queue
        result_queue.put((job_id, JobStatus.PROCESSING, 0.1, "Preparing animation", None))
        
        # Create a temp Python file with the Manim code
        worker_logger.info(f"Generating code for job {job_id}")
        result_queue.put((job_id, JobStatus.RENDERING, 0.3, "Generating animation script with Gemini", None))
        
        # Parse the request dictionary
        request = AnimationRequest(**request_dict)
        prompt = request.prompt
        complexity = request.complexity
        worker_logger.info(f"Processing prompt: '{prompt}' with complexity level {complexity}")
        
        # Use the Gemini API directly within the worker process
        try:
            # Configure Gemini API
            genai.configure(api_key=settings.GEMINI_API_KEY)
            model_name = settings.GEMINI_MODEL
            if not model_name.startswith("models/"):
                model_name = "models/" + model_name
                
            # Craft the prompt for Gemini
            system_prompt = (
                "You are an expert in Manim, the Mathematical Animation Engine. \n"
                "Convert the following animation description into valid, executable Manim Python code.\n\n"
                "Guidelines:\n"
                "1. Always include 'from manim import *' at the top\n"
                "2. Create a class called 'AnimationScene' that inherits from Scene\n"
                "3. Implement the 'construct' method\n"
                "4. Use appropriate Manim objects and animations\n"
                "5. The code should be complete and runnable\n"
                "6. Handle complexity appropriately (1=simple, 2=moderate, 3=complex)\n"
                "7. Include descriptive comments\n"
                "8. Make sure the animation is visually appealing\n"
                "9. Add appropriate wait times between animations\n"
                "10. Ensure the code is error-free and follows Manim best practices\n\n"
                "IMPORTANT: Return ONLY the Python code itself, do NOT wrap it in Markdown code blocks or backticks."
            )
            
            user_prompt = (
                f"{system_prompt}\n\n"
                f"Create a Manim animation for: \"{prompt}\"\n"
                f"Complexity level: {complexity}/3"
            )
            
            # Generate code using Gemini
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(user_prompt)
            
            # Clean up the response
            manim_code = response.text
            
            # Remove any markdown code block formatting
            if "```" in manim_code:
                import re
                pattern = r"```(?:python|py|manim)?(?:\n|\r\n)([\s\S]*?)```"
                matches = re.findall(pattern, manim_code)
                if matches:
                    manim_code = matches[0].strip()
                else:
                    manim_code = manim_code.replace("```python", "").replace("```py", "").replace("```manim", "").replace("```", "").strip()
            
            # Validate the generated code
            if "from manim import" not in manim_code:
                worker_logger.warning("Generated code missing imports, adding them")
                manim_code = f"from manim import *\n\n{manim_code}"
                
            if "class AnimationScene" not in manim_code:
                worker_logger.warning("Generated code missing AnimationScene class, adding structure")
                manim_code = (
                    f"from manim import *\n\n"
                    f"class AnimationScene(Scene):\n"
                    f"    def construct(self):\n"
                    f"        # Generated for: {prompt}\n"
                    f"        {manim_code}"
                )
                
            worker_logger.info(f"Successfully generated unique Manim code for prompt: '{prompt}'")
            
        except Exception as e:
            worker_logger.error(f"Error generating code with Gemini: {str(e)}")
            # Use fallback code with the prompt incorporated to at least show something unique
            manim_code = f"""from manim import *

class AnimationScene(Scene):
    def construct(self):
        # Animation for: {prompt}
        title = Text("{prompt}", font_size=36)
        self.play(Write(title))
        self.wait(1)
        self.play(title.animate.scale(0.5).to_edge(UP))
        
        # Create shapes based on complexity
        if {complexity} == 1:
            shape = Circle(color=BLUE)
            self.play(Create(shape))
            self.wait(1)
            self.play(shape.animate.set_fill(BLUE, opacity=0.5))
        elif {complexity} == 2:
            shapes = VGroup(
                Circle(color=BLUE),
                Square(color=RED),
                Triangle(color=GREEN)
            ).arrange(RIGHT)
            self.play(Create(shapes), run_time=2)
            self.wait(1)
            self.play(shapes.animate.arrange(DOWN), run_time=2)
        else:
            axes = Axes(x_range=[-3, 3], y_range=[-3, 3])
            self.play(Create(axes))
            graph = axes.plot(lambda x: x**2, color=YELLOW)
            self.play(Create(graph))
            dot = Dot().move_to(axes.c2p(2, 4))
            self.play(FadeIn(dot))
            
        self.wait(1)
        end_text = Text("Thank you!", font_size=48)
        self.play(FadeOut(title))
        self.play(Write(end_text))
        self.wait(1)
"""
        
        worker_logger.info(f"Writing Manim code to temporary file for job {job_id}")
        with tempfile.NamedTemporaryFile(suffix='.py', delete=False, dir=temp_dir) as tmp:
            tmp_path = tmp.name
            tmp.write(manim_code.encode('utf-8'))
                
        # Update status
        result_queue.put((job_id, JobStatus.RENDERING, 0.5, "Rendering animation with Manim", None))
        
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
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = process.communicate()
            
            if process.returncode != 0:
                worker_logger.error(f"Manim rendering failed: {stderr.decode('utf-8')}")
                error_msg = f"Manim rendering failed: {stderr.decode('utf-8')}"
                result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))
                return
            
            # Find the generated file
            expected_path = Path(temp_media_dir) / "videos" / "720p30" / output_filename
            if not expected_path.exists():
                # Look for file in different locations
                found_files = list(Path(temp_media_dir).glob(f"**/{output_filename}"))
                if not found_files:
                    error_msg = f"Output video file not found in {temp_media_dir}"
                    result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))
                    return
                expected_path = found_files[0]
            
            # Copy to final destination
            final_path = Path(output_dir) / output_filename
            shutil.copy2(expected_path, final_path)
            
            if not final_path.exists():
                error_msg = f"Failed to copy output file to {final_path}"
                result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))
                return
            
            # Update status to completed
            completion_msg = f"Animation completed: {output_filename}"
            result_queue.put((job_id, JobStatus.COMPLETED, 1.0, completion_msg, None))
        
        # Cleanup temp file
        os.unlink(tmp_path)
        worker_logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        error_msg = f"Error rendering animation for job {job_id}: {str(e)}"
        result_queue.put((job_id, JobStatus.FAILED, None, None, error_msg))


class ManimRenderer:
    def __init__(self):
        self.output_dir = Path(settings.MANIM_OUTPUT_DIR)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create a temporary directory for worker process files
        self.temp_dir = tempfile.mkdtemp(prefix="manim_workers_")
        
        # Set up multiprocessing
        self.max_workers = min(multiprocessing.cpu_count(), settings.MAX_PARALLEL_RENDERINGS)
        self.process_pool = None
        
        # Set up a result processing thread
        self.result_thread = threading.Thread(target=self._process_results, daemon=True)
        self.running = False
        
        # Set up a job submission thread
        self.submit_thread = threading.Thread(target=self._submit_jobs, daemon=True)
    
    def render_animation(self, job_id, request: AnimationRequest):
        """Add a job to the multiprocessing queue for rendering.
        
        This method is called by the JobQueue to submit jobs for rendering.
        The actual rendering happens in worker processes.
        """
        if job_queue.mp_job_queue:
            job_queue.mp_job_queue.put((job_id, request))
            logger.info(f"Job {job_id} added to multiprocessing queue for rendering")
        else:
            raise RuntimeError("Multiprocessing queue not initialized. Call start() first.")
    
    def start(self):
        """Start the renderer worker processes and threads."""
        # Initialize the job queue's multiprocessing manager
        job_queue.start_mp_manager()
        
        # Start the process pool
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_workers)
        
        # Start the threads
        self.running = True
        self.result_thread.start()
        self.submit_thread.start()
        
        logger.info(f"Manim renderer started with {self.max_workers} worker processes")
    
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
            
        # Stop the job queue's multiprocessing manager
        job_queue.stop_mp_manager()
        
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
        logger.info("Manim renderer stopped")
    
    def _submit_jobs(self):
        """Submit jobs to the process pool from the queue."""
        while self.running:
            try:
                if job_queue.mp_job_queue and not job_queue.mp_job_queue.empty():
                    job_id, request = job_queue.mp_job_queue.get()
                    
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
                time.sleep(0.5)
            except Exception as e:
                logger.exception(f"Error processing results: {str(e)}")
                time.sleep(1)

    # ------------------- New Asynchronous Code Generation Methods ------------------- #
    async def _assess_topic_complexity(self, topic: str) -> tuple[str, int]:
        """Use Gemini to dynamically evaluate topic difficulty and suggest duration."""
        PROMPT = (
            "Rate the conceptual complexity of this topic for a computer science animation "
            f"and suggest video duration (in seconds): '{topic}'\n\n"
            "Respond ONLY in this JSON format:\n"
            '{"complexity": "low/medium/high", "duration_seconds": int}\n'
            "Guidelines:\n"
            "- 'low' (2-4 min): Simple topics (e.g., linear search, basic sorting)\n"
            "- 'medium' (5-8 min): Moderate topics (e.g., Dijkstra's, recursion)\n"
            "- 'high' (9-15 min): Complex topics (e.g., AVL trees, dynamic programming)"
        )

        response = await gemini_service.generate_text(PROMPT)
        try:
            assessment = json.loads(response.strip())
            return assessment["complexity"], assessment["duration_seconds"]
        except Exception as e:
            logger.error(f"Error parsing complexity assessment: {e}")
            return "medium", 300  # Fallback default (5 min)
    
    def _build_manim_prompt(self, user_prompt: str, duration_sec: int) -> str:
        """Generate a prompt that enforces duration-aware animations."""
        return (
            f"Generate a Manim animation explaining: '{user_prompt}'\n"
            f"- *Video length*: Exactly {duration_sec // 60} minutes ({duration_sec} sec).\n"
            "- *Key Requirements*:\n"
            "  1. Adjust animation speed to fit the duration.\n"
            "  2. Break complex steps into subtasks if needed.\n"
            "  3. Use annotations/voiceover hints (no audio code).\n"
            "- *Output*: ONLY Python code (no explanations)."
        )
    
    async def _generate_manim_code_async(self, request: AnimationRequest) -> str:
        """Dynamic duration-adjusted code generation."""
        # Step 1: Let Gemini judge topic complexity and suggest duration
        complexity, duration_sec = await self._assess_topic_complexity(request.prompt)
        
        # Step 2: Build a prompt with duration constraints
        modified_prompt = self._build_manim_prompt(request.prompt, duration_sec)
        
        # Step 3: Generate Manim code using Gemini with the updated prompt and complexity value
        return await gemini_service.generate_manim_code(modified_prompt, complexity)

# Create a singleton instance of the renderer
renderer = ManimRenderer()
