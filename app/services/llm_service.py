import os
import re
import time
import asyncio
import logging
import base64
import google.generativeai as genai
from app.config import settings
import faiss
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        try:
            # Configure the API key globally first
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Then create the model without passing the API key parameter
            self.client = genai.GenerativeModel(
                model_name=settings.GEMINI_MODEL,
                generation_config={
                    "temperature": 0.1,  # Very low temperature for deterministic code generation
                    "top_p": 0.9,
                    "top_k": 30,
                    "max_output_tokens": 2048,
                },
            )
            logger.info(f"Gemini service initialized with model: {settings.GEMINI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            self.client = None

    def _get_example_for_topic(self, topic):
        """Get a relevant example based on the topic to help guide the model."""
        if any(word in topic.lower() for word in ['vector', 'vectors', 'arrow', 'arrows', 'direction']):
            return """Example of a vector animation with dynamic layout management:
from manim import *

class AnimationScene(Scene):
    def construct(self):
        # Create title
        title = Text("Understanding Vectors", font_size=36)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP, buff=0.5))
        self.wait(0.5)
        
        # Define a coordinate system
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"include_tip": True, "numbers_to_exclude": [0]},
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        coord_system = VGroup(axes, axes_labels)
        
        # Create and display the coordinate system
        self.play(Create(coord_system))
        self.wait(0.5)
        
        # Create a vector
        vector_1 = Arrow(axes.coords_to_point(0, 0), axes.coords_to_point(2, 1), 
                        buff=0, color=RED, stroke_width=4)
        vector_1_label = MathTex(r"\\vec{v}", font_size=24).next_to(vector_1.get_center(), UP+RIGHT, buff=0.1)
        vector_group_1 = VGroup(vector_1, vector_1_label)
        
        # Show the vector
        self.play(GrowArrow(vector_1))
        self.play(Write(vector_1_label))
        self.wait(1)
        
        # Create a second vector
        vector_2 = Arrow(axes.coords_to_point(0, 0), axes.coords_to_point(1, 2), 
                         buff=0, color=BLUE, stroke_width=4)
        vector_2_label = MathTex(r"\\vec{w}", font_size=24).next_to(vector_2.get_center(), UP+LEFT, buff=0.1)
        vector_group_2 = VGroup(vector_2, vector_2_label)
        
        # Show the second vector
        self.play(GrowArrow(vector_2))
        self.play(Write(vector_2_label))
        self.wait(1)
        
        # Create an explanation of vector addition
        explanation = VGroup(
            Text("Vector Addition", font_size=28),
            MathTex(r"\\vec{v} + \\vec{w}", font_size=24),
        ).arrange(DOWN, buff=0.3)
        explanation.to_edge(RIGHT, buff=0.5)
        
        self.play(Write(explanation))
        self.wait(0.5)
        
        # Show vector addition
        result_vector = Arrow(
            axes.coords_to_point(0, 0), 
            axes.coords_to_point(3, 3), 
            buff=0, color=GREEN, stroke_width=4
        )
        result_label = MathTex(r"\\vec{v} + \\vec{w}", font_size=24).next_to(result_vector.get_center(), DOWN, buff=0.2)
        result_group = VGroup(result_vector, result_label)
        
        # Create parallel vector for demonstration
        parallel_v = Arrow(
            axes.coords_to_point(0, 0),
            axes.coords_to_point(2, 1),
            buff=0, color=RED, stroke_width=4, stroke_opacity=0.5
        )
        parallel_w = Arrow(
            axes.coords_to_point(2, 1),
            axes.coords_to_point(3, 3),
            buff=0, color=BLUE, stroke_width=4, stroke_opacity=0.5
        )
        
        # Animate the addition
        self.play(
            ReplacementTransform(vector_1.copy(), parallel_v),
            ReplacementTransform(vector_2.copy(), parallel_w)
        )
        self.wait(0.5)
        self.play(GrowArrow(result_vector))
        self.play(Write(result_label))
        self.wait(1)
        
        # Create an explanation of vector scaling
        scale_explanation = VGroup(
            Text("Vector Scaling", font_size=28),
            MathTex(r"2 \cdot \\vec{v}", font_size=24),
        ).arrange(DOWN, buff=0.3)
        scale_explanation.next_to(explanation, DOWN, buff=1)
        
        self.play(Write(scale_explanation))
        self.wait(0.5)
        
        # Show vector scaling
        scaled_vector = Arrow(
            axes.coords_to_point(0, 0),
            axes.coords_to_point(4, 2),
            buff=0, color=YELLOW, stroke_width=4
        )
        scaled_label = MathTex(r"2 \cdot \\vec{v}", font_size=24).next_to(scaled_vector.get_center(), DOWN, buff=0.2)
        scaled_group = VGroup(scaled_vector, scaled_label)
        
        self.play(GrowArrow(scaled_vector))
        self.play(Write(scaled_label))
        self.wait(1)
        
        # Clean up the scene
        self.play(
            FadeOut(parallel_v),
            FadeOut(parallel_w),
            FadeOut(result_group),
            FadeOut(scaled_group)
        )
        self.wait(0.5)
        
        # Define unit vectors
        unit_i = Arrow(
            axes.coords_to_point(0, 0),
            axes.coords_to_point(1, 0),
            buff=0, color=RED, stroke_width=4
        )
        unit_i_label = MathTex(r"\\hat{i}", font_size=24).next_to(unit_i, DOWN, buff=0.2)
        
        unit_j = Arrow(
            axes.coords_to_point(0, 0),
            axes.coords_to_point(0, 1),
            buff=0, color=GREEN, stroke_width=4
        )
        unit_j_label = MathTex(r"\\hat{j}", font_size=24).next_to(unit_j, LEFT, buff=0.2)
        
        unit_vectors = VGroup(unit_i, unit_i_label, unit_j, unit_j_label)
        
        # Show the unit vectors
        self.play(
            FadeOut(vector_group_1),
            FadeOut(vector_group_2),
            FadeOut(explanation),
            FadeOut(scale_explanation)
        )
        
        unit_explanation = Text("Unit Vectors", font_size=28).to_edge(RIGHT, buff=0.5)
        self.play(Write(unit_explanation))
        
        self.play(GrowArrow(unit_i))
        self.play(Write(unit_i_label))
        self.play(GrowArrow(unit_j))
        self.play(Write(unit_j_label))
        self.wait(1)
        
        # Show how to represent vectors with unit vectors
        vector_decomposition = MathTex(
            r"\\vec{v} = 2\\hat{i} + 1\\hat{j}",
            font_size=28
        ).next_to(unit_explanation, DOWN, buff=0.5)
        
        self.play(Write(vector_decomposition))
        self.wait(2)
        
        # Final animation - fade everything out
        self.play(
            FadeOut(unit_vectors),
            FadeOut(coord_system),
            FadeOut(unit_explanation),
            FadeOut(vector_decomposition),
            FadeOut(title)
        )
        self.wait(1)"""
        elif any(word in topic.lower() for word in ['theorem', 'proof', 'pythagoras', 'geometric']):
            return """Example of a mathematical theorem animation with dynamic layout management:
from manim import *

class AnimationScene(Scene):
    def construct(self):
        # Create title
        title = Text("Mathematical Theorem", font_size=32)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP, buff=0.5))
        
        # Function to dynamically position elements and avoid overlap
        def arrange_elements(*elements, direction=RIGHT, buff=0.5):
            # Create a VGroup to manage layout
            group = VGroup(*elements)
            group.arrange(direction, buff=buff)
            
            # Check if any elements are outside the frame and adjust if needed
            frame_width = config.frame_width - 1  # Leave some margin
            frame_height = config.frame_height - 1
            
            if group.width > frame_width:
                scale_factor = frame_width / group.width
                group.scale(scale_factor * 0.9)  # Scale with some margin
            
            return group
            
        # Create geometric objects with appropriate sizes
        square = Square(side_length=1.5)
        circle = Circle(radius=0.75)
        
        # Use our dynamic arrangement function
        shapes = arrange_elements(square, circle, buff=1)
        shapes.center()
        
        # Add elements sequentially instead of all at once
        self.play(Create(square))
        self.play(Create(circle))
        self.wait(0.5)
        
        # Add labels with proper positioning relative to the objects
        square_label = Text("Square", font_size=20)
        square_label.next_to(square, DOWN, buff=0.5)
        
        circle_label = Text("Circle", font_size=20)
        circle_label.next_to(circle, DOWN, buff=0.5)
        
        # Add formula with careful positioning
        formula = MathTex("A = \\pi r^2", font_size=24)
        formula.next_to(circle, UP, buff=0.5)
        
        # Add labels sequentially
        self.play(Write(square_label), Write(circle_label))
        self.play(Write(formula))
        
        # Organize complex layouts with VGroup
        explanation = VGroup(
            Text("Circle Area Formula", font_size=22),
            MathTex("A = \\pi \\cdot r^2", font_size=22),
            Text("Where r is the radius", font_size=18)
        ).arrange(DOWN, buff=0.3)
        
        explanation.next_to(shapes, DOWN, buff=1)
        
        # Check if explanation fits in frame, if not adjust
        if explanation.get_bottom()[1] < -config.frame_height/2 + 0.5:
            explanation.next_to(shapes, RIGHT, buff=1)
            
        self.play(FadeIn(explanation))
        
        # Always clean up elements when done
        self.wait(2)
        self.play(
            FadeOut(square),
            FadeOut(circle),
            FadeOut(square_label),
            FadeOut(circle_label),
            FadeOut(formula),
            FadeOut(explanation),
            FadeOut(title)
        )"""
        else:
            return """Example of animation with dynamic layout management:
from manim import *

class AnimationScene(Scene):
    def construct(self):
        # Create title with modest size
        title = Text("Dynamic Layout Demo", font_size=32)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP, buff=0.5))
        
        # Function to arrange elements with auto-scaling if needed
        def arrange_with_scaling(*elements, direction=RIGHT, buff=0.7):
            group = VGroup(*elements)
            group.arrange(direction, buff=buff)
            
            # Auto-scale if too large for frame
            max_width = config.frame_width - 1  # Leave margin
            max_height = config.frame_height - 2  # Leave more margin for height
            
            if group.width > max_width:
                scale = 0.9 * max_width / group.width
                group.scale(scale)
                
            if group.height > max_height:
                scale = 0.9 * max_height / group.height
                group.scale(scale)
                
            return group
        
        # Create elements with modest sizes
        elements = []
        for i in range(3):
            shape = Square(side_length=1) if i % 2 == 0 else Circle(radius=0.5)
            shape.set_fill(color=[BLUE, RED, GREEN][i], opacity=0.5)
            label = Text(f"Element {i+1}", font_size=20)
            label.next_to(shape, DOWN, buff=0.3)
            elements.append(VGroup(shape, label))
        
        # Arrange horizontally with auto-scaling
        row = arrange_with_scaling(*elements)
        row.center()
        
        # Add elements sequentially
        for element in elements:
            self.play(FadeIn(element))
        self.wait()
        
        # Create another row of elements
        more_elements = []
        for i in range(2):
            rect = Rectangle(width=1.5, height=0.8)
            text = Text(f"Info {i+1}", font_size=18)
            text.next_to(rect, DOWN, buff=0.3)
            more_elements.append(VGroup(rect, text))
        
        # Position second row below first row
        second_row = arrange_with_scaling(*more_elements)
        second_row.next_to(row, DOWN, buff=1)
        
        # Add second row
        self.play(FadeIn(second_row))
        self.wait()
        
        # Clean up when done
        self.play(
            FadeOut(row),
            FadeOut(second_row),
            FadeOut(title)
        )"""

    def _extract_python_code(self, text):
        """Extract Python code from the response text, handling various formats."""
        # Clean the text first
        text = text.strip()
        
        # If already starts with Python code, return as is
        if text.startswith("from manim import") or text.startswith("import manim"):
            return text
            
        # Try to extract code from markdown code blocks
        pattern = r"```(?:python|py|manim)?(?:\n|\r\n)([\s\S]*?)```"
        matches = re.findall(pattern, text)
        if matches:
            extracted = matches[0].strip()
            if extracted.startswith("from manim import") or extracted.startswith("import manim"):
                return extracted
        
        # Try to find code starting anywhere in the text
        if "from manim import" in text:
            index = text.find("from manim import")
            return text[index:].strip()
        elif "import manim" in text:
            index = text.find("import manim")
            return text[index:].strip()
            
        # No valid code found
        logger.warning(f"Failed to extract valid Python code from response: {text[:100]}...")
        return text  # Return original text for error reporting

    def _is_valid_manim_code(self, code):
        """Validate if the given code is proper Manim Python code."""
        if not code:
            return False
            
        # Check if starts with explanatory text
        explanatory_starts = [
            "ok", "okay", "alright", "here", "so", "now", "let", "i'll", "for", "to",
            "first", "this", "the", "we", "in", "pi", "what"
        ]
        
        first_word = code.strip().lower().split(" ")[0].strip()
        if first_word in explanatory_starts:
            logger.warning(f"Code starts with explanatory text: '{first_word}'")
            return False
            
        # Check for required elements - simple structural validation
        required_elements = [
            lambda c: "from manim import" in c or "import manim" in c,  # Import statement
            lambda c: "class AnimationScene" in c,  # Scene class
            lambda c: "def construct" in c,  # Construct method
            lambda c: "self.play" in c or "self.wait" in c  # At least one animation
        ]
        
        for check in required_elements:
            if not check(code):
                return False
                
        # Only perform the most basic syntax check - actual errors will be handled during rendering
        try:
            # Check for syntax errors by compiling the code
            compile(code, '<string>', 'exec')
            return True
        except SyntaxError as e:
            logger.warning(f"Syntax error in generated code: {str(e)}")
            return False

    def _run_async_in_sync(self, coro):
        """Helper method to run async code in sync context."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
            loop.close()
            return result
        except Exception as e:
            logger.error(f"Error running async code in sync context: {str(e)}")
            raise

    async def generate_manim_code(self, prompt: str, complexity) -> str:
        """Generate Manim code from a prompt using Gemini (async version)."""
        return self.generate_manim_code_sync(prompt, complexity)

    def generate_manim_code_sync(self, prompt: str, complexity) -> str:
        """Generate Manim code from a prompt using Gemini (synchronous version)."""
        if not self.client:
            error_msg = "Gemini client not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Map complexity
        if isinstance(complexity, str):
            complexity_val = {"low": 1, "medium": 2, "high": 3}.get(complexity.lower(), 2)
        else:
            complexity_val = int(complexity)

        # Step 1: Get relevant examples using RAG
        retrieved_context = ""
        examples_found = False
        
        if settings.RAG_ENABLED and rag_service:
            try:
                # Run RAG search with retry
                for attempt in range(2):
                    try:
                        similar_examples = self._run_async_in_sync(
                            rag_service.search(prompt, k=settings.RAG_MAX_EXAMPLES)
                        )
                        
                        # Filter and format examples
                        examples = [
                            (example, score) for example, score in similar_examples
                            if score >= settings.RAG_MIN_SCORE
                        ]
                        
                        if examples:
                            retrieved_context = "\n\n".join([
                                f"Relevant example {i+1} (similarity: {score:.2f}):\n{example}" 
                                for i, (example, score) in enumerate(examples)
                            ])
                            examples_found = True
                            logger.info(f"Found {len(examples)} relevant examples using RAG")
                            break
                        
                    except Exception as e:
                        if attempt == 0:
                            logger.warning(f"RAG attempt {attempt + 1} failed: {str(e)}, retrying...")
                            time.sleep(1)  # Brief pause before retry
                        else:
                            raise  # Re-raise on final attempt
                            
            except Exception as e:
                logger.error(f"RAG search failed: {str(e)}")
                if settings.ERROR_RECOVERY_ENABLED:
                    logger.info("RAG failed, falling back to template examples")
        
        # Get template example only if no RAG examples found
        template_example = None
        if not examples_found:
            template_example = self._get_example_for_topic(prompt)
            logger.info("Using template example as fallback")
        
        # Build the prompt using available examples
        code_prompt = f"""ROLE: You are a Manim Python code generator specializing in mathematical animations.
TASK: Create a clear, step-by-step Manim animation that explains and visualizes: "{prompt}"
Complexity level: {complexity_val}/3

TECHNICAL REQUIREMENTS:
// ...existing code...

DYNAMIC LAYOUT MANAGEMENT (CRITICAL):
// ...existing code...

ELEMENT SIZING AND POSITIONING:
// ...existing code...

SYNTAX REQUIREMENTS (CRITICAL):
// ...existing code...

COMMON SYNTAX ERRORS TO AVOID:
// ...existing code...

MANIM API REQUIREMENTS:
// ...existing code...

MATHEMATICAL VISUALIZATION GUIDELINES:
// ...existing code...

RESPOND WITH VALID PYTHON CODE ONLY. START DIRECTLY WITH "from manim import *"
DO NOT INCLUDE ANY EXPLANATIONS BEFORE OR AFTER THE CODE.
DO NOT USE MARKDOWN CODE BLOCKS."""

        # Add retrieved examples if available
        if retrieved_context:
            code_prompt += f"\n\nHere are some relevant examples from similar animations:\n{retrieved_context}"

        # Add template example if available and no good RAG examples found
        if template_example and not retrieved_context:
            code_prompt += f"\n\nHere's a similar example to follow (adapt this structure):\n{template_example}"

        # Rest of the method remains the same
        max_attempts = 3
        current_attempt = 0
        last_error = None
        
        while current_attempt < max_attempts:
            current_attempt += 1
            try:
                logger.info(f"Generating Manim code, attempt {current_attempt}/{max_attempts}")
                
                # Add error details on retry attempts
                if current_attempt > 1 and last_error:
                    retry_prompt = f"""PREVIOUS ATTEMPT FAILED: {last_error}
IMPORTANT: YOUR RESPONSE MUST BE VALID PYTHON CODE ONLY.
START EXACTLY WITH: "from manim import *"
DO NOT INCLUDE ANY EXPLANATORY TEXT OR CODE EXAMPLES.
ENSURE ALL PARENTHESES AND STRINGS ARE PROPERLY CLOSED.
NEVER LEAVE INCOMPLETE METHOD CALLS OR PARAMETERS.

CRITICAL SYNTAX ISSUES TO FIX:
- Check all parentheses balance correctly: count opening "(" and closing ")" - they must match
- Each method call must have its own complete set of parentheses
- Always use semicolons or new lines between separate statements
- Never place commas between separate statements
- If using helper methods like save_all_objects(), call them in separate statements
- Keep each function call on its own line for clarity

SCENE FRAMING ISSUES TO FIX:
- Use default camera settings (don't set camera.frame_width)
- Use smaller sizes for all objects (radius=0.5-1, side_length=1-2)
- Scale down text with font_size=24-36 instead of larger values
- Position elements with moderate spacing (LEFT*2.5, RIGHT*2.5)
- Use smaller shift values (UP*1.5 instead of UP*3)
- Add buff=0.5 to all next_to() calls
- Use VGroup for organizing related elements
- Remove objects when no longer needed with FadeOut() or self.remove()

MANIM API ERRORS TO FIX:
- Don't use font_size with next_to() - it's not a valid parameter
- Ensure vector operations use Manim methods (get_length, get_angle) not numpy methods
- Only use UP, DOWN, LEFT, RIGHT constants with Brace direction
- Don't try to manipulate numpy arrays directly - use Manim object methods

{code_prompt}"""
                else:
                    retry_prompt = code_prompt
                
                # Generate content using the synchronous API
                response = self.client.generate_content(retry_prompt)
                
                # Extract Python code from the response
                manim_code = self._extract_python_code(response.text)
                
                # Validate the generated code
                if self._is_valid_manim_code(manim_code):
                    logger.info(f"Successfully generated valid Manim code on attempt {current_attempt}")
                    return manim_code
                
                # Code validation failed, prepare for retry
                last_error = "Generated code was not valid Manim code or contained explanatory text"
                logger.warning(f"Code validation failed on attempt {current_attempt}: {last_error}")
                
            except Exception as e:
                last_error = str(e)
                logger.exception(f"Error on attempt {current_attempt}: {last_error}")
        
        # All attempts failed
        error_msg = f"Failed to generate valid Manim code after {max_attempts} attempts. Last error: {last_error}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    # Simple async text generation method for non-code use cases
    async def generate_text(self, prompt: str) -> str:
        """Generate text response from a prompt."""
        try:
            if not self.client:
                error_msg = "Gemini client not initialized"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
                
            # Use the synchronous API for simplicity
            response = self.client.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.exception(f"Error generating text response: {str(e)}")
            raise RuntimeError(f"Failed to generate text: {str(e)}")

class RAGService:
    def __init__(self, index_path: str, blocks_path: str, embedding_model: str = "models/embedding-001"):
        self.index_path = index_path
        self.blocks_path = blocks_path
        self.embedding_model = embedding_model
        self.index = None
        self.blocks = None
        self.embeddings = genai.GenerativeModel(embedding_model)
        self.load_index()
    
    def load_index(self):
        """Load the FAISS index and code blocks."""
        if not Path(self.index_path).exists() or not Path(self.blocks_path).exists():
            raise FileNotFoundError("FAISS index or blocks file not found")
            
        self.index = faiss.read_index(self.index_path)
        self.blocks = np.load(self.blocks_path, allow_pickle=True)
        
    async def search(self, query: str, k: int = 5) -> list[tuple[str, float]]:
        """Search for relevant code examples."""
        # Get query embedding using Gemini embeddings API
        query_embedding = await self._get_embedding(query)
        
        # Search the FAISS index
        scores, indices = self.index.search(query_embedding.reshape(1, -1), k)
        
        # Return results with scores
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx != -1:  # FAISS returns -1 for empty results
                # Convert cosine similarity to a 0-1 score
                normalized_score = (score + 1) / 2
                if normalized_score >= settings.RAG_MIN_SCORE:
                    results.append((self.blocks[idx], normalized_score))
                
        return sorted(results, key=lambda x: x[1], reverse=True)[:settings.RAG_MAX_EXAMPLES]
        
    async def _get_embedding(self, text: str, max_retries: int = 2) -> np.ndarray:
        """Get embedding for text using Gemini embeddings API with retries."""
        last_error = None
        for attempt in range(max_retries):
            try:
                # Get embeddings from Gemini
                embedding = await self.embeddings.generate_embeddings(text)
                embedding_array = np.array(embedding.values, dtype=np.float32)
                
                # Basic validation
                if embedding_array.size == 0:
                    raise ValueError("Received empty embedding array")
                if np.isnan(embedding_array).any():
                    raise ValueError("Embedding contains NaN values")
                    
                return embedding_array
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Embedding generation failed (attempt {attempt + 1}/{max_retries}): {last_error}")
                if attempt < max_retries - 1:
                    # Wait with exponential backoff before retrying
                    await asyncio.sleep(2 ** attempt)
                    continue
                
        logger.error(f"All embedding generation attempts failed: {last_error}")
        raise RuntimeError(f"Failed to generate embeddings after {max_retries} attempts: {last_error}")

# Initialize RAG service with settings
if settings.RAG_ENABLED:
    try:
        rag_service = RAGService(
            index_path=settings.RAG_INDEX_PATH,
            blocks_path=settings.RAG_BLOCKS_PATH
        )
        logger.info(f"RAG service initialized with index from {settings.RAG_INDEX_PATH}")
    except FileNotFoundError as e:
        logger.warning(f"Could not initialize RAG service: {str(e)}")
        rag_service = None
else:
    rag_service = None

# Create singleton instance 
gemini_service = GeminiService()