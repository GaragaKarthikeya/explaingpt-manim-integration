import os
import re
import logging
import base64
import google.generativeai as genai
from app.config import settings

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
        if any(word in topic.lower() for word in ['theorem', 'proof', 'pythagoras', 'geometric']):
            return """Example of a mathematical theorem animation:
from manim import *

class AnimationScene(Scene):
    def construct(self):
        # Create title
        title = Text("Mathematical Theorem", font_size=40)
        self.play(Write(title))
        self.play(title.animate.to_edge(UP))
        
        # Create geometric objects
        square = Square(side_length=2)
        circle = Circle(radius=1)
        
        # Position and animate
        square.next_to(title, DOWN, buff=1)
        circle.next_to(square, RIGHT, buff=0.5)
        
        self.play(Create(square))
        self.play(Create(circle))
        self.wait()
        
        # Add labels and formulas
        formula = MathTex("A = \\pi r^2")
        formula.next_to(circle, DOWN)
        self.play(Write(formula))
        self.wait(2)"""
        else:
            return None

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
            
        # Check for required elements
        required_elements = [
            lambda c: "from manim import" in c or "import manim" in c,  # Import statement
            lambda c: "class AnimationScene" in c,  # Scene class
            lambda c: "def construct" in c,  # Construct method
            lambda c: "self.play" in c or "self.wait" in c  # At least one animation
        ]
        
        for check in required_elements:
            if not check(code):
                return False
                
        return True

    async def generate_manim_code(self, prompt: str, complexity) -> str:
        """Generate Manim code from a prompt using Gemini."""
        if not self.client:
            error_msg = "Gemini client not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        # Map complexity
        if isinstance(complexity, str):
            complexity_val = {"low": 1, "medium": 2, "high": 3}.get(complexity.lower(), 2)
        else:
            complexity_val = int(complexity)

        # Get a relevant example for complex topics
        example = self._get_example_for_topic(prompt)
        
        # Build a very explicit prompt
        code_prompt = f"""ROLE: You are a Manim Python code generator specializing in mathematical animations.
TASK: Create a clear, step-by-step Manim animation that explains and visualizes: "{prompt}"
Complexity level: {complexity_val}/3

TECHNICAL REQUIREMENTS:
1. Start with exactly "from manim import *"
2. Create a class named AnimationScene that inherits from Scene
3. Implement the construct method
4. Use appropriate mathematical objects (MathTex, Text, geometric shapes)
5. Break down complex concepts into clear visual steps
6. Include descriptive comments for each major step
7. Add appropriate wait times between animations
8. Follow Manim best practices

MATHEMATICAL VISUALIZATION GUIDELINES:
- For theorems: Show step-by-step construction and proof
- For geometric concepts: Build shapes gradually and show relationships
- For equations: Write them step by step with clear transitions
- Use different colors to highlight important elements
- Add labels to clarify components
- Use transformations to show relationships
- Include minimal but clear text explanations as Text/MathTex

RESPOND WITH VALID PYTHON CODE ONLY. START DIRECTLY WITH "from manim import *"
DO NOT INCLUDE ANY EXPLANATIONS BEFORE OR AFTER THE CODE.
DO NOT USE MARKDOWN CODE BLOCKS."""

        # If we have a relevant example, add it to the prompt
        if example:
            code_prompt += f"\n\nHere's a similar example to follow (adapt this structure):\n{example}"
        
        # Track attempts
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
DO NOT INCLUDE ANY EXPLANATORY TEXT.

{code_prompt}"""
                else:
                    retry_prompt = code_prompt
                
                # Generate content
                response = await self.client.generate_content_async(retry_prompt)
                
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

# Create singleton instance
gemini_service = GeminiService()