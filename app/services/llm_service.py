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
                
        # Try to detect common syntax errors
        try:
            # Check for unclosed parentheses or syntax errors
            compile(code, '<string>', 'exec')
            
            # Additional checks for common issues:
            
            # Check for unterminated triple quotes
            triple_quotes = ['"""', "'''"]
            for quote in triple_quotes:
                if code.count(quote) % 2 != 0:
                    logger.warning(f"Found unterminated triple quote: {quote}")
                    return False
            
            # Check for lines ending with incomplete methods/properties
            lines = code.split('\n')
            for i, line in enumerate(lines):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Check for lines ending with dots without completing the reference
                    if line.endswith('.'):
                        logger.warning(f"Line {i+1} ends with a dot without completing reference")
                        return False
                    
                    # Check for incomplete method chains
                    incomplete_ends = ['.animate', '.next_to', '.set_', '.get_', '.to_', 'buff=']
                    for end in incomplete_ends:
                        if line.endswith(end):
                            logger.warning(f"Line {i+1} ends with incomplete method: {end}")
                            return False
            
            # Check for common Manim API usage errors
            common_manim_errors = [
                # Check for font_size in next_to (common error seen in logs)
                r'\.next_to\([^,)]*,[^,)]*,\s*[^,)]*font_size\s*=',
                # Check for using numpy array methods on vector objects
                r'vector_arrow\.get_\w+\(\)\.\w+',
                # Check for incorrect brace usage
                r'Brace\([^,)]*,\s*direction\s*=\s*[^UP|DOWN|LEFT|RIGHT][^,)]*\)',
                # Check for common wrong parameter names - FIXING THIS ONE
                r'(\.\w+\()([^)]*)(\bcolors\s*=|\bcolour\s*=)([^)]*)(\))',
                # Check for incorrect vector handling
                r'([^a-zA-Z0-9_])(rotate|translate|normalize|orthogonalize)(\s*\()',
            ]
            
            for i, pattern in enumerate(common_manim_errors):
                match = re.search(pattern, code)
                if match:
                    # Log what specifically matched to help with debugging
                    logger.warning(f"Found common Manim API usage error with pattern {i}: {match.group(0)}")
                    return False
                
            return True
            
        except SyntaxError as e:
            logger.warning(f"Syntax error in generated code: {str(e)}")
            return False

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

DYNAMIC LAYOUT MANAGEMENT (CRITICAL):
1. Create a helper function at the beginning of your construct method to manage layout:
   ```python
   def manage_layout(*elements, direction=RIGHT, buff=0.7):
       group = VGroup(*elements)
       group.arrange(direction, buff=buff)
       
       # Auto-scale if elements are too large for frame
       max_width = config.frame_width - 1
       max_height = config.frame_height - 2
       if group.width > max_width:
           scale = 0.9 * max_width / group.width
           group.scale(scale)
       if group.height > max_height:
           scale = 0.9 * max_height / group.height
           group.scale(scale)
       return group
   ```
2. Use VGroup for all related elements and arrange them with proper buffers
3. Check frame boundaries before finalizing positions: 
   ```python
   if some_group.get_bottom()[1] < -3.5:  # Too low
       some_group.shift(UP * (abs(some_group.get_bottom()[1]) - 3))
   ```
4. Position complex element groups using the helper function:
   ```python
   shapes = manage_layout(square, circle, triangle, buff=0.8)
   shapes.center()
   ```
5. For multiple rows of content, create them separately and position them relative to each other:
   ```python
   row1 = manage_layout(elem1, elem2, elem3)
   row2 = manage_layout(elem4, elem5)
   row2.next_to(row1, DOWN, buff=1)
   ```

ELEMENT SIZING AND POSITIONING:
1. Use modest sizes for elements (radius=0.5-1, side_length=1-2, font_size=24-36)
2. For text associated with shapes, group them together:
   ```python
   square = Square(side_length=1.5)
   square_label = Text("Square", font_size=24)
   square_label.next_to(square, DOWN, buff=0.5)
   square_group = VGroup(square, square_label)
   ```
3. Add elements sequentially, not all at once
4. Always clean up elements when they're no longer needed

SYNTAX REQUIREMENTS (IMPORTANT):
1. Make sure ALL parentheses, brackets, and braces are properly closed
2. Ensure triple-quoted strings ('''...''' or \"""...\""") are properly closed
3. Complete all method calls - never leave a line ending with a dot or a method name
4. Always complete parameters on the same line or use proper line continuation with backslash
5. When a line exceeds 80 characters, break it before operators with proper indentation
6. Never include examples, corrected_code sections, or any non-executable code

MANIM API REQUIREMENTS:
1. For Brace objects, only use UP, DOWN, LEFT, RIGHT for direction parameter
2. Never use font_size parameter with next_to() method
3. Don't try to access numpy array attributes directly - use get_center(), get_end(), etc.
4. For vector operations, use proper methods: get_length(), get_angle(), not numpy methods

MATHEMATICAL VISUALIZATION GUIDELINES:
- For theorems: Show step-by-step construction and proof
- For geometric concepts: Build shapes gradually and show relationships
- For equations: Write them step by step with clear transitions
- Use different colors to highlight important elements
- Add labels to clarify components
- Use transformations to show relationships

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
DO NOT INCLUDE ANY EXPLANATORY TEXT OR CODE EXAMPLES.
ENSURE ALL PARENTHESES AND STRINGS ARE PROPERLY CLOSED.
NEVER LEAVE INCOMPLETE METHOD CALLS OR PARAMETERS.

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

# Create singleton instance
gemini_service = GeminiService()