import os
import google.generativeai as genai
import logging
import re
from app.config import settings

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        try:
            # Configure the generative AI module with your API key.
            genai.configure(api_key=settings.GEMINI_API_KEY)
            
            # Check if the model name already has the "models/" prefix
            if settings.GEMINI_MODEL.startswith("models/"):
                self.model_name = settings.GEMINI_MODEL
            else:
                self.model_name = "models/" + settings.GEMINI_MODEL
                
            logger.info(f"Gemini service initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {str(e)}")
            self.model_name = None

    def _clean_markdown_code_blocks(self, text):
        """Remove markdown code block formatting from text."""
        # Pattern to match code blocks with language specifier
        pattern = r"```(?:python|py|manim)?(?:\n|\r\n)([\s\S]*?)```"
        
        # Find all code blocks
        matches = re.findall(pattern, text)
        
        if matches:
            # Use the content of the first code block
            return matches[0].strip()
        
        # If no code blocks found, just remove any triple backticks
        cleaned = re.sub(r"```.*\n", "", text)
        cleaned = re.sub(r"```", "", cleaned)
        
        return cleaned.strip()

    async def generate_manim_code(self, prompt: str, complexity: int) -> str:
        """Generate Manim code from a prompt using Gemini."""
        if not self.model_name:
            logger.error("Gemini client not initialized")
            return self._get_fallback_code(prompt, complexity)
        
        try:
            # Craft a specific prompt for Gemini
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
            
            # Create a generative model object
            model = genai.GenerativeModel(self.model_name)
            
            # Use the generative model to generate content
            response = model.generate_content(user_prompt)
            
            # Extract the generated code from the response and clean markdown
            raw_text = response.text
            manim_code = self._clean_markdown_code_blocks(raw_text)
            
            # Validate the generated code
            if "from manim import" not in manim_code:
                logger.warning("Generated code missing imports, adding them")
                manim_code = f"from manim import *\n\n{manim_code}"
            
            if "class AnimationScene" not in manim_code:
                logger.warning("Generated code missing AnimationScene class, adding structure")
                manim_code = (
                    f"from manim import *\n\n"
                    f"class AnimationScene(Scene):\n"
                    f"    def construct(self):\n"
                    f"        # Generated for: {prompt}\n"
                    f"        {manim_code}"
                )
            
            logger.info(f"Successfully generated Manim code for prompt: '{prompt}'")
            return manim_code
            
        except Exception as e:
            logger.exception(f"Error generating Manim code with Gemini: {str(e)}")
            return self._get_fallback_code(prompt, complexity)
    
    def _get_fallback_code(self, prompt: str, complexity: int) -> str:
        """Provide fallback code when Gemini fails."""
        logger.warning(f"Using fallback code for prompt: '{prompt}'")
        
        # Basic template based on complexity
        complexity = min(3, max(1, complexity))
        
        if complexity == 1:
            animation_code = """
        circle = Circle()
        self.play(Create(circle))
        self.wait(1)
        circle.set_fill(BLUE, opacity=0.5)
        self.play(circle.animate.shift(RIGHT * 2))
        self.wait(1)
"""
        elif complexity == 2:
            animation_code = """
        circle = Circle()
        square = Square()
        
        self.play(Create(circle))
        self.wait(0.5)
        self.play(Transform(circle, square))
        self.wait(0.5)
        self.play(square.animate.set_fill(YELLOW, opacity=0.5))
        self.wait(1)
"""
        else:  # complexity == 3
            animation_code = """
        # Create a coordinate system
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"include_tip": True}
        )
        
        # Add labels
        x_label = axes.get_x_axis_label("x")
        y_label = axes.get_y_axis_label("y")
        
        self.play(Create(axes), Create(x_label), Create(y_label))
        
        # Create and plot a function
        func = lambda x: np.sin(x)
        graph = axes.plot(func, color=BLUE)
        graph_label = axes.get_graph_label(graph, "\\sin(x)", x_val=2)
        
        self.play(Create(graph), Create(graph_label))
        self.wait(1)
        
        # Show an area under the curve
        area = axes.get_area(graph, x_range=[0, 2], color=YELLOW, opacity=0.4)
        self.play(FadeIn(area))
        self.wait(1)
"""
        
        manim_code = f"""from manim import *

class AnimationScene(Scene):
    def construct(self):
        # Fallback animation for prompt: {prompt}
        # Complexity level: {complexity}
        
        title = Text("{prompt}")
        self.play(Write(title))
        self.wait(1)
        self.play(FadeOut(title))
        
{animation_code}
        
        self.wait(2)
"""
        return manim_code

# Create singleton instance
gemini_service = GeminiService()