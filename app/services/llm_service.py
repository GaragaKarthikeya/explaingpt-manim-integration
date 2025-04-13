import os
import google.generativeai as genai
import logging
import re
import json
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
        pattern = r"```(?:python|py|manim)?(?:\n|\r\n)([\s\S]*?)```"
        matches = re.findall(pattern, text)
        if matches:
            return matches[0].strip()
        cleaned = re.sub(r"```.*\n", "", text)
        cleaned = re.sub(r"```", "", cleaned)
        return cleaned.strip()

    async def generate_manim_code(self, prompt: str, complexity) -> str:
        """
        Generate Manim code from a prompt using Gemini.
        The complexity parameter can be an integer (1=simple, 2=moderate, 3=complex) or
        a string ("low", "medium", "high"), which will be internally mapped to the corresponding integer.
        """
        if not self.model_name:
            error_msg = "Gemini client not initialized"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        try:
            # Map complexity from string to integer if needed
            if isinstance(complexity, str):
                mapping = {"low": 1, "medium": 2, "high": 3}
                complexity_val = mapping.get(complexity.lower(), 2)
            else:
                complexity_val = int(complexity)
            
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
                f"Complexity level: {complexity_val}/3"
            )
            
            # Create the generative model object
            model = genai.GenerativeModel(self.model_name)
            
            # Generate the content using Gemini
            response = model.generate_content(user_prompt)
            raw_text = response.text
            manim_code = self._clean_markdown_code_blocks(raw_text)
            
            # Validate that the generated code includes necessary elements.
            if "from manim import" not in manim_code:
                logger.warning("Generated code is missing 'from manim import' - please verify Gemini output.")
            if "class AnimationScene" not in manim_code:
                logger.warning("Generated code is missing 'AnimationScene' class - please verify Gemini output.")
            
            logger.info(f"Successfully generated Manim code for prompt: '{prompt}'")
            return manim_code
            
        except Exception as e:
            logger.exception(f"Error generating Manim code with Gemini: {str(e)}")
            raise RuntimeError(f"Gemini failed to generate Manim code: {e}")

# Create singleton instance
gemini_service = GeminiService()
