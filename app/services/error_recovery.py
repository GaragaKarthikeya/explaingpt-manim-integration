"""
Autonomous error recovery framework for Manim animation generation.

This module provides functionality to detect, analyze, and recover from errors
during Manim code generation and rendering, creating a feedback loop with the LLM.
"""

import re
import logging
import tempfile
import subprocess
from pathlib import Path
import google.generativeai as genai
from typing import Dict, List, Tuple, Optional, Any

from app.config import settings

logger = logging.getLogger(__name__)

class ErrorPatterns:
    """Common error patterns and their human-readable descriptions."""
    
    # Dictionary mapping regex patterns to human-readable error descriptions
    PATTERNS = {
        r"ValueError: invalid literal for int\(\) with base 16: '([^']+)'": 
            "Color parsing error in hex code: '{0}'",
            
        r"latex error converting to dvi": 
            "LaTeX compilation error. Check LaTeX syntax and ensure all backslashes are doubled.",
            
        r"AttributeError: 'NoneType' object has no attribute '([^']+)'": 
            "Trying to access attribute '{0}' on a None object. This often happens when an object wasn't properly created.",
            
        r"NameError: name '([^']+)' is not defined": 
            "Using variable '{0}' before defining it.",
            
        r"ValueError: list.remove\(x\): x not in list": 
            "Trying to remove an element that doesn't exist in a list.",
            
        r"IndexError: list index out of range": 
            "Accessing an index that's outside the bounds of a list.",
            
        r"TypeError: '([^']+)' object is not callable": 
            "Trying to call '{0}' which is not a function or method.",
            
        r"ZeroDivisionError": 
            "Division by zero error.",
            
        r"ImportError: No module named '([^']+)'": 
            "Trying to import non-existent module '{0}'.",
            
        r"Code\(\) takes [0-9]+ positional arguments but ([0-9]+) were given": 
            "Incorrect number of arguments supplied to Code() constructor.",
            
        r"ConnectionResetError": 
            "Connection was reset, possibly due to a crash in the renderer process when handling complex LaTeX or animations.",
            
        r"Connection reset by peer":
            "Connection was reset by peer, likely due to memory issues or process crash when handling complex animations.",
            
        r"Mobjects?.*out of frame": 
            "Elements are positioned out of the visible frame. Use smaller multipliers for position vectors.",
            
        r"Mobjects?.*too large": 
            "Elements are too large for the frame. Use smaller sizes for objects.",
            
        r"Camera.*frame.*width": 
            "Issue with camera frame width. Use default camera settings rather than custom width."
    }
    
    # Add math-specific common error patterns
    MATH_PATTERN_FIXES = {
        r"\\pi": r"\\\\pi",
        r"\\theta": r"\\\\theta", 
        r"\\alpha": r"\\\\alpha",
        r"\\beta": r"\\\\beta",
        r"\\gamma": r"\\\\gamma",
        r"\\delta": r"\\\\delta",
        r"\\epsilon": r"\\\\epsilon",
        r"\\zeta": r"\\\\zeta",
        r"\\eta": r"\\\\eta",
        r"\\sigma": r"\\\\sigma",
        r"\\omega": r"\\\\omega",
        r"\\phi": r"\\\\phi",
        r"\\frac{([^}]+)}{([^}]+)}": r"\\\\frac{\1}{\2}",
        r"\\sqrt{([^}]+)}": r"\\\\sqrt{\1}",
        r"\\sum": r"\\\\sum",
        r"\\prod": r"\\\\prod",
        r"\\int": r"\\\\int",
        r"\\mathbb{([^}]+)}": r"\\\\mathbb{\1}",
        r"\\infty": r"\\\\infty",
    }
    
    @classmethod
    def match_error(cls, error_text: str) -> List[Tuple[str, List[str]]]:
        """
        Match error text against known patterns.
        
        Args:
            error_text: The error text to analyze
            
        Returns:
            List of tuples containing (error_description, [matched_groups])
        """
        matches = []
        for pattern, description in cls.PATTERNS.items():
            for match in re.finditer(pattern, error_text, re.MULTILINE):
                groups = match.groups() if match.groups() else []
                formatted_desc = description.format(*groups)
                matches.append((formatted_desc, groups))
        
        return matches
        
    @classmethod
    def fix_math_patterns(cls, tex_string: str) -> str:
        """
        Fix common mathematical LaTeX patterns by ensuring they're properly escaped.
        
        Args:
            tex_string: The LaTeX string to fix
            
        Returns:
            Fixed LaTeX string with proper escaping
        """
        result = tex_string
        for pattern, replacement in cls.MATH_PATTERN_FIXES.items():
            # Don't replace patterns that already have double backslashes
            safe_pattern = pattern.replace(r"\\", r"(?<!\\)\\")
            result = re.sub(safe_pattern, replacement, result)
            
        return result


class CodeAnalyzer:
    """Analyzes Manim code to identify potential issues before execution."""
    
    @staticmethod
    def analyze(code: str) -> List[str]:
        """
        Analyze code for common issues.
        
        Args:
            code: The Manim code to analyze
            
        Returns:
            List of potential issues found
        """
        issues = []
        
        # Check for Code() mobject
        if "Code(" in code:
            issues.append("Using problematic Code() mobject which often causes color parsing errors")
        
        # Check for missing r prefix in Tex strings
        tex_without_r = len(re.findall(r'Tex\(\s*"', code))
        if tex_without_r > 0:
            issues.append(f"Found {tex_without_r} Tex() calls without r prefix for raw strings")
        
        # Check for single backslashes in LaTeX that should be doubled
        tex_matches = re.findall(r'Tex\(r[\'"]([^\'"]+)[\'"]', code)
        for match in tex_matches:
            if re.search(r'(?<!\\)\\(?!\\)[a-zA-Z]+', match):
                issues.append("Found LaTeX commands with single backslashes that should be doubled")
                break
        
        # Check for missing tex_template parameter in Tex objects
        if "my_template" in code and re.search(r'Tex\(.+(?<!tex_template=my_template)\)', code):
            issues.append("Found Tex() calls missing tex_template parameter")
        
        # Check for objects not being removed from screen
        if code.count("FadeIn(") + code.count("Create(") - code.count("FadeOut(") - code.count("self.clear()") > 3:
            issues.append("Objects are being added to the scene without being removed later")
        
        # Check for potential variable references before creation
        potential_vars = re.findall(r'self\.play\([^)]*([a-zA-Z_][a-zA-Z0-9_]*)', code)
        var_definitions = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', code)
        for var in potential_vars:
            if var not in var_definitions and var not in ["self", "Transform", "Create", "FadeIn", "FadeOut"]:
                issues.append(f"Potential reference to undefined variable '{var}'")
        
        # Check for dynamic layout management
        if "def manage_layout" not in code and "def arrange" not in code and "auto-scale" not in code:
            issues.append("Missing dynamic layout management function - may cause overlapping elements")
            
        # Check for VGroup usage with multiple objects
        object_creations = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?:Circle|Square|Rectangle|Triangle|Line|Arrow|Text|MathTex|Tex)\(', code)
        if len(object_creations) > 3 and "VGroup" not in code:
            issues.append("Multiple objects created without using VGroup for organization")
            
        # Check if objects are arranged with proper buffers
        if code.count("next_to(") > 0 and not re.search(r'next_to\([^,]+,[^,]+,\s*buff\s*=', code):
            issues.append("Using next_to() without specifying buffer size - will cause overlapping")
            
        # Check for boundary checks after positioning
        if len(object_creations) > 3 and not re.search(r'if\s+[^.]+\.get_(bottom|top|left|right)\(\)', code):
            issues.append("Multiple objects positioned without checking frame boundaries")
        
        # Check for scene framing and positioning issues
        if "self.camera.frame_width" in code:
            issues.append("Custom camera frame width detected - may cause elements to appear too wide or stretched")
            
        # Check for elements with excessive size
        large_size_patterns = [
            r'Circle\(radius\s*=\s*([0-9.]+)',
            r'Square\(side_length\s*=\s*([0-9.]+)',
            r'Rectangle\(width\s*=\s*([0-9.]+)',
            r'font_size\s*=\s*([0-9.]+)'
        ]
        
        for pattern in large_size_patterns:
            for match in re.finditer(pattern, code):
                try:
                    size = float(match.group(1))
                    if ('Circle' in pattern or 'Square' in pattern) and size > 2:
                        issues.append(f"Object size too large: {match.group(0)}. Use sizes between 0.5 and 2")
                    elif 'Rectangle' in pattern and size > 4:
                        issues.append(f"Rectangle width too large: {match.group(0)}. Use width between 1 and 4")
                    elif 'font_size' in pattern and size > 48:
                        issues.append(f"Font size too large: {match.group(0)}. Use font_size between 24 and 36")
                except (ValueError, IndexError):
                    pass
        
        # Check for excessive position values
        position_patterns = [
            r'move_to\([^)]*([A-Z]+)\s*\*\s*([0-9.]+)',
            r'shift\([^)]*([A-Z]+)\s*\*\s*([0-9.]+)'
        ]
        
        for pattern in position_patterns:
            for match in re.finditer(pattern, code):
                try:
                    multiplier = float(match.group(2))
                    if multiplier > 3.5:
                        issues.append(f"Position multiplier too large: {match.group(0)}. Use values between 1 and 3")
                except (ValueError, IndexError):
                    pass
        
        # Check if complex animations with multiple elements are properly organized
        if len(object_creations) > 5 and "VGroup" not in code:
            issues.append("Complex animation with multiple elements but not using VGroup for organization")
        
        return issues
        
    @staticmethod
    def suggest_layout_fixes(code: str) -> str:
        """
        Add dynamic layout management to code that's missing it.
        
        Args:
            code: The Manim code to enhance
            
        Returns:
            Code with added dynamic layout management
        """
        # If already has layout management, return as is
        if "def manage_layout" in code or "def arrange_elements" in code:
            return code
            
        # Split code into parts
        lines = code.split('\n')
        construct_start_index = -1
        
        # Find the start of the construct method
        for i, line in enumerate(lines):
            if "def construct(self)" in line:
                construct_start_index = i
                break
                
        if construct_start_index == -1:
            return code  # No construct method found
            
        # Insert the layout management function after the construct method declaration
        layout_function = [
            "        # Function to manage layout and prevent overlapping",
            "        def manage_layout(*elements, direction=RIGHT, buff=0.7):",
            "            group = VGroup(*elements)",
            "            group.arrange(direction, buff=buff)",
            "            ",
            "            # Auto-scale if elements are too large for frame",
            "            max_width = config.frame_width - 1",
            "            max_height = config.frame_height - 2",
            "            if group.width > max_width:",
            "                scale = 0.9 * max_width / group.width",
            "                group.scale(scale)",
            "            if group.height > max_height:",
            "                scale = 0.9 * max_height / group.height",
            "                group.scale(scale)",
            "            return group",
            ""
        ]
        
        # Insert the function definition
        modified_lines = lines[:construct_start_index+1] + layout_function + lines[construct_start_index+1:]
        
        return '\n'.join(modified_lines)
    
    @staticmethod
    def suggest_sizing_fixes(code: str) -> str:
        """
        Suggest fixes for sizing issues in the code.
        
        Args:
            code: The Manim code to analyze
            
        Returns:
            Code with suggested size adjustments
        """
        modified_code = code
        
        # Fix camera frame width if present
        modified_code = re.sub(
            r'self\.camera\.frame_width\s*=\s*[0-9.]+', 
            '# Default camera settings are better for standard animations', 
            modified_code
        )
        
        # Fix circle and square sizes
        modified_code = re.sub(
            r'(Circle\(radius\s*=\s*)([0-9.]+)(\))', 
            lambda m: f"{m.group(1)}{min(1.0, float(m.group(2)) / 2.5)}{m.group(3)}" if float(m.group(2)) > 2 else m.group(0),
            modified_code
        )
        
        modified_code = re.sub(
            r'(Square\(side_length\s*=\s*)([0-9.]+)(\))', 
            lambda m: f"{m.group(1)}{min(2.0, float(m.group(2)) / 2)}{m.group(3)}" if float(m.group(2)) > 3 else m.group(0),
            modified_code
        )
        
        # Fix font sizes
        modified_code = re.sub(
            r'(font_size\s*=\s*)([0-9.]+)', 
            lambda m: f"{m.group(1)}{min(36, float(m.group(2)) / 2)}" if float(m.group(2)) > 60 else m.group(0),
            modified_code
        )
        
        # Fix excessive positioning
        modified_code = re.sub(
            r'(move_to\([^)]*[A-Z]+\s*\*\s*)([0-9.]+)(\))', 
            lambda m: f"{m.group(1)}{min(2.5, float(m.group(2)) / 1.5)}{m.group(3)}" if float(m.group(2)) > 3 else m.group(0),
            modified_code
        )
        
        return modified_code


class ErrorRecovery:
    """
    Main class for autonomous error recovery during Manim animation generation.
    
    This class handles:
    1. Analyzing code for potential issues before execution
    2. Running code in a sandbox to detect errors
    3. Generating detailed feedback for the LLM
    4. Managing retry attempts with progressive strategies
    """
    
    def __init__(self):
        self.max_retries = settings.ERROR_RECOVERY_MAX_RETRIES if hasattr(settings, 'ERROR_RECOVERY_MAX_RETRIES') else 3
        self.previous_errors = []  # Track previous errors to inform future attempts
        
    def _format_error_feedback(self, error_text: str, code: str) -> str:
        """
        Format error information into a structured feedback for the LLM.
        
        Args:
            error_text: The raw error text
            code: The code that caused the error
            
        Returns:
            Formatted feedback string
        """
        matches = ErrorPatterns.match_error(error_text)
        error_desc = "Unknown error"
        
        if matches:
            # Use the first match (most specific)
            error_desc = matches[0][0]
        
        # Extract the relevant part of the error message
        error_lines = error_text.split('\n')
        relevant_error = ""
        for line in error_lines:
            if "Error:" in line or "Exception:" in line:
                relevant_error = line.strip()
                break
        
        # If no specific error line found, use the last line
        if not relevant_error and error_lines:
            relevant_error = error_lines[-1].strip()
        
        # Analyze the code for potential issues
        code_issues = CodeAnalyzer.analyze(code)
        issues_text = "\n".join([f"- {issue}" for issue in code_issues])
        
        feedback = (
            f"ERROR ANALYSIS:\n"
            f"Error type: {error_desc}\n"
            f"Error message: {relevant_error}\n\n"
            f"POTENTIAL ISSUES IN CODE:\n"
            f"{issues_text if issues_text else '- No specific issues found through static analysis'}\n\n"
            f"PREVIOUS ERRORS:\n"
            f"{self._format_previous_errors()}"
        )
        
        return feedback
    
    def _format_previous_errors(self) -> str:
        """Format the history of previous errors."""
        if not self.previous_errors:
            return "- No previous errors recorded"
            
        result = []
        for i, error in enumerate(self.previous_errors[-3:]):  # Only show the most recent 3
            result.append(f"- Attempt {i+1}: {error}")
        
        return "\n".join(result)
    
    def _sandbox_test(self, code: str) -> Tuple[bool, Optional[str]]:
        """
        Test Manim code in a sandbox environment to catch errors before full rendering.
        
        Args:
            code: The Manim code to test
            
        Returns:
            (success, error_message)
        """
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False) as tmp:
                tmp_path = tmp.name
                tmp.write(code.encode('utf-8'))
            
            # Run manim with the --dry_run flag which validates the code without rendering
            cmd = [
                "manim", 
                tmp_path, 
                "AnimationScene", 
                "--dry_run"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            stdout, stderr = process.communicate()
            Path(tmp_path).unlink()  # Clean up
            
            if process.returncode != 0:
                return False, stderr.decode('utf-8') if stderr else stdout.decode('utf-8')
                
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    async def recover(self, error_text: str, original_code: str, prompt: str, model) -> str:
        """
        Main recovery method that processes errors and requests fixed code.
        
        Args:
            error_text: The error output from Manim
            original_code: The Manim code that caused the error
            prompt: The original user prompt
            model: The Gemini model instance
            
        Returns:
            Improved code that should fix the error
        """
        # Save this error for future reference
        error_summary = error_text.split('\n')[0] if error_text else "Unknown error"
        self.previous_errors.append(error_summary)
        
        # Format detailed feedback for the LLM
        feedback = self._format_error_feedback(error_text, original_code)
        logger.info(f"Error recovery feedback generated:\n{feedback}")
        
        # Construct the recovery prompt
        recovery_prompt = self._build_recovery_prompt(feedback, original_code, prompt)
        
        # Get improved code from the LLM
        retry_count = 0
        while retry_count < self.max_retries:
            retry_count += 1
            logger.info(f"Requesting fixed code (attempt {retry_count}/{self.max_retries})")
            
            try:
                response = model.generate_content(recovery_prompt)
                improved_code = self._extract_code(response.text)
                
                # Analyze the new code
                issues = CodeAnalyzer.analyze(improved_code)
                if issues:
                    logger.warning(f"Generated code still has potential issues: {issues}")
                
                # Test in sandbox if possible
                logger.info("Running sandbox test of generated code")
                success, sandbox_error = self._sandbox_test(improved_code)
                if not success:
                    logger.warning(f"Sandbox test failed: {sandbox_error}")
                    
                    # Update feedback with new error info and retry
                    feedback = self._format_error_feedback(sandbox_error, improved_code)
                    recovery_prompt = self._build_recovery_prompt(feedback, improved_code, prompt)
                    continue
                
                # If we reach here, the code passed the sandbox test
                logger.info("Recovery successful - code passed sandbox test")
                return improved_code
                
            except Exception as e:
                logger.error(f"Error during recovery attempt: {str(e)}")
                
        # If all retries fail, return the best attempt (the last one)
        logger.warning("Maximum retry attempts reached without successful recovery")
        return improved_code if 'improved_code' in locals() else original_code
    
    def _build_recovery_prompt(self, feedback: str, original_code: str, prompt: str) -> str:
        """Build a prompt for the LLM to fix the code."""
        # Check if we have positioning/overlap issues based on feedback
        has_positioning_issues = any(issue in feedback.lower() for issue in [
            "overlap", "off screen", "out of", "position", "frame_width",
            "move_to", "shift", "next_to", "buffer", "elements"
        ])
        
        has_size_issues = any(issue in feedback.lower() for issue in [
            "too large", "too wide", "too big", "font_size", "radius", "side_length",
            "width", "height", "scale", "stretch"
        ])
        
        positioning_guidance = ""
        if has_positioning_issues:
            positioning_guidance = (
                "POSITIONING AND OVERLAP FIXES NEEDED:\n"
                "1. Remove any custom camera.frame_width settings\n"
                "2. Use absolute positioning with move_to() - examples: obj.move_to(LEFT*2), obj.move_to(RIGHT*3+UP)\n"
                "3. Always add buff=0.5 or greater to next_to() calls to prevent overlap\n" 
                "4. Use proper layout techniques like:\n"
                "   - arrange() for organizing elements in a row\n"
                "   - arrange_in_grid() for organizing elements in a grid\n"
                "   - VGroup() to group related elements and position them together\n"
                "5. Use shift() to fine-tune positions: obj.shift(UP*0.5)\n"
                "6. Remove elements when they're no longer needed\n\n"
            )
            
        size_guidance = ""
        if has_size_issues:
            size_guidance = (
                "ELEMENT SIZING FIXES NEEDED:\n"
                "1. Use smaller sizes for all shapes:\n"
                "   - Circle: radius=0.5 to 1.0 (not larger)\n" 
                "   - Square: side_length=1.0 to 1.5 (not larger)\n"
                "   - Rectangle: width=2.0 to 3.0, height=1.0 to 2.0\n"
                "2. Use appropriate text sizes:\n"
                "   - Titles: font_size=36\n"
                "   - Regular text: font_size=24\n"
                "   - Small labels: font_size=20\n"
                "3. Use moderate spacing between elements (LEFT*2, RIGHT*2)\n"
                "4. For positioning vectors, use values between 1 and 2.5\n"
                "5. Scale down any complex diagrams to ensure they fit properly\n\n"
            )
        
        # Try to apply automatic fixes to the code for sizing issues
        improved_code = original_code
        if has_size_issues:
            improved_code = CodeAnalyzer.suggest_sizing_fixes(original_code)
            if improved_code != original_code:
                original_code = improved_code
        
        return (
            f"You need to fix the following Manim animation code that generated errors.\n\n"
            f"ORIGINAL PROMPT: {prompt}\n\n"
            f"ERROR FEEDBACK:\n{feedback}\n\n"
            f"{positioning_guidance}"
            f"{size_guidance}"
            f"ORIGINAL CODE:\n```python\n{original_code}\n```\n\n"
            f"Please generate fixed code that will work correctly in Manim. "
            f"Specifically address the errors identified above. "
            f"Make sure to:\n"
            f"1. Fix all syntax and runtime errors\n"
            f"2. Use only stable Manim features\n"
            f"3. Ensure objects are properly removed from the screen when no longer needed\n"
            f"4. Double all backslashes in LaTeX expressions\n"
            f"5. Always use raw strings (r prefix) for LaTeX content\n"
            f"6. Use the custom tex_template for all LaTeX elements\n"
            f"7. Replace any Code() mobjects with Text() mobjects\n"
            f"8. Make sure all variables are defined before use\n"
            f"9. Use proper positioning and layout to avoid overlaps and elements going off-screen\n"
            f"10. Use appropriate sizing for all elements (keep shapes and text modest in size)\n\n"
            f"Respond with ONLY the fixed Python code, surrounded by ```python and ``` markers. No explanations."
        )
    
    def _extract_code(self, response_text: str) -> str:
        """Extract code from the LLM response."""
        if "```python" in response_text:
            # Extract code from markdown code blocks
            pattern = r"```python\s*([\s\S]*?)\s*```"
            matches = re.findall(pattern, response_text)
            if matches:
                return matches[0].strip()
                
        # If no code block found, use the whole text but strip markdown
        return response_text.replace("```python", "").replace("```", "").strip()