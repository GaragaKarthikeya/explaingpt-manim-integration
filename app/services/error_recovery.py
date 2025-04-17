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
            "Connection was reset by peer, likely due to memory issues or process crash when handling complex animations."
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
        
        return issues


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
        return (
            f"You need to fix the following Manim animation code that generated errors.\n\n"
            f"ORIGINAL PROMPT: {prompt}\n\n"
            f"ERROR FEEDBACK:\n{feedback}\n\n"
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
            f"8. Make sure all variables are defined before use\n\n"
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