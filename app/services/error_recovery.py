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
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

# Import Manim constants for positioning
from manim import RIGHT, UP, LEFT, DOWN, ORIGIN, config, VGroup, Mobject

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
            "Issue with camera frame width. Use default camera settings rather than custom width.",
            
        r"AttributeError: 'Camera' object has no attribute 'frame'":
            "Using camera.frame which is not available in standard Camera. Use self.camera.frame_width or adjust camera configuration.",
            
        r"SyntaxError: unmatched [\(\)]":
            "Syntax error: There's a missing or extra parenthesis in the code.",
            
        r"SyntaxError: [\(\)]":
            "Syntax error related to parentheses or brackets. Check for mismatched pairs.",
            
        r"SyntaxError: invalid syntax\.\s+Perhaps you forgot a comma\?":
            "Syntax error: Missing comma in a function call or sequence.",
            
        r"SyntaxError: unterminated string literal":
            "Syntax error: Unclosed string - add a matching quote to close the string.",
            
        r"SyntaxError: EOF while scanning":
            "Syntax error: Reached the end of file while parsing a structure - check for unclosed brackets, parentheses, or strings.",
            
        r"Error: Command failed:.*manim.*":
            "Manim command execution failed. This may indicate a problem in the rendering environment.",
            
        r"Code\(code_string=.*\):":
            "Error in Code object with code_string parameter. Check for unterminated strings or syntax errors in the code string."
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
    """
    Analyzes and fixes issues in Manim code, particularly focused on preventing
    blank screens and improving animation quality.
    """
    
    @staticmethod
    def analyze(code: str) -> List[str]:
        """
        Analyzes Manim code for potential issues that might cause rendering problems.
        
        Args:
            code: The Manim code to analyze
            
        Returns:
            A list of potential issues found in the code
        """
        issues = []
        
        # Check for basic structural issues
        if "class AnimationScene(Scene)" not in code:
            issues.append("Missing AnimationScene class")
        
        if "def construct(self)" not in code:
            issues.append("Missing construct method")
        
        # Check for potential positioning issues
        if "next_to" in code and "buff=" not in code:
            issues.append("Using next_to without buffer parameter may cause overlapping")
        
        if "set_z_index" not in code and code.count("add(") > 5:
            issues.append("Multiple objects without z-index management may cause visibility issues")
        
        # Check for layout issues
        if "VGroup" not in code and code.count("add(") > 5:
            issues.append("Multiple objects without VGroup organization may cause layout issues")
        
        # Check for timing issues that could cause blank screens
        if "wait" not in code:
            issues.append("No wait() calls may cause animation to be too fast")
        
        if "play" not in code:
            issues.append("No play() calls suggest no animation will occur")
        
        # Check for blank screen issues where there are large gaps with no content
        wait_calls = re.findall(r'self\.wait\(([^)]*)\)', code)
        if any(call and float(call.strip()) > 2.0 for call in wait_calls if call.strip().replace('.', '', 1).isdigit()):
            issues.append("Long wait times (>2s) may result in blank screens")
        
        return issues
    
    @staticmethod
    def fix_blank_screen_issues(code: str) -> str:
        """
        Fixes common issues that cause blank screens in Manim animations by ensuring
        there is always visual content and smooth transitions between animations.
        
        Args:
            code: The Manim code to fix
            
        Returns:
            Fixed Manim code with reduced likelihood of blank screens
        """
        logger.info("Fixing blank screen issues in Manim code")
        
        # Fix 1: Ensure there's always a title or something visible at the beginning
        if "class AnimationScene(Scene)" in code and "def construct(self)" in code:
            # Check if the animation starts with something visible
            construct_body = re.search(r'def construct\(self\):(.*?)(?:def|\Z)', code, re.DOTALL)
            if construct_body:
                body = construct_body.group(1)
                # If the first action isn't creating and showing something, add a title
                first_actions = body.strip().split('\n')[:5]  # Look at first few lines
                has_initial_content = any('self.add(' in line or 'play(' in line for line in first_actions)
                
                if not has_initial_content:
                    # Insert initial title before other code
                    title_code = """
        # Ensure there's content at the start to prevent blank screens
        title = Text("Animation", font_size=72)
        self.play(Write(title))
        self.wait(0.5)
        self.play(title.animate.scale(0.8).to_edge(UP))
"""
                    construct_pattern = r'(def\s+construct\s*\(\s*self\s*\)\s*:)'
                    code = re.sub(construct_pattern, f'\\1{title_code}', code)
        
        # Fix 2: Replace long waits with shorter ones
        code = re.sub(r'self\.wait\(([3-9]|[1-9][0-9]+)(\.[0-9]+)?\)', r'self.wait(1.5)', code)
        
        # Fix 3: Ensure continuous visual interest by adding transitions between major sections
        wait_pattern = r'self\.wait\(([^)]*)\)(?!\s*#.*transition)'  # Waits not marked as transitions
        code = re.sub(wait_pattern, lambda m: 
                   f'self.wait({m.group(1)})\n        '
                   f'# Add subtle camera movement to maintain visual interest\n        '
                   f'self.camera.frame.save_state()\n        '
                   f'self.play(self.camera.frame.animate.scale(1.05).shift(RIGHT*0.05), run_time=2)', code)
        
        # Fix 4: Ensure objects remain visible throughout the animation
        # Add an improved save_all_objects function to keep track of all created objects
        if "save_all_objects" not in code:
            save_objects_code = """
    def save_all_objects(self, *objects):
        # Keep track of all objects to prevent them from disappearing
        if not hasattr(self, 'all_objects'):
            self.all_objects = []
        
        # Process each object, handling both animation objects and mobjects
        for obj in objects:
            # Extract the actual mobject from animation objects if needed
            if hasattr(obj, 'mobject'):
                actual_obj = obj.mobject
            else:
                actual_obj = obj
                
            if actual_obj not in self.all_objects:
                self.all_objects.append(actual_obj)
"""
            class_pattern = r'(class\s+AnimationScene\s*\(\s*Scene\s*\)\s*:)'
            code = re.sub(class_pattern, f'\\1{save_objects_code}', code)
        
        # Fix 5: Modify play calls to save objects - improved version that avoids syntax errors
        # Instead of trying to extract parameters, we'll call with individual objects
        create_pattern = r'self\.play\((Create|Write|DrawBorderThenFill|FadeIn)\(([a-zA-Z0-9_]+)\)(.*?)\)'
        
        def save_obj_replacement(match):
            anim_type = match.group(1)
            obj_name = match.group(2)
            rest = match.group(3)
            return f"self.play({anim_type}({obj_name}){rest})\n        self.save_all_objects({obj_name})"
        
        code = re.sub(create_pattern, save_obj_replacement, code)
        
        # Fix 6: Ensure smooth animation transitions with rate_functions
        if "rate_functions" not in code:
            code = re.sub(r'from manim import \*', 
                        'from manim import *\nfrom manim.utils.rate_functions import smooth, ease_in_out_sine',
                        code)
        
        # Add rate_functions to animations where they're missing
        code = re.sub(r'\.animate\.', '.animate.set_rate_function(ease_in_out_sine).', code)
        
        # Fix 7: Add a dynamic scene management approach for complex scenes
        if "def manage_scene_complexity" not in code and code.count("self.play") > 5:
            manage_scene_code = """
    def manage_scene_complexity(self):
        # Dynamically manage scene complexity to prevent blank screens
        # and maintain visual interest throughout the animation
        if hasattr(self, 'all_objects') and len(self.all_objects) > 8:
            # If we have too many objects, create a smooth transition
            # by scaling some objects down or moving them to the side
            background_objects = self.all_objects[:-5]  # Keep the 5 most recent objects prominent
            
            animations = []
            for i, obj in enumerate(background_objects):
                if obj in self.mobjects:
                    if i % 2 == 0:
                        animations.append(obj.animate.scale(0.7).to_edge(LEFT, buff=0.5))
                    else:
                        animations.append(obj.animate.scale(0.7).to_edge(RIGHT, buff=0.5))
            
            if animations:
                self.play(*animations, run_time=1.5)
                self.wait(0.3)
"""
            # Add the management function to the class
            class_pattern = r'(class\s+AnimationScene\s*\(\s*Scene\s*\)\s*:)'
            code = re.sub(class_pattern, f'\\1{manage_scene_code}', code)
            
            # Call the management function periodically
            sections = code.split('self.play')
            if len(sections) > 5:
                # Insert management calls after every 4-5 animations
                new_code = sections[0]
                for i, section in enumerate(sections[1:], 1):
                    new_code += 'self.play' + section
                    if i % 4 == 0:
                        new_code += '\n        self.manage_scene_complexity()\n        '
                code = new_code
        
        logger.info("Applied comprehensive fixes to prevent blank screens in animation")
        return code
    
    @staticmethod
    def suggest_sizing_fixes(code: str) -> str:
        """
        Adds appropriate sizing and scaling to objects to prevent them from being too large
        or too small, which can lead to visual issues.
        
        Args:
            code: The Manim code to fix
            
        Returns:
            Fixed Manim code with better sizing
        """
        # Add font_size parameters where text is created without size
        code = re.sub(r'Text\(([\'"](.*?)[\'"])\)', r'Text(\1, font_size=36)', code)
        code = re.sub(r'MathTex\(([\'"](.*?)[\'"])\)', r'MathTex(\1, font_size=36)', code)
        
        # Fix excessive scaling that might push objects off-screen
        code = re.sub(r'\.scale\(([3-9]|[1-9][0-9]+)(\.[0-9]+)?\)', r'.scale(2.0)', code)
        
        return code
    
    @staticmethod
    def suggest_layout_fixes(code: str) -> str:
        """
        Adds layout management code to help organize objects on screen.
        
        Args:
            code: The Manim code to fix
            
        Returns:
            Fixed Manim code with better layout management
        """
        # Add a utility method for arranging elements with proper scaling
        arrange_method = """
    def arrange_with_scaling(self, *objects, direction=DOWN, buff=0.5, center=True):
        # Create a group with all objects
        group = VGroup(*objects)
        
        # Check if the group is too wide or tall
        width = group.width
        height = group.height
        
        # Scale down if necessary
        if width > config.frame_width * 0.8:
            scale_factor = config.frame_width * 0.8 / width
            group.scale(scale_factor)
        
        if height > config.frame_height * 0.8:
            scale_factor = config.frame_height * 0.8 / height
            group.scale(scale_factor)
        
        # Arrange the objects
        group.arrange(direction, buff=buff)
        
        # Center if requested
        if center:
            group.move_to(ORIGIN)
        
        return group
"""
        
        # Add the method to the class if it doesn't exist
        if "def arrange_with_scaling" not in code:
            class_pattern = r'(class\s+AnimationScene\s*\(\s*Scene\s*\)\s*:)'
            code = re.sub(class_pattern, f'\\1{arrange_method}', code)
        
        return code
    
    @staticmethod
    def fix_positioning_issues(code: str) -> str:
        """
        Fixes positioning issues that could cause objects to overlap or go off-screen.
        """
        # Add very strict size limits for all objects
        size_patterns = [
            (r'Circle\(radius=([0-9.]+)\)', lambda m: f'Circle(radius={min(float(m.group(1)), 0.5)})', 0.5),
            (r'Square\(side_length=([0-9.]+)\)', lambda m: f'Square(side_length={min(float(m.group(1)), 1.0)})', 1.0),
            (r'Rectangle\(width=([0-9.]+),\s*height=([0-9.]+)\)', 
             lambda m: f'Rectangle(width={min(float(m.group(1)), 2.0)}, height={min(float(m.group(2)), 1.0)})', None),
            (r'Arrow\([^)]*\)', lambda m: m.group(0).replace(')', ', buff=0.3, stroke_width=3)'), None),
            (r'Text\([^)]*\)', lambda m: m.group(0).replace(')', ', font_size=24)'), None),
            (r'Tex\([^)]*\)', lambda m: m.group(0).replace(')', ', font_size=24)'), None),
            (r'MathTex\([^)]*\)', lambda m: m.group(0).replace(')', ', font_size=24)'), None),
        ]
        
        for pattern, replacement, max_val in size_patterns:
            if max_val:  # If we have a max value, use regex with capture group for numeric validation
                matches = re.finditer(pattern, code)
                for match in matches:
                    try:
                        val = float(match.group(1))
                        if val > max_val:
                            code = code.replace(match.group(0), replacement(match))
                    except:
                        continue
            else:  # Simple string replacement for patterns without numeric validation
                code = re.sub(pattern, replacement, code)
        
        # Add scale reductions for all objects
        if "scale(" not in code:
            # Add after object creation
            lines = code.split('\n')
            new_lines = []
            for line in lines:
                new_lines.append(line)
                if any(create_pattern in line for create_pattern in ['Circle(', 'Square(', 'Rectangle(', 'Text(', 'MathTex(']):
                    obj_name = re.search(r'(\w+)\s*=', line)
                    if obj_name:
                        new_lines.append(f"        {obj_name.group(1)}.scale(0.7)  # Scale down for better spacing")
            code = '\n'.join(new_lines)
        
        # Ensure minimum buffer in all next_to calls
        code = re.sub(r'\.next_to\(([^,)]+)\)', r'.next_to(\1, buff=2.0)', code)
        code = re.sub(r'\.next_to\(([^,)]+),([^,)]+)\)', r'.next_to(\1, \2, buff=2.0)', code)
        
        # Add minimum spacing in arrange calls
        code = re.sub(r'\.arrange\(([^,)]+)\)', r'.arrange(\1, buff=2.0)', code)
        
        # Replace standard add with safe_add
        code = re.sub(r'self\.add\((.*?)\)', r'self.safe_add(\1)', code)
        
        # Add position checking after each transformation
        transform_pattern = r'(self\.play\([^)]*Transform[^)]*\))'
        code = re.sub(transform_pattern, r'\1\n        self.clear_overlaps()', code)
        
        # Add centered positioning for new elements
        lines = code.split('\n')
        new_lines = []
        for line in lines:
            new_lines.append(line)
            if '.set_z_index(' not in line and any(pattern in line for pattern in ['Create(', 'Write(', 'FadeIn(']):
                obj_match = re.search(r'[Create|Write|FadeIn]\((\w+)\)', line)
                if obj_match:
                    new_lines.append(f"        {obj_match.group(1)}.move_to(ORIGIN)  # Center new elements")
                    
        # Add shift adjustments for any elements that might be too close
        shift_code = """
        # Adjust positions to prevent overlaps
        for mobject in self.mobjects:
            if isinstance(mobject, Mobject):
                pos = mobject.get_center()
                if abs(pos[0]) < 0.5 and abs(pos[1]) < 0.5:  # Too close to center
                    mobject.shift(RIGHT * 1.5)  # Move right if too close to center
        """
        
        construct_pattern = r'(def\s+construct\s*\(\s*self\s*\)\s*:)'
        code = re.sub(construct_pattern, f'\\1\n{shift_code}', code)
        
        return '\n'.join(new_lines)

    @staticmethod
    def auto_fix_syntax_errors(code: str) -> str:
        """
        Automatically fix common syntax errors in the code before validation.
        
        Args:
            code: The Manim code to fix
            
        Returns:
            Fixed code with common syntax errors corrected
        """
        logger.info("Applying automatic syntax error fixes")
        
        # Fix 1: Ensure all self.play() method calls have properly closed parentheses
        lines = code.split('\n')
        for i in range(len(lines)):
            # Look for self.play calls with unbalanced parentheses
            if 'self.play(' in lines[i]:
                # Count parentheses
                open_count = lines[i].count('(')
                close_count = lines[i].count(')')
                
                # If unbalanced, add missing closing parentheses
                if open_count > close_count:
                    lines[i] += ')' * (open_count - close_count)
                    logger.info(f"Fixed unbalanced parentheses in line {i+1}: {lines[i]}")
        
        # Fix 2: Fix unterminated string literals in Code objects
        code_fixed = '\n'.join(lines)
        code_string_pattern = r'(Code\s*\(\s*code_string\s*=\s*["\'])([^"\']*?)(?!\s*["\'])'
        
        def fix_code_string(match):
            prefix = match.group(1)
            content = match.group(2)
            quote_char = prefix[-1]  # Get the quote character used
            # Add closing quote if missing
            return f"{prefix}{content}{quote_char})"
        
        code_fixed = re.sub(code_string_pattern, fix_code_string, code_fixed)
        
        # Fix 3: Add missing commas between animation objects in self.play() calls
        play_pattern = r'(self\.play\s*\()([^)]*?)(\))'
        
        def add_missing_commas(match):
            full_match = match.group(0)
            method_start = match.group(1)
            args = match.group(2)
            method_end = match.group(3)
            
            # Check for animation objects without commas between them
            animation_names = ['Create', 'Write', 'FadeIn', 'FadeOut', 'DrawBorderThenFill', 
                               'MoveToTarget', 'AnimationGroup', 'Indicate']
            
            # Look for patterns like Create(obj)Write(obj) and add comma
            for anim in animation_names:
                for other_anim in animation_names:
                    pattern = f'({anim}\\([^)]+\\))\\s*({other_anim}\\()'
                    args = re.sub(pattern, r'\1, \2', args)
            
            return f"{method_start}{args}{method_end}"
        
        code_fixed = re.sub(play_pattern, add_missing_commas, code_fixed)
        
        # Fix 4: Check for missing parentheses in specific statements causing common errors
        error_prone_statements = [
            # Fix incomplete self.play statements
            (r'(self\.play\([^)]*Create\([^)]*\))([^),)])', r'\1)\2'),
            (r'(self\.play\([^)]*Write\([^)]*\))([^),)])', r'\1)\2'),
            (r'(self\.play\([^)]*FadeIn\([^)]*\))([^),)])', r'\1)\2'),
            (r'(self\.play\([^)]*FadeOut\([^)]*\))([^),)])', r'\1)\2'),
        ]
        
        for pattern, replacement in error_prone_statements:
            code_fixed = re.sub(pattern, replacement, code_fixed)
        
        # Fix 5: Ensure proper command separation - replace commas between statements with semicolons
        # Look for statement sequences like self.play(...), self.add(...) 
        statement_separation_patterns = [
            (r'(self\.\w+\([^)]*\))\s*,\s*(self\.\w+)', r'\1;\n        \2'),
        ]
        
        for pattern, replacement in statement_separation_patterns:
            code_fixed = re.sub(pattern, replacement, code_fixed)
            
        # Fix 6: Remove any orphaned closing parentheses that might cause syntax errors
        orphaned_patterns = [
            (r'^\s*\)', ''),
            (r'(\S)\s+\)', r'\1'),
        ]
        
        for pattern, replacement in orphaned_patterns:
            code_fixed = re.sub(pattern, replacement, code_fixed, flags=re.MULTILINE)
            
        return code_fixed

    @staticmethod
    def fix_vector_specific_issues(code: str) -> str:
        """
        Fix common issues in vector-related animations.
        
        Args:
            code: The Manim code to fix
            
        Returns:
            Fixed code with vector-specific fixes applied
        """
        logger.info("Applying vector-specific fixes to animation code")
        
        # Fix 1: Ensure Arrow objects use proper parameters
        arrow_pattern = r'Arrow\s*\([^)]*\)'
        arrow_matches = re.findall(arrow_pattern, code)
        
        for arrow in arrow_matches:
            # Check if the arrow doesn't specify buff
            if "buff=" not in arrow:
                fixed_arrow = arrow.replace(")", ", buff=0)")
                code = code.replace(arrow, fixed_arrow)
            
            # Check if the arrow doesn't specify stroke_width
            if "stroke_width=" not in arrow:
                fixed_arrow = arrow.replace(")", ", stroke_width=4)")
                code = code.replace(arrow, fixed_arrow)
        
        # Fix 2: Make sure vector labels use MathTex with proper notation
        vector_label_patterns = [
            (r'Text\s*\([\'"]([vw])[\'"]', r'MathTex(r"\\vec{\1}"'),
            (r'Text\s*\([\'"]vector ([vw])[\'"]', r'MathTex(r"\\vec{\1}"'),
        ]
        
        for pattern, replacement in vector_label_patterns:
            code = re.sub(pattern, replacement, code)
        
        # Fix 3: Ensure proper vector operations
        code = re.sub(r'\.normalize\(\)', r'.get_unit_vector()', code)  # Use Manim methods
        code = re.sub(r'\.length\(\)', r'.get_length()', code)  # Use Manim methods
        
        # Fix 4: Fix incorrect vector rotation/angle calculation
        code = re.sub(r'\.angle\(\)', r'.get_angle()', code)
        
        # Fix 5: Make sure we have a proper coordinate system for vectors
        if "Arrow" in code and "Axes" not in code and "NumberPlane" not in code:
            coord_system = """        # Create coordinate system for vectors
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            axis_config={"include_tip": True},
        )
        axes_labels = axes.get_axis_labels(x_label="x", y_label="y")
        coord_system = VGroup(axes, axes_labels)
        
        # Show coordinate system
        self.play(Create(coord_system))
        self.wait(0.5)
"""
            # Insert after construct definition
            construct_pattern = r'(def\s+construct\s*\(\s*self\s*\)\s*:)'
            code = re.sub(construct_pattern, f'\\1\n{coord_system}', code)
            
            # Also ensure we clean up the coord system at the end
            # Find the last FadeOut (if any)
            fadeout_matches = list(re.finditer(r'self\.play\(\s*FadeOut\(([^)]*)\)', code))
            if fadeout_matches:
                last_fadeout = fadeout_matches[-1]
                # Add coord_system to the last FadeOut
                current_fadeout = last_fadeout.group(0)
                modified_fadeout = current_fadeout.replace('FadeOut(', 'FadeOut(coord_system, ')
                code = code.replace(current_fadeout, modified_fadeout)
            else:
                # No existing FadeOut, add one at the end
                code = code + "\n        # Clean up coordinate system\n        self.play(FadeOut(coord_system))\n        self.wait(0.5)"
        
        # Fix 6: Add GrowArrow animation for vectors instead of just Create
        code = re.sub(r'Create\(\s*([a-zA-Z0-9_]+_*(?:vector|arrow)[a-zA-Z0-9_]*)\s*\)', r'GrowArrow(\1)', code)
        
        logger.info("Vector-specific fixes applied")
        return code


class LayoutManager:
    """Helper class for managing layout and preventing overlaps in Manim animations."""
    
    @staticmethod
    def check_collision(obj1, obj2, min_buffer=0.5):
        """Check if two objects are too close to each other."""
        if not (hasattr(obj1, 'get_center') and hasattr(obj2, 'get_center')):
            return False
            
        center1 = obj1.get_center()
        center2 = obj2.get_center()
        
        # Get bounding dimensions
        width1 = obj1.width if hasattr(obj1, 'width') else 0.5
        height1 = obj1.height if hasattr(obj1, 'height') else 0.5
        width2 = obj2.width if hasattr(obj2, 'width') else 0.5
        height2 = obj2.height if hasattr(obj2, 'height') else 0.5
        
        # Add buffer to dimensions
        width1 += min_buffer
        height1 += min_buffer
        width2 += min_buffer
        height2 += min_buffer
        
        # Check for overlap in x and y directions
        dx = abs(center1[0] - center2[0])
        dy = abs(center1[1] - center2[1])
        
        min_dx = (width1 + width2) / 2
        min_dy = (height1 + height2) / 2
        
        return dx < min_dx and dy < min_dy

    @staticmethod 
    def resolve_collisions(elements, min_buffer=0.5):
        """Resolve any collisions between elements by adjusting positions."""
        from manim import RIGHT, UP, LEFT, DOWN
        
        changes_made = True
        max_iterations = 10  # Prevent infinite loops
        iteration = 0
        
        while changes_made and iteration < max_iterations:
            changes_made = False
            iteration += 1
            
            for i, elem1 in enumerate(elements):
                for j, elem2 in enumerate(elements):
                    if i != j and LayoutManager.check_collision(elem1, elem2, min_buffer):
                        # Get centers
                        c1 = elem1.get_center()
                        c2 = elem2.get_center()
                        
                        # Calculate direction to push
                        dx = c1[0] - c2[0]
                        dy = c1[1] - c2[1]
                        
                        # Push elements apart
                        push_distance = min_buffer * 1.5
                        if abs(dx) > abs(dy):
                            # Push horizontally
                            direction = RIGHT if dx > 0 else LEFT
                            elem1.shift(direction * push_distance)
                            elem2.shift(direction * -push_distance)
                        else:
                            # Push vertically
                            direction = UP if dy > 0 else DOWN
                            elem1.shift(direction * push_distance)
                            elem2.shift(direction * -push_distance)
                            
                        changes_made = True

    def arrange_elements(self, *elements, direction=RIGHT, buff=1.5):
        """Arrange elements with proper spacing and scaling."""
        group = VGroup(*elements)
        group.arrange(direction, buff=buff)
        
        # Auto-scale if too large
        max_width = config.frame_width * 0.7  # More conservative width
        max_height = config.frame_height * 0.7  # More conservative height
        
        if group.width > max_width:
            scale_factor = max_width / group.width
            group.scale(scale_factor * 0.85)  # More aggressive scaling
            
        if group.height > max_height:
            scale_factor = max_height / group.height
            group.scale(scale_factor * 0.85)  # More aggressive scaling
        
        # Resolve any remaining collisions
        self.resolve_collisions(elements, min_buffer=1.0)
        
        # Center the group
        group.move_to(ORIGIN)
        
        return group
        
    def position_element(self, element, anchor_element=None, direction=RIGHT, buff=1.5):
        """Position an element relative to another with strict collision avoidance."""
        if anchor_element:
            # Start with larger initial buffer
            element.next_to(anchor_element, direction, buff=buff)
            
            # Keep increasing buffer until no collision
            current_buff = buff
            while self.check_collision(element, anchor_element):
                current_buff += 0.5
                element.next_to(anchor_element, direction, buff=current_buff)
                
                if current_buff > 5:  # Safety limit
                    break
        
        # Ensure element is within frame bounds with larger margins
        frame_width = config.frame_width / 2 - 1.0  # Larger margin
        frame_height = config.frame_height / 2 - 1.0  # Larger margin
        
        # Check and adjust position if needed
        pos = element.get_center()
        if abs(pos[0]) > frame_width:
            shift_x = frame_width * (1 if pos[0] < 0 else -1) - pos[0]
            element.shift(RIGHT * shift_x)
            
        if abs(pos[1]) > frame_height:
            shift_y = frame_height * (1 if pos[1] < 0 else -1) - pos[1]
            element.shift(UP * shift_y)
        
        return element
        
    @staticmethod
    def add_layout_methods(code: str) -> str:
        """Add layout management methods to the scene class."""
        layout_methods = """
    def setup(self):
        super().setup()
        # Initialize a list to track objects and their positions
        self.object_positions = []
        
    def clear_overlaps(self):
        \"\"\"Clear any overlapping elements by adjusting positions.\"\"\"
        all_mobjects = [m for m in self.mobjects if isinstance(m, Mobject)]
        moved = True
        iterations = 0
        
        while moved and iterations < 10:
            moved = False
            iterations += 1
            
            for i, obj1 in enumerate(all_mobjects):
                for j, obj2 in enumerate(all_mobjects[i+1:], i+1):
                    if obj1 == obj2 or not (hasattr(obj1, 'get_center') and hasattr(obj2, 'get_center')):
                        continue
                        
                    # Calculate distance between objects
                    c1 = obj1.get_center()
                    c2 = obj2.get_center()
                    dx = c2[0] - c1[0]
                    dy = c2[1] - c1[1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    # Get object sizes
                    w1 = obj1.width if hasattr(obj1, 'width') else 0.5
                    h1 = obj1.height if hasattr(obj1, 'height') else 0.5
                    w2 = obj2.width if hasattr(obj2, 'width') else 0.5
                    h2 = obj2.height if hasattr(obj2, 'height') else 0.5
                    
                    # Calculate minimum required distance
                    min_dist = max(w1 + w2, h1 + h2) * 0.75  # Add 50% extra space
                    
                    if dist < min_dist:
                        # Move objects apart
                        move = (min_dist - dist) * 0.6  # Move each object by half the overlap
                        if abs(dx) > abs(dy):
                            obj1.shift(LEFT * move)
                            obj2.shift(RIGHT * move)
                        else:
                            obj1.shift(DOWN * move)
                            obj2.shift(UP * move)
                        moved = True
        
    def safe_add(self, *mobjects):
        \"\"\"Add objects to scene while preventing overlaps.\"\"\"
        # First add the objects
        self.add(*mobjects)
        
        # Then ensure no overlaps
        self.clear_overlaps()
        
        # Finally ensure everything is within frame bounds
        frame_width = config.frame_width * 0.4  # Use only 80% of frame
        frame_height = config.frame_height * 0.4
        
        for mobject in mobjects:
            if not hasattr(mobject, 'get_center'):
                continue
                
            pos = mobject.get_center()
            # Move object if it's too far from center
            if abs(pos[0]) > frame_width:
                mobject.shift(RIGHT * (frame_width - abs(pos[0])) * np.sign(-pos[0]))
            if abs(pos[1]) > frame_height:
                mobject.shift(UP * (frame_height - abs(pos[1])) * np.sign(-pos[1]))
        
    def arrange_elements(self, *elements, direction=RIGHT, buff=2.0):
        \"\"\"Arrange elements with guaranteed spacing.\"\"\"
        if not elements:
            return None
            
        # Create group and arrange with large buffer
        group = VGroup(*elements)
        group.arrange(direction, buff=buff)
        
        # Scale down if too large
        max_width = config.frame_width * 0.7
        max_height = config.frame_height * 0.7
        
        if group.width > max_width:
            scale = max_width / group.width * 0.9
            group.scale(scale)
            
        if group.height > max_height:
            scale = max_height / group.height * 0.9
            group.scale(scale)
        
        # Center the group
        group.move_to(ORIGIN)
        
        # Apply z-index based on creation order
        for i, elem in enumerate(elements):
            elem.set_z_index(i)
        
        return group
        
    def create_grid_layout(self, elements, cols=2, buff_x=2.0, buff_y=2.0):
        \"\"\"Create a grid layout with guaranteed spacing.\"\"\"
        if not elements:
            return None
            
        # Calculate rows needed
        n = len(elements)
        rows = (n + cols - 1) // cols
        
        # Create grid
        grid = VGroup()
        for i in range(rows):
            row_elements = elements[i * cols : min((i + 1) * cols, n)]
            row = VGroup(*row_elements).arrange(RIGHT, buff=buff_x)
            grid.add(row)
        
        grid.arrange(DOWN, buff=buff_y)
        
        # Ensure grid fits in frame
        if grid.width > config.frame_width * 0.8:
            grid.scale_to_fit_width(config.frame_width * 0.8)
        if grid.height > config.frame_height * 0.8:
            grid.scale_to_fit_height(config.frame_height * 0.8)
            
        # Center grid
        grid.move_to(ORIGIN)
        
        return grid
        
    def position_element(self, element, reference=None, direction=RIGHT, buff=2.0):
        \"\"\"Position an element with collision avoidance.\"\"\"
        if reference:
            element.next_to(reference, direction, buff=buff)
        
        # Ensure element is within frame bounds
        max_x = config.frame_width * 0.4
        max_y = config.frame_height * 0.4
        
        pos = element.get_center()
        if abs(pos[0]) > max_x:
            element.shift(RIGHT * (max_x - abs(pos[0])) * np.sign(-pos[0]))
        if abs(pos[1]) > max_y:
            element.shift(UP * (max_y - abs(pos[1])) * np.sign(-pos[1]))
        
        # Check for collisions with existing objects
        self.clear_overlaps()
        
        return element"""
        
        # Add imports if needed
        if "import numpy as np" not in code:
            code = code.replace("from manim import *", "from manim import *\nimport numpy as np")
        
        # Insert layout methods after class definition
        class_pattern = r'(class\s+AnimationScene\s*\(\s*Scene\s*\)\s*:)'
        return re.sub(class_pattern, f'\\1\n{layout_methods}', code)
        
    @staticmethod
    def apply_layout_management(code: str) -> str:
        """Apply layout management to existing animations."""
        code = LayoutManager.add_layout_methods(code)
        
        # Replace direct next_to calls with position_element
        code = re.sub(
            r'(\w+)\.next_to\((.*?)\)',
            r'self.position_element(\1, \2)',
            code
        )
        
        # Replace direct arrange calls with arrange_elements
        code = re.sub(
            r'(\w+)\.arrange\((.*?)\)',
            r'self.arrange_elements(\1, \2)',
            code
        )
        
        # Add proper z-indexing
        if "set_z_index" not in code:
            lines = code.split('\n')
            new_lines = []
            z_index = 0
            for line in lines:
                new_lines.append(line)
                if any(create_pattern in line for create_pattern in ['Create(', 'Write(', 'FadeIn(']):
                    obj_name = re.search(r'(Create|Write|FadeIn)\((\w+)', line)
                    if obj_name and obj_name.group(2):
                        new_lines.append(f'        {obj_name.group(2)}.set_z_index({z_index})')
                        z_index += 1
            code = '\n'.join(new_lines)
            
        return code


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