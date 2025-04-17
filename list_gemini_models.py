#!/usr/bin/env python3
"""
Script to list all available Gemini models for the configured API key.
This will help identify the correct model names to use in the application.
"""

import os
import google.generativeai as genai
import logging
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def list_available_models():
    """List all available models for the configured API key."""
    try:
        # Load environment variables from .env file if it exists
        load_dotenv()
        
        # Get API key from environment
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY environment variable is not set")
            return
            
        # Configure the API
        genai.configure(api_key=api_key)
        
        # List available models
        models = genai.list_models()
        
        logger.info("Available models:")
        for model in models:
            logger.info(f"- Name: {model.name}")
            logger.info(f"  Display name: {model.display_name}")
            logger.info(f"  Description: {model.description}")
            logger.info(f"  Supported generation methods: {model.supported_generation_methods}")
            logger.info("--------------------------------------------------")
            
        # Find models that specifically support text generation
        logger.info("\nModels that support text generation:")
        for model in models:
            if "generateContent" in model.supported_generation_methods:
                logger.info(f"- {model.name} ({model.display_name})")
                
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")

if __name__ == "__main__":
    list_available_models()