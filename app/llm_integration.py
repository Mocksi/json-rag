"""
LLM Integration Module

This module provides functionality for integrating with Language Models (LLMs).
It includes functions for sending prompts to LLMs and processing their responses.
"""

import os
import json
import logging
import requests
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Look for environment variables for LLM API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def get_llm_response(
    prompt: str, 
    system_message: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1000
) -> str:
    """
    Get a response from a Language Model for a given prompt.
    
    Args:
        prompt: The prompt to send to the LLM
        system_message: Optional system message for models that support it
        temperature: Control randomness (0.0-1.0)
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        str: The LLM's response text
        
    Note:
        - Uses OpenAI API if OPENAI_API_KEY is set
        - Falls back to Cohere if COHERE_API_KEY is set
        - If neither is available, returns a placeholder message
    """
    logger.info("Generating LLM response for prompt")
    
    # Try OpenAI first
    if OPENAI_API_KEY:
        try:
            return _get_openai_response(prompt, system_message, temperature, max_tokens)
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")
    
    # Fallback to Cohere
    if COHERE_API_KEY:
        try:
            return _get_cohere_response(prompt, temperature, max_tokens)
        except Exception as e:
            logger.error(f"Error with Cohere API: {e}")
    
    # No API keys available or both APIs failed
    logger.warning("No LLM API keys available or all LLM API calls failed")
    return f"I don't have access to an LLM API right now. Here's what I was asked: {prompt[:100]}..."

def _get_openai_response(
    prompt: str, 
    system_message: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1000
) -> str:
    """Get a response from OpenAI API."""
    import openai
    
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise

def _get_cohere_response(
    prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 1000
) -> str:
    """Get a response from Cohere API."""
    url = "https://api.cohere.ai/v1/generate"
    
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "command",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result.get("generations", [{}])[0].get("text", "").strip()
    except Exception as e:
        logger.error(f"Cohere API error: {e}")
        raise 