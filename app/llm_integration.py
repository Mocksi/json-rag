"""
LLM Integration Module

This module provides functionality for integrating with Language Models (LLMs).
It includes functions for sending prompts to LLMs and processing their responses.
"""

import os
import json
import logging
import requests
import re
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Look for environment variables for LLM API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

def estimate_token_count(text: str) -> int:
    """
    Estimate the number of tokens in a text string.
    
    This is a simple approximation - OpenAI tokens are roughly 4 characters per token
    but this varies by language and content.
    
    Args:
        text: The text to estimate token count for
        
    Returns:
        int: Estimated token count
    """
    # More conservative estimation - 1 token ~= 3.5 chars for English text with JSON
    # This helps prevent underestimation
    return int(len(text) / 3.5)

def truncate_text_to_token_limit(text: str, max_tokens: int) -> str:
    """
    Truncate text to fit within token limit.
    
    Args:
        text: Text to truncate
        max_tokens: Maximum tokens allowed
        
    Returns:
        str: Truncated text
    """
    estimated_tokens = estimate_token_count(text)
    
    if estimated_tokens <= max_tokens:
        return text
    
    # Calculate approximate character limit
    char_limit = max_tokens * 4
    truncated = text[:char_limit]
    
    # Try to truncate at a reasonable point like a line break or period
    last_newline = truncated.rfind('\n')
    last_period = truncated.rfind('.')
    
    cut_point = max(last_newline, last_period)
    if cut_point > char_limit * 0.8:  # Only use if reasonably close to the end
        truncated = truncated[:cut_point + 1]
    
    return truncated + "\n[Context truncated due to token limits]"

def chunk_context(context: str, system_message: str, query: str, max_context_tokens: int = 3000) -> List[str]:
    """
    Split large context into manageable chunks that fit within token limits.
    
    Args:
        context: The context text to split
        system_message: System message that will be included with each chunk
        query: User query that will be included with each chunk
        max_context_tokens: Maximum tokens for context portion
        
    Returns:
        List[str]: List of context chunks
    """
    # Estimate tokens for fixed parts (more conservative overhead)
    fixed_tokens = estimate_token_count(system_message + query) + 500  # 500 for overhead
    
    # Calculate available tokens for context
    available_tokens = max_context_tokens - fixed_tokens
    if available_tokens <= 0:
        logger.warning("Not enough token space for context after system message and query")
        return ["[Context omitted due to token constraints]"]
    
    # Split context on natural boundaries if it's too large
    estimated_context_tokens = estimate_token_count(context)
    
    if estimated_context_tokens <= available_tokens:
        return [context]
    
    # For structured JSON context, try to split on meaningful boundaries
    chunks = []
    current_chunk = ""
    current_tokens = 0
    
    # Try to split on JSON objects first
    json_objects = re.split(r'(\n\s*\{\s*"[^"]+"\s*:)', context)
    
    # If we couldn't split on JSON objects or there's only one, try splitting on sections
    if len(json_objects) <= 3:  # Original string + at most one split
        sections = re.split(r'(\n###\s+[^\n]+\n|\n##\s+[^\n]+\n|\n#\s+[^\n]+\n)', context)
        
        for i in range(0, len(sections), 2):
            section_header = sections[i] if i == 0 else sections[i-1] + sections[i]
            section_content = sections[i+1] if i+1 < len(sections) else ""
            
            section = section_header + section_content
            section_tokens = estimate_token_count(section)
            
            if current_tokens + section_tokens <= available_tokens:
                current_chunk += section
                current_tokens += section_tokens
            else:
                # If we can't fit this section, add current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)
                
                # If the section itself exceeds available tokens, we need to split it further
                if section_tokens > available_tokens:
                    # Split on paragraphs
                    paragraphs = re.split(r'(\n\n+)', section)
                    paragraph_chunk = ""
                    paragraph_tokens = 0
                    
                    for j in range(0, len(paragraphs), 2):
                        para = paragraphs[j]
                        para_separator = paragraphs[j+1] if j+1 < len(paragraphs) else ""
                        para_full = para + para_separator
                        para_tokens = estimate_token_count(para_full)
                        
                        if paragraph_tokens + para_tokens <= available_tokens:
                            paragraph_chunk += para_full
                            paragraph_tokens += para_tokens
                        else:
                            if paragraph_chunk:
                                chunks.append(paragraph_chunk)
                            
                            # If a single paragraph is too large, truncate it
                            if para_tokens > available_tokens:
                                truncated_para = truncate_text_to_token_limit(para_full, available_tokens)
                                chunks.append(truncated_para)
                                paragraph_chunk = ""
                                paragraph_tokens = 0
                            else:
                                paragraph_chunk = para_full
                                paragraph_tokens = para_tokens
                    
                    if paragraph_chunk:
                        chunks.append(paragraph_chunk)
                else:
                    current_chunk = section
                    current_tokens = section_tokens
    else:
        # Process JSON objects
        for i in range(0, len(json_objects), 2):
            json_part = json_objects[i]
            json_separator = json_objects[i+1] if i+1 < len(json_objects) else ""
            json_full = json_part + json_separator
            json_tokens = estimate_token_count(json_full)
            
            if current_tokens + json_tokens <= available_tokens:
                current_chunk += json_full
                current_tokens += json_tokens
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                
                if json_tokens > available_tokens:
                    truncated_json = truncate_text_to_token_limit(json_full, available_tokens)
                    chunks.append(truncated_json)
                    current_chunk = ""
                    current_tokens = 0
                else:
                    current_chunk = json_full
                    current_tokens = json_tokens
    
    # Add the last chunk if it has content
    if current_chunk:
        chunks.append(current_chunk)
    
    # If we still have no chunks (maybe our splitting failed), do a simple length-based split
    if not chunks:
        logger.warning("Falling back to basic chunk splitting")
        chunk_size = available_tokens * 3  # Convert token limit to char limit (approx)
        for i in range(0, len(context), chunk_size):
            chunks.append(context[i:i+chunk_size])
    
    return chunks

def summarize_chunk(chunk: str, query: str, max_summary_tokens: int = 250) -> str:
    """
    Summarize a chunk of context to reduce token usage.
    
    Args:
        chunk: The chunk of context to summarize
        query: The user's query to focus the summary
        max_summary_tokens: Maximum tokens for the summary
        
    Returns:
        str: Summarized chunk
    """
    # If chunk is already small, return it as is
    if estimate_token_count(chunk) <= max_summary_tokens:
        return chunk
        
    # Create a summarization prompt
    system_message = "You are an extreme summarization assistant. Create the most concise summary possible that preserves only essential information relevant to the user's query. Focus only on key facts. Use bullet points where possible."
    
    summarize_prompt = f"""Summarize the following text in at most {max_summary_tokens} tokens, focusing ONLY on information directly relevant to: "{query}"
    
TEXT TO SUMMARIZE:
{chunk}

EXTREMELY CONCISE SUMMARY (bullet points preferred):"""

    try:
        # Use a lower temperature for more deterministic summaries
        summary = _get_openai_response(summarize_prompt, system_message, temperature=0.1, max_tokens=max_summary_tokens)
        return summary
    except Exception as e:
        logger.error(f"Error summarizing chunk: {e}")
        # Fall back to basic truncation if summarization fails
        return truncate_text_to_token_limit(chunk, max_summary_tokens)

def apply_map_reduce(context: str, query: str, system_message: str, max_final_context_tokens: int = 3000) -> str:
    """
    Apply map-reduce pattern to large context:
    1. Split context into chunks
    2. Summarize each chunk (map)
    3. Combine summaries (reduce)
    
    Args:
        context: Full context text
        query: User's query
        system_message: System message for the LLM
        max_final_context_tokens: Maximum tokens for final combined context
        
    Returns:
        str: Processed context ready for the LLM
    """
    logger.info("Applying map-reduce to large context")
    
    # Split context into manageable chunks
    context_chunks = chunk_context(context, system_message, query, max_context_tokens=6000)
    
    if not context_chunks:
        logger.warning("No context chunks extracted.")
        return "No relevant context information available."
    
    # Always summarize all chunks, even if there's only one
    # Map phase: Summarize each chunk
    logger.info(f"Map phase: Summarizing {len(context_chunks)} chunks")
    summaries = []
    for i, chunk in enumerate(context_chunks):
        logger.info(f"Summarizing chunk {i+1}/{len(context_chunks)}")
        summary = summarize_chunk(chunk, query)
        if summary and not summary.isspace():
            summaries.append(summary)
    
    if not summaries:
        logger.warning("No valid summaries generated.")
        return "No relevant context information available."
    
    # Reduce phase: Combine summaries
    logger.info("Reduce phase: Combining summaries")
    combined_summaries = "\n\n".join(summaries)
    
    # Check if combined summaries are still too large
    if estimate_token_count(combined_summaries) > max_final_context_tokens:
        # Further reduce by creating a summary of summaries
        logger.info("Creating summary of summaries")
        reduce_system_message = "You are a data distillation expert. Combine these summaries into a MINIMAL context containing ONLY information directly relevant to the user's question. Be ruthlessly concise. Use bullet points."
        reduce_prompt = f"""The following are summaries of different parts of a document. 
Create the most concise possible synthesis (max {max_final_context_tokens} tokens) containing ONLY information directly relevant to: "{query}"

SUMMARIES TO COMBINE:
{combined_summaries}

ULTRA-CONCISE SYNTHESIS:"""

        try:
            final_context = _get_openai_response(reduce_prompt, reduce_system_message, temperature=0.1, max_tokens=max_final_context_tokens)
            return final_context
        except Exception as e:
            logger.error(f"Error in reduce phase: {e}")
            # Fall back to more aggressive truncation if reduce fails
            return truncate_text_to_token_limit(combined_summaries, max_final_context_tokens // 3)
    
    return combined_summaries

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
    
    # Separate context from query by looking for the "Question:" marker
    context = ""
    query = prompt
    
    if "Question:" in prompt:
        parts = prompt.split("Question:", 1)
        context = parts[0].strip()
        query = "Question:" + parts[1]
    
    # Set default system message if none provided
    if not system_message:
        system_message = "You are a helpful AI assistant focused on answering precisely and concisely. Answer based only on the provided context."
    
    # Try OpenAI first
    if OPENAI_API_KEY:
        try:
            # Check if context is small enough to send as a single message
            total_tokens = estimate_token_count(prompt) + estimate_token_count(system_message) + max_tokens
            model_max_tokens = 10000  # Even more conservative limit for GPT-3.5-turbo
            
            if total_tokens <= model_max_tokens:
                # Context fits within token limit, process normally
                return _get_openai_response(prompt, system_message, temperature, max_tokens)
            else:
                # Context is too large, need to handle in chunks
                logger.info(f"Context is too large ({total_tokens} estimated tokens), processing with map-reduce")
                
                # Always use map-reduce for large contexts
                processed_context = apply_map_reduce(context, query.replace("Question:", "").strip(), system_message)
                
                # Create new prompt with processed context
                new_prompt = processed_context + "\n\n" + query
                new_total_tokens = estimate_token_count(new_prompt) + estimate_token_count(system_message) + max_tokens
                
                if new_total_tokens <= model_max_tokens:
                    # If processed context fits, use it
                    logger.info(f"Reduced context to {estimate_token_count(processed_context)} tokens")
                    return _get_openai_response(new_prompt, system_message, temperature, max_tokens)
                else:
                    # If still too large, aggressively reduce the context further
                    logger.warning(f"Map-reduce result still too large ({new_total_tokens} tokens), reducing further")
                    
                    # Calculate how much we need to reduce by
                    reduction_factor = new_total_tokens / model_max_tokens
                    reduced_context_tokens = int(estimate_token_count(processed_context) / (reduction_factor * 1.5))  # Extra safety margin
                    
                    # Truncate with prioritization of important parts
                    truncated_context = truncate_text_to_token_limit(processed_context, reduced_context_tokens)
                    final_prompt = truncated_context + "\n\n" + query
                    
                    logger.info(f"Aggressively reduced context to approximately {reduced_context_tokens} tokens")
                    
                    return _get_openai_response(final_prompt, system_message, temperature, max_tokens) + " [Note: Response based on heavily reduced context due to length limitations]"
        except Exception as e:
            logger.error(f"Error with OpenAI API: {e}")
    
    # Fallback to Cohere
    if COHERE_API_KEY:
        try:
            # For Cohere, we'll simplify and just truncate if needed
            if estimate_token_count(prompt) > 4000:  # Cohere has different token limits
                prompt = truncate_text_to_token_limit(prompt, 4000)
                logger.info("Truncated prompt for Cohere API")
            
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
    
    # Estimate tokens before sending
    estimated_tokens = estimate_token_count(prompt) + (estimate_token_count(system_message) if system_message else 0) + max_tokens
    logger.info(f"Sending request to OpenAI API with estimated {estimated_tokens} tokens")
    
    # If we're getting close to limits, proactively reduce
    if estimated_tokens > 15000:
        logger.warning(f"Estimated tokens ({estimated_tokens}) approaching limit, reducing prompt")
        
        # Calculate how much to reduce by
        overage_factor = estimated_tokens / 15000
        reduced_prompt_tokens = int(estimate_token_count(prompt) / overage_factor)
        
        # Truncate prompt
        prompt = truncate_text_to_token_limit(prompt, reduced_prompt_tokens)
        
        # Rebuild messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Re-estimate after reduction
        estimated_tokens = estimate_token_count(prompt) + (estimate_token_count(system_message) if system_message else 0) + max_tokens
        logger.info(f"Reduced to estimated {estimated_tokens} tokens")
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        error_str = str(e)
        logger.error(f"OpenAI API error: {e}")
        
        # Handle context length exceeded error specifically
        if "context_length_exceeded" in error_str:
            # Try again with more aggressive reduction
            logger.warning("Context length exceeded, attempting with smaller context")
            
            # Extract the actual token count from the error message if possible
            token_count_match = re.search(r'resulted in (\d+) tokens', error_str)
            current_tokens = int(token_count_match.group(1)) if token_count_match else estimated_tokens
            
            # Calculate reduction factor (target 80% of max)
            max_tokens_from_error = 16385  # Maximum from error message
            target_tokens = int(max_tokens_from_error * 0.8)
            reduction_ratio = target_tokens / current_tokens
            
            # Reduce prompt size accordingly
            prompt_tokens = estimate_token_count(prompt)
            reduced_prompt_tokens = int(prompt_tokens * reduction_ratio)
            
            if reduced_prompt_tokens < 1000:
                # Too small to be useful
                raise ValueError("Prompt cannot be reduced enough to fit token limits")
            
            reduced_prompt = truncate_text_to_token_limit(prompt, reduced_prompt_tokens)
            
            # Try again with reduced prompt
            try:
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": reduced_prompt})
                
                logger.info(f"Retrying with reduced prompt of ~{reduced_prompt_tokens} tokens")
                
                response = openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip() + " [Note: Response based on reduced context due to token limitations]"
            except Exception as e2:
                logger.error(f"Failed even with reduced prompt: {e2}")
                raise
        else:
            # Other error types, just re-raise
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