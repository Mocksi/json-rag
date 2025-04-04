"""
Query Intent Module

This module provides functionality for analyzing user queries to determine intent.
It re-exports the analyze_query_intent function from app.analysis.intent.
"""

from app.analysis.intent import analyze_query_intent

# Re-export the function 
__all__ = ["analyze_query_intent"] 