"""
LLM Interface for Agentic-RIS
==============================

Handles communication with LLM APIs (Cerebras, OpenAI, Anthropic).

Author: Research Team
Date: January 2026
"""

import os
import json
from typing import Dict, List, Optional
from cerebras.cloud.sdk import Cerebras
from dotenv import load_dotenv

load_dotenv()  # loads .env from project root

class LLMInterface:
    """Interface to LLM APIs."""
    
    def __init__(
        self,
        provider: str = "cerebras",  
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize LLM interface.
        
        Args:
            provider: LLM provider
            api_key: API key (if None, reads from environment)
            model: Model name (if None, uses default)
        """
        self.provider = provider.lower()
        
        # Get API key
        if api_key is None:
            if self.provider == "cerebras":
                api_key = os.getenv("CEREBRAS_API_KEY")
        
        if not api_key:
            raise ValueError(f"API key not found for provider: {self.provider}")
        
        # Initialize client
        if self.provider == "cerebras":
            # Cerebras uses OpenAI-compatible API
            self.client = Cerebras(
                api_key=api_key
            )
            self.model = model or "qwen-3-32b"
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def query(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """
        Query LLM with prompt.
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            temperature: Sampling temperature
            max_tokens: Maximum response tokens
            
        Returns:
            response: LLM response text
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
            
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    
    def extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        try:
            # Find JSON block
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                return json.loads(json_str)
            return None
        except Exception:
            return None