"""
LLM client for interacting with OpenAI API.
"""
import os
import logging
import time
from typing import Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class LLMClient:
    """Client for interacting with OpenAI's API."""
    
    def __init__(self, model: str = None):
        """Initialize OpenAI client."""
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv('DEFAULT_MODEL', 'gpt-3.5-turbo')
        self.temperature = float(os.getenv('DEFAULT_TEMPERATURE', '0.7'))
        self.max_tokens = int(os.getenv('DEFAULT_MAX_TOKENS', '500'))
        
        logger.info(f"LLM Client initialized with model: {self.model}")
    
    def generate_response(self, prompt: str, temperature: Optional[float] = None,
                         max_tokens: Optional[int] = None) -> Dict:
        """
        Generate a response from the LLM.
        
        Returns:
            dict with keys: response_text, tokens_used, response_time
        """
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens
            )
            
            response_time = time.time() - start_time
            
            result = {
                'response_text': response.choices[0].message.content,
                'tokens_used': response.usage.total_tokens,
                'response_time': response_time,
                'model': self.model
            }
            
            logger.info(f"Response generated in {response_time:.2f}s using {result['tokens_used']} tokens")
            return result
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def batch_generate(self, prompts: list, temperature: Optional[float] = None,
                      max_tokens: Optional[int] = None) -> list:
        """
        Generate responses for multiple prompts.
        
        Returns:
            list of result dictionaries
        """
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Processing prompt {i+1}/{len(prompts)}")
            try:
                result = self.generate_response(prompt, temperature, max_tokens)
                results.append(result)
            except Exception as e:
                logger.error(f"Error on prompt {i+1}: {e}")
                results.append({
                    'response_text': f"ERROR: {str(e)}",
                    'tokens_used': 0,
                    'response_time': 0.0,
                    'model': self.model
                })
        
        return results
