"""
Response evaluator using simple heuristics.
"""
import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)


class ResponseEvaluator:
    """Evaluates LLM responses using heuristic methods."""
    
    def __init__(self):
        """Initialize evaluator."""
        pass
    
    def evaluate(self, prompt: str, response: str, response_time: float, 
                 tokens_used: int) -> Dict[str, float]:
        """
        Evaluate a response across multiple dimensions.
        Returns scores between 0.0 and 1.0 for each metric.
        """
        scores = {
            'relevance': self._evaluate_relevance(prompt, response),
            'accuracy': self._evaluate_accuracy(response),
            'completeness': self._evaluate_completeness(response),
            'consistency': self._evaluate_consistency(response),
            'efficiency': self._evaluate_efficiency(response_time, tokens_used),
            'bias': self._evaluate_bias(response)
        }
        
        logger.debug(f"Evaluation scores: {scores}")
        return scores
    
    def _evaluate_relevance(self, prompt: str, response: str) -> float:
        """
        Evaluate relevance by checking keyword overlap between prompt and response.
        Score: 0.0 to 1.0
        """
        # Extract meaningful words (exclude common stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                      'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been'}
        
        prompt_words = set(re.findall(r'\b\w+\b', prompt.lower())) - stop_words
        response_words = set(re.findall(r'\b\w+\b', response.lower())) - stop_words
        
        if not prompt_words:
            return 0.5  # Default if no meaningful words in prompt
        
        # Calculate overlap
        overlap = len(prompt_words & response_words)
        score = min(overlap / len(prompt_words), 1.0)
        
        return score
    
    def _evaluate_accuracy(self, response: str) -> float:
        """
        Evaluate accuracy based on presence of confidence indicators and hedge words.
        Higher score = more confident/definitive language
        Score: 0.0 to 1.0
        """
        # Confidence indicators (positive)
        confidence_words = ['definitely', 'certainly', 'clearly', 'specifically', 
                           'precisely', 'exactly', 'proven', 'established']
        
        # Hedge words (negative)
        hedge_words = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'uncertain',
                      'unclear', 'probably', 'likely', 'seems', 'appears']
        
        response_lower = response.lower()
        
        confidence_count = sum(1 for word in confidence_words if word in response_lower)
        hedge_count = sum(1 for word in hedge_words if word in response_lower)
        
        # Base score starts at 0.7 (neutral)
        score = 0.7
        score += confidence_count * 0.05
        score -= hedge_count * 0.05
        
        return max(0.0, min(score, 1.0))
    
    def _evaluate_completeness(self, response: str) -> float:
        """
        Evaluate completeness based on response length and structure.
        Score: 0.0 to 1.0
        """
        word_count = len(response.split())
        sentence_count = len(re.split(r'[.!?]+', response.strip()))
        
        # Optimal range: 50-200 words, 3-8 sentences
        word_score = min(word_count / 100, 1.0) if word_count < 100 else max(1.0 - (word_count - 200) / 200, 0.5)
        sentence_score = min(sentence_count / 5, 1.0) if sentence_count < 5 else 1.0
        
        # Check for list/structure indicators
        has_structure = any(indicator in response for indicator in ['1.', '2.', '-', '•', 'First', 'Second'])
        structure_bonus = 0.1 if has_structure else 0.0
        
        score = (word_score + sentence_score) / 2 + structure_bonus
        
        return min(score, 1.0)
    
    def _evaluate_consistency(self, response: str) -> float:
        """
        Evaluate consistency by checking for contradictions and coherence.
        Score: 0.0 to 1.0
        """
        # Look for contradiction indicators
        contradictions = ['however', 'but', 'although', 'contrary', 'despite', 
                         'on the other hand', 'conversely']
        
        response_lower = response.lower()
        contradiction_count = sum(1 for word in contradictions if word in response_lower)
        
        # Multiple contradictions may indicate inconsistency
        if contradiction_count == 0:
            score = 0.9  # Likely consistent
        elif contradiction_count == 1:
            score = 0.7  # One contrast is okay
        else:
            score = max(0.5 - (contradiction_count - 2) * 0.1, 0.3)
        
        # Check for repeated information (may indicate confusion)
        sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
        if len(sentences) > 1:
            unique_ratio = len(set(sentences)) / len(sentences)
            score *= unique_ratio
        
        return max(0.0, min(score, 1.0))
    
    def _evaluate_efficiency(self, response_time: float, tokens_used: int) -> float:
        """
        Evaluate efficiency based on response time and token usage.
        Score: 0.0 to 1.0 (higher = more efficient)
        """
        # Time score (faster is better, optimal < 2 seconds)
        if response_time < 2.0:
            time_score = 1.0
        elif response_time < 5.0:
            time_score = 0.8
        elif response_time < 10.0:
            time_score = 0.6
        else:
            time_score = 0.4
        
        # Token score (fewer tokens for similar information is better)
        if tokens_used < 150:
            token_score = 1.0
        elif tokens_used < 300:
            token_score = 0.8
        elif tokens_used < 500:
            token_score = 0.6
        else:
            token_score = 0.4
        
        return (time_score + token_score) / 2
    
    def _evaluate_bias(self, response: str) -> float:
        """
        Evaluate potential bias by checking for loaded language.
        Score: 0.0 to 1.0 (higher = less bias detected)
        """
        # Loaded/biased language indicators
        bias_indicators = [
            'obviously', 'clearly', 'everyone knows', 'it is well known',
            'always', 'never', 'all', 'none', 'must', 'should',
            'superior', 'inferior', 'better', 'worse', 'best', 'worst'
        ]
        
        response_lower = response.lower()
        bias_count = sum(1 for indicator in bias_indicators if indicator in response_lower)
        
        # Start with high score (assume no bias)
        score = 1.0 - (bias_count * 0.1)
        
        # Check for balanced language
        has_balance = any(word in response_lower for word in ['consider', 'perspective', 'may', 'can'])
        if has_balance:
            score = min(score + 0.1, 1.0)
        
        return max(0.0, min(score, 1.0))
