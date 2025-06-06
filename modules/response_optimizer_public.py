import logging
from collections import defaultdict
from typing import Dict

logger = logging.getLogger(__name__)

class ResponseOptimizer:
    """🚀 Public Response Optimizer (Basic Filtering + Length Control)"""

    def __init__(self):
        self.response_cache = {}  # Stores past responses to reduce redundant processing
        self.optim_rules = defaultdict(list)  # Holds rule-based response refinements
        self.cache_limit = 100  # ✅ Limit cache size to avoid overflow

    async def optimize_response(self, response: str, context: Dict) -> str:
        """✅ Optimizes AI-generated responses (Fast + Secure)."""

        # ✅ Return cached response if available
        if response in self.response_cache:
            logger.info("✅ Returning cached optimized response.")
            return self.response_cache[response]

        # ✅ Apply context-based optimization (length, profanity)
        optimized_response = self.apply_optimizations(response, context)

        # ✅ Store in cache (Respect size limit)
        if len(self.response_cache) >= self.cache_limit:
            oldest_response = next(iter(self.response_cache))
            del self.response_cache[oldest_response]

        self.response_cache[response] = optimized_response
        return optimized_response

    def apply_optimizations(self, response: str, context: Dict) -> str:
        """✅ Applies context-specific response optimization."""
        if "filter_profanity" in context and context["filter_profanity"]:
            response = self.remove_profanity(response)

        if "trim_length" in context:
            response = response[:context["trim_length"]].strip() + "..."  # Trims to desired length

        return response

    def remove_profanity(self, response: str) -> str:
        """🚫 Removes flagged words from AI-generated responses."""
        banned_words = [
            "badword1", "badword2", "badword3", "shit", "fuck", "damn", "bitch", "asshole"
        ]  # ✅ Add or remove based on testing

        for word in banned_words:
            response = response.replace(word, "***")  # ✅ Replace with censorship symbol

        return response
