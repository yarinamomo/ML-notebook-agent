import logging
from minisweagent.models.litellm_model import LitellmModel

# Configure logging
from src.utils.log import logger

class LoggingLitellmModel(LitellmModel):
    def __init__(self, *args, cost_per_prompt_token: float = 0.00001, cost_per_completion_token: float = 0.00003, **kwargs):
        # Extract pricing parameters before passing to parent
        self.cost_per_prompt_token = cost_per_prompt_token
        self.cost_per_completion_token = cost_per_completion_token
        
        logger.info("Initializing LoggingLitellmModel with args: %s, kwargs: %s", args, kwargs)
        self.step = 0
        self.cost = 0.0  # Cost tracking
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        super().__init__(*args, **kwargs)

    def query(self, *args, **kwargs):
        logger.debug("Calling query with args: %s, kwargs: %s", args, kwargs)
        result = super().query(*args, **kwargs)
        self._log_response(result)
        return result


    def _log_response(self, response):
        """Log the response and track costs."""
        self.step += 1
        
        # Handle dict responses (from litellm)
        if isinstance(response, dict):
            content = response.get('content', '')
            extra = response.get('extra', {})
            
            # Track costs from usage data
            if 'response' in extra:
                llm_response = extra['response']
                usage = llm_response.get('usage', {})
                
                # Get token counts
                prompt_tokens = usage.get('prompt_tokens', 0)
                completion_tokens = usage.get('completion_tokens', 0)
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens
                
                # Try to get cost from litellm (for known models)
                response_cost = llm_response.get('_hidden_params', {}).get('response_cost', 0.0)
                if response_cost > 0:
                    # Use LiteLLM's cost if available
                    self.cost += response_cost
                else:
                    # Calculate manually for custom models
                    calculated_cost = (
                        prompt_tokens * self.cost_per_prompt_token +
                        completion_tokens * self.cost_per_completion_token
                    )
                    self.cost += calculated_cost
            
            logger.debug(f"\n{'='*60}")
            logger.debug(f"🤖 AGENT RESPONSE (Step {self.step}):")
            logger.debug(f"CONTENT (what agent wants to execute):")
            logger.debug(content)
            logger.debug(f"\nMETADATA:")
            if 'response' in extra:
                llm_response = extra['response']
                logger.debug(f"  Model: {llm_response.get('model', 'unknown')}")
                logger.debug(f"  Tokens: {llm_response.get('usage', {})}")
            logger.debug(f"{'='*60}\n")
        else:
            # String response
            response_str = str(response)
            
            logger.debug(f"\n{'='*60}")
            logger.debug(f"🤖 AGENT RESPONSE (Step {self.step}):")
            logger.debug(response_str[:1000])
            if len(response_str) > 1000:
                    logger.debug(f"\n... (truncated, total length: {len(response_str)} chars)")

