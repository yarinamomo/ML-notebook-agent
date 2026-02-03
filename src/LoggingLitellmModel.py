import logging
from minisweagent.models.litellm_model import LitellmModel

# Configure logging
from src.utils.log import logger

class LoggingLitellmModel(LitellmModel):
    def __init__(self, *args, **kwargs):
        logger.info("Initializing LoggingLitellmModel with args: %s, kwargs: %s", args, kwargs)
        self.step = 0
        super().__init__(*args, **kwargs)

    def query(self, *args, **kwargs):
        logger.info("Calling query with args: %s, kwargs: %s", args, kwargs)
        result = super().query(*args, **kwargs)
        self._log_response(result)
        return result


    def _log_response(self, response):
        """Log the response."""
        self.step += 1
        
        # Handle dict responses (from litellm)
        if isinstance(response, dict):
            content = response.get('content', '')
            extra = response.get('extra', {})
            
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

