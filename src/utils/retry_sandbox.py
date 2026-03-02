"""
Retry decorator for sandbox operations with automatic kernel restart.
"""
import time
from functools import wraps
from typing import Callable, TYPE_CHECKING
from src.utils.log import logger
from src.ui_agent import EnvironmentUnavailable

if TYPE_CHECKING:
    from src.sandbox import DockerSandbox


def retry_with_kernel_restart(max_retries=3):
    """Decorator that retries operations with automatic websocket/kernel restart on failure.
    
    Tracks failures across multiple calls:
    - First failure (after all retries): marks operation as failed, re-raises original exception
    - Second failure (after all retries): raises EnvironmentUnavailable to signal flow interruption
    
    Args:
        max_retries: Maximum number of retry attempts (default: 3)
    
    Returns:
        Decorated function with retry logic
    """
    # Track if operation has failed before
    has_failed_before = False
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: "DockerSandbox", *args, **kwargs):
            nonlocal has_failed_before
            
            for attempt in range(max_retries + 1):
                try:
                    # Ensure WebSocket is connected before attempting operation
                    if not self._is_websocket_connected():
                        if attempt == 0:
                            logger.warning("WebSocket disconnected, attempting to reconnect...")
                        else:
                            logger.info("Reconnection attempt %d/%d", attempt, max_retries)
                        
                        self.restart_kernel()
                    
                    # Attempt the actual operation
                    result = func(self, *args, **kwargs)
                    # Success - clear failure flag
                    has_failed_before = False
                    return result
                except Exception as e:
                    logger.info("Attempt %d/%d failed with error: %s", attempt + 1, max_retries, str(e))
                    if attempt < max_retries:
                        time.sleep(1)  # Brief pause before retrying
                        continue
                    else:
                        if has_failed_before:
                            logger.error("Operation %s failed after all retries on multiple attempts, interrupting agent flow")
                            raise EnvironmentUnavailable({
                                "role": "exit",
                                "content": "EnvironmentUnavailable",
                                "extra": {"exit_status": "EnvironmentUnavailable", "submission": f"Kernel execution failed repeatedly after all retry attempts"},
                            }) from e
                        else:
                            has_failed_before = True  # Mark as failed after exhausting retries
                            raise
            
        return wrapper
    return decorator
