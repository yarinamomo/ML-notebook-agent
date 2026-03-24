"""
Retry decorators for sandbox operations.
"""

import time
from functools import wraps
from typing import Callable, TYPE_CHECKING
from src.utils.log import logger
from src.ui_agent import EnvironmentUnavailable

if TYPE_CHECKING:
    from src.sandbox import DockerSandbox


def check_websocket_connected():
    """Decorator that checks websocket is connected before executing operation.

    Attempts to reconnect via restart_kernel if disconnected.

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: "DockerSandbox", *args, **kwargs):
            if not self._is_websocket_connected():
                logger.warning("WebSocket disconnected, attempting to reconnect...")
                self.restart_kernel()

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def retry_on_failure(max_retries=3, delay_seconds=1.0):
    """Decorator that retries a function on failure, raises EnvironmentUnavailable after max retries.

    Args:
        max_retries: Number of retry attempts (default: 3)
        delay_seconds: Delay between retries (default: 1.0)

    Returns:
        Decorated function with retry logic
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: "DockerSandbox", *args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    result = func(self, *args, **kwargs)
                    return result
                except Exception as exc:
                    logger.info(
                        f"Attempt {attempt + 1}/{max_retries + 1} failed with error: {str(exc)}"
                    )

                    if attempt < max_retries:
                        time.sleep(delay_seconds)
                        continue
                    else:
                        # All retries exhausted
                        logger.error(
                            "Operation failed after %d retries, raising EnvironmentUnavailable",
                            max_retries + 1,
                        )
                        raise EnvironmentUnavailable(
                            {
                                "role": "exit",
                                "content": "EnvironmentUnavailable",
                                "extra": {
                                    "exit_status": "EnvironmentUnavailable",
                                    "submission": "Operation failed after all retry attempts",
                                },
                            }
                        ) from exc

        return wrapper

    return decorator
