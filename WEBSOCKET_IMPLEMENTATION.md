# WebSocket-Based Jupyter Kernel Implementation

## Overview
The `sandbox.py` file has been refactored to use Jupyter's **REST API + WebSocket** approach instead of `KernelClient`. This provides better control over kernel execution, timeouts, and error handling.

## Key Features

### 1. **WebSocket Communication**
- Direct WebSocket connection to Jupyter kernel
- Persistent connection managed by a background thread
- Full control over the Jupyter messaging protocol

### 2. **Robust Timeout Handling**
```python
result = sandbox.run(code, timeout=30)  # Default 30 seconds
result = sandbox.run(code, timeout=60, max_retries=2)  # Custom timeout and retries
```
- Executes code with configurable timeout
- Automatically interrupts long-running/infinite loops
- Restarts kernel after timeout to ensure clean state
- Automatic reconnection with configurable retries (default: 2 attempts)

### 3. **Automatic Kernel Interruption**
- Uses Jupyter REST API to send interrupt signal (`/api/kernels/{id}/interrupt`)
- Independent of WebSocket, so works even if kernel becomes unresponsive
- Non-blocking interrupt operation with timeout protection

### 4. **Graceful Kernel Restart**
When timeout occurs:
1. Send interrupt signal to kernel
2. Wait for interrupt to take effect
3. Close WebSocket connection
4. Delete old kernel via REST API
5. Create fresh kernel and reconnect WebSocket
6. Resume operations seamlessly

### 5. **Thread-Safe WebSocket Communication**
- Uses `threading.Lock` (`ws_lock`) to protect WebSocket sends
- Prevents race conditions from multiple threads
- Safe connection state checks before sending

### 6. **Automatic Reconnection with Retries**
- Detects WebSocket disconnection before and during execution
- Automatically reconnects with configurable retries (default: 2)
- Delays between reconnection attempts (1 second) to allow server recovery
- Distinguishes between connection errors (retryable) and execution errors (not retried)

### 7. **Complete Message Handling**
Supports all Jupyter message types:
- `stream` - stdout/stderr output
- `execute_result` - expression results
- `display_data` - rich media (images, HTML, etc.)
- `error` - exceptions with traceback
- `execute_reply` - execution completion signal

## Architecture

```
┌─────────────────────────────────────────┐
│       DockerSandbox                     │
└─────────────────────────────────────────┘
         │
         ├─ Docker Container
         │  └─ Jupyter Server
         │
         └─ WebSocket (Background Thread)
            │
            ├─ REST API for Kernel Management
            │  ├─ /api/kernels (POST) - Create kernel
            │  ├─ /api/kernels/{id}/interrupt (POST) - Interrupt
            │  └─ /api/kernels/{id} (DELETE) - Delete kernel
            │
            └─ Jupyter Kernel Protocol
               ├─ execute_request
               ├─ execute_reply
               ├─ stream
               ├─ execute_result
               ├─ error
               └─ ...
```

## Usage Example

```python
from src.sandbox import DockerSandbox

# Initialize sandbox
sandbox = DockerSandbox(
    image_name="jupyter/scipy-notebook",
    port=8888,
    token="your_auth_token",
    mount_volume="/path/to/volume"
)

# Start container and kernel
sandbox.start()

# Execute code with default timeout and retries
try:
    result = sandbox.run(
        """
        import time
        for i in range(100):
            print(i)
            time.sleep(0.1)
        """,
        timeout=30  # Default: 30 seconds, 2 retries
    )
    print(result['outputs'])
except TimeoutError:
    print("Code execution timed out and kernel was restarted")
except RuntimeError as e:
    print(f"Execution failed after all retries: {e}")
finally:
    sandbox.stop()

# Execute with custom timeout and retry settings
try:
    result = sandbox.run(
        "import pandas as pd\ndf = pd.DataFrame({'a': [1, 2, 3]})",
        timeout=60,      # 60-second timeout
        max_retries=3    # Up to 3 reconnection attempts
    )
except RuntimeError as e:
    print(f"Failed after 3 reconnection attempts: {e}")
finally:
    sandbox.stop()
```

## Methods

### `start()`
- Removes old container if exists
- Starts new Docker container
- Waits for Jupyter server to be ready
- Establishes WebSocket connection to kernel

### `run(code: str, timeout=30, max_retries=2) -> CellExecutionResult`
- Executes code with timeout protection
- Automatic reconnection if WebSocket is disconnected (up to `max_retries` times)
- Retries on connection errors, but not on execution errors
- Returns execution result with outputs
- Automatically restarts kernel on timeout
- Raises `TimeoutError` or `RuntimeError` on failure

**Parameters:**
- `code` - Python code to execute
- `timeout` - Max execution time in seconds (default: 30)
- `max_retries` - Max reconnection attempts (default: 2)

### `_start_kernel_websocket()`
- Creates new kernel via REST API
- Connects WebSocket with callbacks
- Starts background thread to handle messages
- Waits for connection to be properly established

### `_wait_for_websocket_connection(timeout=10)`
- Actively polls for WebSocket connection status
- Checks every 100ms if socket is ready
- Raises `RuntimeError` if connection not established within timeout
- Prevents execution before connection is actually ready

### `_is_websocket_connected() -> bool`
- Safely checks if WebSocket is properly connected
- Returns `True` only if socket exists, has socket object, and is connected
- Handles exceptions gracefully
- **Note:** Used internally to verify connection state before operations

### `_execute_code(code: str, timeout=30) -> CellExecutionResult`
- Internal method that performs actual code execution
- Called by `run()` after connection is established
- Handles message sending and result accumulation
- Does not include retry logic (handled by `run()`)

### `_interrupt_kernel()`
- Sends interrupt signal via REST API
- Non-blocking operation
- Works independently of WebSocket state

### `_restart_kernel_websocket()`
- Gracefully closes WebSocket
- Deletes old kernel
- Creates fresh kernel with new connection
- Clears pending execution results

### `restart_kernel()`
- Public method to restart kernel without stopping container
- Calls internal `_restart_kernel_websocket()`
- Useful for `run_all()` operations that need fresh state

### `stop()`
- Closes WebSocket gracefully
- Deletes kernel
- Kills and removes Docker container

## Exception Handling

- **TimeoutError** - Code execution exceeded timeout (kernel is interrupted and restarted)
- **RuntimeError** - Kernel unresponsive, initialization failed, or execution error
  - Connection errors: Automatically retried up to `max_retries` times
  - Execution errors: Raised immediately without retry
- WebSocket disconnection automatically triggers reconnection on next `run()` or before first execution
- Failed reconnection attempts raise `RuntimeError` with message indicating retry count

## Thread Safety

- WebSocket operations protected by `ws_lock`
- Message handling processed asynchronously without blocking
- Background thread-safe shutdown procedures

## Reconnection Behavior

The implementation provides robust automatic reconnection on connection failures:

**Before Execution:**
- Every `run()` call checks if WebSocket is connected
- If disconnected, automatically reconnects (up to `max_retries` times)
- Connection is verified active before sending any code

**During Execution:**
- If WebSocket disconnects while executing, the current execution is retried
- Retry includes reconnection and re-sending the code
- Execution errors (code raises exception) are NOT retried

**Retry Delays:**
- 1 second delay between reconnection attempts
- Allows server time to recover if temporarily unavailable
- Configurable via `max_retries` parameter

**Example Scenario:**
```
run(code, max_retries=2)
├─ Check connection → Disconnected
├─ Reconnect attempt 1 → Success, send code
├─ Kernel crashes during execution
├─ Reconnect attempt 2 → Success, re-send code
├─ Code executes successfully
└─ Return result
```

## Advantages Over KernelClient

| Feature | KernelClient | WebSocket + REST |
|---------|--------------|------------------|
| Timeout Handling | Limited, blocking | Full control, configurable |
| Unresponsive Kernel | Can hang threads | Independent interrupt signal |
| Connection Stability | No retries | Auto-reconnect with retries |
| Protocol Control | High-level abstraction | Direct low-level control |
| Language Support | Python only | Language agnostic |
| Error Recovery | Manual reconnection | Automatic graceful restart |
| Code Clarity | More dependencies | Explicit protocol handling |

## Dependencies

```python
requests      # REST API calls
docker        # Container management
websocket-client  # WebSocket communication
json          # Message serialization
threading     # Concurrent execution
```

## Configuration

Key parameters in `__init__()`:
- `image_name` - Docker image to run
- `base_url` - Jupyter server URL (default: `http://127.0.0.1`)
- `port` - Jupyter port (default: `8888`)
- `token` - Jupyter authentication token
- `mount_volume` - Host directory to mount in container
- `start_command` - Custom container startup command

## Logging

The implementation provides detailed logging:
- `logger.info()` - Kernel lifecycle events
- `logger.warning()` - Connection issues, timeouts
- `logger.error()` - Failures and interruptions
- `logger.debug()` - Message details
- `logger.exception()` - Full exception traces

Enable debug logging to monitor protocol interactions:
```python
import logging
logging.getLogger('src.utils.log').setLevel(logging.DEBUG)
```

## Future Enhancements

1. **Connection Pooling** - Support multiple kernels
2. **Async/Await** - Async execution support
3. **Progress Tracking** - Monitor long-running operations
4. **Kernel Warm-up Pool** - Pre-create kernels for faster starts
5. **Resource Limits** - Memory/CPU constraints per kernel
6. **Execution History** - Query previous results
