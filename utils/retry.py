# Copyright (c) 2025 Loong Ma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from functools import wraps
import time
import logging

logger = logging.getLogger(__name__)

def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    retryable_status_codes: tuple = (408, 429, 500, 502, 503, 504)
):
    """Retry decorator with exponential backoff for API calls.
    
    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        backoff_factor: Multiplicative factor for delay after each retry
        retryable_status_codes: HTTP status codes that should trigger a retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for retry in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    
                    # Check if result is a dict with error status
                    if isinstance(result, dict) and result.get("status") == "error":
                        error_msg = result.get("message", "Unknown error")
                        if any(str(code) in error_msg for code in retryable_status_codes):
                            raise Exception(error_msg)
                        return result  # Non-retryable error
                    
                    return result  # Successful result
                    
                except Exception as e:
                    last_exception = e
                    if retry == max_retries:
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        return {"status": "error", "message": str(e)}
                    
                    # Calculate next delay
                    sleep_time = min(delay, max_delay)
                    logger.warning(f"Attempt {retry + 1} failed: {str(e)}. Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
                    delay *= backoff_factor
            
            return {"status": "error", "message": str(last_exception)}
        return wrapper
    return decorator