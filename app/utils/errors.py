from http import HTTPStatus
from typing import Optional, Union

def create_error_response(
    message: str,
    err_type: str = "internal_error",
    status_code: Union[int, HTTPStatus] = HTTPStatus.INTERNAL_SERVER_ERROR,
    param: str = None,
    code: str = None,
    request_id: Optional[str] = None
):
    """
    Create an OpenAI-compatible error response.

    Args:
        message: Error message
        err_type: Error type (e.g., "invalid_request_error", "rate_limit_exceeded")
        status_code: HTTP status code
        param: Parameter that caused the error (optional)
        code: Error code (optional)
        request_id: Request ID for tracking (optional)

    Returns:
        Dictionary with error details and optional request_id
    """
    error_response = {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": str(code or (status_code.value if isinstance(status_code, HTTPStatus) else status_code))
        }
    }

    # Add request_id at the top level if provided (OpenAI format)
    if request_id:
        error_response["request_id"] = request_id

    return error_response