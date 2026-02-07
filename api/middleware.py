import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "predictions.jsonl"

_lock = threading.Lock()


class PredictionLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or request.url.path != "/predict":
            return await call_next(request)

        body = await request.body()
        try:
            input_data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            input_data = {}

        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk if isinstance(chunk, bytes) else chunk.encode()

        try:
            response_data = json.loads(response_body)
        except (json.JSONDecodeError, ValueError):
            response_data = {}

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "input_features": input_data,
            "prediction": response_data.get("prediction"),
            "probability_default": response_data.get("probability_default"),
            "credit_decision": response_data.get("credit_decision"),
            "error": response_data.get("detail") if response.status_code >= 400 else None,
        }

        with _lock:
            with open(LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
