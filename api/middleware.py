import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from api.database import insert_prediction, is_db_enabled

logger = logging.getLogger(__name__)

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "predictions.jsonl"


class PredictionLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method != "POST" or request.url.path != "/predict":
            return await call_next(request)

        body = await request.body()
        try:
            input_data = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            input_data = {}

        response = await call_next(request)

        response_body = b""
        async for chunk in response.body_iterator:
            response_body += chunk if isinstance(chunk, bytes) else chunk.encode()

        if response.status_code == 200:
            try:
                response_data = json.loads(response_body)
            except (json.JSONDecodeError, ValueError):
                response_data = {}

            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "input_features": input_data,
                "prediction": response_data.get("prediction"),
                "probability_default": response_data.get("probability_default"),
                "credit_decision": response_data.get("credit_decision"),
            }

            try:
                if is_db_enabled():
                    await insert_prediction(log_entry)
                else:
                    with open(LOG_FILE, "a") as f:
                        f.write(json.dumps(log_entry) + "\n")
            except Exception:
                logger.exception("Failed to log prediction")

        return Response(
            content=response_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )
