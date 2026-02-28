"""
api/main.py

Flask application for Medical-NER inference.

Loads the fine-tuned model once at startup and exposes a
POST /predict endpoint that accepts raw text and returns
detected entity spans.

Good-practice features
----------------------
- Rate limiting (flask-limiter)
- CORS (flask-cors)
- Structured error handlers (400, 404, 405, 413, 422, 429, 500)
- Request-ID header for traceability
- Response-time header for observability
- Content-Type enforcement on POST
- Input length cap via Pydantic schema
- Graceful model-not-loaded guard
- Logging with timestamps

Usage
-----
  python -m api.main                              # default checkpoint
  python -m api.main --checkpoint outputs/models/best --port 5000
  FLASK_DEBUG=1 python -m api.main                # development mode
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from pathlib import Path

from flask import Flask, g, jsonify, request as flask_request
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from pydantic import ValidationError

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.schemas import (
    EntitySpan,
    ErrorResponse,
    PredictRequest,
    PredictResponse,
)
from src.inference.predict import NERPredictor

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("api")

# ---------------------------------------------------------------------------
# Globals (populated in create_app)
# ---------------------------------------------------------------------------
predictor: NERPredictor | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _error(code: str, message: str, status: int):
    """Return a JSON error response using the ErrorResponse schema."""
    body = ErrorResponse(error=code, message=message).model_dump()
    return jsonify(body), status


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    checkpoint_dir: str = "outputs/models/best",
    device: str = "auto",
) -> Flask:
    """Create and configure the Flask application."""

    app = Flask(__name__)

    # -- configuration -----------------------------------------------------
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024   # 1 MB body limit

    # -- CORS --------------------------------------------------------------
    CORS(app, resources={r"/*": {"origins": "*"}})

    # -- Rate limiting -----------------------------------------------------
    limiter = Limiter(
        key_func=get_remote_address,
        app=app,
        default_limits=["120 per minute"],
        storage_uri="memory://",
    )

    # -- Load model --------------------------------------------------------
    global predictor
    predictor = NERPredictor(checkpoint_dir=checkpoint_dir, device=device)
    logger.info("Model loaded from: %s", checkpoint_dir)

    # -- Before / after request hooks --------------------------------------

    @app.before_request
    def _before():
        """Attach a unique request ID and start timer."""
        g.request_id = flask_request.headers.get(
            "X-Request-ID", str(uuid.uuid4()),
        )
        g.start_time = time.perf_counter()

    @app.after_request
    def _after(response):
        """Inject traceability and timing headers."""
        response.headers["X-Request-ID"] = g.get("request_id", "")
        elapsed = time.perf_counter() - g.get("start_time", time.perf_counter())
        response.headers["X-Response-Time-ms"] = f"{elapsed * 1000:.1f}"

        # Log every request
        logger.info(
            "%s %s %s  %.0fms  [%s]",
            flask_request.method,
            flask_request.path,
            response.status_code,
            elapsed * 1000,
            g.get("request_id", ""),
        )
        return response

    # -- Error handlers ----------------------------------------------------

    @app.errorhandler(400)
    def bad_request(exc):
        return _error("bad_request", str(exc), 400)

    @app.errorhandler(404)
    def not_found(exc):
        return _error("not_found", "The requested URL was not found.", 404)

    @app.errorhandler(405)
    def method_not_allowed(exc):
        return _error("method_not_allowed", "HTTP method not allowed.", 405)

    @app.errorhandler(413)
    def payload_too_large(exc):
        return _error("payload_too_large",
                       "Request body exceeds the 1 MB limit.", 413)

    @app.errorhandler(429)
    def rate_limited(exc):
        return _error("rate_limited",
                       "Too many requests. Please slow down.", 429)

    @app.errorhandler(500)
    def internal_error(exc):
        logger.exception("Unhandled server error")
        return _error("internal_error",
                       "An unexpected error occurred.", 500)

    # -- Routes ------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    @limiter.exempt
    def health():
        """Liveness / readiness check."""
        model_ready = predictor is not None
        status_code = 200 if model_ready else 503
        return jsonify({
            "status": "ok" if model_ready else "unavailable",
            "model_loaded": model_ready,
        }), status_code

    @app.route("/predict", methods=["POST"])
    @limiter.limit("60 per minute")
    def predict():
        """Run NER on the submitted text.

        Expects JSON: {"text": "..."}
        Returns JSON:  {"text": "...", "entities": [...]}
        """
        # Enforce Content-Type
        if not flask_request.is_json:
            return _error(
                "unsupported_media_type",
                "Content-Type must be application/json.",
                415,
            )

        payload = flask_request.get_json(silent=True)
        if payload is None:
            return _error("bad_request", "Request body must be valid JSON.", 400)

        # Validate with Pydantic
        try:
            req = PredictRequest(**payload)
        except ValidationError as exc:
            details = exc.errors()
            messages = "; ".join(
                f"{e['loc'][-1]}: {e['msg']}" for e in details
            )
            return _error("validation_error", messages, 422)

        # Guard against missing model
        if predictor is None:
            return _error(
                "service_unavailable",
                "Model is not loaded. Check /health.",
                503,
            )

        # Inference
        try:
            entities = predictor.predict(req.text)
        except Exception:
            logger.exception("Inference failed for request %s",
                             g.get("request_id", ""))
            return _error("inference_error",
                          "Model inference failed.", 500)

        # Build response
        resp = PredictResponse(
            text=req.text,
            entities=[
                EntitySpan(
                    text=e.text,
                    label=e.label,
                    start=e.start,
                    end=e.end,
                )
                for e in entities
            ],
        )

        return jsonify(resp.model_dump()), 200

    return app


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Medical-NER Flask API server.")
    p.add_argument("--checkpoint", type=str, default="outputs/models/best",
                    help="Path to the saved model directory.")
    p.add_argument("--device", type=str, default="auto",
                    help="Inference device: cpu, cuda, mps, or auto.")
    p.add_argument("--host", type=str, default="0.0.0.0",
                    help="Host to bind to (default: 0.0.0.0).")
    p.add_argument("--port", type=int, default=5000,
                    help="Port to listen on (default: 5000).")
    p.add_argument("--debug", action="store_true",
                    help="Run Flask in debug mode.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(checkpoint_dir=args.checkpoint, device=args.device)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
