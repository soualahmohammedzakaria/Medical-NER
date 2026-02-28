"""
api/main.py

Flask application for Medical-NER inference.

Loads the fine-tuned model once at startup and exposes a
POST /predict endpoint that accepts raw text and returns
detected entity spans.

Usage
-----
  python -m api.main                              # default checkpoint
  python -m api.main --checkpoint outputs/models/best --port 5000
  FLASK_DEBUG=1 python -m api.main                # development mode
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from flask import Flask, jsonify, request as flask_request

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.schemas import PredictRequest, PredictResponse, EntitySpan
from src.inference.predict import NERPredictor

# ---------------------------------------------------------------------------
# Globals (populated in create_app)
# ---------------------------------------------------------------------------
predictor: NERPredictor | None = None


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(checkpoint_dir: str = "outputs/models/best",
               device: str = "auto") -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    # Load model once on startup
    global predictor
    predictor = NERPredictor(checkpoint_dir=checkpoint_dir, device=device)
    print(f"Model loaded from: {checkpoint_dir}")

    # -- routes ------------------------------------------------------------

    @app.route("/health", methods=["GET"])
    def health():
        """Simple liveness check."""
        return jsonify({"status": "ok"})

    @app.route("/predict", methods=["POST"])
    def predict():
        """Run NER on the submitted text.

        Expects JSON: {"text": "..."}
        Returns JSON:  {"text": "...", "entities": [...]}
        """
        payload = flask_request.get_json(silent=True)
        if payload is None:
            return jsonify({"error": "Request body must be JSON."}), 400

        # Validate with Pydantic
        try:
            req = PredictRequest(**payload)
        except Exception as exc:
            return jsonify({"error": str(exc)}), 422

        # Inference
        entities = predictor.predict(req.text)

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
