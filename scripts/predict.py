"""
predict.py
----------
CLI entry-point for running inference on new text.

Usage:
    python scripts/predict.py --input "The patient was prescribed Metformin."
    python scripts/predict.py --file notes.txt

Responsibilities:
 - Accept text via CLI argument or file.
 - Run the inference pipeline and pretty-print extracted entities.
"""
