"""
evaluate.py
-----------
CLI entry-point for model evaluation.

Usage:
    python scripts/evaluate.py --config config/config.yaml --checkpoint outputs/models/best

Responsibilities:
 - Load a trained checkpoint and the test split.
 - Compute entity-level and token-level metrics.
 - Print a classification report and save results to disk.
"""
