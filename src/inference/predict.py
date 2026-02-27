"""
predict.py
----------
Inference pipeline for running NER on new text.

Responsibilities:
 - Load a trained checkpoint and tokeniser.
 - Accept raw strings or files and return structured entity spans.
 - Post-process BIO tags into human-readable entity dicts.
 - Support batched and single-sample inference.
"""
